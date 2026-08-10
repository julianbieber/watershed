use std::collections::HashMap;

use glam::{UVec2, Vec2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::field::{Field, FieldId};
use crate::layer::{Blend, LayerOp, Mask, Remap, SlopeMode};
use crate::noise::Noise;
use crate::raster::{CellRect, Raster, raster_coord, resolution, step, texel_center};
use crate::regions::{CompiledOutput, RegionMap, RegionOutput};
use crate::water::{WaterSpec, WaterState};

#[derive(Debug, Error)]
pub enum BakeError {
    #[error("terrain size has a zero component: {0} by {1}")]
    ZeroSize(u32, u32),
    #[error("two fields share the id `{0}`")]
    DuplicateField(String),
    #[error("field `{referenced}`, read by `{reader}`, is not in the document")]
    UnknownField { referenced: String, reader: String },
    #[error("fields depend on each other in a cycle: {0}")]
    Cycle(String),
    #[error("column `{column}`, read by `{reader}`, is not in the region table")]
    UnknownRegionColumn { column: String, reader: String },
}

/// TODO(jb-doc): what a terrain is here — a size, a set of named fields, a solved water
/// state, and nothing else that the caller does not own.
///
/// TODO(jb-comment): why the water is not part of the serialized document, on the same
/// terms as a field's bake.
///
/// TODO(jb-comment): why the spec that produced the water *is* part of the document where
/// the water itself is not — a derived thing needs the recipe that re-derives it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Terrain {
    pub size: UVec2,
    pub fields: Vec<Field>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub water_spec: Option<WaterSpec>,
    #[serde(skip)]
    pub(crate) water: Option<WaterState>,
}

impl Terrain {
    pub fn new(size: UVec2) -> Self {
        Self {
            size,
            fields: Vec::new(),
            water_spec: None,
            water: None,
        }
    }

    pub fn with_field(mut self, field: Field) -> Self {
        self.fields.push(field);
        self
    }

    pub fn field(&self, id: &str) -> Option<&Field> {
        self.fields.iter().find(|field| field.id.as_str() == id)
    }

    pub fn field_mut(&mut self, id: &str) -> Option<&mut Field> {
        self.fields.iter_mut().find(|field| field.id.as_str() == id)
    }

    /// TODO(jb-doc): the coordinate space this takes, and why it answers `None` for a
    /// field the document does not carry rather than zero.
    pub fn sample(&self, id: &str, x: f32, y: f32) -> Option<f32> {
        self.field(id).map(|field| field.sample(x, y))
    }

    pub fn rect(&self) -> CellRect {
        CellRect::from_size(self.size)
    }

    pub fn bake(&mut self) -> Result<(), BakeError> {
        self.bake_rect(self.rect())
    }

    /// The order a bake visits the fields in: every field after the ones it reads.
    ///
    /// This is the whole of what a caller needs to drive a bake a stage at a time, and it
    /// is a *plan* rather than a running bake — nothing is borrowed, so the document
    /// stays readable between stages and the caller decides the pacing. Anything that
    /// would make a bake fail late (a cycle, a missing field, a duplicate id) fails here
    /// instead, before a single texel is written.
    ///
    /// TODO(jb-doc): why this hands back ids rather than indices, given the caller then
    /// looks each one up again.
    pub fn bake_order(&self) -> Result<Vec<FieldId>, BakeError> {
        let index_of = self.index_fields()?;
        let dependencies = self.resolve_dependencies(&index_of)?;
        let order = topological_order(&dependencies, &self.fields)?;
        Ok(order
            .into_iter()
            .map(|index| self.fields[index].id.clone())
            .collect())
    }

    /// Bake one field over the whole document, assuming everything it reads is baked.
    ///
    /// **The assumption is the caller's to keep**, and [`Terrain::bake_order`] is how:
    /// walking that order calls this on a field only after its dependencies. Called out
    /// of order it does not fail — it reads whatever those fields currently hold, which
    /// for an unbaked one is zero. That is the same fallback a document has before any
    /// bake at all, and it is what makes a partially baked document *displayable* rather
    /// than an error state.
    pub fn bake_field(&mut self, id: &str) -> Result<(), BakeError> {
        if self.size.x == 0 || self.size.y == 0 {
            return Err(BakeError::ZeroSize(self.size.x, self.size.y));
        }
        let index_of = self.index_fields()?;
        let reader = FieldId::from(id);
        let target = lookup(&index_of, &reader, &reader)?;
        // **Only the target**, where a whole-document bake reallocates everything. That
        // is what makes [`Terrain::release`] mean something: reallocating every field
        // here would hand a released one its full-size raster straight back, and a
        // staged bake would peak at the sum of the document however carefully the caller
        // dropped things. A field still empty when something reads it samples as zero,
        // which is the documented reading of an unbaked field either way.
        let wanted = resolution(self.size, self.fields[target].shift);
        if self.fields[target].baked().size() != wanted {
            *self.fields[target].baked_mut() = Raster::new(wanted, 0.0);
        }

        let shifts: Vec<u8> = self.fields.iter().map(|field| field.shift).collect();
        let categorical: Vec<bool> = self.fields.iter().map(Field::is_categorical).collect();
        let mut baked: Vec<Raster<f32>> = self
            .fields
            .iter_mut()
            .map(|field| field.take_baked())
            .collect();

        let rect = self.rect();
        let result = self.evaluate(target, rect, &index_of, &shifts, &categorical, &mut baked);

        for (field, raster) in self.fields.iter_mut().zip(baked) {
            field.put_baked(raster);
        }
        result
    }

    /// Drop a field's baked raster, keeping the layers that would rebuild it.
    ///
    /// **This is what makes a staged bake affordable rather than merely visible.** A
    /// document's fields do not all have to be resident at once: an intermediate is dead
    /// as soon as everything downstream of it has been baked, and at a whole-world size a
    /// single shift-0 field is tens of megabytes. Releasing as the order advances turns
    /// the peak from the sum of every field into the widest live set.
    ///
    /// A released field samples as zero, exactly as one that has never been baked — so
    /// releasing something still to be read is not an error, it is a wrong answer, and
    /// the caller owns that distinction the same way it owns the bake order.
    pub fn release(&mut self, id: &str) -> bool {
        match self.field_mut(id) {
            Some(field) => {
                *field.baked_mut() = Raster::default();
                true
            }
            None => false,
        }
    }

    /// How many bytes the baked rasters currently hold.
    ///
    /// TODO(jb-doc): why this counts only the bakes and not the layers, given a painted
    /// layer carries a raster of its own.
    pub fn baked_bytes(&self) -> usize {
        self.fields
            .iter()
            .map(|field| {
                let size = field.baked().size();
                size.x as usize * size.y as usize * size_of::<f32>()
            })
            .sum()
    }

    /// One field's texels, written into `baked[target]`.
    ///
    /// The one implementation of "evaluate this stack here", shared by the whole-document
    /// bake, the rect re-bake and the staged one, so a stage cannot drift from a bake.
    fn evaluate(
        &self,
        target: usize,
        rect: CellRect,
        index_of: &HashMap<String, usize>,
        shifts: &[u8],
        categorical: &[bool],
        baked: &mut [Raster<f32>],
    ) -> Result<(), BakeError> {
        let field = &self.fields[target];
        let texels = rect.to_texels(field.shift, baked[target].size());
        if texels.is_empty() {
            return Ok(());
        }
        let layers = compile_layers(field, index_of, self.size)?;
        let bounds = field.bounds();
        let shift = field.shift;
        let rows = {
            let context = Evaluator {
                size: self.size,
                baked,
                shifts,
                categorical,
            };
            map_rows(texels.min.y, texels.max.y, |j| {
                (texels.min.x..texels.max.x)
                    .map(|i| context.texel(&layers, shift, bounds, i, j))
                    .collect()
            })
        };
        let target_raster = &mut baked[target];
        for (offset, row) in rows.into_iter().enumerate() {
            let j = texels.min.y + offset as u32;
            for (n, value) in row.into_iter().enumerate() {
                target_raster.set(texels.min.x + n as u32, j, value);
            }
        }
        Ok(())
    }

    /// TODO(jb-doc): the contract this carries — that the document is already baked, and
    /// that what comes back inside the rectangle is what a full bake would have written.
    pub fn bake_rect(&mut self, rect: CellRect) -> Result<(), BakeError> {
        if self.size.x == 0 || self.size.y == 0 {
            return Err(BakeError::ZeroSize(self.size.x, self.size.y));
        }

        let index_of = self.index_fields()?;
        let dependencies = self.resolve_dependencies(&index_of)?;
        let order = topological_order(&dependencies, &self.fields)?;
        self.reallocate_rasters();

        let rect = rect.intersect(self.rect());
        if rect.is_empty() {
            return Ok(());
        }
        let required = self.required_rects(rect, &order, &dependencies);

        let shifts: Vec<u8> = self.fields.iter().map(|field| field.shift).collect();
        let categorical: Vec<bool> = self.fields.iter().map(Field::is_categorical).collect();
        let mut baked: Vec<Raster<f32>> = self
            .fields
            .iter_mut()
            .map(|field| field.take_baked())
            .collect();

        let mut result = Ok(());
        for &target in &order {
            result = self.evaluate(
                target,
                required[target],
                &index_of,
                &shifts,
                &categorical,
                &mut baked,
            );
            if result.is_err() {
                break;
            }
        }

        for (field, raster) in self.fields.iter_mut().zip(baked) {
            field.put_baked(raster);
        }
        result
    }

    // TODO(jb-comment): why the index owns its keys rather than borrowing the ids out of
    // the fields it indexes.
    fn index_fields(&self) -> Result<HashMap<String, usize>, BakeError> {
        let mut index_of = HashMap::with_capacity(self.fields.len());
        for (index, field) in self.fields.iter().enumerate() {
            if index_of.insert(field.id.to_string(), index).is_some() {
                return Err(BakeError::DuplicateField(field.id.to_string()));
            }
        }
        Ok(index_of)
    }

    fn resolve_dependencies(
        &self,
        index_of: &HashMap<String, usize>,
    ) -> Result<Vec<Vec<usize>>, BakeError> {
        let mut dependencies = vec![Vec::new(); self.fields.len()];
        for (index, field) in self.fields.iter().enumerate() {
            for id in field.dependencies() {
                let referenced = lookup(index_of, id, &field.id)?;
                if !dependencies[index].contains(&referenced) {
                    dependencies[index].push(referenced);
                }
            }
        }
        Ok(dependencies)
    }

    fn reallocate_rasters(&mut self) {
        let size = self.size;
        for field in &mut self.fields {
            let wanted = resolution(size, field.shift);
            if field.baked().size() != wanted {
                // TODO(jb-comment): why a resolution change discards the bake rather than
                // resampling it, and what that means for a rect re-bake after one.
                *field.baked_mut() = Raster::new(wanted, 0.0);
            }
        }
    }

    /// TODO(jb-comment): why the halo is deliberately generous, and why over-computing
    /// is safe where under-computing is not.
    fn required_rects(
        &self,
        rect: CellRect,
        order: &[usize],
        dependencies: &[Vec<usize>],
    ) -> Vec<CellRect> {
        let document = self.rect();
        let mut required = vec![rect; self.fields.len()];
        for &index in order.iter().rev() {
            if required[index].is_empty() || dependencies[index].is_empty() {
                continue;
            }
            for &referenced in &dependencies[index] {
                let halo = self.halo_between(index, referenced);
                let widened = required[index].expand(halo).intersect(document);
                required[referenced] = required[referenced].union(widened);
            }
        }
        required
    }

    /// How far from the cell it is about a read by `reader` of `referenced` can fall.
    ///
    /// TODO(jb-comment): what each of the four terms is paying for, and why the bound is
    /// symmetric — that the same number widens what a bake *needs* and what an edit
    /// *reaches*.
    fn halo_between(&self, reader: usize, referenced: usize) -> u32 {
        let field = &self.fields[reader];
        let reach = field
            .layers
            .iter()
            .filter(|layer| layer.enabled)
            .filter_map(|layer| match &layer.op {
                LayerOp::Slope { sample_tiles, .. } => Some(sample_tiles.abs()),
                _ => None,
            })
            .fold(0.0f32, f32::max);
        let reach = if reach.is_finite() {
            reach.ceil() as u32
        } else {
            0
        };
        step(field.shift) + reach + 2 * step(self.fields[referenced].shift) + 2
    }

    /// Which cells of the *whole document* could bake differently because one field changed
    /// over `rect` — the fields that read it, the fields that read those, and the halo each
    /// hop adds.
    ///
    /// TODO(jb-doc): why this is the rectangle a stroke re-bakes rather than the one it
    /// painted, and what a document that answered with the painted rectangle would leave
    /// behind in a field one hop downstream.
    ///
    /// TODO(jb-comment): why a stack that will not compile answers with the whole document
    /// rather than with nothing.
    pub fn influence_of(&self, changed: &str, rect: CellRect) -> CellRect {
        let document = self.rect();
        let rect = rect.intersect(document);
        if rect.is_empty() {
            return CellRect::EMPTY;
        }
        let Ok(index_of) = self.index_fields() else {
            return document;
        };
        let Some(&start) = index_of.get(changed) else {
            return CellRect::EMPTY;
        };
        let Ok(dependencies) = self.resolve_dependencies(&index_of) else {
            return document;
        };
        let Ok(order) = topological_order(&dependencies, &self.fields) else {
            return document;
        };

        // The order puts a field after everything it reads, so a dependency's rectangle is
        // final by the time the field that reads it is reached.
        let mut affected = vec![CellRect::EMPTY; self.fields.len()];
        affected[start] = rect;
        for &index in &order {
            for &referenced in &dependencies[index] {
                if affected[referenced].is_empty() {
                    continue;
                }
                let halo = self.halo_between(index, referenced);
                let widened = affected[referenced].expand(halo).intersect(document);
                affected[index] = affected[index].union(widened);
            }
        }
        affected.into_iter().fold(CellRect::EMPTY, CellRect::union)
    }
}

fn lookup(
    index_of: &HashMap<String, usize>,
    referenced: &FieldId,
    reader: &FieldId,
) -> Result<usize, BakeError> {
    index_of
        .get(referenced.as_str())
        .copied()
        .ok_or_else(|| BakeError::UnknownField {
            referenced: referenced.to_string(),
            reader: reader.to_string(),
        })
}

#[derive(Clone, Copy, PartialEq)]
enum Mark {
    Unvisited,
    InProgress,
    Done,
}

/// TODO(jb-comment): why the walk is depth-first in declaration order — that the bake
/// order has to be the same on every machine and in every run.
fn topological_order(
    dependencies: &[Vec<usize>],
    fields: &[Field],
) -> Result<Vec<usize>, BakeError> {
    let mut marks = vec![Mark::Unvisited; dependencies.len()];
    let mut order = Vec::with_capacity(dependencies.len());
    let mut stack = Vec::new();
    for index in 0..dependencies.len() {
        visit(
            index,
            dependencies,
            fields,
            &mut marks,
            &mut order,
            &mut stack,
        )?;
    }
    Ok(order)
}

fn visit(
    index: usize,
    dependencies: &[Vec<usize>],
    fields: &[Field],
    marks: &mut [Mark],
    order: &mut Vec<usize>,
    stack: &mut Vec<usize>,
) -> Result<(), BakeError> {
    match marks[index] {
        Mark::Done => return Ok(()),
        Mark::InProgress => {
            let start = stack.iter().position(|&i| i == index).unwrap_or(0);
            let mut names: Vec<String> = stack[start..]
                .iter()
                .map(|&i| fields[i].id.to_string())
                .collect();
            names.push(fields[index].id.to_string());
            return Err(BakeError::Cycle(names.join(" -> ")));
        }
        Mark::Unvisited => {}
    }
    marks[index] = Mark::InProgress;
    stack.push(index);
    for &referenced in &dependencies[index] {
        visit(referenced, dependencies, fields, marks, order, stack)?;
    }
    stack.pop();
    marks[index] = Mark::Done;
    order.push(index);
    Ok(())
}

enum CompiledOp<'a> {
    Constant(f32),
    Noise(Noise),
    Raster(&'a Raster<f32>),
    Slope {
        of: usize,
        sample_tiles: f32,
        mode: SlopeMode,
    },
    FieldRef(usize),
    Regions(RegionMap, CompiledOutput),
}

enum CompiledMask<'a> {
    Constant(f32),
    Painted(&'a Raster<u8>),
    Field(usize, Remap),
}

struct CompiledLayer<'a> {
    op: CompiledOp<'a>,
    mask: CompiledMask<'a>,
    blend: Blend,
    amplitude: f32,
}

fn compile_layers<'a>(
    field: &'a Field,
    index_of: &HashMap<String, usize>,
    size: UVec2,
) -> Result<Vec<CompiledLayer<'a>>, BakeError> {
    let mut compiled = Vec::with_capacity(field.layers.len());
    for layer in field.layers.iter().filter(|layer| layer.enabled) {
        let op = match &layer.op {
            LayerOp::Constant(value) => CompiledOp::Constant(*value),
            LayerOp::Noise(spec) => CompiledOp::Noise(Noise::new(spec)),
            LayerOp::Paint(raster) | LayerOp::External(raster) => CompiledOp::Raster(raster),
            LayerOp::Slope {
                of,
                sample_tiles,
                mode,
            } => CompiledOp::Slope {
                of: lookup(index_of, of, &field.id)?,
                sample_tiles: *sample_tiles,
                mode: *mode,
            },
            LayerOp::FieldRef(id) => CompiledOp::FieldRef(lookup(index_of, id, &field.id)?),
            LayerOp::Regions { spec, output } => {
                let compiled = match output {
                    RegionOutput::Blended(column) => {
                        CompiledOutput::Blended(spec.column_index(column).ok_or_else(|| {
                            BakeError::UnknownRegionColumn {
                                column: column.clone(),
                                reader: field.id.to_string(),
                            }
                        })?)
                    }
                    RegionOutput::RegionId => CompiledOutput::RegionId,
                    RegionOutput::CoverClass => CompiledOutput::CoverClass,
                };
                CompiledOp::Regions(RegionMap::new(spec, size), compiled)
            }
        };
        let mask = match &layer.mask {
            Mask::Constant(value) => CompiledMask::Constant(*value),
            Mask::Painted(raster) => CompiledMask::Painted(raster),
            Mask::Field(id, remap) => CompiledMask::Field(lookup(index_of, id, &field.id)?, *remap),
        };
        compiled.push(CompiledLayer {
            op,
            mask,
            blend: layer.blend,
            amplitude: layer.amplitude,
        });
    }
    Ok(compiled)
}

struct Evaluator<'a> {
    size: UVec2,
    baked: &'a [Raster<f32>],
    shifts: &'a [u8],
    /// Answered once per bake rather than per texel, where [`Field::sample`] asks the
    /// field itself — this is the read inside the loop, and the question is a property of
    /// the stack rather than of the position.
    categorical: &'a [bool],
}

impl Evaluator<'_> {
    /// TODO(jb-comment): why a categorical field is read at its nearest texel here as well
    /// as in [`Field::sample`], and which of the two a mask goes through.
    fn field(&self, index: usize, position: Vec2) -> f32 {
        let shift = self.shifts[index];
        let u = raster_coord(position.x, shift);
        let v = raster_coord(position.y, shift);
        if self.categorical[index] {
            self.baked[index].sample_nearest(u, v)
        } else {
            self.baked[index].sample_bilinear(u, v)
        }
    }

    fn mask(&self, mask: &CompiledMask<'_>, position: Vec2) -> f32 {
        let weight = match mask {
            CompiledMask::Constant(value) => *value,
            CompiledMask::Painted(raster) => raster.sample_over(self.size, position.x, position.y),
            CompiledMask::Field(index, remap) => remap.apply(self.field(*index, position)),
        };
        weight.clamp(0.0, 1.0)
    }

    /// TODO(jb-comment): why the slope is a central difference in document cells rather
    /// than in texels of the field it reads.
    fn slope(&self, index: usize, sample_tiles: f32, mode: SlopeMode, position: Vec2) -> f32 {
        let reach = sample_tiles.abs().max(f32::EPSILON);
        match mode {
            SlopeMode::Gradient => {
                let dx = self.field(index, position + Vec2::new(reach, 0.0))
                    - self.field(index, position - Vec2::new(reach, 0.0));
                let dy = self.field(index, position + Vec2::new(0.0, reach))
                    - self.field(index, position - Vec2::new(0.0, reach));
                let scale = 2.0 * reach;
                ((dx / scale).powi(2) + (dy / scale).powi(2)).sqrt()
            }
            SlopeMode::SteepestAxis => {
                let here = self.field(index, position);
                let dx = (self.field(index, position + Vec2::new(reach, 0.0)) - here).abs();
                let dy = (self.field(index, position + Vec2::new(0.0, reach)) - here).abs();
                dx.max(dy) / reach
            }
        }
    }

    fn value(&self, op: &CompiledOp<'_>, position: Vec2) -> f32 {
        match op {
            CompiledOp::Constant(value) => *value,
            CompiledOp::Noise(noise) => noise.sample(position.x, position.y),
            CompiledOp::Raster(raster) => raster.sample_over(self.size, position.x, position.y),
            CompiledOp::Slope {
                of,
                sample_tiles,
                mode,
            } => self.slope(*of, *sample_tiles, *mode, position),
            CompiledOp::FieldRef(index) => self.field(*index, position),
            CompiledOp::Regions(map, output) => map.sample(*output, position.x, position.y),
        }
    }

    fn texel(
        &self,
        layers: &[CompiledLayer<'_>],
        shift: u8,
        bounds: (f32, f32),
        i: u32,
        j: u32,
    ) -> f32 {
        let position = Vec2::new(texel_center(i, shift), texel_center(j, shift));
        let mut under = 0.0f32;
        for layer in layers {
            let weight = self.mask(&layer.mask, position);
            if weight <= 0.0 {
                continue;
            }
            let value = self.value(&layer.op, position) * layer.amplitude;
            let blended = layer.blend.apply(under, value);
            // TODO(jb-comment): why the endpoints are branches rather than falling out of
            // the interpolation.
            under = if weight >= 1.0 {
                blended
            } else {
                under + (blended - under) * weight
            };
        }
        under.clamp(bounds.0, bounds.1)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn map_rows<F>(start: u32, end: u32, row: F) -> Vec<Vec<f32>>
where
    F: Fn(u32) -> Vec<f32> + Send + Sync,
{
    use rayon::prelude::*;
    (start..end).into_par_iter().map(row).collect()
}

#[cfg(target_arch = "wasm32")]
fn map_rows<F>(start: u32, end: u32, row: F) -> Vec<Vec<f32>>
where
    F: Fn(u32) -> Vec<f32> + Send + Sync,
{
    (start..end).map(row).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::Layer;
    use crate::noise::{NoiseKind, NoiseSpec};

    fn noise_layer(seed: u32) -> Layer {
        Layer::new(LayerOp::Noise(NoiseSpec::new(seed, NoiseKind::Fbm, 0.05)))
            .with_blend(Blend::Replace)
    }

    fn two_field_document() -> Terrain {
        Terrain::new(UVec2::new(96, 80))
            .with_field(
                Field::new("moisture")
                    .with_shift(3)
                    .with_layer(noise_layer(7)),
            )
            .with_field(
                Field::new("height")
                    .with_shift(0)
                    .with_layer(noise_layer(11).with_mask(Mask::Field(
                        FieldId::from("moisture"),
                        Remap::new((0.3, 0.7), (0.0, 1.0)),
                    ))),
            )
    }

    /// The whole point of the staged bake: a caller that walks the order one field at a
    /// time has to end up with exactly the document a single `bake()` would have written,
    /// or the intermediate results it showed were of a different world.
    #[test]
    fn baking_a_stage_at_a_time_writes_what_one_bake_would_have() {
        let mut whole = two_field_document().with_field(
            Field::new("relief")
                .with_range((-1.0, 1.0))
                .with_layer(Layer::new(LayerOp::Slope {
                    of: FieldId::from("height"),
                    sample_tiles: 5.0,
                    mode: SlopeMode::SteepestAxis,
                })),
        );
        let mut staged = whole.clone();

        whole.bake().unwrap();
        for id in staged.bake_order().unwrap() {
            staged.bake_field(id.as_str()).unwrap();
        }

        for field in &whole.fields {
            let one = field.baked();
            let other = staged.field(field.id.as_str()).unwrap().baked();
            assert_eq!(one.size(), other.size(), "{} changed size", field.id);
            assert!(
                one.data().iter().zip(other.data()).all(|(a, b)| a == b),
                "{} differs between a staged bake and a whole one",
                field.id
            );
        }
    }

    /// Releasing is what keeps a staged bake's peak below the sum of its fields, and it
    /// has to leave the document able to rebuild what it dropped.
    #[test]
    fn a_released_field_reads_as_zero_and_bakes_back() {
        let mut terrain = two_field_document();
        terrain.bake().unwrap();

        let before = terrain.baked_bytes();
        let sampled = terrain.sample("height", 12.5, 9.5).unwrap();
        assert!(
            sampled != 0.0,
            "the sample to compare against is already zero"
        );

        assert!(terrain.release("height"));
        assert!(terrain.baked_bytes() < before);
        assert_eq!(terrain.sample("height", 12.5, 9.5), Some(0.0));

        terrain.bake_field("height").unwrap();
        assert_eq!(terrain.baked_bytes(), before);
        assert_eq!(terrain.sample("height", 12.5, 9.5), Some(sampled));
    }

    /// The guard on the whole point of releasing: a staged bake that drops what it no
    /// longer needs must actually *hold* less, not hand the raster straight back on the
    /// next stage.
    #[test]
    fn a_staged_bake_does_not_re_allocate_what_the_caller_released() {
        let mut terrain = two_field_document();
        let order = terrain.bake_order().unwrap();
        terrain.bake_field(order[0].as_str()).unwrap();

        let with_first = terrain.baked_bytes();
        terrain.release(order[0].as_str());
        let released = terrain.baked_bytes();
        assert!(released < with_first);

        // Baking the *next* field must not resurrect the one just dropped.
        terrain.bake_field(order[1].as_str()).unwrap();
        assert_eq!(
            terrain.field(order[0].as_str()).unwrap().baked().size(),
            UVec2::ZERO,
            "a released field came back when the next stage baked"
        );
    }

    #[test]
    fn releasing_a_field_that_is_not_there_says_so_rather_than_panicking() {
        let mut terrain = two_field_document();
        assert!(!terrain.release("nowhere"));
    }

    /// A stage list has to be an order rather than a listing: a field that reads another
    /// cannot come first, or the caller walking it would bake against zeros.
    #[test]
    fn the_stage_order_puts_a_field_after_everything_it_reads() {
        let terrain = two_field_document();
        let order = terrain.bake_order().unwrap();
        let position = |id: &str| order.iter().position(|got| got.as_str() == id).unwrap();
        assert!(position("moisture") < position("height"));
        assert_eq!(order.len(), terrain.fields.len());
    }

    /// The failures a bake can only discover late are the ones a caller most wants early,
    /// because a staged bake has already drawn half a world by the time it hits one.
    #[test]
    fn a_stage_list_refuses_a_document_a_bake_would_refuse() {
        let cyclic = Terrain::new(UVec2::splat(16))
            .with_field(Field::new("a").with_layer(Layer::new(LayerOp::FieldRef("b".into()))))
            .with_field(Field::new("b").with_layer(Layer::new(LayerOp::FieldRef("a".into()))));
        assert!(matches!(cyclic.bake_order(), Err(BakeError::Cycle(_))));

        let dangling = Terrain::new(UVec2::splat(16))
            .with_field(Field::new("a").with_layer(Layer::new(LayerOp::FieldRef("gone".into()))));
        assert!(matches!(
            dangling.bake_order(),
            Err(BakeError::UnknownField { .. })
        ));
    }

    /// Baking a field the document does not carry is the caller's mistake and has to say
    /// so, rather than quietly doing nothing.
    #[test]
    fn baking_a_field_that_is_not_there_says_which_one() {
        let mut terrain = two_field_document();
        assert!(matches!(
            terrain.bake_field("nowhere"),
            Err(BakeError::UnknownField { .. })
        ));
    }

    #[test]
    fn a_two_field_document_with_a_field_masked_layer_bakes() {
        let mut terrain = two_field_document();
        terrain.bake().unwrap();

        let moisture = terrain.field("moisture").unwrap();
        assert_eq!(moisture.baked().size(), UVec2::new(12, 10));
        let height = terrain.field("height").unwrap();
        assert_eq!(height.baked().size(), UVec2::new(96, 80));

        let values = height.baked().data();
        assert!(values.iter().all(|value| value.is_finite()));
        let lowest = values.iter().copied().fold(f32::INFINITY, f32::min);
        let highest = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(highest - lowest > 0.05, "{lowest} to {highest}");
    }

    #[test]
    fn a_rect_re_bake_is_bit_identical_to_a_full_one() {
        let mut terrain = two_field_document();
        terrain.bake().unwrap();
        let full: Vec<Vec<f32>> = terrain
            .fields
            .iter()
            .map(|field| field.baked().data().to_vec())
            .collect();

        for field in &mut terrain.fields {
            field.baked_mut().fill(f32::NAN);
        }
        let rect = CellRect::new(UVec2::new(33, 21), UVec2::new(70, 64));
        terrain.bake_rect(rect).unwrap();

        for (index, field) in terrain.fields.iter().enumerate() {
            let texels = rect.to_texels(field.shift, field.baked().size());
            let width = field.baked().width();
            for j in texels.min.y..texels.max.y {
                for i in texels.min.x..texels.max.x {
                    let at = (j * width + i) as usize;
                    assert_eq!(
                        field.baked().data()[at].to_bits(),
                        full[index][at].to_bits(),
                        "field {} at {i},{j}",
                        field.id
                    );
                }
            }
        }
    }

    #[test]
    fn a_rect_re_bake_of_the_whole_document_reproduces_every_texel() {
        let mut terrain = two_field_document();
        terrain.bake().unwrap();
        let full: Vec<Vec<f32>> = terrain
            .fields
            .iter()
            .map(|field| field.baked().data().to_vec())
            .collect();

        for field in &mut terrain.fields {
            field.baked_mut().fill(f32::NAN);
        }
        terrain.bake_rect(terrain.rect()).unwrap();

        for (index, field) in terrain.fields.iter().enumerate() {
            let bits: Vec<u32> = field.baked().data().iter().map(|v| v.to_bits()).collect();
            let wanted: Vec<u32> = full[index].iter().map(|v| v.to_bits()).collect();
            assert_eq!(bits, wanted, "field {}", field.id);
        }
    }

    #[test]
    fn a_slope_layer_reads_the_gradient_of_the_field_it_names() {
        let size = UVec2::new(64, 64);
        let ramp = Raster::from_vec(
            size,
            (0..size.x * size.y)
                .map(|n| (n % size.x) as f32 / size.x as f32)
                .collect(),
        )
        .unwrap();
        let mut terrain = Terrain::new(size)
            .with_field(
                Field::new("height")
                    .with_layer(Layer::new(LayerOp::External(ramp)).with_blend(Blend::Replace)),
            )
            .with_field(
                Field::new("soil").with_range((0.0, 10.0)).with_layer(
                    Layer::new(LayerOp::Slope {
                        of: FieldId::from("height"),
                        sample_tiles: 2.0,
                        mode: SlopeMode::default(),
                    })
                    .with_blend(Blend::Replace),
                ),
            );
        terrain.bake().unwrap();

        let slope = terrain.sample("soil", 32.5, 32.5).unwrap();
        assert!(
            (slope - 1.0 / 64.0).abs() < 1e-4,
            "expected the ramp's gradient, got {slope}"
        );
    }

    #[test]
    fn a_field_is_baked_before_the_field_that_reads_it() {
        let mut terrain = Terrain::new(UVec2::new(8, 8))
            .with_field(
                Field::new("derived").with_range((0.0, 10.0)).with_layer(
                    Layer::new(LayerOp::FieldRef(FieldId::from("source")))
                        .with_blend(Blend::Replace)
                        .with_amplitude(2.0),
                ),
            )
            .with_field(
                Field::new("source")
                    .with_range((0.0, 10.0))
                    .with_layer(Layer::new(LayerOp::Constant(3.0)).with_blend(Blend::Replace)),
            );
        terrain.bake().unwrap();
        assert_eq!(terrain.sample("derived", 4.5, 4.5).unwrap(), 6.0);
    }

    #[test]
    fn a_dependency_cycle_is_an_error_rather_than_a_hang() {
        let mut terrain = Terrain::new(UVec2::new(8, 8))
            .with_field(
                Field::new("a").with_layer(Layer::new(LayerOp::FieldRef(FieldId::from("b")))),
            )
            .with_field(
                Field::new("b").with_layer(Layer::new(LayerOp::FieldRef(FieldId::from("a")))),
            );
        let error = terrain.bake().unwrap_err();
        assert!(matches!(error, BakeError::Cycle(_)), "{error}");
    }

    #[test]
    fn a_cycle_that_only_exists_through_a_disabled_layer_is_not_a_cycle() {
        let mut terrain = Terrain::new(UVec2::new(8, 8))
            .with_field(
                Field::new("a").with_layer(Layer::new(LayerOp::FieldRef(FieldId::from("b")))),
            )
            .with_field(
                Field::new("b")
                    .with_layer(Layer::new(LayerOp::FieldRef(FieldId::from("a"))).disabled()),
            );
        terrain.bake().unwrap();
    }

    #[test]
    fn a_reference_to_a_field_that_is_not_there_is_an_error() {
        let mut terrain = Terrain::new(UVec2::new(8, 8)).with_field(
            Field::new("a").with_layer(Layer::new(LayerOp::FieldRef(FieldId::from("gone")))),
        );
        let error = terrain.bake().unwrap_err();
        assert!(
            matches!(&error, BakeError::UnknownField { referenced, reader }
                if referenced == "gone" && reader == "a"),
            "{error}"
        );
    }

    #[test]
    fn two_fields_may_not_share_an_id() {
        let mut terrain = Terrain::new(UVec2::new(8, 8))
            .with_field(Field::new("height"))
            .with_field(Field::new("height"));
        let error = terrain.bake().unwrap_err();
        assert!(matches!(error, BakeError::DuplicateField(_)), "{error}");
    }

    #[test]
    fn a_document_with_no_extent_is_an_error() {
        let mut terrain = Terrain::new(UVec2::new(0, 16));
        assert!(matches!(
            terrain.bake().unwrap_err(),
            BakeError::ZeroSize(0, 16)
        ));
    }

    #[test]
    fn a_mask_of_zero_is_a_no_op_for_every_blend_mode() {
        for blend in [
            Blend::Add,
            Blend::Mul,
            Blend::Replace,
            Blend::Max,
            Blend::Min,
        ] {
            let mut terrain = Terrain::new(UVec2::new(8, 8)).with_field(
                Field::new("height")
                    .with_range((-10.0, 10.0))
                    .with_layer(Layer::new(LayerOp::Constant(4.0)).with_blend(Blend::Replace))
                    .with_layer(
                        Layer::new(LayerOp::Constant(9.0))
                            .with_blend(blend)
                            .with_mask(Mask::Constant(0.0)),
                    ),
            );
            terrain.bake().unwrap();
            assert_eq!(
                terrain.sample("height", 4.5, 4.5).unwrap(),
                4.0,
                "{blend:?}"
            );
        }
    }

    #[test]
    fn a_mask_of_one_applies_the_layer_whole() {
        let mut terrain = Terrain::new(UVec2::new(8, 8)).with_field(
            Field::new("height")
                .with_range((-20.0, 20.0))
                .with_layer(Layer::new(LayerOp::Constant(4.0)).with_blend(Blend::Replace))
                .with_layer(
                    Layer::new(LayerOp::Constant(9.0))
                        .with_blend(Blend::Add)
                        .with_mask(Mask::Constant(1.0)),
                ),
        );
        terrain.bake().unwrap();
        assert_eq!(terrain.sample("height", 4.5, 4.5).unwrap(), 13.0);
    }

    #[test]
    fn a_half_mask_lands_halfway_between_blending_and_not() {
        let mut terrain = Terrain::new(UVec2::new(8, 8)).with_field(
            Field::new("height")
                .with_range((-10.0, 10.0))
                .with_layer(Layer::new(LayerOp::Constant(4.0)).with_blend(Blend::Replace))
                .with_layer(
                    Layer::new(LayerOp::Constant(8.0))
                        .with_blend(Blend::Add)
                        .with_mask(Mask::Constant(0.5)),
                ),
        );
        terrain.bake().unwrap();
        assert_eq!(terrain.sample("height", 4.5, 4.5).unwrap(), 8.0);
    }

    #[test]
    fn a_field_is_clamped_to_the_range_it_declares() {
        let mut terrain = Terrain::new(UVec2::new(8, 8)).with_field(
            Field::new("height")
                .with_range((0.0, 1.0))
                .with_layer(Layer::new(LayerOp::Constant(5.0)).with_blend(Blend::Replace)),
        );
        terrain.bake().unwrap();
        assert_eq!(terrain.sample("height", 4.5, 4.5).unwrap(), 1.0);
    }

    #[test]
    fn a_disabled_layer_contributes_nothing() {
        let mut terrain = Terrain::new(UVec2::new(8, 8)).with_field(
            Field::new("height")
                .with_range((-10.0, 10.0))
                .with_layer(Layer::new(LayerOp::Constant(4.0)).with_blend(Blend::Replace))
                .with_layer(Layer::new(LayerOp::Constant(9.0)).disabled()),
        );
        terrain.bake().unwrap();
        assert_eq!(terrain.sample("height", 4.5, 4.5).unwrap(), 4.0);
    }

    #[test]
    fn a_painted_mask_lets_a_layer_through_where_it_is_white() {
        let mask = Raster::from_vec(UVec2::new(2, 1), vec![0u8, 255]).unwrap();
        let mut terrain = Terrain::new(UVec2::new(8, 8)).with_field(
            Field::new("height")
                .with_range((-10.0, 10.0))
                .with_layer(Layer::new(LayerOp::Constant(1.0)).with_blend(Blend::Replace))
                .with_layer(
                    Layer::new(LayerOp::Constant(4.0))
                        .with_blend(Blend::Add)
                        .with_mask(Mask::Painted(mask)),
                ),
        );
        terrain.bake().unwrap();
        assert_eq!(terrain.sample("height", 0.5, 4.5).unwrap(), 1.0);
        assert_eq!(terrain.sample("height", 7.5, 4.5).unwrap(), 5.0);
    }

    /// TODO(jb-doc): what these figures are for, and against which machine they were
    /// taken. `cargo test --release -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn a_full_size_document_measures_what_a_bake_costs() {
        let size = UVec2::new(4096, 4096);
        let mut terrain = Terrain::new(size)
            .with_field(
                Field::new("moisture")
                    .with_shift(4)
                    .with_layer(noise_layer(3).with_amplitude(1.0)),
            )
            .with_field(
                Field::new("height")
                    .with_layer(
                        Layer::new(LayerOp::Noise(NoiseSpec::new(5, NoiseKind::Fbm, 0.0015)))
                            .with_blend(Blend::Replace),
                    )
                    .with_layer(
                        Layer::new(LayerOp::Noise(NoiseSpec::new(7, NoiseKind::Fbm, 0.04)))
                            .with_amplitude(0.25),
                    )
                    .with_layer(
                        Layer::new(LayerOp::Noise(NoiseSpec::new(11, NoiseKind::Ridged, 0.01)))
                            .with_amplitude(0.3)
                            .with_mask(Mask::Field(
                                FieldId::from("moisture"),
                                Remap::new((0.4, 0.7), (0.0, 1.0)),
                            )),
                    ),
            )
            .with_field(
                Field::new("soil").with_layer(
                    Layer::new(LayerOp::Slope {
                        of: FieldId::from("height"),
                        sample_tiles: 2.0,
                        mode: SlopeMode::default(),
                    })
                    .with_blend(Blend::Replace)
                    .with_amplitude(20.0),
                ),
            );

        let started = std::time::Instant::now();
        terrain.bake().unwrap();
        let full = started.elapsed();

        let rect = CellRect::new(UVec2::new(1024, 1024), UVec2::new(1088, 1088));
        let started = std::time::Instant::now();
        terrain.bake_rect(rect).unwrap();
        let patch = started.elapsed();

        println!("full bake of {} by {}: {full:?}", size.x, size.y);
        println!("64 by 64 rect re-bake: {patch:?}");
        for field in &terrain.fields {
            let data = field.baked().data();
            let lowest = data.iter().copied().fold(f32::INFINITY, f32::min);
            let highest = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            println!(
                "{}: {} texels, {lowest:.3} to {highest:.3}",
                field.id,
                data.len()
            );
        }
    }

    fn region_spec() -> crate::regions::RegionSpec {
        use crate::noise::WarpSpec;
        use crate::regions::{Region, RegionSpec};
        RegionSpec::new(0x5eed_0036, 128, 16, ["base", "ridge"])
            .with_region(Region::new(6, [0.20, 0.0]))
            .with_region(Region::new(4, [0.52, 0.0]))
            .with_region(Region::new(3, [0.70, 0.34]))
            .with_warp(WarpSpec {
                seed: 0x7a1d_0b37,
                amplitude: 48.0,
                scale: 1.0 / (128.0 * 0.75),
                octaves: 3,
                salts: None,
            })
    }

    fn region_document() -> Terrain {
        Terrain::new(UVec2::new(512, 512))
            .with_field(
                Field::new("ridge_weight").with_layer(
                    Layer::new(LayerOp::Regions {
                        spec: region_spec(),
                        output: RegionOutput::Blended("ridge".to_owned()),
                    })
                    .with_blend(Blend::Replace),
                ),
            )
            .with_field(
                Field::new("height")
                    .with_layer(
                        Layer::new(LayerOp::Regions {
                            spec: region_spec(),
                            output: RegionOutput::Blended("base".to_owned()),
                        })
                        .with_blend(Blend::Replace),
                    )
                    .with_layer(
                        Layer::new(LayerOp::Noise(NoiseSpec::new(11, NoiseKind::Ridged, 0.01)))
                            .with_amplitude(0.4)
                            .with_mask(Mask::Field(
                                FieldId::from("ridge_weight"),
                                Remap::new((0.0, 0.34), (0.0, 1.0)),
                            )),
                    ),
            )
    }

    #[test]
    fn a_regions_layer_bakes_a_blended_column_into_a_field() {
        let mut terrain = region_document();
        terrain.bake().unwrap();

        let values = terrain.field("height").unwrap().baked().data();
        assert!(values.iter().all(|value| value.is_finite()));
        let lowest = values.iter().copied().fold(f32::INFINITY, f32::min);
        let highest = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            highest - lowest > 0.2,
            "a region-blended height spanning only {lowest} to {highest}"
        );
    }

    #[test]
    fn a_rect_re_bake_of_a_regions_field_is_bit_identical_to_a_full_one() {
        let mut terrain = region_document();
        terrain.bake().unwrap();
        let full: Vec<Vec<f32>> = terrain
            .fields
            .iter()
            .map(|field| field.baked().data().to_vec())
            .collect();

        for field in &mut terrain.fields {
            field.baked_mut().fill(f32::NAN);
        }
        let rect = CellRect::new(UVec2::new(97, 61), UVec2::new(300, 288));
        terrain.bake_rect(rect).unwrap();

        for (index, field) in terrain.fields.iter().enumerate() {
            let texels = rect.to_texels(field.shift, field.baked().size());
            let width = field.baked().width();
            for j in texels.min.y..texels.max.y {
                for i in texels.min.x..texels.max.x {
                    let at = (j * width + i) as usize;
                    assert_eq!(
                        field.baked().data()[at].to_bits(),
                        full[index][at].to_bits(),
                        "field {} at {i},{j}",
                        field.id
                    );
                }
            }
        }
    }

    #[test]
    fn a_categorical_region_field_at_shift_zero_reads_back_a_whole_index() {
        let mut terrain = Terrain::new(UVec2::new(256, 256)).with_field(
            Field::new("region").with_range((0.0, 8.0)).with_layer(
                Layer::new(LayerOp::Regions {
                    spec: region_spec(),
                    output: RegionOutput::RegionId,
                })
                .with_blend(Blend::Replace),
            ),
        );
        terrain.bake().unwrap();

        let baked = terrain.field("region").unwrap().baked();
        assert_eq!(baked.size(), UVec2::new(256, 256));
        for value in baked.data() {
            assert_eq!(*value, value.round(), "a region id baked as {value}");
            assert!((0.0..3.0).contains(value), "a region id of {value}");
        }
    }

    /// The defect this guards: a value standing for a class has no midpoint, so a read
    /// between region 1 and region 3 must be one of the two rather than the region 2 that
    /// interpolating them invents.
    #[test]
    fn a_categorical_field_is_never_read_between_two_of_its_classes() {
        let mut terrain = Terrain::new(UVec2::new(256, 256))
            .with_field(
                Field::new("region").with_range((0.0, 8.0)).with_layer(
                    Layer::new(LayerOp::Regions {
                        spec: region_spec(),
                        output: RegionOutput::RegionId,
                    })
                    .with_blend(Blend::Replace),
                ),
            )
            .with_field(Field::new("copy").with_range((0.0, 8.0)).with_layer(
                Layer::new(LayerOp::FieldRef(FieldId::from("region"))).with_blend(Blend::Replace),
            ));
        terrain.bake().unwrap();

        let region = terrain.field("region").unwrap();
        assert!(region.is_categorical());
        for value in terrain.field("copy").unwrap().baked().data() {
            assert_eq!(*value, value.round(), "a region id read as {value}");
        }

        // The same question asked of the public read, which is the one wusel will go
        // through — off a texel centre, where a bilinear read is at its worst.
        for step in 0..64 {
            let at = 4.0 * step as f32 + 2.5;
            let value = region.sample(at, at);
            assert_eq!(value, value.round(), "a region id sampled as {value}");
        }
    }

    #[test]
    fn a_field_of_blended_region_columns_is_still_read_smoothly() {
        let terrain = region_document();
        assert!(
            !terrain.field("height").unwrap().is_categorical(),
            "a blended column is a number, not a class"
        );
    }

    #[test]
    fn a_field_is_categorical_only_while_the_layer_saying_so_is_enabled() {
        let field = Field::new("region").with_layer(
            Layer::new(LayerOp::Regions {
                spec: region_spec(),
                output: RegionOutput::CoverClass,
            })
            .disabled(),
        );
        assert!(!field.is_categorical());
    }

    #[test]
    fn a_regions_layer_reports_no_field_dependency() {
        let layer = Layer::new(LayerOp::Regions {
            spec: region_spec(),
            output: RegionOutput::CoverClass,
        });
        assert_eq!(layer.dependencies().count(), 0);
    }

    #[test]
    fn a_region_column_that_is_not_in_the_table_is_an_error() {
        let mut terrain = Terrain::new(UVec2::new(64, 64)).with_field(
            Field::new("height").with_layer(Layer::new(LayerOp::Regions {
                spec: region_spec(),
                output: RegionOutput::Blended("humidity".to_owned()),
            })),
        );
        let error = terrain.bake().unwrap_err();
        assert!(
            matches!(&error, BakeError::UnknownRegionColumn { column, reader }
                if column == "humidity" && reader == "height"),
            "{error}"
        );
    }

    #[test]
    fn a_document_carrying_a_regions_layer_round_trips_through_serde() {
        let terrain = region_document();
        let encoded = serde_json::to_string(&terrain).unwrap();
        let decoded: Terrain = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, terrain);
    }

    #[test]
    fn a_bake_leaves_no_texel_of_a_field_untouched() {
        let mut terrain = Terrain::new(UVec2::new(37, 23)).with_field(
            Field::new("height")
                .with_shift(2)
                .with_layer(Layer::new(LayerOp::Constant(0.5)).with_blend(Blend::Replace)),
        );
        terrain.bake().unwrap();
        let field = terrain.field("height").unwrap();
        assert_eq!(field.baked().size(), UVec2::new(10, 6));
        assert!(field.baked().data().iter().all(|value| *value == 0.5));
    }

    #[test]
    fn a_document_round_trips_through_serde_without_its_bakes() {
        let terrain = two_field_document();
        let encoded = serde_json::to_string(&terrain).unwrap();
        let decoded: Terrain = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, terrain);
        assert!(decoded.field("height").unwrap().baked().is_empty());
    }

    #[test]
    fn a_change_reaches_further_than_the_rectangle_it_was_made_in() {
        let terrain = two_field_document();
        let painted = CellRect::new(UVec2::new(40, 30), UVec2::new(50, 40));
        let reached = terrain.influence_of("moisture", painted);

        assert!(!reached.is_empty());
        assert_eq!(reached.union(painted), reached, "{reached:?}");
        assert!(reached.width() > painted.width());
    }

    /// The two slope modes answer different questions, and on a plane tilted along one
    /// axis the difference is arithmetic rather than a matter of degree: a gradient of a
    /// pure x-slope is that slope, and so is the steepest axis, so a plane cannot tell
    /// them apart. A plane tilted along *both* can — the gradient takes the hypotenuse
    /// where the steepest axis takes the longer leg.
    #[test]
    fn the_two_slope_modes_differ_by_the_hypotenuse_on_a_tilted_plane() {
        let rise = 0.001_f32;
        let plane = |mode| {
            let mut terrain =
                Terrain::new(UVec2::splat(64))
                    .with_field(
                        Field::new("ground")
                            .with_range((-10.0, 10.0))
                            .with_layer(Layer::new(LayerOp::Constant(0.0))),
                    )
                    .with_field(Field::new("tilt").with_range((-10.0, 10.0)).with_layer(
                        Layer::new(LayerOp::Slope {
                            of: FieldId::from("ground"),
                            sample_tiles: 4.0,
                            mode,
                        }),
                    ));
            // A plane written straight into the baked raster, so the slope reads exactly
            // what the arithmetic says and no noise enters the comparison.
            let size = terrain.size;
            let mut raster = Raster::new(size, 0.0);
            for y in 0..size.y {
                for x in 0..size.x {
                    raster.set(x, y, (x as f32 + y as f32) * rise);
                }
            }
            let layer = Layer::new(LayerOp::External(raster)).with_blend(Blend::Replace);
            terrain.field_mut("ground").unwrap().layers.push(layer);
            terrain.bake().unwrap();
            terrain.sample("tilt", 32.5, 32.5).unwrap()
        };

        let gradient = plane(SlopeMode::Gradient);
        let steepest = plane(SlopeMode::SteepestAxis);
        assert!(
            (gradient - rise * 2.0_f32.sqrt()).abs() < 1e-6,
            "a gradient of a doubly tilted plane is {gradient}, not the hypotenuse"
        );
        assert!(
            (steepest - rise).abs() < 1e-6,
            "the steepest axis of a doubly tilted plane is {steepest}, not one leg"
        );
    }

    /// The rectangle a stroke re-bakes is what keeps the document solvable, so it has to
    /// cover every cell a full bake would have written differently — including the ones in
    /// the fields that only *read* the one that changed.
    #[test]
    fn the_rectangle_a_stroke_reaches_covers_every_cell_a_full_bake_would_move() {
        let mut terrain = two_field_document().with_field(Field::new("relief").with_layer(
            Layer::new(LayerOp::Slope {
                of: FieldId::from("height"),
                sample_tiles: 6.0,
                mode: SlopeMode::default(),
            }),
        ));
        let paint = Raster::new(
            terrain.field("moisture").unwrap().resolution(terrain.size),
            0.0,
        );
        terrain
            .field_mut("moisture")
            .unwrap()
            .layers
            .push(Layer::new(LayerOp::Paint(paint)));
        terrain.bake().unwrap();

        let before: Vec<Vec<f32>> = terrain
            .fields
            .iter()
            .map(|field| field.baked().data().to_vec())
            .collect();

        // A stroke into the one field nothing else is, which `height` masks by and `relief`
        // then reads the slope of — so most of what follows is in fields the paint is not in.
        let brush = crate::brush::Brush {
            radius_cells: 14.0,
            falloff: 0.5,
            strength: 0.8,
            ..crate::brush::Brush::default()
        };
        let size = terrain.size;
        let LayerOp::Paint(raster) = &mut terrain.field_mut("moisture").unwrap().layers[1].op
        else {
            panic!("the paint layer stopped being paint");
        };
        let painted = brush.stroke(raster, size, &[glam::Vec2::new(44.0, 34.0)]);
        assert!(!painted.is_empty());

        let reached = terrain.influence_of("moisture", painted);
        terrain.bake().unwrap();

        let mut moved = 0;
        for (index, field) in terrain.fields.iter().enumerate() {
            let resolution = field.baked().size();
            for (texel, (was, now)) in before[index].iter().zip(field.baked().data()).enumerate() {
                if was == now {
                    continue;
                }
                moved += 1;
                let x = texel_center(texel as u32 % resolution.x, field.shift) as u32;
                let y = texel_center(texel as u32 / resolution.x, field.shift) as u32;
                assert!(
                    reached.contains(x, y),
                    "{}: cell {x},{y} moved outside {reached:?}",
                    field.id
                );
            }
        }
        assert!(moved > 0, "the stroke moved nothing at all");
    }

    #[test]
    fn a_change_to_a_field_the_document_does_not_have_reaches_nothing() {
        let terrain = two_field_document();
        assert!(
            terrain
                .influence_of("nowhere", CellRect::new(UVec2::ZERO, UVec2::splat(8)))
                .is_empty()
        );
        assert!(terrain.influence_of("moisture", CellRect::EMPTY).is_empty());
    }
}
