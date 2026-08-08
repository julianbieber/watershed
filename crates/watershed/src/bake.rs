use std::collections::HashMap;

use glam::{UVec2, Vec2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::field::{Field, FieldId};
use crate::layer::{Blend, LayerOp, Mask, Remap};
use crate::noise::Noise;
use crate::raster::{CellRect, Raster, raster_coord, resolution, step, texel_center};
use crate::regions::{CompiledOutput, RegionMap, RegionOutput};

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

/// TODO(jb-doc): what a terrain is here — a size, a set of named fields, and nothing
/// else that the caller does not own.
///
/// TODO(jb-comment): stage 4 adds `water: Option<WaterState>` beside the fields.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Terrain {
    pub size: UVec2,
    pub fields: Vec<Field>,
}

impl Terrain {
    pub fn new(size: UVec2) -> Self {
        Self {
            size,
            fields: Vec::new(),
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
        let mut baked: Vec<Raster<f32>> = self
            .fields
            .iter_mut()
            .map(|field| field.take_baked())
            .collect();

        for &target in &order {
            let field = &self.fields[target];
            let texels = required[target].to_texels(field.shift, baked[target].size());
            if texels.is_empty() {
                continue;
            }
            let layers = compile_layers(field, &index_of, self.size)?;
            let bounds = field.bounds();
            let shift = field.shift;
            let rows = {
                let context = Evaluator {
                    size: self.size,
                    baked: &baked,
                    shifts: &shifts,
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
        }

        for (field, raster) in self.fields.iter_mut().zip(baked) {
            field.put_baked(raster);
        }
        Ok(())
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
            let field = &self.fields[index];
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
            for &referenced in &dependencies[index] {
                let halo = step(field.shift) + reach + 2 * step(self.fields[referenced].shift) + 2;
                let widened = required[index].expand(halo).intersect(document);
                required[referenced] = required[referenced].union(widened);
            }
        }
        required
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
    Slope { of: usize, sample_tiles: f32 },
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
            LayerOp::Slope { of, sample_tiles } => CompiledOp::Slope {
                of: lookup(index_of, of, &field.id)?,
                sample_tiles: *sample_tiles,
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
}

impl Evaluator<'_> {
    fn field(&self, index: usize, position: Vec2) -> f32 {
        let shift = self.shifts[index];
        self.baked[index].sample_bilinear(
            raster_coord(position.x, shift),
            raster_coord(position.y, shift),
        )
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
    fn slope(&self, index: usize, sample_tiles: f32, position: Vec2) -> f32 {
        let reach = sample_tiles.abs().max(f32::EPSILON);
        let dx = self.field(index, position + Vec2::new(reach, 0.0))
            - self.field(index, position - Vec2::new(reach, 0.0));
        let dy = self.field(index, position + Vec2::new(0.0, reach))
            - self.field(index, position - Vec2::new(0.0, reach));
        let scale = 2.0 * reach;
        ((dx / scale).powi(2) + (dy / scale).powi(2)).sqrt()
    }

    fn value(&self, op: &CompiledOp<'_>, position: Vec2) -> f32 {
        match op {
            CompiledOp::Constant(value) => *value,
            CompiledOp::Noise(noise) => noise.sample(position.x, position.y),
            CompiledOp::Raster(raster) => raster.sample_over(self.size, position.x, position.y),
            CompiledOp::Slope { of, sample_tiles } => self.slope(*of, *sample_tiles, position),
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
}
