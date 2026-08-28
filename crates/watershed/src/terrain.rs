//! The read side of a terrain: what a consuming project holds once a document has
//! been baked, and how it addresses and reads it.

use std::collections::HashMap;

use glam::UVec2;

use crate::field::FieldRole;
use crate::raster::{Raster, raster_coord};
use crate::water::WaterState;

/// Everything about a baked field except its values: what the bake settled once so
/// that reading a texel needs no further reference to the document.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldInfo {
    /// The [`FieldId`](crate::field::FieldId) the field was declared under, and the
    /// key [`Terrain::field`] looks up.
    pub name: String,
    /// The role the field held in the spec.
    pub role: FieldRole,
    /// The [`raster`](crate::raster) shift the field was baked at, which is what
    /// relates a cell to a texel.
    pub shift: u8,
    /// Low end of the interval values were clamped into. Already sorted, whatever
    /// order the spec stated its range in.
    pub range_low: f32,
    /// High end of that interval.
    pub range_high: f32,
    /// Whether values name a class rather than measure a quantity, which decides
    /// how [`FieldView::sample`] interpolates. Settled from the spec's layers at
    /// bake time, so a read does not have to re-derive it.
    pub categorical: bool,
}

/// A baked terrain: an extent, a set of named fields, and optionally a solved water
/// state.
///
/// Nothing that produced it survives here — no layers, no noise specs, no plan — so
/// a consuming project cannot re-bake from a `Terrain` and does not have to carry
/// the machinery that would let it. What it may assume instead is that every field
/// [`Terrain::fields`] names is readable at every cell inside the extent: a field
/// whose raster is missing is dropped from the listing rather than answering
/// nothing.
#[derive(Clone, Debug, Default)]
pub struct Terrain {
    pub(crate) size: UVec2,
    pub(crate) fields: Vec<FieldInfo>,
    pub(crate) baked: HashMap<String, Raster<f32>>,
    pub(crate) water: Option<WaterState>,
}

impl Terrain {
    /// Cells on the x axis.
    pub fn width(&self) -> u32 {
        self.size.x
    }

    /// Cells on the y axis.
    pub fn height(&self) -> u32 {
        self.size.y
    }

    /// The extent in cells. Every field covers all of it, whatever its own shift.
    pub fn size(&self) -> UVec2 {
        self.size
    }

    /// Every readable field, in the order the spec declared them — not bake order
    /// and not hash order, so a project may index by position and get the same field
    /// back across runs.
    ///
    /// A declared field with no baked raster is skipped, so the listing is exactly
    /// what can be read.
    pub fn fields(&self) -> impl Iterator<Item = FieldView<'_>> {
        self.fields.iter().filter_map(|info| self.view_of(info))
    }

    /// The field of that exact name, or `None` if the terrain does not carry one or
    /// it was not baked. Case-sensitive.
    pub fn field(&self, name: &str) -> Option<FieldView<'_>> {
        self.fields
            .iter()
            .find(|info| info.name == name)
            .and_then(|info| self.view_of(info))
    }

    /// The field holding `role`, for a project that wants the height of a document
    /// it did not author.
    ///
    /// At most one field can hold [`FieldRole::Height`] or [`FieldRole::Moisture`] —
    /// a document holding two is rejected before it is ever baked — so the answer is
    /// unambiguous. [`FieldRole::Custom`] is carried by any number of fields and
    /// always resolves to `None`.
    pub fn field_with_role(&self, role: FieldRole) -> Option<FieldView<'_>> {
        if role == FieldRole::Custom {
            return None;
        }
        self.fields
            .iter()
            .find(|info| info.role == role)
            .and_then(|info| self.view_of(info))
    }

    /// The solved water, present only if the spec declared some and the water step
    /// of the bake ran.
    pub fn water(&self) -> Option<&WaterState> {
        self.water.as_ref()
    }

    fn view_of<'a>(&'a self, info: &'a FieldInfo) -> Option<FieldView<'a>> {
        let raster = self.baked.get(&info.name)?;
        Some(FieldView {
            info,
            raster,
            size: self.size,
        })
    }
}

/// A resolved handle to one field of a [`Terrain`], borrowing its metadata and its
/// texels.
///
/// `Copy` and small — two borrows and the extent — so resolving once outside a loop
/// and reading through it copies no part of the grid. It borrows the terrain, which
/// therefore cannot be modified while any view of it is alive.
#[derive(Clone, Copy, Debug)]
pub struct FieldView<'a> {
    info: &'a FieldInfo,
    raster: &'a Raster<f32>,
    size: UVec2,
}

impl<'a> FieldView<'a> {
    /// The field's name in the document.
    pub fn name(&self) -> &'a str {
        &self.info.name
    }

    /// The role the field was declared with.
    pub fn role(&self) -> FieldRole {
        self.info.role
    }

    /// The shift the field was baked at: one texel per `2^shift` cells on each axis.
    pub fn shift(&self) -> u8 {
        self.info.shift
    }

    /// Whether [`FieldView::sample`] reads the nearest texel rather than
    /// interpolating. See [`FieldInfo::categorical`].
    pub fn is_categorical(&self) -> bool {
        self.info.categorical
    }

    /// Low end of the interval every value is inside.
    pub fn range_low(&self) -> f32 {
        self.info.range_low
    }

    /// High end of the interval every value is inside.
    pub fn range_high(&self) -> f32 {
        self.info.range_high
    }

    /// Columns of the underlying raster — the terrain's width only at shift 0.
    pub fn texel_width(&self) -> u32 {
        self.raster.width()
    }

    /// Rows of the underlying raster — the terrain's height only at shift 0.
    pub fn texel_height(&self) -> u32 {
        self.raster.height()
    }

    /// The value at an integer *cell*, in the terrain's own grid whatever the
    /// field's shift: every cell of the block a texel covers reads that texel.
    ///
    /// `None` outside the extent — this is the read that refuses rather than
    /// clamping. Use [`FieldView::sample`] for the clamping one.
    pub fn value_at(&self, x: u32, y: u32) -> Option<f32> {
        if x >= self.size.x || y >= self.size.y {
            return None;
        }
        let shift = self.info.shift;
        self.raster.get(x >> shift, y >> shift).copied()
    }

    /// The value at a continuous position in cells, where a cell centre is at
    /// `x + 0.5`.
    ///
    /// Clamps to the extent instead of failing, so a position outside the terrain
    /// reads its nearest edge. Interpolated between texels, or read to the nearest
    /// one for a [categorical](FieldView::is_categorical) field.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let u = raster_coord(x, self.info.shift);
        let v = raster_coord(y, self.info.shift);
        if self.info.categorical {
            self.raster.sample_nearest(u, v)
        } else {
            self.raster.sample_bilinear(u, v)
        }
    }

    /// The raw texels in row-major order, `texel_width * texel_height` of them —
    /// for uploading a field to a GPU or writing it out, not for point reads.
    pub fn texels(&self) -> &'a [f32] {
        self.raster.data()
    }
}

#[cfg(test)]
mod tests {
    use crate::bake::TerrainSpec;
    use crate::field::{Field, FieldRole};
    use crate::layer::{Blend, Layer, LayerOp};

    use super::*;

    fn baked() -> Terrain {
        TerrainSpec::new(UVec2::new(64, 32))
            .with_field(
                Field::new("height")
                    .with_role(FieldRole::Height)
                    .with_layer(Layer::new(LayerOp::Constant(0.25)).with_blend(Blend::Replace)),
            )
            .with_field(
                Field::new("moisture")
                    .with_role(FieldRole::Moisture)
                    .with_shift(4)
                    .with_layer(Layer::new(LayerOp::Constant(0.5)).with_blend(Blend::Replace)),
            )
            .bake()
            .unwrap()
    }

    // The extent is what every cell read is bounds-checked against, and it is the one
    // thing a coarse field must not be able to change.
    #[test]
    fn a_baked_terrain_answers_the_extent_the_spec_declared() {
        let terrain = baked();
        assert_eq!((terrain.width(), terrain.height()), (64, 32));
    }

    // The order is what a consuming project reads its fields back in, so it is part
    // of the contract rather than an artefact of the map the bake filled.
    #[test]
    fn fields_come_back_in_the_order_the_spec_declared_them() {
        let terrain = baked();
        let names: Vec<_> = terrain
            .fields()
            .map(|view| view.name().to_owned())
            .collect();
        assert_eq!(names, vec!["height", "moisture"]);
    }

    // Role lookup is how a project finds the height of a document it did not author,
    // so it has to work off the declared role rather than off a conventional name.
    #[test]
    fn a_role_resolves_to_the_one_field_holding_it() {
        let terrain = baked();
        let height = terrain.field_with_role(FieldRole::Height).unwrap();
        assert_eq!(height.name(), "height");
        assert_eq!(
            terrain.field_with_role(FieldRole::Moisture).unwrap().name(),
            "moisture"
        );
    }

    // Custom is held by any number of fields, so it is the one role a lookup cannot
    // answer with a single view.
    #[test]
    fn the_custom_role_resolves_to_no_field() {
        assert!(baked().field_with_role(FieldRole::Custom).is_none());
    }

    // A cell read is in the terrain's own grid whatever the field's shift, so a
    // coarse field answers at every cell of the block its texel covers.
    #[test]
    fn every_cell_of_a_block_reads_the_texel_that_covers_it() {
        let terrain = baked();
        let moisture = terrain.field("moisture").unwrap();
        assert_eq!(moisture.texel_width(), 4);
        for (x, y) in [(0, 0), (15, 15), (3, 12)] {
            assert_eq!(moisture.value_at(x, y), Some(0.5));
        }
    }

    // `value_at` bounds-checks against the extent, not against the raster, so a
    // coarse field must not accept the cells past the last one it has a texel for.
    #[test]
    fn a_cell_outside_the_extent_reads_nothing() {
        let terrain = baked();
        let height = terrain.field("height").unwrap();
        assert_eq!(height.value_at(63, 31), Some(0.25));
        assert_eq!(height.value_at(64, 0), None);
        assert_eq!(height.value_at(0, 32), None);
    }

    // A position read clamps where a cell read refuses — the one place the two
    // spellings deliberately differ.
    #[test]
    fn a_position_read_clamps_to_the_extent() {
        let terrain = baked();
        let height = terrain.field("height").unwrap();
        assert_eq!(height.sample(-40.0, -40.0), 0.25);
        assert_eq!(height.sample(4000.0, 4000.0), 0.25);
    }

    // Lookup is by exact name and there is no fallback, so a misspelling has to be a
    // `None` rather than a neighbouring field.
    #[test]
    fn a_name_the_terrain_does_not_carry_resolves_to_nothing() {
        assert!(baked().field("elevation").is_none());
    }

    // A view carries the metadata the bake settled, so a project reading through one
    // never needs the spec that produced the terrain.
    #[test]
    fn a_view_reports_the_range_and_shift_its_field_declared() {
        let terrain = baked();
        let moisture = terrain.field("moisture").unwrap();
        assert_eq!(moisture.shift(), 4);
        assert_eq!((moisture.range_low(), moisture.range_high()), (0.0, 1.0));
        assert_eq!(moisture.role(), FieldRole::Moisture);
    }
}
