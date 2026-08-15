// TODO(jb-doc): what a terrain is once it is baked — the grids and the names, and nothing
// that built them — and why it is a separate type from the spec it came from.

use std::collections::HashMap;

use glam::UVec2;

use crate::field::FieldRole;
use crate::raster::{Raster, raster_coord};
use crate::water::WaterState;

/// TODO(jb-doc): why categorical is settled here rather than re-derived per read.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldInfo {
    pub name: String,
    pub role: FieldRole,
    pub shift: u8,
    pub range_low: f32,
    pub range_high: f32,
    pub categorical: bool,
}

/// TODO(jb-doc): what a baked terrain carries, and what a consuming project may assume of
/// it — that every field it names is readable at every cell inside the extent.
#[derive(Clone, Debug, Default)]
pub struct Terrain {
    pub(crate) size: UVec2,
    pub(crate) fields: Vec<FieldInfo>,
    pub(crate) baked: HashMap<String, Raster<f32>>,
    pub(crate) water: Option<WaterState>,
}

impl Terrain {
    pub fn width(&self) -> u32 {
        self.size.x
    }

    pub fn height(&self) -> u32 {
        self.size.y
    }

    pub fn size(&self) -> UVec2 {
        self.size
    }

    /// TODO(jb-doc): why the order is the order the spec declared, and what a consuming
    /// project is entitled to read into it.
    pub fn fields(&self) -> impl Iterator<Item = FieldView<'_>> {
        self.fields.iter().filter_map(|info| self.view_of(info))
    }

    pub fn field(&self, name: &str) -> Option<FieldView<'_>> {
        self.fields
            .iter()
            .find(|info| info.name == name)
            .and_then(|info| self.view_of(info))
    }

    /// TODO(jb-doc): why this answers at most one field, and which spec check makes that
    /// true before a bake ever runs.
    pub fn field_with_role(&self, role: FieldRole) -> Option<FieldView<'_>> {
        if role == FieldRole::Custom {
            return None;
        }
        self.fields
            .iter()
            .find(|info| info.role == role)
            .and_then(|info| self.view_of(info))
    }

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

/// TODO(jb-doc): why a consuming project resolves a view once and reads through it, and
/// what the view borrows.
#[derive(Clone, Copy, Debug)]
pub struct FieldView<'a> {
    info: &'a FieldInfo,
    raster: &'a Raster<f32>,
    size: UVec2,
}

impl<'a> FieldView<'a> {
    pub fn name(&self) -> &'a str {
        &self.info.name
    }

    pub fn role(&self) -> FieldRole {
        self.info.role
    }

    pub fn shift(&self) -> u8 {
        self.info.shift
    }

    pub fn is_categorical(&self) -> bool {
        self.info.categorical
    }

    pub fn range_low(&self) -> f32 {
        self.info.range_low
    }

    pub fn range_high(&self) -> f32 {
        self.info.range_high
    }

    pub fn texel_width(&self) -> u32 {
        self.raster.width()
    }

    pub fn texel_height(&self) -> u32 {
        self.raster.height()
    }

    /// returns None if the coordinates are outside the range where the terrain is defined.
    pub fn value_at(&self, x: u32, y: u32) -> Option<f32> {
        if x >= self.size.x || y >= self.size.y {
            return None;
        }
        let shift = self.info.shift;
        self.raster.get(x >> shift, y >> shift).copied()
    }

    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let u = raster_coord(x, self.info.shift);
        let v = raster_coord(y, self.info.shift);
        if self.info.categorical {
            self.raster.sample_nearest(u, v)
        } else {
            self.raster.sample_bilinear(u, v)
        }
    }

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

    #[test]
    fn a_baked_terrain_answers_the_extent_the_spec_declared() {
        let terrain = baked();
        assert_eq!((terrain.width(), terrain.height()), (64, 32));
    }

    /// The order is what a consuming project reads its fields back in, so it is part of
    /// the contract rather than an artefact of the map the bake filled.
    #[test]
    fn fields_come_back_in_the_order_the_spec_declared_them() {
        let terrain = baked();
        let names: Vec<_> = terrain
            .fields()
            .map(|view| view.name().to_owned())
            .collect();
        assert_eq!(names, vec!["height", "moisture"]);
    }

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

    /// Custom is held by any number of fields, so it is the one role a lookup cannot
    /// answer with a single view.
    #[test]
    fn the_custom_role_resolves_to_no_field() {
        assert!(baked().field_with_role(FieldRole::Custom).is_none());
    }

    /// A cell read is in the terrain's own grid whatever the field's shift, so a coarse
    /// field answers at every cell the block its texel covers.
    #[test]
    fn every_cell_of_a_block_reads_the_texel_that_covers_it() {
        let terrain = baked();
        let moisture = terrain.field("moisture").unwrap();
        assert_eq!(moisture.texel_width(), 4);
        for (x, y) in [(0, 0), (15, 15), (3, 12)] {
            assert_eq!(moisture.value_at(x, y), Some(0.5));
        }
    }

    #[test]
    fn a_cell_outside_the_extent_reads_nothing() {
        let terrain = baked();
        let height = terrain.field("height").unwrap();
        assert_eq!(height.value_at(63, 31), Some(0.25));
        assert_eq!(height.value_at(64, 0), None);
        assert_eq!(height.value_at(0, 32), None);
    }

    /// A position read clamps where a cell read refuses — the one place the two spellings
    /// deliberately differ.
    #[test]
    fn a_position_read_clamps_to_the_extent() {
        let terrain = baked();
        let height = terrain.field("height").unwrap();
        assert_eq!(height.sample(-40.0, -40.0), 0.25);
        assert_eq!(height.sample(4000.0, 4000.0), 0.25);
    }

    #[test]
    fn a_name_the_terrain_does_not_carry_resolves_to_nothing() {
        assert!(baked().field("elevation").is_none());
    }

    /// A view is a borrow, so what a consuming project holds per field is a pointer and a
    /// shift rather than a copy of the grid.
    #[test]
    fn a_view_reports_the_range_and_shift_its_field_declared() {
        let terrain = baked();
        let moisture = terrain.field("moisture").unwrap();
        assert_eq!(moisture.shift(), 4);
        assert_eq!((moisture.range_low(), moisture.range_high()), (0.0, 1.0));
        assert_eq!(moisture.role(), FieldRole::Moisture);
    }
}
