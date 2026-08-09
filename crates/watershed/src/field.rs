use std::fmt;

use glam::UVec2;
use serde::{Deserialize, Serialize};

use crate::layer::{Layer, LayerOp};
use crate::raster::{Raster, raster_coord, resolution};
use crate::regions::RegionOutput;

/// TODO(jb-doc): why a field is named rather than numbered, and what that costs against
/// what it buys in a header a person edits.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FieldId(String);

impl FieldId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for FieldId {
    fn from(name: &str) -> Self {
        Self(name.to_owned())
    }
}

impl From<String> for FieldId {
    fn from(name: String) -> Self {
        Self(name)
    }
}

impl AsRef<str> for FieldId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for FieldId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// TODO(jb-doc): what a field is — a named stack evaluated onto its own raster — and the
/// three things a caller has to choose: its resolution, its range, and its layers.
///
/// TODO(jb-doc): why `baked` is not part of the serialized document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Field {
    pub id: FieldId,
    pub shift: u8,
    pub range: (f32, f32),
    pub layers: Vec<Layer>,
    #[serde(skip)]
    baked: Raster<f32>,
}

impl Field {
    pub fn new(id: impl Into<FieldId>) -> Self {
        Self {
            id: id.into(),
            shift: 0,
            range: (0.0, 1.0),
            layers: Vec::new(),
            baked: Raster::default(),
        }
    }

    pub fn with_shift(mut self, shift: u8) -> Self {
        self.shift = shift;
        self
    }

    pub fn with_range(mut self, range: (f32, f32)) -> Self {
        self.range = range;
        self
    }

    pub fn with_layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }

    /// TODO(jb-comment): why the range is sorted here rather than being rejected as
    /// invalid when it is set.
    pub fn bounds(&self) -> (f32, f32) {
        if self.range.0 <= self.range.1 {
            self.range
        } else {
            (self.range.1, self.range.0)
        }
    }

    pub fn resolution(&self, size: UVec2) -> UVec2 {
        resolution(size, self.shift)
    }

    pub fn baked(&self) -> &Raster<f32> {
        &self.baked
    }

    pub(crate) fn baked_mut(&mut self) -> &mut Raster<f32> {
        &mut self.baked
    }

    pub(crate) fn take_baked(&mut self) -> Raster<f32> {
        std::mem::take(&mut self.baked)
    }

    pub(crate) fn put_baked(&mut self, raster: Raster<f32>) {
        self.baked = raster;
    }

    /// TODO(jb-doc): why this is derived from the ops rather than declared beside the
    /// shift — that it follows an op parameter in the same edit that changes it, and that
    /// a document written before it existed needs no migration.
    ///
    /// TODO(jb-comment): why a disabled layer does not make a field categorical, on the
    /// same terms as [`Field::dependencies`].
    pub fn is_categorical(&self) -> bool {
        self.layers
            .iter()
            .filter(|layer| layer.enabled)
            .any(|layer| {
                matches!(
                    &layer.op,
                    LayerOp::Regions {
                        output: RegionOutput::RegionId | RegionOutput::CoverClass,
                        ..
                    }
                )
            })
    }

    /// TODO(jb-doc): the coordinate space this takes — document cells, a cell centre at
    /// `x + 0.5` — and why a coarse field is read bilinearly rather than as blocks.
    ///
    /// TODO(jb-doc): the exception, and what the value halfway between two region ids
    /// would name if it were interpolated.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let u = raster_coord(x, self.shift);
        let v = raster_coord(y, self.shift);
        if self.is_categorical() {
            self.baked.sample_nearest(u, v)
        } else {
            self.baked.sample_bilinear(u, v)
        }
    }

    /// TODO(jb-comment): why a disabled layer contributes no dependency, and what that
    /// means for a cycle that only exists while a layer is switched off.
    pub fn dependencies(&self) -> impl Iterator<Item = &FieldId> {
        self.layers
            .iter()
            .filter(|layer| layer.enabled)
            .flat_map(|layer| layer.dependencies())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{LayerOp, Mask, Remap};

    #[test]
    fn a_field_id_round_trips_through_every_way_of_making_one() {
        assert_eq!(FieldId::from("height").as_str(), "height");
        assert_eq!(FieldId::new("height"), FieldId::from("height".to_owned()));
        assert_eq!(FieldId::from("height").to_string(), "height");
    }

    #[test]
    fn a_field_at_shift_zero_holds_one_texel_per_cell() {
        let field = Field::new("height");
        assert_eq!(field.resolution(UVec2::new(64, 32)), UVec2::new(64, 32));
    }

    #[test]
    fn a_coarse_field_holds_one_texel_per_block() {
        let field = Field::new("moisture").with_shift(4);
        assert_eq!(
            field.resolution(UVec2::new(4096, 4096)),
            UVec2::new(256, 256)
        );
    }

    #[test]
    fn a_backwards_range_is_read_in_the_order_a_clamp_needs() {
        let field = Field::new("height").with_range((1.0, -1.0));
        assert_eq!(field.bounds(), (-1.0, 1.0));
    }

    #[test]
    fn a_field_reports_the_dependencies_of_every_enabled_layer_and_no_others() {
        let field = Field::new("height")
            .with_layer(Layer::new(LayerOp::Constant(0.5)))
            .with_layer(
                Layer::new(LayerOp::FieldRef(FieldId::from("relief")))
                    .with_mask(Mask::Field(FieldId::from("ridge"), Remap::IDENTITY)),
            )
            .with_layer(Layer::new(LayerOp::FieldRef(FieldId::from("hidden"))).disabled());
        let deps: Vec<_> = field.dependencies().map(|id| id.as_str()).collect();
        assert_eq!(deps, vec!["relief", "ridge"]);
    }

    #[test]
    fn an_unbaked_field_samples_as_zero() {
        let field = Field::new("height");
        assert_eq!(field.sample(12.5, 3.5), 0.0);
    }
}
