//! A named stack of layers, and the raster the stack bakes onto.

use glam::UVec2;
use serde::{Deserialize, Serialize};

use watershed::field::{FieldId, FieldRole};
use watershed::raster::{Raster, raster_coord, resolution};

use crate::terrain::layer::{Layer, LayerOp};
use crate::terrain::regions::RegionOutput;

/// A named stack of layers together with everything needed to evaluate it onto its
/// own raster, plus that raster once it has been baked.
///
/// The baked raster is not serialized: it is derived from the layers and is
/// re-obtained by baking, so a loaded document starts with every field empty and
/// sampling as `0.0`. Equality does compare it, so two fields differing only in
/// bake state are not equal.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Field {
    /// The name this field is referenced by. Changing it does not rewrite the
    /// layers of other fields that reference the old name.
    pub id: FieldId,
    /// What the bake may do with the field. See [`FieldRole`] for the constraints
    /// a document carrying this has to satisfy.
    #[serde(default)]
    pub role: FieldRole,
    /// Resolution, as the [`raster`](watershed::raster) shift: one texel per cell at 0,
    /// per `2^shift` cells above that.
    pub shift: u8,
    /// The interval baked values are clamped into. Accepted in either order; read
    /// it through [`Field::bounds`] rather than directly.
    pub range: (f32, f32),
    /// Carried through the document and never read by this crate. It is for
    /// whatever consumes a baked terrain to decide what an exported field means.
    #[serde(default)]
    pub export: bool,
    /// Evaluated in order, each blended onto the result of the ones before it.
    pub layers: Vec<Layer>,
    #[serde(skip)]
    baked: Raster<f32>,
}

impl Field {
    /// A field named `id` with no layers: role `Custom`, shift 0, range `0.0..=1.0`,
    /// not exported, and unbaked — so it samples as `0.0` until it is baked.
    pub fn new(id: impl Into<FieldId>) -> Self {
        Self {
            id: id.into(),
            role: FieldRole::Custom,
            shift: 0,
            range: (0.0, 1.0),
            export: false,
            layers: Vec::new(),
            baked: Raster::default(),
        }
    }

    /// Sets the role. Does not check the document-wide constraints on
    /// [`FieldRole`]; a conflict surfaces at plan time.
    pub fn with_role(mut self, role: FieldRole) -> Self {
        self.role = role;
        self
    }

    /// Sets [`Field::export`].
    pub fn with_export(mut self, export: bool) -> Self {
        self.export = export;
        self
    }

    /// Sets the resolution shift. A shift on the `Height` field is rejected at plan
    /// time, not here.
    pub fn with_shift(mut self, shift: u8) -> Self {
        self.shift = shift;
        self
    }

    /// Sets the clamp interval. Either order is accepted — see [`Field::bounds`].
    pub fn with_range(mut self, range: (f32, f32)) -> Self {
        self.range = range;
        self
    }

    /// Appends a layer on top of the ones already there. Order is evaluation order.
    pub fn with_layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }

    /// [`Field::range`] as `(low, high)`, swapped if it was stored backwards. A
    /// backwards range is not an error anywhere; this is the only correct way to
    /// read it.
    pub fn bounds(&self) -> (f32, f32) {
        if self.range.0 <= self.range.1 {
            self.range
        } else {
            (self.range.1, self.range.0)
        }
    }

    /// Texel dimensions this field bakes to in a `size`-cell document. Never zero
    /// on either axis.
    pub fn resolution(&self, size: UVec2) -> UVec2 {
        resolution(size, self.shift)
    }

    /// The values from the last bake. Empty for a field that has never been baked,
    /// was released, or came from a loaded document — read it through
    /// [`Field::sample`] unless the texel grid itself is what you want.
    pub fn baked(&self) -> &Raster<f32> {
        &self.baked
    }

    /// The baked raster, writable in place. Its dimensions are the bake's
    /// invariant, so a caller replacing texels must not change its length.
    pub(crate) fn baked_mut(&mut self) -> &mut Raster<f32> {
        &mut self.baked
    }

    /// Moves the baked raster out, leaving the field unbaked and sampling as
    /// `0.0`. Pair with [`Field::put_baked`] to borrow a raster across a bake that
    /// needs the rest of the document mutably.
    pub(crate) fn take_baked(&mut self) -> Raster<f32> {
        std::mem::take(&mut self.baked)
    }

    /// Installs `raster` as the bake result, dropping whatever was there. Nothing
    /// checks it against [`Field::resolution`].
    pub(crate) fn put_baked(&mut self, raster: Raster<f32>) {
        self.baked = raster;
    }

    /// Whether this field's values name a class rather than measure a quantity,
    /// which is what decides how [`Field::sample`] interpolates.
    ///
    /// Derived from the layers, not declared: true when an *enabled* layer emits
    /// region ids or cover classes. Disabling that layer makes the field
    /// non-categorical again, so the answer can change without the shift or the
    /// range changing.
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

    /// The baked value at a position in document cells, where a cell centre is at
    /// `x + 0.5`. Reads `0.0` for an unbaked field, and clamps rather than failing
    /// outside the document.
    ///
    /// A coarse field is interpolated between its texels, so it reads as a smooth
    /// surface and not as blocks. A [categorical](Field::is_categorical) field is
    /// read to the nearest texel instead: halfway between two region ids is not a
    /// third region, so it must be one of the two.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let u = raster_coord(x, self.shift);
        let v = raster_coord(y, self.shift);
        if self.is_categorical() {
            self.baked.sample_nearest(u, v)
        } else {
            self.baked.sample_bilinear(u, v)
        }
    }

    /// The fields this one reads, through the ops and masks of its *enabled* layers
    /// only; disabling a layer removes its dependencies.
    ///
    /// This is what bake ordering and cycle detection run on, so a cycle that exists
    /// only through a disabled layer is not a cycle and the document plans.
    /// Duplicates are not removed and the order is the layer order.
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
    use crate::terrain::layer::{Mask, Remap};

    // Shift 0 is what the `Height` field is pinned to, so a document's cell grid and
    // its height raster have to be the same grid.
    #[test]
    fn a_field_at_shift_zero_holds_one_texel_per_cell() {
        let field = Field::new("height");
        assert_eq!(field.resolution(UVec2::new(64, 32)), UVec2::new(64, 32));
    }

    // The point of a shift is the allocation it saves; this pins that the saving is
    // quadratic in the shift and not linear.
    #[test]
    fn a_coarse_field_holds_one_texel_per_block() {
        let field = Field::new("moisture").with_shift(4);
        assert_eq!(
            field.resolution(UVec2::new(4096, 4096)),
            UVec2::new(256, 256)
        );
    }

    // Nothing rejects a backwards range, so `bounds` is the only thing standing
    // between one and a clamp that empties the field.
    #[test]
    fn a_backwards_range_is_read_in_the_order_a_clamp_needs() {
        let field = Field::new("height").with_range((1.0, -1.0));
        assert_eq!(field.bounds(), (-1.0, 1.0));
    }

    // Pins both halves of what bake ordering is computed from: a mask contributes a
    // dependency just as an op does, and a disabled layer contributes none.
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

    // Every loaded document is in this state until it is baked, so sampling one has
    // to be a defined read rather than a panic on an empty raster.
    #[test]
    fn an_unbaked_field_samples_as_zero() {
        let field = Field::new("height");
        assert_eq!(field.sample(12.5, 3.5), 0.0);
    }
}
