//! A named raster, the shader that produces its values, and the raster it bakes onto.

use glam::UVec2;
use serde::{Deserialize, Serialize};

use watershed::raster::{Raster, raster_coord, resolution};

use crate::terrain::shader::ShaderLayer;

/// A layer's name: the library's [`FieldId`](watershed::field::FieldId) under the editor's
/// word. Any string is accepted; uniqueness is checked when a document is planned.
pub type LayerId = watershed::field::FieldId;

/// What a bake may do with a layer: the library's `FieldRole`, under the word the
/// editor uses.
pub type LayerRole = watershed::field::FieldRole;

/// A named layer together with everything needed to bake its shader onto its own
/// raster, plus that raster once it has been baked.
///
/// The baked raster is not serialized: it is derived from the shader and is
/// re-obtained by baking, so a loaded document starts with every layer empty and
/// sampling as `0.0`. Equality does compare it, so two layers differing only in
/// bake state are not equal.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    /// The name this layer is referenced by, and the stem of its shader file.
    /// Assigning it rewrites neither the files of other layers that read the old name
    /// nor the water spec that names it.
    pub id: LayerId,
    /// What the bake may do with the layer. See [`LayerRole`] for the constraints
    /// a document carrying this has to satisfy.
    #[serde(default)]
    pub role: LayerRole,
    /// Resolution, as the [`raster`](watershed::raster) shift: one texel per cell at 0,
    /// per `2^shift` cells above that.
    pub shift: u8,
    /// The interval baked values are clamped into. Accepted in either order; read
    /// it through [`Layer::bounds`] rather than directly.
    pub range: (f32, f32),
    /// Carried through the document and never read by this crate. It is for
    /// whatever consumes a baked terrain to decide what an exported layer means.
    #[serde(default)]
    pub export: bool,
    /// Whether the map lights this layer's colour ramp by the layer's own normal.
    ///
    /// Nothing a bake reads looks at this or at [`Layer::light_azimuth`], so
    /// toggling either never makes the bake stale.
    #[serde(default)]
    pub hillshade: bool,
    /// The compass bearing the hillshade light comes from, in degrees: 0 is north
    /// and 90 is east. Read only while [`Layer::hillshade`] is set. Defaults to the
    /// northwest.
    #[serde(default = "default_light_azimuth")]
    pub light_azimuth: f32,
    /// Whether the map draws an iso-line at every multiple of
    /// [`Layer::contour_interval`] over this layer's colour ramp.
    ///
    /// Nothing a bake reads looks at this or at [`Layer::contour_interval`], so
    /// toggling either never makes the bake stale.
    #[serde(default)]
    pub contours: bool,
    /// The spacing between iso-lines, in the layer's own units. Read only while
    /// [`Layer::contours`] is set. Defaults to a tenth, which divides the default
    /// range into ten bands.
    #[serde(default = "default_contour_interval")]
    pub contour_interval: f32,
    /// The parameter values and the layers read of the shader file that produces this
    /// layer's values. Evaluation order is derived from the layers read, not stored.
    pub shader: ShaderLayer,
    #[serde(skip)]
    baked: Raster<f32>,
}

const DEFAULT_LIGHT_AZIMUTH: f32 = 315.0;
const DEFAULT_CONTOUR_INTERVAL: f32 = 0.1;

fn default_light_azimuth() -> f32 {
    DEFAULT_LIGHT_AZIMUTH
}

fn default_contour_interval() -> f32 {
    DEFAULT_CONTOUR_INTERVAL
}

impl Layer {
    /// A layer named `id` with no parameter values: role `Custom`, shift 0, range
    /// `0.0..=1.0`, not exported, not hillshaded, lit from the northwest, without
    /// contours at a tenth-unit interval, and unbaked — so it samples as `0.0`
    /// until it is baked.
    pub fn new(id: impl Into<LayerId>) -> Self {
        Self {
            id: id.into(),
            role: LayerRole::Custom,
            shift: 0,
            range: (0.0, 1.0),
            export: false,
            hillshade: false,
            light_azimuth: DEFAULT_LIGHT_AZIMUTH,
            contours: false,
            contour_interval: DEFAULT_CONTOUR_INTERVAL,
            shader: ShaderLayer::default(),
            baked: Raster::default(),
        }
    }

    /// Sets the role. Does not check the document-wide constraints on
    /// [`LayerRole`]; a conflict surfaces at plan time.
    pub fn with_role(mut self, role: LayerRole) -> Self {
        self.role = role;
        self
    }

    /// Sets [`Layer::export`].
    pub fn with_export(mut self, export: bool) -> Self {
        self.export = export;
        self
    }

    /// Sets the resolution shift. A shift on the `Height` layer is rejected at plan
    /// time, not here.
    pub fn with_shift(mut self, shift: u8) -> Self {
        self.shift = shift;
        self
    }

    /// Sets the clamp interval. Either order is accepted — see [`Layer::bounds`].
    pub fn with_range(mut self, range: (f32, f32)) -> Self {
        self.range = range;
        self
    }

    /// The shader file this layer's values come from: a plain name inside the
    /// document's `shaders` directory, never a path.
    pub fn file(&self) -> String {
        format!("{}.wgsl", self.id)
    }

    /// A copy of everything a person authored and nothing that was derived from it:
    /// the settings and the parameter values, with no shader values and no bake.
    ///
    /// What a history snapshot is made of — the bake and the shader values are
    /// re-obtained by baking and dispatching, so a copy that carried them would cost
    /// the size of the document per edit.
    pub fn authored(&self) -> Self {
        let mut shader = self.shader.clone();
        shader.clear();
        Self {
            id: self.id.clone(),
            role: self.role,
            shift: self.shift,
            range: self.range,
            export: self.export,
            hillshade: self.hillshade,
            light_azimuth: self.light_azimuth,
            contours: self.contours,
            contour_interval: self.contour_interval,
            shader,
            baked: Raster::default(),
        }
    }

    /// [`Layer::range`] as `(low, high)`, swapped if it was stored backwards. A
    /// backwards range is not an error anywhere; this is the only correct way to
    /// read it.
    pub fn bounds(&self) -> (f32, f32) {
        if self.range.0 <= self.range.1 {
            self.range
        } else {
            (self.range.1, self.range.0)
        }
    }

    /// Texel dimensions this layer bakes to in a `size`-cell document. Never zero
    /// on either axis.
    pub fn resolution(&self, size: UVec2) -> UVec2 {
        resolution(size, self.shift)
    }

    /// The values from the last bake. Empty for a layer that has never been baked,
    /// was released, or came from a loaded document — read it through
    /// [`Layer::sample`] unless the texel grid itself is what you want.
    pub fn baked(&self) -> &Raster<f32> {
        &self.baked
    }

    /// The baked raster, writable in place. Its dimensions are the bake's
    /// invariant, so a caller replacing texels must not change its length.
    pub(crate) fn baked_mut(&mut self) -> &mut Raster<f32> {
        &mut self.baked
    }

    /// Moves the baked raster out, leaving the layer unbaked and sampling as
    /// `0.0`. Pair with [`Layer::put_baked`] to borrow a raster across a bake that
    /// needs the rest of the document mutably.
    pub(crate) fn take_baked(&mut self) -> Raster<f32> {
        std::mem::take(&mut self.baked)
    }

    /// Installs `raster` as the bake result, dropping whatever was there. Nothing
    /// checks it against [`Layer::resolution`].
    pub(crate) fn put_baked(&mut self, raster: Raster<f32>) {
        self.baked = raster;
    }

    /// The baked value at a position in document cells, where a cell centre is at
    /// `x + 0.5`. Reads `0.0` for an unbaked layer, and clamps rather than failing
    /// outside the document.
    ///
    /// A coarse layer is interpolated between its texels, so it reads as a smooth
    /// surface and not as blocks.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let u = raster_coord(x, self.shift);
        let v = raster_coord(y, self.shift);
        self.baked.sample_bilinear(u, v)
    }

    /// The layers this one's shader file reads by name, in declaration order, with
    /// duplicates kept.
    ///
    /// This is what bake ordering and cycle detection run on. It is what the file said
    /// when it was last read, so it is empty until the shader directory has been read.
    pub fn dependencies(&self) -> impl Iterator<Item = &LayerId> {
        self.shader.layers.iter()
    }

    /// This layer holding a 1×1 raster of `value` as its shader's values, and `value`
    /// as its `value` parameter.
    #[cfg(test)]
    pub fn held(mut self, value: f32) -> Self {
        self.shader.put_values(Raster::new(UVec2::ONE, value));
        self.shader.params.insert("value".to_owned(), vec![value]);
        self
    }

    /// This layer holding `raster` as its shader's values.
    #[cfg(test)]
    pub fn holding(mut self, raster: Raster<f32>) -> Self {
        self.shader.put_values(raster);
        self
    }

    /// This layer's shader reading the layers `names`, in that order.
    #[cfg(test)]
    pub fn reading(mut self, names: &[&str]) -> Self {
        self.shader.layers = names.iter().map(|name| LayerId::from(*name)).collect();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shift 0 is what the `Height` layer is pinned to, so a document's cell grid and
    // its height raster have to be the same grid.
    #[test]
    fn a_layer_at_shift_zero_holds_one_texel_per_cell() {
        let layer = Layer::new("height");
        assert_eq!(layer.resolution(UVec2::new(64, 32)), UVec2::new(64, 32));
    }

    // The point of a shift is the allocation it saves; this pins that the saving is
    // quadratic in the shift and not linear.
    #[test]
    fn a_coarse_layer_holds_one_texel_per_block() {
        let layer = Layer::new("moisture").with_shift(4);
        assert_eq!(
            layer.resolution(UVec2::new(4096, 4096)),
            UVec2::new(256, 256)
        );
    }

    // Nothing rejects a backwards range, so `bounds` is the only thing standing
    // between one and a clamp that empties the layer.
    #[test]
    fn a_backwards_range_is_read_in_the_order_a_clamp_needs() {
        let layer = Layer::new("height").with_range((1.0, -1.0));
        assert_eq!(layer.bounds(), (-1.0, 1.0));
    }

    // A shader's `@layer` names are the file's dependency on another layer, so bake
    // order and the editor's read checks have to see them.
    #[test]
    fn a_layers_layers_are_its_dependencies() {
        let layer = Layer::new("height").reading(&["base", "relief"]);
        let deps: Vec<_> = layer.dependencies().map(|id| id.as_str()).collect();
        assert_eq!(deps, ["base", "relief"]);
    }

    // A history snapshot is `authored()`, so a display property missing from it is a
    // property undo silently skips.
    #[test]
    fn a_new_layer_is_unlit_from_the_northwest_and_carries_both_into_a_snapshot() {
        let mut layer = Layer::new("height");
        assert!(!layer.hillshade);
        assert_eq!(layer.light_azimuth, 315.0);

        layer.hillshade = true;
        layer.light_azimuth = 90.0;
        let authored = layer.authored();
        assert!(authored.hillshade);
        assert_eq!(authored.light_azimuth, 90.0);
    }

    // Same reason as the hillshade pair above: a contour property dropped by
    // `authored()` is one undo would silently skip.
    #[test]
    fn a_new_layer_has_no_contours_at_a_tenth_and_carries_both_into_a_snapshot() {
        let mut layer = Layer::new("height");
        assert!(!layer.contours);
        assert_eq!(layer.contour_interval, 0.1);

        layer.contours = true;
        layer.contour_interval = 0.25;
        let authored = layer.authored();
        assert!(authored.contours);
        assert_eq!(authored.contour_interval, 0.25);
    }

    // Every loaded document is in this state until it is baked, so sampling one has
    // to be a defined read rather than a panic on an empty raster.
    #[test]
    fn an_unbaked_layer_samples_as_zero() {
        let layer = Layer::new("height");
        assert_eq!(layer.sample(12.5, 3.5), 0.0);
    }
}
