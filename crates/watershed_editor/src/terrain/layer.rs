//! The unit a field's value is built out of: what a layer contributes, where it
//! applies, and how it meets what is already under it.

use serde::{Deserialize, Serialize};

use watershed::field::FieldId;

use crate::terrain::noise::NoiseSpec;
use crate::terrain::regions::{RegionOutput, RegionSpec};
use crate::terrain::shader::ShaderLayer;
use watershed::raster::Raster;

/// How a layer's value meets the accumulated value under it.
///
/// A stack starts at `0.0`, so the bottom layer blends against zero rather than
/// against nothing: `Add` and `Max` pass it through, `Mul` and `Min` erase it, and
/// `Replace` is how a layer states its value outright at any position in the stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Blend {
    /// Sum. The default, so a stack accumulates unless told otherwise.
    #[default]
    Add,
    /// Product.
    Mul,
    /// Discards what is under it.
    Replace,
    /// The larger of the two.
    Max,
    /// The smaller of the two.
    Min,
}

impl Blend {
    /// The blended value, before the layer's mask weight is applied. `under` is the
    /// stack so far and `value` the layer's own contribution, already scaled by its
    /// amplitude. Not clamped — the field's range is applied once, at the end.
    pub fn apply(self, under: f32, value: f32) -> f32 {
        match self {
            Blend::Add => under + value,
            Blend::Mul => under * value,
            Blend::Replace => value,
            Blend::Max => under.max(value),
            Blend::Min => under.min(value),
        }
    }
}

/// A linear rescale from one interval onto another, clamped at both ends of the
/// input.
///
/// This is what makes a field usable as a mask: a field's values span whatever its
/// own range is, and a mask weight has to be in `0.0..=1.0`, so the reader states
/// which band of the field it cares about rather than the field being changed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Remap {
    /// The input band. Values outside it clamp to the nearest end.
    pub from: (f32, f32),
    /// The output band. May run backwards, which inverts the mapping.
    pub to: (f32, f32),
}

impl Remap {
    /// `0.0..=1.0` onto itself — still clamps, so it is not a no-op for a value
    /// outside the unit interval.
    pub const IDENTITY: Self = Self {
        from: (0.0, 1.0),
        to: (0.0, 1.0),
    };

    /// Takes both bands as given. Neither is validated or sorted; `from` running
    /// backwards inverts the mapping just as `to` does.
    pub fn new(from: (f32, f32), to: (f32, f32)) -> Self {
        Self { from, to }
    }

    /// `value` rescaled from `from` onto `to`, clamped to `to`'s ends.
    ///
    /// A `from` band of zero width is not an error and never divides by zero: every
    /// input maps to `to.0`.
    pub fn apply(&self, value: f32) -> f32 {
        let span = self.from.1 - self.from.0;
        let t = if span == 0.0 {
            0.0
        } else {
            ((value - self.from.0) / span).clamp(0.0, 1.0)
        };
        let (lo, hi) = (self.to.0, self.to.1);
        lo + (hi - lo) * t
    }
}

impl Default for Remap {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Where a layer applies, as a weight per position.
///
/// However the weight is obtained it is clamped to `0.0..=1.0` before use: at `0.0`
/// the layer is skipped entirely, at `1.0` its blend lands whole, and between the
/// two the result is interpolated back towards what was under the layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Mask {
    /// The same weight everywhere. The default, at `1.0`.
    Constant(f32),
    /// A painted weight raster, stretched over the whole document whatever its own
    /// resolution. Bytes, not floats: a weight is confined to `0.0..=1.0` anyway,
    /// so 256 steps of it cost a quarter of what a layer's own values do.
    Painted(Raster<u8>),
    /// Another field's value, put through the [`Remap`] that says which band of it
    /// means "applies". Contributes a bake-order dependency.
    Field(FieldId, Remap),
}

impl Mask {
    /// The field this mask reads, if any. Feeds bake ordering and cycle detection.
    pub fn dependency(&self) -> Option<&FieldId> {
        match self {
            Mask::Field(id, _) => Some(id),
            _ => None,
        }
    }
}

impl Default for Mask {
    fn default() -> Self {
        Mask::Constant(1.0)
    }
}

/// How a slope reads the field under it.
///
/// The two answer different questions about the same ground, and which one a caller
/// wants depends on what the slope is *for* rather than on accuracy. A gradient is the
/// better estimate of the surface's true steepness; the steepest axis is what a thing
/// travelling on the lattice — water, a walker, a wagon — actually has to climb, and it
/// reads one sample further ahead rather than one either side, so it is a forward
/// question rather than a symmetric one.
///
/// The difference that matters in practice is at a crest. A central difference
/// takes samples either side of the position, and on a ridge line those two are
/// close to equal, so `Gradient` reads near zero along the ridge itself; a field
/// thresholded just above zero therefore shows a hairline gap running along every
/// crest. A forward difference never cancels that way, so `SteepestAxis` has no
/// such gap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SlopeMode {
    /// A central difference on each axis, taken as a euclidean magnitude.
    #[default]
    Gradient,
    /// A forward difference on each axis, taken as the larger of the two.
    SteepestAxis,
}

/// What a layer produces at a position, before anything is done with it.
///
/// An op answers only "what value is here". Scaling it, deciding where it applies
/// and combining it with the stack belong to [`Layer`], so every op composes with
/// every amplitude, mask and blend.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LayerOp {
    /// The same value everywhere.
    Constant(f32),
    /// Procedural noise evaluated at the position; see [`NoiseSpec`].
    Noise(NoiseSpec),
    /// A raster stretched over the whole document, and the layer an editor brush
    /// writes into. A stroke targets the topmost enabled `Paint` layer of its field.
    Paint(Raster<f32>),
    /// The steepness of another field.
    ///
    /// Read out of that field's *baked* raster, so the answer depends on the shift
    /// the field was baked at, and the field must be baked first — which is what
    /// the dependency this op reports arranges.
    Slope {
        /// The field to measure. Must exist in the document; checked at plan time.
        of: FieldId,
        /// How far apart, in document cells, the samples are taken. The magnitude
        /// is used, and a zero is raised to `f32::EPSILON` rather than dividing by
        /// zero.
        sample_tiles: f32,
        /// Defaults to [`SlopeMode::Gradient`], which is also what a document
        /// written before this field existed deserializes to.
        #[serde(default)]
        mode: SlopeMode,
    },
    /// Another field's value at the position, through that field's own sampling —
    /// so a categorical field is read to the nearest texel, not interpolated.
    FieldRef(FieldId),
    /// A value derived from the region tiling at the position.
    ///
    /// Reads no field and holds its whole input in the [`RegionSpec`], so it can be
    /// evaluated at any position without anything else in the document having been
    /// baked, and re-baking a rectangle needs no halo from it.
    Regions {
        /// The tiling and its per-region table.
        spec: RegionSpec,
        /// Which column, or which identifier, of the tiling to emit. A
        /// `Blended` column that the spec's table does not carry fails at plan
        /// time.
        output: RegionOutput,
    },
    /// A raster produced outside the editor — imported, generated, or handed in by
    /// the host application. Evaluated exactly like [`LayerOp::Paint`]; the
    /// distinction is that a brush stroke will not write into it.
    External(Raster<f32>),
    /// A raster a WGSL shader produced, at the field's own resolution.
    ///
    /// The shader is a function of the position, its own parameters and the extent
    /// it is dispatched over — it reads no field — so this op contributes no
    /// dependency and widens no re-bake. What the shader wrote is read here exactly
    /// as a field reads its own bake, and the scaling, the mask and the blend are
    /// the layer's as they are for every other op.
    ///
    /// The values are not serialized. A loaded document reads the layer as `0.0`
    /// until it has been dispatched again, in the way a loaded field reads as `0.0`
    /// until it has been baked.
    Shader(ShaderLayer),
}

impl LayerOp {
    /// The field this op reads, if any. Feeds bake ordering and cycle detection.
    pub fn dependency(&self) -> Option<&FieldId> {
        match self {
            LayerOp::Slope { of, .. } => Some(of),
            LayerOp::FieldRef(id) => Some(id),
            _ => None,
        }
    }
}

/// One entry of a field's stack: a value, scaled, confined to where it applies, and
/// blended onto what is under it.
///
/// Evaluated in that order — `op` at the position, times `amplitude`, blended by
/// `blend` onto the accumulated value, then interpolated back towards that value by
/// `1 - mask`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    /// What this layer contributes.
    pub op: LayerOp,
    /// How the contribution meets the stack under it.
    pub blend: Blend,
    /// Scales the op's value before blending. Not clamped, and applied to the value
    /// rather than to the weight.
    pub amplitude: f32,
    /// Where the layer applies.
    pub mask: Mask,
    /// A disabled layer is skipped by the bake *and* contributes no dependency, so
    /// switching one off can break a cycle or make a field non-categorical. It is
    /// data rather than a deletion so the stack keeps its shape across the toggle.
    pub enabled: bool,
}

impl Layer {
    /// `op` at full amplitude, added, unmasked and enabled.
    pub fn new(op: LayerOp) -> Self {
        Self {
            op,
            blend: Blend::Add,
            amplitude: 1.0,
            mask: Mask::default(),
            enabled: true,
        }
    }

    /// Sets how the layer meets the stack under it.
    pub fn with_blend(mut self, blend: Blend) -> Self {
        self.blend = blend;
        self
    }

    /// Sets the scale applied to the op's value.
    pub fn with_amplitude(mut self, amplitude: f32) -> Self {
        self.amplitude = amplitude;
        self
    }

    /// Sets where the layer applies, replacing any mask already set.
    pub fn with_mask(mut self, mask: Mask) -> Self {
        self.mask = mask;
        self
    }

    /// Clears [`Layer::enabled`]. There is no matching `enabled()` — a new layer is
    /// enabled already.
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// The fields this layer reads, op before mask. Reports them whether or not the
    /// layer is enabled — [`Field::dependencies`](crate::terrain::field::Field::dependencies)
    /// is what filters on that.
    pub fn dependencies(&self) -> impl Iterator<Item = &FieldId> {
        self.op
            .dependency()
            .into_iter()
            .chain(self.mask.dependency())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Each mode is one line, so the risk is a transposed pair rather than a wrong
    // formula; the asymmetric operands catch `Max`/`Min` and `under`/`value` being
    // swapped, which no symmetric case would.
    #[test]
    fn every_blend_mode_leaves_the_value_under_it_where_it_belongs() {
        assert_eq!(Blend::Add.apply(2.0, 3.0), 5.0);
        assert_eq!(Blend::Mul.apply(2.0, 3.0), 6.0);
        assert_eq!(Blend::Replace.apply(2.0, 3.0), 3.0);
        assert_eq!(Blend::Max.apply(2.0, 3.0), 3.0);
        assert_eq!(Blend::Min.apply(2.0, 3.0), 2.0);
    }

    // `Remap::IDENTITY` is the default every unremapped field mask carries, so a
    // rounding or offset error in `apply` would perturb masks nobody configured.
    #[test]
    fn an_identity_remap_returns_the_value_it_was_given() {
        for value in [0.0f32, 0.25, 0.5, 1.0] {
            assert_eq!(Remap::IDENTITY.apply(value), value);
        }
    }

    // Order matters: scaling first and clamping after would let a value outside the
    // band leave the output band. Pins the clamp on the input side.
    #[test]
    fn a_remap_clamps_to_its_input_band_before_it_scales() {
        let remap = Remap::new((0.4, 0.6), (0.0, 1.0));
        assert_eq!(remap.apply(0.0), 0.0);
        assert_eq!(remap.apply(0.4), 0.0);
        assert!((remap.apply(0.5) - 0.5).abs() < 1e-6);
        assert_eq!(remap.apply(0.6), 1.0);
        assert_eq!(remap.apply(9.0), 1.0);
    }

    // Inverting a mask is done by writing `to` backwards rather than by a flag, so
    // nothing may sort or normalise the output band.
    #[test]
    fn a_remap_may_run_backwards() {
        let remap = Remap::new((0.0, 1.0), (1.0, 0.0));
        assert_eq!(remap.apply(0.0), 1.0);
        assert_eq!(remap.apply(1.0), 0.0);
    }

    // A zero-width band is what dragging both ends of a range widget together
    // produces, so it reaches `apply` from the editor; the guard keeps a NaN out of
    // every downstream texel.
    #[test]
    fn a_remap_over_a_zero_width_band_is_its_low_end_rather_than_a_division_by_zero() {
        let remap = Remap::new((0.5, 0.5), (0.2, 0.9));
        assert_eq!(remap.apply(0.5), 0.2);
        assert!(remap.apply(0.9).is_finite());
    }

    // Bake ordering is computed from this, so a dependency missed on either side
    // schedules a field before the one it reads and bakes it against a stale raster.
    #[test]
    fn a_layer_reports_both_the_field_its_op_reads_and_the_field_its_mask_reads() {
        let layer = Layer::new(LayerOp::FieldRef(FieldId::from("height")))
            .with_mask(Mask::Field(FieldId::from("moisture"), Remap::IDENTITY));
        let deps: Vec<_> = layer.dependencies().map(|id| id.as_str()).collect();
        assert_eq!(deps, vec!["height", "moisture"]);
    }

    // A spurious dependency would be as bad as a missing one: it can invent a cycle
    // in a document that has none.
    #[test]
    fn a_layer_with_no_field_reference_reports_no_dependency() {
        let layer = Layer::new(LayerOp::Constant(1.0));
        assert_eq!(layer.dependencies().count(), 0);
    }
}
