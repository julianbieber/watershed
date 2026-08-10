use serde::{Deserialize, Serialize};

use crate::field::FieldId;
use crate::noise::NoiseSpec;
use crate::raster::Raster;
use crate::regions::{RegionOutput, RegionSpec};

/// TODO(jb-doc): how a layer's value meets the value under it, and why `Replace` is a
/// blend mode rather than a property of being the first layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Blend {
    #[default]
    Add,
    Mul,
    Replace,
    Max,
    Min,
}

impl Blend {
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

/// TODO(jb-doc): why a field-valued mask needs a remap at all — that a field carries its
/// own range and a mask weight does not.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Remap {
    pub from: (f32, f32),
    pub to: (f32, f32),
}

impl Remap {
    pub const IDENTITY: Self = Self {
        from: (0.0, 1.0),
        to: (0.0, 1.0),
    };

    pub fn new(from: (f32, f32), to: (f32, f32)) -> Self {
        Self { from, to }
    }

    /// TODO(jb-comment): why a zero-width input band reads as the low end rather than as
    /// a division by zero.
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

/// TODO(jb-doc): the three ways a layer can be told where it applies, and why a painted
/// mask is bytes where a painted layer is floats.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Mask {
    Constant(f32),
    Painted(Raster<u8>),
    Field(FieldId, Remap),
}

impl Mask {
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
/// TODO(jb-doc): why a forward run is not just a cheaper central one — what the two do
/// differently at a crest, and why that shows up in a field thresholded near zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SlopeMode {
    /// A central difference on each axis, taken as a euclidean magnitude.
    #[default]
    Gradient,
    /// A forward difference on each axis, taken as the larger of the two.
    SteepestAxis,
}

/// TODO(jb-doc): what each op reads and what it deliberately does not — that amplitude,
/// blending and masking belong to the layer and never to the op.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LayerOp {
    Constant(f32),
    Noise(NoiseSpec),
    Paint(Raster<f32>),
    /// TODO(jb-doc): why the slope is taken from a baked raster rather than from a second
    /// analytic evaluation of the field it names.
    Slope {
        of: FieldId,
        sample_tiles: f32,
        /// TODO(jb-comment): why the default has to stay the gradient for a document
        /// written before this existed.
        #[serde(default)]
        mode: SlopeMode,
    },
    FieldRef(FieldId),
    /// TODO(jb-doc): why this op reads no field, and what that buys the halo arithmetic
    /// and the rect re-bake.
    Regions {
        spec: RegionSpec,
        output: RegionOutput,
    },
    /// TODO(jb-doc): what this is the escape hatch for.
    External(Raster<f32>),
}

impl LayerOp {
    pub fn dependency(&self) -> Option<&FieldId> {
        match self {
            LayerOp::Slope { of, .. } => Some(of),
            LayerOp::FieldRef(id) => Some(id),
            _ => None,
        }
    }
}

/// TODO(jb-doc): the unit the stack is built out of, and why `enabled` is data rather
/// than the caller removing the layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub op: LayerOp,
    pub blend: Blend,
    pub amplitude: f32,
    pub mask: Mask,
    pub enabled: bool,
}

impl Layer {
    pub fn new(op: LayerOp) -> Self {
        Self {
            op,
            blend: Blend::Add,
            amplitude: 1.0,
            mask: Mask::default(),
            enabled: true,
        }
    }

    pub fn with_blend(mut self, blend: Blend) -> Self {
        self.blend = blend;
        self
    }

    pub fn with_amplitude(mut self, amplitude: f32) -> Self {
        self.amplitude = amplitude;
        self
    }

    pub fn with_mask(mut self, mask: Mask) -> Self {
        self.mask = mask;
        self
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

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

    #[test]
    fn every_blend_mode_leaves_the_value_under_it_where_it_belongs() {
        assert_eq!(Blend::Add.apply(2.0, 3.0), 5.0);
        assert_eq!(Blend::Mul.apply(2.0, 3.0), 6.0);
        assert_eq!(Blend::Replace.apply(2.0, 3.0), 3.0);
        assert_eq!(Blend::Max.apply(2.0, 3.0), 3.0);
        assert_eq!(Blend::Min.apply(2.0, 3.0), 2.0);
    }

    #[test]
    fn an_identity_remap_returns_the_value_it_was_given() {
        for value in [0.0f32, 0.25, 0.5, 1.0] {
            assert_eq!(Remap::IDENTITY.apply(value), value);
        }
    }

    #[test]
    fn a_remap_clamps_to_its_input_band_before_it_scales() {
        let remap = Remap::new((0.4, 0.6), (0.0, 1.0));
        assert_eq!(remap.apply(0.0), 0.0);
        assert_eq!(remap.apply(0.4), 0.0);
        assert!((remap.apply(0.5) - 0.5).abs() < 1e-6);
        assert_eq!(remap.apply(0.6), 1.0);
        assert_eq!(remap.apply(9.0), 1.0);
    }

    #[test]
    fn a_remap_may_run_backwards() {
        let remap = Remap::new((0.0, 1.0), (1.0, 0.0));
        assert_eq!(remap.apply(0.0), 1.0);
        assert_eq!(remap.apply(1.0), 0.0);
    }

    #[test]
    fn a_remap_over_a_zero_width_band_is_its_low_end_rather_than_a_division_by_zero() {
        let remap = Remap::new((0.5, 0.5), (0.2, 0.9));
        assert_eq!(remap.apply(0.5), 0.2);
        assert!(remap.apply(0.9).is_finite());
    }

    #[test]
    fn a_layer_reports_both_the_field_its_op_reads_and_the_field_its_mask_reads() {
        let layer = Layer::new(LayerOp::FieldRef(FieldId::from("height")))
            .with_mask(Mask::Field(FieldId::from("moisture"), Remap::IDENTITY));
        let deps: Vec<_> = layer.dependencies().map(|id| id.as_str()).collect();
        assert_eq!(deps, vec!["height", "moisture"]);
    }

    #[test]
    fn a_layer_with_no_field_reference_reports_no_dependency() {
        let layer = Layer::new(LayerOp::Constant(1.0));
        assert_eq!(layer.dependencies().count(), 0);
    }
}
