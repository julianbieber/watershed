// TODO(jb-doc): module docs — that this is the whole of how a field becomes a picture,
// and the coupling rule it shares with the WGSL beside it.

use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{Material2d, Material2dPlugin};

/// TODO(jb-doc): why the shader is embedded rather than loaded from an asset root — what
/// a directly invoked binary does without one, and that the ctl invokes it exactly that
/// way.
const SHADER: &str = "embedded://watershed_editor/field.wgsl";

pub struct FieldMaterialPlugin;

impl Plugin for FieldMaterialPlugin {
    fn build(&self, app: &mut App) {
        bevy::asset::embedded_asset!(app, "field.wgsl");
        app.add_plugins(Material2dPlugin::<FieldMaterial>::default());
    }
}

/// TODO(jb-doc): the coupling rule — this struct and `FieldUniform` in `field.wgsl` are
/// the same thing written twice, field order *is* the binding layout, and vectors are
/// declared before scalars so the padding agrees on both sides.
#[derive(Asset, AsBindGroup, TypePath, Clone)]
pub struct FieldMaterial {
    #[uniform(0)]
    pub settings: FieldSettings,
    /// TODO(jb-doc): why this is read with `textureLoad` and therefore carries no sampler
    /// — that one texel is one cell and a filtered read would blur a cell boundary that
    /// the whole view exists to show.
    #[texture(1, sample_type = "float", filterable = false)]
    pub field: Handle<Image>,
    #[texture(2, sample_type = "float", filterable = false)]
    pub water: Handle<Image>,
}

/// TODO(jb-doc): what each number means to the fragment function, and which of them are
/// re-measured every frame rather than written once.
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct FieldSettings {
    pub field_resolution: Vec2,
    pub document_size: Vec2,
    /// The fitted low and high ends of the ramp, in the field's own units.
    pub range: Vec2,
    /// TODO(jb-doc): why this is derived from the fitted range straddling zero rather than
    /// chosen, and what that means a field with no negative values can never draw as.
    pub diverging: f32,
    pub water_overlay: f32,
}

impl Default for FieldSettings {
    fn default() -> Self {
        Self {
            field_resolution: Vec2::ONE,
            document_size: Vec2::ONE,
            range: Vec2::new(0.0, 1.0),
            diverging: 0.0,
            water_overlay: 1.0,
        }
    }
}

impl Material2d for FieldMaterial {
    fn fragment_shader() -> ShaderRef {
        SHADER.into()
    }
}

// The ramp exists twice and no more: here, and in `field.wgsl`. The legend cannot run the
// shader and the shader cannot ask the legend, so the numbers are written on both sides
// and `the_two_ramps_agree_at_their_ends` is what holds them together.
const SEQUENTIAL_LIGHT: Vec3 = Vec3::new(0.933, 0.949, 0.961);
const SEQUENTIAL_DARK: Vec3 = Vec3::new(0.063, 0.157, 0.227);
const DIVERGING_COOL: Vec3 = Vec3::new(0.051, 0.212, 0.420);
const DIVERGING_NEUTRAL: Vec3 = Vec3::new(0.949, 0.937, 0.914);
const DIVERGING_WARM: Vec3 = Vec3::new(0.439, 0.075, 0.071);

pub fn sequential(t: f32) -> Vec3 {
    SEQUENTIAL_LIGHT.lerp(SEQUENTIAL_DARK, t.clamp(0.0, 1.0))
}

/// `t` is on -1..1, with the neutral at zero.
pub fn diverging(t: f32) -> Vec3 {
    let t = t.clamp(-1.0, 1.0);
    if t < 0.0 {
        DIVERGING_NEUTRAL.lerp(DIVERGING_COOL, -t)
    } else {
        DIVERGING_NEUTRAL.lerp(DIVERGING_WARM, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TODO(jb-comment): why only the ends and the midpoint are pinned rather than the
    /// whole curve — that the interpolation is the same `mix` on both sides, so the ends
    /// are the only place a transcription can drift.
    #[test]
    fn the_two_ramps_agree_at_their_ends() {
        assert_eq!(sequential(0.0), SEQUENTIAL_LIGHT);
        assert_eq!(sequential(1.0), SEQUENTIAL_DARK);
        assert_eq!(diverging(-1.0), DIVERGING_COOL);
        assert_eq!(diverging(0.0), DIVERGING_NEUTRAL);
        assert_eq!(diverging(1.0), DIVERGING_WARM);
    }

    /// TODO(jb-comment): what this is really asserting — that "more is darker" holds over
    /// the whole sequential ramp, which is the rule a rainbow breaks.
    #[test]
    fn the_sequential_ramp_only_ever_darkens() {
        let luma = |c: Vec3| 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
        let mut previous = f32::MAX;
        for step in 0..=32 {
            let value = luma(sequential(step as f32 / 32.0));
            assert!(value <= previous, "step {step} brightened");
            previous = value;
        }
    }

    /// TODO(jb-comment): why the arms are checked for equal reach rather than equal colour
    /// — that a view wholly on one side of the midpoint has to draw wholly in that hue.
    #[test]
    fn the_diverging_arms_are_equal_about_the_neutral() {
        for step in 1..=16 {
            let t = step as f32 / 16.0;
            let cool = diverging(-t).distance(DIVERGING_NEUTRAL);
            let warm = diverging(t).distance(DIVERGING_NEUTRAL);
            let reach = cool.max(warm);
            assert!(
                (cool - warm).abs() < reach * 0.5,
                "at {t} the arms reach {cool} and {warm}"
            );
        }
    }
}
