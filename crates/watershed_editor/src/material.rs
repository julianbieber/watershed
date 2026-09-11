//! Turning a field's texels into a picture: the material the viewport quad is drawn
//! with, and the colour ramps that decide what a value looks like.
//!
//! This module and `field.wgsl` beside it are two halves of one thing. The uniform
//! is declared in both and the ramps are written out in both, because the shader
//! cannot call into Rust and the legend cannot run the shader. A change on either
//! side is a change on both.

use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{Material2d, Material2dPlugin};

const SHADER: &str = "embedded://watershed_editor/field.wgsl";

/// Registers the field material and embeds its shader into the binary.
///
/// The shader is compiled in rather than loaded from an asset root, so the editor
/// runs from wherever the binary is — which is how the control client starts it, with
/// no assets directory anywhere near.
pub struct FieldMaterialPlugin;

impl Plugin for FieldMaterialPlugin {
    fn build(&self, app: &mut App) {
        bevy::asset::embedded_asset!(app, "field.wgsl");
        app.add_plugins(Material2dPlugin::<FieldMaterial>::default());
    }
}

/// What the viewport quad is drawn with: the numbers the fragment function needs and
/// the two textures it reads.
#[derive(Asset, AsBindGroup, TypePath, Clone)]
pub struct FieldMaterial {
    /// Bound at slot 0 as `FieldUniform`.
    #[uniform(0)]
    pub settings: FieldSettings,
    /// The active field's texels, one per texel of its baked raster.
    ///
    /// Unfilterable, and fetched exactly rather than sampled: one texel is one cell,
    /// and a filtered read would smear the cell boundaries the view exists to show.
    #[texture(1, sample_type = "float", filterable = false)]
    pub field: Handle<Image>,
    /// The solved water at one texel per document cell, read the same way. Empty
    /// while the document has no water.
    #[texture(2, sample_type = "float", filterable = false)]
    pub water: Handle<Image>,
}

/// The uniform the fragment function reads.
///
/// The same thing as `FieldUniform` in `field.wgsl`, written twice: the field order
/// here *is* the binding layout, and the vectors are declared before the scalars so
/// the padding agrees on both sides. Adding, removing or reordering a field means
/// doing the same in the shader.
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct FieldSettings {
    /// Texel dimensions of the field texture, so the shader can address it. Changes
    /// when the active field or its shift changes.
    pub field_resolution: Vec2,
    /// The document's extent in cells, which the water texture is addressed in.
    pub document_size: Vec2,
    /// The fitted low and high ends of the ramp, in the field's own units. Refitted
    /// to what is on screen as the camera moves, so it changes most frames.
    pub range: Vec2,
    /// Non-zero to draw the diverging ramp instead of the sequential one.
    ///
    /// Follows from `range` straddling zero rather than being chosen: the diverging
    /// ramp's neutral band means "zero", and it means nothing at all on a field whose
    /// values are all one sign — such a field is never drawn diverging.
    pub diverging: f32,
    /// Non-zero to tint standing water and channels over the field.
    pub water_overlay: f32,
    /// Non-zero to light the ramp by the field's own normal.
    pub hillshade: f32,
    /// The compass bearing the hillshade light comes from, in degrees: 0 is north and
    /// 90 is east. Read only while `hillshade` is non-zero.
    pub light_azimuth: f32,
}

impl Default for FieldSettings {
    fn default() -> Self {
        Self {
            field_resolution: Vec2::ONE,
            document_size: Vec2::ONE,
            range: Vec2::new(0.0, 1.0),
            diverging: 0.0,
            water_overlay: 1.0,
            hillshade: 0.0,
            light_azimuth: 315.0,
        }
    }
}

impl Material2d for FieldMaterial {
    fn fragment_shader() -> ShaderRef {
        SHADER.into()
    }
}

const SEQUENTIAL_LIGHT: Vec3 = Vec3::new(0.933, 0.949, 0.961);
const SEQUENTIAL_DARK: Vec3 = Vec3::new(0.063, 0.157, 0.227);
const DIVERGING_COOL: Vec3 = Vec3::new(0.051, 0.212, 0.420);
const DIVERGING_NEUTRAL: Vec3 = Vec3::new(0.949, 0.937, 0.914);
const DIVERGING_WARM: Vec3 = Vec3::new(0.439, 0.075, 0.071);

/// The sequential ramp, `t` on 0..1, clamped. Monotone in lightness from light to
/// dark, so a larger value always reads as darker.
///
/// The same ramp `field.wgsl` draws with. Used by the legend, which cannot run the
/// shader.
pub fn sequential(t: f32) -> Vec3 {
    SEQUENTIAL_LIGHT.lerp(SEQUENTIAL_DARK, t.clamp(0.0, 1.0))
}

/// The diverging ramp, `t` on -1..1 with the neutral at zero, clamped.
///
/// The two arms reach equally far from the neutral, so a view lying wholly on one
/// side of zero draws wholly in that side's hue. The same ramp `field.wgsl` draws
/// with.
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

    // The ramps are transcribed into `field.wgsl` by hand, and this is what holds the
    // two copies together. Only the ends and the midpoint are pinned: both sides
    // interpolate linearly between exactly those, so a transcription can only drift at
    // an endpoint.
    #[test]
    fn the_two_ramps_agree_at_their_ends() {
        assert_eq!(sequential(0.0), SEQUENTIAL_LIGHT);
        assert_eq!(sequential(1.0), SEQUENTIAL_DARK);
        assert_eq!(diverging(-1.0), DIVERGING_COOL);
        assert_eq!(diverging(0.0), DIVERGING_NEUTRAL);
        assert_eq!(diverging(1.0), DIVERGING_WARM);
    }

    // "More is darker" has to hold over the whole ramp, not just at its ends — that is
    // the rule a rainbow ramp breaks, and breaking it makes a reader see boundaries in
    // the data that are not there.
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

    // Equal *reach* rather than equal colour: the two arms are deliberately different
    // hues, and what has to match is how far each gets from the neutral, so that a view
    // lying on one side of zero is not drawn paler than the same view on the other.
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
