#import bevy_sprite::mesh2d_vertex_output::VertexOutput

// The Rust side of this is `FieldSettings` in `material.rs`. Field order is the binding
// layout: vectors before scalars, so the padding agrees on both sides.
struct FieldUniform {
    field_resolution: vec2<f32>,
    document_size: vec2<f32>,
    range: vec2<f32>,
    diverging: f32,
    water_overlay: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> settings: FieldUniform;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var field_map: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var water_map: texture_2d<f32>;

// TODO(jb-comment): why the ends are these two colours specifically, and what a ramp that
// was not monotone in lightness would make the eye invent.
const SEQUENTIAL_LIGHT: vec3<f32> = vec3<f32>(0.933, 0.949, 0.961);
const SEQUENTIAL_DARK: vec3<f32> = vec3<f32>(0.063, 0.157, 0.227);

// TODO(jb-comment): where this pair was validated and against what floors — the same
// measurement wusel's inspection overlay carries.
const DIVERGING_COOL: vec3<f32> = vec3<f32>(0.051, 0.212, 0.420);
const DIVERGING_NEUTRAL: vec3<f32> = vec3<f32>(0.949, 0.937, 0.914);
const DIVERGING_WARM: vec3<f32> = vec3<f32>(0.439, 0.075, 0.071);

const WATER_TINT: vec3<f32> = vec3<f32>(0.114, 0.353, 0.541);
const CHANNEL_TINT: vec3<f32> = vec3<f32>(0.365, 0.749, 0.867);

fn sequential(t: f32) -> vec3<f32> {
    return mix(SEQUENTIAL_LIGHT, SEQUENTIAL_DARK, clamp(t, 0.0, 1.0));
}

// `t` is on -1..1, and the arms are kept equal about the neutral so a view wholly on one
// side of the midpoint draws wholly in that side's hue.
fn diverging(t: f32) -> vec3<f32> {
    let s = clamp(t, -1.0, 1.0);
    if s < 0.0 {
        return mix(DIVERGING_NEUTRAL, DIVERGING_COOL, -s);
    }
    return mix(DIVERGING_NEUTRAL, DIVERGING_WARM, s);
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    // The quad's v runs down from the top while row zero of a raster is the bottom, so
    // the flip is here rather than in every read on the Rust side.
    let uv = vec2<f32>(mesh.uv.x, 1.0 - mesh.uv.y);

    let field_size = max(settings.field_resolution, vec2<f32>(1.0, 1.0));
    let field_texel = vec2<i32>(clamp(
        floor(uv * field_size),
        vec2<f32>(0.0, 0.0),
        field_size - vec2<f32>(1.0, 1.0),
    ));
    let value = textureLoad(field_map, field_texel, 0).r;

    let low = settings.range.x;
    let high = settings.range.y;
    let span = max(high - low, 1e-6);

    var colour: vec3<f32>;
    if settings.diverging > 0.5 {
        // The midpoint is zero and the arms are the wider of the two, so the neutral band
        // sits where the field actually changes sign.
        let reach = max(max(abs(low), abs(high)), 1e-6);
        colour = diverging(value / reach);
    } else {
        colour = sequential((value - low) / span);
    }

    if settings.water_overlay > 0.5 {
        let water_size = max(settings.document_size, vec2<f32>(1.0, 1.0));
        let water_texel = vec2<i32>(clamp(
            floor(uv * water_size),
            vec2<f32>(0.0, 0.0),
            water_size - vec2<f32>(1.0, 1.0),
        ));
        let water = textureLoad(water_map, water_texel, 0);

        // Level first, then flow: a channel drawn under a lake would vanish into it, and
        // the channel is the thing a scenario is asking about.
        colour = mix(colour, WATER_TINT, water.r * 0.78);
        colour = mix(colour, CHANNEL_TINT, water.g * 0.85);
    }

    return vec4<f32>(colour, 1.0);
}
