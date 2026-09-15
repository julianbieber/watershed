// Colouring one layer of a watershed document, with the solved water over it.
//
// The Rust half of this file is `material.rs`: the uniform below is declared there
// as `LayerSettings` and the two ramps are written out there as well, because the
// legend has to draw the same colours and cannot run a shader. A change here is a
// change there.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput

// Layer order is the binding layout: vectors before scalars, so the padding agrees
// with `LayerSettings`.
struct LayerUniform {
    layer_resolution: vec2<f32>,
    document_size: vec2<f32>,
    range: vec2<f32>,
    diverging: f32,
    water_overlay: f32,
    hillshade: f32,
    light_azimuth: f32,
    contours: f32,
    contour_interval: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> settings: LayerUniform;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var layer_map: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var water_map: texture_2d<f32>;

const SEQUENTIAL_LIGHT: vec3<f32> = vec3<f32>(0.933, 0.949, 0.961);
const SEQUENTIAL_DARK: vec3<f32> = vec3<f32>(0.063, 0.157, 0.227);

const DIVERGING_COOL: vec3<f32> = vec3<f32>(0.051, 0.212, 0.420);
const DIVERGING_NEUTRAL: vec3<f32> = vec3<f32>(0.949, 0.937, 0.914);
const DIVERGING_WARM: vec3<f32> = vec3<f32>(0.439, 0.075, 0.071);

const WATER_TINT: vec3<f32> = vec3<f32>(0.114, 0.353, 0.541);
const CHANNEL_TINT: vec3<f32> = vec3<f32>(0.365, 0.749, 0.867);

const LIGHT_ALTITUDE: f32 = 45.0;
const HILLSHADE_GAIN: f32 = 8.0;
const HILLSHADE_DEPTH: f32 = 0.7;

const CONTOUR_HALF_WIDTH: f32 = 0.6;
const CONTOUR_STRENGTH: f32 = 0.85;

// `t` is on 0..1, clamped. Monotone in lightness, so a larger value always reads as
// darker.
fn sequential(t: f32) -> vec3<f32> {
    return mix(SEQUENTIAL_LIGHT, SEQUENTIAL_DARK, clamp(t, 0.0, 1.0));
}

// `t` is on -1..1 with the neutral at zero, clamped. The arms reach equally far from
// the neutral, so a view wholly on one side of the midpoint draws wholly in that
// side's hue.
fn diverging(t: f32) -> vec3<f32> {
    let s = clamp(t, -1.0, 1.0);
    if s < 0.0 {
        return mix(DIVERGING_NEUTRAL, DIVERGING_COOL, -s);
    }
    return mix(DIVERGING_NEUTRAL, DIVERGING_WARM, s);
}

// `texel` is clamped into the raster rather than wrapped, so the outermost row shades
// against itself instead of against the far edge.
fn layer_at(texel: vec2<f32>, size: vec2<f32>) -> f32 {
    let clamped = clamp(texel, vec2<f32>(0.0, 0.0), size - vec2<f32>(1.0, 1.0));
    return textureLoad(layer_map, vec2<i32>(clamped), 0).r;
}

// The multiplier the ramp is lit by: `1.0` on ground that is flat, more where the
// surface turns towards the light and less where it turns away.
//
// The slope is measured in fitted `span`s per texel, so the relief reads the same
// whatever units the layer is in, and flat ground lands on exactly the unlit ramp
// colour — which is what the legend beside it still stands for. `azimuth` is a compass
// bearing in degrees.
fn relief(texel: vec2<f32>, size: vec2<f32>, span: f32, azimuth: f32) -> f32 {
    let east = layer_at(texel + vec2<f32>(1.0, 0.0), size);
    let west = layer_at(texel + vec2<f32>(-1.0, 0.0), size);
    let north = layer_at(texel + vec2<f32>(0.0, 1.0), size);
    let south = layer_at(texel + vec2<f32>(0.0, -1.0), size);

    let gradient = vec2<f32>(east - west, north - south) * 0.5 * HILLSHADE_GAIN / span;
    let normal = normalize(vec3<f32>(-gradient.x, -gradient.y, 1.0));

    let bearing = radians(azimuth);
    let altitude = radians(LIGHT_ALTITUDE);
    let light = vec3<f32>(
        sin(bearing) * cos(altitude),
        cos(bearing) * cos(altitude),
        sin(altitude),
    );
    let flat = max(sin(altitude), 1e-6);
    let lit = max(dot(normal, light), 0.0) / flat;
    return mix(1.0, lit, HILLSHADE_DEPTH);
}

// The layer read as a smooth surface rather than as texels: bilinear between the four
// texels around `uv`, clamped at the border the same way `layer_at` clamps. The colour
// ramp does not go through here and stays an exact per-texel fetch.
fn layer_smooth(uv: vec2<f32>, size: vec2<f32>) -> f32 {
    let p = uv * size - vec2<f32>(0.5, 0.5);
    let base = floor(p);
    let t = p - base;
    let v00 = layer_at(base, size);
    let v10 = layer_at(base + vec2<f32>(1.0, 0.0), size);
    let v01 = layer_at(base + vec2<f32>(0.0, 1.0), size);
    let v11 = layer_at(base + vec2<f32>(1.0, 1.0), size);
    return mix(mix(v00, v10, t.x), mix(v01, v11, t.x), t.y);
}

// How much of this pixel an iso-line covers, on 0..1: `1.0` on a pixel sitting across a
// multiple of `interval`, falling to zero a pixel away from one.
//
// `interval` is in the layer's own units, so the levels are absolute multiples and do
// not move with the fitted range. Flat ground yields zero, having no level crossing
// within a pixel of it, and so does ground steep enough that the lines come within
// about a pixel of each other — which fades a too-small interval out rather than
// drawing it as moire. Must be called from uniform control flow: it takes a screen-space
// derivative.
fn contour_coverage(uv: vec2<f32>, size: vec2<f32>, interval: f32) -> f32 {
    if interval <= 0.0 {
        return 0.0;
    }
    let level = layer_smooth(uv, size) / interval;
    let per_pixel = fwidth(level);
    if per_pixel <= 0.0 {
        return 0.0;
    }
    let to_line = abs(fract(level + 0.5) - 0.5) / per_pixel;
    let line = 1.0 - smoothstep(CONTOUR_HALF_WIDTH, CONTOUR_HALF_WIDTH + 1.0, to_line);
    return line * (1.0 - smoothstep(0.25, 1.0, per_pixel));
}

// The quad's v runs down from the top while row zero of a raster is the bottom, so
// both textures are read with v flipped and the Rust side uploads them unaltered.
//
// The overlays run after the ramp, each taking the colour the one before it left.
@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(mesh.uv.x, 1.0 - mesh.uv.y);

    let layer_size = max(settings.layer_resolution, vec2<f32>(1.0, 1.0));
    let texel = clamp(
        floor(uv * layer_size),
        vec2<f32>(0.0, 0.0),
        layer_size - vec2<f32>(1.0, 1.0),
    );
    let layer_texel = vec2<i32>(texel);
    let value = textureLoad(layer_map, layer_texel, 0).r;

    let low = settings.range.x;
    let high = settings.range.y;
    let span = max(high - low, 1e-6);

    var colour: vec3<f32>;
    if settings.diverging > 0.5 {
        let reach = max(max(abs(low), abs(high)), 1e-6);
        colour = diverging(value / reach);
    } else {
        colour = sequential((value - low) / span);
    }

    if settings.hillshade > 0.5 {
        let lit = relief(texel, layer_size, span, settings.light_azimuth);
        colour = clamp(colour * lit, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    if settings.contours > 0.5 {
        let luma = dot(colour, vec3<f32>(0.2126, 0.7152, 0.0722));
        let ink = select(vec3<f32>(1.0), vec3<f32>(0.0), luma > 0.5);
        let coverage = contour_coverage(uv, layer_size, settings.contour_interval);
        colour = mix(colour, ink, coverage * CONTOUR_STRENGTH);
    }

    if settings.water_overlay > 0.5 {
        let water_size = max(settings.document_size, vec2<f32>(1.0, 1.0));
        let water_texel = vec2<i32>(clamp(
            floor(uv * water_size),
            vec2<f32>(0.0, 0.0),
            water_size - vec2<f32>(1.0, 1.0),
        ));
        let water = textureLoad(water_map, water_texel, 0);

        colour = mix(colour, WATER_TINT, water.r * 0.78);
        colour = mix(colour, CHANNEL_TINT, water.g * 0.85);
    }

    return vec4<f32>(colour, 1.0);
}
