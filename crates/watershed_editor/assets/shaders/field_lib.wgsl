// What every shader layer is compiled against: the bindings a dispatch supplies, and
// the noise primitives a shader is expected to reach for.
//
// The Rust half of this file is `gpu.rs`, which writes `Globals` out as
// `DispatchGlobals` and prepends this source to every shader; and `terrain/noise.rs`,
// which is where the noise below has to keep agreeing with the CPU layers a stack
// mixes it with. A change here is a change there.

// Everything a shader is told about where it is being evaluated. Positions are in
// document cells, not in a normalised coordinate, so a scale means the same thing at
// every shift and in every rectangle a re-bake covers.
struct Globals {
    // The document's extent, in cells.
    document: vec2<u32>,
    // The extent of this dispatch, in texels of the field's own raster.
    texels: vec2<u32>,
    // Where this dispatch starts, in texels of the field's own raster.
    origin: vec2<u32>,
    // The field's raster shift: one texel per cell at 0, per 2^shift cells above.
    shift: u32,
    // The document's seed.
    seed: u32,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read_write> field_out: array<f32>;

const TAU: f32 = 6.2831853;
const RIDGE_GAIN: f32 = 2.0;
const NOISE_GAIN: f32 = 2.6;

// The one hash, and the same one the CPU layers use. Its exact output is part of what
// a saved document means.
fn hash2(x: i32, y: i32) -> u32 {
    var h: u32 = u32(x) * 0x27d4eb2du;
    h = h ^ (u32(y) * 0x165667b1u);
    h = h ^ (h >> 15u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h;
}

fn gradient(ix: i32, iy: i32) -> vec2<f32> {
    let h = hash2(ix, iy);
    let angle = (f32(h) / 4294967295.0) * TAU;
    return vec2<f32>(cos(angle), sin(angle));
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// One octave of gradient noise at a position in lattice units, roughly in [-1, 1] and
// exactly zero at every integer lattice point.
fn gradient_noise(p: vec2<f32>) -> f32 {
    let i0 = vec2<i32>(floor(p));
    let i1 = i0 + vec2<i32>(1, 1);
    let s = p - floor(p);

    let n00 = dot(gradient(i0.x, i0.y), s);
    let n10 = dot(gradient(i1.x, i0.y), s - vec2<f32>(1.0, 0.0));
    let n01 = dot(gradient(i0.x, i1.y), s - vec2<f32>(0.0, 1.0));
    let n11 = dot(gradient(i1.x, i1.y), s - vec2<f32>(1.0, 1.0));

    let u = fade(s.x);
    let v = fade(s.y);
    return mix(mix(n00, n10, u), mix(n01, n11, u), v);
}

// Octaves of `gradient_noise` summed, each at `lacunarity` times the frequency and
// `persistence` times the amplitude of the one before, divided by the total
// amplitude — so changing the octave count refines the field rather than rescaling
// it. Zero octaves gives a NaN, exactly as the CPU reading does.
fn fbm(p: vec2<f32>, octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
    var total = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_amplitude = 0.0;
    for (var i = 0u; i < octaves; i = i + 1u) {
        total = total + gradient_noise(p * frequency) * amplitude;
        max_amplitude = max_amplitude + amplitude;
        amplitude = amplitude * persistence;
        frequency = frequency * lacunarity;
    }
    return total / max_amplitude;
}

// `fbm` read for its creases: each octave contributes (1 - |n|)^2 rather than n, so
// the zero crossings come out as the high ground. In [0, 1] with no midpoint, and
// never negative.
fn ridged_fbm(p: vec2<f32>, octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
    var total = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_amplitude = 0.0;
    for (var i = 0u; i < octaves; i = i + 1u) {
        let crease = max(1.0 - abs(gradient_noise(p * frequency) * RIDGE_GAIN), 0.0);
        total = total + crease * crease * amplitude;
        max_amplitude = max_amplitude + amplitude;
        amplitude = amplitude * persistence;
        frequency = frequency * lacunarity;
    }
    return total / max_amplitude;
}

// `fbm` stretched around its midpoint onto 0..1, which is the reading the noise
// layers take and the one a height field's range expects.
fn fbm_unit(p: vec2<f32>, octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
    return clamp(fbm(p, octaves, persistence, lacunarity) * NOISE_GAIN * 0.5 + 0.5, 0.0, 1.0);
}

// The document position, in cells, of the texel this invocation writes.
fn cell_position(id: vec2<u32>) -> vec2<f32> {
    let texel = id + globals.origin;
    let step = f32(1u << globals.shift);
    return (vec2<f32>(texel) + vec2<f32>(0.5, 0.5)) * step;
}
