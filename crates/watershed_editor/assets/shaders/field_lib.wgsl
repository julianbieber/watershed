// What every field's shader is compiled against: the bindings a dispatch supplies, and
// the noise primitives a shader is expected to reach for.
//
// The Rust half of this file is `gpu.rs`, which writes `Globals` out as
// `DispatchGlobals` and prepends this source to every shader. A change here is a
// change there.

// Everything a shader is told about where it is being evaluated. Positions are in
// document cells, not in a normalised coordinate, so a scale means the same thing at
// every shift.
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

// The one hash. Its exact output is part of what a saved document means.
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
// it. Zero octaves gives a NaN.
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

// `fbm` stretched around its midpoint onto 0..1, which is the reading a height field's
// range expects.
fn fbm_unit(p: vec2<f32>, octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
    return clamp(fbm(p, octaves, persistence, lacunarity) * NOISE_GAIN * 0.5 + 0.5, 0.0, 1.0);
}

// A displacement in lattice units drawn from `seed` and `salt`, added to a noise
// position so two seeds, or two noises under one seed, read different stretches of the
// same lattice. Up to about a thousand lattice cells on each axis.
fn seed_offset(seed: u32, salt: u32) -> vec2<f32> {
    let h = hash2(i32(seed), i32(salt));
    return vec2<f32>(f32(h & 0xffffu), f32(h >> 16u)) / 64.0;
}

// The document position, in cells, of the texel this invocation writes.
fn cell_position(id: vec2<u32>) -> vec2<f32> {
    let texel = id + globals.origin;
    let step = f32(1u << globals.shift);
    return (vec2<f32>(texel) + vec2<f32>(0.5, 0.5)) * step;
}

// The document's extent, in cells.
fn document_extent() -> vec2<f32> {
    return vec2<f32>(globals.document);
}

// A document position as a 0..1 coordinate across the document, which is the reading a
// shader written against a normalised space wants. Not square: a document wider than it
// is tall stretches, because the coordinate runs 0..1 on both axes whatever the extent
// is.
fn uv(p: vec2<f32>) -> vec2<f32> {
    return p / document_extent();
}

// The texel of this field's own raster a document position falls in — the inverse of
// `cell_position`, and how a shader turns a neighbour's position into an index into
// a layer.
fn field_texel(p: vec2<f32>) -> vec2<i32> {
    return vec2<i32>(floor(p / f32(1u << globals.shift)));
}

// One texel of a layer texture, clamped to its edge — so a neighbourhood read at the
// border repeats the edge rather than reading nothing, and a layer with no raster,
// which is one texel of zero, reads 0.0 everywhere.
fn input_texel(source: texture_2d<f32>, at: vec2<i32>) -> f32 {
    let last = vec2<i32>(textureDimensions(source)) - vec2<i32>(1, 1);
    return textureLoad(source, clamp(at, vec2<i32>(0, 0), last), 0).r;
}

// The shift a bound layer was baked at: the smallest one whose raster covers the
// document at the layer texture's own extent. Every shift reads a one-texel layer the
// same way, so which one answers for it does not matter.
fn layer_shift(layer: texture_2d<f32>) -> u32 {
    let size = textureDimensions(layer);
    for (var shift = 0u; shift < 16u; shift = shift + 1u) {
        let step = 1u << shift;
        let fits = max((globals.document + vec2<u32>(step - 1u)) / step, vec2<u32>(1u, 1u));
        if all(fits == size) {
            return shift;
        }
    }
    return 16u;
}

// Another field's value at a document position, interpolated between its texels and
// clamped to its edge, whatever shift it was baked at — the same read a field
// reference makes.
fn layer_value(layer: texture_2d<f32>, p: vec2<f32>) -> f32 {
    let step = f32(1u << layer_shift(layer));
    let last = vec2<f32>(textureDimensions(layer)) - vec2<f32>(1.0, 1.0);
    let at = clamp(p / step - vec2<f32>(0.5, 0.5), vec2<f32>(0.0, 0.0), last);
    let low = floor(at);
    let f = at - low;
    let i0 = vec2<i32>(low);
    let i1 = min(i0 + vec2<i32>(1, 1), vec2<i32>(last));
    let a = textureLoad(layer, i0, 0).r;
    let b = textureLoad(layer, vec2<i32>(i1.x, i0.y), 0).r;
    let c = textureLoad(layer, vec2<i32>(i0.x, i1.y), 0).r;
    let d = textureLoad(layer, i1, 0).r;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
