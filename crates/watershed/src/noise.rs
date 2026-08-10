//! Hand-rolled gradient noise. There is deliberately no noise crate here: the
//! world is a pure function of tile position, and owning the hash means that
//! stays true across platforms and dependency bumps.

use glam::Vec2;
use serde::{Deserialize, Serialize};

const NOISE_OCTAVES: u32 = 5;
const NOISE_PERSISTENCE: f32 = 0.5;
const NOISE_LACUNARITY: f32 = 2.0;
/// Normalized fbm only spans about [0.35, 0.65] in practice — the octaves rarely
/// align and gradient noise peaks well below 1. Stretching it around the midpoint
/// makes the terrain thresholds mean what they say on a [0, 1] scale.
pub const NOISE_GAIN: f32 = 2.6;

/// Spreads `|gradient_noise_2d|` over most of [0, 1] before it is inverted. Without
/// it the creases in a ridged field are shallow, because 2D gradient noise rarely
/// gets near its nominal range.
const RIDGE_GAIN: f32 = 2.0;

// TODO(jb-comment): why the offset stays small, and why a salt of zero has to keep
// hashing to exactly what a bare seed always did.
fn domain_offset(seed: u32, salt: u32) -> Vec2 {
    let h = hash2(seed as i32, salt as i32);
    Vec2::new((h & 0xffff) as f32 / 64.0, (h >> 16) as f32 / 64.0)
}

/// TODO(jb-doc): what a sub-seed is for — one spec owns one seed, but a warp needs
/// two independent fields out of it.
pub fn sub_seed(seed: u32, index: u32) -> u32 {
    hash2(seed as i32, index.wrapping_add(1) as i32)
}

/// One fbm field with its own domain offset, so two fields sampled at the same
/// position are independent rather than two views of the same landscape.
pub struct NoiseField {
    offset: Vec2,
    scale: f32,
    octaves: u32,
}

impl NoiseField {
    pub fn new(seed: u32, salt: u32, scale: f32) -> Self {
        Self::with_octaves(seed, salt, scale, NOISE_OCTAVES)
    }

    /// A field with a chosen octave count. The low-frequency layers of the terrain
    /// want fewer: an octave finer than the feature the layer is there to make is
    /// paid for on every tile and then buried under the layer above it.
    pub fn with_octaves(seed: u32, salt: u32, scale: f32, octaves: u32) -> Self {
        Self {
            offset: domain_offset(seed, salt),
            scale,
            octaves,
        }
    }

    /// Sample the field at a global tile position, remapped to [0, 1].
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let n = fbm(
            x * self.scale + self.offset.x,
            y * self.scale + self.offset.y,
            self.octaves,
            NOISE_PERSISTENCE,
            NOISE_LACUNARITY,
        );
        (0.5 + n * NOISE_GAIN * 0.5).clamp(0.0, 1.0)
    }
}

/// The same fbm read as a **signed** displacement rather than as a height.
///
/// Every other field here is remapped to [0, 1], because everything else asks it
/// "how high / how green / how wet". A meander bias asks "which way does the
/// water lean here", and that question has no natural zero at 0.5 — it has one at
/// 0, where the river runs straight. Remapping and then subtracting a half would
/// give the same numbers only until someone changed [`NOISE_GAIN`], which is
/// tuned for the terrain thresholds and not for this.
///
/// Deliberately few octaves. The value of the field is that its *sign* holds over
/// tens of tiles and then reverses — that alternation is what a meander is — and
/// a fine octave on top only adds a wobble that the lattice cannot represent
/// anyway.
pub struct SignedNoiseField {
    offset: Vec2,
    scale: f32,
    octaves: u32,
}

impl SignedNoiseField {
    pub fn new(seed: u32, salt: u32, scale: f32, octaves: u32) -> Self {
        Self {
            offset: domain_offset(seed, salt),
            scale,
            octaves,
        }
    }

    /// Sample at a global tile position, in [-1, 1]. Stretched by the same
    /// reasoning as [`NOISE_GAIN`] — raw fbm rarely gets near its nominal range,
    /// so an unstretched field would lean the water only feebly and never commit
    /// to a side.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let n = fbm(
            x * self.scale + self.offset.x,
            y * self.scale + self.offset.y,
            self.octaves,
            NOISE_PERSISTENCE,
            NOISE_LACUNARITY,
        );
        (n * NOISE_GAIN).clamp(-1.0, 1.0)
    }
}

/// The same lattice read for its creases instead of its peaks: a ridged field is
/// large where the underlying noise crosses zero, so its maxima form connected
/// *lines* rather than isolated blobs.
///
/// That is the whole reason it exists. A mountain range is a ridge line with
/// spurs; plain fbm over the same domain gives a field of separate lumps, and no
/// amount of thresholding turns one into the other.
///
/// Already in [0, 1] and one-sided — the value is a height to add, not a
/// displacement around a midpoint, so it never lowers the terrain it is added to.
pub struct RidgedNoiseField {
    offset: Vec2,
    scale: f32,
    octaves: u32,
}

impl RidgedNoiseField {
    pub fn new(seed: u32, salt: u32, scale: f32, octaves: u32) -> Self {
        Self {
            offset: domain_offset(seed, salt),
            scale,
            octaves,
        }
    }

    pub fn sample(&self, x: f32, y: f32) -> f32 {
        ridged_fbm(
            x * self.scale + self.offset.x,
            y * self.scale + self.offset.y,
            self.octaves,
            NOISE_PERSISTENCE,
            NOISE_LACUNARITY,
        )
    }
}

/// One fbm field that repeats exactly every `period` noise units in both axes.
///
/// The seam is the point: a field that tiles can be baked into a small texture and
/// scrolled forever, which is how the weather overlay animates without evaluating
/// any noise per fragment. Nothing in the terrain wants this — a world that repeats
/// every few hundred tiles would be visible from the ground.
#[derive(Clone)]
pub struct TilingNoiseField {
    offset: Vec2,
    period: u32,
    octaves: u32,
}

impl TilingNoiseField {
    /// `period` must be a power of two: every octave wraps at `period * frequency`,
    /// and with a lacunarity of 2 that is only an integer if `period` is one.
    ///
    /// `octaves` is a parameter here rather than the module's constant because the
    /// field is baked into a texture: octaves finer than a couple of texels cannot
    /// survive the sampling, and asking for them only buys aliasing.
    pub fn new(seed: u32, period: u32, octaves: u32) -> Self {
        debug_assert!(
            period.is_power_of_two(),
            "a tiling period must be a power of two"
        );
        Self {
            offset: domain_offset(seed, 0),
            period,
            octaves,
        }
    }

    /// Sample the field, remapped to [0, 1]. `u` and `v` are in noise units, of
    /// which the field holds `period` before it repeats.
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        let n = tiling_fbm(
            u + self.offset.x,
            v + self.offset.y,
            self.period,
            self.octaves,
            NOISE_PERSISTENCE,
            NOISE_LACUNARITY,
        );
        (0.5 + n * NOISE_GAIN * 0.5).clamp(0.0, 1.0)
    }
}

/// Hash function to generate pseudo-random gradients from integer coordinates.
/// No external crates — uses a simple bit-mixing hash.
pub fn hash2(x: i32, y: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x27d4eb2d);
    h ^= (y as u32).wrapping_mul(0x165667b1);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

/// Returns a pseudo-random unit gradient vector for a lattice point.
fn gradient(ix: i32, iy: i32) -> (f32, f32) {
    let h = hash2(ix, iy);
    // Map hash to an angle in [0, 2*pi)
    let angle = (h as f32 / u32::MAX as f32) * std::f32::consts::TAU;
    (angle.cos(), angle.sin())
}

/// Smoothstep-style fade curve (6t^5 - 15t^4 + 10t^3), as used in Perlin noise.
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// 2D gradient (Perlin-style) noise, returns values roughly in [-1, 1].
pub fn gradient_noise_2d(x: f32, y: f32) -> f32 {
    gradient_noise_with(x, y, gradient)
}

/// The same lattice, with the corner lookups wrapped, so the noise repeats every
/// `period` units. Generic over the lookup so the untiled path above monomorphizes
/// to exactly what it was before — this is the only interpolation in the crate.
fn tiling_gradient_noise_2d(x: f32, y: f32, period: i32) -> f32 {
    gradient_noise_with(x, y, |ix, iy| {
        gradient(ix.rem_euclid(period), iy.rem_euclid(period))
    })
}

fn gradient_noise_with(x: f32, y: f32, grad: impl Fn(i32, i32) -> (f32, f32)) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let sx = x - x0 as f32;
    let sy = y - y0 as f32;

    // Dot product of gradient and distance vector at each corner.
    let dot_grad = |ix: i32, iy: i32, dx: f32, dy: f32| -> f32 {
        let (gx, gy) = grad(ix, iy);
        gx * dx + gy * dy
    };

    let n00 = dot_grad(x0, y0, sx, sy);
    let n10 = dot_grad(x1, y0, sx - 1.0, sy);
    let n01 = dot_grad(x0, y1, sx, sy - 1.0);
    let n11 = dot_grad(x1, y1, sx - 1.0, sy - 1.0);

    let u = fade(sx);
    let v = fade(sy);

    let nx0 = lerp(n00, n10, u);
    let nx1 = lerp(n01, n11, u);

    lerp(nx0, nx1, v)
}

/// Fractal Brownian Motion: sums multiple octaves of gradient noise
/// with increasing frequency and decreasing amplitude, then normalizes.
pub fn fbm(
    x: f32,
    y: f32,
    octaves: u32,
    persistence: f32, // amplitude multiplier per octave, e.g. 0.5
    lacunarity: f32,  // frequency multiplier per octave, e.g. 2.0
) -> f32 {
    let mut total = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amplitude = 0.0;

    for _ in 0..octaves {
        total += gradient_noise_2d(x * frequency, y * frequency) * amplitude;
        max_amplitude += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    // Normalize so output stays roughly in [-1, 1] regardless of octave count.
    total / max_amplitude
}

/// [`fbm`] read for its creases: each octave contributes `(1 - |n|)^2` instead of
/// `n`, so the zero crossings of the lattice — which are curves, not points — come
/// out as the high ground. Squaring sharpens the crest; without it a ridge is a
/// broad welt.
///
/// Output is in [0, 1] with no midpoint, unlike [`fbm`].
pub fn ridged_fbm(x: f32, y: f32, octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
    let mut total = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amplitude = 0.0;

    for _ in 0..octaves {
        let crease =
            (1.0 - (gradient_noise_2d(x * frequency, y * frequency) * RIDGE_GAIN).abs()).max(0.0);
        total += crease * crease * amplitude;
        max_amplitude += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    total / max_amplitude
}

/// [`fbm`], with every octave's lattice wrapped so the sum repeats every `period`
/// units. Octave `n` runs at `period * lacunarity^n` lattice cells, which is why the
/// period has to be a power of two.
pub fn tiling_fbm(
    x: f32,
    y: f32,
    period: u32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
) -> f32 {
    let mut total = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amplitude = 0.0;

    for _ in 0..octaves {
        let lattice_period = (period as f32 * frequency) as i32;
        total += tiling_gradient_noise_2d(x * frequency, y * frequency, lattice_period) * amplitude;
        max_amplitude += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    total / max_amplitude
}

/// TODO(jb-doc): what strike and aspect mean, and why an isotropic spec cannot
/// express a fold belt or a dune crest.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SampleTransform {
    pub strike_degrees: f32,
    pub aspect: f32,
}

impl SampleTransform {
    pub const IDENTITY: Self = Self {
        strike_degrees: 0.0,
        aspect: 1.0,
    };
}

impl Default for SampleTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// TODO(jb-doc): why the rotation is precomputed here rather than taken per sample.
#[derive(Clone, Copy, Debug)]
pub struct Anisotropy {
    axis: Vec2,
    aspect: f32,
}

impl Anisotropy {
    pub fn new(transform: SampleTransform) -> Self {
        let (sin, cos) = transform.strike_degrees.to_radians().sin_cos();
        Self {
            axis: Vec2::new(cos, sin),
            // TODO(jb-comment): why an aspect below one is clamped away rather than
            // being read as the perpendicular strike.
            aspect: transform.aspect.max(1.0),
        }
    }

    /// Rotates a position onto an axis and squashes the along-axis component, so a
    /// field sampled at the result varies quickly across the axis and slowly along
    /// it — which is what turns blobs into bands.
    ///
    /// The division is what does the work: the caller's field applies one scale to
    /// both components, so pre-dividing the along-axis one by `aspect` means you
    /// must travel `aspect` times as far along the structure to see it change.
    pub fn apply(&self, position: Vec2) -> Vec2 {
        Vec2::new(
            position.x * self.axis.x + position.y * self.axis.y,
            (-position.x * self.axis.y + position.y * self.axis.x) / self.aspect,
        )
    }
}

/// TODO(jb-doc): what a domain warp buys, and the constraint that its finest octave
/// has to be shorter than the feature it is meant to bend.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WarpSpec {
    pub seed: u32,
    pub amplitude: f32,
    pub scale: f32,
    pub octaves: u32,
    /// TODO(jb-doc): why the two component fields can be salted explicitly, what
    /// `None` keeps meaning for a document written before this existed, and which
    /// kind of caller has an offset it does not get to choose.
    #[serde(default)]
    pub salts: Option<(u32, u32)>,
}

/// TODO(jb-doc): why the warp is two fields rather than one.
pub struct Warp {
    x: NoiseField,
    y: NoiseField,
    amplitude: f32,
}

impl Warp {
    pub fn new(spec: &WarpSpec) -> Self {
        let (x, y) = match spec.salts {
            Some((salt_x, salt_y)) => (
                NoiseField::with_octaves(spec.seed, salt_x, spec.scale, spec.octaves),
                NoiseField::with_octaves(spec.seed, salt_y, spec.scale, spec.octaves),
            ),
            None => (
                NoiseField::with_octaves(sub_seed(spec.seed, 0), 0, spec.scale, spec.octaves),
                NoiseField::with_octaves(sub_seed(spec.seed, 1), 0, spec.scale, spec.octaves),
            ),
        };
        Self {
            x,
            y,
            amplitude: spec.amplitude,
        }
    }

    pub fn apply(&self, position: Vec2) -> Vec2 {
        let displacement = Vec2::new(
            self.x.sample(position.x, position.y) - 0.5,
            self.y.sample(position.x, position.y) - 0.5,
        ) * (2.0 * self.amplitude);
        position + displacement
    }
}

/// TODO(jb-doc): which reading of the lattice each kind is, and what range each
/// one hands back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseKind {
    Fbm,
    Signed,
    Ridged,
}

/// TODO(jb-doc): what a spec owns and what it deliberately leaves to the layer that
/// holds it — amplitude, blending and masking are not this type's business.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoiseSpec {
    pub seed: u32,
    pub kind: NoiseKind,
    pub scale: f32,
    pub octaves: u32,
    /// TODO(jb-doc): what the salt separates that the seed does not — one seed owning
    /// a whole document's worth of independent fields — and why zero has to keep
    /// hashing to what a bare seed always did.
    #[serde(default)]
    pub salt: u32,
    #[serde(default)]
    pub transform: SampleTransform,
    #[serde(default)]
    pub warp: Option<WarpSpec>,
}

impl NoiseSpec {
    pub fn new(seed: u32, kind: NoiseKind, scale: f32) -> Self {
        Self {
            seed,
            kind,
            scale,
            octaves: NOISE_OCTAVES,
            salt: 0,
            transform: SampleTransform::IDENTITY,
            warp: None,
        }
    }

    pub fn with_octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }

    pub fn with_salt(mut self, salt: u32) -> Self {
        self.salt = salt;
        self
    }

    pub fn with_transform(mut self, transform: SampleTransform) -> Self {
        self.transform = transform;
        self
    }

    pub fn with_warp(mut self, warp: WarpSpec) -> Self {
        self.warp = Some(warp);
        self
    }
}

enum NoiseSource {
    Fbm(NoiseField),
    Signed(SignedNoiseField),
    Ridged(RidgedNoiseField),
}

/// TODO(jb-doc): the compiled counterpart of a [`NoiseSpec`] — what is precomputed
/// and why a caller builds one per field rather than per sample.
pub struct Noise {
    source: NoiseSource,
    anisotropy: Anisotropy,
    warp: Option<Warp>,
}

impl Noise {
    pub fn new(spec: &NoiseSpec) -> Self {
        let source = match spec.kind {
            NoiseKind::Fbm => NoiseSource::Fbm(NoiseField::with_octaves(
                spec.seed,
                spec.salt,
                spec.scale,
                spec.octaves,
            )),
            NoiseKind::Signed => NoiseSource::Signed(SignedNoiseField::new(
                spec.seed,
                spec.salt,
                spec.scale,
                spec.octaves,
            )),
            NoiseKind::Ridged => NoiseSource::Ridged(RidgedNoiseField::new(
                spec.seed,
                spec.salt,
                spec.scale,
                spec.octaves,
            )),
        };
        Self {
            source,
            anisotropy: Anisotropy::new(spec.transform),
            warp: spec.warp.as_ref().map(Warp::new),
        }
    }

    // TODO(jb-comment): why the warp is applied before the stretch and not after.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let mut position = Vec2::new(x, y);
        if let Some(warp) = &self.warp {
            position = warp.apply(position);
        }
        let position = self.anisotropy.apply(position);
        match &self.source {
            NoiseSource::Fbm(field) => field.sample(position.x, position.y),
            NoiseSource::Signed(field) => field.sample(position.x, position.y),
            NoiseSource::Ridged(field) => field.sample(position.x, position.y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seam is the whole reason the field exists: the weather overlay bakes one
    /// period into a texture and scrolls it forever, so a discontinuity at the wrap
    /// would be a line marching across the sky.
    #[test]
    fn a_tiling_field_matches_itself_across_the_seam() {
        let field = TilingNoiseField::new(0x5eed, 8, 4);
        let period = 8.0;

        for i in 0..64 {
            let t = i as f32 / 64.0 * period;
            for (u, v) in [(t, 1.7), (1.7, t), (t, t)] {
                assert!(
                    (field.sample(u, v) - field.sample(u + period, v + period)).abs() < 1e-5,
                    "the field does not repeat at ({u}, {v})"
                );
            }
        }
    }

    /// Wrapping the lattice must not flatten the field into a constant — a tiling
    /// field that is all one value would also "match across the seam".
    #[test]
    fn a_tiling_field_still_varies_across_its_period() {
        let field = TilingNoiseField::new(0x5eed, 8, 4);
        let samples: Vec<f32> = (0..64)
            .map(|i| field.sample(i as f32 / 8.0, 3.25))
            .collect();
        let min = samples.iter().copied().fold(f32::MAX, f32::min);
        let max = samples.iter().copied().fold(f32::MIN, f32::max);
        assert!(max - min > 0.2, "a tiling field spanning only {min}..{max}");
    }

    // TODO(jb-comment): what a change to these numbers would mean for a world
    // already saved to disk.
    #[test]
    fn the_lattice_hands_back_the_numbers_it_always_has() {
        assert_eq!(hash2(0, 0), 0x0000_0000);
        assert_eq!(hash2(1, 0), 0x80f5_75f2);
        assert_eq!(hash2(0, 1), 0x0c5c_9f25);
        assert_eq!(hash2(-7, 13), 0xfc22_0e42);

        let expected = [
            (0.0_f32, 0.0_f32, 0.0_f32),
            (0.5, 0.0, 0.499_930_77),
            (0.25, 0.75, 0.221_791_15),
            (12.5, -3.25, -0.105_630_64),
        ];
        for (x, y, want) in expected {
            let got = gradient_noise_2d(x, y);
            assert!(
                (got - want).abs() < 1e-6,
                "gradient noise at ({x}, {y}) is {got}, was {want}"
            );
        }
    }

    /// A document written before the salt existed deserializes with `salt: 0`, so
    /// zero has to keep meaning exactly what a bare seed meant — otherwise every
    /// terrain already saved to disk bakes into a different landscape.
    #[test]
    fn an_unsalted_field_samples_where_a_bare_seed_always_put_it() {
        let seed = 0xc0ff_ee01;
        assert_eq!(
            domain_offset(seed, 0),
            Vec2::new(
                (hash2(seed as i32, 0) & 0xffff) as f32 / 64.0,
                (hash2(seed as i32, 0) >> 16) as f32 / 64.0,
            )
        );
        assert_eq!(NoiseSpec::new(seed, NoiseKind::Fbm, 0.01).salt, 0);
    }

    /// The salt is what lets one seed own a whole document's worth of fields that
    /// are independent of each other rather than views of one landscape.
    #[test]
    fn two_salts_off_one_seed_do_not_sample_the_same_landscape() {
        let seed = 0xc0ff_ee01;
        let one = Noise::new(&NoiseSpec::new(seed, NoiseKind::Fbm, 0.01).with_salt(0x0000_0001));
        let two = Noise::new(&NoiseSpec::new(seed, NoiseKind::Fbm, 0.01).with_salt(0x9e37_79b9));
        let differing = (0..256)
            .filter(|i| {
                let x = *i as f32 * 3.0;
                (one.sample(x, 17.0) - two.sample(x, 17.0)).abs() > 1e-3
            })
            .count();
        assert!(
            differing > 200,
            "only {differing} of 256 samples differ between two salts"
        );
    }

    /// A warp that names its salts must place its two component fields where those
    /// salts put them, not where `sub_seed` would have.
    #[test]
    fn a_salted_warp_moves_a_position_somewhere_a_sub_seeded_one_does_not() {
        let base = WarpSpec {
            seed: 0x5eed_0036,
            amplitude: 160.0,
            scale: 1.0 / 288.0,
            octaves: 3,
            salts: None,
        };
        let salted = WarpSpec {
            salts: Some((0x7a1d_0b37, 0x9c3f_1102)),
            ..base
        };
        let unsalted = Warp::new(&base);
        let salted = Warp::new(&salted);
        let differing = (0..256)
            .filter(|i| {
                let at = Vec2::new(*i as f32 * 13.0, 91.0);
                unsalted.apply(at).distance(salted.apply(at)) > 1.0
            })
            .count();
        assert!(
            differing > 200,
            "only {differing} of 256 positions warp differently"
        );
    }

    #[test]
    fn a_field_built_twice_from_one_seed_samples_the_same_both_times() {
        let spec = NoiseSpec::new(0xc0ff_ee01, NoiseKind::Fbm, 0.01)
            .with_transform(SampleTransform {
                strike_degrees: 24.0,
                aspect: 6.0,
            })
            .with_warp(WarpSpec {
                seed: 0x1234_5678,
                amplitude: 40.0,
                scale: 0.004,
                octaves: 3,
                salts: None,
            });
        let one = Noise::new(&spec);
        let two = Noise::new(&spec);

        for i in 0..128 {
            let x = i as f32 * 7.5 - 400.0;
            let y = i as f32 * -3.25 + 90.0;
            assert_eq!(one.sample(x, y), two.sample(x, y));
        }
    }

    #[test]
    fn two_seeds_do_not_sample_the_same_landscape() {
        let one = Noise::new(&NoiseSpec::new(1, NoiseKind::Fbm, 0.01));
        let two = Noise::new(&NoiseSpec::new(2, NoiseKind::Fbm, 0.01));
        let differing = (0..256)
            .filter(|i| {
                let x = *i as f32 * 3.0;
                (one.sample(x, 17.0) - two.sample(x, 17.0)).abs() > 1e-3
            })
            .count();
        assert!(
            differing > 200,
            "only {differing} of 256 samples differ between two seeds"
        );
    }

    #[test]
    fn an_identity_transform_leaves_a_position_where_it_found_it() {
        let anisotropy = Anisotropy::new(SampleTransform::IDENTITY);
        for (x, y) in [(0.0, 0.0), (13.5, -2.25), (-800.0, 640.0)] {
            let moved = anisotropy.apply(Vec2::new(x, y));
            assert!((moved.x - x).abs() < 1e-4 && (moved.y - y).abs() < 1e-4);
        }
    }

    // TODO(jb-comment): which way round the angle runs — the bands come out
    // perpendicular to it, which is the convention wusel's `stretch` already had.
    #[test]
    fn a_stretched_field_holds_its_value_along_the_bands_it_makes() {
        let strike_degrees = 24.0_f32;
        let spec =
            NoiseSpec::new(0xba5e_1a11, NoiseKind::Fbm, 0.006).with_transform(SampleTransform {
                strike_degrees,
                aspect: 6.0,
            });
        let noise = Noise::new(&spec);

        let (sin, cos) = strike_degrees.to_radians().sin_cos();
        let along_bands = Vec2::new(-sin, cos);
        let across_bands = Vec2::new(cos, sin);

        let walk = |direction: Vec2| -> f32 {
            let mut total = 0.0;
            let mut previous = noise.sample(0.0, 0.0);
            for step in 1..=200 {
                let at = direction * (step as f32 * 2.0);
                let current = noise.sample(at.x, at.y);
                total += (current - previous).abs();
                previous = current;
            }
            total
        };

        let along_change = walk(along_bands);
        let across_change = walk(across_bands);
        assert!(
            across_change > along_change * 3.0,
            "change across the bands {across_change} is not much more than along them {along_change}"
        );
    }

    #[test]
    fn a_warp_moves_a_position_no_further_than_its_amplitude() {
        let amplitude = 160.0;
        let warp = Warp::new(&WarpSpec {
            seed: 0xfeed_face,
            amplitude,
            scale: 0.0035,
            octaves: 3,
            salts: None,
        });
        for i in 0..256 {
            let at = Vec2::new(i as f32 * 11.0 - 1000.0, i as f32 * -6.0 + 500.0);
            let moved = warp.apply(at);
            let displacement = moved - at;
            assert!(
                displacement.x.abs() <= amplitude + 1e-3
                    && displacement.y.abs() <= amplitude + 1e-3,
                "a warp of amplitude {amplitude} moved a position by {displacement}"
            );
        }
    }

    /// A warp that only translates a region bodily leaves every edge exactly as
    /// straight as it found it — the displacement has to differ over a distance
    /// shorter than the feature being bent.
    #[test]
    fn a_warp_displaces_nearby_positions_differently() {
        let scale = 0.0035_f32;
        let warp = Warp::new(&WarpSpec {
            seed: 0xfeed_face,
            amplitude: 160.0,
            scale,
            octaves: 3,
            salts: None,
        });
        let base_wavelength = 1.0 / scale;
        let step = base_wavelength / 8.0;

        let spread = (0..64)
            .map(|i| {
                let at = Vec2::new(i as f32 * step, 0.0);
                warp.apply(at) - at
            })
            .fold((f32::MAX, f32::MIN), |(min, max), d| {
                (min.min(d.x), max.max(d.x))
            });
        assert!(
            spread.1 - spread.0 > 20.0,
            "the warp displaces a whole span by nearly the same amount: {spread:?}"
        );
    }

    #[test]
    fn a_warped_field_is_not_the_field_it_warped() {
        let plain = NoiseSpec::new(0x51de_51de, NoiseKind::Fbm, 0.006);
        let warped = plain.with_warp(WarpSpec {
            seed: 0x7a1d_0b37,
            amplitude: 160.0,
            scale: 0.0035,
            octaves: 3,
            salts: None,
        });
        let plain = Noise::new(&plain);
        let warped = Noise::new(&warped);

        let differing = (0..256)
            .filter(|i| {
                let x = *i as f32 * 9.0;
                (plain.sample(x, 40.0) - warped.sample(x, 40.0)).abs() > 1e-3
            })
            .count();
        assert!(differing > 200, "the warp changed only {differing} of 256");
    }

    #[test]
    fn a_ridged_field_never_falls_below_the_ground_it_is_added_to() {
        let noise = Noise::new(&NoiseSpec::new(0xa11e, NoiseKind::Ridged, 0.02));
        for i in 0..512 {
            let x = i as f32 * 3.5;
            let value = noise.sample(x, x * 0.5);
            assert!(
                (0.0..=1.0).contains(&value),
                "a ridged field sampled {value} at ({x}, {})",
                x * 0.5
            );
        }
    }

    #[test]
    fn a_signed_field_leans_both_ways() {
        let noise = Noise::new(&NoiseSpec::new(0x1ea5, NoiseKind::Signed, 0.01).with_octaves(2));
        let samples: Vec<f32> = (0..512)
            .map(|i| noise.sample(i as f32 * 4.0, 11.0))
            .collect();
        assert!(samples.iter().any(|v| *v > 0.05), "never leans positive");
        assert!(samples.iter().any(|v| *v < -0.05), "never leans negative");
        assert!(
            samples.iter().all(|v| (-1.0..=1.0).contains(v)),
            "a signed field left [-1, 1]"
        );
    }

    #[test]
    fn a_noise_spec_survives_a_serde_round_trip() {
        let spec = NoiseSpec::new(0xdead_beef, NoiseKind::Ridged, 0.0015)
            .with_octaves(3)
            .with_transform(SampleTransform {
                strike_degrees: 65.0,
                aspect: 3.0,
            })
            .with_warp(WarpSpec {
                seed: 0x9c3f_1102,
                amplitude: 160.0,
                scale: 0.0035,
                octaves: 3,
                salts: None,
            });

        let encoded = serde_json::to_string(&spec).expect("a spec serializes");
        let decoded: NoiseSpec = serde_json::from_str(&encoded).expect("a spec deserializes");
        assert_eq!(spec, decoded);

        let before = Noise::new(&spec);
        let after = Noise::new(&decoded);
        for i in 0..64 {
            let x = i as f32 * 17.0;
            assert_eq!(before.sample(x, -x), after.sample(x, -x));
        }
    }

    /// A spec written before the transform and the warp existed must still read, or
    /// every terrain saved to disk is invalidated by adding a field.
    #[test]
    fn a_spec_with_no_transform_or_warp_reads_as_the_plain_field() {
        let decoded: NoiseSpec =
            serde_json::from_str(r#"{"seed":7,"kind":"Fbm","scale":0.01,"octaves":5}"#)
                .expect("a bare spec deserializes");
        assert_eq!(decoded.transform, SampleTransform::IDENTITY);
        assert!(decoded.warp.is_none());
        assert_eq!(decoded, NoiseSpec::new(7, NoiseKind::Fbm, 0.01));
    }
}
