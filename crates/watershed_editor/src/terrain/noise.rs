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

const RIDGE_GAIN: f32 = 2.0;

fn domain_offset(seed: u32, salt: u32) -> Vec2 {
    let h = hash2(seed as i32, salt as i32);
    Vec2::new((h & 0xffff) as f32 / 64.0, (h >> 16) as f32 / 64.0)
}

/// A seed derived from `seed` and `index`, for a caller that needs several
/// independent fields but was given one seed.
///
/// Distinct indices give unrelated fields. `index` is offset before hashing, so
/// `sub_seed(s, 0)` is not `hash2(s, 0)` and therefore not the offset an unsalted
/// field of seed `s` lands on.
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
    /// A field at the module's default octave count. `scale` multiplies the
    /// position, so a smaller scale means larger features.
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
    /// Octaves are explicit and deliberately few: the field is useful for the way
    /// its sign holds over a long run and then reverses, and a fine octave only adds
    /// a wobble on top of that.
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
    /// Two fields differing only in `salt` have unrelated ridge lines; two differing
    /// only in `scale` have the same lines at a different size.
    pub fn new(seed: u32, salt: u32, scale: f32, octaves: u32) -> Self {
        Self {
            offset: domain_offset(seed, salt),
            scale,
            octaves,
        }
    }

    /// Sample at a global tile position, in [0, 1] and never negative.
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

/// The crate's one hash: a pure function of two integers, the same on every
/// platform and in every build.
///
/// Every field's lattice and every domain offset comes off this, so its exact
/// output is part of what a saved document means — changing it re-generates every
/// world already on disk. A test pins specific values for that reason.
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

fn gradient(ix: i32, iy: i32) -> (f32, f32) {
    let h = hash2(ix, iy);
    let angle = (h as f32 / u32::MAX as f32) * std::f32::consts::TAU;
    (angle.cos(), angle.sin())
}

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// One octave of gradient noise at a position in lattice units, roughly in
/// [-1, 1] — "roughly" because the nominal bound is rarely approached, which is why
/// the fields built on it stretch their output.
///
/// Exactly zero at every integer lattice point.
pub fn gradient_noise_2d(x: f32, y: f32) -> f32 {
    gradient_noise_with(x, y, gradient)
}

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

/// Fractional Brownian motion: `octaves` of [`gradient_noise_2d`] summed, each at
/// `lacunarity` times the frequency and `persistence` times the amplitude of the
/// one before, divided by the total amplitude.
///
/// That division is what makes the result independent of `octaves`, so changing the
/// octave count refines a field rather than rescaling it. Roughly in [-1, 1], and
/// in practice much narrower — the octaves seldom align. Zero octaves is not an
/// error and gives NaN.
pub fn fbm(x: f32, y: f32, octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
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

    total / max_amplitude
}

/// [`fbm`] read for its creases: each octave contributes `(1 - |n|)^2` instead of
/// `n`, so the zero crossings of the lattice — which are curves, not points — come
/// out as the high ground. Squaring sharpens the crest; without it a ridge is a
/// broad welt.
///
/// Output is in [0, 1] with no midpoint, unlike [`fbm`], and never negative — a
/// ridged field only ever raises the ground it is added to.
///
/// Each octave is stretched before it is inverted, because gradient noise rarely
/// reaches its nominal range and an unstretched crease comes out as a broad welt.
/// The stretch clips: positions where the underlying octave is far from zero all
/// contribute exactly nothing rather than a small negative.
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

/// A direction and an elongation, which is what turns a field of blobs into a field
/// of bands.
///
/// Plain gradient noise is isotropic: its features are the same size in every
/// direction, so it can make hills but not a fold belt or a run of dunes, both of
/// which are long in one direction and short across it. This states that direction
/// and how much longer.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SampleTransform {
    /// The direction the field varies *fastest* in, in degrees counterclockwise
    /// from the x axis. The bands therefore run perpendicular to it.
    pub strike_degrees: f32,
    /// How many times further you must travel along a band to see it change as
    /// across it. `1.0` is isotropic; below `1.0` is clamped up to it, so the
    /// elongation is always along the bands and never across them.
    pub aspect: f32,
}

impl SampleTransform {
    /// No rotation and no elongation — the field is left exactly as it was. The
    /// default, and what a document written before this type existed reads as.
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

/// A [`SampleTransform`] with its rotation resolved, ready to apply per sample.
///
/// The sine and cosine are taken once when this is built rather than at every
/// position, so a caller constructs one per field and reuses it across the bake.
#[derive(Clone, Copy, Debug)]
pub struct Anisotropy {
    axis: Vec2,
    aspect: f32,
}

impl Anisotropy {
    /// An aspect below `1.0` is raised to it rather than being read as an
    /// elongation the other way, so `strike_degrees` alone decides which way the
    /// bands run and the two settings cannot contradict each other.
    pub fn new(transform: SampleTransform) -> Self {
        let (sin, cos) = transform.strike_degrees.to_radians().sin_cos();
        Self {
            axis: Vec2::new(cos, sin),
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

/// A displacement applied to a position before a field is sampled at it, which
/// bends the field's features instead of moving them.
///
/// The bending only happens because the displacement *differs* over short
/// distances: a warp whose finest octave is longer than the feature it is meant to
/// bend translates that feature bodily and leaves its edges exactly as straight as
/// it found them. So `scale` has to be fine relative to the field being warped.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WarpSpec {
    /// Seed of the displacement, independent of the seed of the field it warps.
    pub seed: u32,
    /// Bound on the displacement, in the same units as the position: neither
    /// component ever moves further than this.
    pub amplitude: f32,
    /// Scale of the displacement field. Must be fine relative to the field being
    /// warped for the warp to bend rather than translate.
    pub scale: f32,
    /// Octaves of the displacement field.
    pub octaves: u32,
    /// Salts for the two component fields, for a caller that must place them
    /// exactly — a document assembled elsewhere, or one that needs a warp to match
    /// a field it was authored against.
    ///
    /// `None` derives both from `seed` via [`sub_seed`], which is what a document
    /// written before this field existed reads as and must keep meaning.
    #[serde(default)]
    pub salts: Option<(u32, u32)>,
}

/// A [`WarpSpec`] with its two component fields built.
///
/// Two independent fields, one per axis: a single field would displace x and y by
/// the same amount at every position, which is a slide along the diagonal rather
/// than a warp.
pub struct Warp {
    x: NoiseField,
    y: NoiseField,
    amplitude: f32,
}

impl Warp {
    /// Builds both component fields, from `spec.salts` if it names them and from
    /// [`sub_seed`] of `spec.seed` otherwise.
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

    /// The displaced position. Neither component moves further than the spec's
    /// amplitude, so the result stays inside a square of that half-width.
    pub fn apply(&self, position: Vec2) -> Vec2 {
        let displacement = Vec2::new(
            self.x.sample(position.x, position.y) - 0.5,
            self.y.sample(position.x, position.y) - 0.5,
        ) * (2.0 * self.amplitude);
        position + displacement
    }
}

/// Which reading of the same lattice a [`NoiseSpec`] asks for. The three differ in
/// the range they hand back, so they are not interchangeable in a stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseKind {
    /// Summed octaves, stretched and remapped to [0, 1]. The general-purpose
    /// reading: a height, a wetness, a density.
    Fbm,
    /// The same sum in [-1, 1], for a quantity whose natural zero is "neither way"
    /// rather than "lowest" — a lean or a bias.
    Signed,
    /// The lattice read for its zero crossings, in [0, 1] and never negative. Its
    /// maxima form connected lines rather than isolated lumps, which is what a
    /// mountain range needs and no amount of thresholding gets out of `Fbm`.
    Ridged,
}

/// A complete description of one noise field: everything needed to reproduce it,
/// and nothing about what is done with the result.
///
/// Scaling the value, deciding where it applies and combining it with anything else
/// belong to the [`Layer`](crate::terrain::layer::Layer) holding the spec, so the same spec
/// means the same field wherever it is used.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoiseSpec {
    /// The document's seed. Two specs differing only here sample unrelated
    /// landscapes.
    pub seed: u32,
    /// Which reading of the lattice, and so which range the samples are in.
    pub kind: NoiseKind,
    /// Multiplies the position; a smaller scale gives larger features.
    pub scale: f32,
    /// Octaves summed. More refines the field rather than rescaling it.
    pub octaves: u32,
    /// Separates fields that share a seed, so one document seed can own as many
    /// independent fields as it has layers.
    ///
    /// A salt of `0` is what a document written before this field existed
    /// deserializes to, and it has to keep landing exactly where a bare seed always
    /// did — otherwise every terrain already saved bakes into different ground.
    #[serde(default)]
    pub salt: u32,
    /// Direction and elongation. Defaults to isotropic.
    #[serde(default)]
    pub transform: SampleTransform,
    /// Optional domain warp, applied to the position before the transform.
    #[serde(default)]
    pub warp: Option<WarpSpec>,
}

impl NoiseSpec {
    /// The module's default octave count, no salt, isotropic and unwarped.
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

    /// Sets the octave count.
    pub fn with_octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }

    /// Sets the salt. A non-zero salt moves the field somewhere unrelated.
    pub fn with_salt(mut self, salt: u32) -> Self {
        self.salt = salt;
        self
    }

    /// Sets the direction and elongation.
    pub fn with_transform(mut self, transform: SampleTransform) -> Self {
        self.transform = transform;
        self
    }

    /// Sets the domain warp, replacing any already set.
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

/// A [`NoiseSpec`] with its field, rotation and warp built, ready to sample.
///
/// Everything derivable from the spec is resolved here rather than per position, so
/// a caller builds one per field before a bake and samples it for every texel.
pub struct Noise {
    source: NoiseSource,
    anisotropy: Anisotropy,
    warp: Option<Warp>,
}

impl Noise {
    /// Builds from the spec. Two `Noise` built from equal specs sample identically,
    /// which is what makes a document reproducible.
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

    /// The field's value at a position, in whatever range the spec's
    /// [`NoiseKind`] hands back.
    ///
    /// The warp displaces the position first and the transform stretches the
    /// result, so the warp bends the bands the transform makes. Applied the other
    /// way the warp would be stretched along with them and could not break them up.
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

    // The seam is the whole reason the field exists: the weather overlay bakes one
    // period into a texture and scrolls it forever, so a discontinuity at the wrap
    // would be a line marching across the sky.
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

    // Wrapping the lattice must not flatten the field into a constant — a tiling
    // field that is all one value would also "match across the seam".
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

    // The whole crate is a pure function of this hash, so these literals are the
    // file format's real compatibility surface: changing any of them silently
    // re-generates every world already on disk rather than failing to load it.
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

    // A document written before the salt existed deserializes with `salt: 0`, so
    // zero has to keep meaning exactly what a bare seed meant — otherwise every
    // terrain already saved to disk bakes into a different landscape.
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

    // The salt is what lets one seed own a whole document's worth of fields that
    // are independent of each other rather than views of one landscape.
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

    // A warp that names its salts must place its two component fields where those
    // salts put them, not where `sub_seed` would have.
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

    // Reproducibility is what a seed is for, and the transform and the warp are the
    // two stages with state of their own — either caching a value across
    // construction would break it.
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

    // The other half of what a seed is for. A hash that mixed its arguments too
    // weakly would still be reproducible and would still pass every test above.
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

    // Every unstretched field goes through `Anisotropy::apply` anyway, so an error
    // in the rotation would move fields nobody configured.
    #[test]
    fn an_identity_transform_leaves_a_position_where_it_found_it() {
        let anisotropy = Anisotropy::new(SampleTransform::IDENTITY);
        for (x, y) in [(0.0, 0.0), (13.5, -2.25), (-800.0, 640.0)] {
            let moved = anisotropy.apply(Vec2::new(x, y));
            assert!((moved.x - x).abs() < 1e-4 && (moved.y - y).abs() < 1e-4);
        }
    }

    // Pins which way round `strike_degrees` runs: the bands come out perpendicular
    // to it. A transposed sine and cosine still produces bands, just at ninety
    // degrees to the ones asked for, which nothing else here would catch.
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

    // `amplitude` is a bound callers rely on to know how far outside a rectangle a
    // warped bake has to read; the centring subtraction and the doubling have to
    // cancel exactly or the bound is out by a factor of two.
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

    // A warp that only translates a region bodily leaves every edge exactly as
    // straight as it found it — the displacement has to differ over a distance
    // shorter than the feature being bent.
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

    // A warp built but never applied — dropped on a code path, or applied after the
    // sample — leaves the field intact and fails silently.
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

    // Callers add a ridged field rather than blending it, so a negative sample
    // would carve holes where it was meant to raise mountains. The `max(0.0)` in
    // `ridged_fbm` is the only thing preventing that.
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

    // A signed field is useful only if its sign actually reverses; one that stayed
    // on one side of zero would still be in range and would still look like noise.
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

    // A spec is what a saved document holds, so the round trip has to preserve not
    // just the fields but the field they build: the sample comparison catches a
    // value that survives serde but is read differently on the way back.
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

    // A spec written before the transform and the warp existed must still read, or
    // every terrain saved to disk is invalidated by adding a field.
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
