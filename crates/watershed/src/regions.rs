// TODO(jb-doc): module docs — what a region is here (a weight and a row of numbers),
// why the enum, the kind triple and the recipe table are deliberately not in this crate,
// and what a caller has to supply instead.

use glam::{IVec2, UVec2, Vec2};
use serde::{Deserialize, Serialize};

use crate::noise::{Warp, WarpSpec, hash2};

/// TODO(jb-doc): what confines a site to the middle of its cell, and the two things that
/// stop short of filling the whole cell buy.
pub const SITE_JITTER: f32 = 0.85;

const CELL_SALT: u32 = 0x51de_51de;
const COVER_SALT: i32 = 0x2f6b_1e59;

// TODO(jb-comment): why the cell table is bounded at all, and what a document that
// exceeds the bound falls back to.
const MAX_CACHED_CELLS: usize = 1 << 20;

/// TODO(jb-doc): why a region is a draw weight and a row of numbers, and why a name for
/// the region itself would be the thing this crate must not carry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Region {
    pub weight: u32,
    pub values: Vec<f32>,
}

impl Region {
    pub fn new(weight: u32, values: impl Into<Vec<f32>>) -> Self {
        Self {
            weight,
            values: values.into(),
        }
    }
}

/// TODO(jb-doc): the four things a caller chooses — the lattice, the band, the warp and
/// the table — and which of them a region's *interior* depends on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegionSpec {
    pub seed: u32,
    pub cell_tiles: u32,
    pub blend_tiles: u32,
    #[serde(default)]
    pub warp: Option<WarpSpec>,
    pub columns: Vec<String>,
    pub regions: Vec<Region>,
}

impl RegionSpec {
    pub fn new(
        seed: u32,
        cell_tiles: u32,
        blend_tiles: u32,
        columns: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            seed,
            cell_tiles,
            blend_tiles,
            warp: None,
            columns: columns.into_iter().map(Into::into).collect(),
            regions: Vec::new(),
        }
    }

    pub fn with_warp(mut self, warp: WarpSpec) -> Self {
        self.warp = Some(warp);
        self
    }

    pub fn with_region(mut self, region: Region) -> Self {
        self.regions.push(region);
        self
    }

    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|column| column == name)
    }
}

/// TODO(jb-doc): the three questions a region map can answer, and why two of them are
/// categorical in a crate whose fields are all `f32`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionOutput {
    Blended(String),
    RegionId,
    CoverClass,
}

/// TODO(jb-doc): why the column is resolved to an index once per bake rather than looked
/// up by name per texel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompiledOutput {
    Blended(usize),
    RegionId,
    CoverClass,
}

#[derive(Clone, Copy, Debug)]
struct Site {
    position: Vec2,
    region: u16,
}

// TODO(jb-comment): why the table is precomputed for the whole document rather than
// filled in as texels ask for cells, and what that buys the rect re-bake guard.
//
// TODO(jb-doc): what the table does *not* take off — the measured split between the cell
// lookups and the domain warp, and what that implies for the shift a region column should
// be baked at. Figures in `the_cell_table_measures_what_it_takes_off_a_blend`.
struct CellCache {
    origin: IVec2,
    size: UVec2,
    cells: Vec<Site>,
}

impl CellCache {
    fn empty() -> Self {
        Self {
            origin: IVec2::ZERO,
            size: UVec2::ZERO,
            cells: Vec::new(),
        }
    }

    fn get(&self, cell: IVec2) -> Option<Site> {
        let local = cell - self.origin;
        if local.x < 0 || local.y < 0 {
            return None;
        }
        if local.x >= self.size.x as i32 || local.y >= self.size.y as i32 {
            return None;
        }
        self.cells
            .get((local.y as u32 * self.size.x + local.x as u32) as usize)
            .copied()
    }
}

/// TODO(jb-doc): the compiled counterpart of a [`RegionSpec`], and why the whole
/// neighbourhood is read for every answer it gives.
pub struct RegionMap {
    cell_tiles: f32,
    blend_tiles: f32,
    warp: Option<Warp>,
    warp_reach: f32,
    cell_salt: i32,
    weights: Vec<u32>,
    weight_total: u32,
    columns: usize,
    values: Vec<f32>,
    cache: CellCache,
}

impl RegionMap {
    pub fn new(spec: &RegionSpec, size: UVec2) -> Self {
        Self::with_cache_limit(spec, size, MAX_CACHED_CELLS)
    }

    fn with_cache_limit(spec: &RegionSpec, size: UVec2, limit: usize) -> Self {
        let columns = spec.columns.len();
        let mut values = Vec::with_capacity(spec.regions.len() * columns);
        for region in &spec.regions {
            for index in 0..columns {
                values.push(region.values.get(index).copied().unwrap_or(0.0));
            }
        }

        let weights: Vec<u32> = spec.regions.iter().map(|region| region.weight).collect();
        let weight_total = weights.iter().copied().sum::<u32>().max(1);

        let mut map = Self {
            cell_tiles: (spec.cell_tiles.max(1)) as f32,
            blend_tiles: (spec.blend_tiles as f32).max(1.0),
            warp: spec.warp.as_ref().map(Warp::new),
            warp_reach: spec.warp.map_or(0.0, |warp| warp.amplitude.abs()),
            cell_salt: (spec.seed ^ CELL_SALT) as i32,
            weights,
            weight_total,
            columns,
            values,
            cache: CellCache::empty(),
        };
        map.cache = map.build_cache(size, limit);
        map
    }

    fn build_cache(&self, size: UVec2, limit: usize) -> CellCache {
        if self.weights.is_empty() {
            return CellCache::empty();
        }
        let reach = if self.warp_reach.is_finite() {
            self.warp_reach
        } else {
            0.0
        };
        let low = Vec2::splat(-reach) / self.cell_tiles;
        let high = (Vec2::new(size.x as f32, size.y as f32) + Vec2::splat(reach)) / self.cell_tiles;
        if !low.is_finite() || !high.is_finite() {
            return CellCache::empty();
        }

        let min = low.floor().as_ivec2() - IVec2::ONE;
        let max = high.floor().as_ivec2() + IVec2::ONE;
        let span = (max - min) + IVec2::ONE;
        if span.x <= 0 || span.y <= 0 {
            return CellCache::empty();
        }
        let count = span.x as usize * span.y as usize;
        if count > limit {
            return CellCache::empty();
        }

        let mut cells = Vec::with_capacity(count);
        for y in min.y..=max.y {
            for x in min.x..=max.x {
                cells.push(self.site_at(IVec2::new(x, y)));
            }
        }
        CellCache {
            origin: min,
            size: UVec2::new(span.x as u32, span.y as u32),
            cells,
        }
    }

    // TODO(jb-comment): why the jitter comes off the low bytes and the draw off the high
    // ones, and what a second hash per cell would cost per texel.
    fn site_at(&self, cell: IVec2) -> Site {
        let h = hash2(cell.x ^ self.cell_salt, cell.y);

        let jitter = Vec2::new((h & 0xff) as f32 / 255.0, ((h >> 8) & 0xff) as f32 / 255.0);
        let offset = (jitter - Vec2::splat(0.5)) * SITE_JITTER + Vec2::splat(0.5);
        let position = (cell.as_vec2() + offset) * self.cell_tiles;

        let mut draw = ((h >> 16) % self.weight_total) as i64;
        let mut region = 0u16;
        for (index, weight) in self.weights.iter().enumerate() {
            draw -= *weight as i64;
            region = index as u16;
            if draw < 0 {
                break;
            }
        }

        Site { position, region }
    }

    fn site(&self, cell: IVec2) -> Site {
        match self.cache.get(cell) {
            Some(site) => site,
            None => self.site_at(cell),
        }
    }

    fn value_of(&self, region: u16, column: usize) -> f32 {
        if self.columns == 0 || column >= self.columns {
            return 0.0;
        }
        self.values
            .get(region as usize * self.columns + column)
            .copied()
            .unwrap_or(0.0)
    }

    /// TODO(jb-doc): the coordinate space this takes, and why the query is warped where
    /// the lattice is not.
    pub fn blended(&self, x: f32, y: f32) -> Blended<'_> {
        let position = Vec2::new(x, y);
        let query = match &self.warp {
            Some(warp) => warp.apply(position),
            None => position,
        };

        let base = (query / self.cell_tiles).floor().as_ivec2();

        // TODO(jb-comment): why a 3x3 neighbourhood is enough given the jitter bound, and
        // what a miss would cost against what 5x5 would.
        let mut sites = [Site {
            position: Vec2::ZERO,
            region: 0,
        }; 9];
        let mut distances = [0.0f32; 9];
        let mut nearest = f32::MAX;
        let mut slot = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let site = self.site(base + IVec2::new(dx, dy));
                let distance = site.position.distance(query);
                nearest = nearest.min(distance);
                sites[slot] = site;
                distances[slot] = distance;
                slot += 1;
            }
        }

        let mut weights = [0.0f32; 9];
        let mut total = 0.0;
        let mut dominant = 0u16;
        let mut dominant_weight = -1.0;
        for slot in 0..9 {
            // TODO(jb-comment): what the excess measures, and why a tile B from a
            // boundary is 2B further from the loser than from the winner.
            let excess = (distances[slot] - nearest) / (2.0 * self.blend_tiles);
            if excess >= 1.0 {
                continue;
            }
            let weight = (1.0 - excess) * (1.0 - excess);
            weights[slot] = weight;
            total += weight;
            if weight > dominant_weight {
                dominant_weight = weight;
                dominant = sites[slot].region;
            }
        }

        // TODO(jb-comment): why the cover draw is hashed on the unwarped tile, and why
        // nothing about it is random at run time.
        let mut cover = dominant;
        if total > 0.0 {
            let draw =
                hash2(x.floor() as i32 ^ COVER_SALT, y.floor() as i32) as f32 / u32::MAX as f32;
            let mut climbed = 0.0;
            for slot in 0..9 {
                if weights[slot] == 0.0 {
                    continue;
                }
                climbed += weights[slot] / total;
                if draw <= climbed {
                    cover = sites[slot].region;
                    break;
                }
            }
        }

        Blended {
            map: self,
            sites,
            weights,
            total,
            dominant,
            cover,
        }
    }

    pub fn sample(&self, output: CompiledOutput, x: f32, y: f32) -> f32 {
        let blended = self.blended(x, y);
        match output {
            CompiledOutput::Blended(column) => blended.column(column),
            CompiledOutput::RegionId => blended.dominant as f32,
            CompiledOutput::CoverClass => blended.cover as f32,
        }
    }
}

/// TODO(jb-doc): what this holds and why it borrows the map rather than copying a row of
/// blended numbers out of it.
pub struct Blended<'a> {
    map: &'a RegionMap,
    sites: [Site; 9],
    weights: [f32; 9],
    total: f32,
    pub dominant: u16,
    pub cover: u16,
}

impl Blended<'_> {
    pub fn column(&self, index: usize) -> f32 {
        if self.total <= 0.0 {
            return 0.0;
        }
        let mut sum = 0.0;
        for slot in 0..9 {
            if self.weights[slot] == 0.0 {
                continue;
            }
            let share = self.weights[slot] / self.total;
            sum += self.map.value_of(self.sites[slot].region, index) * share;
        }
        sum
    }

    /// TODO(jb-doc): what makes a tile interior, and why that is the question every guard
    /// in this module asks first.
    pub fn is_interior(&self) -> bool {
        self.weights.iter().filter(|weight| **weight > 0.0).count() == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: usize = 0;
    const RIDGE: usize = 1;

    fn spec() -> RegionSpec {
        RegionSpec::new(0x5eed_0036, 384, 48, ["base", "ridge"])
            .with_region(Region::new(6, [0.20, 0.0]))
            .with_region(Region::new(4, [0.52, 0.0]))
            .with_region(Region::new(4, [0.55, 0.02]))
            .with_region(Region::new(3, [0.70, 0.34]))
            .with_region(Region::new(2, [0.51, 0.04]))
            .with_region(Region::new(2, [0.47, 0.0]))
            .with_warp(WarpSpec {
                seed: 0x7a1d_0b37,
                amplitude: 160.0,
                scale: 1.0 / (384.0 * 0.75),
                octaves: 3,
            })
    }

    fn map() -> RegionMap {
        RegionMap::new(&spec(), UVec2::splat(4096))
    }

    #[test]
    fn a_blend_of_like_regions_is_that_region() {
        let flat = RegionSpec::new(0x1111, 384, 48, ["base"])
            .with_region(Region::new(1, [0.44]))
            .with_region(Region::new(1, [0.44]))
            .with_region(Region::new(1, [0.44]));
        let map = RegionMap::new(&flat, UVec2::splat(4096));

        for i in 0..2048 {
            let (x, y) = ((i * 17 % 4096) as f32, (i * 131 % 4096) as f32);
            let base = map.blended(x, y).column(BASE);
            assert!(
                (base - 0.44).abs() < 1e-5,
                "a blend of like regions sampled {base} rather than 0.44"
            );
        }
    }

    #[test]
    fn the_cover_draw_is_a_no_op_inside_a_region() {
        let map = map();
        let mut interior = 0;

        for i in 0..8192 {
            let (x, y) = ((i * 7 % 4096) as f32, (i * 97 % 4096) as f32);
            let blended = map.blended(x, y);
            if blended.is_interior() {
                interior += 1;
                assert_eq!(
                    blended.cover, blended.dominant,
                    "a tile inside region {} took its cover from {}",
                    blended.dominant, blended.cover
                );
            }
        }

        assert!(interior > 1000, "only {interior} interior tiles sampled");
    }

    #[test]
    fn a_weight_is_zero_beyond_the_blend_band() {
        let map = map();
        let mut interior = 0;

        for i in 0..4096 {
            let (x, y) = ((i * 7 % 4096) as f32, (i * 97 % 4096) as f32);
            let blended = map.blended(x, y);
            if !blended.is_interior() {
                continue;
            }
            interior += 1;
            let pure = map.value_of(blended.dominant, BASE);
            let mixed = blended.column(BASE);
            assert!(
                (mixed - pure).abs() < 1e-5,
                "an interior tile blended {mixed} where its own region is {pure}"
            );
        }

        assert!(
            interior > 4096 / 2,
            "only {interior}/4096 sampled tiles carry an unblended row"
        );
    }

    #[test]
    fn a_blend_is_a_pure_function_of_position() {
        let map = map();
        for i in 0..64 {
            let (x, y) = ((1000 + i * 37) as f32, (2000 + i * 53) as f32);
            let first = map.blended(x, y);
            let second = map.blended(x, y);
            assert_eq!(first.dominant, second.dominant);
            assert_eq!(first.cover, second.cover);
            assert_eq!(first.column(BASE).to_bits(), second.column(BASE).to_bits());
        }
    }

    #[test]
    fn a_blended_column_stays_within_the_range_of_its_parts() {
        let spec = spec();
        let map = RegionMap::new(&spec, UVec2::splat(4096));
        let lowest = spec
            .regions
            .iter()
            .map(|region| region.values[BASE])
            .fold(f32::MAX, f32::min);
        let highest = spec
            .regions
            .iter()
            .map(|region| region.values[BASE])
            .fold(f32::MIN, f32::max);

        for i in 0..2048 {
            let (x, y) = ((i * 17 % 4096) as f32, (i * 131 % 4096) as f32);
            let base = map.blended(x, y).column(BASE);
            assert!(
                (lowest - 1e-4..=highest + 1e-4).contains(&base),
                "a blended column sampled {base} outside {lowest}..{highest}"
            );
        }
    }

    #[test]
    fn a_blended_column_is_continuous_across_a_boundary() {
        let spec = spec();
        let map = RegionMap::new(&spec, UVec2::splat(4096));
        let spread = spec
            .regions
            .iter()
            .map(|region| region.values[BASE])
            .fold(f32::MIN, f32::max)
            - spec
                .regions
                .iter()
                .map(|region| region.values[BASE])
                .fold(f32::MAX, f32::min);

        let mut worst = 0.0f32;
        for i in 0..4096 {
            let (x, y) = ((i * 13 % 4096) as f32, (i * 211 % 4096) as f32);
            let here = map.blended(x, y);
            for (dx, dy) in [(1.0, 0.0), (0.0, 1.0)] {
                let next = map.blended(x + dx, y + dy);
                worst = worst.max((here.column(BASE) - next.column(BASE)).abs());
                worst = worst.max((here.column(RIDGE) - next.column(RIDGE)).abs());
            }
        }

        let bound = spread * 0.05;
        assert!(
            worst < bound,
            "a column steps by {worst} between adjacent tiles, over the {bound} allowed"
        );
    }

    #[test]
    fn the_cover_draw_mixes_both_regions_across_a_boundary() {
        let map = map();
        let mixed = (0..16384)
            .filter(|i| {
                let (x, y) = ((i * 13 % 4096) as f32, (i * 211 % 4096) as f32);
                let blended = map.blended(x, y);
                blended.cover != blended.dominant
            })
            .count();

        assert!(
            mixed > 100,
            "only {mixed}/16384 tiles take their cover from a neighbouring region"
        );
    }

    #[test]
    fn every_region_is_drawn_somewhere() {
        let map = map();
        for region in 0..6u16 {
            let found = (0..64 * 64).any(|i| {
                let cell = IVec2::new(i % 64, i / 64);
                map.site(cell).region == region
            });
            assert!(found, "region {region} is never drawn");
        }
    }

    #[test]
    fn a_zero_weight_region_is_never_drawn() {
        let spec = RegionSpec::new(0x2222, 64, 8, ["base"])
            .with_region(Region::new(1, [0.1]))
            .with_region(Region::new(0, [0.9]));
        let map = RegionMap::new(&spec, UVec2::splat(2048));

        for i in 0..64 * 64 {
            let cell = IVec2::new(i % 64, i / 64);
            assert_ne!(map.site(cell).region, 1, "a zero-weight region was drawn");
        }
    }

    #[test]
    fn the_cell_cache_does_not_change_a_single_sample() {
        let spec = spec();
        let cached = RegionMap::new(&spec, UVec2::splat(4096));
        let uncached = RegionMap::with_cache_limit(&spec, UVec2::splat(4096), 0);

        assert!(!cached.cache.cells.is_empty(), "the cache was not built");
        assert!(
            uncached.cache.cells.is_empty(),
            "the cache was not disabled"
        );

        for i in 0..4096 {
            let (x, y) = ((i * 13 % 4096) as f32, (i * 211 % 4096) as f32);
            let one = cached.blended(x, y);
            let two = uncached.blended(x, y);
            assert_eq!(one.dominant, two.dominant, "at {x},{y}");
            assert_eq!(one.cover, two.cover, "at {x},{y}");
            assert_eq!(
                one.column(BASE).to_bits(),
                two.column(BASE).to_bits(),
                "at {x},{y}"
            );
        }
    }

    #[test]
    fn the_cell_cache_covers_what_the_warp_can_reach_outside_the_document() {
        let map = map();
        let reach = map.warp_reach;
        for (x, y) in [(0.0f32, 0.0f32), (4095.0, 4095.0), (0.0, 4095.0)] {
            for (dx, dy) in [(-reach, -reach), (reach, reach)] {
                let cell = (Vec2::new(x + dx, y + dy) / map.cell_tiles)
                    .floor()
                    .as_ivec2();
                assert!(
                    map.cache.get(cell).is_some(),
                    "cell {cell} inside the warp's reach is not in the table"
                );
            }
        }
    }

    #[test]
    fn a_region_map_with_no_regions_samples_as_zero() {
        let spec = RegionSpec::new(0x3333, 64, 8, ["base"]);
        let map = RegionMap::new(&spec, UVec2::splat(256));
        assert_eq!(map.sample(CompiledOutput::Blended(BASE), 12.5, 3.5), 0.0);
        assert_eq!(map.sample(CompiledOutput::RegionId, 12.5, 3.5), 0.0);
    }

    #[test]
    fn a_column_that_is_not_in_the_table_reads_as_zero_rather_than_a_panic() {
        let map = map();
        assert_eq!(map.blended(120.0, 340.0).column(9), 0.0);
    }

    #[test]
    fn a_region_id_is_the_index_of_the_region_the_tile_is_in() {
        let map = map();
        for i in 0..512 {
            let (x, y) = ((i * 37 % 4096) as f32, (i * 71 % 4096) as f32);
            let blended = map.blended(x, y);
            assert_eq!(
                map.sample(CompiledOutput::RegionId, x, y),
                blended.dominant as f32
            );
            assert_eq!(
                map.sample(CompiledOutput::CoverClass, x, y),
                blended.cover as f32
            );
        }
    }

    /// TODO(jb-doc): what these figures are for, which part of a blend the table removes
    /// and which part it cannot. `cargo test --release -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn the_cell_table_measures_what_it_takes_off_a_blend() {
        let spec = spec();
        let size = UVec2::splat(4096);
        let cached = RegionMap::new(&spec, size);
        let uncached = RegionMap::with_cache_limit(&spec, size, 0);

        let walk = |map: &RegionMap| -> (std::time::Duration, f32) {
            let started = std::time::Instant::now();
            let mut sink = 0.0f32;
            for j in 0..512 {
                for i in 0..512 {
                    sink += map.blended(i as f32 * 8.0, j as f32 * 8.0).column(BASE);
                }
            }
            (started.elapsed(), sink)
        };

        let (warm, _) = walk(&cached);
        let (cold, _) = walk(&uncached);
        let (with_table, sink) = walk(&cached);
        let (without_table, other) = walk(&uncached);
        assert_eq!(sink.to_bits(), other.to_bits());

        println!("first pass: cached {warm:?}, uncached {cold:?}");
        println!("262144 blends with the table:    {with_table:?}");
        println!("262144 blends without the table: {without_table:?}");
        println!(
            "the table holds {} cells and takes {:.1}% off a blend",
            cached.cache.cells.len(),
            100.0 * (1.0 - with_table.as_secs_f64() / without_table.as_secs_f64())
        );

        let unwarped = RegionMap::new(
            &RegionSpec {
                warp: None,
                ..spec.clone()
            },
            size,
        );
        let (no_warp, _) = walk(&unwarped);
        println!("262144 blends with no warp at all: {no_warp:?}");
    }

    #[test]
    fn a_region_spec_survives_a_serde_round_trip() {
        let spec = spec();
        let encoded = serde_json::to_string(&spec).expect("a spec serializes");
        let decoded: RegionSpec = serde_json::from_str(&encoded).expect("a spec deserializes");
        assert_eq!(spec, decoded);

        let before = RegionMap::new(&spec, UVec2::splat(1024));
        let after = RegionMap::new(&decoded, UVec2::splat(1024));
        for i in 0..64 {
            let x = i as f32 * 17.0;
            assert_eq!(
                before.blended(x, -x).column(BASE).to_bits(),
                after.blended(x, -x).column(BASE).to_bits()
            );
        }
    }

    #[test]
    fn a_spec_with_no_warp_reads_as_the_unwarped_lattice() {
        let decoded: RegionSpec = serde_json::from_str(
            r#"{"seed":7,"cell_tiles":384,"blend_tiles":48,"columns":["base"],
                "regions":[{"weight":1,"values":[0.5]}]}"#,
        )
        .expect("a bare spec deserializes");
        assert!(decoded.warp.is_none());
        assert_eq!(decoded.column_index("base"), Some(0));
        assert_eq!(decoded.column_index("ridge"), None);
    }
}
