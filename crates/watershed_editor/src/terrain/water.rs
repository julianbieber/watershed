//! Where water stands and where it runs on a baked height field, and the document
//! operations that produce and discard that answer.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::f32::consts::SQRT_2;

use glam::{UVec2, Vec2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::terrain::bake::TerrainSpec;
use watershed::field::FieldId;
use watershed::raster::Raster;

/// Why a document could not be solved. Every variant is about the document's state
/// rather than the solve, which cannot itself fail: on a readable height field the
/// solve always produces an answer.
#[derive(Debug, Error)]
pub enum WaterError {
    /// The document has no cells to solve over.
    #[error("terrain size has a zero component: {0} by {1}")]
    ZeroSize(u32, u32),
    /// The spec names a height field the document does not carry.
    #[error("height field `{0}` is not in the document")]
    UnknownHeightField(String),
    /// The spec names a moisture field the document does not carry.
    #[error("moisture field `{0}` is not in the document")]
    UnknownMoistureField(String),
    /// The height field is coarser than the document. Refused rather than
    /// resampled: a filled surface derived from interpolated texels would route
    /// water down slopes that are not in the document.
    #[error("height field `{0}` is at shift {1}; the solve reads one texel per cell")]
    CoarseHeight(String, u8),
    /// The height field has no baked raster at the document's size — it has not
    /// been baked, or was baked and released.
    #[error("height field `{0}` has not been baked at the document's size")]
    UnbakedHeight(String),
}

/// The flow direction code of a cell with no outflow — the bottom of a lake, or a
/// cell on the border of the document.
///
/// The other codes are `1..=8`, naming the eight neighbours in order starting at
/// `+x` and turning through `+y`: `1` is `(+1, 0)`, `2` is `(+1, +1)`, and so on
/// round to `8` at `(+1, -1)`. The numbering is part of what a saved document means
/// — [`WaterState::downstream`] and [`WaterState::flow_vector`] resolve it for a
/// caller that would rather not depend on it.
pub const SINK: u8 = 0;

const D8: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

const D8_DISTANCE: [f32; 8] = [1.0, SQRT_2, 1.0, SQRT_2, 1.0, SQRT_2, 1.0, SQRT_2];

const ACCUM_QUANT: f32 = 3900.0;

/// What a solve is to be run over, and the one threshold it takes.
///
/// The height field is named rather than found by role, so a document can solve
/// water over a field that is not its `Height` — a second surface, or a candidate
/// being previewed — without changing which field its consumers read as the ground.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WaterSpec {
    /// The field to solve over. Must be in the document, at shift 0, and baked.
    pub height: FieldId,
    /// A field weighting how much water each cell contributes. `None` gives every
    /// cell a weight of `1.0`, so an accumulation counts cells. Sampled at cell
    /// centres and floored at zero.
    pub moisture: Option<FieldId>,
    /// The smallest connected body of standing water that gets a lake id. Anything
    /// smaller keeps its depth and stays unlabelled.
    pub lake_min_cells: u32,
}

impl Default for WaterSpec {
    fn default() -> Self {
        Self {
            height: FieldId::from("height"),
            moisture: None,
            lake_min_cells: 64,
        }
    }
}

impl WaterSpec {
    /// Solves over `height`, with unit weights and the default lake threshold.
    pub fn new(height: impl Into<FieldId>) -> Self {
        Self {
            height: height.into(),
            moisture: None,
            lake_min_cells: 64,
        }
    }

    /// Weights each cell's contribution by a field instead of by `1.0`.
    pub fn with_moisture(mut self, moisture: impl Into<FieldId>) -> Self {
        self.moisture = Some(moisture.into());
        self
    }

    /// Sets the smallest body of water that earns a lake id.
    pub fn with_lake_min_cells(mut self, cells: u32) -> Self {
        self.lake_min_cells = cells;
        self
    }
}

/// The solved answer: how deep the standing water is, where each cell drains, how
/// much reaches it, and which lake it belongs to.
///
/// Four rasters at one texel per cell, and nothing derived from them — a flow
/// vector is two floats per cell that [`WaterState::flow_vector`] reconstructs from
/// one byte, so it is computed on read rather than stored.
///
/// A default state is empty: every read answers `None`, `false` or `0.0` rather
/// than panicking, which is what an unsolved or invalidated document holds.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WaterState {
    depth: Raster<f32>,
    flow_dir: Raster<u8>,
    flow_accum: Raster<u16>,
    lake_id: Raster<u32>,
    lakes: u32,
}

impl WaterState {
    /// Solves over a height raster at one texel per cell, with one weight per cell
    /// in the same row-major order.
    ///
    /// Returns an empty state — not an error — if the raster has no cells or
    /// `weight` is not exactly as long as it. Water leaves the document only at its
    /// border, so the whole grid is solved at once: a rectangle cannot be re-solved
    /// on its own, because where the water inside it goes depends on ground outside
    /// it.
    ///
    /// A pure function of its arguments: the same inputs give a bit-identical state.
    pub fn solve(height: &Raster<f32>, weight: &[f32], lake_min_cells: u32) -> Self {
        let size = height.size();
        let width = size.x as usize;
        let rows = size.y as usize;
        let cells = width * rows;
        if cells == 0 || weight.len() != cells {
            return Self::default();
        }

        let (filled, routing) = fill(height);
        let flow_dir = directions(&filled, &routing, width, rows);
        let flow_accum = accumulate(&flow_dir, weight, width, rows);
        let (lake_id, lakes) = label_lakes(&filled, height.data(), width, rows, lake_min_cells);

        let source = height.data();
        let depth: Vec<f32> = filled
            .iter()
            .zip(source)
            .map(|(surface, ground)| (surface - ground).max(0.0))
            .collect();

        Self {
            depth: Raster::from_vec(size, depth).unwrap_or_default(),
            flow_dir: Raster::from_vec(size, flow_dir).unwrap_or_default(),
            flow_accum: Raster::from_vec(size, flow_accum).unwrap_or_default(),
            lake_id: Raster::from_vec(size, lake_id).unwrap_or_default(),
            lakes,
        }
    }

    /// The extent the state was solved over, in cells. Zero for an empty state.
    pub fn size(&self) -> UVec2 {
        self.depth.size()
    }

    /// Whether nothing was solved. True for a default state.
    pub fn is_empty(&self) -> bool {
        self.depth.is_empty()
    }

    /// Standing water depth per cell, never negative. For a single cell prefer
    /// [`WaterState::depth_at`].
    pub fn depth(&self) -> &Raster<f32> {
        &self.depth
    }

    /// Flow direction codes per cell; see [`SINK`] for what they mean.
    pub fn flow_dir(&self) -> &Raster<u8> {
        &self.flow_dir
    }

    /// Quantized accumulation codes per cell. Comparable directly, but not a
    /// quantity — put one through [`dequantize_accumulation`] first.
    pub fn flow_accum(&self) -> &Raster<u16> {
        &self.flow_accum
    }

    /// Lake ids per cell: `0` for dry ground and for standing water in a body under
    /// the spec's threshold, otherwise `1..=lakes()`.
    pub fn lake_id(&self) -> &Raster<u32> {
        &self.lake_id
    }

    /// How many bodies of water met the threshold. Ids run `1..=lakes()`.
    pub fn lakes(&self) -> u32 {
        self.lakes
    }

    /// How far the water surface stands above the ground at a cell, in the height
    /// field's own units. Never negative.
    ///
    /// `None` outside the extent, and for every cell of an empty state — a caller
    /// that would treat "no water here" and "no answer here" alike wants
    /// [`WaterState::is_water`].
    pub fn depth_at(&self, x: u32, y: u32) -> Option<f32> {
        self.depth.get(x, y).copied()
    }

    /// Whether any water stands at a cell, at any depth.
    ///
    /// Reads the depth, not the lake id, so a puddle too small to have earned a lake
    /// id still counts as water. `false` outside the extent.
    pub fn is_water(&self, x: u32, y: u32) -> bool {
        self.depth.get(x, y).is_some_and(|depth| *depth > 0.0)
    }

    /// How much water reaches a cell: its own weight plus everything draining
    /// through it, so with unit weights it is a count of upstream cells.
    ///
    /// Recovered from a 16-bit log-scaled code, which costs about a part in 3900 of
    /// the value wherever on the scale it sits. `0.0` outside the extent.
    pub fn accumulation(&self, x: u32, y: u32) -> f32 {
        self.flow_accum
            .get(x, y)
            .map(|code| dequantize_accumulation(*code))
            .unwrap_or(0.0)
    }

    /// The stored code behind [`WaterState::accumulation`], for a caller comparing
    /// or thresholding rather than measuring.
    ///
    /// The quantization is monotone, so ordering codes orders accumulations and a
    /// threshold can be converted once with [`quantize_accumulation`] instead of
    /// decoding every cell. Nothing else about the code is a contract: it is not
    /// proportional to the accumulation and its scale may change.
    ///
    /// `0` outside the extent.
    pub fn accumulation_code(&self, x: u32, y: u32) -> u16 {
        self.flow_accum.get(x, y).copied().unwrap_or(0)
    }

    /// Whether at least `threshold` reaches the cell — the test that turns an
    /// accumulation field into a river network. Compares codes, so the quantization
    /// costs nothing here. `false` outside the extent.
    pub fn channel(&self, x: u32, y: u32, threshold: f32) -> bool {
        self.accumulation_code(x, y) >= quantize_accumulation(threshold)
    }

    /// A unit vector pointing the way water leaves a cell, reconstructed from the
    /// stored direction code rather than held per cell.
    ///
    /// `None` outside the extent and at a [`SINK`] — a cell with no outflow has no
    /// direction, which is not the same as a zero one.
    pub fn flow_vector(&self, x: u32, y: u32) -> Option<Vec2> {
        let code = *self.flow_dir.get(x, y)?;
        let (dx, dy) = *D8.get(direction_index(code)?)?;
        Some(Vec2::new(dx as f32, dy as f32).normalize_or_zero())
    }

    /// The cell this one drains into, in document cells.
    ///
    /// `None` outside the extent and at a [`SINK`]. Following this from any cell
    /// reaches a sink in finitely many steps — the routing surface strictly descends
    /// along every step, so a flow path cannot cycle.
    pub fn downstream(&self, x: u32, y: u32) -> Option<UVec2> {
        let code = *self.flow_dir.get(x, y)?;
        let index = direction_index(code)?;
        let size = self.size();
        neighbour(
            x as usize,
            y as usize,
            index,
            size.x as usize,
            size.y as usize,
        )
        .map(|target| {
            UVec2::new(
                (target % size.x as usize) as u32,
                (target / size.x as usize) as u32,
            )
        })
    }
}

impl TerrainSpec {
    /// The solved water, if the document has been solved and not invalidated since.
    pub fn water(&self) -> Option<&WaterState> {
        self.water.as_ref()
    }

    /// Removes the water from the document entirely — the solved state *and* the
    /// spec that produced it, so nothing will re-solve it.
    ///
    /// This is "this document has no water", not "this answer is stale"; for the
    /// latter use [`TerrainSpec::invalidate_water`].
    pub fn clear_water(&mut self) {
        self.water = None;
        self.water_spec = None;
    }

    /// Drops the solved state and **keeps the spec**, which is the difference between
    /// "forget about the water" and "the height moved, so this answer is stale".
    ///
    /// This is what an edit to the height calls: the answer no longer matches the
    /// ground, but the document still wants water, and the spec is the only record
    /// of what it wanted. A document that loses it looks like one that never had
    /// water at all, and no later solve can recover it.
    pub fn invalidate_water(&mut self) {
        self.water = None;
    }

    /// Solves water over the field `spec` names and stores both the answer and the
    /// spec on the document, replacing whatever was there.
    ///
    /// The height field must be in the document, at shift 0, and baked at the
    /// document's size; a moisture field, if named, must be in the document but may
    /// be at any shift. On any [`WaterError`] the document is left exactly as it
    /// was.
    pub fn solve_water(&mut self, spec: &WaterSpec) -> Result<(), WaterError> {
        if self.size.x == 0 || self.size.y == 0 {
            return Err(WaterError::ZeroSize(self.size.x, self.size.y));
        }

        let state = {
            let height = self
                .field(spec.height.as_str())
                .ok_or_else(|| WaterError::UnknownHeightField(spec.height.to_string()))?;
            if height.shift != 0 {
                return Err(WaterError::CoarseHeight(
                    spec.height.to_string(),
                    height.shift,
                ));
            }
            let raster = height.baked();
            if raster.size() != self.size {
                return Err(WaterError::UnbakedHeight(spec.height.to_string()));
            }

            let weight = match &spec.moisture {
                Some(id) => {
                    let field = self
                        .field(id.as_str())
                        .ok_or_else(|| WaterError::UnknownMoistureField(id.to_string()))?;
                    let width = self.size.x as usize;
                    (0..(width * self.size.y as usize))
                        .map(|index| {
                            let x = (index % width) as f32 + 0.5;
                            let y = (index / width) as f32 + 0.5;
                            field.sample(x, y).max(0.0)
                        })
                        .collect()
                }
                None => vec![1.0; self.size.x as usize * self.size.y as usize],
            };

            WaterState::solve(raster, &weight, spec.lake_min_cells)
        };

        self.water = Some(state);
        self.water_spec = Some(spec.clone());
        Ok(())
    }
}

fn fill(height: &Raster<f32>) -> (Vec<f32>, Vec<f32>) {
    let width = height.width() as usize;
    let rows = height.height() as usize;
    let cells = width * rows;
    let source = height.data();

    let mut filled = vec![0.0f32; cells];
    let mut routing = vec![0.0f32; cells];
    let mut seen = vec![false; cells];
    let mut queue: BinaryHeap<Pending> = BinaryHeap::new();

    for y in 0..rows {
        for x in 0..width {
            if x != 0 && y != 0 && x + 1 != width && y + 1 != rows {
                continue;
            }
            let index = y * width + x;
            filled[index] = source[index];
            routing[index] = source[index];
            seen[index] = true;
            queue.push(Pending {
                key: order_key(source[index]),
                index: index as u32,
            });
        }
    }

    while let Some(Pending { index, .. }) = queue.pop() {
        let index = index as usize;
        let x = index % width;
        let y = index / width;
        for step in 0..D8.len() {
            let Some(target) = neighbour(x, y, step, width, rows) else {
                continue;
            };
            if seen[target] {
                continue;
            }
            seen[target] = true;
            filled[target] = source[target].max(filled[index]);
            routing[target] = source[target].max(routing[index].next_up());
            queue.push(Pending {
                key: order_key(filled[target]),
                index: target as u32,
            });
        }
    }

    (filled, routing)
}

fn directions(filled: &[f32], routing: &[f32], width: usize, rows: usize) -> Vec<u8> {
    let mut flow = vec![SINK; width * rows];
    for y in 0..rows {
        for x in 0..width {
            let index = y * width + x;
            let mut downhill = SINK;
            let mut steepest = 0.0f32;
            let mut across = SINK;
            let mut across_steepest = 0.0f32;
            for step in 0..D8.len() {
                let Some(target) = neighbour(x, y, step, width, rows) else {
                    continue;
                };
                if filled[target] > filled[index] {
                    continue;
                }
                if filled[target] < filled[index] {
                    let slope = (filled[index] - filled[target]) / D8_DISTANCE[step];
                    if slope > steepest {
                        steepest = slope;
                        downhill = step as u8 + 1;
                    }
                } else {
                    let drop = routing[index] - routing[target];
                    if drop <= 0.0 {
                        continue;
                    }
                    let slope = drop / D8_DISTANCE[step];
                    if slope > across_steepest {
                        across_steepest = slope;
                        across = step as u8 + 1;
                    }
                }
            }
            flow[index] = if downhill == SINK { across } else { downhill };
        }
    }
    flow
}

fn accumulate(flow: &[u8], weight: &[f32], width: usize, rows: usize) -> Vec<u16> {
    let cells = width * rows;
    let downstream_of = |index: usize| -> Option<usize> {
        let step = direction_index(flow[index])?;
        neighbour(index % width, index / width, step, width, rows)
    };

    let mut incoming = vec![0u8; cells];
    for index in 0..cells {
        if let Some(target) = downstream_of(index) {
            incoming[target] += 1;
        }
    }

    let mut total: Vec<f32> = weight.to_vec();
    let mut ready: VecDeque<usize> = (0..cells).filter(|index| incoming[*index] == 0).collect();
    while let Some(index) = ready.pop_front() {
        let Some(target) = downstream_of(index) else {
            continue;
        };
        total[target] += total[index];
        incoming[target] -= 1;
        if incoming[target] == 0 {
            ready.push_back(target);
        }
    }

    total.into_iter().map(quantize_accumulation).collect()
}

fn label_lakes(
    filled: &[f32],
    source: &[f32],
    width: usize,
    rows: usize,
    lake_min_cells: u32,
) -> (Vec<u32>, u32) {
    let cells = width * rows;
    let mut label = vec![0u32; cells];
    let mut lakes = 0u32;
    let mut component: Vec<usize> = Vec::new();
    let mut frontier: VecDeque<usize> = VecDeque::new();
    let mut claimed = vec![false; cells];

    for start in 0..cells {
        if claimed[start] || filled[start] <= source[start] {
            continue;
        }
        component.clear();
        frontier.clear();
        claimed[start] = true;
        frontier.push_back(start);
        while let Some(index) = frontier.pop_front() {
            component.push(index);
            let x = index % width;
            let y = index / width;
            for step in [0usize, 2, 4, 6] {
                let Some(target) = neighbour(x, y, step, width, rows) else {
                    continue;
                };
                if claimed[target] || filled[target] <= source[target] {
                    continue;
                }
                claimed[target] = true;
                frontier.push_back(target);
            }
        }
        if component.len() as u64 >= lake_min_cells as u64 {
            lakes += 1;
            for index in &component {
                label[*index] = lakes;
            }
        }
    }

    (label, lakes)
}

fn neighbour(x: usize, y: usize, step: usize, width: usize, rows: usize) -> Option<usize> {
    let (dx, dy) = D8[step];
    let nx = x as i64 + dx as i64;
    let ny = y as i64 + dy as i64;
    if nx < 0 || ny < 0 || nx >= width as i64 || ny >= rows as i64 {
        return None;
    }
    Some(ny as usize * width + nx as usize)
}

fn direction_index(code: u8) -> Option<usize> {
    if code == SINK || code as usize > D8.len() {
        return None;
    }
    Some(code as usize - 1)
}

/// The code an accumulation is stored as. Monotone non-decreasing in `value`, which
/// is what lets a caller convert a threshold once and then compare codes.
///
/// Negative values are read as zero, and anything past the top of the scale
/// saturates at `u16::MAX` rather than wrapping.
pub fn quantize_accumulation(value: f32) -> u16 {
    let code = value.max(0.0).ln_1p() * ACCUM_QUANT;
    code.round().clamp(0.0, u16::MAX as f32) as u16
}

/// The accumulation a stored code stands for.
///
/// Inverse of [`quantize_accumulation`] up to the rounding, which costs about a
/// part in 3900 of the value — a fixed *relative* error, so a large accumulation is
/// no less accurate in proportion than a small one. The scale reaches a few tens of
/// millions before it saturates.
pub fn dequantize_accumulation(code: u16) -> f32 {
    (code as f32 / ACCUM_QUANT).exp_m1()
}

fn order_key(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Pending {
    key: u32,
    index: u32,
}

impl Ord for Pending {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .key
            .cmp(&self.key)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for Pending {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::field::Field;
    use crate::terrain::graph::NodeOp;

    fn raster_from(size: UVec2, cell: impl Fn(u32, u32) -> f32) -> Raster<f32> {
        let mut raster = Raster::new(size, 0.0f32);
        for y in 0..size.y {
            for x in 0..size.x {
                raster.set(x, y, cell(x, y));
            }
        }
        raster
    }

    fn cone(size: UVec2) -> Raster<f32> {
        let centre = Vec2::new(size.x as f32 - 1.0, size.y as f32 - 1.0) * 0.5;
        raster_from(size, |x, y| {
            -Vec2::new(x as f32, y as f32).distance(centre) / size.x as f32
        })
    }

    fn bowl(size: UVec2) -> Raster<f32> {
        raster_from(size, |x, y| {
            let edge = x == 0 || y == 0 || x + 1 == size.x || y + 1 == size.y;
            let rim = x == 1 || y == 1 || x + 2 == size.x || y + 2 == size.y;
            if edge {
                0.5
            } else if rim {
                1.0
            } else {
                0.0
            }
        })
    }

    fn unit_weights(size: UVec2) -> Vec<f32> {
        vec![1.0; size.x as usize * size.y as usize]
    }

    fn solved(height: &Raster<f32>) -> WaterState {
        WaterState::solve(height, &unit_weights(height.size()), 4)
    }

    // The one invariant everything downstream rests on: if a step could go up, a
    // flow path could cycle and the accumulation pass would deadlock rather than
    // give a wrong answer.
    #[test]
    fn no_cell_drains_uphill_on_the_filled_surface() {
        let size = UVec2::new(41, 37);
        let height = raster_from(size, |x, y| {
            let u = x as f32 * 0.31;
            let v = y as f32 * 0.27;
            (u.sin() + v.cos() + (u * 0.5 + v * 0.7).sin()) * 0.25
        });
        let state = solved(&height);
        let (filled, _) = fill(&height);

        let mut steps = 0;
        for y in 0..size.y {
            for x in 0..size.x {
                let Some(target) = state.downstream(x, y) else {
                    continue;
                };
                steps += 1;
                let here = filled[(y * size.x + x) as usize];
                let there = filled[(target.y * size.x + target.x) as usize];
                assert!(
                    there <= here,
                    "cell {x},{y} at {here} drains to {},{} at {there}",
                    target.x,
                    target.y
                );
            }
        }
        assert!(steps > 0);
    }

    // States the accumulation pass locally, which is the form an error in the
    // topological order shows up in: a cell drained before one of its contributors
    // is short by exactly that contributor and by nothing else.
    #[test]
    fn a_cells_accumulation_is_one_plus_what_flows_into_it() {
        let size = UVec2::new(29, 23);
        let height = raster_from(size, |x, y| {
            let u = x as f32 * 0.4;
            let v = y as f32 * 0.35;
            (u.cos() + v.sin()) * 0.3 - (x + y) as f32 * 0.01
        });
        let state = solved(&height);

        let mut contributed = vec![0.0f32; (size.x * size.y) as usize];
        for y in 0..size.y {
            for x in 0..size.x {
                if let Some(target) = state.downstream(x, y) {
                    contributed[(target.y * size.x + target.x) as usize] +=
                        state.accumulation(x, y);
                }
            }
        }

        for y in 0..size.y {
            for x in 0..size.x {
                let expected = 1.0 + contributed[(y * size.x + x) as usize];
                let found = state.accumulation(x, y);
                assert!(
                    (found - expected).abs() <= expected * 1e-3,
                    "cell {x},{y} holds {found} against {expected}"
                );
            }
        }
    }

    // The global counterpart: every unit of weight has to leave through some sink,
    // so nothing may be lost in a depression or counted twice on the way down.
    #[test]
    fn the_total_accumulation_equals_the_weighted_cell_count() {
        let size = UVec2::new(31, 31);
        let height = cone(size);
        let weight: Vec<f32> = (0..(size.x * size.y))
            .map(|index| 0.25 + (index % 7) as f32 * 0.1)
            .collect();
        let state = WaterState::solve(&height, &weight, 4);

        let mut delivered = 0.0f32;
        for y in 0..size.y {
            for x in 0..size.x {
                if state.downstream(x, y).is_none() {
                    delivered += state.accumulation(x, y);
                }
            }
        }
        let total: f32 = weight.iter().sum();
        assert!(
            (delivered - total).abs() <= total * 1e-3,
            "{delivered} reached the sinks against {total} laid down"
        );
    }

    // A surface whose correct answer is known everywhere, so the routing can be
    // checked against something other than itself — and one with no depressions, so
    // it also pins that the fill leaves an already-draining surface alone.
    #[test]
    fn a_cone_drains_radially() {
        let size = UVec2::new(33, 33);
        let state = solved(&cone(size));
        let centre = Vec2::new(size.x as f32 - 1.0, size.y as f32 - 1.0) * 0.5;

        for y in 0..size.y {
            for x in 0..size.x {
                let outward = Vec2::new(x as f32, y as f32) - centre;
                assert!(!state.is_water(x, y));
                if outward.length() < 2.0 || x == 0 || y == 0 || x + 1 == size.x || y + 1 == size.y
                {
                    continue;
                }
                let flow = state
                    .flow_vector(x, y)
                    .unwrap_or_else(|| panic!("cell {x},{y} has no outflow"));
                assert!(
                    flow.dot(outward.normalize()) > 0.0,
                    "cell {x},{y} flows {flow} against an outward {}",
                    outward.normalize()
                );
            }
        }
    }

    // Pins where the fill stops. Below the rim the depression is not resolved and
    // the routing has nowhere to go; above it the water spills across ground it
    // never reached.
    #[test]
    fn a_bowl_fills_to_exactly_its_rim() {
        let size = UVec2::new(11, 11);
        let height = bowl(size);
        let state = solved(&height);

        for y in 0..size.y {
            for x in 0..size.x {
                let ground = *height.get(x, y).unwrap();
                let surface = ground + state.depth().get(x, y).unwrap();
                if ground < 1.0 && x > 1 && y > 1 && x + 2 < size.x && y + 2 < size.y {
                    assert_eq!(surface, 1.0, "cell {x},{y} stands at {surface}");
                    assert!(state.is_water(x, y));
                } else {
                    assert_eq!(*state.depth().get(x, y).unwrap(), 0.0);
                }
            }
        }
        assert_eq!(state.lakes(), 1);
        assert_eq!(*state.lake_id().get(5, 5).unwrap(), 1);
        assert_eq!(*state.lake_id().get(0, 0).unwrap(), 0);
    }

    // The threshold suppresses a label, not the water: a puddle still has a depth
    // and still reads as water, so a consumer drawing surfaces and a consumer
    // listing lakes see different things by design.
    #[test]
    fn a_lake_under_the_minimum_keeps_its_depth_and_loses_its_label() {
        let size = UVec2::new(11, 11);
        let height = bowl(size);
        let state = WaterState::solve(&height, &unit_weights(size), 10_000);
        assert!(state.is_water(5, 5));
        assert_eq!(state.lakes(), 0);
        assert_eq!(*state.lake_id().get(5, 5).unwrap(), 0);
    }

    // The two public readings of one stored byte are resolved by separate code
    // paths; a neighbour table indexed off by one would leave them consistent with
    // themselves and disagreeing with each other.
    #[test]
    fn a_flow_code_and_the_cell_it_names_agree() {
        let size = UVec2::new(33, 33);
        let state = solved(&cone(size));
        for y in 0..size.y {
            for x in 0..size.x {
                let Some(target) = state.downstream(x, y) else {
                    continue;
                };
                let vector = state.flow_vector(x, y).unwrap();
                let step = Vec2::new(target.x as f32 - x as f32, target.y as f32 - y as f32);
                assert!((vector - step.normalize()).length() < 1e-6);
            }
        }
    }

    // Walks the scale from zero to past sixteen million, which is what pins the
    // error as relative: a linear quantization would pass at the small end and lose
    // whole orders of magnitude at the large one.
    #[test]
    fn an_accumulation_round_trips_through_its_quantization() {
        for value in [0.0f32, 1.0, 2.0, 17.0, 1024.0, 65_536.0, 16_777_216.0] {
            let found = dequantize_accumulation(quantize_accumulation(value));
            assert!(
                (found - value).abs() <= (value * 1e-3).max(1e-3),
                "{value} came back as {found}"
            );
        }
    }

    // Thresholding by code and by value have to agree, and this pins how far they
    // may not: only where the accumulation is within a quantization step of the
    // threshold. There the code compare is the more faithful of the two — a cell
    // holding exactly 1.0 comes back from the round trip as 0.99985945 — so a caller
    // wanting an exact count of upstream cells cannot get one from either and has to
    // decide which side of the step it wants.
    #[test]
    fn comparing_codes_answers_what_comparing_accumulations_does() {
        let size = UVec2::new(37, 29);
        let height = raster_from(size, |x, y| {
            let u = x as f32 * 0.23;
            let v = y as f32 * 0.19;
            (u.sin() + v.cos()) * 0.3 - (x + y) as f32 * 0.008
        });
        let state = solved(&height);

        let mut channels = 0u32;
        for threshold in [1.0f32, 2.5, 8.0, 40.0, 250.0] {
            let code = quantize_accumulation(threshold);
            for y in 0..size.y {
                for x in 0..size.x {
                    let value = state.accumulation(x, y);
                    let by_code = state.accumulation_code(x, y) >= code;
                    channels += u32::from(by_code);
                    if by_code == (value >= threshold) {
                        continue;
                    }
                    assert!(
                        (value - threshold).abs() <= threshold * 1e-3,
                        "cell {x},{y} at {value} disagrees about {threshold} by more than \
                         the quantization step"
                    );
                }
            }
        }
        assert!(channels > 0, "no cell cleared any threshold");
    }

    // The priority flood pops equal-height cells in an order the heap does not
    // otherwise fix; without the index tiebreak a document would solve differently
    // between runs and its saved water would not match a re-solve.
    #[test]
    fn a_solve_is_the_same_every_time_it_is_run() {
        let size = UVec2::new(37, 29);
        let height = raster_from(size, |x, y| {
            ((x * 7 + y * 13) % 11) as f32 * 0.1 - ((x + y) % 5) as f32 * 0.05
        });
        let first = solved(&height);
        let second = solved(&height);
        assert_eq!(first, second);
    }

    // The document-level path, which resolves the field by name and stores the
    // answer on the spec — everything the state tests exercise is reached through
    // this.
    #[test]
    fn a_document_solves_water_over_the_field_it_names() {
        let mut terrain = TerrainSpec::new(UVec2::new(16, 16))
            .with_field(Field::new("height").with_op(NodeOp::Constant(0.5)));
        terrain.bake_in_place().unwrap();
        assert!(terrain.water().is_none());
        terrain.solve_water(&WaterSpec::default()).unwrap();
        let water = terrain.water().unwrap();
        assert_eq!(water.size(), UVec2::new(16, 16));
        terrain.clear_water();
        assert!(terrain.water().is_none());
    }

    // The two ways of getting rid of a solved state are not interchangeable, and the
    // difference is the *recipe*: forgetting about the water drops it, invalidating a
    // stale answer keeps it. Getting this wrong makes a document that can never be solved
    // again, and looks exactly like an ordinary one.
    #[test]
    fn invalidating_the_water_keeps_the_recipe_where_clearing_it_does_not() {
        let mut terrain = TerrainSpec::new(UVec2::new(16, 16))
            .with_field(Field::new("height").with_op(NodeOp::Constant(0.5)));
        terrain.bake_in_place().unwrap();
        terrain.solve_water(&WaterSpec::default()).unwrap();

        terrain.invalidate_water();
        assert!(terrain.water().is_none());
        assert!(
            terrain.water_spec.is_some(),
            "an invalidated document has to be solvable again"
        );
        terrain
            .solve_water(&terrain.water_spec.clone().unwrap())
            .expect("the recipe survived, so the solve does");

        terrain.clear_water();
        assert!(terrain.water().is_none());
        assert!(terrain.water_spec.is_none());
    }

    // The moisture field is sampled per cell and multiplies what that cell
    // contributes; a constant field makes the total exactly predictable, so a
    // sampling offset or a dropped weight shows up as a proportional shortfall.
    #[test]
    fn a_named_moisture_field_weights_what_the_sinks_deliver() {
        let size = UVec2::new(24, 24);
        let mut terrain = TerrainSpec::new(size)
            .with_field(Field::new("height").with_op(NodeOp::Constant(0.5)))
            .with_field(
                Field::new("moisture")
                    .with_op(NodeOp::Constant(0.25))
                    .with_range((0.0, 1.0)),
            );
        terrain.bake_in_place().unwrap();
        terrain
            .solve_water(&WaterSpec::default().with_moisture("moisture"))
            .unwrap();

        let water = terrain.water().unwrap();
        let mut delivered = 0.0f32;
        for y in 0..size.y {
            for x in 0..size.x {
                if water.downstream(x, y).is_none() {
                    delivered += water.accumulation(x, y);
                }
            }
        }
        let expected = 0.25 * (size.x * size.y) as f32;
        assert!(
            (delivered - expected).abs() <= expected * 1e-3,
            "{delivered} reached the sinks against {expected}"
        );
    }

    // Each refusal is a distinct variant a caller matches on, and all three leave
    // the document untouched — a partially applied solve would be worse than none.
    #[test]
    fn a_solve_refuses_a_field_it_cannot_read() {
        let mut terrain = TerrainSpec::new(UVec2::new(8, 8))
            .with_field(Field::new("height").with_op(NodeOp::Constant(0.5)))
            .with_field(Field::new("coarse").with_shift(2));
        terrain.bake_in_place().unwrap();

        assert!(matches!(
            terrain.solve_water(&WaterSpec::new("absent")),
            Err(WaterError::UnknownHeightField(_))
        ));
        assert!(matches!(
            terrain.solve_water(&WaterSpec::new("coarse")),
            Err(WaterError::CoarseHeight(_, 2))
        ));
        assert!(matches!(
            terrain.solve_water(&WaterSpec::default().with_moisture("absent")),
            Err(WaterError::UnknownMoistureField(_))
        ));
        assert!(terrain.water().is_none());
    }

    // An unbaked field has an empty raster rather than a wrong one, so without this
    // check the solve would quietly return an empty state instead of an error.
    #[test]
    fn a_solve_of_an_unbaked_document_is_refused() {
        let mut terrain = TerrainSpec::new(UVec2::new(8, 8))
            .with_field(Field::new("height").with_op(NodeOp::Constant(0.5)));
        assert!(matches!(
            terrain.solve_water(&WaterSpec::default()),
            Err(WaterError::UnbakedHeight(_))
        ));

        let mut empty = TerrainSpec::new(UVec2::ZERO);
        assert!(matches!(
            empty.solve_water(&WaterSpec::default()),
            Err(WaterError::ZeroSize(0, 0))
        ));
    }

    // Every reader has to answer on an empty state, because that is what a document
    // holds between an invalidation and the next solve.
    #[test]
    fn an_empty_height_raster_solves_to_nothing() {
        let state = WaterState::solve(&Raster::default(), &[], 4);
        assert!(state.is_empty());
        assert_eq!(state.lakes(), 0);
        assert!(state.flow_vector(0, 0).is_none());
        assert_eq!(state.accumulation(0, 0), 0.0);
    }

    // Not a pass/fail test: it prints what one solve of a full-size document costs,
    // which is the figure that decides whether the editor can re-solve on an edit or
    // has to defer it. Ignored because it is a measurement.
    #[test]
    #[ignore]
    fn a_full_size_document_measures_what_a_water_solve_costs() {
        let size = UVec2::new(4096, 4096);
        let height = raster_from(size, |x, y| {
            let u = x as f32 * 0.01;
            let v = y as f32 * 0.01;
            (u.sin() + v.cos()) * 0.25
        });
        let start = std::time::Instant::now();
        let state = WaterState::solve(&height, &unit_weights(size), 64);
        let elapsed = start.elapsed();
        println!(
            "solved {}x{} in {:?}, {} lakes",
            size.x,
            size.y,
            elapsed,
            state.lakes()
        );
    }
}
