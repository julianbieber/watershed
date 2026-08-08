use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::f32::consts::SQRT_2;

use glam::{UVec2, Vec2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::bake::Terrain;
use crate::field::FieldId;
use crate::raster::Raster;

#[derive(Debug, Error)]
pub enum WaterError {
    #[error("terrain size has a zero component: {0} by {1}")]
    ZeroSize(u32, u32),
    #[error("height field `{0}` is not in the document")]
    UnknownHeightField(String),
    #[error("moisture field `{0}` is not in the document")]
    UnknownMoistureField(String),
    #[error("height field `{0}` is at shift {1}; the solve reads one texel per cell")]
    CoarseHeight(String, u8),
    #[error("height field `{0}` has not been baked at the document's size")]
    UnbakedHeight(String),
}

/// TODO(jb-doc): what a flow direction code is, why zero means no outflow, and which
/// neighbour each of the eight remaining codes names.
pub const SINK: u8 = 0;

/// TODO(jb-comment): why the neighbour order is fixed and written out rather than
/// generated, and what a change to it would do to a solved document.
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

/// TODO(jb-doc): the range this scale has to cover, the relative precision it leaves, and
/// why the quantization is logarithmic rather than linear.
const ACCUM_QUANT: f32 = 3900.0;

/// TODO(jb-doc): what a caller chooses here, and why the height field is named rather
/// than assumed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WaterSpec {
    pub height: FieldId,
    pub moisture: Option<FieldId>,
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
    pub fn new(height: impl Into<FieldId>) -> Self {
        Self {
            height: height.into(),
            moisture: None,
            lake_min_cells: 64,
        }
    }

    pub fn with_moisture(mut self, moisture: impl Into<FieldId>) -> Self {
        self.moisture = Some(moisture.into());
        self
    }

    pub fn with_lake_min_cells(mut self, cells: u32) -> Self {
        self.lake_min_cells = cells;
        self
    }
}

/// TODO(jb-doc): what the solve produces — standing water, where each cell drains, how
/// much reaches it, and which lake it belongs to — and what it deliberately does not
/// store.
///
/// TODO(jb-comment): why no flow vector is stored per cell.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WaterState {
    level: Raster<f32>,
    flow_dir: Raster<u8>,
    flow_accum: Raster<u16>,
    lake_id: Raster<u32>,
    lakes: u32,
}

impl WaterState {
    /// TODO(jb-doc): the contract this takes — a height raster at one texel per cell and
    /// one weight per cell in the same order — and why the whole grid is solved at once.
    ///
    /// TODO(jb-comment): why this pass is single threaded where the bake is not.
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
        let level: Vec<f32> = filled
            .iter()
            .zip(source)
            .map(|(surface, ground)| (surface - ground).max(0.0))
            .collect();

        Self {
            level: Raster::from_vec(size, level).unwrap_or_default(),
            flow_dir: Raster::from_vec(size, flow_dir).unwrap_or_default(),
            flow_accum: Raster::from_vec(size, flow_accum).unwrap_or_default(),
            lake_id: Raster::from_vec(size, lake_id).unwrap_or_default(),
            lakes,
        }
    }

    /// TODO(jb-comment): why the loader rebuilds a state from its rasters rather than
    /// re-solving, and what it is trusted to have checked first.
    pub(crate) fn from_parts(
        level: Raster<f32>,
        flow_dir: Raster<u8>,
        flow_accum: Raster<u16>,
        lake_id: Raster<u32>,
        lakes: u32,
    ) -> Self {
        Self {
            level,
            flow_dir,
            flow_accum,
            lake_id,
            lakes,
        }
    }

    pub fn size(&self) -> UVec2 {
        self.level.size()
    }

    pub fn is_empty(&self) -> bool {
        self.level.is_empty()
    }

    pub fn level(&self) -> &Raster<f32> {
        &self.level
    }

    pub fn flow_dir(&self) -> &Raster<u8> {
        &self.flow_dir
    }

    pub fn flow_accum(&self) -> &Raster<u16> {
        &self.flow_accum
    }

    pub fn lake_id(&self) -> &Raster<u32> {
        &self.lake_id
    }

    pub fn lakes(&self) -> u32 {
        self.lakes
    }

    /// TODO(jb-doc): what counts as water here, and why it is the fill depth rather than
    /// the lake labelling.
    pub fn is_water(&self, x: u32, y: u32) -> bool {
        self.level.get(x, y).is_some_and(|depth| *depth > 0.0)
    }

    /// TODO(jb-doc): the unit this answers in, and the precision the stored quantization
    /// leaves it with.
    pub fn accumulation(&self, x: u32, y: u32) -> f32 {
        self.flow_accum
            .get(x, y)
            .map(|code| dequantize_accumulation(*code))
            .unwrap_or(0.0)
    }

    pub fn channel(&self, x: u32, y: u32, threshold: f32) -> bool {
        self.accumulation(x, y) >= threshold
    }

    /// TODO(jb-doc): why this is derived on read, and what `None` means.
    pub fn flow_vector(&self, x: u32, y: u32) -> Option<Vec2> {
        let code = *self.flow_dir.get(x, y)?;
        let (dx, dy) = *D8.get(direction_index(code)?)?;
        Some(Vec2::new(dx as f32, dy as f32).normalize_or_zero())
    }

    /// TODO(jb-doc): the coordinate this answers with, and why a sink has none.
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

impl Terrain {
    pub fn water(&self) -> Option<&WaterState> {
        self.water.as_ref()
    }

    /// TODO(jb-comment): why this drops the spec as well as the state, and what a saved
    /// document would otherwise do on the next load.
    pub fn clear_water(&mut self) {
        self.water = None;
        self.water_spec = None;
    }

    /// TODO(jb-doc): what this reads, what it replaces, and why a coarse height field is
    /// refused rather than resampled.
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

/// TODO(jb-comment): why two surfaces come out of one traversal, and what each is for.
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

/// TODO(jb-comment): why a step is chosen on the filled surface and only broken on the
/// epsilon one, and what choosing on the epsilon surface alone did.
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

/// TODO(jb-comment): why a component is grown four-connected, and why a component under
/// the threshold keeps its depth but loses its label.
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

fn quantize_accumulation(value: f32) -> u16 {
    let code = value.max(0.0).ln_1p() * ACCUM_QUANT;
    code.round().clamp(0.0, u16::MAX as f32) as u16
}

/// TODO(jb-doc): the inverse of the stored quantization, and the error it carries.
pub fn dequantize_accumulation(code: u16) -> f32 {
    (code as f32 / ACCUM_QUANT).exp_m1()
}

/// TODO(jb-comment): why the priority queue orders on an integer key rather than on the
/// float it stands for, and why the cell index is part of the order.
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
    use crate::field::Field;
    use crate::layer::{Layer, LayerOp};

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

    #[test]
    fn a_bowl_fills_to_exactly_its_rim() {
        let size = UVec2::new(11, 11);
        let height = bowl(size);
        let state = solved(&height);

        for y in 0..size.y {
            for x in 0..size.x {
                let ground = *height.get(x, y).unwrap();
                let surface = ground + state.level().get(x, y).unwrap();
                if ground < 1.0 && x > 1 && y > 1 && x + 2 < size.x && y + 2 < size.y {
                    assert_eq!(surface, 1.0, "cell {x},{y} stands at {surface}");
                    assert!(state.is_water(x, y));
                } else {
                    assert_eq!(*state.level().get(x, y).unwrap(), 0.0);
                }
            }
        }
        assert_eq!(state.lakes(), 1);
        assert_eq!(*state.lake_id().get(5, 5).unwrap(), 1);
        assert_eq!(*state.lake_id().get(0, 0).unwrap(), 0);
    }

    #[test]
    fn a_lake_under_the_minimum_keeps_its_depth_and_loses_its_label() {
        let size = UVec2::new(11, 11);
        let height = bowl(size);
        let state = WaterState::solve(&height, &unit_weights(size), 10_000);
        assert!(state.is_water(5, 5));
        assert_eq!(state.lakes(), 0);
        assert_eq!(*state.lake_id().get(5, 5).unwrap(), 0);
    }

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

    #[test]
    fn a_document_solves_water_over_the_field_it_names() {
        let mut terrain = Terrain::new(UVec2::new(16, 16))
            .with_field(Field::new("height").with_layer(Layer::new(LayerOp::Constant(0.5))));
        terrain.bake().unwrap();
        assert!(terrain.water().is_none());
        terrain.solve_water(&WaterSpec::default()).unwrap();
        let water = terrain.water().unwrap();
        assert_eq!(water.size(), UVec2::new(16, 16));
        terrain.clear_water();
        assert!(terrain.water().is_none());
    }

    #[test]
    fn a_named_moisture_field_weights_what_the_sinks_deliver() {
        let size = UVec2::new(24, 24);
        let mut terrain = Terrain::new(size)
            .with_field(Field::new("height").with_layer(Layer::new(LayerOp::Constant(0.5))))
            .with_field(
                Field::new("moisture")
                    .with_layer(Layer::new(LayerOp::Constant(0.25)))
                    .with_range((0.0, 1.0)),
            );
        terrain.bake().unwrap();
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

    #[test]
    fn a_solve_refuses_a_field_it_cannot_read() {
        let mut terrain = Terrain::new(UVec2::new(8, 8))
            .with_field(Field::new("height").with_layer(Layer::new(LayerOp::Constant(0.5))))
            .with_field(Field::new("coarse").with_shift(2));
        terrain.bake().unwrap();

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

    #[test]
    fn a_solve_of_an_unbaked_document_is_refused() {
        let mut terrain = Terrain::new(UVec2::new(8, 8))
            .with_field(Field::new("height").with_layer(Layer::new(LayerOp::Constant(0.5))));
        assert!(matches!(
            terrain.solve_water(&WaterSpec::default()),
            Err(WaterError::UnbakedHeight(_))
        ));

        let mut empty = Terrain::new(UVec2::ZERO);
        assert!(matches!(
            empty.solve_water(&WaterSpec::default()),
            Err(WaterError::ZeroSize(0, 0))
        ));
    }

    #[test]
    fn an_empty_height_raster_solves_to_nothing() {
        let state = WaterState::solve(&Raster::default(), &[], 4);
        assert!(state.is_empty());
        assert_eq!(state.lakes(), 0);
        assert!(state.flow_vector(0, 0).is_none());
        assert_eq!(state.accumulation(0, 0), 0.0);
    }

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
