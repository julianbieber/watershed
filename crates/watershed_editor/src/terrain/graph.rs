//! What a field's value is built out of: the nodes, the edges between them, and the
//! one node the field is read from.

use serde::{Deserialize, Serialize};

use watershed::field::FieldId;
use watershed::raster::Raster;

use crate::terrain::noise::NoiseSpec;
use crate::terrain::regions::{RegionOutput, RegionSpec};
use crate::terrain::shader::ShaderLayer;

/// How far apart cards are laid out when nothing says where one goes.
///
/// Document data rather than a canvas measurement: a position is saved, so the spacing
/// a graph is built with has to be the same wherever it was built from.
pub const NODE_STEP: [f32; 2] = [260.0, 150.0];

/// Names one node for the life of a document.
///
/// Assigned from [`FieldGraph::next_id`] and never reused, so an edge, a control
/// path and the canvas all name the same node across every edit — nothing is
/// renumbered when a node is removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId(pub u32);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// A linear rescale from one interval onto another, clamped at both ends of the
/// input.
///
/// This is what makes one field's values usable as a weight: they span whatever the
/// field's own range is, and a [`NodeOp::Lerp`] weight has to be in `0.0..=1.0`, so
/// the reader states which band it cares about rather than the field being changed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Remap {
    /// The input band. Values outside it clamp to the nearest end.
    pub from: (f32, f32),
    /// The output band. May run backwards, which inverts the mapping.
    pub to: (f32, f32),
}

impl Remap {
    /// `0.0..=1.0` onto itself — still clamps, so it is not a no-op for a value
    /// outside the unit interval.
    pub const IDENTITY: Self = Self {
        from: (0.0, 1.0),
        to: (0.0, 1.0),
    };

    /// Takes both bands as given. Neither is validated or sorted; `from` running
    /// backwards inverts the mapping just as `to` does.
    pub fn new(from: (f32, f32), to: (f32, f32)) -> Self {
        Self { from, to }
    }

    /// `value` rescaled from `from` onto `to`, clamped to `to`'s ends.
    ///
    /// A `from` band of zero width is not an error and never divides by zero: every
    /// input maps to `to.0`.
    pub fn apply(&self, value: f32) -> f32 {
        let span = self.from.1 - self.from.0;
        let t = if span == 0.0 {
            0.0
        } else {
            ((value - self.from.0) / span).clamp(0.0, 1.0)
        };
        let (lo, hi) = (self.to.0, self.to.1);
        lo + (hi - lo) * t
    }
}

impl Default for Remap {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// How a slope reads the values under it.
///
/// The two answer different questions about the same ground, and which one a caller
/// wants depends on what the slope is *for* rather than on accuracy. A gradient is the
/// better estimate of the surface's true steepness; the steepest axis is what a thing
/// travelling on the lattice — water, a walker, a wagon — actually has to climb, and it
/// reads one sample further ahead rather than one either side, so it is a forward
/// question rather than a symmetric one.
///
/// The difference that matters in practice is at a crest. A central difference
/// takes samples either side of the position, and on a ridge line those two are
/// close to equal, so `Gradient` reads near zero along the ridge itself; a field
/// thresholded just above zero therefore shows a hairline gap running along every
/// crest. A forward difference never cancels that way, so `SteepestAxis` has no
/// such gap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SlopeMode {
    /// A central difference on each axis, taken as a euclidean magnitude.
    #[default]
    Gradient,
    /// A forward difference on each axis, taken as the larger of the two.
    SteepestAxis,
}

/// How a [`NodeOp::Binary`] node combines its two inputs.
///
/// What a layer's blend mode was, less `Replace`: in a graph, replacing what is
/// under a value is done by connecting the other edge instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Binary {
    /// Sum. The default, so an unconfigured node accumulates.
    #[default]
    Add,
    /// Product.
    Mul,
    /// The larger of the two.
    Max,
    /// The smaller of the two.
    Min,
}

impl Binary {
    /// The combined value. Not clamped — the field's range is applied once, at the
    /// end of the bake.
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            Binary::Add => a + b,
            Binary::Mul => a * b,
            Binary::Max => a.max(b),
            Binary::Min => a.min(b),
        }
    }
}

/// One point of a [`Curve`]: an input value and what it maps to.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    /// Where on the input axis this point sits.
    pub input: f32,
    /// What the curve reads at that input.
    pub output: f32,
}

/// An arbitrary mapping from one value to another, through a list of control points.
///
/// [`Remap`] for a mapping that is not a straight line. The points need not be
/// sorted or distinct: they are sorted when the curve is evaluated, and where two
/// share an input the later one wins.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Curve {
    /// The control points, in any order.
    pub points: Vec<CurvePoint>,
}

impl Curve {
    /// Takes the points as given.
    pub fn new(points: Vec<CurvePoint>) -> Self {
        Self { points }
    }

    /// `value` mapped through the curve, clamped to the first and last point.
    ///
    /// A curve with no points passes its input through unchanged, and one with a
    /// single point reads that point's output everywhere — so a curve being built
    /// in the inspector is never a hole in the graph.
    pub fn apply(&self, value: f32) -> f32 {
        let sorted = self.sorted();
        match sorted.as_slice() {
            [] => value,
            [only] => only.output,
            points => {
                let first = points[0];
                let last = points[points.len() - 1];
                if value <= first.input {
                    return first.output;
                }
                if value >= last.input {
                    return last.output;
                }
                let upper = points
                    .iter()
                    .position(|point| point.input >= value)
                    .unwrap_or(points.len() - 1)
                    .max(1);
                let (a, b) = (points[upper - 1], points[upper]);
                let span = b.input - a.input;
                if span == 0.0 {
                    return b.output;
                }
                let t = (value - a.input) / span;
                a.output + (b.output - a.output) * t
            }
        }
    }

    fn sorted(&self) -> Vec<CurvePoint> {
        let mut points = self.points.clone();
        points.sort_by(|a, b| a.input.total_cmp(&b.input));
        points
    }
}

/// What a node produces, given whatever is wired into it.
///
/// Every op reads its inputs at the texel it is writing, except [`NodeOp::Slope`],
/// which reads a neighbourhood — that is what makes it the only op to widen a
/// rectangle re-bake.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NodeOp {
    /// The same value everywhere.
    Constant(f32),
    /// Procedural noise evaluated at the position; see [`NoiseSpec`].
    Noise(NoiseSpec),
    /// A painted raster, stretched over the whole document, and what an editor brush
    /// writes into.
    ///
    /// Bytes, not floats: a stroke is read as its byte over 255 and so lives in
    /// `0.0..=1.0`, and any other band is reached by the [`NodeOp::Scale`] or
    /// [`NodeOp::Remap`] under it.
    Paint(Raster<u8>),
    /// A raster produced outside the editor — imported, generated, or handed in by
    /// the host application. A brush stroke will not write into it.
    External(Raster<f32>),
    /// A raster a WGSL shader produced, at the field's own resolution.
    ///
    /// The shader reads no field, so this op contributes no dependency and widens no
    /// re-bake. The values are not serialized: a loaded document reads the node as
    /// `0.0` until it has been dispatched again.
    Shader(ShaderLayer),
    /// A value derived from the region tiling at the position.
    ///
    /// Holds its whole input in the [`RegionSpec`], so it can be evaluated without
    /// anything else in the document having been baked.
    Regions {
        /// The tiling and its per-region table.
        spec: RegionSpec,
        /// Which column, or which identifier, of the tiling to emit.
        output: RegionOutput,
    },
    /// Another field's value at the position, through that field's own sampling — so
    /// a categorical field is read to the nearest texel, not interpolated.
    ///
    /// The only op that reads another field.
    FieldRef(FieldId),
    /// The steepness of its input.
    ///
    /// Reads a neighbourhood rather than a single texel, so it widens the rectangle
    /// its input must be evaluated over, and the widening sums along a chain.
    Slope {
        /// How far apart, in document cells, the samples are taken. The magnitude is
        /// used, and a zero is raised to `f32::EPSILON` rather than dividing by zero.
        sample_tiles: f32,
        /// Defaults to [`SlopeMode::Gradient`].
        #[serde(default)]
        mode: SlopeMode,
    },
    /// Combines its two inputs; see [`Binary`].
    Binary(Binary),
    /// Interpolates from its first input to its second by its third, clamped to
    /// `0.0..=1.0`. What a mask was.
    Lerp,
    /// Multiplies its input by a constant. What a layer's amplitude was.
    Scale(f32),
    /// Rescales one band of its input onto another; see [`Remap`].
    Remap(Remap),
    /// Maps its input through a [`Curve`].
    Curve(Curve),
}

impl NodeOp {
    /// How many inputs this op reads. An op is always stored with exactly this many
    /// input pins, connected or not.
    pub fn arity(&self) -> usize {
        match self {
            NodeOp::Constant(_)
            | NodeOp::Noise(_)
            | NodeOp::Paint(_)
            | NodeOp::External(_)
            | NodeOp::Shader(_)
            | NodeOp::Regions { .. }
            | NodeOp::FieldRef(_) => 0,
            NodeOp::Slope { .. } | NodeOp::Scale(_) | NodeOp::Remap(_) | NodeOp::Curve(_) => 1,
            NodeOp::Binary(_) => 2,
            NodeOp::Lerp => 3,
        }
    }

    /// The field this op reads, if any. Feeds bake ordering and cycle detection.
    pub fn dependency(&self) -> Option<&FieldId> {
        match self {
            NodeOp::FieldRef(id) => Some(id),
            _ => None,
        }
    }

    /// Whether this op reads a neighbourhood of its input rather than one texel, and
    /// so widens the rectangle that input must be evaluated over.
    pub fn widens(&self) -> bool {
        matches!(self, NodeOp::Slope { .. })
    }
}

/// One node of a field's graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    /// Names this node for the life of the document.
    pub id: NodeId,
    /// What a person called it, if anything. Never `n<digits>`, which would shadow
    /// the identifier form of a control path.
    pub name: Option<String>,
    /// What the node produces.
    pub op: NodeOp,
    /// One entry per [`NodeOp::arity`], in pin order. `None` is an unconnected pin,
    /// which reads `0.0`.
    pub inputs: Vec<Option<NodeId>>,
    /// A bypassed node passes its first input through instead of evaluating, and
    /// contributes no dependency.
    pub bypassed: bool,
    /// Where the node's card sits on the canvas. Authoring data: it is written to the
    /// document but reaches no bake.
    pub position: [f32; 2],
}

impl GraphNode {
    /// A node of `op` at `position`, with every pin unconnected.
    pub fn new(id: NodeId, op: NodeOp, position: [f32; 2]) -> Self {
        let inputs = vec![None; op.arity()];
        Self {
            id,
            name: None,
            op,
            inputs,
            bypassed: false,
            position,
        }
    }

    /// The nodes wired into this one, in pin order, skipping unconnected pins.
    pub fn sources(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.inputs.iter().flatten().copied()
    }
}

/// Why an edit to a graph was refused. The graph is left exactly as it was.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GraphError {
    /// No node in this graph carries that id.
    #[error("this field has no node {0:?}")]
    UnknownNode(NodeId),
    /// The pin index is past what the target op reads.
    #[error("node {node:?} has {arity} inputs, so there is no pin {pin}")]
    PinOutOfRange {
        /// The node written to.
        node: NodeId,
        /// The pin asked for.
        pin: usize,
        /// How many the node actually has.
        arity: usize,
    },
    /// The edge would make the graph feed itself.
    #[error("connecting {from:?} to {to:?} would close a cycle")]
    WouldCycle {
        /// The node the edge comes from.
        from: NodeId,
        /// The node it would be written into.
        to: NodeId,
    },
    /// Another node in this graph already carries that name.
    #[error("this field already has a node named `{0}`")]
    DuplicateName(String),
    /// The name is the identifier form of a control path and would shadow a node id.
    #[error("`{0}` is the path form of a node id and cannot be a name")]
    ReservedName(String),
}

/// The nodes of one field, the edges between them, and the node the field's value is
/// read from.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct FieldGraph {
    /// Every node, in no meaningful order — evaluation order is derived, not stored.
    pub nodes: Vec<GraphNode>,
    /// The node the field's value is read from. A graph with none bakes zero.
    pub output: Option<NodeId>,
    /// The id the next node created will take. Only ever increases.
    pub next_id: u32,
}

impl FieldGraph {
    /// An empty graph: no nodes, no output.
    pub fn new() -> Self {
        Self::default()
    }

    /// The node with that id.
    pub fn node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.iter().find(|node| node.id == id)
    }

    /// The node with that id, to be modified.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.iter_mut().find(|node| node.id == id)
    }

    /// A spot no card is sitting on, for a node added without one.
    ///
    /// Laid out in rows below the origin, so a graph built entirely through control
    /// verbs comes out readable rather than as a stack of cards on one another.
    pub fn free_position(&self) -> [f32; 2] {
        let clear = |at: [f32; 2]| {
            !self.nodes.iter().any(|node| {
                (node.position[0] - at[0]).abs() < NODE_STEP[0] * 0.5
                    && (node.position[1] - at[1]).abs() < NODE_STEP[1] * 0.5
            })
        };
        for row in 0..64 {
            for column in 0..8 {
                let at = [column as f32 * NODE_STEP[0], -(row as f32) * NODE_STEP[1]];
                if clear(at) {
                    return at;
                }
            }
        }
        [0.0, 0.0]
    }

    /// Adds an unconnected node of `op` and returns the id it was given.
    ///
    /// If the graph had no output node, the new node becomes it — the first node
    /// added to an empty graph is what that field is, and leaving it outputless
    /// would make the field bake zero until it was wired by hand.
    pub fn add_node(&mut self, op: NodeOp, position: [f32; 2]) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes.push(GraphNode::new(id, op, position));
        if self.output.is_none() {
            self.output = Some(id);
        }
        id
    }

    /// Adds a node of `op` with `sources` wired into its pins in order.
    ///
    /// The shorthand a graph is built with in one expression. Every edge it makes is
    /// one [`FieldGraph::connect`], so a source that would close a cycle or overrun
    /// the op's arity panics rather than being quietly dropped — which is what a
    /// caller building a graph from a literal wants to hear.
    pub fn node_with(&mut self, op: NodeOp, sources: &[NodeId]) -> NodeId {
        let id = self.add_node(op, [0.0, 0.0]);
        for (pin, &source) in sources.iter().enumerate() {
            self.connect(source, id, pin)
                .expect("a graph built in one expression wires only pins its ops carry");
        }
        id
    }

    /// Takes a node out along with every edge touching it.
    ///
    /// A node with exactly one connected input and exactly one reader is spliced out:
    /// the reader is reconnected to that input, so pulling a node out of a chain does
    /// not break the chain. In every other case each reader's pin is left
    /// unconnected. If the node was the output, the graph is left with none.
    pub fn remove_node(&mut self, id: NodeId) -> Result<(), GraphError> {
        let node = self.node(id).ok_or(GraphError::UnknownNode(id))?;
        let only_source = match node.inputs.iter().flatten().collect::<Vec<_>>().as_slice() {
            [single] => Some(**single),
            _ => None,
        };
        let readers: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|other| other.sources().any(|source| source == id))
            .map(|other| other.id)
            .collect();
        let splice = match (only_source, readers.as_slice()) {
            (Some(source), [_one]) => Some(source),
            _ => None,
        };
        for other in &mut self.nodes {
            for pin in &mut other.inputs {
                if *pin == Some(id) {
                    *pin = splice;
                }
            }
        }
        self.nodes.retain(|node| node.id != id);
        if self.output == Some(id) {
            self.output = splice;
        }
        Ok(())
    }

    /// Writes one input pin, replacing whatever was on it.
    ///
    /// Refused, leaving the graph unchanged, if either node is unknown, if the pin is
    /// past the target's arity, or if the edge would close a cycle. The cycle check
    /// reads every edge, bypassed or not, so un-bypassing a node can never produce a
    /// graph that will not plan.
    pub fn connect(&mut self, from: NodeId, to: NodeId, pin: usize) -> Result<(), GraphError> {
        if self.node(from).is_none() {
            return Err(GraphError::UnknownNode(from));
        }
        let target = self.node(to).ok_or(GraphError::UnknownNode(to))?;
        let arity = target.inputs.len();
        if pin >= arity {
            return Err(GraphError::PinOutOfRange {
                node: to,
                pin,
                arity,
            });
        }
        if from == to || self.reaches(from, to) {
            return Err(GraphError::WouldCycle { from, to });
        }
        self.node_mut(to).expect("checked above").inputs[pin] = Some(from);
        Ok(())
    }

    /// Clears one input pin, which then reads `0.0`.
    pub fn disconnect(&mut self, to: NodeId, pin: usize) -> Result<(), GraphError> {
        let target = self.node_mut(to).ok_or(GraphError::UnknownNode(to))?;
        let arity = target.inputs.len();
        if pin >= arity {
            return Err(GraphError::PinOutOfRange {
                node: to,
                pin,
                arity,
            });
        }
        target.inputs[pin] = None;
        Ok(())
    }

    /// Sets whether a node is bypassed.
    pub fn set_bypassed(&mut self, id: NodeId, bypassed: bool) -> Result<(), GraphError> {
        self.node_mut(id)
            .ok_or(GraphError::UnknownNode(id))?
            .bypassed = bypassed;
        Ok(())
    }

    /// Writes a node's canvas position and nothing else.
    pub fn place(&mut self, id: NodeId, position: [f32; 2]) -> Result<(), GraphError> {
        self.node_mut(id)
            .ok_or(GraphError::UnknownNode(id))?
            .position = position;
        Ok(())
    }

    /// Names a node, or clears its name with `None`.
    ///
    /// Refused if another node in this graph carries the name, or if the name is the
    /// `n<digits>` form a control path uses for an id, which would otherwise let a
    /// name shadow a different node.
    pub fn rename(&mut self, id: NodeId, name: Option<String>) -> Result<(), GraphError> {
        if self.node(id).is_none() {
            return Err(GraphError::UnknownNode(id));
        }
        if let Some(name) = &name {
            if is_id_path(name) {
                return Err(GraphError::ReservedName(name.clone()));
            }
            if self
                .nodes
                .iter()
                .any(|node| node.id != id && node.name.as_deref() == Some(name.as_str()))
            {
                return Err(GraphError::DuplicateName(name.clone()));
            }
        }
        self.node_mut(id).expect("checked above").name = name;
        Ok(())
    }

    /// Names the node the field's value is read from.
    pub fn set_output(&mut self, id: Option<NodeId>) -> Result<(), GraphError> {
        if let Some(id) = id
            && self.node(id).is_none()
        {
            return Err(GraphError::UnknownNode(id));
        }
        self.output = id;
        Ok(())
    }

    /// The node the field's value actually comes from, following bypassed nodes to
    /// what they pass through.
    ///
    /// This, not [`FieldGraph::output`], is what decides whether a field is
    /// categorical: a bypassed node at the output emits its input's values, not its
    /// own.
    pub fn effective_output(&self) -> Option<NodeId> {
        let mut seen = Vec::new();
        let mut current = self.output?;
        loop {
            if seen.contains(&current) {
                return None;
            }
            seen.push(current);
            let node = self.node(current)?;
            if !node.bypassed {
                return Some(current);
            }
            current = *node.inputs.first()?.as_ref()?;
        }
    }

    /// The inputs a node actually contributes, which for a bypassed node is only the
    /// first pin it passes through.
    pub fn effective_sources(&self, id: NodeId) -> Vec<NodeId> {
        let Some(node) = self.node(id) else {
            return Vec::new();
        };
        if node.bypassed {
            node.inputs
                .first()
                .and_then(|pin| *pin)
                .into_iter()
                .collect()
        } else {
            node.sources().collect()
        }
    }

    /// Every node reachable from the output, in an order that puts each node after
    /// everything it reads.
    ///
    /// A bypassed node is walked through to the input it passes on, so a node only
    /// that node reached is not in the order and is evaluated never. A graph with no
    /// output yields an empty order rather than an error — it bakes zero.
    ///
    /// # Errors
    ///
    /// [`GraphError::WouldCycle`] naming the node the walk re-entered. An edit made
    /// through [`FieldGraph::connect`] cannot produce one; a loaded or hand-edited
    /// document can.
    pub fn evaluation_order(&self) -> Result<Vec<NodeId>, GraphError> {
        let Some(output) = self.output else {
            return Ok(Vec::new());
        };
        if self.node(output).is_none() {
            return Err(GraphError::UnknownNode(output));
        }
        let mut order = Vec::new();
        let mut open = Vec::new();
        let mut done = Vec::new();
        let mut stack = vec![(output, false)];
        while let Some((id, leaving)) = stack.pop() {
            if leaving {
                open.retain(|other| *other != id);
                done.push(id);
                order.push(id);
                continue;
            }
            if done.contains(&id) {
                continue;
            }
            if open.contains(&id) {
                return Err(GraphError::WouldCycle { from: id, to: id });
            }
            open.push(id);
            stack.push((id, true));
            for source in self.effective_sources(id) {
                stack.push((source, false));
            }
        }
        Ok(order)
    }

    /// The fields this graph reads, in evaluation order and with duplicates kept.
    ///
    /// Only [`NodeOp::FieldRef`] nodes the walk actually reached, so an unreachable
    /// or bypassed reference contributes no bake-order dependency and cannot make a
    /// cycle between fields.
    pub fn dependencies(&self) -> Vec<&FieldId> {
        self.evaluation_order()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|id| self.node(id)?.op.dependency())
            .collect()
    }

    /// Whether `target` can be reached from `start` by walking inputs — that is,
    /// whether `start` already depends on `target`.
    ///
    /// Reads every edge regardless of bypass, because bypass is not a property of the
    /// graph's shape.
    pub fn reaches(&self, start: NodeId, target: NodeId) -> bool {
        let mut stack = vec![start];
        let mut seen = Vec::new();
        while let Some(id) = stack.pop() {
            if id == target {
                return true;
            }
            if seen.contains(&id) {
                continue;
            }
            seen.push(id);
            if let Some(node) = self.node(id) {
                stack.extend(node.sources());
            }
        }
        false
    }
}

fn is_id_path(name: &str) -> bool {
    match name.strip_prefix('n') {
        Some(digits) => !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> FieldGraph {
        FieldGraph::new()
    }

    // `Remap::IDENTITY` is what an unconfigured remap carries, so a rounding or offset
    // error in `apply` would perturb every graph nobody has configured.
    #[test]
    fn an_identity_remap_returns_the_value_it_was_given() {
        for value in [0.0f32, 0.25, 0.5, 1.0] {
            assert_eq!(Remap::IDENTITY.apply(value), value);
        }
    }

    // Order matters: scaling first and clamping after would let a value outside the
    // band leave the output band. Pins the clamp on the input side.
    #[test]
    fn a_remap_clamps_to_its_input_band_before_it_scales() {
        let remap = Remap::new((0.4, 0.6), (0.0, 1.0));
        assert_eq!(remap.apply(0.0), 0.0);
        assert_eq!(remap.apply(0.4), 0.0);
        assert!((remap.apply(0.5) - 0.5).abs() < 1e-6);
        assert_eq!(remap.apply(0.6), 1.0);
        assert_eq!(remap.apply(9.0), 1.0);
    }

    // Inverting a mapping is done by writing `to` backwards rather than by a flag, so
    // nothing may sort or normalise the output band.
    #[test]
    fn a_remap_may_run_backwards() {
        let remap = Remap::new((0.0, 1.0), (1.0, 0.0));
        assert_eq!(remap.apply(0.0), 1.0);
        assert_eq!(remap.apply(1.0), 0.0);
    }

    // A zero-width band is what dragging both ends of a range widget together produces,
    // so it reaches `apply` from the editor; the guard keeps a NaN out of every
    // downstream texel.
    #[test]
    fn a_remap_over_a_zero_width_band_is_its_low_end_rather_than_a_division_by_zero() {
        let remap = Remap::new((0.5, 0.5), (0.2, 0.9));
        assert_eq!(remap.apply(0.5), 0.2);
        assert!(remap.apply(0.9).is_finite());
    }

    fn point(input: f32, output: f32) -> CurvePoint {
        CurvePoint { input, output }
    }

    // An id names a node for the life of the document, so removing one must not free
    // its id for the next node: an edge or a control path would silently retarget.
    #[test]
    fn an_id_is_never_reused_after_the_node_is_removed() {
        let mut graph = graph();
        let first = graph.add_node(NodeOp::Constant(1.0), [0.0, 0.0]);
        graph.remove_node(first).unwrap();
        let second = graph.add_node(NodeOp::Constant(2.0), [0.0, 0.0]);
        assert_ne!(first, second);
    }

    // A field is what its output node reads, and a graph whose first node is not the
    // output would bake zero until wired by hand.
    #[test]
    fn the_first_node_added_becomes_the_output() {
        let mut graph = graph();
        let id = graph.add_node(NodeOp::Constant(1.0), [0.0, 0.0]);
        assert_eq!(graph.output, Some(id));
    }

    // A node feeding itself is the shortest cycle and the easiest to draw on a canvas.
    #[test]
    fn a_node_cannot_be_connected_to_itself() {
        let mut graph = graph();
        let id = graph.add_node(NodeOp::Scale(2.0), [0.0, 0.0]);
        assert_eq!(
            graph.connect(id, id, 0),
            Err(GraphError::WouldCycle { from: id, to: id })
        );
    }

    // The graph must be acyclic at all times, so the edge that would close a cycle is
    // refused rather than left for the plan to reject.
    #[test]
    fn an_edge_closing_a_cycle_is_refused_and_changes_nothing() {
        let mut graph = graph();
        let a = graph.add_node(NodeOp::Scale(2.0), [0.0, 0.0]);
        let b = graph.add_node(NodeOp::Scale(2.0), [0.0, 0.0]);
        graph.connect(a, b, 0).unwrap();
        let before = graph.clone();
        assert_eq!(
            graph.connect(b, a, 0),
            Err(GraphError::WouldCycle { from: b, to: a })
        );
        assert_eq!(graph, before);
    }

    // Bypass is not part of the graph's shape: if the cycle check read the
    // bypass-reduced graph, un-bypassing a node would produce a graph that cannot
    // plan, with no verb left to refuse it.
    #[test]
    fn the_cycle_check_reads_bypassed_edges_too() {
        let mut graph = graph();
        let a = graph.add_node(NodeOp::Scale(2.0), [0.0, 0.0]);
        let b = graph.add_node(NodeOp::Scale(2.0), [0.0, 0.0]);
        graph.connect(a, b, 0).unwrap();
        graph.set_bypassed(b, true).unwrap();
        assert_eq!(
            graph.connect(b, a, 0),
            Err(GraphError::WouldCycle { from: b, to: a })
        );
    }

    // Pulling one node out of a chain should not break the chain.
    #[test]
    fn removing_a_node_with_one_input_and_one_reader_splices_it_out() {
        let mut graph = graph();
        let source = graph.add_node(NodeOp::Constant(1.0), [0.0, 0.0]);
        let middle = graph.add_node(NodeOp::Scale(2.0), [0.0, 0.0]);
        let reader = graph.add_node(NodeOp::Scale(3.0), [0.0, 0.0]);
        graph.connect(source, middle, 0).unwrap();
        graph.connect(middle, reader, 0).unwrap();
        graph.remove_node(middle).unwrap();
        assert_eq!(graph.node(reader).unwrap().inputs, vec![Some(source)]);
    }

    // The splice is only defined for one input and one reader; with two readers there
    // is no single chain to preserve, so every pin is cleared instead.
    #[test]
    fn removing_a_node_with_two_readers_clears_both_pins() {
        let mut graph = graph();
        let source = graph.add_node(NodeOp::Constant(1.0), [0.0, 0.0]);
        let middle = graph.add_node(NodeOp::Scale(2.0), [0.0, 0.0]);
        let first = graph.add_node(NodeOp::Scale(3.0), [0.0, 0.0]);
        let second = graph.add_node(NodeOp::Scale(4.0), [0.0, 0.0]);
        graph.connect(source, middle, 0).unwrap();
        graph.connect(middle, first, 0).unwrap();
        graph.connect(middle, second, 0).unwrap();
        graph.remove_node(middle).unwrap();
        assert_eq!(graph.node(first).unwrap().inputs, vec![None]);
        assert_eq!(graph.node(second).unwrap().inputs, vec![None]);
    }

    // A source node has nothing to splice in, so its reader is left unconnected.
    #[test]
    fn removing_a_node_with_no_input_clears_its_readers_pin() {
        let mut graph = graph();
        let source = graph.add_node(NodeOp::Constant(1.0), [0.0, 0.0]);
        let reader = graph.add_node(NodeOp::Scale(2.0), [0.0, 0.0]);
        graph.connect(source, reader, 0).unwrap();
        graph.remove_node(source).unwrap();
        assert_eq!(graph.node(reader).unwrap().inputs, vec![None]);
    }

    // Removing the output node must not leave the graph naming a node that is gone.
    #[test]
    fn removing_the_output_node_leaves_no_dangling_output() {
        let mut graph = graph();
        let only = graph.add_node(NodeOp::Constant(1.0), [0.0, 0.0]);
        graph.remove_node(only).unwrap();
        assert_eq!(graph.output, None);
        assert!(graph.effective_output().is_none());
    }

    // A name is an address in a control path, so two nodes sharing one would make a
    // path ambiguous.
    #[test]
    fn a_name_already_taken_is_refused() {
        let mut graph = graph();
        let first = graph.add_node(NodeOp::Constant(1.0), [0.0, 0.0]);
        let second = graph.add_node(NodeOp::Constant(2.0), [0.0, 0.0]);
        graph.rename(first, Some("ridge".to_owned())).unwrap();
        assert_eq!(
            graph.rename(second, Some("ridge".to_owned())),
            Err(GraphError::DuplicateName("ridge".to_owned()))
        );
    }

    // A control path names a node as `n<id>` or by name, so a node named `n7` would
    // shadow node 7.
    #[test]
    fn a_name_in_the_id_path_form_is_refused() {
        let mut graph = graph();
        let id = graph.add_node(NodeOp::Constant(1.0), [0.0, 0.0]);
        assert_eq!(
            graph.rename(id, Some("n7".to_owned())),
            Err(GraphError::ReservedName("n7".to_owned()))
        );
        graph.rename(id, Some("north".to_owned())).unwrap();
    }

    // Whether a field is categorical is read off the node its value actually comes
    // from, and a bypassed node emits its input's values rather than its own.
    #[test]
    fn the_effective_output_follows_a_bypassed_node_to_its_input() {
        let mut graph = graph();
        let source = graph.add_node(NodeOp::Constant(1.0), [0.0, 0.0]);
        let scale = graph.add_node(NodeOp::Scale(2.0), [0.0, 0.0]);
        graph.connect(source, scale, 0).unwrap();
        graph.set_output(Some(scale)).unwrap();
        graph.set_bypassed(scale, true).unwrap();
        assert_eq!(graph.effective_output(), Some(source));
    }

    // A bypassed node with nothing wired in passes zero through, so there is no node
    // the field's value comes from.
    #[test]
    fn a_bypassed_output_with_no_input_has_no_effective_output() {
        let mut graph = graph();
        let scale = graph.add_node(NodeOp::Scale(2.0), [0.0, 0.0]);
        graph.set_bypassed(scale, true).unwrap();
        assert_eq!(graph.effective_output(), None);
    }

    // Writing past a node's pins is a canvas or control-path mistake, not a silent
    // no-op.
    #[test]
    fn a_pin_past_the_arity_is_refused() {
        let mut graph = graph();
        let source = graph.add_node(NodeOp::Constant(1.0), [0.0, 0.0]);
        let scale = graph.add_node(NodeOp::Scale(2.0), [0.0, 0.0]);
        assert_eq!(
            graph.connect(source, scale, 1),
            Err(GraphError::PinOutOfRange {
                node: scale,
                pin: 1,
                arity: 1,
            })
        );
    }

    // Every op is stored with exactly as many pins as it reads, so the canvas can draw
    // them without consulting a table of its own.
    #[test]
    fn a_node_carries_one_pin_per_input_the_op_reads() {
        assert_eq!(NodeOp::Constant(1.0).arity(), 0);
        assert_eq!(NodeOp::Scale(2.0).arity(), 1);
        assert_eq!(NodeOp::Binary(Binary::Add).arity(), 2);
        assert_eq!(NodeOp::Lerp.arity(), 3);
        let node = GraphNode::new(NodeId(0), NodeOp::Lerp, [0.0, 0.0]);
        assert_eq!(node.inputs.len(), 3);
    }

    // Slope is the only op that reads a neighbourhood, and the whole halo calculation
    // is derived from that.
    #[test]
    fn slope_is_the_only_op_that_widens_a_rebake() {
        assert!(
            NodeOp::Slope {
                sample_tiles: 1.0,
                mode: SlopeMode::default(),
            }
            .widens()
        );
        assert!(!NodeOp::Scale(2.0).widens());
        assert!(!NodeOp::Constant(1.0).widens());
    }

    // A curve being built in the inspector is empty for a moment, and a hole in the
    // graph there would be worse than passing the value through.
    #[test]
    fn a_curve_with_no_points_passes_its_input_through() {
        assert_eq!(Curve::default().apply(0.25), 0.25);
    }

    // One point states a value and nothing to interpolate towards.
    #[test]
    fn a_curve_with_one_point_reads_that_point_everywhere() {
        let curve = Curve::new(vec![point(0.5, 0.75)]);
        assert_eq!(curve.apply(0.0), 0.75);
        assert_eq!(curve.apply(1.0), 0.75);
    }

    // The curve is defined only between its ends, and clamping there is what keeps it
    // usable on a field whose range is wider than the curve was drawn for.
    #[test]
    fn a_curve_interpolates_between_its_points_and_clamps_outside_them() {
        let curve = Curve::new(vec![point(0.0, 0.0), point(1.0, 2.0)]);
        assert_eq!(curve.apply(0.5), 1.0);
        assert_eq!(curve.apply(-1.0), 0.0);
        assert_eq!(curve.apply(9.0), 2.0);
    }

    // The inspector edits the points as a list, so they arrive in whatever order they
    // were added and may share an input.
    #[test]
    fn a_curve_sorts_its_points_and_takes_the_later_of_two_sharing_an_input() {
        let curve = Curve::new(vec![point(1.0, 2.0), point(0.0, 0.0)]);
        assert_eq!(curve.apply(0.5), 1.0);
        let doubled = Curve::new(vec![point(0.0, 0.0), point(1.0, 5.0), point(1.0, 9.0)]);
        assert_eq!(doubled.apply(1.0), 9.0);
    }
}
