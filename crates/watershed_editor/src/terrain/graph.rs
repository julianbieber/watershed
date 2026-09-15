//! What a field's value is built out of: the nodes, the edges between them, and the
//! one node the field is read from.

use serde::{Deserialize, Serialize};

use watershed::field::FieldId;

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

/// What a node produces, given whatever is wired into it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NodeOp {
    /// A raster a WGSL shader produced, at the field's own resolution.
    ///
    /// One input pin per input the file declares, in declaration order, each read
    /// inside the shader as a texture of the whole upstream raster; an unwired pin
    /// reads `0.0`. Every field the file names in a `@layer` annotation is a
    /// dependency, bound as that field's baked raster. A wired pin or a layer read
    /// widens the re-bake: by the reach the file declares, or — for a file that
    /// declares none, which may read any texel of what it is handed — by baking the
    /// field whole instead. The values are not serialized: a loaded document reads the
    /// node as `0.0` until it has been dispatched again.
    Shader(ShaderLayer),
    /// Another field's value at the position, interpolated between that field's
    /// texels.
    FieldRef(FieldId),
}

impl NodeOp {
    /// How many inputs this op reads. An op is always stored with exactly this many
    /// input pins, connected or not.
    pub fn arity(&self) -> usize {
        match self {
            NodeOp::FieldRef(_) => 0,
            NodeOp::Shader(shader) => shader.inputs.len(),
        }
    }

    /// The field a [`NodeOp::FieldRef`] names, and `None` for every other op —
    /// including a shader that reads fields by name, which [`NodeOp::reads`] reports.
    pub fn dependency(&self) -> Option<&FieldId> {
        match self {
            NodeOp::FieldRef(id) => Some(id),
            NodeOp::Shader(_) => None,
        }
    }

    /// Every field this op reads: a reference's one field, or the fields a shader's
    /// file names in `@layer` annotations, in declaration order and with duplicates
    /// kept. Feeds bake ordering and cycle detection.
    pub fn reads(&self) -> Vec<&FieldId> {
        match self {
            NodeOp::FieldRef(id) => vec![id],
            NodeOp::Shader(shader) => shader.layers.iter().collect(),
        }
    }
}

#[cfg(test)]
impl NodeOp {
    /// A shader node that reads `value` everywhere while no shader runtime is
    /// installed: it carries a `value` parameter and holds a single texel of it.
    pub fn held(value: f32) -> Self {
        let mut layer = ShaderLayer::new("held.wgsl");
        layer.params.insert("value".to_owned(), vec![value]);
        layer.put_values(watershed::raster::Raster::new(glam::UVec2::ONE, value));
        Self::Shader(layer)
    }

    /// A shader node that reads `raster` while no shader runtime is installed. The
    /// raster has to be at the resolution of the field the node is placed in.
    pub fn holding(raster: watershed::raster::Raster<f32>) -> Self {
        let mut layer = ShaderLayer::new("held.wgsl");
        layer.put_values(raster);
        Self::Shader(layer)
    }

    /// A shader node with `pins` input pins and no values, which reads `0.0` wherever
    /// it is not bypassed.
    pub fn piped(pins: usize) -> Self {
        let mut layer = ShaderLayer::new("piped.wgsl");
        layer.inputs = (0..pins).map(|pin| format!("in{pin}")).collect();
        Self::Shader(layer)
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

    fn clear(&self, at: [f32; 2]) -> bool {
        !self.nodes.iter().any(|node| {
            (node.position[0] - at[0]).abs() < NODE_STEP[0] * 0.5
                && (node.position[1] - at[1]).abs() < NODE_STEP[1] * 0.5
        })
    }

    /// A spot no card is sitting on, for a node added without one.
    ///
    /// Laid out in rows below the origin, so a graph built entirely through control
    /// verbs comes out readable rather than as a stack of cards on one another.
    pub fn free_position(&self) -> [f32; 2] {
        for row in 0..64 {
            for column in 0..8 {
                let at = [column as f32 * NODE_STEP[0], -(row as f32) * NODE_STEP[1]];
                if self.clear(at) {
                    return at;
                }
            }
        }
        [0.0, 0.0]
    }

    /// A spot no card is sitting on, one `NODE_STEP` right of `anchor`.
    ///
    /// An `anchor` the graph does not hold is not an error: it falls back to the
    /// output node, as `None` does, and a graph with neither falls back to
    /// `free_position`. The spot returned overlaps no existing card.
    pub fn free_position_beside(&self, anchor: Option<NodeId>) -> [f32; 2] {
        let beside = anchor
            .filter(|id| self.node(*id).is_some())
            .or(self.output)
            .and_then(|id| self.node(id));
        let Some(beside) = beside else {
            return self.free_position();
        };
        let column = beside.position[0] + NODE_STEP[0];
        for row in 0..64 {
            let at = [column, beside.position[1] - row as f32 * NODE_STEP[1]];
            if self.clear(at) {
                return at;
            }
        }
        self.free_position()
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
    /// unconnected.
    ///
    /// If the node was the output, the output passes to the first of these the graph
    /// still holds once the node is gone: the input spliced in; the node on the removed
    /// node's first pin, if nothing reads it any more; the highest-id node nothing
    /// reads; the highest-id node. Only a graph left with no nodes has no output.
    pub fn remove_node(&mut self, id: NodeId) -> Result<(), GraphError> {
        let node = self.node(id).ok_or(GraphError::UnknownNode(id))?;
        let first_input = node.inputs.first().copied().flatten();
        let only_source = match node.inputs.iter().flatten().collect::<Vec<_>>().as_slice() {
            [single] => Some(**single),
            _ => None,
        };
        let readers: Vec<NodeId> = self.readers(id).collect();
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
            self.output = self.successor(splice, first_input);
        }
        Ok(())
    }

    /// The nodes that read `id` on any pin, bypassed or not.
    pub fn readers(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes
            .iter()
            .filter(move |other| other.sources().any(|source| source == id))
            .map(|other| other.id)
    }

    fn successor(&self, splice: Option<NodeId>, first_input: Option<NodeId>) -> Option<NodeId> {
        let read: Vec<NodeId> = self.nodes.iter().flat_map(GraphNode::sources).collect();
        let held = |id: &NodeId| self.node(*id).is_some();
        let unread = |id: &NodeId| !read.contains(id);
        let ids = || self.nodes.iter().map(|node| node.id);
        splice
            .filter(held)
            .or_else(|| first_input.filter(held).filter(unread))
            .or_else(|| ids().filter(unread).max())
            .or_else(|| ids().max())
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
    /// This, not [`FieldGraph::output`], is the node whose values the field bakes: a
    /// bypassed node at the output emits its input's values, not its own.
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
    /// Only what [`NodeOp::reads`] reports for the nodes the walk actually reached, so
    /// an unreachable or bypassed node contributes no bake-order dependency and cannot
    /// make a cycle between fields.
    pub fn dependencies(&self) -> Vec<&FieldId> {
        self.evaluation_order()
            .unwrap_or_default()
            .into_iter()
            .flat_map(|id| {
                self.node(id)
                    .map(|node| node.op.reads())
                    .unwrap_or_default()
            })
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

    // An id names a node for the life of the document, so removing one must not free
    // its id for the next node: an edge or a control path would silently retarget.
    #[test]
    fn an_id_is_never_reused_after_the_node_is_removed() {
        let mut graph = graph();
        let first = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        graph.remove_node(first).unwrap();
        let second = graph.add_node(NodeOp::held(2.0), [0.0, 0.0]);
        assert_ne!(first, second);
    }

    // A field is what its output node reads, and a graph whose first node is not the
    // output would bake zero until wired by hand.
    #[test]
    fn the_first_node_added_becomes_the_output() {
        let mut graph = graph();
        let id = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        assert_eq!(graph.output, Some(id));
    }

    // A node feeding itself is the shortest cycle and the easiest to draw on a canvas.
    #[test]
    fn a_node_cannot_be_connected_to_itself() {
        let mut graph = graph();
        let id = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
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
        let a = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
        let b = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
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
        let a = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
        let b = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
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
        let source = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        let middle = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
        let reader = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
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
        let source = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        let middle = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
        let first = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
        let second = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
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
        let source = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        let reader = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
        graph.connect(source, reader, 0).unwrap();
        graph.remove_node(source).unwrap();
        assert_eq!(graph.node(reader).unwrap().inputs, vec![None]);
    }

    // Removing the output node must not leave the graph naming a node that is gone.
    #[test]
    fn removing_the_output_node_leaves_no_dangling_output() {
        let mut graph = graph();
        let only = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        graph.remove_node(only).unwrap();
        assert_eq!(graph.output, None);
        assert!(graph.effective_output().is_none());
    }

    // Deleting the end of a chain auto-chooses the node now at the end of the graph as
    // the output, rather than leaving the field baking zero.
    #[test]
    fn removing_the_output_at_the_end_of_a_chain_hands_it_to_the_node_before() {
        let mut graph = graph();
        let first = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        let middle = graph.node_with(NodeOp::piped(1), &[first]);
        let last = graph.node_with(NodeOp::piped(1), &[middle]);
        graph.set_output(Some(last)).unwrap();
        graph.remove_node(last).unwrap();
        assert_eq!(graph.output, Some(middle));
    }

    // An output something reads is spliced out of its chain, and the input spliced in
    // outranks every other successor.
    #[test]
    fn removing_an_output_that_is_read_hands_it_to_the_spliced_input() {
        let mut graph = graph();
        let source = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        let middle = graph.node_with(NodeOp::piped(1), &[source]);
        graph.node_with(NodeOp::piped(1), &[middle]);
        graph.set_output(Some(middle)).unwrap();
        graph.remove_node(middle).unwrap();
        assert_eq!(graph.output, Some(source));
    }

    // With the first pin empty there is no chain to step back along, so the output goes
    // to the newest node nothing reads.
    #[test]
    fn removing_an_output_with_an_empty_first_pin_hands_it_to_the_newest_unread_node() {
        let mut graph = graph();
        let second_pin = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        let output = graph.add_node(NodeOp::piped(2), [0.0, 0.0]);
        graph.connect(second_pin, output, 1).unwrap();
        let newest = graph.add_node(NodeOp::held(2.0), [0.0, 0.0]);
        graph.set_output(Some(output)).unwrap();
        graph.remove_node(output).unwrap();
        assert_eq!(graph.output, Some(newest));
    }

    // A first-pin node that still feeds another reader is not the end of the graph, so
    // the output passes over it.
    #[test]
    fn removing_an_output_whose_input_is_read_elsewhere_skips_that_input() {
        let mut graph = graph();
        let source = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        let output = graph.node_with(NodeOp::piped(1), &[source]);
        let other = graph.node_with(NodeOp::piped(1), &[source]);
        graph.set_output(Some(output)).unwrap();
        graph.remove_node(output).unwrap();
        assert_eq!(graph.output, Some(other));
    }

    // A loaded graph can hold a cycle, in which every node is read; one with nodes left
    // must still come out with an output.
    #[test]
    fn removing_the_output_beside_a_cycle_still_leaves_an_output() {
        let mut graph = graph();
        let a = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
        let b = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
        graph.node_mut(a).unwrap().inputs[0] = Some(b);
        graph.node_mut(b).unwrap().inputs[0] = Some(a);
        let output = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        graph.set_output(Some(output)).unwrap();
        graph.remove_node(output).unwrap();
        assert_eq!(graph.output, Some(b));
    }

    // A loaded node can read itself, which makes its own id the splice source; the output
    // must never be left naming the node that was removed.
    #[test]
    fn removing_a_self_reading_output_never_leaves_the_removed_id() {
        let mut graph = graph();
        let looped = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
        graph.node_mut(looped).unwrap().inputs[0] = Some(looped);
        let other = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        graph.set_output(Some(looped)).unwrap();
        graph.remove_node(looped).unwrap();
        assert_eq!(graph.output, Some(other));
    }

    // A name is an address in a control path, so two nodes sharing one would make a
    // path ambiguous.
    #[test]
    fn a_name_already_taken_is_refused() {
        let mut graph = graph();
        let first = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        let second = graph.add_node(NodeOp::held(2.0), [0.0, 0.0]);
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
        let id = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        assert_eq!(
            graph.rename(id, Some("n7".to_owned())),
            Err(GraphError::ReservedName("n7".to_owned()))
        );
        graph.rename(id, Some("north".to_owned())).unwrap();
    }

    // A field bakes the values of the node its value actually comes from, and a
    // bypassed node emits its input's values rather than its own.
    #[test]
    fn the_effective_output_follows_a_bypassed_node_to_its_input() {
        let mut graph = graph();
        let source = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        let piped = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
        graph.connect(source, piped, 0).unwrap();
        graph.set_output(Some(piped)).unwrap();
        graph.set_bypassed(piped, true).unwrap();
        assert_eq!(graph.effective_output(), Some(source));
    }

    // A bypassed node with nothing wired in passes zero through, so there is no node
    // the field's value comes from.
    #[test]
    fn a_bypassed_output_with_no_input_has_no_effective_output() {
        let mut graph = graph();
        let piped = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
        graph.set_bypassed(piped, true).unwrap();
        assert_eq!(graph.effective_output(), None);
    }

    // Writing past a node's pins is a canvas or control-path mistake, not a silent
    // no-op.
    #[test]
    fn a_pin_past_the_arity_is_refused() {
        let mut graph = graph();
        let source = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        let piped = graph.add_node(NodeOp::piped(1), [0.0, 0.0]);
        assert_eq!(
            graph.connect(source, piped, 1),
            Err(GraphError::PinOutOfRange {
                node: piped,
                pin: 1,
                arity: 1,
            })
        );
    }

    // Every op is stored with exactly as many pins as it reads, so the canvas can draw
    // them without consulting a table of its own — and a shader's pins come from the
    // file it names, so a node added for a two-input shader is born with two.
    #[test]
    fn a_node_carries_one_pin_per_input_the_op_reads() {
        assert_eq!(NodeOp::FieldRef(FieldId::from("base")).arity(), 0);
        let mut layer = ShaderLayer::new("blur.wgsl");
        layer.inputs = vec!["a".to_owned(), "b".to_owned()];
        let op = NodeOp::Shader(layer);
        assert_eq!(op.arity(), 2);
        assert_eq!(GraphNode::new(NodeId(0), op, [0.0, 0.0]).inputs.len(), 2);
    }

    // The whole point of the change: a node added while something is selected lands
    // beside that selection rather than in the next free grid slot.
    #[test]
    fn a_spot_beside_an_anchor_is_one_step_right_of_it() {
        let mut graph = graph();
        let anchor = graph.add_node(NodeOp::held(1.0), [400.0, -300.0]);
        assert_eq!(
            graph.free_position_beside(Some(anchor)),
            [400.0 + NODE_STEP[0], -300.0]
        );
    }

    // Landing beside the selection is worth nothing if the card lands on top of one
    // that is already there, which is exactly the crowded case the probe exists for.
    #[test]
    fn a_spot_beside_an_anchor_avoids_a_card_already_sitting_there() {
        let mut graph = graph();
        let anchor = graph.add_node(NodeOp::held(1.0), [0.0, 0.0]);
        graph.add_node(NodeOp::held(2.0), [NODE_STEP[0], 0.0]);
        let at = graph.free_position_beside(Some(anchor));
        assert!(graph.nodes.iter().all(|node| {
            (node.position[0] - at[0]).abs() >= NODE_STEP[0] * 0.5
                || (node.position[1] - at[1]).abs() >= NODE_STEP[1] * 0.5
        }));
    }

    // Nothing selected is the ordinary state of a fresh field, and the issue asks for
    // the output node to stand in as the anchor there.
    #[test]
    fn a_spot_beside_nothing_anchors_on_the_output_node() {
        let mut graph = graph();
        let first = graph.add_node(NodeOp::held(1.0), [700.0, -100.0]);
        assert_eq!(graph.output, Some(first));
        assert_eq!(
            graph.free_position_beside(None),
            [700.0 + NODE_STEP[0], -100.0]
        );
    }
}
