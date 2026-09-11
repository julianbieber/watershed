//! What the document can go back to: the snapshot a change leaves behind, and the two
//! stacks that turn snapshots into undo and redo.
//!
//! A snapshot holds what a person authored and nothing derived from it, so the
//! history costs the size of the graphs rather than the size of the document.

use crate::terrain::graph::NodeOp;
use crate::terrain::{Field, TerrainSpec};

/// How many changes can be undone. Recording past this drops the oldest.
pub const HISTORY_DEPTH: usize = 100;

/// The authored state of every field on the far side of one change, and whether
/// crossing that change reaches the bake.
pub struct Snapshot {
    fields: Vec<Field>,
    reaches_bake: bool,
}

impl Snapshot {
    /// Copies every field's authored state as it stands: no bake, no shader values,
    /// every paint and external raster still held. Call [`Snapshot::shed`] once the
    /// change has been made, or the copy keeps rasters the document still has.
    pub fn take(terrain: &TerrainSpec, reaches_bake: bool) -> Self {
        Self {
            fields: terrain.fields.iter().map(Field::authored).collect(),
            reaches_bake,
        }
    }

    /// Whether crossing the change this snapshot sits beside reaches the bake.
    pub fn reaches_bake(&self) -> bool {
        self.reaches_bake
    }

    /// Drops every raster `live` still holds under the same node and the same op, so
    /// the snapshot retains a raster only for a node the change dropped.
    pub fn shed(&mut self, live: &TerrainSpec) {
        for field in &mut self.fields {
            let Some(current) = live.field(field.id.as_str()) else {
                continue;
            };
            for node in &mut field.graph.nodes {
                let Some(held) = current.graph.node(node.id) else {
                    continue;
                };
                match (&mut node.op, &held.op) {
                    (NodeOp::Paint(raster), NodeOp::Paint(_)) => *raster = Default::default(),
                    (NodeOp::External(raster), NodeOp::External(_)) => {
                        *raster = Default::default();
                    }
                    _ => {}
                }
            }
        }
    }

    /// Puts the fields back into `terrain`, keeping what the history does not own:
    /// each live field's bake, and the live raster or shader values of a node that
    /// is the same op under the same id on both sides. A graph's next id is never
    /// lowered, so an id freed by an undo is not handed out again.
    pub fn restore(self, terrain: &mut TerrainSpec) {
        let mut fields = self.fields;
        for field in &mut fields {
            let Some(live) = terrain.field_mut(field.id.as_str()) else {
                continue;
            };
            field.put_baked(live.take_baked());
            field.graph.next_id = field.graph.next_id.max(live.graph.next_id);
            for node in &mut field.graph.nodes {
                let Some(held) = live.graph.node_mut(node.id) else {
                    continue;
                };
                match (&mut node.op, &mut held.op) {
                    (NodeOp::Paint(raster), NodeOp::Paint(theirs)) if raster.is_empty() => {
                        *raster = std::mem::take(theirs);
                    }
                    (NodeOp::External(raster), NodeOp::External(theirs)) if raster.is_empty() => {
                        *raster = std::mem::take(theirs);
                    }
                    (NodeOp::Shader(shader), NodeOp::Shader(theirs)) => {
                        shader.put_values(theirs.take_values());
                    }
                    _ => {}
                }
            }
        }
        terrain.fields = fields;
    }
}

/// How far the history reaches in each direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryDepth {
    /// Changes that can be undone.
    pub undo: usize,
    /// Changes that can be redone.
    pub redo: usize,
}

/// The snapshots behind the document and the ones ahead of it.
///
/// Every entry sits on the far side of exactly one change, and undoing or redoing
/// swaps it with the document — so the entry that comes back carries the same
/// reaches-bake mark, and the two stacks together always hold one snapshot per
/// change made.
#[derive(Default)]
pub struct History {
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
}

impl History {
    /// Records a change that has just been made: `before` is the snapshot taken
    /// ahead of it and `live` is the document as it now stands. Forgets everything
    /// that could have been redone, and the oldest entry past [`HISTORY_DEPTH`].
    pub fn record(&mut self, mut before: Snapshot, live: &TerrainSpec) {
        before.shed(live);
        self.undo.push(before);
        self.redo.clear();
        if self.undo.len() > HISTORY_DEPTH {
            self.undo.remove(0);
        }
    }

    /// Puts the document back to before the last change and says whether that
    /// change reached the bake, or `None` with nothing to undo.
    pub fn undo(&mut self, terrain: &mut TerrainSpec) -> Option<bool> {
        let entry = self.undo.pop()?;
        let reaches_bake = swap(entry, terrain);
        self.redo.push(reaches_bake.1);
        Some(reaches_bake.0)
    }

    /// Replays the last change undone and says whether it reaches the bake, or
    /// `None` with nothing to redo.
    pub fn redo(&mut self, terrain: &mut TerrainSpec) -> Option<bool> {
        let entry = self.redo.pop()?;
        let reaches_bake = swap(entry, terrain);
        self.undo.push(reaches_bake.1);
        Some(reaches_bake.0)
    }

    /// Forgets everything, for a document that has just been replaced.
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }

    /// How far the history reaches in each direction.
    pub fn depth(&self) -> HistoryDepth {
        HistoryDepth {
            undo: self.undo.len(),
            redo: self.redo.len(),
        }
    }
}

fn swap(entry: Snapshot, terrain: &mut TerrainSpec) -> (bool, Snapshot) {
    let reaches_bake = entry.reaches_bake();
    let mut now = Snapshot::take(terrain, reaches_bake);
    entry.restore(terrain);
    now.shed(terrain);
    (reaches_bake, now)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::graph::NodeId;
    use bevy::math::UVec2;
    use watershed::raster::Raster;

    fn painted(size: u32) -> TerrainSpec {
        let mut terrain = TerrainSpec::new(UVec2::splat(size)).with_field(
            Field::new("height").with_op(NodeOp::Paint(Raster::new(UVec2::splat(size), 7u8))),
        );
        terrain.bake_in_place().unwrap();
        terrain
    }

    fn paint_of(terrain: &TerrainSpec) -> Option<&Raster<u8>> {
        match &terrain.field("height")?.graph.node(NodeId(0))?.op {
            NodeOp::Paint(raster) => Some(raster),
            _ => None,
        }
    }

    // The whole reason a snapshot is affordable: a raster the document still holds
    // is not kept twice, and a bake is never kept at all.
    #[test]
    fn a_recorded_snapshot_sheds_the_raster_the_document_still_holds() {
        let terrain = painted(8);
        let mut history = History::default();
        history.record(Snapshot::take(&terrain, true), &terrain);

        let held = &history.undo[0].fields[0];
        assert!(held.baked().is_empty(), "a bake was kept");
        let NodeOp::Paint(raster) = &held.graph.node(NodeId(0)).unwrap().op else {
            panic!("the op changed");
        };
        assert!(raster.is_empty(), "a raster the document holds was kept");
    }

    // And the other half of that rule: a raster whose node the change dropped is the
    // only copy left, so it has to come back with the node.
    #[test]
    fn undoing_a_removal_brings_the_nodes_raster_back() {
        let mut terrain = painted(8);
        let mut history = History::default();
        let before = Snapshot::take(&terrain, true);
        terrain
            .field_mut("height")
            .unwrap()
            .graph
            .remove_node(NodeId(0))
            .unwrap();
        history.record(before, &terrain);
        assert!(paint_of(&terrain).is_none());

        assert_eq!(history.undo(&mut terrain), Some(true));
        assert_eq!(paint_of(&terrain).unwrap().data(), &[7u8; 64][..]);
    }

    // A stroke is not in the history, so what the brush painted after a change has to
    // survive that change being undone and redone — the live raster wins both ways.
    #[test]
    fn a_live_raster_survives_an_undo_and_a_redo_that_keep_its_node() {
        let mut terrain = painted(8);
        let mut history = History::default();
        let before = Snapshot::take(&terrain, true);
        terrain.field_mut("height").unwrap().shift = 2;
        history.record(before, &terrain);
        if let NodeOp::Paint(raster) = &mut terrain
            .field_mut("height")
            .unwrap()
            .graph
            .node_mut(NodeId(0))
            .unwrap()
            .op
        {
            raster.data_mut()[3] = 200;
        }

        history.undo(&mut terrain).unwrap();
        assert_eq!(terrain.field("height").unwrap().shift, 0);
        assert_eq!(paint_of(&terrain).unwrap().data()[3], 200);
        history.redo(&mut terrain).unwrap();
        assert_eq!(terrain.field("height").unwrap().shift, 2);
        assert_eq!(paint_of(&terrain).unwrap().data()[3], 200);
    }

    // The bake is derived and stays with the document across a restore, so the picture
    // on screen is the last one until the re-bake lands rather than a blank.
    #[test]
    fn restoring_keeps_the_bake_the_document_has() {
        let mut terrain = painted(8);
        let mut history = History::default();
        let before = Snapshot::take(&terrain, true);
        terrain.field_mut("height").unwrap().range = (0.0, 2.0);
        history.record(before, &terrain);

        history.undo(&mut terrain).unwrap();
        assert!(!terrain.field("height").unwrap().baked().is_empty());
    }

    // A node id names its node for the life of the document, and an undo must not
    // break that promise by handing an id out twice.
    #[test]
    fn an_id_freed_by_an_undo_is_not_reused() {
        let mut terrain = painted(8);
        let mut history = History::default();
        let before = Snapshot::take(&terrain, true);
        let first = terrain
            .field_mut("height")
            .unwrap()
            .graph
            .add_node(NodeOp::Constant(1.0), [0.0, 0.0]);
        history.record(before, &terrain);

        history.undo(&mut terrain).unwrap();
        let second = terrain
            .field_mut("height")
            .unwrap()
            .graph
            .add_node(NodeOp::Constant(2.0), [0.0, 0.0]);
        assert_ne!(first, second);
    }

    // The two stacks are one sequence of changes: a new change forgets what could have
    // been redone, and the depth is bounded.
    #[test]
    fn a_new_change_forgets_the_redo_stack_and_the_depth_is_bounded() {
        let mut terrain = painted(8);
        let mut history = History::default();
        for step in 0..(HISTORY_DEPTH + 5) {
            let before = Snapshot::take(&terrain, true);
            terrain.field_mut("height").unwrap().range = (0.0, step as f32);
            history.record(before, &terrain);
        }
        assert_eq!(history.depth().undo, HISTORY_DEPTH);

        history.undo(&mut terrain).unwrap();
        history.undo(&mut terrain).unwrap();
        assert_eq!(history.depth().redo, 2);
        let before = Snapshot::take(&terrain, true);
        terrain.field_mut("height").unwrap().range = (0.0, 1.0);
        history.record(before, &terrain);
        assert_eq!(history.depth().redo, 0);
        assert!(history.redo(&mut terrain).is_none());
    }
}
