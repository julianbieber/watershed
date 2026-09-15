//! What the document can go back to: what one change leaves behind, and the two stacks
//! that turn those into undo and redo.
//!
//! An entry does not cost the size of the document. It is kept as the authored state of
//! every field and nothing derived from it, so it costs the graphs.

use crate::terrain::graph::NodeOp;
use crate::terrain::{Field, TerrainSpec, WaterSpec};

/// How many changes can be undone. Recording past this drops the oldest.
pub const HISTORY_DEPTH: usize = 100;

/// The authored state of every field on the far side of one change, the water spec
/// there, the field that was on screen, and whether crossing that change reaches the
/// bake.
///
/// Which field was on screen is part of what one change leaves behind because a
/// change may move the view: a field added is the one shown afterwards, so going
/// back across that change has to put the earlier one back, in the same step. The
/// water spec is held for the same reason: a change may rewrite it, as a field renamed
/// rewrites the name the spec solves over.
pub struct Snapshot {
    fields: Vec<Field>,
    water_spec: Option<WaterSpec>,
    active: String,
    reaches_bake: bool,
}

impl Snapshot {
    /// Copies every field's authored state as it stands: no bake and no shader values.
    ///
    /// `active` is the field on screen at the moment of the copy, which crossing the
    /// change puts back.
    pub fn take(terrain: &TerrainSpec, reaches_bake: bool, active: &str) -> Self {
        Self {
            fields: terrain.fields.iter().map(Field::authored).collect(),
            water_spec: terrain.water_spec.clone(),
            active: active.to_owned(),
            reaches_bake,
        }
    }

    /// Puts the fields and the water spec back into `terrain`, keeping what the
    /// history does not own: each live field's bake, and the values and declared reach
    /// of a shader node under the same id on both sides. A live field the snapshot does
    /// not name keeps nothing — it is not in the document afterwards. A graph's next id
    /// is never lowered, so an id freed by an undo is not handed out again.
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
                if let (NodeOp::Shader(shader), NodeOp::Shader(theirs)) =
                    (&mut node.op, &mut held.op)
                {
                    shader.put_values(theirs.take_values());
                    shader.reach = theirs.reach;
                }
            }
        }
        terrain.fields = fields;
        terrain.water_spec = self.water_spec;
    }
}

/// What crossing one change put back, and what the document has to do about it.
pub struct Restored {
    /// Whether crossing this change reaches the bake.
    pub reaches_bake: bool,
    /// The field that was on screen on the far side of the change. Equal to the
    /// current one for every change that did not move the view.
    pub active: String,
}

/// How far the history reaches in each direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryDepth {
    /// Changes that can be undone.
    pub undo: usize,
    /// Changes that can be redone.
    pub redo: usize,
}

/// The changes behind the document and the ones ahead of it.
///
/// Every entry sits on the far side of exactly one change, and undoing or redoing swaps
/// it with the document — so the entry that comes back describes the same change from
/// the other side, and the two stacks together always hold one entry per change made.
#[derive(Default)]
pub struct History {
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
}

impl History {
    /// Records a change that has just been made, `before` being the snapshot taken
    /// ahead of it. Forgets everything that could have been redone, and the oldest
    /// entry past [`HISTORY_DEPTH`].
    pub fn record(&mut self, before: Snapshot) {
        self.undo.push(before);
        self.redo.clear();
        if self.undo.len() > HISTORY_DEPTH {
            self.undo.remove(0);
        }
    }

    /// Puts the document back to before the last change and says what came back, or
    /// `None` with nothing to undo.
    ///
    /// `active` is the field on screen now, recorded so that redoing the change puts
    /// it back.
    pub fn undo(&mut self, terrain: &mut TerrainSpec, active: &str) -> Option<Restored> {
        let entry = self.undo.pop()?;
        let (restored, back) = swap(entry, terrain, active);
        self.redo.push(back);
        Some(restored)
    }

    /// Replays the last change undone and says what came back, or `None` with nothing to
    /// redo.
    ///
    /// `active` is the field on screen now, recorded so that undoing the change again
    /// puts it back.
    pub fn redo(&mut self, terrain: &mut TerrainSpec, active: &str) -> Option<Restored> {
        let entry = self.redo.pop()?;
        let (restored, back) = swap(entry, terrain, active);
        self.undo.push(back);
        Some(restored)
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

fn swap(entry: Snapshot, terrain: &mut TerrainSpec, active: &str) -> (Restored, Snapshot) {
    let restored = Restored {
        reaches_bake: entry.reaches_bake,
        active: entry.active.clone(),
    };
    let now = Snapshot::take(terrain, entry.reaches_bake, active);
    entry.restore(terrain);
    (restored, now)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::graph::NodeId;
    use bevy::math::UVec2;
    use watershed::raster::Raster;

    fn held(size: u32) -> TerrainSpec {
        let mut terrain = TerrainSpec::new(UVec2::splat(size))
            .with_field(Field::new("height").with_op(NodeOp::held(0.5)));
        terrain.bake_in_place().unwrap();
        terrain
    }

    fn values_of(terrain: &TerrainSpec) -> &Raster<f32> {
        match &terrain
            .field("height")
            .unwrap()
            .graph
            .node(NodeId(0))
            .unwrap()
            .op
        {
            NodeOp::Shader(shader) => shader.values(),
            other => panic!("the fixture node is {other:?}"),
        }
    }

    // The whole reason a snapshot is affordable: neither a bake nor a shader node's
    // values is kept, since both are re-obtained by baking.
    #[test]
    fn a_recorded_snapshot_keeps_no_bake_and_no_shader_values() {
        let terrain = held(8);
        let mut history = History::default();
        history.record(Snapshot::take(&terrain, true, "height"));

        let kept = &history.undo[0].fields[0];
        assert!(kept.baked().is_empty(), "a bake was kept");
        let NodeOp::Shader(shader) = &kept.graph.node(NodeId(0)).unwrap().op else {
            panic!("the op changed");
        };
        assert!(shader.values().is_empty(), "the shader values were kept");
    }

    // A change to the graph must not move what a shader node last produced: values
    // landed after that change have to survive it being undone and redone, because
    // nothing in the history holds them.
    #[test]
    fn a_shader_nodes_values_survive_an_undo_and_a_redo_that_keep_its_node() {
        let mut terrain = held(8);
        let mut history = History::default();
        let before = Snapshot::take(&terrain, true, "height");
        terrain.field_mut("height").unwrap().shift = 2;
        history.record(before);
        if let NodeOp::Shader(shader) = &mut terrain
            .field_mut("height")
            .unwrap()
            .graph
            .node_mut(NodeId(0))
            .unwrap()
            .op
        {
            shader.put_values(Raster::new(UVec2::splat(8), 0.75));
        }

        history.undo(&mut terrain, "height").unwrap();
        assert_eq!(terrain.field("height").unwrap().shift, 0);
        assert_eq!(values_of(&terrain).data()[3], 0.75);
        history.redo(&mut terrain, "height").unwrap();
        assert_eq!(terrain.field("height").unwrap().shift, 2);
        assert_eq!(values_of(&terrain).data()[3], 0.75);
    }

    // The bake is derived and stays with the document across a restore, so the picture
    // on screen is the last one until the re-bake lands rather than a blank.
    #[test]
    fn restoring_keeps_the_bake_the_document_has() {
        let mut terrain = held(8);
        let mut history = History::default();
        let before = Snapshot::take(&terrain, true, "height");
        terrain.field_mut("height").unwrap().range = (0.0, 2.0);
        history.record(before);

        history.undo(&mut terrain, "height").unwrap();
        assert!(!terrain.field("height").unwrap().baked().is_empty());
    }

    // A node id names its node for the life of the document, and an undo must not
    // break that promise by handing an id out twice.
    #[test]
    fn an_id_freed_by_an_undo_is_not_reused() {
        let mut terrain = held(8);
        let mut history = History::default();
        let before = Snapshot::take(&terrain, true, "height");
        let first = terrain
            .field_mut("height")
            .unwrap()
            .graph
            .add_node(NodeOp::held(1.0), [0.0, 0.0]);
        history.record(before);

        history.undo(&mut terrain, "height").unwrap();
        let second = terrain
            .field_mut("height")
            .unwrap()
            .graph
            .add_node(NodeOp::held(2.0), [0.0, 0.0]);
        assert_ne!(first, second);
    }

    // The two stacks are one sequence of changes: a new change forgets what could have
    // been redone, and the depth is bounded.
    #[test]
    fn a_new_change_forgets_the_redo_stack_and_the_depth_is_bounded() {
        let mut terrain = held(8);
        let mut history = History::default();
        for step in 0..(HISTORY_DEPTH + 5) {
            let before = Snapshot::take(&terrain, true, "height");
            terrain.field_mut("height").unwrap().range = (0.0, step as f32);
            history.record(before);
        }
        assert_eq!(history.depth().undo, HISTORY_DEPTH);

        history.undo(&mut terrain, "height").unwrap();
        history.undo(&mut terrain, "height").unwrap();
        assert_eq!(history.depth().redo, 2);
        let before = Snapshot::take(&terrain, true, "height");
        terrain.field_mut("height").unwrap().range = (0.0, 1.0);
        history.record(before);
        assert_eq!(history.depth().redo, 0);
        assert!(history.redo(&mut terrain, "height").is_none());
    }
}
