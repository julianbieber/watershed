//! What the document can go back to: what one change leaves behind, and the two stacks
//! that turn those into undo and redo.
//!
//! Neither kind of entry costs the size of the document. A change to the stack is kept
//! as the authored state of every field and nothing derived from it, so it costs the
//! graphs; a stroke is kept as the texels that were under it, so it costs its own
//! footprint.

use bevy::math::UVec2;
use watershed::CellRect;
use watershed::raster::Raster;

use crate::terrain::graph::{NodeId, NodeOp};
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

/// One frame's worth of a stroke: the texels that were under it before it painted.
///
/// Addressed in the raster's own texels rather than in document cells, and only
/// meaningful against a raster of [`StrokePatch::raster_size`] — a raster that has since
/// been replaced at another resolution is left alone rather than written at the wrong
/// offsets.
pub struct StrokePatch {
    rect: CellRect,
    texels: Vec<u8>,
    raster_size: UVec2,
    empty: bool,
}

impl StrokePatch {
    /// Copies what is under `rect` out of `raster` as it stands, against a raster of
    /// `raster_size`.
    ///
    /// An empty raster is recorded as empty rather than as zeros, so putting the patch
    /// back leaves the node with no raster at all — which is what it had before a stroke
    /// allocated one. Call it *before* the raster is allocated, or a first stroke
    /// records the zeros it just made instead.
    pub fn before(raster: &Raster<u8>, rect: CellRect, raster_size: UVec2) -> Self {
        Self {
            rect,
            texels: raster.copy_rect(rect).unwrap_or_default(),
            raster_size,
            empty: raster.is_empty(),
        }
    }

    /// Puts the patch back into `raster`, which is what crossing the change does in
    /// either direction.
    ///
    /// A patch taken from an empty raster leaves the node with none, so the first stroke
    /// into a paint node undoes to what it had rather than to a raster of zeros. A raster
    /// at some other size than the patch was taken against is left alone: the patch
    /// cannot address it, and writing it anyway would smear across the wrong rows.
    pub fn apply(&self, raster: &mut Raster<u8>) {
        if self.empty {
            *raster = Raster::default();
            return;
        }
        if raster.is_empty() {
            *raster = Raster::new(self.raster_size, 0u8);
        } else if raster.size() != self.raster_size {
            return;
        }
        raster.paste_rect(self.rect, &self.texels);
    }
}

/// A stroke as the history holds it: the node it painted into and the pieces it was laid
/// down in, oldest first.
///
/// A drag is many pieces, one per frame the document was free to take one, and they
/// overlap — so the order is load-bearing. Only running the whole sequence backwards,
/// each piece putting back what the piece after it found, inverts the drag; running it
/// forwards replays it.
pub struct Stroke {
    field: String,
    node: NodeId,
    pieces: Vec<StrokePatch>,
    cells: CellRect,
}

impl Stroke {
    fn swap(mut self, terrain: &mut TerrainSpec, order: Order) -> Self {
        let Some(raster) = paint_raster(terrain, &self.field, self.node) else {
            return self;
        };
        let last = self.pieces.len();
        for step in 0..last {
            let at = match order {
                Order::Backward => last - 1 - step,
                Order::Forward => step,
            };
            let now =
                StrokePatch::before(raster, self.pieces[at].rect, self.pieces[at].raster_size);
            self.pieces[at].apply(raster);
            self.pieces[at] = now;
        }
        self
    }
}

#[derive(Clone, Copy)]
enum Order {
    Backward,
    Forward,
}

enum Change {
    Fields(Snapshot),
    Stroke(Stroke),
}

/// What crossing one change put back, and what the document has to do about it.
///
/// The history says which cells a stroke covered; how far that reaches through the
/// fields that read them is the document's question, not this module's.
pub enum Restored {
    /// Every field's authored state, from a change to the stack.
    Fields {
        /// Whether crossing this change reaches the bake.
        reaches_bake: bool,
    },
    /// A stroke, naming the field it painted and the cells it covered.
    Stroke {
        /// The field whose paint node was written.
        field: String,
        /// The document cells the stroke painted.
        cells: CellRect,
    },
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
    undo: Vec<Change>,
    redo: Vec<Change>,
}

impl History {
    /// Records a change to the stack that has just been made: `before` is the snapshot
    /// taken ahead of it and `live` is the document as it now stands. Forgets everything
    /// that could have been redone, and the oldest entry past [`HISTORY_DEPTH`].
    pub fn record(&mut self, mut before: Snapshot, live: &TerrainSpec) {
        before.shed(live);
        self.push(Change::Fields(before));
    }

    /// Records a stroke that has just painted `painted` into `node` of `field`, `piece`
    /// being what was under it beforehand.
    ///
    /// `joins` asks for this to extend the entry the same drag opened rather than making
    /// one, which it does only while that entry is still the one on top and nothing has
    /// been undone since — so a drag is one undo step, and a piece laid down after a
    /// Ctrl+Z or an edit starts its own.
    pub fn record_stroke(
        &mut self,
        field: &str,
        node: NodeId,
        piece: StrokePatch,
        painted: CellRect,
        joins: bool,
    ) {
        if joins
            && self.redo.is_empty()
            && let Some(Change::Stroke(open)) = self.undo.last_mut()
            && open.field == field
            && open.node == node
        {
            open.pieces.push(piece);
            open.cells = open.cells.union(painted);
            return;
        }
        self.push(Change::Stroke(Stroke {
            field: field.to_owned(),
            node,
            pieces: vec![piece],
            cells: painted,
        }));
    }

    /// Puts the document back to before the last change and says what came back, or
    /// `None` with nothing to undo.
    pub fn undo(&mut self, terrain: &mut TerrainSpec) -> Option<Restored> {
        let entry = self.undo.pop()?;
        let (restored, back) = swap(entry, terrain, Order::Backward);
        self.redo.push(back);
        Some(restored)
    }

    /// Replays the last change undone and says what came back, or `None` with nothing to
    /// redo.
    pub fn redo(&mut self, terrain: &mut TerrainSpec) -> Option<Restored> {
        let entry = self.redo.pop()?;
        let (restored, back) = swap(entry, terrain, Order::Forward);
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

    fn push(&mut self, change: Change) {
        self.undo.push(change);
        self.redo.clear();
        if self.undo.len() > HISTORY_DEPTH {
            self.undo.remove(0);
        }
    }
}

fn paint_raster<'a>(
    terrain: &'a mut TerrainSpec,
    field: &str,
    node: NodeId,
) -> Option<&'a mut Raster<u8>> {
    match &mut terrain.field_mut(field)?.graph.node_mut(node)?.op {
        NodeOp::Paint(raster) => Some(raster),
        _ => None,
    }
}

fn swap(entry: Change, terrain: &mut TerrainSpec, order: Order) -> (Restored, Change) {
    match entry {
        Change::Fields(entry) => {
            let reaches_bake = entry.reaches_bake();
            let mut now = Snapshot::take(terrain, reaches_bake);
            entry.restore(terrain);
            now.shed(terrain);
            (Restored::Fields { reaches_bake }, Change::Fields(now))
        }
        Change::Stroke(stroke) => {
            let restored = Restored::Stroke {
                field: stroke.field.clone(),
                cells: stroke.cells,
            };
            (restored, Change::Stroke(stroke.swap(terrain, order)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::UVec2;
    use watershed::raster::Raster;

    fn painted(size: u32) -> TerrainSpec {
        let mut terrain = TerrainSpec::new(UVec2::splat(size)).with_field(
            Field::new("height").with_op(NodeOp::Paint(Raster::new(UVec2::splat(size), 7u8))),
        );
        terrain.bake_in_place().unwrap();
        terrain
    }

    fn unpainted(size: u32) -> TerrainSpec {
        TerrainSpec::new(UVec2::splat(size))
            .with_field(Field::new("height").with_op(NodeOp::Paint(Raster::default())))
    }

    fn paint_of(terrain: &TerrainSpec) -> Option<&Raster<u8>> {
        match &terrain.field("height")?.graph.node(NodeId(0))?.op {
            NodeOp::Paint(raster) => Some(raster),
            _ => None,
        }
    }

    fn texels_of(terrain: &TerrainSpec) -> Vec<u8> {
        paint_of(terrain)
            .map(|raster| raster.data().to_vec())
            .unwrap_or_default()
    }

    fn rect(min: u32, max: u32) -> CellRect {
        CellRect::new(UVec2::splat(min), UVec2::splat(max))
    }

    fn stroke(
        terrain: &mut TerrainSpec,
        history: &mut History,
        area: CellRect,
        value: u8,
        joins: bool,
    ) {
        let size = terrain.size;
        let raster = paint_raster(terrain, "height", NodeId(0)).expect("the fixture paints");
        let texels = if raster.is_empty() {
            size
        } else {
            raster.size()
        };
        let piece = StrokePatch::before(raster, area, texels);
        if raster.is_empty() {
            *raster = Raster::new(texels, 0u8);
        }
        for y in area.min.y..area.max.y {
            for x in area.min.x..area.max.x {
                raster.set(x, y, value);
            }
        }
        history.record_stroke("height", NodeId(0), piece, area, joins);
    }

    fn edit(terrain: &mut TerrainSpec, history: &mut History, shift: u8) {
        let before = Snapshot::take(terrain, true);
        terrain.field_mut("height").unwrap().shift = shift;
        history.record(before, terrain);
    }

    fn undone(history: &mut History, terrain: &mut TerrainSpec) -> Restored {
        history.undo(terrain).expect("there is something to undo")
    }

    // The whole reason a snapshot is affordable: a raster the document still holds
    // is not kept twice, and a bake is never kept at all.
    #[test]
    fn a_recorded_snapshot_sheds_the_raster_the_document_still_holds() {
        let terrain = painted(8);
        let mut history = History::default();
        history.record(Snapshot::take(&terrain, true), &terrain);

        let Change::Fields(entry) = &history.undo[0] else {
            panic!("a change to the stack was not recorded as one");
        };
        let held = &entry.fields[0];
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

        assert!(matches!(
            undone(&mut history, &mut terrain),
            Restored::Fields { reaches_bake: true }
        ));
        assert_eq!(paint_of(&terrain).unwrap().data(), &[7u8; 64][..]);
    }

    // A change to the stack must not move paint: what the brush laid down after that
    // change has to survive it being undone and redone, and only a stroke's own entry
    // ever puts texels back.
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

    // The task's own claim, at the level the history can make it: a stroke goes back to
    // the texels it found, and the redo repaints exactly what it painted. Overlapping
    // strokes are the case a single before-and-after rectangle cannot get right, so the
    // two here overlap.
    #[test]
    fn overlapping_strokes_undo_and_redo_to_the_texels_they_found() {
        let mut terrain = painted(8);
        let mut history = History::default();
        let clean = texels_of(&terrain);

        stroke(&mut terrain, &mut history, rect(0, 4), 1, false);
        let after_first = texels_of(&terrain);
        stroke(&mut terrain, &mut history, rect(2, 6), 2, false);
        let after_second = texels_of(&terrain);
        assert_ne!(after_first, after_second);

        assert!(matches!(
            undone(&mut history, &mut terrain),
            Restored::Stroke { ref field, cells } if field == "height" && cells == rect(2, 6)
        ));
        assert_eq!(texels_of(&terrain), after_first);
        undone(&mut history, &mut terrain);
        assert_eq!(
            texels_of(&terrain),
            clean,
            "the first stroke left a texel behind"
        );

        history.redo(&mut terrain).unwrap();
        assert_eq!(texels_of(&terrain), after_first);
        history.redo(&mut terrain).unwrap();
        assert_eq!(texels_of(&terrain), after_second);
    }

    // What makes a drag one undo step: the pieces a drag is laid down in join the entry
    // it opened, so one Ctrl+Z takes the whole line back rather than one frame of it.
    #[test]
    fn the_pieces_of_one_drag_are_one_entry_undone_in_one_step() {
        let mut terrain = painted(8);
        let mut history = History::default();
        let clean = texels_of(&terrain);

        stroke(&mut terrain, &mut history, rect(0, 4), 1, false);
        stroke(&mut terrain, &mut history, rect(2, 6), 2, true);
        stroke(&mut terrain, &mut history, rect(4, 8), 3, true);
        let drawn = texels_of(&terrain);
        assert_eq!(history.depth().undo, 1, "a drag left more than one entry");

        let Restored::Stroke { cells, .. } = undone(&mut history, &mut terrain) else {
            panic!("a drag did not come back as a stroke");
        };
        assert_eq!(
            cells,
            rect(0, 8),
            "the entry forgot ground the drag covered"
        );
        assert_eq!(texels_of(&terrain), clean);

        history.redo(&mut terrain).unwrap();
        assert_eq!(texels_of(&terrain), drawn);
    }

    // The first stroke into an empty paint node allocates its raster, so undoing it has
    // to leave the node with none — a raster of zeros is a different document from a
    // node that has never been painted. What makes that possible is capturing the patch
    // before the allocation, which is the order the fixture above shares with
    // `apply_stroke`.
    #[test]
    fn undoing_the_stroke_that_allocated_the_raster_leaves_the_node_without_one() {
        let mut terrain = unpainted(8);
        let mut history = History::default();
        stroke(&mut terrain, &mut history, rect(1, 5), 9, false);
        let drawn = texels_of(&terrain);
        assert_eq!(paint_of(&terrain).unwrap().size(), UVec2::splat(8));

        undone(&mut history, &mut terrain);
        assert!(
            paint_of(&terrain).unwrap().is_empty(),
            "a raster of zeros was left where there had been none"
        );

        history.redo(&mut terrain).unwrap();
        assert_eq!(texels_of(&terrain), drawn);
        undone(&mut history, &mut terrain);
        assert!(paint_of(&terrain).unwrap().is_empty());
    }

    // The reason a stroke goes into the same stack as an edit rather than one of its
    // own: they undo in the order they were made, whichever kind came last.
    #[test]
    fn a_stroke_and_an_edit_undo_in_the_order_they_were_made() {
        let mut terrain = painted(8);
        let mut history = History::default();
        let clean = texels_of(&terrain);

        stroke(&mut terrain, &mut history, rect(0, 4), 1, false);
        let after_stroke = texels_of(&terrain);
        edit(&mut terrain, &mut history, 2);

        assert!(matches!(
            undone(&mut history, &mut terrain),
            Restored::Fields { .. }
        ));
        assert_eq!(terrain.field("height").unwrap().shift, 0);
        assert_eq!(
            texels_of(&terrain),
            after_stroke,
            "an edit's undo moved paint"
        );

        assert!(matches!(
            undone(&mut history, &mut terrain),
            Restored::Stroke { .. }
        ));
        assert_eq!(texels_of(&terrain), clean);
    }

    // A piece joins the entry its own drag opened and no other. After a Ctrl+Z that
    // entry is on the redo stack, and joining onto whatever is under it would make one
    // undo step out of two separate strokes.
    #[test]
    fn a_piece_laid_down_after_an_undo_opens_its_own_entry() {
        let mut terrain = painted(8);
        let mut history = History::default();
        stroke(&mut terrain, &mut history, rect(0, 4), 1, false);
        undone(&mut history, &mut terrain);
        let clean = texels_of(&terrain);

        stroke(&mut terrain, &mut history, rect(2, 6), 2, true);
        assert_eq!(history.depth(), HistoryDepth { undo: 1, redo: 0 });

        undone(&mut history, &mut terrain);
        assert_eq!(
            texels_of(&terrain),
            clean,
            "two strokes were merged into one entry"
        );
    }

    // What keeps a painted document's history affordable: an entry holds the texels its
    // pieces covered, not the raster and not the document.
    #[test]
    fn a_stroke_entry_costs_its_footprint_rather_than_the_raster() {
        let mut terrain = painted(64);
        let mut history = History::default();
        stroke(&mut terrain, &mut history, rect(10, 14), 1, false);

        let Change::Stroke(entry) = &history.undo[0] else {
            panic!("a stroke was not recorded as one");
        };
        let held: usize = entry.pieces.iter().map(|piece| piece.texels.len()).sum();
        assert_eq!(held, 16);
        assert!(held * 100 < paint_of(&terrain).unwrap().len());
    }
}
