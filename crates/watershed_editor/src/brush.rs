//! Painting into a document with the pointer.
//!
//! A stroke is an ordinary edit — it writes into a paint layer of a field, and
//! nothing about the field's graph is special-cased for it. What is special is what it
//! tells the document afterwards: a rectangle rather than "this is stale", so a
//! stroke provokes a re-bake of the ground it reached instead of the whole document.
//!
//! A drag outlives the re-bake it provokes. The cells the cursor crosses while the
//! document is busy are queued and laid down as one polyline once it is free, so a
//! drag is a continuous line however many frames its own baking takes; and a drag
//! refused — started over a panel, or aimed at a field with nowhere to paint — is
//! refused once, not once a frame while the button is down. Every piece after the
//! first joins the history entry the drag opened, so the whole drag is one undo step.

use crate::terrain::Field;
use crate::terrain::brush::Brush;
use crate::terrain::graph::{NodeId, NodeOp};
use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use serde_json::{Value, json};
use watershed::raster::{Raster, resolution};

use crate::document::{Document, EditorSystems};
use crate::history::StrokePatch;
use crate::view::{EditorCamera, cell_at_cursor};

/// Runs the pointer-driven painting system, ahead of the document's own systems each
/// frame so a stroke made this frame is baked this frame.
pub struct BrushPlugin;

impl Plugin for BrushPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BrushSettings>()
            .init_resource::<Painting>()
            .configure_sets(Update, EditorSystems::Brush.before(EditorSystems::Document))
            .add_systems(Update, paint.in_set(EditorSystems::Brush));
    }
}

/// The brush as the editor currently has it set.
///
/// A setting of the tool rather than of the document: it survives loading, closing
/// and creating one, because a radius and a strength are how the person is working
/// and not something the document has an opinion about.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct BrushSettings(pub Brush);

#[derive(Resource, Default)]
struct Painting {
    pending: Vec<Vec2>,
    blocked: bool,
    laid: bool,
}

impl Painting {
    fn advance(&mut self, cell: Vec2, busy: bool) -> Option<Vec<Vec2>> {
        if self.pending.last() != Some(&cell) {
            self.pending.push(cell);
        }
        if busy {
            return None;
        }
        let points = std::mem::take(&mut self.pending);
        self.pending = points.last().copied().into_iter().collect();
        Some(points)
    }
}

/// The node a stroke would land in: the one `Paint` node of `field`, or `None` when
/// it has none or has several.
///
/// Derived from the graph every time rather than remembered, so adding or deleting a
/// node moves the target with no separate selection to keep in step. A field with two
/// `Paint` nodes is ambiguous and answers `None` rather than guessing — nothing about
/// a graph makes one of them the obvious target, where the top of a stack once did.
pub fn paint_node(field: &Field) -> Option<NodeId> {
    let mut painted = field
        .graph
        .nodes
        .iter()
        .filter(|node| matches!(node.op, NodeOp::Paint(_)));
    let first = painted.next()?;
    painted.next().is_none().then_some(first.id)
}

/// What the panel names and `observe brush` reports: the field the brush would paint into,
/// and the node of it, or nothing when the active field has none.
pub fn target_of(document: &Document) -> Option<(String, NodeId)> {
    let terrain = document.terrain()?;
    let field = terrain.field(document.active())?;
    paint_node(field).map(|id| (field.id.to_string(), id))
}

/// Applies a stroke, records it in the history and tells the document what it reached —
/// the one path a brush reaches a document by, whether the points came from a drag or
/// from the control client.
///
/// `points` are in document cells. Refused, with a message fit to show, if there are
/// no points, if a job is already running, if there is no document, or if the active
/// field has no single paint node. A stroke that could move no texel at all writes
/// nothing, allocates nothing and records nothing, and answers with no cells.
///
/// `joins` asks for this to extend the history entry the same drag opened rather than
/// to open one; a stroke that starts a drag, and one from the control client, passes
/// `false`.
///
/// The target node's raster is allocated on first use at the field's own resolution,
/// so a texel of the field reads exactly one painted texel and a stroke is never finer
/// than what the field can hold. One that arrived at some other resolution — from a
/// file, or from a shift changed under it — is kept and stretched over the document
/// instead, and the reported rectangle is widened to cover the cells either side of
/// each painted texel.
///
/// The reply names the field, the node, the cells painted and the wider rectangle
/// the change reaches through the fields that read it.
pub fn apply_stroke(
    document: &mut Document,
    brush: &Brush,
    points: &[Vec2],
    joins: bool,
) -> Result<Value, String> {
    if points.is_empty() {
        return Err("a stroke needs somewhere to go".to_owned());
    }
    if document.is_busy() {
        return Err("a job is running".to_owned());
    }
    let name = document.active().to_owned();

    let (id, piece, painted, bleed) = {
        let terrain = document
            .terrain_mut()
            .ok_or("there is no document to paint on")?;
        let size = terrain.size;
        let field = terrain
            .field_mut(&name)
            .ok_or_else(|| format!("no field named `{name}`"))?;
        let id = paint_node(field)
            .ok_or_else(|| format!("`{name}` has no single paint node to paint into"))?;
        let shift = field.shift;
        let node = field.graph.node_mut(id).expect("resolved above");
        let NodeOp::Paint(raster) = &mut node.op else {
            return Err("the brush's target stopped being a paint node".to_owned());
        };
        let texels = if raster.is_empty() {
            resolution(size, shift)
        } else {
            raster.size()
        };
        let Some(footprint) = brush.footprint(texels, size, points) else {
            return Ok(
                json!({ "field": name, "node": crate::edit::node_path(id), "cells": Value::Null }),
            );
        };
        let piece = StrokePatch::before(raster, footprint, texels);
        if raster.is_empty() {
            *raster = Raster::new(texels, 0u8);
        }
        let bleed = (size.x.div_ceil(raster.width().max(1)))
            .max(size.y.div_ceil(raster.height().max(1)))
            + 1;
        (id, piece, brush.stroke(raster, size, points), bleed)
    };

    let painted = painted.expand(bleed);
    let reached = document.record_stroke(&name, id, piece, painted, joins);

    Ok(json!({
        "field": name,
        "node": crate::edit::node_path(id),
        "cells": [painted.min.x, painted.min.y, painted.max.x, painted.max.y],
        "reached": [reached.min.x, reached.min.y, reached.max.x, reached.max.y],
        "points": points.len(),
    }))
}

fn paint(
    buttons: Res<ButtonInput<MouseButton>>,
    hover: Res<HoverMap>,
    nodes: Query<(), With<Node>>,
    window: Option<Single<&Window, With<PrimaryWindow>>>,
    camera: Single<(&Transform, &Projection), With<EditorCamera>>,
    settings: Res<BrushSettings>,
    mut painting: ResMut<Painting>,
    mut document: ResMut<Document>,
) {
    if !buttons.pressed(MouseButton::Left) {
        painting.pending.clear();
        painting.blocked = false;
        painting.laid = false;
        return;
    }
    if painting.blocked {
        return;
    }
    if painting.pending.is_empty() && crate::ui::pointer_over_ui(&hover, &nodes) {
        painting.blocked = true;
        return;
    }
    let Some(window) = window else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let (transform, projection) = camera.into_inner();
    let Some(cell) = cell_at_cursor(
        transform,
        projection,
        Vec2::new(window.width(), window.height()),
        cursor,
        document.size,
    ) else {
        return;
    };

    let Some(points) = painting.advance(cell, document.is_busy()) else {
        return;
    };
    let brush = settings.0;
    match apply_stroke(&mut document, &brush, &points, painting.laid) {
        Ok(_) => painting.laid = true,
        Err(error) => {
            painting.blocked = true;
            warn!("{error}");
            document.refuse(error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::TerrainSpec;
    use crate::terrain::graph::NodeOp;
    use watershed::raster::CellRect;

    fn painted_document() -> Document {
        let mut document = Document::default();
        let terrain = TerrainSpec::new(UVec2::splat(64)).with_field(
            Field::new("height")
                .with_sum([NodeOp::Constant(0.25), NodeOp::Paint(Raster::default())]),
        );
        document.adopt(terrain);
        document
    }

    fn height_texels(document: &Document) -> Vec<u8> {
        let field = document.terrain().unwrap().field("height").unwrap();
        let Some(target) = paint_node(field) else {
            return Vec::new();
        };
        match &field.graph.node(target).unwrap().op {
            NodeOp::Paint(raster) => raster.data().to_vec(),
            _ => Vec::new(),
        }
    }

    fn field_with(ops: Vec<NodeOp>) -> Field {
        ops.into_iter()
            .fold(Field::new("height"), |field, op| field.with_op(op))
    }

    // The one paint node is the target wherever it sits in the graph, since a graph
    // has no top for a stroke to land on.
    #[test]
    fn the_brush_paints_into_the_one_paint_node_of_the_field() {
        let field = field_with(vec![
            NodeOp::Constant(0.5),
            NodeOp::Paint(Raster::default()),
            NodeOp::Constant(0.1),
        ]);
        let painted = field
            .graph
            .nodes
            .iter()
            .find(|node| matches!(node.op, NodeOp::Paint(_)))
            .map(|node| node.id);
        assert_eq!(paint_node(&field), painted);
    }

    // Nothing about a graph makes one of two paint nodes the obvious target, so an
    // ambiguous field has to answer "no target" rather than guess and paint somewhere
    // the person is not looking. A field with none answers the same way.
    #[test]
    fn a_field_with_no_paint_node_and_one_with_two_are_both_no_target() {
        assert_eq!(paint_node(&field_with(vec![NodeOp::Constant(0.5)])), None);
        assert_eq!(
            paint_node(&field_with(vec![
                NodeOp::Paint(Raster::default()),
                NodeOp::Paint(Raster::default()),
            ])),
            None
        );
    }

    // The defect this guards was found by driving the editor: a stroke's own re-bake is
    // still running on the next frame, and a drag that gave up there laid down nothing
    // at all for as long as the button was held. The queue keeps every cell the cursor
    // crossed while it was busy, and the one it was last laid down at, so the next
    // piece joins the last rather than starting beside it.
    #[test]
    fn a_drag_keeps_the_cells_it_crossed_while_the_document_was_busy() {
        let mut painting = Painting::default();
        assert_eq!(
            painting.advance(Vec2::new(0.0, 0.0), false).unwrap().len(),
            1
        );

        assert_eq!(painting.advance(Vec2::new(1.0, 0.0), true), None);
        assert_eq!(painting.advance(Vec2::new(2.0, 0.0), true), None);
        let laid = painting
            .advance(Vec2::new(3.0, 0.0), false)
            .expect("the queue is laid down once the document is free");

        assert_eq!(
            laid,
            vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(2.0, 0.0),
                Vec2::new(3.0, 0.0),
            ]
        );
    }

    // A drag is one undo step: its pieces join the entry it opened, and a stroke from
    // the control client — which passes no drag — is its own.
    #[test]
    fn a_drag_is_one_undo_step_and_an_unjoined_stroke_is_another() {
        let mut document = painted_document();
        let brush = Brush {
            radius_cells: 6.0,
            strength: 0.9,
            ..Brush::default()
        };

        apply_stroke(&mut document, &brush, &[Vec2::splat(20.0)], false).unwrap();
        apply_stroke(&mut document, &brush, &[Vec2::splat(24.0)], true).unwrap();
        apply_stroke(&mut document, &brush, &[Vec2::splat(28.0)], true).unwrap();
        assert_eq!(document.history().undo, 1);

        apply_stroke(&mut document, &brush, &[Vec2::splat(40.0)], false).unwrap();
        assert_eq!(document.history().undo, 2);
    }

    // A stroke that could move no texel — every point off the document — must not
    // allocate the raster, because an allocation nothing recorded cannot be undone.
    #[test]
    fn a_stroke_that_reaches_no_texel_allocates_nothing_and_records_nothing() {
        let mut document = painted_document();
        let reply = apply_stroke(
            &mut document,
            &Brush::default(),
            &[Vec2::splat(4000.0)],
            false,
        )
        .unwrap();

        assert!(reply["cells"].is_null());
        assert_eq!(document.history().undo, 0);
        assert!(
            height_texels(&document).is_empty(),
            "an untracked raster was left behind"
        );
    }

    // A held button reports the same cell every frame; queuing each one would grow the
    // polyline without bound and make a stationary brush behave differently from a
    // moving one.
    #[test]
    fn a_cursor_that_has_not_moved_lays_the_same_cell_down_once() {
        let mut painting = Painting::default();
        painting.advance(Vec2::splat(4.0), false).unwrap();
        assert_eq!(
            painting.advance(Vec2::splat(4.0), false).unwrap(),
            vec![Vec2::splat(4.0)]
        );
        assert_eq!(painting.advance(Vec2::splat(4.0), true), None);
    }

    // A drag is one line, so the piece laid down this frame has to start where the last
    // one ended — a queue that kept nothing would leave the join between them unpainted.
    #[test]
    fn every_piece_of_a_drag_starts_where_the_one_before_it_ended() {
        let mut painting = Painting::default();
        let mut ended = None;
        for step in 0..6 {
            let cell = Vec2::new(step as f32 * 10.0, 0.0);
            let laid = painting.advance(cell, false).expect("nothing is busy");
            if let Some(ended) = ended {
                assert_eq!(laid.first().copied(), Some(ended));
            }
            ended = laid.last().copied();
        }
        assert_eq!(ended, Some(Vec2::new(50.0, 0.0)));
    }

    // Both refusals reach a person as a message, so they have to be errors rather than
    // a silent no-op that looks like a brush with no effect.
    #[test]
    fn a_stroke_is_refused_where_there_is_nothing_to_paint_into() {
        let mut document = Document::default();
        assert!(
            apply_stroke(&mut document, &Brush::default(), &[Vec2::splat(4.0)], false).is_err()
        );
        assert!(apply_stroke(&mut document, &Brush::default(), &[], false).is_err());
    }

    // The seam the whole module is written against: a stroke lands in the layer, and
    // what it reports to the document is a rectangle rather than "everything".
    #[test]
    fn a_stroke_lands_in_the_node_and_leaves_the_bake_where_it_was() {
        let mut document = Document::default();
        let terrain = TerrainSpec::new(UVec2::splat(64)).with_field(
            Field::new("height")
                .with_sum([NodeOp::Constant(0.25), NodeOp::Paint(Raster::default())]),
        );
        document.adopt(terrain);

        let brush = Brush {
            radius_cells: 6.0,
            strength: 0.5,
            ..Brush::default()
        };
        let reply = apply_stroke(&mut document, &brush, &[Vec2::splat(32.0)], false).unwrap();
        assert!(document.is_dirty());

        let field = document.terrain().unwrap().field("height").unwrap();
        let target = paint_node(field).expect("the fixture has one paint node");
        assert_eq!(reply["node"], target.to_string());
        let NodeOp::Paint(raster) = &field.graph.node(target).unwrap().op else {
            panic!("the target stopped being a paint node");
        };
        assert_eq!(raster.size(), UVec2::splat(64));
        assert!(raster.data().iter().any(|byte| *byte > 0));
        assert_eq!(raster.data()[0], 0);

        let cells = &reply["cells"];
        assert!(
            !CellRect::new(
                UVec2::new(
                    cells[0].as_u64().unwrap() as u32,
                    cells[1].as_u64().unwrap() as u32
                ),
                UVec2::new(
                    cells[2].as_u64().unwrap() as u32,
                    cells[3].as_u64().unwrap() as u32
                ),
            )
            .is_empty()
        );
    }
}
