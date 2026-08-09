// TODO(jb-doc): module docs — that the brush writes into a layer like any other edit, and
// why a stroke is the one edit that names a rectangle instead of dropping the whole bake.

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::input::EguiWantsInput;
use serde_json::{Value, json};
use watershed::Field;
use watershed::brush::Brush;
use watershed::layer::LayerOp;
use watershed::raster::{Raster, resolution};

use crate::document::{Document, EditorSystems};
use crate::view::{EditorCamera, cell_at_cursor};

pub struct BrushPlugin;

impl Plugin for BrushPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BrushSettings>()
            .init_resource::<Painting>()
            .configure_sets(Update, EditorSystems::Brush.before(EditorSystems::Document))
            .add_systems(Update, paint.in_set(EditorSystems::Brush));
    }
}

/// TODO(jb-doc): why the brush is a knob that outlives a document, on the same terms the
/// new-terrain dialog's fields are.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct BrushSettings(pub Brush);

/// The drag in progress: the cells it has passed through that have not been laid down yet,
/// starting with the last one that was.
///
/// Cells rather than pixels, because a stroke is written in document cells and the two part
/// company the moment the camera moves. A queue rather than one position, because the bake a
/// stroke provokes takes frames and the cursor does not wait for it — see [`paint`].
#[derive(Resource, Default)]
struct Painting {
    pending: Vec<Vec2>,
    /// This drag belongs to something else — a panel it started over, or a refusal it has
    /// already reported. Held until the button comes up rather than tested again each frame.
    blocked: bool,
}

impl Painting {
    /// One frame of a drag: where the cursor is, and whether the document can take a stroke
    /// this frame. Answers with the points to lay down, or with nothing while it cannot.
    ///
    /// TODO(jb-doc): what a drag laid down by this is and is not — that the cells are exact
    /// and the *amount* at a join is not, and which of the two a person notices.
    fn advance(&mut self, cell: Vec2, busy: bool) -> Option<Vec<Vec2>> {
        if self.pending.last() != Some(&cell) {
            self.pending.push(cell);
        }
        if busy {
            return None;
        }
        let points = std::mem::take(&mut self.pending);
        // The last cell stays behind as the next segment's start, or the drag would be laid
        // down as a row of disconnected pieces with the joins between them unpainted.
        self.pending = points.last().copied().into_iter().collect();
        Some(points)
    }
}

/// The layer a stroke lands in: the topmost enabled paint layer of the field on screen.
///
/// TODO(jb-doc): why the target is derived every stroke rather than being a selection the
/// panel and the ctl would each have to keep in step with a stack that can be reordered.
pub fn paint_layer(field: &Field) -> Option<usize> {
    field
        .layers
        .iter()
        .enumerate()
        .rev()
        .find(|(_, layer)| layer.enabled && matches!(layer.op, LayerOp::Paint(_)))
        .map(|(index, _)| index)
}

/// What the panel names and `observe brush` reports: the field the brush would paint into,
/// and the layer of it, or nothing when the active field has none.
pub fn target_of(document: &Document) -> Option<(String, usize)> {
    let terrain = document.terrain()?;
    let field = terrain.field(document.active())?;
    paint_layer(field).map(|index| (field.id.to_string(), index))
}

/// A stroke, applied and noted — the one path a brush reaches the document by, whether the
/// points came from a drag or from the ctl.
pub fn apply_stroke(
    document: &mut Document,
    brush: &Brush,
    points: &[Vec2],
) -> Result<Value, String> {
    if points.is_empty() {
        return Err("a stroke needs somewhere to go".to_owned());
    }
    if document.is_busy() {
        return Err("a job is running".to_owned());
    }
    let name = document.active().to_owned();

    let (index, painted, bleed) = {
        let terrain = document
            .terrain_mut()
            .ok_or("there is no document to paint on")?;
        let size = terrain.size;
        let field = terrain
            .field_mut(&name)
            .ok_or_else(|| format!("no field named `{name}`"))?;
        let index = paint_layer(field)
            .ok_or_else(|| format!("`{name}` has no enabled paint layer to paint into"))?;
        let shift = field.shift;
        let LayerOp::Paint(raster) = &mut field.layers[index].op else {
            return Err("the brush's target stopped being a paint layer".to_owned());
        };
        // One texel per texel of the field it is a layer of, so a texel of the field reads
        // exactly one texel of this and a stroke cannot be finer than what the field holds.
        if raster.is_empty() {
            *raster = Raster::new(resolution(size, shift), 0.0);
        }
        // A raster that arrived at some other resolution — from a file, or from a shift
        // changed under it — is stretched over the document instead, so a painted texel is
        // read from a cell either side of the block it covers.
        let bleed = (size.x.div_ceil(raster.width().max(1)))
            .max(size.y.div_ceil(raster.height().max(1)))
            + 1;
        (index, brush.stroke(raster, size, points), bleed)
    };

    if painted.is_empty() {
        return Ok(json!({ "field": name, "layer": index, "cells": Value::Null }));
    }
    let painted = painted.expand(bleed);

    let reached = match document.terrain() {
        Some(terrain) => terrain.influence_of(&name, painted),
        None => painted,
    };
    document.note_stroke(reached);

    Ok(json!({
        "field": name,
        "layer": index,
        "cells": [painted.min.x, painted.min.y, painted.max.x, painted.max.y],
        "reached": [reached.min.x, reached.min.y, reached.max.x, reached.max.y],
        "points": points.len(),
    }))
}

/// TODO(jb-comment): why egui is asked only on the frame a stroke starts, and what a drag
/// that wandered over the panel would otherwise paint.
fn paint(
    buttons: Res<ButtonInput<MouseButton>>,
    wants: Option<Res<EguiWantsInput>>,
    window: Option<Single<&Window, With<PrimaryWindow>>>,
    camera: Single<(&Transform, &Projection), With<EditorCamera>>,
    settings: Res<BrushSettings>,
    mut painting: ResMut<Painting>,
    mut document: ResMut<Document>,
) {
    if !buttons.pressed(MouseButton::Left) {
        painting.pending.clear();
        painting.blocked = false;
        return;
    }
    if painting.blocked {
        return;
    }
    if painting.pending.is_empty()
        && wants
            .as_deref()
            .is_some_and(EguiWantsInput::wants_any_pointer_input)
    {
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

    // The whole reason the queue exists: a stroke's own re-bake is still in flight on the
    // frame after it, so a drag that painted only when the document was free would lay down
    // every other frame — and one that *refused* there would lose the cells in between.
    let Some(points) = painting.advance(cell, document.is_busy()) else {
        return;
    };
    let brush = settings.0;
    match apply_stroke(&mut document, &brush, &points) {
        Ok(_) => {}
        Err(error) => {
            // The refusal a person can see, on the same terms the panel's buttons report
            // one — once for the drag, rather than once a frame while the button is held.
            painting.blocked = true;
            warn!("{error}");
            document.refuse(error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use watershed::Terrain;
    use watershed::layer::Layer;
    use watershed::raster::CellRect;

    fn field_with(ops: Vec<LayerOp>) -> Field {
        ops.into_iter().fold(Field::new("height"), |field, op| {
            field.with_layer(Layer::new(op))
        })
    }

    #[test]
    fn the_brush_paints_into_the_topmost_paint_layer_of_the_field() {
        let field = field_with(vec![
            LayerOp::Constant(0.5),
            LayerOp::Paint(Raster::default()),
            LayerOp::Constant(0.1),
            LayerOp::Paint(Raster::default()),
        ]);
        assert_eq!(paint_layer(&field), Some(3));
    }

    #[test]
    fn a_field_with_no_paint_layer_and_one_that_is_switched_off_are_both_no_target() {
        assert_eq!(paint_layer(&field_with(vec![LayerOp::Constant(0.5)])), None);

        let mut field = field_with(vec![LayerOp::Paint(Raster::default())]);
        field.layers[0].enabled = false;
        assert_eq!(paint_layer(&field), None);
    }

    /// The defect this guards was found by driving the editor: a stroke's own re-bake is
    /// still running on the next frame, and a drag that gave up there laid down nothing at
    /// all for as long as the button was held.
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

        // Every cell the cursor crossed while it was busy, and the one it was last laid
        // down at — so the piece joins the one before it rather than starting beside it.
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

    /// A drag is one line, so the piece laid down this frame has to start where the last one
    /// ended — a queue that kept nothing would leave the join between them unpainted.
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

    #[test]
    fn a_stroke_is_refused_where_there_is_nothing_to_paint_into() {
        let mut document = Document::default();
        assert!(apply_stroke(&mut document, &Brush::default(), &[Vec2::splat(4.0)]).is_err());
        assert!(apply_stroke(&mut document, &Brush::default(), &[]).is_err());
    }

    /// The seam the whole module is written against: a stroke lands in the layer, and what
    /// it reports to the document is a rectangle rather than "everything".
    #[test]
    fn a_stroke_lands_in_the_layer_and_leaves_the_bake_where_it_was() {
        let mut document = Document::default();
        let terrain = Terrain::new(UVec2::splat(64)).with_field(
            Field::new("height")
                .with_layer(Layer::new(LayerOp::Constant(0.25)))
                .with_layer(Layer::new(LayerOp::Paint(Raster::default()))),
        );
        document.adopt(terrain);

        let brush = Brush {
            radius_cells: 6.0,
            strength: 0.5,
            ..Brush::default()
        };
        let reply = apply_stroke(&mut document, &brush, &[Vec2::splat(32.0)]).unwrap();
        assert_eq!(reply["layer"], 1);
        assert!(document.is_dirty());

        let LayerOp::Paint(raster) =
            &document.terrain().unwrap().field("height").unwrap().layers[1].op
        else {
            panic!("the target stopped being a paint layer");
        };
        assert_eq!(raster.size(), UVec2::splat(64));
        assert!(*raster.get(32, 32).unwrap() > 0.0);
        assert!(*raster.get(0, 0).unwrap() == 0.0);

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
