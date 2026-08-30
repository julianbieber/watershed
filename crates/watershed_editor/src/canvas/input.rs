//! Panning, zooming, dragging and selecting on the canvas, and the edits those make.

use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use super::{
    CanvasCameraTag, CanvasFrame, Grab, MAX_SCALE, MIN_SCALE, NodeCard, NodePin, Selection,
    ZOOM_PER_STEP, open_graph,
};
use crate::document::Document;
use crate::edit::Edit;
use crate::terrain::graph::NodeId;
use crate::ui::{pointer_over_ui, report};

/// How far the pointer may travel, in the viewport's own pixels, and still count as a
/// click rather than a drag.
///
/// Measured against where the press landed rather than against the frame before, so a
/// slow drag is still a drag; and in viewport pixels rather than canvas units, so
/// panning the camera under the pointer does not read as pointer movement.
const CLICK_SLOP: f32 = 4.0;

/// What the canvas holds between the frame a drag is finished on and the frame the
/// edit it makes is applied.
#[derive(Resource, Default)]
pub struct Finished(pub Option<Edit>);

/// Pans and zooms the canvas.
///
/// The cursor is converted to canvas space from the camera's own translation and
/// scale rather than through the picking backend, so a pan applied this frame is read
/// back this frame and not next.
pub fn canvas_camera(
    mut wheel: MessageReader<MouseWheel>,
    window: Option<Single<&Window, With<PrimaryWindow>>>,
    frame: Res<CanvasFrame>,
    camera: Option<Single<(&mut Transform, &mut Projection), With<CanvasCameraTag>>>,
    grab: Res<Grab>,
) {
    let (Some(window), Some(camera)) = (window, camera) else {
        wheel.clear();
        return;
    };
    let (mut transform, mut projection) = camera.into_inner();
    let Projection::Orthographic(ortho) = &mut *projection else {
        return;
    };

    let mut steps = 0.0;
    for message in wheel.read() {
        steps += match message.unit {
            MouseScrollUnit::Line => message.y,
            MouseScrollUnit::Pixel => message.y / 32.0,
        };
    }

    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Some(cursor) = canvas_cursor(&window, &frame, cursor) else {
        return;
    };

    if steps != 0.0 {
        let centre = transform.translation.truncate();
        let before = centre + cursor * Vec2::new(ortho.scale, -ortho.scale);
        ortho.scale = (ortho.scale / ZOOM_PER_STEP.powf(steps)).clamp(MIN_SCALE, MAX_SCALE);
        let after = centre + cursor * Vec2::new(ortho.scale, -ortho.scale);
        transform.translation += (before - after).extend(0.0);
    }

    if let Grab::Pan { anchor } = *grab {
        let world =
            transform.translation.truncate() + cursor * Vec2::new(ortho.scale, -ortho.scale);
        transform.translation += (anchor - world).extend(0.0);
    }
}

/// Takes and moves what the pointer is holding, and records the edit that finishes it.
///
/// Writes only the canvas's own state and the card being dragged. The edits it decides
/// on are handed to [`canvas_commit`], which is the one system here that writes the
/// document — so an edit lands in the frame's one pass over it rather than wherever the
/// pointer happened to be read.
pub fn canvas_drag(
    mouse: Res<ButtonInput<MouseButton>>,
    hover: Res<HoverMap>,
    ui: Query<(), With<Node>>,
    window: Option<Single<&Window, With<PrimaryWindow>>>,
    frame: Res<CanvasFrame>,
    document: Res<Document>,
    camera: Option<Single<(&Transform, &Projection), (With<CanvasCameraTag>, Without<NodeCard>)>>,
    mut cards: Query<(Entity, &NodeCard, &mut Transform), Without<CanvasCameraTag>>,
    pins: Query<(&NodePin, &GlobalTransform)>,
    mut grab: ResMut<Grab>,
    mut selection: ResMut<Selection>,
    mut finished: ResMut<Finished>,
    mut pressed_at: Local<Option<Vec2>>,
) {
    let (Some(window), Some(camera)) = (window, camera) else {
        return;
    };
    let (camera_at, projection) = camera.into_inner();
    let Projection::Orthographic(ortho) = projection else {
        return;
    };
    let Some(cursor) = window
        .cursor_position()
        .and_then(|cursor| canvas_cursor(&window, &frame, cursor))
    else {
        return;
    };
    let world = camera_at.translation.truncate() + cursor * Vec2::new(ortho.scale, -ortho.scale);

    if mouse.just_pressed(MouseButton::Left) && !pointer_over_ui(&hover, &ui) {
        let on_pin = pins
            .iter()
            .find(|(_, at)| at.translation().truncate().distance(world) <= super::PIN_RADIUS * 2.0)
            .map(|(pin, _)| *pin);
        let on_card = cards
            .iter()
            .find(|(_, card, at)| {
                Rect::from_center_size(at.translation.truncate(), card.size).contains(world)
            })
            .map(|(entity, card, at)| (entity, card.node, at.translation.truncate()));

        *pressed_at = Some(cursor);
        *grab = match (on_pin, on_card) {
            (Some(pin), _) => {
                selection.select(Some(pin.node));
                Grab::Wire {
                    node: pin.node,
                    pin: pin.input,
                }
            }
            (None, Some((entity, node, at))) => {
                selection.select(Some(node));
                Grab::Card {
                    entity,
                    node,
                    offset: at - world,
                    at,
                }
            }
            (None, None) => Grab::Pan { anchor: world },
        };
    }

    if mouse.pressed(MouseButton::Left)
        && let Grab::Card {
            entity,
            node,
            offset,
            ..
        } = *grab
    {
        let target = world + offset;
        *grab = Grab::Card {
            entity,
            node,
            offset,
            at: target,
        };
        if let Ok((_, _, mut held)) = cards.get_mut(entity) {
            held.translation.x = target.x;
            held.translation.y = target.y;
        }
    }

    if mouse.just_released(MouseButton::Left) {
        let held = std::mem::take(&mut *grab);
        let dragged = pressed_at
            .take()
            .is_some_and(|from| from.distance(cursor) > CLICK_SLOP);
        let active = document.active().to_owned();
        match held {
            Grab::Card { node, at, .. } => {
                // Written from where the drag pulled the card to, and only when that is
                // not where the document already has it — so a click on a card is not
                // an edit, and a move of any size is.
                let recorded = open_graph(&document)
                    .and_then(|graph| graph.node(node))
                    .map(|node| node.position);
                let placed = [at.x, at.y];
                if recorded != Some(placed) {
                    finished.0 = Some(Edit::PlaceNode {
                        field: active,
                        node: node.to_string(),
                        position: placed,
                    });
                }
            }
            Grab::Wire { node, pin } => {
                let landed = pins.iter().find(|(_, at)| {
                    at.translation().truncate().distance(world) <= super::PIN_RADIUS * 2.0
                });
                if let Some((landed, _)) = landed {
                    finished.0 = connection(&document, node, pin, landed).map(|(from, to, pin)| {
                        Edit::Connect {
                            field: active,
                            from: from.to_string(),
                            to: to.to_string(),
                            pin,
                        }
                    });
                }
            }
            Grab::Pan { .. } => {
                if !dragged {
                    selection.select(None);
                }
            }
            Grab::Idle => {}
        }
    }
}

/// Which way round a wire dragged from one pin and released over another goes.
///
/// A wire is drawn in either direction, so this is what decides which end is the
/// source: an edge always leaves an output and arrives at an input, and a release over
/// two outputs or two inputs joins nothing.
fn connection(
    document: &Document,
    from_node: NodeId,
    from_pin: Option<usize>,
    landed: &NodePin,
) -> Option<(NodeId, NodeId, usize)> {
    open_graph(document)?;
    match (from_pin, landed.input) {
        (None, Some(pin)) => Some((from_node, landed.node, pin)),
        (Some(pin), None) => Some((landed.node, from_node, pin)),
        _ => None,
    }
}

/// Applies the edit the last drag finished on.
///
/// The one system here that writes the document, and it runs before the document
/// decides what to bake — so a wire joined this frame is baked this frame rather than
/// next.
pub fn canvas_commit(mut document: ResMut<Document>, mut finished: ResMut<Finished>) {
    let Some(edit) = finished.0.take() else {
        return;
    };
    let result = document.apply(&edit).map(|_| ());
    report(&mut document, result);
}

/// The cursor in the canvas viewport's own pixels, with the origin at its centre and
/// y downwards, as a camera counts a viewport.
///
/// `None` when the pointer is outside the canvas — over the map or over the panel —
/// which is what keeps a drag on one from reaching the other.
fn canvas_cursor(window: &Window, frame: &CanvasFrame, cursor: Vec2) -> Option<Vec2> {
    let scale = window.scale_factor();
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let at = cursor * scale;
    let low = frame.position;
    let high = frame.position + frame.size;
    if at.x < low.x || at.x >= high.x || at.y < low.y || at.y >= high.y {
        return None;
    }
    Some((at - low - frame.size * 0.5) / scale)
}

/// Puts the selected node on the map, and takes it off again.
///
/// A solo lasts as long as the selection that made it — deselecting clears it — so the
/// map at rest is always what the field bakes rather than something left on it and
/// forgotten.
pub fn canvas_solo(keys: Res<ButtonInput<KeyCode>>, mut selection: ResMut<Selection>) {
    if !keys.just_pressed(KeyCode::Digit1) {
        return;
    }
    let Some(node) = selection.node else {
        return;
    };
    selection.soloed = if selection.soloed == Some(node) {
        None
    } else {
        Some(node)
    };
}
