//! Panning, zooming and framing the canvas, and the keys that take a change back.

use bevy::input::keyboard::Key;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::input_focus::InputFocus;
use bevy::prelude::*;
use bevy::text::EditableText;
use bevy::window::PrimaryWindow;

use super::{
    CanvasCameraTag, CanvasFrame, Grab, MAX_SCALE, MIN_SCALE, ZOOM_PER_STEP, frame_canvas,
};
use crate::document::Document;
use crate::ui::{report, typing};

/// How long after a click a second one on the same card still reads as a double-click,
/// in seconds.
pub(super) const DOUBLE_CLICK: f32 = 0.4;

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

/// Ctrl+Z undoes the last change to the document and Ctrl+Shift+Z redoes it, unless a
/// text field has the keyboard.
///
/// Reads the logical key rather than the physical position, so the key printed Z
/// undoes on every layout. Runs first in the canvas chain, ahead of the rebuild, so the
/// cards drawn this frame are the undone document's — and touches the document only on
/// a frame a chord was pressed.
pub fn undo_keys(
    keys: Res<ButtonInput<Key>>,
    focus: Option<Res<InputFocus>>,
    fields: Query<(), With<EditableText>>,
    mut document: ResMut<Document>,
) {
    if typing(focus.as_deref(), &fields) || !keys.pressed(Key::Control) {
        return;
    }
    let z = keys
        .get_just_pressed()
        .any(|key| matches!(key, Key::Character(c) if c.eq_ignore_ascii_case("z")));
    if !z {
        return;
    }
    let result = if keys.pressed(Key::Shift) {
        document.redo()
    } else {
        document.undo()
    };
    report(&mut document, result);
}

/// F frames every layer card, unless a text field has the keyboard.
///
/// Reads the logical key rather than the physical position, so the key printed F fits
/// on every layout. Does nothing while a pan is held: the pan keeps its anchor under
/// the cursor every frame, so it would drag the camera straight back off the frame.
pub fn fit_key(
    keys: Res<ButtonInput<Key>>,
    focus: Option<Res<InputFocus>>,
    fields: Query<(), With<EditableText>>,
    grab: Res<Grab>,
    document: Res<Document>,
    frame: Res<CanvasFrame>,
    window: Option<Single<&Window, With<PrimaryWindow>>>,
    camera: Option<Single<(&mut Transform, &mut Projection), With<CanvasCameraTag>>>,
) {
    if typing(focus.as_deref(), &fields) || !matches!(*grab, Grab::Idle) {
        return;
    }
    let pressed = keys
        .get_just_pressed()
        .any(|key| matches!(key, Key::Character(c) if c.eq_ignore_ascii_case("f")));
    if !pressed {
        return;
    }
    let (Some(window), Some(camera)) = (window, camera) else {
        return;
    };
    let (mut transform, mut projection) = camera.into_inner();
    frame_canvas(
        &document,
        &frame,
        window.into_inner(),
        &mut transform,
        &mut projection,
    );
}

/// The cursor in the canvas viewport's own pixels, with the origin at its centre and
/// y downwards, as a camera counts a viewport.
///
/// `None` when the pointer is outside the canvas — over the map or over the panel —
/// which is what keeps a drag on one from reaching the other.
pub(super) fn canvas_cursor(window: &Window, frame: &CanvasFrame, cursor: Vec2) -> Option<Vec2> {
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
