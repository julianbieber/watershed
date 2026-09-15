//! The document's fields as a thing on screen: one card per field with ribbons for what
//! each reads, and how that canvas is panned, zoomed and framed.

mod edges;
mod input;
mod overview;
mod thumb;

use bevy::camera::visibility::RenderLayers;
use bevy::camera::{Camera, ClearColorConfig, Viewport};
use bevy::prelude::*;
use bevy::ui::UiSystems;

use crate::document::{Document, EditorSystems};

/// The width and height of a card, in canvas units.
pub const CARD: Vec2 = Vec2::new(240.0, 140.0);
const TITLE_BAR: f32 = 30.0;
const THUMB: f32 = 78.0;
const MARGIN: f32 = 8.0;
const ROW_STEP: f32 = 19.0;
const TITLE_SIZE: f32 = 17.0;
const DETAIL_SIZE: f32 = 12.0;
const PARAM_SIZE: f32 = 11.0;
const ZOOM_PER_STEP: f32 = 1.2;
const MIN_SCALE: f32 = 0.05;
const MAX_SCALE: f32 = 8.0;

const CHIP_ZOOM: f32 = 0.40;

const CANVAS_LAYER: usize = 1;

const CANVAS_GROUND: Color = Color::srgb(0.115, 0.125, 0.165);

/// Draws the document's fields, and keeps the drawing in step with the document.
pub struct CanvasPlugin;

impl Plugin for CanvasPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Grab>()
            .init_resource::<CanvasFrame>()
            .init_resource::<CanvasShape>()
            .add_message::<OpenField>()
            .add_systems(Startup, spawn_canvas_camera)
            .add_systems(
                Update,
                (
                    apply_open_field,
                    input::undo_keys,
                    overview::rebuild_overview,
                    frame_overview,
                    thumb::sync_thumbnails,
                    input::canvas_camera,
                    input::fit_key,
                    overview::overview_drag,
                    overview::route_field_ribbons,
                    scale_canvas_labels,
                )
                    .chain()
                    .before(EditorSystems::Document),
            )
            .add_systems(PostUpdate, size_canvas_viewport.after(UiSystems::Layout));
    }
}

/// The camera the canvas is drawn through. Orthographic, and the only camera that
/// draws [`CANVAS_LAYER`].
#[derive(Component)]
pub struct CanvasCameraTag;

/// A world-space label, and the height it is meant to read at.
///
/// World-space text is rasterised once and then scaled by the camera, so a label left
/// alone blurs as it is zoomed past 1:1. The height kept here is what
/// [`scale_canvas_labels`] rasterises against the current zoom.
#[derive(Component)]
pub struct CanvasLabel(pub f32);

/// Asks for another field of the open document to be the active one.
///
/// The one way anything asks for a field change: the canvas's double-click, the
/// panel's `reads` and `read by` buttons, and whatever else comes to want it all write
/// this rather than reaching for the document themselves. It is a view change and not
/// an edit — nothing is written to the terrain, no history entry is made, and no bake
/// is started — so a field opened this way can be shut again by opening the first one.
///
/// A name no field of the document carries is refused the way any other bad reference
/// is: the refusal is reported and the active field stays where it was.
#[derive(Message)]
pub struct OpenField {
    /// The field to open. Must be one the document carries.
    pub field: String,
}

fn apply_open_field(mut open: MessageReader<OpenField>, mut document: ResMut<Document>) {
    for message in open.read() {
        let opened = document.set_active(&message.field);
        crate::ui::report(&mut document, opened);
    }
}

/// What the pointer is currently doing on the canvas.
#[derive(Resource, Default)]
pub enum Grab {
    /// Nothing is held.
    #[default]
    Idle,
    /// The canvas is being panned. The anchor is the canvas point held under the
    /// cursor.
    Pan {
        /// The canvas point the drag started at.
        anchor: Vec2,
    },
}

/// The strip of the window the canvas is drawn into.
///
/// A UI node with nothing in it: it exists so the layout reserves the space and the
/// canvas camera's viewport can be measured from something the layout agrees with,
/// rather than from a fraction computed twice.
#[derive(Component, Default, Clone)]
pub struct CanvasViewport;

/// The rectangle the layout gave the canvas, in physical pixels from the window's top
/// left.
///
/// Written by the UI once the layout has run and read by the camera and by every hit
/// test, so the space reserved for the canvas and the space it draws into are one
/// answer rather than two.
///
/// The pan and the zoom are deliberately absent: they live on the camera itself,
/// because the cursor is converted to canvas space from the camera's own translation
/// and scale.
#[derive(Resource)]
pub struct CanvasFrame {
    /// The top left corner, in physical pixels.
    pub position: Vec2,
    /// Width and height, in physical pixels. Never zero on either axis.
    pub size: Vec2,
}

impl Default for CanvasFrame {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            size: Vec2::ONE,
        }
    }
}

/// What the canvas was last built from, and whether it has been framed since fields
/// appeared.
#[derive(Resource, Default)]
pub struct CanvasShape {
    /// The fingerprint the cards on screen were built from.
    pub key: String,
    /// Whether the camera has framed the cards since the document last had fields, so
    /// it frames them once rather than fighting a pan the person made afterwards.
    pub framed: bool,
}

fn spawn_canvas_camera(mut commands: Commands) {
    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            clear_color: ClearColorConfig::Custom(CANVAS_GROUND),
            ..default()
        },
        Projection::Orthographic(OrthographicProjection {
            scale: 1.0,
            ..OrthographicProjection::default_2d()
        }),
        RenderLayers::layer(CANVAS_LAYER),
        CanvasCameraTag,
    ));
}

fn size_canvas_viewport(
    frame: Res<CanvasFrame>,
    window: Option<Single<&Window, With<bevy::window::PrimaryWindow>>>,
    camera: Option<Single<&mut Camera, With<CanvasCameraTag>>>,
) {
    let (Some(camera), Some(window)) = (camera, window) else {
        return;
    };
    let target = UVec2::new(window.physical_width(), window.physical_height());
    if target.x == 0 || target.y == 0 {
        return;
    }
    let mut camera = camera.into_inner();
    let position = frame
        .position
        .max(Vec2::ZERO)
        .as_uvec2()
        .min(target - UVec2::ONE);
    let size = frame
        .size
        .max(Vec2::ONE)
        .as_uvec2()
        .min(target - position)
        .max(UVec2::ONE);
    let wanted = Viewport {
        physical_position: position,
        physical_size: size,
        depth: 0.0..1.0,
    };
    if camera
        .viewport
        .as_ref()
        .is_none_or(|held| held.physical_position != position || held.physical_size != size)
    {
        camera.viewport = Some(wanted);
    }
}

fn frame_overview(
    document: Res<Document>,
    frame: Res<CanvasFrame>,
    window: Option<Single<&Window, With<bevy::window::PrimaryWindow>>>,
    mut shape: ResMut<CanvasShape>,
    camera: Option<Single<(&mut Transform, &mut Projection), With<CanvasCameraTag>>>,
) {
    let has_fields = document
        .terrain()
        .is_some_and(|terrain| !terrain.fields.is_empty());
    if !has_fields {
        if shape.framed {
            shape.framed = false;
        }
        return;
    }
    if shape.framed {
        return;
    }
    let (Some(camera), Some(window)) = (camera, window) else {
        return;
    };
    let (mut transform, mut projection) = camera.into_inner();
    if frame_canvas(
        &document,
        &frame,
        window.into_inner(),
        &mut transform,
        &mut projection,
    ) {
        shape.framed = true;
    }
}

/// Puts every field card in view on the canvas camera.
///
/// What the canvas's Fit button and the key F both do. Answers whether the camera was
/// moved: `false` when there is no document, when the document has no fields, and when
/// the canvas has not been measured yet.
pub fn frame_canvas(
    document: &Document,
    frame: &CanvasFrame,
    window: &Window,
    transform: &mut Transform,
    projection: &mut Projection,
) -> bool {
    let scale_factor = window.scale_factor();
    if !scale_factor.is_finite() || scale_factor <= 0.0 {
        return false;
    }
    let Projection::Orthographic(ortho) = projection else {
        return false;
    };
    let viewport = frame.size / scale_factor;
    let Some((centre, scale)) = document
        .terrain()
        .and_then(|terrain| overview::overview_fit(terrain, viewport))
    else {
        return false;
    };
    ortho.scale = scale;
    transform.translation.x = centre.x;
    transform.translation.y = centre.y;
    true
}

fn scale_canvas_labels(
    camera: Option<Single<&Projection, With<CanvasCameraTag>>>,
    mut labels: Query<(&CanvasLabel, &mut TextFont, &mut Transform, &mut Visibility)>,
    added: Query<(), Added<CanvasLabel>>,
    mut applied: Local<Option<f32>>,
) {
    let Some(camera) = camera else {
        return;
    };
    let Projection::Orthographic(ortho) = camera.into_inner() else {
        return;
    };
    if *applied == Some(ortho.scale) && added.is_empty() {
        return;
    }
    *applied = Some(ortho.scale);

    let readable = labels_readable(ortho.scale);
    let shown = if readable {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for (label, mut font, mut transform, mut visibility) in &mut labels {
        *visibility = shown;
        if !readable {
            continue;
        }
        let size = (label.0 / ortho.scale).clamp(6.0, 192.0).round();
        font.font_size = bevy::text::FontSize::Px(size);
        transform.scale = Vec3::splat(label.0 / size);
    }
}

/// Whether the pointer is over the canvas rather than the map.
///
/// The question both views ask before they take a wheel event: the map and the canvas
/// zoom independently, so a scroll has to land on exactly one of them.
pub fn pointer_over_canvas(window: &Window, frame: &CanvasFrame) -> bool {
    let Some(cursor) = window.cursor_position() else {
        return false;
    };
    let scale = window.scale_factor();
    if !scale.is_finite() || scale <= 0.0 {
        return false;
    }
    let at = cursor * scale;
    let high = frame.position + frame.size;
    at.x >= frame.position.x && at.x < high.x && at.y >= frame.position.y && at.y < high.y
}

fn labels_readable(scale: f32) -> bool {
    1.0 / scale >= CHIP_ZOOM
}

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;

    use super::*;
    use crate::terrain::{Field, TerrainSpec};

    fn open_field_world() -> World {
        let mut document = Document::default();
        document.adopt(
            TerrainSpec::new(UVec2::splat(16))
                .with_field(Field::new("base").held(0.25))
                .with_field(Field::new("height").held(0.5)),
        );
        document.set_active("height").unwrap();

        let mut world = World::new();
        world.insert_resource(document);
        world.init_resource::<Messages<OpenField>>();
        world
    }

    // A card rebuilt while zoomed out must not put its labels back on screen until the
    // next zoom.
    #[test]
    fn a_label_spawned_below_the_chip_zoom_is_hidden() {
        let mut world = World::new();
        world.spawn((
            CanvasCameraTag,
            Projection::Orthographic(OrthographicProjection {
                scale: 2.0 / CHIP_ZOOM,
                ..OrthographicProjection::default_2d()
            }),
        ));
        let system = world.register_system(scale_canvas_labels);
        world.run_system(system).unwrap();

        let label = world
            .spawn((
                CanvasLabel(TITLE_SIZE),
                TextFont::default(),
                Transform::default(),
                Visibility::Inherited,
            ))
            .id();
        world.run_system(system).unwrap();

        assert_eq!(world.get::<Visibility>(label), Some(&Visibility::Hidden));
    }

    // Opening a field is a view change, so the active field moves while the history and
    // the dirty flag stand still — otherwise following a `reads` link would make a
    // document that has to be saved.
    #[test]
    fn opening_a_field_moves_the_view_without_making_an_edit() {
        let mut world = open_field_world();
        let before = world.resource::<Document>().history();
        world.write_message(OpenField {
            field: "base".to_owned(),
        });
        world.run_system_once(apply_open_field).unwrap();

        let document = world.resource::<Document>();
        assert_eq!(document.active(), "base");
        assert_eq!(document.history().undo, before.undo);
        assert_eq!(document.history().redo, before.redo);
        assert!(!document.is_dirty());
    }

    // A name no field carries has to leave the active field where it was and say so,
    // because the panel and the canvas both write this message from names they read off
    // a document that may have moved under them.
    #[test]
    fn opening_a_field_that_is_not_there_leaves_the_view_alone() {
        let mut world = open_field_world();
        world.write_message(OpenField {
            field: "nowhere".to_owned(),
        });
        world.run_system_once(apply_open_field).unwrap();

        let document = world.resource::<Document>();
        assert_eq!(document.active(), "height");
        assert!(document.error().is_some(), "the refusal was not reported");
    }
}
