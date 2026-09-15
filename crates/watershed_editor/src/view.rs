//! Showing a document: the quad it is drawn on, the camera over it, and the
//! conversions between what is on screen and what is in the document.
//!
//! One document cell is one world unit and the quad is centred on the origin, so the
//! camera's scale is *cells per unit of projection area* — a scale of 1 puts one cell
//! in the space one cell would take at the default zoom, and everything that talks
//! about the view talks in cells rather than in pixels.

use crate::terrain::WaterState;
use bevy::asset::RenderAssetUsages;
use bevy::camera::CameraUpdateSystems;
use bevy::image::{ImageSampler, TextureFormatPixelInfo};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::input_focus::InputFocus;
use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::sprite_render::MeshMaterial2d;
use bevy::text::EditableText;
use bevy::ui::IsDefaultUiCamera;
use watershed::CellRect;

use crate::document::{Document, EditorSystems};
use crate::material::{FieldMaterial, FieldMaterialPlugin, FieldSettings};

/// How much accumulated flow makes a cell a channel, for the overlay and for anything
/// asking the editor what it is showing.
///
/// One number for both: a caller asserting on channels has to be asking about the
/// channels a person can see, and two thresholds would let the picture and the answer
/// disagree.
pub const CHANNEL_THRESHOLD: f32 = 64.0;

const FIT_SAMPLES: u32 = 24;

const FIT_TRIM: f32 = 1.0 / 40.0;

const MIN_SPAN: f32 = 1.0 / 512.0;

const PAN_CELLS_PER_SECOND: f32 = 600.0;
const ZOOM_PER_STEP: f32 = 1.2;
const MIN_CELLS_ACROSS: f32 = 8.0;

/// Spawns the camera and the quad the document is drawn on, and runs the systems that
/// keep the picture, the fitted ramp and the visible rectangle in step with it.
pub struct ViewPlugin;

impl Plugin for ViewPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FieldMaterialPlugin)
            .init_resource::<ViewRange>()
            .init_resource::<VisibleCells>()
            .init_resource::<FreeView>()
            .add_systems(Startup, spawn_view)
            .add_systems(
                Update,
                (pan_zoom, sync_maps)
                    .chain()
                    .in_set(EditorSystems::View)
                    .after(EditorSystems::Document),
            )
            .add_systems(
                PostUpdate,
                (fit_ramp, track_visible_cells).after(CameraUpdateSystems),
            );
    }
}

/// The camera the document is viewed through. Orthographic, and the only camera the
/// view systems will act on.
#[derive(Component)]
pub struct EditorCamera;

#[derive(Component)]
struct FieldQuad;

/// What the colour ramp is currently fitted to.
///
/// Written once a frame from what is on screen, and read by everything that has to
/// agree with the picture — the legend, and anything asking the editor what it is
/// showing. Deriving it a second time would be a second answer to one question.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct ViewRange {
    /// The value the ramp's low end stands for, in the field's own units.
    pub low: f32,
    /// The value the ramp's high end stands for.
    pub high: f32,
    /// Whether the diverging ramp is in use, which follows from `low` and `high`
    /// straddling zero.
    pub diverging: bool,
}

/// The document cells the camera can see, written once a frame beside [`ViewRange`]
/// and for the same reason: the re-bake and the screen have to be answering one
/// question.
///
/// Rounded outwards and clipped to the document, so a cell only half on screen is
/// inside it. A rectangle one cell short of the view would leave the outermost row of
/// cells unbaked after an edit — a stale fringe along the edge of the screen that
/// moves whenever the camera does.
#[derive(Resource, Clone, Copy, Debug)]
pub struct VisibleCells(pub CellRect);

impl Default for VisibleCells {
    fn default() -> Self {
        Self(CellRect::EMPTY)
    }
}

/// What the panels have left the world, as fractions of the window.
///
/// The world is drawn across the whole window with the panels laid on top of it, so
/// the only thing that has to know they are there is [`fit_camera`]. Both fractions
/// are needed: the size says how much of the document fits, and the centre says where
/// to put it — a fit using the size alone would centre the document on the window and
/// leave a strip of it behind the toolbar and the panel.
#[derive(Resource, Clone, Copy, Debug)]
pub struct FreeView {
    /// The uncovered fraction of the window's width and height.
    pub size: Vec2,
    /// Where the uncovered rectangle's centre sits against the window's, as a fraction of
    /// the window and counted the way the camera counts: y upwards.
    pub centre: Vec2,
}

impl Default for FreeView {
    fn default() -> Self {
        Self {
            size: Vec2::ONE,
            centre: Vec2::ZERO,
        }
    }
}

impl FreeView {
    /// The fractions `free` is of `window`.
    ///
    /// A degenerate window or rectangle gives the whole window rather than itself,
    /// which is also what a caller gets for the frames before a layout has run.
    pub fn new(free: Rect, window: Vec2) -> Self {
        if window.x <= 0.0 || window.y <= 0.0 || free.width() <= 0.0 || free.height() <= 0.0 {
            return Self::default();
        }
        let centre = free.center();
        Self {
            size: Vec2::new(free.width() / window.x, free.height() / window.y),
            centre: Vec2::new(
                (centre.x - window.x * 0.5) / window.x,
                (window.y * 0.5 - centre.y) / window.y,
            ),
        }
    }
}

#[derive(Component)]
struct MapRevisions {
    field: Option<(u64, u64)>,
    water: Option<u64>,
}

fn blank(format: TextureFormat) -> Image {
    let mut image = Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![
            0u8;
            format
                .pixel_size()
                .expect("a blank map is only ever made of formats with a pixel size")
        ],
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::nearest();
    image
}

fn spawn_view(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<FieldMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let field = images.add(blank(TextureFormat::R32Float));
    let water = images.add(blank(TextureFormat::Rg8Unorm));

    let material = materials.add(FieldMaterial {
        settings: FieldSettings::default(),
        field,
        water,
    });

    commands.spawn((
        Camera2d,
        Projection::Orthographic(OrthographicProjection {
            scale: 1.0,
            ..OrthographicProjection::default_2d()
        }),
        IsDefaultUiCamera,
        EditorCamera,
    ));

    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(1.0, 1.0))),
        MeshMaterial2d(material),
        Transform::default(),
        FieldQuad,
        MapRevisions {
            field: None,
            water: None,
        },
    ));
}

fn sync_maps(
    document: Res<Document>,
    solo: Res<crate::canvas::Solo>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<FieldMaterial>>,
    mut images: ResMut<Assets<Image>>,
    quad: Single<(&Mesh2d, &MeshMaterial2d<FieldMaterial>, &mut MapRevisions), With<FieldQuad>>,
) {
    let (mesh, material, mut revisions) = quad.into_inner();
    let Some(mut material) = materials.get_mut(&material.0) else {
        return;
    };
    let Some(terrain) = document.terrain() else {
        return;
    };

    if let Some(field) = terrain.field(document.active()) {
        material.settings.hillshade = if field.hillshade { 1.0 } else { 0.0 };
        material.settings.light_azimuth = field.light_azimuth;
        material.settings.contours = if field.contours { 1.0 } else { 0.0 };
        material.settings.contour_interval = field.contour_interval;
    }

    if revisions.field != Some((document.revision(), solo.generation)) {
        revisions.field = Some((document.revision(), solo.generation));

        let Some(field) = terrain.field(document.active()) else {
            return;
        };
        let baked = solo.raster.as_ref().unwrap_or(field.baked());

        if let Some(mut mesh) = meshes.get_mut(&mesh.0) {
            *mesh = Rectangle::new(terrain.size.x as f32, terrain.size.y as f32).into();
        }

        let resolution = if baked.is_empty() {
            UVec2::ONE
        } else {
            baked.size()
        };
        material.settings.field_resolution = resolution.as_vec2();
        material.settings.document_size = terrain.size.as_vec2();

        if baked.is_empty() {
            material.field = images.add(blank(TextureFormat::R32Float));
        } else {
            let mut bytes = Vec::with_capacity(baked.len() * size_of::<f32>());
            for value in baked.data() {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
            let mut image = Image::new(
                Extent3d {
                    width: resolution.x,
                    height: resolution.y,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                bytes,
                TextureFormat::R32Float,
                RenderAssetUsages::RENDER_WORLD,
            );
            image.sampler = ImageSampler::nearest();
            material.field = images.add(image);
        }
    }

    if revisions.water != Some(document.water_revision()) {
        revisions.water = Some(document.water_revision());
        material.water = images.add(match terrain.water() {
            Some(state) => water_map(state),
            None => blank(TextureFormat::Rg8Unorm),
        });
    }
}

fn water_map(state: &WaterState) -> Image {
    let size = state.size();
    let mut bytes = Vec::with_capacity((size.x * size.y) as usize * 2);
    for y in 0..size.y {
        for x in 0..size.x {
            bytes.push(if state.is_water(x, y) { 255 } else { 0 });
            bytes.push(if state.channel(x, y, CHANNEL_THRESHOLD) {
                255
            } else {
                0
            });
        }
    }

    let mut image = Image::new(
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rg8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::nearest();
    image
}

fn visible_rect(camera: &Transform, projection: &OrthographicProjection) -> Rect {
    let centre = camera.translation.truncate();
    Rect::from_corners(centre + projection.area.min, centre + projection.area.max)
}

fn world_to_cell(world: Vec2, size: UVec2) -> Vec2 {
    world + size.as_vec2() * 0.5
}

fn cell_to_world(cell: Vec2, size: UVec2) -> Vec2 {
    cell - size.as_vec2() * 0.5
}

fn pan_zoom(
    keys: Res<ButtonInput<KeyCode>>,
    focus: Option<Res<InputFocus>>,
    fields: Query<(), With<EditableText>>,
    hover: Res<HoverMap>,
    nodes: Query<(), With<Node>>,
    mut wheel: MessageReader<MouseWheel>,
    time: Res<Time>,
    frame: Res<crate::canvas::CanvasFrame>,
    window: Option<Single<&Window, With<bevy::window::PrimaryWindow>>>,
    camera: Single<(&mut Transform, &mut Projection), With<EditorCamera>>,
) {
    let (mut transform, mut projection) = camera.into_inner();
    let Projection::Orthographic(projection) = &mut *projection else {
        return;
    };

    let typing = crate::ui::typing(focus.as_deref(), &fields);

    let mut direction = Vec2::ZERO;
    if !typing {
        if keys.pressed(KeyCode::KeyA) {
            direction.x -= 1.0;
        }
        if keys.pressed(KeyCode::KeyD) {
            direction.x += 1.0;
        }
        if keys.pressed(KeyCode::KeyS) {
            direction.y -= 1.0;
        }
        if keys.pressed(KeyCode::KeyW) {
            direction.y += 1.0;
        }
    }
    if direction != Vec2::ZERO {
        let step = direction.normalize() * PAN_CELLS_PER_SECOND * projection.scale;
        transform.translation += (step * time.delta_secs()).extend(0.0);
    }

    let mut steps = 0.0;
    for message in wheel.read() {
        steps += match message.unit {
            MouseScrollUnit::Line => message.y,
            MouseScrollUnit::Pixel => message.y / 32.0,
        };
    }
    let over_canvas = window
        .as_deref()
        .is_some_and(|window| crate::canvas::pointer_over_canvas(window, &frame));
    if over_canvas || crate::ui::pointer_over_ui(&hover, &nodes) {
        steps = 0.0;
    }
    if !typing {
        if keys.just_pressed(KeyCode::Equal) || keys.just_pressed(KeyCode::NumpadAdd) {
            steps += 1.0;
        }
        if keys.just_pressed(KeyCode::Minus) || keys.just_pressed(KeyCode::NumpadSubtract) {
            steps -= 1.0;
        }
    }
    if steps != 0.0 {
        projection.scale = (projection.scale / ZOOM_PER_STEP.powf(steps)).max(1e-4);
    }
}

fn fit_ramp(
    document: Res<Document>,
    mut range: ResMut<ViewRange>,
    mut materials: ResMut<Assets<FieldMaterial>>,
    camera: Single<(&Transform, &Projection), With<EditorCamera>>,
    quad: Single<&MeshMaterial2d<FieldMaterial>, With<FieldQuad>>,
) {
    let (transform, projection) = camera.into_inner();
    let Projection::Orthographic(projection) = projection else {
        return;
    };
    let Some(mut material) = materials.get_mut(&quad.0) else {
        return;
    };
    let Some(terrain) = document.terrain() else {
        return;
    };
    let Some(field) = terrain.field(document.active()) else {
        return;
    };
    if field.baked().is_empty() {
        return;
    }

    let view = visible_rect(transform, projection);
    let low_cell = world_to_cell(view.min, terrain.size);
    let high_cell = world_to_cell(view.max, terrain.size);

    let size = terrain.size.as_vec2();
    let min = low_cell.max(Vec2::ZERO).min(size - Vec2::ONE);
    let max = high_cell.max(Vec2::ZERO).min(size - Vec2::ONE);

    let mut samples = Vec::with_capacity((FIT_SAMPLES * FIT_SAMPLES) as usize);
    for row in 0..FIT_SAMPLES {
        for column in 0..FIT_SAMPLES {
            let t = Vec2::new(
                column as f32 / (FIT_SAMPLES - 1) as f32,
                row as f32 / (FIT_SAMPLES - 1) as f32,
            );
            let cell = min + (max - min) * t;
            let value = field.sample(cell.x, cell.y);
            if value.is_finite() {
                samples.push(value);
            }
        }
    }
    if samples.is_empty() {
        return;
    }

    samples.sort_by(f32::total_cmp);
    let trim = ((samples.len() as f32 * FIT_TRIM) as usize).min(samples.len() / 4);
    let mut low = samples[trim];
    let mut high = samples[samples.len() - 1 - trim];

    if high - low < MIN_SPAN {
        let middle = (low + high) * 0.5;
        low = middle - MIN_SPAN * 0.5;
        high = middle + MIN_SPAN * 0.5;
    }

    let diverging = low < 0.0 && high > 0.0;

    *range = ViewRange {
        low,
        high,
        diverging,
    };
    material.settings.range = Vec2::new(low, high);
    material.settings.diverging = if diverging { 1.0 } else { 0.0 };
}

fn track_visible_cells(
    document: Res<Document>,
    mut visible: ResMut<VisibleCells>,
    camera: Single<(&Transform, &Projection), With<EditorCamera>>,
) {
    let (transform, projection) = camera.into_inner();
    let Projection::Orthographic(projection) = projection else {
        return;
    };
    let size = document.size;
    let view = visible_rect(transform, projection);

    let bounds = size.as_vec2();
    let low = world_to_cell(view.min, size)
        .floor()
        .clamp(Vec2::ZERO, bounds);
    let high = world_to_cell(view.max, size)
        .ceil()
        .clamp(Vec2::ZERO, bounds);
    visible.0 = if high.x <= low.x || high.y <= low.y {
        CellRect::EMPTY
    } else {
        CellRect::new(low.as_uvec2(), high.as_uvec2())
    };
}

/// Puts the whole document in the space the panels have left, in one step.
///
/// The only thing in the crate that knows the panels are there, and it has to be: the
/// world is drawn across the whole window with the panels over it, so a document
/// fitted to the window alone would be partly behind them.
///
/// A jump rather than an animation, so the camera is where it was asked to be by the
/// time the call returns — a caller that fits and then reads what is on screen gets
/// the fitted view, not a frame of a transition.
pub fn fit_camera(
    transform: &mut Transform,
    projection: &mut Projection,
    size: UVec2,
    free: FreeView,
) {
    let Projection::Orthographic(projection) = projection else {
        return;
    };

    let area = projection.area.size();
    if area.x <= 0.0 || area.y <= 0.0 {
        return;
    }

    let uncovered = area * free.size;
    if uncovered.x <= 0.0 || uncovered.y <= 0.0 {
        return;
    }
    let document = size.as_vec2();
    let factor = (document.x / uncovered.x).max(document.y / uncovered.y);
    projection.scale *= factor;

    transform.translation = (-free.centre * area * factor).extend(transform.translation.z);
}

/// Centres the view on a document cell. Absolute rather than relative, so a caller
/// says where to look instead of how far to travel, and the zoom is left alone.
pub fn look_at_cell(transform: &mut Transform, size: UVec2, cell: Vec2) {
    let world = cell_to_world(cell, size);
    transform.translation = world.extend(transform.translation.z);
}

/// Zooms so that `cells` document cells span the width of the view.
///
/// In cells rather than as a bare scale, so a caller says how much of the document it
/// wants to see without knowing the window's size. Floored at a few cells, and does
/// nothing on a projection with no width yet.
pub fn set_cells_across(projection: &mut Projection, cells: f32) {
    let Projection::Orthographic(projection) = projection else {
        return;
    };
    let width = projection.area.width();
    if width > 0.0 {
        projection.scale *= cells.max(MIN_CELLS_ACROSS) / width;
    }
}

/// How many document cells currently span the width of the view. `0.0` for a
/// projection that is not orthographic.
pub fn cells_across(projection: &Projection) -> f32 {
    match projection {
        Projection::Orthographic(projection) => projection.area.width(),
        _ => 0.0,
    }
}

/// The document cell the view is centred on, fraction kept. Outside the document if
/// the camera has been panned off it.
pub fn view_centre_cell(transform: &Transform, size: UVec2) -> Vec2 {
    world_to_cell(transform.translation.truncate(), size)
}
