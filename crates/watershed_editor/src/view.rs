// TODO(jb-doc): module docs — that one cell is one world unit, and what that makes the
// camera's scale mean.

use bevy::asset::RenderAssetUsages;
use bevy::camera::CameraUpdateSystems;
use bevy::image::{ImageSampler, TextureFormatPixelInfo};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::sprite_render::MeshMaterial2d;
use watershed::WaterState;

use crate::document::{Document, EditorSystems};
use crate::material::{FieldMaterial, FieldMaterialPlugin, FieldSettings};

/// TODO(jb-doc): why the overlay and the observation share one number rather than each
/// carrying its own — that a scenario asserting on channels must be asking about the ones
/// it can see.
pub const CHANNEL_THRESHOLD: f32 = 64.0;

/// Cells the ramp is fitted over. The visible rectangle is what a screen holds, and a
/// screen holds a slice of the field rather than all of it.
const FIT_SAMPLES: u32 = 24;

/// TODO(jb-comment): why a fortieth is trimmed from each end, and what one deep-water cell
/// does to a ramp fitted without it.
const FIT_TRIM: f32 = 1.0 / 40.0;

/// TODO(jb-comment): what this floor is defending — that genuinely flat ground has to look
/// flat rather than have its last quantization step magnified to full contrast.
const MIN_SPAN: f32 = 1.0 / 512.0;

const PAN_CELLS_PER_SECOND: f32 = 600.0;
const ZOOM_PER_STEP: f32 = 1.2;
const MIN_CELLS_ACROSS: f32 = 8.0;

pub struct ViewPlugin;

impl Plugin for ViewPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FieldMaterialPlugin)
            .init_resource::<ViewRange>()
            .add_systems(Startup, spawn_view)
            .add_systems(
                Update,
                (pan_zoom, sync_maps)
                    .chain()
                    .in_set(EditorSystems::View)
                    .after(EditorSystems::Document),
            )
            // The fit reads `OrthographicProjection::area`, which `camera_system` writes in
            // `PostUpdate` — so fitting in `Update` would read the area belonging to the
            // *previous* frame's scale, and a scenario would see a stale range for one
            // frame after every zoom.
            .add_systems(PostUpdate, fit_ramp.after(CameraUpdateSystems));
    }
}

#[derive(Component)]
pub struct EditorCamera;

#[derive(Component)]
struct FieldQuad;

/// What the ramp is currently fitted to. Written once a frame by [`fit_ramp`] and read by
/// everything that has to agree with the screen — the legend and the ctl — because
/// deriving it a second time would be a second answer to what the view is showing.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct ViewRange {
    pub low: f32,
    pub high: f32,
    pub diverging: bool,
}

/// TODO(jb-doc): why the maps are tracked by revision rather than rebuilt when the terrain
/// is touched.
#[derive(Component)]
struct MapRevisions {
    field: Option<u64>,
    water: Option<u64>,
}

fn blank(format: TextureFormat) -> Image {
    // A one-by-one of zeroes rather than no texture at all: bevy's fallback for an absent
    // image is opaque white, which in the water map would mean water everywhere. Zero is
    // the fallback each map already documents — no field and no water.
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

/// TODO(jb-comment): why the whole map is rebuilt and re-added rather than the texture
/// being written in place, and what the alternative would have cost per landed bake.
fn sync_maps(
    document: Res<Document>,
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

    if revisions.field != Some(document.revision()) {
        revisions.field = Some(document.revision());

        let Some(field) = terrain.field(document.active()) else {
            return;
        };
        let baked = field.baked();

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

/// Two channels rather than four: the level says where water stands and the flow says
/// where it runs, and neither the lake labelling nor the direction is something the eye
/// can read off a colour.
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

/// `OrthographicProjection::area` is **already multiplied by `scale`** — bevy's
/// `camera_system` writes it that way — so it is the visible world rectangle relative to
/// the camera, and scaling it again is the bug this whole module got wrong once. Nothing
/// here may multiply by `scale`.
fn visible_rect(camera: &Transform, projection: &OrthographicProjection) -> Rect {
    let centre = camera.translation.truncate();
    Rect::from_corners(centre + projection.area.min, centre + projection.area.max)
}

/// Cell coordinates of a world position. The quad is centred on the origin and one cell is
/// one unit, so this is the document's own space shifted by half its size.
fn world_to_cell(world: Vec2, size: UVec2) -> Vec2 {
    world + size.as_vec2() * 0.5
}

fn cell_to_world(cell: Vec2, size: UVec2) -> Vec2 {
    cell - size.as_vec2() * 0.5
}

fn pan_zoom(
    keys: Res<ButtonInput<KeyCode>>,
    mut wheel: MessageReader<MouseWheel>,
    time: Res<Time>,
    camera: Single<(&mut Transform, &mut Projection), With<EditorCamera>>,
) {
    let (mut transform, mut projection) = camera.into_inner();
    let Projection::Orthographic(projection) = &mut *projection else {
        return;
    };

    let mut direction = Vec2::ZERO;
    if keys.any_pressed([KeyCode::KeyA, KeyCode::ArrowLeft]) {
        direction.x -= 1.0;
    }
    if keys.any_pressed([KeyCode::KeyD, KeyCode::ArrowRight]) {
        direction.x += 1.0;
    }
    if keys.any_pressed([KeyCode::KeyS, KeyCode::ArrowDown]) {
        direction.y -= 1.0;
    }
    if keys.any_pressed([KeyCode::KeyW, KeyCode::ArrowUp]) {
        direction.y += 1.0;
    }
    if direction != Vec2::ZERO {
        // Scaled by the zoom, so a drag covers the same fraction of the screen however far
        // out the view is.
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
    if keys.just_pressed(KeyCode::Equal) || keys.just_pressed(KeyCode::NumpadAdd) {
        steps += 1.0;
    }
    if keys.just_pressed(KeyCode::Minus) || keys.just_pressed(KeyCode::NumpadSubtract) {
        steps -= 1.0;
    }
    if steps != 0.0 {
        projection.scale = (projection.scale / ZOOM_PER_STEP.powf(steps)).max(1e-4);
    }
}

/// TODO(jb-comment): why the sort is over a fixed grid rather than the whole visible
/// rectangle, and what reading every cell would cost at the zoom that shows all of a
/// 4096-square document.
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

    // Clamped to the document: a view that is mostly empty space would otherwise fit the
    // ramp to whatever a read outside the raster clamps to.
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

    // Polarity is read off the data rather than configured: a range that straddles zero is
    // the only thing that gives the neutral band a meaning.
    let diverging = low < 0.0 && high > 0.0;

    *range = ViewRange {
        low,
        high,
        diverging,
    };
    material.settings.range = Vec2::new(low, high);
    material.settings.diverging = if diverging { 1.0 } else { 0.0 };
}

/// TODO(jb-doc): why fitting is a jump rather than an animation, and what the ctl needs
/// from that.
pub fn fit_camera(transform: &mut Transform, projection: &mut Projection, size: UVec2) {
    let Projection::Orthographic(projection) = projection else {
        return;
    };
    transform.translation = Vec3::new(0.0, 0.0, transform.translation.z);

    // The area already carries the scale, so this is a ratio against what is on screen
    // now rather than an absolute — see the note on [`visible_rect`].
    let area = projection.area.size();
    if area.x > 0.0 && area.y > 0.0 {
        let size = size.as_vec2();
        projection.scale *= (size.x / area.x).max(size.y / area.y);
    }
}

/// Centres the view on a document cell. Absolute rather than relative so a scenario says
/// where to look instead of how far to travel.
pub fn look_at_cell(transform: &mut Transform, size: UVec2, cell: Vec2) {
    let world = cell_to_world(cell, size);
    transform.translation = world.extend(transform.translation.z);
}

/// TODO(jb-doc): the unit this takes and why it is cells rather than a bare scale — that a
/// scenario can say how much of the document it wants to see without knowing the window.
pub fn set_cells_across(projection: &mut Projection, cells: f32) {
    let Projection::Orthographic(projection) = projection else {
        return;
    };
    let width = projection.area.width();
    if width > 0.0 {
        projection.scale *= cells.max(MIN_CELLS_ACROSS) / width;
    }
}

pub fn cells_across(projection: &Projection) -> f32 {
    match projection {
        Projection::Orthographic(projection) => projection.area.width(),
        _ => 0.0,
    }
}

pub fn view_centre_cell(transform: &Transform, size: UVec2) -> Vec2 {
    world_to_cell(transform.translation.truncate(), size)
}
