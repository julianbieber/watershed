//! The document as one graph of fields: what a field is drawn as, where the cards sit,
//! and what the pointer does among them.

use bevy::asset::AssetServer;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::feathers::constants::fonts;
use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::FontSize;
use bevy::window::PrimaryWindow;

use super::thumb::CardThumb;
use super::{
    CANVAS_LAYER, CARD, CanvasCameraTag, CanvasFrame, CanvasLabel, CanvasShape, DETAIL_SIZE, Grab,
    MARGIN, MAX_SCALE, MIN_SCALE, OpenField, PARAM_SIZE, ROW_STEP, THUMB, TITLE_BAR, TITLE_SIZE,
    edges, input,
};
use crate::document::Document;
use crate::gpu::ShaderLibrary;
use crate::terrain::TerrainSpec;
use crate::ui::{pointer_over_ui, report};

const FIELD: Color = Color::srgb(0.42, 0.33, 0.55);
const BODY: Color = Color::srgb(0.16, 0.17, 0.21);
const SELECTED: Color = Color::srgb(0.85, 0.87, 0.95);
const WIRE: Color = Color::srgb(0.48, 0.64, 0.86);
const BROKEN: Color = Color::srgb(0.72, 0.24, 0.24);
const FAULT: Color = Color::srgb(1.0, 0.74, 0.72);
const DETAIL: Color = Color::srgb(0.72, 0.76, 0.85);
const PARAM: Color = Color::srgb(0.62, 0.66, 0.76);

const FAULT_CHARS: usize = 34;

const CARD_COLUMN: f32 = CARD.x * 1.6;

const CARD_ROW: f32 = CARD.y * 1.3;

/// Everything the canvas spawns, so a rebuild can take it all down in one query.
#[derive(Component)]
pub(super) struct CanvasOwned;

/// One field's card.
///
/// Carries the field's name rather than an index, so a hit test answers something the
/// document can be addressed by after the cards have been rebuilt under it.
#[derive(Component)]
pub struct FieldCard {
    /// The field this card stands for.
    pub field: String,
    /// Width and height in canvas units.
    pub size: Vec2,
}

/// One ribbon, as the two points it joins.
///
/// The points rather than the two card entities: a card never moves, because its place
/// comes from the layout and is not in the document.
#[derive(Component)]
pub(super) struct FieldRibbon {
    from: Vec2,
    to: Vec2,
}

struct Placed {
    field: String,
    at: Vec2,
}

fn layout(terrain: &TerrainSpec) -> (Vec<Placed>, Option<String>) {
    let (names, cycle) = match terrain.bake_order() {
        Ok(order) => (
            order
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<String>>(),
            None,
        ),
        Err(error) => (
            terrain
                .fields
                .iter()
                .map(|field| field.id.to_string())
                .collect(),
            Some(error.to_string()),
        ),
    };

    let mut columns: Vec<(String, usize)> = names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), if cycle.is_some() { index } else { 0 }))
        .collect();
    for _ in 0..columns.len() * usize::from(cycle.is_none()) {
        let mut moved = false;
        for index in 0..columns.len() {
            let Some(field) = terrain.field(&columns[index].0) else {
                continue;
            };
            let wanted = crate::edit::reads_of(field)
                .iter()
                .filter_map(|read| columns.iter().find(|(held, _)| held == read))
                .map(|(_, column)| column + 1)
                .max()
                .unwrap_or(0);
            if wanted > columns[index].1 {
                columns[index].1 = wanted;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }

    let mut rows: Vec<usize> = Vec::new();
    let mut placed = Vec::with_capacity(columns.len());
    for (field, column) in columns {
        if rows.len() <= column {
            rows.resize(column + 1, 0);
        }
        let row = rows[column];
        rows[column] += 1;
        placed.push(Placed {
            field,
            at: Vec2::new(column as f32 * CARD_COLUMN, -(row as f32) * CARD_ROW),
        });
    }
    (placed, cycle)
}

fn edges_between(placed: &[Placed], terrain: &TerrainSpec) -> Vec<(Vec2, Vec2)> {
    let mut wires = Vec::new();
    for reader in placed {
        let Some(field) = terrain.field(&reader.field) else {
            continue;
        };
        for read in crate::edit::reads_of(field) {
            let Some(from) = placed
                .iter()
                .find(|held| held.field == read)
                .map(|held| held.at)
            else {
                continue;
            };
            wires.push((
                from + Vec2::new(CARD.x * 0.5, 0.0),
                reader.at - Vec2::new(CARD.x * 0.5, 0.0),
            ));
        }
    }
    wires
}

/// Where the canvas camera has to sit, and at what scale, for every field card to be
/// inside a viewport that many logical pixels across.
///
/// What the key `F` and the Fit button reach. `None` for a document with no fields or a
/// viewport with no area — there is nothing to frame in either case.
pub(super) fn overview_fit(terrain: &TerrainSpec, viewport: Vec2) -> Option<(Vec2, f32)> {
    if terrain.fields.is_empty() || viewport.x <= 1.0 || viewport.y <= 1.0 {
        return None;
    }
    let (placed, _) = layout(terrain);
    let mut low = Vec2::splat(f32::INFINITY);
    let mut high = Vec2::splat(f32::NEG_INFINITY);
    for card in &placed {
        low = low.min(card.at - CARD * 0.5);
        high = high.max(card.at + CARD * 0.5);
    }
    if !low.is_finite() || !high.is_finite() {
        return None;
    }
    let span = (high - low).max(Vec2::splat(1.0)) + Vec2::splat(CARD.x * 0.4);
    let scale = (span / viewport).max_element().clamp(MIN_SCALE, MAX_SCALE);
    Some(((low + high) * 0.5, scale))
}

/// The field whose card contains that canvas point, as its name and the card's centre.
///
/// The hit test a press is resolved through. `None` for a point between cards.
pub fn field_at<'a>(
    cards: impl IntoIterator<Item = (&'a FieldCard, &'a Transform)>,
    world: Vec2,
) -> Option<(String, Vec2)> {
    cards
        .into_iter()
        .find(|(card, at)| {
            Rect::from_center_size(at.translation.truncate(), card.size).contains(world)
        })
        .map(|(card, at)| (card.field.clone(), at.translation.truncate()))
}

/// Respawns the field cards and their ribbons when what they show changes.
///
/// Driven by a fingerprint rather than by an event, because an edit can arrive from the
/// panel, from the control socket, from a preset load or from a shader file changing on
/// disk, and a fingerprint catches all of them without each having to announce itself.
/// The key carries the names, the roles, the shifts, the relation and every fault a card
/// shows, so a field added, a read added and a shader broken or fixed each bring the
/// cards up to date.
pub(super) fn rebuild_overview(
    mut commands: Commands,
    mut document: ResMut<Document>,
    library: Res<ShaderLibrary>,
    assets: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut shape: ResMut<CanvasShape>,
    mut grab: ResMut<Grab>,
    owned: Query<Entity, With<CanvasOwned>>,
) {
    let key = fingerprint(&document, &library);
    if key == shape.key {
        return;
    }
    shape.key = key;

    for entity in &owned {
        commands.entity(entity).despawn();
    }
    *grab = Grab::Idle;

    let active = document.active().to_owned();
    let Some(terrain) = document.terrain() else {
        return;
    };
    let (placed, cycle) = layout(terrain);
    let wires = edges_between(&placed, terrain);
    let faults = terrain.field_faults();
    let cards: Vec<(String, &'static str, u8, Vec2, Option<String>)> = placed
        .iter()
        .filter_map(|card| {
            let field = terrain.field(&card.field)?;
            let fault = faults
                .iter()
                .find(|(id, _)| *id == field.id)
                .map(|(_, fault)| fault.clone())
                .or_else(|| library_fault(&library, &field.file()));
            Some((
                card.field.clone(),
                field.role.as_str(),
                field.shift,
                card.at,
                fault,
            ))
        })
        .collect();

    if let Some(cycle) = cycle {
        report(&mut document, Err(cycle));
    }

    let font = assets.load(fonts::REGULAR);
    for (name, role, shift, at, fault) in cards {
        let selected = name == active;
        spawn_field_card(
            &mut commands,
            &font,
            &name,
            role,
            shift,
            at,
            selected,
            fault,
        );
    }

    let wire_material = materials.add(WIRE);
    for (from, to) in wires {
        commands.spawn((
            Mesh2d(meshes.add(edges::blank_ribbon())),
            MeshMaterial2d(wire_material.clone()),
            Transform::from_xyz(0.0, 0.0, -1.0),
            NoFrustumCulling,
            RenderLayers::layer(CANVAS_LAYER),
            FieldRibbon { from, to },
            CanvasOwned,
        ));
    }
}

fn library_fault(library: &ShaderLibrary, file: &str) -> Option<String> {
    library.entry(file).and_then(|entry| entry.error.clone())
}

fn spawn_field_card(
    commands: &mut Commands,
    font: &Handle<Font>,
    name: &str,
    role: &str,
    shift: u8,
    at: Vec2,
    selected: bool,
    fault: Option<String>,
) {
    let title_y = (CARD.y - TITLE_BAR) * 0.5;
    let accent = match (&fault, selected) {
        (Some(_), _) => BROKEN,
        (None, true) => SELECTED,
        (None, false) => FIELD,
    };
    let entity = commands
        .spawn((
            Sprite {
                color: BODY,
                custom_size: Some(CARD),
                ..default()
            },
            Transform::from_translation(at.extend(0.0)),
            RenderLayers::layer(CANVAS_LAYER),
            FieldCard {
                field: name.to_owned(),
                size: CARD,
            },
            CanvasOwned,
        ))
        .id();

    commands.entity(entity).with_children(|parent| {
        parent.spawn((
            Sprite {
                color: accent,
                custom_size: Some(Vec2::new(CARD.x, TITLE_BAR)),
                ..default()
            },
            Transform::from_xyz(0.0, title_y, 0.01),
            RenderLayers::layer(CANVAS_LAYER),
        ));
        parent.spawn((
            Text2d::new(name.to_owned()),
            TextFont {
                font: bevy::text::FontSource::Handle(font.clone()),
                font_size: FontSize::Px(TITLE_SIZE),
                ..default()
            },
            TextColor(Color::srgb(0.96, 0.97, 1.0)),
            Anchor::CENTER_LEFT,
            Transform::from_xyz(-CARD.x * 0.5 + 12.0, title_y, 0.02),
            RenderLayers::layer(CANVAS_LAYER),
            CanvasLabel(TITLE_SIZE),
        ));
        let blank = FIELD.with_alpha(0.35);
        parent.spawn((
            Sprite {
                color: blank,
                custom_size: Some(Vec2::splat(THUMB)),
                ..default()
            },
            Transform::from_translation(thumb_centre().extend(0.01)),
            RenderLayers::layer(CANVAS_LAYER),
            CardThumb::new(name.to_owned(), blank),
        ));
        let mut row = |index: usize, size: f32, colour: Color, text: String| {
            parent.spawn((
                Text2d::new(text),
                TextFont {
                    font: bevy::text::FontSource::Handle(font.clone()),
                    font_size: FontSize::Px(size),
                    ..default()
                },
                TextColor(colour),
                Anchor::CENTER_LEFT,
                Transform::from_xyz(text_left(), row_y(index), 0.02),
                RenderLayers::layer(CANVAS_LAYER),
                CanvasLabel(size),
            ));
        };
        row(0, DETAIL_SIZE, DETAIL, role.to_owned());
        row(1, PARAM_SIZE, PARAM, format!("shift {shift}"));
        if let Some(fault) = fault {
            row(2, PARAM_SIZE, FAULT, fault_line(&fault));
        }
    });
}

fn fault_line(fault: &str) -> String {
    clip(fault, FAULT_CHARS)
}

fn clip(text: &str, chars: usize) -> String {
    let first = text.lines().next().unwrap_or("").trim();
    if first.chars().count() <= chars {
        return first.to_owned();
    }
    first
        .chars()
        .take(chars - 1)
        .chain(std::iter::once('…'))
        .collect()
}

fn body_top() -> f32 {
    CARD.y * 0.5 - TITLE_BAR
}

fn row_y(index: usize) -> f32 {
    body_top() - 11.0 - ROW_STEP * index as f32
}

fn text_left() -> f32 {
    -CARD.x * 0.5 + MARGIN * 2.0 + THUMB
}

fn thumb_centre() -> Vec2 {
    Vec2::new(
        -CARD.x * 0.5 + MARGIN + THUMB * 0.5,
        body_top() - 6.0 - THUMB * 0.5,
    )
}

/// Rebuilds each ribbon's mesh from the two points it joins.
///
/// Only when the zoom changed or a ribbon was just spawned: a card never moves, so those
/// are the only two things the mesh depends on — the half-width is in screen pixels, and
/// so follows the camera's scale.
pub(super) fn route_field_ribbons(
    camera: Option<Single<&Projection, With<CanvasCameraTag>>>,
    ribbons: Query<(&FieldRibbon, &Mesh2d)>,
    spawned: Query<(), Added<FieldRibbon>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut applied: Local<Option<f32>>,
) {
    let Some(camera) = camera else {
        return;
    };
    let Projection::Orthographic(ortho) = camera.into_inner() else {
        return;
    };
    let zoomed = *applied != Some(ortho.scale);
    if zoomed {
        *applied = Some(ortho.scale);
    }
    if !zoomed && spawned.is_empty() {
        return;
    }
    let half_width = edges::EDGE_PIXELS * 0.5 * ortho.scale;
    for (ribbon, handle) in &ribbons {
        if let Some(mut mesh) = meshes.get_mut(&handle.0) {
            *mesh = edges::ribbon_between(ribbon.from, ribbon.to, half_width);
        }
    }
}

/// Opens a field on a double-click on its card, and pans the canvas on a press between
/// cards.
///
/// Writes nothing to the document: opening goes through [`OpenField`], and card
/// positions come from the layout and are never stored.
pub(super) fn overview_drag(
    mouse: Res<ButtonInput<MouseButton>>,
    hover: Res<HoverMap>,
    ui: Query<(), With<Node>>,
    window: Option<Single<&Window, With<PrimaryWindow>>>,
    frame: Res<CanvasFrame>,
    camera: Option<Single<(&Transform, &Projection), (With<CanvasCameraTag>, Without<FieldCard>)>>,
    cards: Query<(&FieldCard, &Transform), Without<CanvasCameraTag>>,
    time: Res<Time>,
    mut open: MessageWriter<OpenField>,
    mut grab: ResMut<Grab>,
    mut last_click: Local<Option<(f32, String)>>,
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
        .and_then(|cursor| input::canvas_cursor(&window, &frame, cursor))
    else {
        return;
    };
    let world = camera_at.translation.truncate() + cursor * Vec2::new(ortho.scale, -ortho.scale);

    if mouse.just_pressed(MouseButton::Left) && !pointer_over_ui(&hover, &ui) {
        let now = time.elapsed_secs();
        match field_at(cards.iter(), world) {
            Some((field, _)) => {
                let again = last_click.as_ref().is_some_and(|(at, before)| {
                    *before == field && now - at <= input::DOUBLE_CLICK
                });
                if again {
                    open.write(OpenField { field });
                    *last_click = None;
                } else {
                    *last_click = Some((now, field));
                }
            }
            None => {
                *last_click = None;
                *grab = Grab::Pan { anchor: world };
            }
        }
    }

    if mouse.just_released(MouseButton::Left) {
        *grab = Grab::Idle;
    }
}

fn fingerprint(document: &Document, library: &ShaderLibrary) -> String {
    let mut key = String::from("overview|");
    key.push_str(document.active());
    let Some(terrain) = document.terrain() else {
        key.push_str("|none");
        return key;
    };
    for field in &terrain.fields {
        key.push_str(&format!(
            "|{}:{}:{}:{}",
            field.id,
            field.role.as_str(),
            field.shift,
            crate::edit::reads_of(field).join(","),
        ));
        if let Some(error) = library_fault(library, &field.file()) {
            key.push_str(&format!("|error {}:{error}", field.id));
        }
    }
    for (id, fault) in terrain.field_faults() {
        key.push_str(&format!("|fault {id}:{fault}"));
    }
    key
}

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;

    use super::*;
    use crate::terrain::{Field, TerrainSpec};

    fn height_reads_base_and_relief() -> TerrainSpec {
        TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("moisture").held(0.5))
            .with_field(Field::new("base").held(0.25))
            .with_field(Field::new("relief").held(0.75))
            .with_field(Field::new("height").reading(&["base", "relief"]))
    }

    fn cyclic() -> TerrainSpec {
        TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("here").reading(&["there"]))
            .with_field(Field::new("there").reading(&["here"]))
    }

    fn at(placed: &[Placed], name: &str) -> Vec2 {
        placed
            .iter()
            .find(|card| card.field == name)
            .unwrap_or_else(|| panic!("no card for `{name}`"))
            .at
    }

    // What a field reads is drawn to its left, so the document is read left to right in
    // the order it is baked in rather than in the order it happens to be declared in.
    #[test]
    fn a_field_is_placed_right_of_everything_it_reads() {
        let terrain = height_reads_base_and_relief();
        let (placed, cycle) = layout(&terrain);
        assert!(cycle.is_none(), "a document that orders reported a cycle");
        assert!(at(&placed, "base").x < at(&placed, "height").x);
        assert!(at(&placed, "relief").x < at(&placed, "height").x);
    }

    // A field at neither end of the relation carries no ribbon at all, rather than a
    // stub to nothing.
    #[test]
    fn a_field_nothing_reads_and_that_reads_nothing_has_no_ribbon() {
        let terrain = height_reads_base_and_relief();
        let (placed, _) = layout(&terrain);
        let moisture = at(&placed, "moisture");
        for (from, to) in edges_between(&placed, &terrain) {
            assert!(from.y != moisture.y || from.x < moisture.x - CARD.x);
            assert_ne!(to, moisture - Vec2::new(CARD.x * 0.5, 0.0));
        }
    }

    // One ribbon per declared read and no more: `height` reads two fields, so a double
    // count here would draw two ribbons over each other.
    #[test]
    fn every_declared_read_is_joined_once() {
        let terrain = height_reads_base_and_relief();
        let (placed, _) = layout(&terrain);
        let wires = edges_between(&placed, &terrain);
        assert_eq!(wires.len(), 2);
        let ends: Vec<Vec2> = wires.iter().map(|(from, _)| *from).collect();
        assert!(ends.contains(&(at(&placed, "base") + Vec2::new(CARD.x * 0.5, 0.0))));
        assert!(ends.contains(&(at(&placed, "relief") + Vec2::new(CARD.x * 0.5, 0.0))));
    }

    // A document whose fields cannot be ordered still draws every card — in declaration
    // order, one per column — and says what the cycle is, rather than showing an empty
    // canvas with no account of why.
    #[test]
    fn a_cycle_falls_back_to_declaration_order_and_names_itself() {
        let terrain = cyclic();
        let (placed, cycle) = layout(&terrain);
        let named = cycle.expect("a cyclic document reports its cycle");
        assert!(named.contains("cycle"), "{named:?} does not name the cycle");
        assert_eq!(placed.len(), 2);
        assert_eq!(placed[0].field, "here");
        assert_eq!(placed[1].field, "there");
        assert!(placed[0].at.x < placed[1].at.x);
    }

    // The hit test is what a press is resolved through, so it has to answer the card the
    // point is inside and nothing at all for the gap between two — a press in the gap
    // pans, and one that guessed a neighbour would not.
    #[test]
    fn the_hit_test_answers_the_card_under_the_point_and_nothing_between_two() {
        let left = (
            FieldCard {
                field: "base".to_owned(),
                size: CARD,
            },
            Transform::from_translation(Vec3::ZERO),
        );
        let right = (
            FieldCard {
                field: "height".to_owned(),
                size: CARD,
            },
            Transform::from_xyz(CARD_COLUMN, 0.0, 0.0),
        );
        let cards = vec![(&left.0, &left.1), (&right.0, &right.1)];

        let hit = field_at(cards.clone(), Vec2::new(CARD.x * 0.25, 0.0));
        assert_eq!(hit.map(|(field, _)| field), Some("base".to_owned()));
        let over = field_at(cards.clone(), Vec2::new(CARD_COLUMN, 0.0));
        assert_eq!(over.map(|(field, _)| field), Some("height".to_owned()));
        assert!(field_at(cards, Vec2::new(CARD_COLUMN * 0.5, 0.0)).is_none());
    }

    // Pinned at the arithmetic the key F runs: whatever the fit answers has to put every
    // card of a document wider than the viewport inside the framed rectangle.
    #[test]
    fn a_fit_puts_every_field_card_of_a_wide_document_in_view() {
        let mut terrain = TerrainSpec::new(UVec2::splat(16));
        let mut previous: Option<String> = None;
        for step in 0..8 {
            let name = format!("f{step}");
            let field = match &previous {
                Some(read) => Field::new(name.as_str()).reading(&[read.as_str()]),
                None => Field::new(name.as_str()).held(0.5),
            };
            terrain = terrain.with_field(field);
            previous = Some(name);
        }

        let viewport = Vec2::new(800.0, 600.0);
        let (centre, scale) = overview_fit(&terrain, viewport).expect("a fit");
        let framed = Rect::from_center_size(centre, viewport * scale);
        let (placed, _) = layout(&terrain);
        for card in &placed {
            let rect = Rect::from_center_size(card.at, CARD);
            assert!(framed.contains(rect.min), "a card's corner left the frame");
            assert!(framed.contains(rect.max), "a card's corner left the frame");
        }
    }

    // Fitting has to answer with nothing rather than a centre and a scale when there is
    // nothing to frame, or the Fit button would throw the camera at whatever the empty
    // bounds came out as.
    #[test]
    fn there_is_nothing_to_fit_without_fields_or_without_a_viewport() {
        assert!(
            overview_fit(&TerrainSpec::new(UVec2::splat(16)), Vec2::new(800.0, 600.0)).is_none()
        );
        assert!(overview_fit(&height_reads_base_and_relief(), Vec2::ZERO).is_none());
    }

    fn overview_app(terrain: TerrainSpec, library: ShaderLibrary) -> App {
        let mut document = Document::default();
        document.adopt(terrain);

        let mut app = App::new();
        app.add_plugins((
            bevy::app::TaskPoolPlugin::default(),
            bevy::asset::AssetPlugin::default(),
        ));
        app.init_asset::<Mesh>();
        app.init_asset::<ColorMaterial>();
        app.init_asset::<Font>();
        app.insert_resource(document);
        app.insert_resource(library);
        app.init_resource::<CanvasShape>();
        app.init_resource::<Grab>();
        app
    }

    fn texts(app: &mut App) -> Vec<String> {
        app.world_mut()
            .query::<&Text2d>()
            .iter(app.world())
            .map(|text| text.0.clone())
            .collect()
    }

    // Drawing the cards is a way of looking, so building them leaves the history, the
    // dirty flag and what is baked exactly where they were — otherwise glancing at the
    // document would be something that has to be saved.
    #[test]
    fn drawing_the_overview_makes_no_edit_and_starts_no_bake() {
        let mut app = overview_app(height_reads_base_and_relief(), ShaderLibrary::default());
        let before = {
            let document = app.world().resource::<Document>();
            (
                document.history().undo,
                document.history().redo,
                document.is_dirty(),
                document.baked().name(),
                document.revision(),
            )
        };

        app.world_mut().run_system_once(rebuild_overview).unwrap();

        let document = app.world().resource::<Document>();
        assert_eq!(document.history().undo, before.0);
        assert_eq!(document.history().redo, before.1);
        assert_eq!(document.is_dirty(), before.2);
        assert_eq!(document.baked().name(), before.3);
        assert_eq!(document.revision(), before.4);
        assert_eq!(
            app.world_mut()
                .query::<&FieldCard>()
                .iter(app.world())
                .count(),
            4,
            "one card per field"
        );
    }

    // A card carries the name rather than an index, so the double-click addresses a
    // field that survives the cards being rebuilt under it.
    #[test]
    fn every_field_gets_a_card_carrying_its_name() {
        let mut app = overview_app(height_reads_base_and_relief(), ShaderLibrary::default());
        app.world_mut().run_system_once(rebuild_overview).unwrap();
        let mut names: Vec<String> = app
            .world_mut()
            .query::<&FieldCard>()
            .iter(app.world())
            .map(|card| card.field.clone())
            .collect();
        names.sort();
        assert_eq!(names, ["base", "height", "moisture", "relief"]);
    }

    // The fingerprint is what brings the cards up to date, so adding a field and adding a
    // read each have to move it.
    #[test]
    fn the_fingerprint_moves_when_a_field_or_a_reference_is_added() {
        let library = ShaderLibrary::default();
        let mut document = Document::default();
        document.adopt(height_reads_base_and_relief());
        let before = fingerprint(&document, &library);

        let mut added = Document::default();
        added.adopt(height_reads_base_and_relief().with_field(Field::new("temperature")));
        assert_ne!(
            fingerprint(&added, &library),
            before,
            "a new field left the key alone"
        );

        let mut read = Document::default();
        read.adopt(
            TerrainSpec::new(UVec2::splat(16))
                .with_field(Field::new("moisture").reading(&["base"]))
                .with_field(Field::new("base").held(0.25))
                .with_field(Field::new("relief").held(0.75))
                .with_field(Field::new("height").reading(&["base", "relief"])),
        );
        assert_ne!(
            fingerprint(&read, &library),
            before,
            "a new reference left the key alone"
        );
    }

    // A shader breaking on disk changes nothing in the document, so the fingerprint has
    // to carry the library's error or the card keeps looking whole until some edit.
    #[test]
    fn the_fingerprint_moves_when_a_fields_shader_stops_compiling() {
        let mut document = Document::default();
        document.adopt(height_reads_base_and_relief());
        let good = fingerprint(&document, &ShaderLibrary::default());
        let broken = fingerprint(
            &document,
            &ShaderLibrary::with_fault("height.wgsl", "line 2: unknown type"),
        );
        assert_ne!(good, broken);
    }

    // A field that cannot bake reads as zero, so its card has to say why, or the
    // overview shows a document that looks whole while one field of it is empty.
    #[test]
    fn a_field_naming_no_field_carries_the_fault_on_its_card() {
        let mut app = overview_app(
            TerrainSpec::new(UVec2::splat(16))
                .with_field(Field::new("height").reading(&["nowhere"])),
            ShaderLibrary::default(),
        );
        app.world_mut().run_system_once(rebuild_overview).unwrap();
        let texts = texts(&mut app);
        assert!(
            texts.iter().any(|text| text.contains("nowhere")),
            "{texts:?}"
        );
    }

    // A field is its shader file, so a file that will not compile is that field's fault
    // and has to reach its card and redden its bar, not only the status line.
    #[test]
    fn a_field_whose_shader_will_not_compile_carries_the_error_on_its_card() {
        let mut app = overview_app(
            height_reads_base_and_relief(),
            ShaderLibrary::with_fault("height.wgsl", "line 2: unknown type: 'vec4'"),
        );
        app.world_mut().run_system_once(rebuild_overview).unwrap();
        let texts = texts(&mut app);
        assert!(
            texts.iter().any(|text| text.starts_with("line 2:")),
            "{texts:?}"
        );
        let broken = app
            .world_mut()
            .query::<&Sprite>()
            .iter(app.world())
            .filter(|sprite| sprite.color == BROKEN)
            .count();
        assert_eq!(broken, 1, "only the broken field's bar is reddened");
    }

    // Every line of a card has to sit inside the card and clear the line above it,
    // including the fault row — which is the layout the constants are easiest to nudge
    // out of.
    #[test]
    fn every_line_of_a_card_sits_inside_it_and_clears_the_one_above() {
        let title_y = (CARD.y - TITLE_BAR) * 0.5;
        let low = -CARD.y * 0.5;
        let high = CARD.y * 0.5;
        let rows: Vec<f32> = (0..3).map(row_y).collect();

        for y in rows.iter().copied().chain([title_y]) {
            assert!(y > low && y < high, "a line at {y} left the card");
        }
        assert!(
            title_y - TITLE_SIZE > rows[0],
            "the role row is under the bar"
        );
        for pair in rows.windows(2) {
            assert!(pair[0] - pair[1] >= DETAIL_SIZE, "rows {pair:?} overlap");
        }
        let thumb = thumb_centre();
        assert!(
            thumb.x - THUMB * 0.5 > -CARD.x * 0.5
                && thumb.y + THUMB * 0.5 < title_y - TITLE_BAR * 0.5
                && thumb.y - THUMB * 0.5 > low,
            "the picture left the card's body"
        );
        assert!(
            text_left() > thumb.x + THUMB * 0.5,
            "the text is on the picture"
        );
    }

    // A naga message can run to a paragraph and a card is narrow, so the line has to be
    // cut to fit — and cut by character, since a message can quote a token that is not
    // ASCII.
    #[test]
    fn fault_line_keeps_the_first_line_and_cuts_it_to_the_cards_width() {
        let line =
            fault_line("line 7: expected ‘;’ but found an identifier of some length\nnote: here");
        assert!(!line.contains('\n'), "{line:?} spans lines");
        assert!(line.chars().count() <= FAULT_CHARS, "{line:?} is too wide");
        assert!(line.ends_with('…'), "{line:?} is not marked as cut");
    }
}
