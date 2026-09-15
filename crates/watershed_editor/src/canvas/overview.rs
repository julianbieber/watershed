//! The document as one graph of fields: what a field is drawn as when the canvas is
//! showing the whole document rather than one field's nodes, where the cards sit, and
//! what the pointer does among them.

use bevy::asset::AssetServer;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::feathers::constants::fonts;
use bevy::input::keyboard::Key;
use bevy::input_focus::InputFocus;
use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::{EditableText, FontSize};
use bevy::window::PrimaryWindow;

use super::scene::CanvasOwned;
use super::thumb::{CardThumb, ThumbSource};
use super::{
    CANVAS_LAYER, CARD, CanvasCameraTag, CanvasFrame, CanvasLabel, CanvasShape, DETAIL_SIZE, Grab,
    MAX_SCALE, MIN_SCALE, OpenField, PARAM_SIZE, Selection, THUMB, TITLE_BAR, TITLE_SIZE, edges,
    input, scene,
};
use crate::document::Document;
use crate::edit::{Edit, node_of_path};
use crate::terrain::TerrainSpec;
use crate::terrain::graph::NodeOp;
use crate::ui::{pointer_over_ui, report, typing};

const FIELD: Color = Color::srgb(0.42, 0.33, 0.55);

const CARD_COLUMN: f32 = CARD.x * 1.6;

const CARD_ROW: f32 = CARD.y * 1.3;

/// Whether the canvas is showing the document's fields rather than the open field's
/// graph.
///
/// Switched by the canvas bar's `fields` toggle and by the key `O`, and cleared by
/// opening a field. It is a way of looking and not an edit: nothing about it is
/// written to the terrain, no history entry is made and no bake is started.
#[derive(Resource, Default)]
pub struct Overview {
    /// Whether the overview is on screen.
    pub showing: bool,
}

/// The dependency a finished drag asks for, as the field it will read and the field
/// that will read it.
///
/// What the overview holds between the frame a drag is released on and the frame the
/// edit it makes is applied, the way [`input::Finished`] does for the node graph. The
/// pair is a request and not a decision: whether the two fields may read each other at
/// all is answered by the edit.
#[derive(Resource, Default)]
pub(super) struct Wired(Option<(String, String)>);

/// How many changes were behind the document when a drag landed one, so that undoing
/// exactly that change can put the overview back.
///
/// The history carries the terrain and the field being looked at, not which of the two
/// things the canvas draws was on screen — so the one change a drag makes is marked
/// here instead. `None` once an undo would no longer be crossing that change, whether
/// because it has been undone or because another change was made on top of it.
#[derive(Resource, Default)]
pub(super) struct WiredStep(Option<usize>);

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
/// The points rather than the two card entities an edge of a node graph carries: a
/// card in the overview never moves, because its place comes from the layout and is
/// not in the document.
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
/// The twin of the graph's own fit, and what the key `F` and the Fit button reach while
/// the overview is showing. `None` for a document with no fields or a viewport with no
/// area — there is nothing to frame in either case.
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
/// The overview's hit test, and the one a press and a release are each resolved
/// through. `None` for a point between cards.
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

/// `O` switches the canvas between the open field's graph and the document's fields,
/// unless a text field has the keyboard.
///
/// The key half of the canvas bar's toggle. Reads the logical key rather than the
/// physical position, so the key printed O fits on every layout.
pub fn overview_key(
    keys: Res<ButtonInput<Key>>,
    focus: Option<Res<InputFocus>>,
    fields: Query<(), With<EditableText>>,
    mut overview: ResMut<Overview>,
) {
    if typing(focus.as_deref(), &fields) {
        return;
    }
    let pressed = keys
        .get_just_pressed()
        .any(|key| matches!(key, Key::Character(c) if c.eq_ignore_ascii_case("o")));
    if !pressed {
        return;
    }
    overview.showing = !overview.showing;
}

/// Respawns the field cards and their ribbons when the document's fields change.
///
/// Driven by the same fingerprint the graph's rebuild is, and against the same key, so
/// whichever of the two views is enabled after a toggle takes the other's entities
/// down. The key carries the names, the roles, the shifts and the relation, so a field
/// added from the panel and a reference added to a graph each bring the overview up to
/// date without it being switched off and on.
pub(super) fn rebuild_overview(
    mut commands: Commands,
    mut document: ResMut<Document>,
    assets: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut shape: ResMut<CanvasShape>,
    mut grab: ResMut<Grab>,
    owned: Query<Entity, With<CanvasOwned>>,
) {
    let key = fingerprint(&document);
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
    let cards: Vec<(String, &'static str, u8, Vec2)> = placed
        .iter()
        .filter_map(|card| {
            let field = terrain.field(&card.field)?;
            Some((
                card.field.clone(),
                field.role.as_str(),
                field.shift,
                card.at,
            ))
        })
        .collect();

    if let Some(cycle) = cycle {
        report(&mut document, Err(cycle));
    }

    let font = assets.load(fonts::REGULAR);
    for (name, role, shift, at) in cards {
        let selected = name == active;
        spawn_field_card(&mut commands, &font, &name, role, shift, at, selected);
    }

    let wire_material = materials.add(scene::WIRE);
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

fn spawn_field_card(
    commands: &mut Commands,
    font: &Handle<Font>,
    name: &str,
    role: &str,
    shift: u8,
    at: Vec2,
    selected: bool,
) {
    let title_y = (CARD.y - TITLE_BAR) * 0.5;
    let accent = if selected { scene::SELECTED } else { FIELD };
    let entity = commands
        .spawn((
            Sprite {
                color: scene::BODY,
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
            Transform::from_translation(scene::thumb_centre().extend(0.01)),
            RenderLayers::layer(CANVAS_LAYER),
            CardThumb::new(ThumbSource::Field(name.to_owned()), blank),
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
                Transform::from_xyz(scene::text_left(), scene::row_y(index), 0.02),
                RenderLayers::layer(CANVAS_LAYER),
                CanvasLabel(size),
            ));
        };
        row(0, DETAIL_SIZE, scene::DETAIL, role.to_owned());
        row(1, PARAM_SIZE, scene::PARAM, format!("shift {shift}"));
    });
}

/// Rebuilds each ribbon's mesh from the two points it joins.
///
/// Only when the zoom changed or a ribbon was just spawned: a card in the overview
/// never moves, so those are the only two things the mesh depends on — the half-width
/// is in screen pixels, and so follows the camera's scale.
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

/// Opens a field on a double-click, wires one field into another on a drag between
/// their cards, and pans the overview otherwise.
///
/// Decides the dependency a drag asks for and hands it to [`commit_field_wire`] rather
/// than writing it, the way `input::canvas_drag` hands its edit on — so the one system
/// that writes the document is the one that runs in the chain's commit slot. Card
/// positions still come from the layout and are never written.
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
    mut wired: ResMut<Wired>,
    mut last_click: Local<Option<(f32, String)>>,
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
                    open.write(OpenField {
                        field: field.clone(),
                        select: None,
                    });
                    *last_click = None;
                } else {
                    *last_click = Some((now, field.clone()));
                }
                *pressed_at = Some(cursor);
                *grab = Grab::FieldWire { from: field };
            }
            None => {
                *last_click = None;
                *pressed_at = None;
                *grab = Grab::Pan { anchor: world };
            }
        }
    }

    if mouse.just_released(MouseButton::Left) {
        let held = std::mem::take(&mut *grab);
        let travelled = pressed_at.take().map_or(0.0, |from| from.distance(cursor));
        if let Grab::FieldWire { from } = held {
            wired.0 = wired_by(
                from,
                travelled,
                field_at(cards.iter(), world).map(|(name, _)| name),
            );
        }
    }
}

fn wired_by(from: String, travelled: f32, landed: Option<String>) -> Option<(String, String)> {
    if travelled <= input::CLICK_SLOP {
        return None;
    }
    Some((from, landed?))
}

/// Adds the reference a finished drag asked for and opens the field that now holds it
/// with that node selected.
///
/// The one system in the overview that writes the document, as `input::canvas_commit`
/// is for the node graph. The node carries no edges, and sits where the panel's own
/// Add node button puts one with nothing selected. A pair the reference guard refuses
/// — one that would make the fields read each other in a circle, a card dragged onto
/// itself included — is reported, and the view is left where it was.
pub(super) fn commit_field_wire(
    mut document: ResMut<Document>,
    mut wired: ResMut<Wired>,
    mut step: ResMut<WiredStep>,
    mut open: MessageWriter<OpenField>,
) {
    let Some((source, target)) = wired.0.take() else {
        return;
    };
    let position = document
        .terrain()
        .and_then(|terrain| terrain.field(target.as_str()))
        .map(|field| field.graph.free_position_beside(None));
    let before = document.history().undo;
    let added = document.apply(&Edit::AddNode {
        field: target.clone(),
        op: NodeOp::FieldRef(watershed::FieldId::from(source.as_str())),
        position,
    });
    if let Ok(reply) = &added {
        let node = reply
            .get("node")
            .and_then(|node| node.as_str())
            .and_then(node_of_path);
        open.write(OpenField {
            field: target,
            select: node,
        });
        let now = document.history().undo;
        if now > before {
            step.0 = Some(now);
        }
    }
    report(&mut document, added.map(|_| ()));
}

/// Puts the overview back when the one change a drag made is undone.
///
/// The drag is one step: it adds a node and opens the field it landed in, so undoing it
/// has to take the view back as well as the node. Runs on every frame rather than only
/// while the overview is showing, because the view it restores is off at the point the
/// undo arrives.
pub(super) fn overview_after_undo(
    document: Res<Document>,
    mut overview: ResMut<Overview>,
    mut selection: ResMut<Selection>,
    mut step: ResMut<WiredStep>,
) {
    let Some(marked) = step.0 else {
        return;
    };
    let now = document.history().undo;
    if now < marked {
        overview.showing = true;
        selection.select(None);
        step.0 = None;
    } else if now > marked {
        step.0 = None;
    }
}

fn fingerprint(document: &Document) -> String {
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
    }
    key
}

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;

    use super::*;
    use crate::terrain::graph::NodeOp;
    use crate::terrain::{Field, TerrainSpec};
    use watershed::FieldId;

    fn height_reads_base_and_relief() -> TerrainSpec {
        TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("moisture").with_op(NodeOp::held(0.5)))
            .with_field(Field::new("base").with_op(NodeOp::held(0.25)))
            .with_field(Field::new("relief").with_op(NodeOp::held(0.75)))
            .with_field(
                Field::new("height")
                    .with_op(NodeOp::FieldRef(FieldId::from("base")))
                    .with_op(NodeOp::FieldRef(FieldId::from("relief"))),
            )
    }

    fn cyclic() -> TerrainSpec {
        TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("here").with_op(NodeOp::FieldRef(FieldId::from("there"))))
            .with_field(Field::new("there").with_op(NodeOp::FieldRef(FieldId::from("here"))))
    }

    fn at(placed: &[Placed], name: &str) -> Vec2 {
        placed
            .iter()
            .find(|card| card.field == name)
            .unwrap_or_else(|| panic!("no card for `{name}`"))
            .at
    }

    // The first acceptance criterion: what a field reads is drawn to its left, so the
    // document is read left to right in the order it is baked in rather than in the
    // order it happens to be declared in.
    #[test]
    fn a_field_is_placed_right_of_everything_it_reads() {
        let terrain = height_reads_base_and_relief();
        let (placed, cycle) = layout(&terrain);
        assert!(cycle.is_none(), "a document that orders reported a cycle");
        assert!(at(&placed, "base").x < at(&placed, "height").x);
        assert!(at(&placed, "relief").x < at(&placed, "height").x);
    }

    // The defect the scenario run found: a reference node wired into nothing is a read
    // the bake does not have to order, so `bake_order` can visit `moisture` before the
    // `base` it reads. The card still has to sit right of it, or the one ribbon added
    // by acceptance criterion six is drawn backwards between two cards in one column.
    #[test]
    fn a_field_is_placed_right_of_a_read_the_bake_order_visits_after_it() {
        let terrain = TerrainSpec::new(UVec2::splat(16))
            .with_field(
                Field::new("moisture")
                    .with_op(NodeOp::held(0.5))
                    .with_op(NodeOp::FieldRef(FieldId::from("base"))),
            )
            .with_field(Field::new("base").with_op(NodeOp::held(0.25)));
        let order: Vec<String> = terrain
            .bake_order()
            .expect("a dangling reference does not stop the order")
            .iter()
            .map(|id| id.to_string())
            .collect();
        assert_eq!(
            order,
            ["moisture", "base"],
            "the premise moved: the bake already orders the reader after the read"
        );

        let (placed, cycle) = layout(&terrain);
        assert!(cycle.is_none());
        assert!(at(&placed, "base").x < at(&placed, "moisture").x);
    }

    // The second half of that criterion: a field at neither end of the relation carries
    // no ribbon at all, rather than a stub to nothing.
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

    // One ribbon per declared read and no more: `height` reads two fields through two
    // reference nodes, so a double count here would draw two ribbons over each other.
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

    // Acceptance criterion eight: a document whose fields cannot be ordered still draws
    // every card — in declaration order, one per column — and says what the cycle is,
    // rather than showing an empty canvas with no account of why.
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

    // The hit test is what a press and a release are each resolved through, so it has
    // to answer the card the point is inside and nothing at all for the gap between
    // two — a press in the gap pans, and one that guessed a neighbour would not.
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

    // Acceptance criterion four, pinned at the arithmetic the key F runs: whatever the
    // fit answers has to put every card of a document wider than the viewport inside
    // the framed rectangle.
    #[test]
    fn a_fit_puts_every_field_card_of_a_wide_document_in_view() {
        let mut terrain = TerrainSpec::new(UVec2::splat(16));
        let mut previous: Option<String> = None;
        for step in 0..8 {
            let name = format!("f{step}");
            let field = match &previous {
                Some(read) => Field::new(name.as_str())
                    .with_op(NodeOp::FieldRef(FieldId::from(read.as_str()))),
                None => Field::new(name.as_str()).with_op(NodeOp::held(0.5)),
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

    fn overview_app(terrain: TerrainSpec) -> App {
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
        app.init_resource::<CanvasShape>();
        app.init_resource::<Grab>();
        app
    }

    // Acceptance criterion nine: the overview is a way of looking, so building it leaves
    // the history, the dirty flag and what is baked exactly where they were — otherwise
    // glancing at the document would be something that has to be saved.
    #[test]
    fn drawing_the_overview_makes_no_edit_and_starts_no_bake() {
        let mut app = overview_app(height_reads_base_and_relief());
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

    // A card carries the name rather than an index, so #43's drag and the double-click
    // both address a field that survives the cards being rebuilt under them.
    #[test]
    fn every_field_gets_a_card_carrying_its_name() {
        let mut app = overview_app(height_reads_base_and_relief());
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

    // The fingerprint is what brings the overview up to date without it being switched
    // off and on, so adding a field and adding a reference each have to move it —
    // acceptance criteria five and six.
    #[test]
    fn the_fingerprint_moves_when_a_field_or_a_reference_is_added() {
        let mut document = Document::default();
        document.adopt(height_reads_base_and_relief());
        let before = fingerprint(&document);

        let mut added = Document::default();
        added.adopt(height_reads_base_and_relief().with_field(Field::new("temperature")));
        assert_ne!(
            fingerprint(&added),
            before,
            "a new field left the key alone"
        );

        let mut read = Document::default();
        read.adopt(
            TerrainSpec::new(UVec2::splat(16))
                .with_field(Field::new("moisture").with_op(NodeOp::FieldRef(FieldId::from("base"))))
                .with_field(Field::new("base").with_op(NodeOp::held(0.25)))
                .with_field(Field::new("relief").with_op(NodeOp::held(0.75)))
                .with_field(
                    Field::new("height")
                        .with_op(NodeOp::FieldRef(FieldId::from("base")))
                        .with_op(NodeOp::FieldRef(FieldId::from("relief"))),
                ),
        );
        assert_ne!(
            fingerprint(&read),
            before,
            "a new reference left the key alone"
        );
    }
    fn wired_world(terrain: TerrainSpec, active: &str) -> World {
        let mut document = Document::default();
        document.adopt(terrain);
        document.set_active(active).unwrap();

        let mut world = World::new();
        world.insert_resource(document);
        world.init_resource::<Wired>();
        world.init_resource::<WiredStep>();
        world.insert_resource(Overview { showing: true });
        world.init_resource::<Selection>();
        world.init_resource::<Messages<OpenField>>();
        world
    }

    fn with_temperature() -> TerrainSpec {
        height_reads_base_and_relief().with_field(Field::new("temperature"))
    }

    fn nodes_of(world: &World, field: &str) -> Vec<NodeOp> {
        world
            .resource::<Document>()
            .terrain()
            .and_then(|terrain| terrain.field(field))
            .map(|field| {
                field
                    .graph
                    .nodes
                    .iter()
                    .map(|node| node.op.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    // Acceptance criterion one, at the decision alone: a press that travelled and was
    // released over another card asks for that field to read the one the drag left.
    #[test]
    fn a_drag_between_two_cards_asks_for_the_dependency_it_crossed() {
        let asked = wired_by(
            "moisture".to_owned(),
            input::CLICK_SLOP * 4.0,
            Some("temperature".to_owned()),
        );
        assert_eq!(
            asked,
            Some(("moisture".to_owned(), "temperature".to_owned()))
        );
    }

    // A press that never travelled is how a card is opened, so it must ask for nothing —
    // otherwise every click on a card would try to add a reference to it.
    #[test]
    fn a_press_that_did_not_travel_asks_for_nothing() {
        assert_eq!(
            wired_by("moisture".to_owned(), 0.0, Some("moisture".to_owned())),
            None
        );
    }

    // Acceptance criterion five: a release anywhere but on a card does nothing.
    #[test]
    fn a_release_between_cards_asks_for_nothing() {
        assert_eq!(
            wired_by("moisture".to_owned(), input::CLICK_SLOP * 4.0, None),
            None
        );
    }

    // Acceptance criterion one, end to end: the reference lands in the field the drag
    // arrived at, and that field is opened with the new node selected.
    #[test]
    fn a_wired_pair_adds_the_reference_and_opens_the_field_it_landed_in() {
        let mut world = wired_world(with_temperature(), "moisture");
        world.resource_mut::<Wired>().0 = Some(("moisture".to_owned(), "temperature".to_owned()));
        world.run_system_once(commit_field_wire).unwrap();
        world
            .run_system_once(super::super::apply_open_field)
            .unwrap();

        assert_eq!(world.resource::<Document>().error(), None);
        let held = nodes_of(&world, "temperature");
        assert!(
            matches!(held.as_slice(), [NodeOp::FieldRef(read)] if read.as_str() == "moisture"),
            "{held:?}"
        );
        let document = world.resource::<Document>();
        assert_eq!(document.active(), "temperature");
        assert_eq!(document.history().undo, 1, "the drag is one change");
        let selected = world.resource::<Selection>().node;
        assert_eq!(
            selected,
            Some(
                world
                    .resource::<Document>()
                    .terrain()
                    .unwrap()
                    .field("temperature")
                    .unwrap()
                    .graph
                    .nodes[0]
                    .id
            ),
            "the node the drag added is not the selected one"
        );
        assert!(!world.resource::<Overview>().showing);
    }

    // Acceptance criterion six: the whole drag is one undo step, and undoing it puts
    // back the view the drag was made from as well as taking the node out.
    #[test]
    fn undoing_the_drag_removes_the_node_and_returns_to_the_overview() {
        let mut world = wired_world(with_temperature(), "moisture");
        world.resource_mut::<Wired>().0 = Some(("moisture".to_owned(), "temperature".to_owned()));
        world.run_system_once(commit_field_wire).unwrap();
        world
            .run_system_once(super::super::apply_open_field)
            .unwrap();

        world.resource_mut::<Document>().undo().unwrap();
        world.run_system_once(overview_after_undo).unwrap();

        assert!(nodes_of(&world, "temperature").is_empty());
        assert_eq!(world.resource::<Document>().history().undo, 0);
        assert!(world.resource::<Overview>().showing);
        assert_eq!(world.resource::<Selection>().node, None);
    }

    // Acceptance criterion three: the drag goes through the same reference guard the
    // panel and the socket do, so a pair that would make the two fields read each other
    // in a circle is refused, named, and adds nothing.
    #[test]
    fn a_drag_that_would_close_a_cycle_is_refused_and_named() {
        let mut world = wired_world(height_reads_base_and_relief(), "moisture");
        let before = nodes_of(&world, "base").len();
        world.resource_mut::<Wired>().0 = Some(("height".to_owned(), "base".to_owned()));
        world.run_system_once(commit_field_wire).unwrap();

        let document = world.resource::<Document>();
        let refusal = document.error().expect("a cycle is refused");
        assert!(refusal.contains("base -> height -> base"), "{refusal}");
        assert_eq!(nodes_of(&world, "base").len(), before);
        assert_eq!(world.resource::<Document>().active(), "moisture");
        assert_eq!(world.resource::<Document>().history().undo, 0);
        assert!(world.resource::<Overview>().showing);
    }

    // Acceptance criterion four: a card dragged onto itself reaches the same guard, and
    // the shortest cycle there is is refused like any other.
    #[test]
    fn a_card_dragged_onto_itself_is_refused() {
        let mut world = wired_world(height_reads_base_and_relief(), "moisture");
        world.resource_mut::<Wired>().0 = Some(("moisture".to_owned(), "moisture".to_owned()));
        world.run_system_once(commit_field_wire).unwrap();

        let document = world.resource::<Document>();
        let refusal = document.error().expect("a self-reference is refused");
        assert!(refusal.contains("moisture -> moisture"), "{refusal}");
        assert!(nodes_of(&world, "moisture").len() == 1, "a node was added");
        assert_eq!(document.history().undo, 0);
    }
}
