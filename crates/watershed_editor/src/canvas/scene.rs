//! What one field's graph is drawn as, and how the drawing is kept in step with it.

use bevy::asset::AssetServer;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::feathers::constants::fonts;
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::FontSize;

use super::{
    CANVAS_LAYER, CARD, CanvasLabel, CanvasShape, DETAIL_SIZE, Grab, NodeCard, NodeEdge, NodePin,
    PIN_RADIUS, Selection, TITLE_BAR, TITLE_SIZE, THUMB, open_graph,
};
use crate::document::Document;
use crate::edit::{op_name, op_summary};
use crate::terrain::graph::{FieldGraph, GraphNode, NodeId, NodeOp};

const BODY: Color = Color::srgb(0.16, 0.17, 0.21);
const SOURCE: Color = Color::srgb(0.29, 0.42, 0.62);
const OPERATOR: Color = Color::srgb(0.46, 0.35, 0.24);
const OUTPUT: Color = Color::srgb(0.30, 0.55, 0.38);
const SOLOED: Color = Color::srgb(0.72, 0.58, 0.22);
const SELECTED: Color = Color::srgb(0.85, 0.87, 0.95);
const WIRE: Color = Color::srgb(0.48, 0.64, 0.86);
const PIN: Color = Color::srgb(0.78, 0.81, 0.88);

/// Everything the canvas owns, so a rebuild can take it all down in one query.
#[derive(Component)]
pub struct CanvasOwned;

/// The title bar of a card, which is what carries the output and solo marks.
#[derive(Component)]
pub struct CardTitleBar(pub NodeId);

/// The text on a card that names the node.
#[derive(Component)]
pub struct CardTitle(pub NodeId);

/// The text on a card that says what the op is set to.
#[derive(Component)]
pub struct CardDetail(pub NodeId);

/// Respawns the cards, pins and edges when the open field's graph changes shape.
///
/// Driven by a fingerprint rather than by an event, because an edit can arrive from
/// the panel, from the control socket or from a preset load, and a fingerprint catches
/// all three without every one of them having to remember to announce itself.
pub fn rebuild_canvas(
    mut commands: Commands,
    document: Res<Document>,
    assets: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut shape: ResMut<CanvasShape>,
    mut selection: ResMut<Selection>,
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

    let Some(graph) = open_graph(&document) else {
        selection.select(None);
        return;
    };
    if let Some(node) = selection.node
        && graph.node(node).is_none()
    {
        selection.forget(node);
    }
    if let Some(node) = selection.soloed
        && graph.node(node).is_none()
    {
        selection.soloed = None;
    }

    let font = assets.load(fonts::REGULAR);
    let pin_mesh = meshes.add(Circle::new(PIN_RADIUS));
    let pin_material = materials.add(PIN);
    let wire_material = materials.add(WIRE);

    let mut cards = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let entity = spawn_card(
            &mut commands,
            &pin_mesh,
            &pin_material,
            &font,
            node,
            graph.output == Some(node.id),
        );
        cards.push((node.id, entity));
    }

    let entity_of = |id: NodeId| {
        cards
            .iter()
            .find(|(held, _)| *held == id)
            .map(|(_, entity)| *entity)
    };
    for node in &graph.nodes {
        for (pin, source) in node.inputs.iter().enumerate() {
            let (Some(source), Some(to)) = (source, entity_of(node.id)) else {
                continue;
            };
            let Some(from) = entity_of(*source) else {
                continue;
            };
            commands.spawn((
                Mesh2d(meshes.add(super::edges::blank_ribbon())),
                MeshMaterial2d(wire_material.clone()),
                Transform::from_xyz(0.0, 0.0, -1.0),
                NoFrustumCulling,
                RenderLayers::layer(CANVAS_LAYER),
                NodeEdge { from, to, pin },
                CanvasOwned,
            ));
        }
    }
}

/// Writes what a card shows without respawning it.
///
/// Position and name reach the document but not the bake, so they must not reach a
/// rebuild either: dragging a card would otherwise despawn every card on the canvas
/// the moment the button came up.
///
/// The card a drag is holding is left alone. While it is held the drag owns where it
/// sits, and writing the document's position over it would put the card back under the
/// cursor's own hand — and on the frame the button came up, back where it started.
pub fn sync_canvas(
    document: Res<Document>,
    selection: Res<Selection>,
    grab: Res<Grab>,
    mut cards: Query<(&NodeCard, &mut Transform)>,
    mut bars: Query<(&CardTitleBar, &mut Sprite)>,
    mut titles: Query<(&CardTitle, &mut Text2d)>,
    mut details: Query<(&CardDetail, &mut Text2d), Without<CardTitle>>,
) {
    let Some(graph) = open_graph(&document) else {
        return;
    };
    let held = match *grab {
        Grab::Card { node, .. } => Some(node),
        _ => None,
    };
    for (card, mut transform) in &mut cards {
        let Some(node) = graph.node(card.node) else {
            continue;
        };
        if held == Some(card.node) {
            continue;
        }
        transform.translation.x = node.position[0];
        transform.translation.y = node.position[1];
    }
    for (bar, mut sprite) in &mut bars {
        sprite.color = title_colour(graph, &selection, bar.0);
    }
    for (title, mut text) in &mut titles {
        if let Some(node) = graph.node(title.0) {
            **text = caption(node);
        }
    }
    for (detail, mut text) in &mut details {
        if let Some(node) = graph.node(detail.0) {
            **text = op_summary(&node.op);
        }
    }
}

/// What a card is called: the name a person gave it, or the op it carries.
fn caption(node: &GraphNode) -> String {
    match &node.name {
        Some(name) => name.clone(),
        None => op_name(&node.op).to_owned(),
    }
}

/// The colour of a card's title bar, which is where a card says what it is.
///
/// The output node and the soloed one are marked here, and the mark is on the bar
/// rather than on a label so it survives the zoom at which labels are dropped.
fn title_colour(graph: &FieldGraph, selection: &Selection, node: NodeId) -> Color {
    if selection.soloed == Some(node) {
        return SOLOED;
    }
    if selection.node == Some(node) {
        return SELECTED;
    }
    if graph.output == Some(node) {
        return OUTPUT;
    }
    match graph.node(node).map(|node| node.op.arity()) {
        Some(0) => SOURCE,
        _ => OPERATOR,
    }
}

fn spawn_card(
    commands: &mut Commands,
    pin_mesh: &Handle<Mesh>,
    pin_material: &Handle<ColorMaterial>,
    font: &Handle<Font>,
    node: &GraphNode,
    is_output: bool,
) -> Entity {
    let inputs = node.inputs.len();
    let card = NodeCard {
        node: node.id,
        size: CARD,
        inputs,
    };
    let title_y = (CARD.y - TITLE_BAR) * 0.5;
    let row_y = -TITLE_BAR * 0.5 - 2.0;
    let accent = if is_output {
        OUTPUT
    } else if inputs == 0 {
        SOURCE
    } else {
        OPERATOR
    };
    let dim = if node.bypassed { 0.55 } else { 1.0 };

    let entity = commands
        .spawn((
            Sprite {
                color: BODY.with_alpha(dim),
                custom_size: Some(CARD),
                ..default()
            },
            Transform::from_translation(Vec3::new(node.position[0], node.position[1], 0.0)),
            RenderLayers::layer(CANVAS_LAYER),
            card,
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
            CardTitleBar(node.id),
        ));
        parent.spawn((
            Text2d::new(caption(node)),
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
            CardTitle(node.id),
        ));
        parent.spawn((
            Sprite {
                color: accent.with_alpha(0.35),
                custom_size: Some(Vec2::splat(THUMB)),
                ..default()
            },
            Transform::from_xyz(-CARD.x * 0.5 + THUMB * 0.5 + 12.0, row_y, 0.01),
            RenderLayers::layer(CANVAS_LAYER),
        ));
        parent.spawn((
            Text2d::new(op_summary(&node.op)),
            TextFont {
                font: bevy::text::FontSource::Handle(font.clone()),
                font_size: FontSize::Px(DETAIL_SIZE),
                ..default()
            },
            TextColor(Color::srgb(0.72, 0.76, 0.85)),
            Anchor::CENTER_LEFT,
            Transform::from_xyz(-CARD.x * 0.5 + THUMB + 20.0, row_y, 0.02),
            RenderLayers::layer(CANVAS_LAYER),
            CanvasLabel(DETAIL_SIZE),
            CardDetail(node.id),
        ));

        let output = super::output_offset(&card);
        parent.spawn((
            Mesh2d(pin_mesh.clone()),
            MeshMaterial2d(pin_material.clone()),
            Transform::from_xyz(output.x, output.y, 0.03),
            RenderLayers::layer(CANVAS_LAYER),
            NodePin {
                node: node.id,
                input: None,
            },
        ));
        for index in 0..inputs {
            let at = super::input_offset(&card, index);
            parent.spawn((
                Mesh2d(pin_mesh.clone()),
                MeshMaterial2d(pin_material.clone()),
                Transform::from_xyz(at.x, at.y, 0.03),
                RenderLayers::layer(CANVAS_LAYER),
                NodePin {
                    node: node.id,
                    input: Some(index),
                },
            ));
        }
    });

    entity
}

/// What the canvas is drawn from, as one string.
///
/// Everything that decides how many entities there are and what each one is, and
/// nothing that only decides where they sit or what they are called.
fn fingerprint(document: &Document) -> String {
    let mut key = String::new();
    key.push_str(document.active());
    let Some(graph) = open_graph(document) else {
        key.push_str("|none");
        return key;
    };
    key.push_str(&format!(
        "|out:{}",
        graph
            .output
            .map_or_else(|| "none".to_owned(), |id| id.to_string())
    ));
    for node in &graph.nodes {
        key.push_str(&format!(
            "|{}:{}:{}:{:?}",
            node.id,
            op_name(&node.op),
            node.bypassed,
            node.inputs,
        ));
        if let NodeOp::Regions { output, .. } = &node.op {
            key.push_str(&crate::edit::region_output_name(output));
        }
    }
    key
}

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;

    use super::*;
    use crate::terrain::graph::NodeOp;
    use crate::terrain::{Field, TerrainSpec};

    /// A world holding one field of one node, with that node's card on the canvas at
    /// `dragged_to`, and a drag holding that card when `held`.
    fn world_with(held: bool, dragged_to: Vec2) -> (World, Entity) {
        let mut document = Document::default();
        document.adopt(
            TerrainSpec::new(UVec2::splat(16))
                .with_field(Field::new("height").with_op(NodeOp::Constant(0.5))),
        );
        let node = document
            .terrain()
            .unwrap()
            .field("height")
            .unwrap()
            .graph
            .nodes[0]
            .id;

        let mut world = World::new();
        world.insert_resource(document);
        world.insert_resource(Selection::default());
        let card = world
            .spawn((
                NodeCard {
                    node,
                    size: CARD,
                    inputs: 0,
                },
                Transform::from_translation(dragged_to.extend(0.0)),
            ))
            .id();
        world.insert_resource(if held {
            Grab::Card {
                entity: card,
                node,
                offset: Vec2::ZERO,
                at: dragged_to,
            }
        } else {
            Grab::Idle
        });
        (world, card)
    }

    fn at(world: &World, card: Entity) -> Vec2 {
        world
            .get::<Transform>(card)
            .unwrap()
            .translation
            .truncate()
    }

    // The defect this guards was the whole of "I drag a node and it jumps back": the
    // sync runs before the drag, so on the frame the button comes up it would write the
    // document's position over the card the drag had just placed — putting it back
    // where it was picked up, and taking the release's answer with it.
    #[test]
    fn the_card_a_drag_is_holding_is_left_where_the_drag_put_it() {
        let dragged_to = Vec2::new(120.0, -80.0);
        let (mut world, card) = world_with(true, dragged_to);
        world.run_system_once(sync_canvas).unwrap();
        assert_eq!(at(&world, card), dragged_to);
    }

    // And the other half: a card nothing is holding follows the document, which is what
    // makes a position written by a control verb show up on the canvas at all.
    #[test]
    fn a_card_nothing_is_holding_follows_the_document() {
        let (mut world, card) = world_with(false, Vec2::new(120.0, -80.0));
        world.run_system_once(sync_canvas).unwrap();
        assert_eq!(at(&world, card), Vec2::ZERO);
    }
}
