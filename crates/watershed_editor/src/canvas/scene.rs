//! What one field's graph is drawn as, and how the drawing is kept in step with it.

use bevy::asset::AssetServer;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::feathers::constants::fonts;
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::FontSize;

use super::{
    CANVAS_LAYER, CARD, CanvasLabel, CanvasShape, DETAIL_SIZE, Grab, NodeCard, NodeEdge, NodePin,
    PIN_RADIUS, Selection, THUMB, TITLE_BAR, TITLE_SIZE, open_graph,
};
use crate::document::Document;
use crate::edit::{op_name, op_summary};
use crate::gpu::ShaderLibrary;
use crate::terrain::graph::{FieldGraph, GraphNode, NodeId, NodeOp};

const BODY: Color = Color::srgb(0.16, 0.17, 0.21);
const SOURCE: Color = Color::srgb(0.29, 0.42, 0.62);
const OPERATOR: Color = Color::srgb(0.46, 0.35, 0.24);
const OUTPUT: Color = Color::srgb(0.30, 0.55, 0.38);
const SOLOED: Color = Color::srgb(0.72, 0.58, 0.22);
const SELECTED: Color = Color::srgb(0.85, 0.87, 0.95);
const WIRE: Color = Color::srgb(0.48, 0.64, 0.86);
const PIN: Color = Color::srgb(0.78, 0.81, 0.88);
const BROKEN: Color = Color::srgb(0.72, 0.24, 0.24);
const FAULT: Color = Color::srgb(1.0, 0.74, 0.72);

const FAULT_SIZE: f32 = 11.0;
const FAULT_CHARS: usize = 34;

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

/// The text on a card that says why its shader will not compile. Empty while the
/// shader is good.
#[derive(Component)]
pub struct CardFault(pub NodeId);

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
    mut faults: Query<(&CardFault, &mut Text2d), (Without<CardTitle>, Without<CardDetail>)>,
    library: Option<Res<ShaderLibrary>>,
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
        sprite.color = title_colour(graph, &selection, library.as_deref(), bar.0);
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
    for (fault, mut text) in &mut faults {
        if let Some(node) = graph.node(fault.0) {
            **text = fault_of(node, library.as_deref())
                .map(fault_line)
                .unwrap_or_default();
        }
    }
}

fn fault_of<'a>(node: &GraphNode, library: Option<&'a ShaderLibrary>) -> Option<&'a str> {
    let NodeOp::Shader(shader) = &node.op else {
        return None;
    };
    library?.entry(&shader.file)?.error.as_deref()
}

fn fault_line(fault: &str) -> String {
    let first = fault.lines().next().unwrap_or("").trim();
    if first.chars().count() <= FAULT_CHARS {
        return first.to_owned();
    }
    first
        .chars()
        .take(FAULT_CHARS - 1)
        .chain(std::iter::once('…'))
        .collect()
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
/// rather than on a label so it survives the zoom at which labels are dropped. A node
/// whose shader will not compile outranks all three, so clicking the card to find out
/// what is wrong with it does not take the mark away.
fn title_colour(
    graph: &FieldGraph,
    selection: &Selection,
    library: Option<&ShaderLibrary>,
    node: NodeId,
) -> Color {
    if graph
        .node(node)
        .is_some_and(|node| fault_of(node, library).is_some())
    {
        return BROKEN;
    }
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
        if matches!(node.op, NodeOp::Shader(_)) {
            parent.spawn((
                Text2d::new(String::new()),
                TextFont {
                    font: bevy::text::FontSource::Handle(font.clone()),
                    font_size: FontSize::Px(FAULT_SIZE),
                    ..default()
                },
                TextColor(FAULT),
                Anchor::CENTER_LEFT,
                Transform::from_xyz(-CARD.x * 0.5 + 12.0, -CARD.y * 0.5 + 7.0, 0.02),
                RenderLayers::layer(CANVAS_LAYER),
                CanvasLabel(FAULT_SIZE),
                CardFault(node.id),
            ));
        }

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
    use crate::gpu::ShaderLibrary;
    use crate::terrain::graph::NodeOp;
    use crate::terrain::shader::ShaderLayer;
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
        world.get::<Transform>(card).unwrap().translation.truncate()
    }

    fn shader_world(fault: Option<&str>) -> (World, Entity, Entity) {
        let mut document = Document::default();
        document.adopt(TerrainSpec::new(UVec2::splat(16)).with_field(
            Field::new("height").with_op(NodeOp::Shader(ShaderLayer::new("broken.wgsl"))),
        ));
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
        world.insert_resource(Grab::Idle);
        world.insert_resource(match fault {
            Some(fault) => ShaderLibrary::with_fault("broken.wgsl", fault),
            None => ShaderLibrary::default(),
        });
        let text = world
            .spawn((Text2d::new(String::new()), CardFault(node)))
            .id();
        let bar = world
            .spawn((
                Sprite {
                    color: BODY,
                    ..default()
                },
                CardTitleBar(node),
            ))
            .id();
        (world, text, bar)
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

    // The point of the task: the fault the compiler reported reaches the card, which is
    // where the graph is being read, instead of only the status bar line that has
    // already scrolled away by the time the node is looked at.
    #[test]
    fn the_card_of_a_shader_that_will_not_compile_shows_the_fault_and_reddens_the_bar() {
        let (mut world, text, bar) = shader_world(Some("line 2: unknown type: 'vec4'"));
        world.run_system_once(sync_canvas).unwrap();
        let shown = world.get::<Text2d>(text).unwrap();
        assert!(shown.starts_with("line 2:"), "card shows {:?}", shown.0);
        assert_eq!(world.get::<Sprite>(bar).unwrap().color, BROKEN);
    }

    // The second half of the acceptance: fixing the file has to clear the card by
    // itself, so the good path writes the empty string rather than leaving the last
    // fault on the card until something else respawns it.
    #[test]
    fn a_card_whose_shader_compiles_carries_no_fault_line() {
        let (mut world, text, bar) = shader_world(None);
        world.run_system_once(sync_canvas).unwrap();
        assert_eq!(world.get::<Text2d>(text).unwrap().0, "");
        assert_ne!(world.get::<Sprite>(bar).unwrap().color, BROKEN);
    }

    // A naga message can run to a paragraph and a card is 210 units wide, so the line
    // has to be cut to fit — and cut by character, since a message can quote a token
    // that is not ASCII.
    #[test]
    fn fault_line_keeps_the_first_line_and_cuts_it_to_the_cards_width() {
        let line =
            fault_line("line 7: expected ‘;’ but found an identifier of some length\nnote: here");
        assert!(!line.contains('\n'), "{line:?} spans lines");
        assert!(line.chars().count() <= FAULT_CHARS, "{line:?} is too wide");
        assert!(line.ends_with('…'), "{line:?} is not marked as cut");
    }
}
