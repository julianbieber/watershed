//! What one field's graph is drawn as, and how the drawing is kept in step with it.

use bevy::asset::AssetServer;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::feathers::constants::fonts;
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::FontSize;

use super::thumb::CardThumb;
use super::{
    CANVAS_LAYER, CARD, CanvasLabel, CanvasShape, DETAIL_SIZE, Grab, MARGIN, NodeCard, NodeEdge,
    NodePin, PARAM_SIZE, PIN_RADIUS, ROW_STEP, Selection, THUMB, TITLE_BAR, TITLE_SIZE, open_graph,
};
use crate::document::Document;
use crate::edit::{op_name, op_params};
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

const DETAIL: Color = Color::srgb(0.72, 0.76, 0.85);
const PARAM: Color = Color::srgb(0.62, 0.66, 0.76);

const FAULT_SIZE: f32 = 11.0;
const FAULT_CHARS: usize = 34;
const PARAM_CHARS: usize = 23;
const FILE_CHARS: usize = 21;

/// Everything the canvas owns, so a rebuild can take it all down in one query.
#[derive(Component)]
pub struct CanvasOwned;

/// The title bar of a card, which is what carries the output and solo marks.
#[derive(Component)]
pub struct CardTitleBar(pub NodeId);

/// Which of a card's lines a label is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LineKind {
    /// The name in the title bar.
    Title,
    /// The op's own name.
    Op,
    /// What the op is set to.
    Params,
    /// The file a shader node runs. Only a shader node carries one.
    File,
}

/// One line of text on a card: which node it reads from and which line it is.
#[derive(Component)]
pub struct CardLine {
    /// The node the line is read from.
    pub node: NodeId,
    /// Which line of the card it is.
    pub kind: LineKind,
}

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
    mut lines: Query<(&CardLine, &mut Text2d)>,
    mut faults: Query<(&CardFault, &mut Text2d), Without<CardLine>>,
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
        sprite.color = title_colour(&document, graph, &selection, library.as_deref(), bar.0);
    }
    for (line, mut text) in &mut lines {
        let Some(node) = graph.node(line.node) else {
            continue;
        };
        **text = match line.kind {
            LineKind::Title => caption(node),
            LineKind::Op => op_name(&node.op).to_owned(),
            LineKind::Params => clip(&params_line(node, library.as_deref()), PARAM_CHARS),
            LineKind::File => match &node.op {
                NodeOp::Shader(shader) => clip(&shader.file, FILE_CHARS),
                _ => String::new(),
            },
        };
    }
    for (fault, mut text) in &mut faults {
        if let Some(node) = graph.node(fault.0) {
            **text = fault_of(&document, node, library.as_deref())
                .map(|fault| fault_line(&fault))
                .unwrap_or_default();
        }
    }
}

fn fault_of(
    document: &Document,
    node: &GraphNode,
    library: Option<&ShaderLibrary>,
) -> Option<String> {
    match &node.op {
        NodeOp::Shader(shader) => library?.entry(&shader.file)?.error.clone(),
        NodeOp::FieldRef(id) => document
            .terrain()
            .filter(|terrain| terrain.field(id.as_str()).is_none())
            .map(|_| format!("no field named `{id}`")),
        _ => None,
    }
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

fn params_line(node: &GraphNode, library: Option<&ShaderLibrary>) -> String {
    let NodeOp::Shader(shader) = &node.op else {
        return op_params(&node.op);
    };
    shader.params_line(
        library
            .and_then(|held| held.entry(&shader.file))
            .map(|entry| &entry.layout),
    )
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
/// rather than on a label so it survives the zoom at which labels are dropped. A broken
/// node outranks all three, so clicking the card to find out what is wrong with it does
/// not take the mark away — broken being a shader that will not compile or a reference
/// to a field the document does not carry.
fn title_colour(
    document: &Document,
    graph: &FieldGraph,
    selection: &Selection,
    library: Option<&ShaderLibrary>,
    node: NodeId,
) -> Color {
    if graph
        .node(node)
        .is_some_and(|node| fault_of(document, node, library).is_some())
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
            CardLine {
                node: node.id,
                kind: LineKind::Title,
            },
        ));
        let blank = accent.with_alpha(0.35);
        parent.spawn((
            Sprite {
                color: blank,
                custom_size: Some(Vec2::splat(THUMB)),
                ..default()
            },
            Transform::from_translation(thumb_centre().extend(0.01)),
            RenderLayers::layer(CANVAS_LAYER),
            CardThumb::new(node.id, blank),
        ));
        let row = |parent: &mut ChildSpawnerCommands,
                   index: usize,
                   kind: LineKind,
                   size: f32,
                   colour: Color,
                   text: String| {
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
                CardLine {
                    node: node.id,
                    kind,
                },
            ));
        };
        row(
            parent,
            0,
            LineKind::Op,
            DETAIL_SIZE,
            DETAIL,
            op_name(&node.op).to_owned(),
        );
        row(
            parent,
            1,
            LineKind::Params,
            PARAM_SIZE,
            PARAM,
            clip(&op_params(&node.op), PARAM_CHARS),
        );
        if let NodeOp::Shader(shader) = &node.op {
            row(
                parent,
                2,
                LineKind::File,
                DETAIL_SIZE,
                DETAIL,
                clip(&shader.file, FILE_CHARS),
            );
        }
        if matches!(node.op, NodeOp::Shader(_) | NodeOp::FieldRef(_)) {
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
    use watershed::FieldId;

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

    fn line_of(world: &mut World, kind: LineKind) -> String {
        world
            .query::<(&CardLine, &Text2d)>()
            .iter(world)
            .find(|(line, _)| line.kind == kind)
            .map(|(_, text)| text.0.clone())
            .expect("a line of that kind")
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
        for kind in [LineKind::Op, LineKind::Params, LineKind::File] {
            world.spawn((Text2d::new(String::new()), CardLine { node, kind }));
        }
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

    fn fieldref_world(target_exists: bool) -> (World, Entity, Entity) {
        let mut terrain = TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("height").with_op(NodeOp::FieldRef(FieldId::from("base"))));
        if target_exists {
            terrain = terrain.with_field(Field::new("base").with_op(NodeOp::Constant(0.25)));
        }
        let mut document = Document::default();
        document.adopt(terrain);
        document.set_active("height").unwrap();
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
        world.insert_resource(ShaderLibrary::default());
        let text = world
            .spawn((Text2d::new(String::new()), CardFault(node)))
            .id();
        for kind in [LineKind::Op, LineKind::Params, LineKind::File] {
            world.spawn((Text2d::new(String::new()), CardLine { node, kind }));
        }
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

    // Acceptance criterion six: a reference naming a field the document does not carry
    // is broken the way a shader that will not compile is, rather than a blank card
    // giving no sign that the graph cannot be baked.
    #[test]
    fn a_reference_to_a_missing_field_shows_a_fault_and_reddens_the_bar() {
        let (mut world, text, bar) = fieldref_world(false);
        world.run_system_once(sync_canvas).unwrap();
        assert_eq!(
            world.get::<Text2d>(text).unwrap().0,
            "no field named `base`"
        );
        assert_eq!(world.get::<Sprite>(bar).unwrap().color, BROKEN);
    }

    // And a reference that resolves is an ordinary card: putting the field back has to
    // clear the fault by itself, the way fixing a shader does.
    #[test]
    fn a_reference_to_a_field_that_is_there_carries_no_fault() {
        let (mut world, text, bar) = fieldref_world(true);
        world.run_system_once(sync_canvas).unwrap();
        assert_eq!(world.get::<Text2d>(text).unwrap().0, "");
        assert_ne!(world.get::<Sprite>(bar).unwrap().color, BROKEN);
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

    // The acceptance in unit form: a shader card says which file it runs and what op
    // it is, on lines of their own, rather than leaving both to a title bar a person
    // may have renamed.
    #[test]
    fn a_shader_card_names_its_file_and_its_op_on_their_own_lines() {
        let (mut world, _, _) = shader_world(None);
        world.run_system_once(sync_canvas).unwrap();
        assert_eq!(line_of(&mut world, LineKind::File), "broken.wgsl");
        assert_eq!(line_of(&mut world, LineKind::Op), "shader");
    }

    // Every line of a card has to sit inside the card and clear the line above it,
    // including the frame a shader is broken and the fault line is on the card with
    // all the rest — which is the layout the constants are easiest to nudge out of.
    #[test]
    fn every_line_of_a_card_sits_inside_it_and_clears_the_one_above() {
        let title_y = (CARD.y - TITLE_BAR) * 0.5;
        let fault_y = -CARD.y * 0.5 + 7.0;
        let low = -CARD.y * 0.5;
        let high = CARD.y * 0.5;
        let rows: Vec<f32> = (0..3).map(row_y).collect();

        for y in rows.iter().copied().chain([title_y, fault_y]) {
            assert!(y > low && y < high, "a line at {y} left the card");
        }
        assert!(
            title_y - TITLE_SIZE > rows[0],
            "the op row is under the bar"
        );
        for pair in rows.windows(2) {
            assert!(pair[0] - pair[1] >= DETAIL_SIZE, "rows {pair:?} overlap");
        }
        let thumb = thumb_centre();
        assert!(
            thumb.y - THUMB * 0.5 > fault_y + FAULT_SIZE * 0.5,
            "the picture covers the fault line"
        );
        assert!(
            thumb.x - THUMB * 0.5 > -CARD.x * 0.5
                && thumb.y + THUMB * 0.5 < title_y - TITLE_BAR * 0.5,
            "the picture left the card's body"
        );
        assert!(
            text_left() > thumb.x + THUMB * 0.5,
            "the text is on the picture"
        );
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
