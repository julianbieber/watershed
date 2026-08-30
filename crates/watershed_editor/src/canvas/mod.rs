//! The node graph as a thing on screen: what the open field's graph is drawn as, how
//! it is panned, zoomed and rewired, and how one node's values are put on the map.

mod edges;
mod input;
mod scene;

use bevy::camera::visibility::RenderLayers;
use bevy::camera::{Camera, ClearColorConfig, Viewport};
use bevy::prelude::*;
use bevy::ui::UiSystems;

use crate::document::{Document, EditorSystems};
use crate::terrain::graph::NodeId;

/// The width and height of a card, in canvas units.
pub const CARD: Vec2 = Vec2::new(210.0, 118.0);
const TITLE_BAR: f32 = 30.0;
const THUMB: f32 = 60.0;
const TITLE_SIZE: f32 = 17.0;
const DETAIL_SIZE: f32 = 13.0;
const PIN_RADIUS: f32 = 7.0;
const ZOOM_PER_STEP: f32 = 1.2;
const MIN_SCALE: f32 = 0.05;
const MAX_SCALE: f32 = 8.0;

/// The zoom below which a card is drawn as a chip and its labels are dropped.
///
/// The floor the prototype stayed readable to. Below it a label costs a text layout
/// nobody can read, so the labels come off rather than being drawn illegibly.
const CHIP_ZOOM: f32 = 0.40;

/// The render layer the canvas draws on, so the map's camera does not draw it and the
/// canvas camera does not draw the map.
const CANVAS_LAYER: usize = 1;

/// What the canvas is drawn on.
///
/// Deliberately not the map's ground: the two sit one above the other in the same
/// window, and a person has to be able to tell at a glance which of them the pointer
/// is about to act on.
const CANVAS_GROUND: Color = Color::srgb(0.115, 0.125, 0.165);

/// Draws the open field's graph, and keeps it in step with the document.
pub struct CanvasPlugin;

impl Plugin for CanvasPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Selection>()
            .init_resource::<Grab>()
            .init_resource::<CanvasFrame>()
            .init_resource::<CanvasShape>()
            .init_resource::<Solo>()
            .init_resource::<input::Finished>()
            .add_systems(Startup, spawn_canvas_camera)
            .add_systems(
                Update,
                (
                    scene::rebuild_canvas,
                    frame_graph,
                    scene::sync_canvas,
                    input::canvas_camera,
                    input::canvas_solo,
                    input::canvas_drag,
                    input::canvas_commit,
                    solo_preview,
                    edges::route_edges,
                    scale_canvas_labels,
                )
                    .chain()
                    .before(EditorSystems::Document),
            )
            .add_systems(
                PostUpdate,
                size_canvas_viewport.after(UiSystems::Layout),
            );
    }
}

/// The camera the canvas is drawn through. Orthographic, and the only camera that
/// draws [`CANVAS_LAYER`].
#[derive(Component)]
pub struct CanvasCameraTag;

/// One node's card. Carries what the pins are placed from, so a pin's position is
/// derived rather than stored twice.
#[derive(Component, Clone, Copy)]
pub struct NodeCard {
    /// The node this card stands for.
    pub node: NodeId,
    /// Width and height in canvas units.
    pub size: Vec2,
    /// How many input pins run down its left edge.
    pub inputs: usize,
}

/// One pin of one card: the output when `input` is `None`, otherwise that input.
#[derive(Component, Clone, Copy)]
pub struct NodePin {
    /// The node the pin belongs to.
    pub node: NodeId,
    /// Which input pin, or `None` for the single output.
    pub input: Option<usize>,
}

/// One edge, as the two card entities it joins.
///
/// Entities rather than ids because the routing reads their transforms every frame,
/// and looking each up by id would be a scan per edge per frame.
#[derive(Component)]
pub struct NodeEdge {
    /// The card the value comes from.
    pub from: Entity,
    /// The card it is written into.
    pub to: Entity,
    /// Which input pin of `to`.
    pub pin: usize,
}

/// A world-space label, and the height it is meant to read at.
///
/// World-space text is rasterised once and then scaled by the camera, so a label left
/// alone blurs as it is zoomed past 1:1. The height kept here is what
/// [`scale_canvas_labels`] rasterises against the current zoom.
#[derive(Component)]
pub struct CanvasLabel(pub f32);

/// Which node the panel is showing and which one the map is showing.
#[derive(Resource, Default)]
pub struct Selection {
    /// The node the inspector is built for.
    pub node: Option<NodeId>,
    /// The node the map is showing instead of the field's output.
    ///
    /// Never set while `node` is `None`: a solo is an inspection that ends with the
    /// selection that made it, so the map at rest is what the field bakes.
    pub soloed: Option<NodeId>,
}

impl Selection {
    /// Selects a node, or clears the selection — which clears any solo with it.
    pub fn select(&mut self, node: Option<NodeId>) {
        self.node = node;
        if node.is_none() {
            self.soloed = None;
        }
    }

    /// Drops a node that is no longer in the document from both the selection and the
    /// solo, so neither can name something that is gone.
    pub fn forget(&mut self, gone: NodeId) {
        if self.node == Some(gone) {
            self.select(None);
        }
        if self.soloed == Some(gone) {
            self.soloed = None;
        }
    }
}

/// What the pointer is currently doing on the canvas.
///
/// An enum rather than a set of flags because dragging a card, dragging a wire and
/// panning are three things the pointer cannot be doing at once, and a shape that
/// admitted all three would need a rule for what to do when it held them.
#[derive(Resource, Default)]
pub enum Grab {
    /// Nothing is held.
    #[default]
    Idle,
    /// A card is being moved. The offset keeps the point under the cursor under it.
    Card {
        /// The card entity.
        entity: Entity,
        /// The node it stands for.
        node: NodeId,
        /// Where the card sat relative to the cursor when it was taken.
        offset: Vec2,
    },
    /// A wire is being dragged from a pin.
    Wire {
        /// The node the wire comes from.
        node: NodeId,
        /// Which pin it left, so releasing over another can tell which end is which.
        pin: Option<usize>,
    },
    /// The canvas is being panned. The anchor is the canvas point held under the
    /// cursor.
    Pan {
        /// The canvas point the drag started at.
        anchor: Vec2,
    },
}

/// The strip of the window the node canvas is drawn into.
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

/// The raster the map is showing instead of the field's output, if any.
///
/// Held here rather than on the document because it is a way of looking at the
/// document, not a part of it: nothing about a solo is saved, and clearing one puts
/// the map back without anything having been edited.
#[derive(Resource, Default)]
pub struct Solo {
    /// Which field and node the raster was baked for.
    pub node: Option<(String, NodeId)>,
    /// The values, at the field's own resolution.
    pub raster: Option<watershed::raster::Raster<f32>>,
    /// Bumped whenever the raster changes, so the map knows to upload it again.
    pub generation: u64,
}

/// What the canvas was last built from.
///
/// The graph's shape only: which nodes there are, what op each carries, how many pins
/// it has and what is wired into them. A position or a name is deliberately absent, so
/// dragging a card or naming a node writes onto the cards that are already there
/// rather than despawning every one of them.
#[derive(Resource, Default)]
pub struct CanvasShape {
    /// The shape the cards on screen were built from.
    pub key: String,
    /// The field the camera was last framed for, so opening another field frames it
    /// once rather than fighting a pan the person made afterwards.
    pub framed: Option<String>,
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

/// Gives the canvas camera the rectangle the layout reserved for it.
///
/// Two cameras with no viewports would each draw the whole window, so this is what
/// keeps the map above the canvas rather than behind it.
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
    // A viewport that leaves the render target is a validation error the renderer
    // quits on, so it is clamped here rather than trusted from the layout.
    let position = frame.position.max(Vec2::ZERO).as_uvec2().min(target - UVec2::ONE);
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

/// Bakes the soloed node onto the map, or puts the field's output back.
///
/// Runs only when the solo moves or the document changes under it: a preview is a
/// whole field's bake, so doing it every frame would cost a bake a frame.
fn solo_preview(
    document: Res<Document>,
    selection: Res<Selection>,
    mut solo: ResMut<Solo>,
    mut baked_at: Local<Option<u64>>,
) {
    let wanted = selection
        .soloed
        .map(|node| (document.active().to_owned(), node));
    let revision = document.revision();
    if solo.node == wanted && *baked_at == Some(revision) {
        return;
    }
    *baked_at = Some(revision);
    solo.node = wanted.clone();
    solo.generation = solo.generation.wrapping_add(1);
    solo.raster = wanted.and_then(|(field, node)| {
        document
            .terrain()
            .and_then(|terrain| terrain.preview_node(&field, node))
    });
}

/// Puts the whole of the open field's graph in view, once per field.
///
/// Once, because after that the pan and the zoom are the person's: a frame on every
/// edit would drag the view out from under someone adding a node at the far edge.
fn frame_graph(
    document: Res<Document>,
    frame: Res<CanvasFrame>,
    window: Option<Single<&Window, With<bevy::window::PrimaryWindow>>>,
    mut shape: ResMut<CanvasShape>,
    camera: Option<Single<(&mut Transform, &mut Projection), With<CanvasCameraTag>>>,
) {
    let active = document.active().to_owned();
    if shape.framed.as_deref() == Some(active.as_str()) {
        return;
    }
    let (Some(camera), Some(window)) = (camera, window) else {
        return;
    };
    let scale_factor = window.scale_factor();
    if !scale_factor.is_finite() || scale_factor <= 0.0 {
        return;
    }
    let Some(graph) = open_graph(&document) else {
        return;
    };
    if graph.nodes.is_empty() {
        shape.framed = Some(active);
        return;
    }
    let mut low = Vec2::splat(f32::INFINITY);
    let mut high = Vec2::splat(f32::NEG_INFINITY);
    for node in &graph.nodes {
        let at = Vec2::new(node.position[0], node.position[1]);
        low = low.min(at - CARD * 0.5);
        high = high.max(at + CARD * 0.5);
    }
    if !low.is_finite() || !high.is_finite() || frame.size.x <= 1.0 || frame.size.y <= 1.0 {
        return;
    }
    shape.framed = Some(active);

    let span = (high - low).max(Vec2::splat(1.0)) + Vec2::splat(CARD.x * 0.4);
    let (mut transform, mut projection) = camera.into_inner();
    let Projection::Orthographic(ortho) = &mut *projection else {
        return;
    };
    let wanted = (span / (frame.size / scale_factor)).max_element();
    ortho.scale = wanted.clamp(MIN_SCALE, MAX_SCALE);
    let centre = (low + high) * 0.5;
    transform.translation.x = centre.x;
    transform.translation.y = centre.y;
}

/// Rasterises each label at the size it is about to be shown at.
///
/// World-space text is rasterised at its font size and the camera then scales that, so
/// a label zoomed past 1:1 is blurred by exactly the zoom. Each label is given a font
/// size of its world height divided by the camera's scale and a counter-scale that puts
/// the height back, so what is rasterised is what is shown.
///
/// Runs only when the zoom changes, and against a size rounded to the pixel: every
/// distinct size is a font atlas, and a smooth wheel zoom would otherwise build one per
/// frame.
fn scale_canvas_labels(
    camera: Option<Single<&Projection, With<CanvasCameraTag>>>,
    mut labels: Query<(&CanvasLabel, &mut TextFont, &mut Transform, &mut Visibility)>,
    mut applied: Local<Option<f32>>,
) {
    let Some(camera) = camera else {
        return;
    };
    let Projection::Orthographic(ortho) = camera.into_inner() else {
        return;
    };
    if *applied == Some(ortho.scale) {
        return;
    }
    *applied = Some(ortho.scale);

    let readable = 1.0 / ortho.scale >= CHIP_ZOOM;
    for (label, mut font, mut transform, mut visibility) in &mut labels {
        *visibility = if readable {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
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

/// The open field's graph, or `None` when there is no document.
fn open_graph(document: &Document) -> Option<&crate::terrain::graph::FieldGraph> {
    let terrain = document.terrain()?;
    Some(&terrain.field(document.active())?.graph)
}

/// Where a card's single output pin sits, relative to the card's centre.
fn output_offset(card: &NodeCard) -> Vec2 {
    Vec2::new(card.size.x * 0.5, 0.0)
}

/// Where one of a card's input pins sits, relative to the card's centre.
fn input_offset(card: &NodeCard, index: usize) -> Vec2 {
    let step = card.size.y / (card.inputs.max(1) as f32 + 1.0);
    Vec2::new(
        -card.size.x * 0.5,
        card.size.y * 0.5 - step * (index as f32 + 1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::graph::{FieldGraph, NodeOp};
    use crate::terrain::{Field, TerrainSpec};

    fn node(id: u32) -> NodeId {
        NodeId(id)
    }

    // A solo is an inspection that ends with the selection that made it, so every path
    // that clears a selection has to clear the solo with it — otherwise the map keeps
    // showing a node nothing on the canvas is marking.
    #[test]
    fn deselecting_clears_the_solo_with_it() {
        let mut selection = Selection::default();
        selection.select(Some(node(3)));
        selection.soloed = Some(node(3));
        selection.select(None);
        assert_eq!(selection.node, None);
        assert_eq!(selection.soloed, None);
    }

    // A node removed from the document must not be left named by either, or the panel
    // builds against a node that is gone and the map previews one that cannot be baked.
    #[test]
    fn a_removed_node_is_forgotten_by_the_selection_and_the_solo() {
        let mut selection = Selection::default();
        selection.select(Some(node(1)));
        selection.soloed = Some(node(1));
        selection.forget(node(1));
        assert_eq!(selection.node, None);
        assert_eq!(selection.soloed, None);
    }

    // Selecting another node leaves the first node's solo behind, which would put the
    // map and the canvas out of step: what the map shows is always marked on the card
    // that is selected.
    #[test]
    fn selecting_another_node_does_not_carry_the_solo_over() {
        let mut selection = Selection::default();
        selection.select(Some(node(1)));
        selection.soloed = Some(node(1));
        selection.select(Some(node(2)));
        assert_eq!(selection.soloed, Some(node(1)));
        selection.forget(node(1));
        assert_eq!(selection.node, Some(node(2)));
        assert_eq!(selection.soloed, None);
    }

    // The pins have to be where the routing thinks they are: an output on the right
    // edge and one input per arity evenly down the left, or every wire lands beside the
    // card rather than on it.
    #[test]
    fn the_pins_sit_on_the_edges_of_the_card_they_belong_to() {
        let card = NodeCard {
            node: node(0),
            size: CARD,
            inputs: 3,
        };
        assert_eq!(output_offset(&card).x, CARD.x * 0.5);
        assert_eq!(output_offset(&card).y, 0.0);
        for index in 0..3 {
            let at = input_offset(&card, index);
            assert_eq!(at.x, -CARD.x * 0.5);
            assert!(at.y.abs() <= CARD.y * 0.5, "pin {index} left the card");
        }
        assert!(input_offset(&card, 0).y > input_offset(&card, 2).y);
    }

    // A preview is what the map shows while a node is soloed, so it has to be that
    // node's values and not the field's — and it must leave the document exactly as it
    // was, since nothing about looking at a node is an edit.
    #[test]
    fn previewing_a_node_reads_that_node_and_moves_nothing() {
        let mut graph = FieldGraph::new();
        let under = graph.node_with(NodeOp::Constant(0.25), &[]);
        let over = graph.node_with(NodeOp::Constant(0.75), &[]);
        let sum = graph.node_with(NodeOp::Binary(crate::terrain::graph::Binary::Add), &[under, over]);
        graph.set_output(Some(sum)).unwrap();

        let mut terrain = TerrainSpec::new(UVec2::splat(8))
            .with_field(Field::new("height").with_range((0.0, 4.0)).with_graph(graph));
        terrain.bake_in_place().unwrap();
        let before = terrain.clone();

        let preview = terrain.preview_node("height", under).expect("a preview");
        assert!(preview.data().iter().all(|value| *value == 0.25));
        assert_eq!(terrain.sample("height", 4.5, 4.5).unwrap(), 1.0);
        assert_eq!(terrain, before, "a preview moved the document");
    }

    // A node the document does not carry has to answer with nothing rather than baking
    // whatever the id happens to land on.
    #[test]
    fn previewing_a_node_that_is_not_there_answers_with_nothing() {
        let terrain = TerrainSpec::new(UVec2::splat(8))
            .with_field(Field::new("height").with_op(NodeOp::Constant(0.5)));
        assert!(terrain.preview_node("height", node(9)).is_none());
        assert!(terrain.preview_node("nowhere", node(0)).is_none());
    }
}
