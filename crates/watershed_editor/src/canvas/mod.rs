//! The node graph as a thing on screen: what the open field's graph is drawn as, how
//! it is panned, zoomed and rewired, and how one node's values are put on the map.

mod edges;
mod input;
mod overview;
mod scene;
mod thumb;

pub use overview::Overview;

use bevy::camera::visibility::RenderLayers;
use bevy::camera::{Camera, ClearColorConfig, Viewport};
use bevy::prelude::*;
use bevy::ui::UiSystems;

use crate::document::{Document, EditorSystems};
use crate::terrain::graph::{FieldGraph, NodeId};

/// The width and height of a card, in canvas units.
///
/// Stays under the `NODE_STEP` an auto-placed graph is laid out on, or every
/// control-built graph is a stack of overlapping cards.
pub const CARD: Vec2 = Vec2::new(240.0, 140.0);
const TITLE_BAR: f32 = 30.0;
const THUMB: f32 = 78.0;
const MARGIN: f32 = 8.0;
const ROW_STEP: f32 = 19.0;
const TITLE_SIZE: f32 = 17.0;
const DETAIL_SIZE: f32 = 12.0;
const PARAM_SIZE: f32 = 11.0;
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
            .init_resource::<Overview>()
            .init_resource::<Grab>()
            .init_resource::<CanvasFrame>()
            .init_resource::<CanvasShape>()
            .init_resource::<Solo>()
            .init_resource::<input::Finished>()
            .init_resource::<overview::Wired>()
            .init_resource::<overview::WiredStep>()
            .add_message::<OpenField>()
            .add_systems(Startup, spawn_canvas_camera)
            .add_systems(
                Update,
                (
                    apply_open_field,
                    input::undo_keys,
                    overview::overview_after_undo,
                    scene::rebuild_canvas.run_if(not(overview_showing)),
                    overview::rebuild_overview.run_if(overview_showing),
                    frame_graph,
                    scene::sync_canvas.run_if(not(overview_showing)),
                    thumb::sync_thumbnails,
                    input::canvas_camera,
                    input::fit_key,
                    overview::overview_key,
                    input::canvas_solo,
                    input::canvas_drag.run_if(not(overview_showing)),
                    overview::overview_drag.run_if(overview_showing),
                    input::canvas_commit,
                    overview::commit_field_wire,
                    solo_preview,
                    edges::route_edges,
                    overview::route_field_ribbons.run_if(overview_showing),
                    scale_canvas_labels,
                )
                    .chain()
                    .before(EditorSystems::Document),
            )
            .add_systems(PostUpdate, size_canvas_viewport.after(UiSystems::Layout));
    }
}

/// Whether the canvas is showing the document's fields rather than one field's graph.
fn overview_showing(overview: Res<Overview>) -> bool {
    overview.showing
}

/// Which of the two things the canvas draws is on screen.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CanvasView {
    /// One field's node graph, by the field's name.
    Graph(String),
    /// The whole document, as one card per field.
    Overview,
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

/// Asks for another field of the open document to be put on screen.
///
/// The one way anything asks for a field change: the canvas's double-click, the
/// panel's `reads` and `read by` buttons, and whatever else comes to want it all write
/// this rather than reaching for the document themselves. It is a view change and not
/// an edit — nothing is written to the terrain, no history entry is made, and no bake
/// is started — so a field opened this way can be shut again by opening the first one.
///
/// Opening a field leaves the overview, because the field's graph is what was asked
/// for. A name no field of the document carries is refused the way any other bad
/// reference is, and the view stays where it was — overview and all.
#[derive(Message)]
pub struct OpenField {
    /// The field to open. Must be one the document carries.
    pub field: String,
    /// A node of the field being opened to select once it is up, or `None` to leave
    /// nothing selected.
    pub select: Option<NodeId>,
}

fn apply_open_field(
    mut open: MessageReader<OpenField>,
    mut document: ResMut<Document>,
    mut selection: ResMut<Selection>,
    mut overview: ResMut<Overview>,
) {
    for message in open.read() {
        let opened = document.set_active(&message.field);
        if opened.is_ok() {
            selection.select(message.select);
            overview.showing = false;
        }
        crate::ui::report(&mut document, opened);
    }
}

/// What the pointer is currently doing on the canvas.
///
/// An enum rather than a set of flags because dragging a card, dragging a wire, wiring
/// two fields together and panning are four things the pointer cannot be doing at once,
/// and a shape that admitted several would need a rule for what to do when it held them.
#[derive(Resource, Default)]
pub enum Grab {
    /// Nothing is held.
    #[default]
    Idle,
    /// A card is being moved. The offset keeps the point under the cursor under it.
    ///
    /// While this is held the drag owns the card's position, not the document: `at` is
    /// where the card has been pulled to, and it is what the release writes. Reading it
    /// back off the transform instead would read whatever ran last that frame.
    Card {
        /// The card entity.
        entity: Entity,
        /// The node it stands for.
        node: NodeId,
        /// Where the card sat relative to the cursor when it was taken.
        offset: Vec2,
        /// Where the card has been dragged to.
        at: Vec2,
    },
    /// A wire is being dragged from a pin.
    Wire {
        /// The node the wire comes from.
        node: NodeId,
        /// Which pin it left, so releasing over another can tell which end is which.
        pin: Option<usize>,
    },
    /// A dependency is being dragged from one field's card to another's.
    FieldWire {
        /// The field the drag left, which the reference it makes will read.
        from: String,
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
    /// The view the camera was last framed for, so switching to another one frames it
    /// once rather than fighting a pan the person made afterwards.
    pub framed: Option<CanvasView>,
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

/// Puts the whole of whatever the canvas is showing in view, once per view.
///
/// Once, because after that the pan and the zoom are the person's: a frame on every
/// edit would drag the view out from under someone adding a node at the far edge.
/// Switching the overview on is another view, so it is framed once too. The on-demand
/// path is [`frame_canvas`], which the canvas's Fit button and the key F both take.
fn frame_graph(
    document: Res<Document>,
    overview: Res<Overview>,
    frame: Res<CanvasFrame>,
    window: Option<Single<&Window, With<bevy::window::PrimaryWindow>>>,
    mut shape: ResMut<CanvasShape>,
    camera: Option<Single<(&mut Transform, &mut Projection), With<CanvasCameraTag>>>,
) {
    let wanted = if overview.showing {
        CanvasView::Overview
    } else {
        CanvasView::Graph(document.active().to_owned())
    };
    if shape.framed.as_ref() == Some(&wanted) {
        return;
    }
    let (Some(camera), Some(window)) = (camera, window) else {
        return;
    };
    let empty = if overview.showing {
        document.terrain().map(|terrain| terrain.fields.is_empty())
    } else {
        open_graph(&document).map(|graph| graph.nodes.is_empty())
    };
    match empty {
        None => return,
        Some(true) => {
            shape.framed = Some(wanted);
            return;
        }
        Some(false) => {}
    }
    let (mut transform, mut projection) = camera.into_inner();
    if frame_canvas(
        &document,
        &overview,
        &frame,
        window.into_inner(),
        &mut transform,
        &mut projection,
    ) {
        shape.framed = Some(wanted);
    }
}

/// Puts the whole of whatever the canvas is showing in view on the canvas camera.
///
/// What the canvas's Fit button and the key F both do, for the open field's graph and
/// for the document's fields alike. Answers whether the camera was moved: `false` when
/// there is no document, when the view has nothing in it to frame, and when the canvas
/// has not been measured yet.
pub fn frame_canvas(
    document: &Document,
    overview: &Overview,
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
    let fitted = if overview.showing {
        document
            .terrain()
            .and_then(|terrain| overview::overview_fit(terrain, viewport))
    } else {
        open_graph(document).and_then(|graph| graph_fit(graph, viewport))
    };
    let Some((centre, scale)) = fitted else {
        return false;
    };
    ortho.scale = scale;
    transform.translation.x = centre.x;
    transform.translation.y = centre.y;
    true
}

/// Where the canvas camera has to sit, and at what scale, for every one of a graph's
/// cards to be inside a viewport that many logical pixels across.
///
/// The scale is already clamped to what the canvas allows, so a graph too large to fit
/// is framed as closely as the zoom permits rather than not at all. `None` when the
/// graph has no nodes or the viewport has no area — there is nothing to frame in either
/// case.
fn graph_fit(graph: &FieldGraph, viewport: Vec2) -> Option<(Vec2, f32)> {
    if graph.nodes.is_empty() || viewport.x <= 1.0 || viewport.y <= 1.0 {
        return None;
    }
    let mut low = Vec2::splat(f32::INFINITY);
    let mut high = Vec2::splat(f32::NEG_INFINITY);
    for node in &graph.nodes {
        let at = Vec2::new(node.position[0], node.position[1]);
        low = low.min(at - CARD * 0.5);
        high = high.max(at + CARD * 0.5);
    }
    if !low.is_finite() || !high.is_finite() {
        return None;
    }
    let span = (high - low).max(Vec2::splat(1.0)) + Vec2::splat(CARD.x * 0.4);
    let scale = (span / viewport).max_element().clamp(MIN_SCALE, MAX_SCALE);
    Some(((low + high) * 0.5, scale))
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
    use bevy::ecs::system::RunSystemOnce;

    use super::*;
    use crate::terrain::graph::{FieldGraph, NodeOp};
    use crate::terrain::{Field, TerrainSpec};

    fn node(id: u32) -> NodeId {
        NodeId(id)
    }

    fn open_field_world() -> World {
        let mut document = Document::default();
        document.adopt(
            TerrainSpec::new(UVec2::splat(16))
                .with_field(Field::new("base").with_op(NodeOp::Constant(0.25)))
                .with_field(Field::new("height").with_op(NodeOp::Constant(0.5))),
        );
        document.set_active("height").unwrap();

        let mut world = World::new();
        world.insert_resource(document);
        world.insert_resource(Selection::default());
        world.insert_resource(Overview { showing: true });
        world.init_resource::<Messages<OpenField>>();
        world
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

    // The acceptance, pinned at the arithmetic: whatever a fit answers has to put every
    // card of a graph spread wider than the viewport inside the framed rectangle.
    #[test]
    fn a_fit_puts_every_card_of_a_spread_out_graph_in_view() {
        let mut graph = FieldGraph::new();
        let at = [[-1500.0, 0.0], [1500.0, 120.0], [0.0, -600.0]];
        for position in at {
            let id = graph.node_with(NodeOp::Constant(0.0), &[]);
            graph.place(id, position).unwrap();
        }
        let viewport = Vec2::new(800.0, 600.0);
        let (centre, scale) = graph_fit(&graph, viewport).expect("a fit");
        let framed = Rect::from_center_size(centre, viewport * scale);
        for position in at {
            let card = Rect::from_center_size(Vec2::new(position[0], position[1]), CARD);
            assert!(framed.contains(card.min), "a card's corner left the frame");
            assert!(framed.contains(card.max), "a card's corner left the frame");
        }
    }

    // Fitting has to answer with nothing rather than a centre and a scale when there is
    // nothing to frame, or the Fit button would throw the camera at whatever the empty
    // bounds came out as.
    #[test]
    fn there_is_nothing_to_fit_without_nodes_or_without_a_viewport() {
        let empty = FieldGraph::new();
        assert!(graph_fit(&empty, Vec2::new(800.0, 600.0)).is_none());

        let mut graph = FieldGraph::new();
        graph.node_with(NodeOp::Constant(0.0), &[]);
        assert!(graph_fit(&graph, Vec2::ZERO).is_none());
    }

    // A preview is what the map shows while a node is soloed, so it has to be that
    // node's values and not the field's — and it must leave the document exactly as it
    // was, since nothing about looking at a node is an edit.
    #[test]
    fn previewing_a_node_reads_that_node_and_moves_nothing() {
        let mut graph = FieldGraph::new();
        let under = graph.node_with(NodeOp::Constant(0.25), &[]);
        let over = graph.node_with(NodeOp::Constant(0.75), &[]);
        let sum = graph.node_with(
            NodeOp::Binary(crate::terrain::graph::Binary::Add),
            &[under, over],
        );
        graph.set_output(Some(sum)).unwrap();

        let mut terrain = TerrainSpec::new(UVec2::splat(8)).with_field(
            Field::new("height")
                .with_range((0.0, 4.0))
                .with_graph(graph),
        );
        terrain.bake_in_place().unwrap();
        let before = terrain.clone();

        let preview = terrain.preview_node("height", under).expect("a preview");
        assert!(preview.data().iter().all(|value| *value == 0.25));
        assert_eq!(terrain.sample("height", 4.5, 4.5).unwrap(), 1.0);
        assert_eq!(terrain, before, "a preview moved the document");
    }

    // Opening a field is a view change, so the field on screen moves while the history
    // and the dirty flag stand still — otherwise following a `reads` link would make a
    // document that has to be saved. It also leaves the overview, which is what makes
    // the canvas bar's toggle read as off after a double-click on a card.
    #[test]
    fn opening_a_field_moves_the_view_without_making_an_edit() {
        let mut world = open_field_world();
        let before = world.resource::<Document>().history();
        world.write_message(OpenField {
            field: "base".to_owned(),
            select: Some(node(4)),
        });
        world.run_system_once(apply_open_field).unwrap();

        let document = world.resource::<Document>();
        assert_eq!(document.active(), "base");
        assert_eq!(document.history().undo, before.undo);
        assert_eq!(document.history().redo, before.redo);
        assert!(!document.is_dirty());
        assert_eq!(world.resource::<Selection>().node, Some(node(4)));
        assert!(!world.resource::<Overview>().showing);
    }

    // A name no field carries has to leave the view where it was and say so, because
    // the panel and the canvas both write this message from names they read off a
    // document that may have moved under them.
    #[test]
    fn opening_a_field_that_is_not_there_leaves_the_view_alone() {
        let mut world = open_field_world();
        world.write_message(OpenField {
            field: "nowhere".to_owned(),
            select: None,
        });
        world.run_system_once(apply_open_field).unwrap();

        let document = world.resource::<Document>();
        assert_eq!(document.active(), "height");
        assert!(document.error().is_some(), "the refusal was not reported");
        assert!(
            world.resource::<Overview>().showing,
            "a refusal moved the view"
        );
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
