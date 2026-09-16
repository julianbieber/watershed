//! The editor's chrome: the panels, the dialog, and the state that belongs to the
//! interface rather than to the document.
//!
//! Nothing here decides what an action does. Every control calls the same entry point
//! the control client's verb calls, so the two cannot drift apart and anything the
//! window can do can be driven from outside the process. The rule runs that way round:
//! a new action is written where both can reach it, and then given a button.

use bevy::feathers::controls::{ButtonVariant, FeathersButton};
use bevy::feathers::theme::{ThemeBackgroundColor, ThemedText};
use bevy::feathers::tokens;
use bevy::input_focus::InputFocus;
use bevy::input_focus::tab_navigation::TabGroup;
use bevy::picking::Pickable;
use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use bevy::text::EditableText;
use bevy::ui::UiSystems;
use bevy::ui_widgets::Activate;
use bevy::window::PrimaryWindow;

use crate::canvas::{CanvasCameraTag, CanvasFrame, CanvasViewport, frame_canvas};
use crate::document::Document;
use crate::preset::Preset;
use crate::view::FreeView;

mod bind;
mod dialog;
mod legend;
mod log;
mod scroll;
mod stack;
mod toolbar;
mod widgets;

/// Spawns the editor's chrome and runs the systems that keep it in step with the
/// document.
pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NewDialog>()
            .init_resource::<NewLayer>()
            .init_resource::<Expanded>()
            .init_resource::<toolbar::LayerChoices>()
            .init_resource::<stack::Shape>()
            .init_resource::<log::Open>()
            .add_systems(Startup, shell.spawn())
            .add_systems(
                Update,
                (
                    toolbar::sync,
                    toolbar::rebuild_layer_menu,
                    stack::rebuild,
                    stack::sync,
                    stack::seed_layer_name,
                    stack::prune,
                    dialog::sync,
                    legend::sync,
                    log::sync,
                    scroll::send,
                ),
            )
            .add_observer(scroll::apply)
            .add_systems(
                PostUpdate,
                (
                    bind::push,
                    (measure_free_view, measure_canvas_frame).after(UiSystems::Layout),
                ),
            );
    }
}

/// What the new-terrain dialog is currently set to, and whether it is showing.
///
/// Held apart from the document because it describes a terrain that does not exist
/// yet: it survives creating one, so a person who got the size wrong reopens the
/// dialog on what they last typed rather than on what they ended up with.
#[derive(Resource)]
pub struct NewDialog {
    /// Whether the dialog is showing.
    pub open: bool,
    /// Cells across, for the terrain Create would build.
    pub width: u32,
    /// Cells down, for the terrain Create would build.
    pub height: u32,
    /// The seed the preset would be built from.
    pub seed: u32,
    /// Which preset Create would build.
    pub preset: Preset,
}

impl Default for NewDialog {
    fn default() -> Self {
        Self {
            open: false,
            width: 1024,
            height: 1024,
            seed: 1,
            preset: Preset::default(),
        }
    }
}

/// The name typed into the panel's field box: what "Add layer" will call a new layer.
///
/// Held across frames because typing a name and pressing the button are two separate
/// acts, and because the panel is rebuilt whenever the document changes shape — a name
/// left only in the text input would be lost with it.
#[derive(Resource, Default)]
pub struct NewLayer(pub String);

/// Which of the panel's collapsible sections are open.
///
/// Kept here rather than in the toggles themselves because the panel is rebuilt
/// whenever the document changes shape — adding a layer despawns every toggle in it,
/// and state left in one would be lost with it, closing every section on each edit.
#[derive(Resource, Default)]
pub struct Expanded {
    /// Whether the shader reference is open.
    pub reference: bool,
}

const PANEL_WIDTH: f32 = 320.0;

#[derive(Component, Default, Clone)]
struct WorldViewport;

fn shell() -> impl SceneList {
    bsn_list! { @chrome() -- @dialog::dialog() }
}

fn canvas_divider() -> impl Scene {
    bsn! {
        Node {
            height: px(2),
            min_height: px(2),
            flex_shrink: 0.0,
        }
        Pickable { should_block_lower: false, is_hoverable: false }
        ThemeBackgroundColor(tokens::GROUP_BORDER)
    }
}

fn canvas_bar() -> impl Scene {
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(6),
            padding: px(4),
            flex_shrink: 0.0,
        }
        ThemeBackgroundColor(tokens::WINDOW_BG)
        Children [
            @FeathersButton {
                @caption: bsn! { Text("Fit") ThemedText },
                @variant: ButtonVariant::Plain,
            }
            on(|_: On<Activate>,
                document: Res<Document>,
                frame: Res<CanvasFrame>,
                window: Single<&Window, With<PrimaryWindow>>,
                camera: Single<(&mut Transform, &mut Projection), With<CanvasCameraTag>>| {
                let (mut transform, mut projection) = camera.into_inner();
                frame_canvas(
                    &document,
                    &frame,
                    *window,
                    &mut transform,
                    &mut projection,
                );
            })
        ]
    }
}

fn chrome() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            height: percent(100),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Stretch,
        }
        TabGroup
        Pickable { should_block_lower: false, is_hoverable: false }
        Children [
            @toolbar::toolbar()
            --
            Node {
                flex_grow: 1.0,
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Stretch,
                min_height: px(0),
            }
            Pickable { should_block_lower: false, is_hoverable: false }
            Children [
                Node {
                    flex_grow: 1.0,
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Stretch,
                    min_width: px(0),
                }
                Pickable { should_block_lower: false, is_hoverable: false }
                Children [
                    Node {
                        flex_grow: 1.0,
                        display: Display::Flex,
                        flex_direction: FlexDirection::Column,
                        justify_content: JustifyContent::End,
                        align_items: AlignItems::Start,
                        padding: px(12),
                        min_height: px(0),
                    }
                    Pickable { should_block_lower: false, is_hoverable: false }
                    WorldViewport
                    Children [ @legend::legend() ]
                    --
                    @canvas_divider()
                    --
                    @canvas_bar()
                    --
                    Node {
                        flex_grow: 0.72,
                        min_height: px(0),
                    }
                    Pickable { should_block_lower: false, is_hoverable: false }
                    CanvasViewport
                ]
                --
                @stack::panel()
            ]
            --
            @log::panel()
        ]
    }
}

fn measure_canvas_frame(
    mut frame: ResMut<CanvasFrame>,
    window: Option<Single<&Window, With<PrimaryWindow>>>,
    viewport: Option<Single<(&ComputedNode, &UiGlobalTransform), With<CanvasViewport>>>,
) {
    let (Some(window), Some(viewport)) = (window, viewport) else {
        return;
    };
    let (node, transform) = viewport.into_inner();
    let size = node.size();
    let centre = Vec2::new(transform.translation.x, transform.translation.y);
    let target = Vec2::new(
        window.physical_width() as f32,
        window.physical_height() as f32,
    );
    if target.x < 1.0 || target.y < 1.0 {
        return;
    }
    let low = (centre - size * 0.5)
        .max(Vec2::ZERO)
        .min(target - Vec2::ONE);
    frame.position = low;
    frame.size = size.min(target - low).max(Vec2::ONE);
}

fn measure_free_view(
    mut free_view: ResMut<FreeView>,
    window: Option<Single<&Window, With<PrimaryWindow>>>,
    viewport: Option<Single<(&ComputedNode, &UiGlobalTransform), With<WorldViewport>>>,
) {
    let (Some(window), Some(viewport)) = (window, viewport) else {
        return;
    };
    let (node, transform) = viewport.into_inner();
    let size = node.size();
    let centre = transform.translation;
    let free = Rect::from_center_size(Vec2::new(centre.x, centre.y), size);
    *free_view = FreeView::new(
        free,
        Vec2::new(
            window.physical_width() as f32,
            window.physical_height() as f32,
        ),
    );
}

/// Whether the pointer is over any node the UI owns.
///
/// The question a tool asks before it acts on a click. It is about the node under the
/// pointer, not about whether a widget wants the event — a node either takes a hit or
/// it does not, and there is no third answer to interpret.
pub fn pointer_over_ui(hover: &HoverMap, nodes: &Query<(), With<Node>>) -> bool {
    hover
        .values()
        .flat_map(|hits| hits.keys())
        .any(|entity| nodes.contains(*entity))
}

/// Whether a text field has the keyboard, in which case a key chord belongs to what
/// is being typed and not to the editor: the keyboard twin of [`pointer_over_ui`].
pub fn typing(focus: Option<&InputFocus>, fields: &Query<(), With<EditableText>>) -> bool {
    focus
        .and_then(InputFocus::get)
        .is_some_and(|entity| fields.contains(entity))
}

/// Records a refusal on the document so the toolbar shows it, and logs it.
///
/// Every control routes its refusals through here. Logging alone is not enough: the
/// control client reads the log and a person watching the window does not, and both
/// have to get the answer to "why did that button do nothing".
pub fn report(document: &mut Document, result: Result<(), String>) {
    if let Err(error) = result {
        warn!("{error}");
        document.refuse(error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout_app() -> App {
        let mut app = App::new();
        app.add_plugins((
            bevy::app::TaskPoolPlugin::default(),
            bevy::asset::AssetPlugin::default(),
            bevy::transform::TransformPlugin,
            bevy::text::TextPlugin,
            bevy::picking::PickingPlugin,
            bevy::picking::InteractionPlugin,
            bevy::input::InputPlugin,
            bevy::image::ImagePlugin::default(),
            bevy::time::TimePlugin,
            bevy::ui::UiPlugin,
        ));
        app.init_asset::<bevy::image::TextureAtlasLayout>();
        app
    }

    // The defect this guards drew the whole world into four pixels: the rectangle left
    // over after the panels is what the camera is given, and a degenerate one is not a
    // visible mistake in the panel — it is a missing picture everywhere else. The second
    // half pins that it is the space the panels are *not* in, rather than merely a big
    // rectangle.
    #[test]
    fn the_panels_leave_the_world_a_rectangle_it_can_be_drawn_in() {
        let mut app = layout_app();
        let screen = Vec2::new(592.0, 720.0);
        spawn_camera(app.world_mut(), screen);

        let world = app.world_mut();
        let root = world
            .spawn(Node {
                width: percent(100),
                height: percent(100),
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Stretch,
                ..default()
            })
            .id();
        let toolbar = world
            .spawn(Node {
                height: px(32),
                ..default()
            })
            .id();
        let middle = world
            .spawn(Node {
                flex_grow: 1.0,
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Stretch,
                min_height: px(0),
                ..default()
            })
            .id();
        let viewport = world
            .spawn((
                Node {
                    flex_grow: 1.0,
                    ..default()
                },
                WorldViewport,
            ))
            .id();
        let panel = world
            .spawn(Node {
                width: px(PANEL_WIDTH),
                ..default()
            })
            .id();
        world.entity_mut(root).add_children(&[toolbar, middle]);
        world.entity_mut(middle).add_children(&[viewport, panel]);

        app.update();

        let world = app.world_mut();
        let free = world.get::<ComputedNode>(viewport).unwrap().size();
        assert!(
            free.x > screen.x * 0.25,
            "the world was left {} points of {}",
            free.x,
            screen.x
        );
        assert!(
            free.y > screen.y * 0.5,
            "the world was left {} points of {}",
            free.y,
            screen.y
        );

        let centre = world
            .get::<UiGlobalTransform>(viewport)
            .unwrap()
            .translation;
        let rect = Rect::from_center_size(Vec2::new(centre.x, centre.y), free);
        assert!(rect.min.y > 0.0, "the world overlaps the toolbar");
        assert!(rect.max.x < screen.x, "the world overlaps the panel");
    }

    fn spawn_camera(world: &mut World, size: Vec2) {
        world.spawn((
            Camera2d,
            Camera {
                computed: bevy::camera::ComputedCameraValues {
                    target_info: Some(bevy::camera::RenderTargetInfo {
                        physical_size: size.as_uvec2(),
                        scale_factor: 1.0,
                    }),
                    ..default()
                },
                viewport: Some(bevy::camera::Viewport {
                    physical_size: size.as_uvec2(),
                    ..default()
                }),
                ..default()
            },
        ));
    }
}
