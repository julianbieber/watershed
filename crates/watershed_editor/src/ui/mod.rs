// TODO(jb-doc): module docs — that every button here has a ctl verb behind it, and that
// the rule runs that way round rather than the other.

use bevy::input_focus::tab_navigation::TabGroup;
use bevy::picking::Pickable;
use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use bevy::ui::UiSystems;
use bevy::window::PrimaryWindow;

use crate::document::Document;
use crate::preset::Preset;
use crate::view::FreeView;

mod bind;
mod dialog;
mod legend;
mod stack;
mod toolbar;
mod widgets;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NewDialog>()
            .init_resource::<FilePath>()
            .init_resource::<AddLayer>()
            .init_resource::<Expanded>()
            .init_resource::<toolbar::FieldChoices>()
            .init_resource::<stack::Shape>()
            .add_systems(Startup, shell.spawn())
            .add_systems(
                Update,
                (
                    toolbar::sync,
                    toolbar::rebuild_field_menu,
                    toolbar::seed_path,
                    stack::rebuild,
                    stack::sync,
                    dialog::sync,
                    legend::sync,
                ),
            )
            // After the scenes a rebuild queued have been spawned, so a field that has just
            // appeared holds its number on the frame it appears rather than the one after.
            .add_systems(
                PostUpdate,
                (bind::push, measure_free_view.after(UiSystems::Layout)),
            );
    }
}

/// TODO(jb-doc): why the dialog's fields are held here rather than read out of the
/// document — that a dialog is what you are *about* to make, and the document is what you
/// have.
#[derive(Resource)]
pub struct NewDialog {
    pub open: bool,
    pub width: u32,
    pub height: u32,
    pub seed: u32,
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

#[derive(Resource)]
pub struct FilePath(pub String);

impl Default for FilePath {
    fn default() -> Self {
        // A text field rather than a native file dialog: the editor is driven far more
        // often than it is clicked, and a path is what the ctl takes anyway.
        Self("terrain.watershed".to_owned())
    }
}

/// The op the add button will make. Held across frames because a menu is a choice standing
/// until it is acted on, exactly as the new-terrain dialog's fields are.
#[derive(Resource)]
pub struct AddLayer(pub String);

impl Default for AddLayer {
    fn default() -> Self {
        Self("noise".to_owned())
    }
}

/// Which of the panel's collapsible sections are open.
///
/// TODO(jb-doc): why this is a resource the layer panel's shape is read from rather than
/// state left in the disclosure toggles, and what a rebuild would otherwise lose.
#[derive(Resource, Default)]
pub struct Expanded {
    pub brush: bool,
    pub layers: Vec<usize>,
}

impl Expanded {
    pub fn has(&self, index: usize) -> bool {
        self.layers.contains(&index)
    }

    pub fn set(&mut self, index: usize, open: bool) {
        self.layers.retain(|held| *held != index);
        if open {
            self.layers.push(index);
        }
    }
}

/// Points the layer stack asks for. Wide enough for a region table's columns.
const PANEL_WIDTH: f32 = 320.0;

/// The ops a button can make, against the [`crate::edit::parse_op`] grammar the ctl takes.
/// A regions op is deliberately absent from both — a region table is not something either
/// a command line or a single button can write.
pub const ADDABLE: [&str; 5] = ["noise", "constant", "fieldref", "slope", "paint"];

/// The rectangle of the window the world is drawn in, as a node rather than as arithmetic:
/// the toolbar and the layer panel are its siblings, so whatever they take it does not
/// have. See [`FreeView`] for why the camera is not given this rectangle as a viewport.
#[derive(Component, Default, Clone)]
struct WorldViewport;

fn shell() -> impl SceneList {
    bsn_list![chrome(), dialog::dialog()]
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
            toolbar::toolbar(),
            (
                Node {
                    flex_grow: 1.0,
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Stretch,
                    min_height: px(0),
                }
                Pickable { should_block_lower: false, is_hoverable: false }
                Children [
                    (
                        Node {
                            flex_grow: 1.0,
                            display: Display::Flex,
                            flex_direction: FlexDirection::Column,
                            justify_content: JustifyContent::End,
                            align_items: AlignItems::Start,
                            padding: px(12),
                        }
                        Pickable { should_block_lower: false, is_hoverable: false }
                        WorldViewport
                        Children [ legend::legend() ]
                    ),
                    stack::panel(),
                ]
            )
        ]
    }
}

/// TODO(jb-comment): why this is measured after the layout rather than derived from
/// [`PANEL_WIDTH`], and what a wrapped toolbar would do to the arithmetic version.
fn measure_free_view(
    mut free_view: ResMut<FreeView>,
    window: Option<Single<&Window, With<PrimaryWindow>>>,
    viewport: Option<Single<(&ComputedNode, &UiGlobalTransform), With<WorldViewport>>>,
) {
    let (Some(window), Some(viewport)) = (window, viewport) else {
        return;
    };
    let (node, transform) = viewport.into_inner();
    // Physical throughout, which cancels: [`FreeView`] is fractions of the window, and both
    // ends of every fraction are measured in the same units.
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

/// Whether the pointer is over something the UI owns. The brush asks this instead of
/// asking whether a widget wants the event, because a bevy_ui node either takes a hit or
/// it does not — there is no third answer to interpret.
pub fn pointer_over_ui(hover: &HoverMap, nodes: &Query<(), With<Node>>) -> bool {
    hover
        .values()
        .flat_map(|hits| hits.keys())
        .any(|entity| nodes.contains(*entity))
}

/// A refusal is synchronous, where an error is something a job hands back — but both are
/// the answer to "why did that button do nothing", so both go where the toolbar can show
/// them. Logging alone was not enough: `observe log` reads it and a person does not.
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
            // The viewport widget's picking runs whether or not anything is being picked,
            // so a layout test has to carry it even though nothing here is ever clicked.
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

    /// The defect this guards drew the whole world into four pixels: the rectangle left
    /// over after the panels is what the camera is given, and a degenerate one is not a
    /// visible mistake in the panel — it is a missing picture everywhere else.
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

        // And it has to be the space the panels are not in, rather than merely a big
        // rectangle: the toolbar is above it and the layer stack to the right of it.
        let centre = world
            .get::<UiGlobalTransform>(viewport)
            .unwrap()
            .translation;
        let rect = Rect::from_center_size(Vec2::new(centre.x, centre.y), free);
        assert!(rect.min.y > 0.0, "the world overlaps the toolbar");
        assert!(rect.max.x < screen.x, "the world overlaps the panel");
    }

    /// A camera with a render target invented on the spot: there is no window in a test,
    /// and a camera that has never been told how big its target is lays nothing out.
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
