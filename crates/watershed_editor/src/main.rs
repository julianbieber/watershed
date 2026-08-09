// TODO(jb-doc): crate-level docs — what the editor is for, and the standing rule that
// nothing it can do may be unreachable from `watershed-ctl`.

use bevy::prelude::*;

mod brush;
mod control;
mod document;
mod edit;
mod material;
mod preset;
mod ui;
mod view;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(bevy::log::LogPlugin {
            // The layer has to exist before the logger does, and `LogPlugin` is built
            // first — so it is wired in here rather than by `ControlPlugin`.
            custom_layer: control::log_layer,
            ..default()
        }))
        .add_plugins(bevy_egui::EguiPlugin::default())
        .add_plugins((
            document::DocumentPlugin,
            brush::BrushPlugin,
            view::ViewPlugin,
            ui::UiPlugin,
            control::ControlPlugin,
        ))
        .run();
}
