// TODO(jb-doc): crate-level docs — what the editor is for, and the standing rule that
// nothing it can do may be unreachable from `watershed-ctl`.

use bevy::feathers::FeathersPlugins;
use bevy::feathers::dark_theme::create_dark_theme;
use bevy::feathers::theme::UiTheme;
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
        .add_plugins(FeathersPlugins)
        .insert_resource(UiTheme(create_dark_theme()))
        .add_plugins((
            document::DocumentPlugin,
            brush::BrushPlugin,
            view::ViewPlugin,
            ui::UiPlugin,
            control::ControlPlugin,
        ))
        .run();
}
