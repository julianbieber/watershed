//! A graphical editor for a watershed document: authoring the fields and the shader
//! files that produce them, and previewing the baked terrain.
//!
//! Every capability the editor has is reachable from `watershed-ctl` as well as from
//! the window. That is a standing rule rather than a convenience: a change that can
//! only be exercised by a person holding the keys cannot be verified, so an action
//! added to the UI is added to [`control`] too.

use bevy::feathers::FeathersPlugins;
use bevy::feathers::dark_theme::create_dark_theme;
use bevy::feathers::theme::UiTheme;
use bevy::prelude::*;

mod canvas;
mod control;
mod document;
mod edit;
mod gpu;
mod history;
mod material;
mod preset;
mod terrain;
mod ui;
mod view;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(bevy::log::LogPlugin {
            custom_layer: control::log_layer,
            ..default()
        }))
        .add_plugins(FeathersPlugins)
        .insert_resource(UiTheme(create_dark_theme()))
        .add_plugins((
            document::DocumentPlugin,
            view::ViewPlugin,
            canvas::CanvasPlugin,
            gpu::ShaderPlugin,
            ui::UiPlugin,
            control::ControlPlugin,
        ))
        .run();
}
