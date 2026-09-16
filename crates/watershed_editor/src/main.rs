//! A graphical editor for a watershed document: authoring the layers and the shader
//! files that produce them, and previewing the baked terrain.
//!
//! Every capability the editor has is reachable from `watershed-ctl` as well as from
//! the window. That is a standing rule rather than a convenience: a change that can
//! only be exercised by a person holding the keys cannot be verified, so an action
//! added to the UI is added to [`control`] too.

use bevy::app::AppExit;
use bevy::feathers::FeathersPlugins;
use bevy::feathers::dark_theme::create_dark_theme;
use bevy::feathers::theme::UiTheme;
use bevy::prelude::*;
use bevy::window::{Window, WindowPlugin};

mod canvas;
mod control;
mod document;
mod edit;
mod gpu;
mod history;
mod material;
mod open;
mod preset;
mod project;
mod terrain;
mod ui;
mod view;

fn main() -> AppExit {
    let project = match project::from_arguments() {
        Ok(project) => project,
        Err(error) => {
            eprintln!("{error}");
            return AppExit::error();
        }
    };
    let title = project::title(&project);

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(bevy::log::LogPlugin {
                    custom_layer: control::log_layer,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window { title, ..default() }),
                    ..default()
                }),
        )
        .add_plugins(FeathersPlugins)
        .insert_resource(UiTheme(create_dark_theme()))
        .insert_resource(project)
        .add_plugins((
            document::DocumentPlugin,
            view::ViewPlugin,
            canvas::CanvasPlugin,
            gpu::ShaderPlugin,
            ui::UiPlugin,
            control::ControlPlugin,
            project::ProjectPlugin,
        ))
        .run()
}
