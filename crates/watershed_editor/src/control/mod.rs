//! Driving the running editor from outside the process.
//!
//! The editor keeps its real window and its real swapchain; this is a way to talk to
//! it, not a second way to run it. Present only when `WATERSHED_CONTROL` is set in
//! the environment.
//!
//! **A command is synchronous from the client's side.** The reply is held until the
//! effect has actually happened — `solve-water` answers when the water is solved,
//! `capture` when the PNG is on disk — so a caller never sleeps and hopes. What
//! blocks is the *client*; the editor runs on undisturbed.

use bevy::{log::BoxedLayer, prelude::*};

/// Installs the control server, if `WATERSHED_CONTROL` names a socket to listen on.
/// Without it the plugin is inert and the editor behaves as though it were absent.
pub struct ControlPlugin;

/// The extra tracing layer that keeps the run's warnings and errors where the window's
/// log panel and `observe log` can both reach them. Always installed.
///
/// Passed to `LogPlugin::custom_layer` in `main.rs` rather than installed by
/// [`ControlPlugin`], because a log layer has to exist before the logger does and
/// `LogPlugin` is built first. It is a bare `fn` pointer by `LogPlugin`'s definition, so
/// it cannot capture and inserts its own resource.
pub fn log_layer(app: &mut App) -> Option<BoxedLayer> {
    log::layer(app)
}

impl Plugin for ControlPlugin {
    fn build(&self, app: &mut App) {
        server::build(app);
    }
}

pub(crate) use log::{LogBuffer, LogView};

mod command;
mod log;
mod observe;
mod server;
