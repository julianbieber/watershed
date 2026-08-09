//! Drive the running editor from outside the process.
//!
//! Transplanted from wusel's `control/`, and for the same reason: a change that only shows
//! up on screen cannot otherwise be verified without a person holding the keys. The editor
//! keeps its real window and its real swapchain — this adds a way to talk to it, not a
//! second way to run it.
//!
//! **A command is synchronous from the client's side.** The reply is held until the effect
//! has actually happened — `solve-water` answers when the water is solved, `capture` when
//! the PNG is on disk — so a caller never sleeps and hopes. What blocks is the *client*;
//! the editor runs on undisturbed.
//!
//! Activation is the `WATERSHED_CONTROL` environment variable rather than a cargo feature:
//! CI builds `--all-features`, so a feature would need care in every recipe to buy nothing.

use bevy::{log::BoxedLayer, prelude::*};

pub struct ControlPlugin;

/// The extra tracing layer that keeps the run's warnings and errors where `observe log`
/// can reach them.
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

mod command;
mod log;
mod observe;
mod server;
