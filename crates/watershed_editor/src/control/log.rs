//! Everything the run complained about, kept where a scenario can read it.
//!
//! "Did the shader compile" is the first question anyone has about a render change, and
//! the answer is otherwise on stderr mixed into a few thousand lines of engine chatter.
//! A capture cannot show it: a pass that failed to build silently draws nothing, and the
//! frame looks merely wrong rather than broken.
//!
//! Only `WARN` and `ERROR` are kept. `INFO` is where the engine narrates startup, and a
//! scenario that had to read past it would be no better off than reading the log by hand.
//!
//! A shared buffer rather than the channel-and-transfer-system that bevy's
//! `log_layers_ecs` example uses: that example wants log lines as ECS messages, and this
//! wants a pile that one exclusive system reads on demand. `Arc<Mutex<_>>` is `Send +
//! Sync`, so it can be an ordinary resource and needs no per-frame system to move
//! anything.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use bevy::{
    log::{
        BoxedLayer, Level,
        tracing::{self, Subscriber},
        tracing_subscriber::{self, Layer},
    },
    prelude::*,
};
use serde_json::{Value, json};

/// Records held before the oldest start falling off. A run that is being watched gets
/// drained every few commands; one that is not would otherwise grow without limit, and an
/// unbounded buffer nobody reads is a leak.
const CAPACITY: usize = 512;

#[derive(Resource, Clone, Default)]
pub(super) struct LogBuffer(Arc<Mutex<Records>>);

#[derive(Default)]
struct Records {
    entries: VecDeque<Record>,
    /// Lost to [`CAPACITY`] since the last drain. Reported rather than hidden — a
    /// truncated log that says it is complete is worse than no log.
    dropped: u64,
}

struct Record {
    level: &'static str,
    target: String,
    message: String,
}

/// Installed through `LogPlugin::custom_layer`, which is a bare `fn` pointer and so
/// cannot capture — the resource has to be put into the app from in here.
///
/// Returns `None` unless the editor is being driven, because a buffer with no reader is
/// only a slow leak.
pub(super) fn layer(app: &mut App) -> Option<BoxedLayer> {
    std::env::var(super::server::SOCKET_ENV).ok()?;

    let buffer = LogBuffer::default();
    app.insert_resource(buffer.clone());
    Some(CaptureLayer { buffer }.boxed())
}

struct CaptureLayer {
    buffer: LogBuffer,
}

impl<S: Subscriber> Layer<S> for CaptureLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _context: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let metadata = event.metadata();
        // Compared rather than matched: `tracing::Level` is a struct with associated
        // constants, not an enum whose variants can appear in a pattern.
        let level = *metadata.level();
        let level = if level == Level::ERROR {
            "ERROR"
        } else if level == Level::WARN {
            "WARN"
        } else {
            return;
        };

        let mut message = None;
        event.record(&mut MessageVisitor(&mut message));
        let Some(message) = message else {
            return;
        };

        // A poisoned lock means a previous logger panicked mid-push. Dropping the record
        // is the right answer: this is a diagnostic, and panicking inside the log layer
        // while handling a log would take the app down for the sake of one line.
        if let Ok(mut records) = self.buffer.0.lock() {
            records.push(Record {
                level,
                target: metadata.target().to_owned(),
                message,
            });
        }
    }
}

impl Records {
    fn push(&mut self, record: Record) {
        if self.entries.len() == CAPACITY {
            self.entries.pop_front();
            self.dropped += 1;
        }
        self.entries.push_back(record);
    }
}

impl LogBuffer {
    /// Takes what has accumulated and leaves the buffer empty.
    ///
    /// Draining rather than snapshotting so that a scenario can bracket one step — "clear,
    /// do the thing, see what it said" — which is the question actually being asked. A
    /// snapshot would make every read include the whole session's startup noise.
    pub(super) fn drain(&self) -> Value {
        let Ok(mut records) = self.0.lock() else {
            return json!({ "available": false });
        };

        let dropped = std::mem::take(&mut records.dropped);
        let entries: Vec<Value> = records
            .entries
            .drain(..)
            .map(|record| {
                json!({
                    "level": record.level,
                    "target": record.target,
                    "message": record.message,
                })
            })
            .collect();

        let errors = entries
            .iter()
            .filter(|entry| entry["level"] == "ERROR")
            .count();

        json!({
            "available": true,
            "count": entries.len(),
            "errors": errors,
            "dropped": dropped,
            "entries": entries,
        })
    }
}

/// Pulls the formatted message out of an event. A `tracing` event is a set of fields, and
/// the one the macros put the text in is called `message`.
struct MessageVisitor<'a>(&'a mut Option<String>);

impl tracing::field::Visit for MessageVisitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            *self.0 = Some(format!("{value:?}"));
        }
    }
}
