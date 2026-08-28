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
//! Present only when the editor is being driven through the control socket: a buffer
//! nobody reads is a slow leak.

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

const CAPACITY: usize = 512;

/// What the run has complained about, shared between the log layer that fills it and
/// the system that drains it. Bounded: the oldest records fall off, and how many were
/// lost is reported beside the ones that survived.
#[derive(Resource, Clone, Default)]
pub(super) struct LogBuffer(Arc<Mutex<Records>>);

#[derive(Default)]
struct Records {
    entries: VecDeque<Record>,
    dropped: u64,
}

struct Record {
    level: &'static str,
    target: String,
    message: String,
}

/// The tracing layer that keeps warnings and errors, and the resource the drain reads
/// them from — both installed here, because `LogPlugin::custom_layer` takes a bare
/// `fn` pointer that cannot capture one to hand back.
///
/// `None` unless the control socket is named in the environment, which leaves the
/// editor logging exactly as it would without this module.
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
    /// Takes what has accumulated and leaves the buffer empty, so a caller can bracket
    /// one step — clear, do the thing, see what it said — rather than re-reading the
    /// whole session's startup noise on every look.
    ///
    /// Reports how many records were lost to the capacity since the last drain: a
    /// truncated log that says it is complete is worse than no log. `available: false`
    /// if the buffer's lock is poisoned, which is the only failure.
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

struct MessageVisitor<'a>(&'a mut Option<String>);

impl tracing::field::Visit for MessageVisitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            *self.0 = Some(format!("{value:?}"));
        }
    }
}
