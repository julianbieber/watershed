//! Everything the run complained about, kept where a scenario and the window can both
//! read it.
//!
//! "Did the shader compile" is the first question anyone has about a render change, and
//! the answer is otherwise on stderr mixed into a few thousand lines of engine chatter.
//! A capture cannot show it: a pass that failed to build silently draws nothing, and the
//! frame looks merely wrong rather than broken.
//!
//! Only `WARN` and `ERROR` are kept. `INFO` is where the engine narrates startup, and a
//! scenario that had to read past it would be no better off than reading the log by hand.
//!
//! Two readers share the one buffer and want different things, so the records are
//! retained and the control client's position is a cursor: `observe log` answers with
//! what arrived since it last asked, the window's log panel with everything still held.

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
/// the two readers. Bounded: the oldest records fall off, and how many were lost is
/// reported beside the ones that survived.
#[derive(Resource, Clone, Default)]
pub(crate) struct LogBuffer(Arc<Mutex<Records>>);

#[derive(Default)]
struct Records {
    entries: VecDeque<Record>,
    next: u64,
    drained: u64,
}

struct Record {
    seq: u64,
    level: &'static str,
    target: String,
    message: String,
}

/// One held record, as the window shows it.
pub(crate) struct LogLine {
    pub(crate) level: &'static str,
    pub(crate) message: String,
}

/// The newest records still held, and what is behind them.
///
/// `older` counts what is not in `lines`, whether it is still held or was lost to the
/// capacity, so a panel showing `lines` can say how much it is not showing.
#[derive(Default)]
pub(crate) struct LogView {
    pub(crate) lines: Vec<LogLine>,
    pub(crate) older: u64,
    pub(crate) errors: usize,
    pub(crate) held: usize,
}

/// The tracing layer that keeps warnings and errors, and the resource the readers take
/// them from — both installed here, because `LogPlugin::custom_layer` takes a bare
/// `fn` pointer that cannot capture one to hand back.
pub(super) fn layer(app: &mut App) -> Option<BoxedLayer> {
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

        self.buffer
            .record(level, metadata.target().to_owned(), message);
    }
}

impl Records {
    fn push(&mut self, level: &'static str, target: String, message: String) {
        if self.entries.len() == CAPACITY {
            self.entries.pop_front();
        }
        self.entries.push_back(Record {
            seq: self.next,
            level,
            target,
            message,
        });
        self.next += 1;
    }

    fn oldest(&self) -> u64 {
        self.next - self.entries.len() as u64
    }
}

impl LogBuffer {
    /// Keeps one warning or error, evicting the oldest once the capacity is reached.
    ///
    /// Does nothing if the buffer's lock is poisoned, which is the only failure.
    pub(crate) fn record(&self, level: &'static str, target: String, message: String) {
        if let Ok(mut records) = self.0.lock() {
            records.push(level, target, message);
        }
    }

    /// What has accumulated since this was last called, so a caller can bracket one
    /// step — read, do the thing, see what it said — rather than re-reading the whole
    /// session's startup noise on every look.
    ///
    /// The records stay in the buffer for the window; it is the caller's position that
    /// moves. Reports how many records were lost to the capacity before this caller
    /// reached them: a truncated log that says it is complete is worse than no log.
    /// `available: false` if the buffer's lock is poisoned, which is the only failure.
    pub(crate) fn since_last_read(&self) -> Value {
        let Ok(mut records) = self.0.lock() else {
            return json!({ "available": false });
        };

        let oldest = records.oldest();
        let first = records.drained.max(oldest);
        let dropped = oldest.saturating_sub(records.drained);
        records.drained = records.next;

        let entries: Vec<Value> = records
            .entries
            .iter()
            .filter(|record| record.seq >= first)
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

    /// The `limit` newest records still held, newest first, with the count of everything
    /// behind them. Reading it does not consume anything.
    ///
    /// An empty view if the buffer's lock is poisoned, which is the only failure.
    pub(crate) fn newest(&self, limit: usize) -> LogView {
        let Ok(records) = self.0.lock() else {
            return LogView::default();
        };

        let lines: Vec<LogLine> = records
            .entries
            .iter()
            .rev()
            .take(limit)
            .map(|record| LogLine {
                level: record.level,
                message: record.message.clone(),
            })
            .collect();

        LogView {
            older: records.next - lines.len() as u64,
            errors: records
                .entries
                .iter()
                .filter(|record| record.level == "ERROR")
                .count(),
            held: records.entries.len(),
            lines,
        }
    }

    /// How many records have ever been pushed, so a reader can tell whether anything has
    /// arrived before paying for [`LogBuffer::newest`]. 0 if the lock is poisoned.
    pub(crate) fn seq(&self) -> u64 {
        self.0.lock().map(|records| records.next).unwrap_or(0)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn warn(buffer: &LogBuffer, message: &str) {
        buffer.record("WARN", "test".to_owned(), message.to_owned());
    }

    // The split the whole retaining buffer exists for: the control client reading the log
    // must not take it away from the window's panel.
    #[test]
    fn a_record_the_control_client_read_is_still_there_for_the_window() {
        let buffer = LogBuffer::default();
        warn(&buffer, "ridged.wesl: line 9: no annotation");

        let read = buffer.since_last_read();
        assert_eq!(read["count"], 1);

        let view = buffer.newest(64);
        assert_eq!(view.lines.len(), 1);
        assert_eq!(view.lines[0].message, "ridged.wesl: line 9: no annotation");
    }

    // The contract the scenarios/*.txt files depend on, and the one thing this must not
    // break: a second `observe log` answers with what arrived after the first.
    #[test]
    fn a_second_read_returns_only_what_arrived_after_the_first() {
        let buffer = LogBuffer::default();
        warn(&buffer, "first");
        buffer.since_last_read();

        warn(&buffer, "second");
        let read = buffer.since_last_read();

        assert_eq!(read["count"], 1);
        assert_eq!(read["entries"][0]["message"], "second");
        assert_eq!(read["dropped"], 0);
    }

    // A truncated log has to say it is truncated: records evicted before the control
    // client reached them are counted, not silently missing.
    #[test]
    fn records_evicted_by_the_capacity_are_counted_against_a_reader_that_never_saw_them() {
        let buffer = LogBuffer::default();
        for index in 0..CAPACITY + 3 {
            warn(&buffer, &format!("{index}"));
        }

        let read = buffer.since_last_read();
        assert_eq!(read["dropped"], 3);
        assert_eq!(read["count"], CAPACITY);
    }

    // The panel's change signal has to survive eviction, so it counts pushes rather than
    // held records.
    #[test]
    fn the_sequence_counts_every_push_and_the_view_reports_what_is_behind_it() {
        let buffer = LogBuffer::default();
        for index in 0..5 {
            warn(&buffer, &format!("{index}"));
        }

        assert_eq!(buffer.seq(), 5);

        let view = buffer.newest(2);
        assert_eq!(view.lines[0].message, "4");
        assert_eq!(view.older, 3);
        assert_eq!(view.held, 5);
    }
}
