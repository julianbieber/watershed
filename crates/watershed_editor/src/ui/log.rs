//! The window's standing list of what the run has complained about.
//!
//! The status bar says one thing — the latest — and the next refusal overwrites it, so
//! a compile failure read half a second too late is gone. This is the same stream kept
//! as a list, so it can still be read after the bar has moved on. It is the history;
//! what a layer currently says about itself is on its card.

use bevy::feathers::controls::FeathersDisclosureToggle;
use bevy::feathers::theme::ThemeBackgroundColor;
use bevy::feathers::tokens;
use bevy::prelude::*;
use bevy::ui_widgets::ValueChange;

use crate::control::{LogBuffer, LogView};
use crate::ui::widgets::{self, one, set_text};

const ROWS: usize = 64;

const BODY_HEIGHT: f32 = 160.0;

/// Whether the body is showing. A resource rather than the toggle's own state because
/// [`sync`] is what shows and hides the body, and it cannot read the widget.
#[derive(Resource, Default)]
pub struct Open(pub bool);

/// The header line, which says how much the panel holds without it having to be opened.
#[derive(Component, Default, Clone)]
pub struct LogCaption;

/// The scrolling column of records, shown only while [`Open`] is set.
#[derive(Component, Default, Clone)]
pub struct LogBody;

/// One row, carrying its distance from the newest record.
#[derive(Component, Default, Clone)]
pub struct LogRow(usize);

/// The panel's scene, built closed and with every row empty — [`sync`] fills it in as
/// records arrive.
pub fn panel() -> impl Scene {
    let rows: Vec<Box<dyn SceneList>> = (0..ROWS)
        .map(|index| one(bsn! { (widgets::small("") LogRow({index})) }))
        .collect();
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Stretch,
            flex_shrink: 0.0,
            padding: px(6),
            row_gap: px(4),
        }
        ThemeBackgroundColor(tokens::WINDOW_BG)
        Children [
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(6),
                }
                Children [
                    (
                        @FeathersDisclosureToggle
                        on(|change: On<ValueChange<bool>>, mut open: ResMut<Open>| {
                            open.0 = change.value;
                        })
                    ),
                    (widgets::small("log") LogCaption),
                ]
            ),
            (
                Node {
                    display: Display::None,
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Stretch,
                    max_height: {px(BODY_HEIGHT)},
                    overflow: {Overflow::scroll_y()},
                }
                LogBody
                Children [ {rows} ]
            ),
        ]
    }
}

/// Shows or hides the body from [`Open`], and rewrites the rows when a record has
/// arrived since the last time it did.
///
/// The rows are written only on a change: the strings are cloned out of the buffer
/// behind its lock, and a warning arrives far less often than a frame does.
///
/// Does nothing but the show-and-hide when the log buffer is absent, which is the case
/// in an app built without a `LogPlugin`: the buffer is the log layer's resource.
pub fn sync(
    buffer: Option<Res<LogBuffer>>,
    open: Res<Open>,
    mut caption: Single<&mut Text, With<LogCaption>>,
    mut body: Single<&mut Node, With<LogBody>>,
    mut rows: Query<(&LogRow, &mut Text), Without<LogCaption>>,
    mut shown: Local<Option<u64>>,
) {
    let display = if open.0 { Display::Flex } else { Display::None };
    if body.display != display {
        body.display = display;
    }

    let Some(buffer) = buffer else {
        return;
    };
    let seq = buffer.seq();
    if *shown == Some(seq) {
        return;
    }
    *shown = Some(seq);

    let view = buffer.newest(ROWS);
    set_text(&mut caption, &caption_for(&view));

    let lines = lines_for(&view, ROWS);
    for (row, mut text) in &mut rows {
        set_text(&mut text, lines.get(row.0).map_or("", String::as_str));
    }
}

fn caption_for(view: &LogView) -> String {
    if view.held == 0 {
        return "log".to_owned();
    }
    if view.errors == 0 {
        return format!("log — {} held", view.held);
    }
    let plural = if view.errors == 1 { "" } else { "s" };
    format!("log — {} held, {} error{plural}", view.held, view.errors)
}

fn lines_for(view: &LogView, rows: usize) -> Vec<String> {
    let mut lines: Vec<String> = view
        .lines
        .iter()
        .take(rows)
        .map(|line| format!("{}  {}", line.level, line.message))
        .collect();

    let hidden = view.older + view.lines.len().saturating_sub(rows) as u64;
    if hidden > 0 && !lines.is_empty() {
        let behind = hidden + 1;
        let last = lines.len() - 1;
        lines[last] = format!("… {behind} earlier");
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(messages: &[&str]) -> LogView {
        let buffer = LogBuffer::default();
        for message in messages {
            buffer.record("WARN", "test".to_owned(), (*message).to_owned());
        }
        buffer.newest(ROWS)
    }

    // The acceptance observation, in the one place it can be asserted without a window:
    // a compile failure is listed with the file and line the log gave it, unaltered.
    #[test]
    fn a_compile_failure_is_listed_with_its_file_and_line() {
        let view = view(&["ridged.wgsl: line 9: parameter has no annotation"]);
        let lines = lines_for(&view, ROWS);
        assert_eq!(
            lines,
            vec!["WARN  ridged.wgsl: line 9: parameter has no annotation"]
        );
    }

    // The panel exists to be read right after something went wrong, and there is no
    // auto-scroll, so the thing that just happened has to be at the top.
    #[test]
    fn the_newest_record_is_the_first_row() {
        let view = view(&["oldest", "middle", "newest"]);
        let lines = lines_for(&view, ROWS);
        assert_eq!(lines[0], "WARN  newest");
        assert_eq!(lines[2], "WARN  oldest");
    }

    // A list that silently ends is worse than a short one: what does not fit is counted,
    // and the count includes the record the counter displaced.
    #[test]
    fn what_does_not_fit_is_counted_including_the_row_the_counter_displaced() {
        let mut view = view(&["c", "b", "a"]);
        view.older = 4;
        let lines = lines_for(&view, 3);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "WARN  a");
        assert_eq!(lines[2], "… 5 earlier");
    }

    // The header is what a closed panel says, so it has to distinguish nothing-happened
    // from something-did without being opened.
    #[test]
    fn the_header_says_what_is_held_and_how_much_of_it_failed() {
        assert_eq!(caption_for(&LogView::default()), "log");

        let mut view = view(&["a", "b"]);
        assert_eq!(caption_for(&view), "log — 2 held");

        view.errors = 1;
        assert_eq!(caption_for(&view), "log — 2 held, 1 error");
    }
}
