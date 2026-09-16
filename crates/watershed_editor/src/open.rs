//! Handing a file on disk to whatever the person edits files with.

use std::path::Path;
use std::process::{Command, Stdio};

/// Starts the person's editor on `path` and answers the program that started.
///
/// Tries `$VISUAL` first when it is set to something that is not blank, split on
/// whitespace so a program with arguments works (`code -w`), then `xdg-open`. The
/// path is the last argument.
///
/// Does not wait: the program keeps running after this returns, its standard streams
/// are null, and it is reaped on a thread of its own so no frame is held.
///
/// `Err` names every program tried, and means none of them could be started — on a
/// machine with neither on `PATH`, that is every press. Only a failure to *start* is
/// reported: a program that starts and then exits non-zero answers `Ok`.
pub fn open(path: &Path) -> Result<String, String> {
    start(&programs(std::env::var("VISUAL").ok().as_deref()), path)
}

fn programs(visual: Option<&str>) -> Vec<Vec<String>> {
    let mut programs = Vec::new();
    if let Some(visual) = visual
        && !visual.trim().is_empty()
    {
        programs.push(visual.split_whitespace().map(str::to_owned).collect());
    }
    programs.push(vec!["xdg-open".to_owned()]);
    programs
}

fn start(programs: &[Vec<String>], path: &Path) -> Result<String, String> {
    for words in programs {
        let Some((program, arguments)) = words.split_first() else {
            continue;
        };
        let child = Command::new(program)
            .args(arguments)
            .arg(path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        if let Ok(mut child) = child {
            std::thread::spawn(move || {
                let _ = child.wait();
            });
            return Ok(program.clone());
        }
    }
    let tried = programs
        .iter()
        .filter_map(|words| words.first())
        .map(|program| format!("`{program}`"))
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "could not open {}: none of {tried} would start",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A $VISUAL that lost its arguments starts the wrong thing, and one that lands
    // behind xdg-open is never reached.
    #[test]
    fn visual_keeps_its_arguments_and_comes_first() {
        assert_eq!(
            programs(Some("code -w")),
            vec![
                vec!["code".to_owned(), "-w".to_owned()],
                vec!["xdg-open".to_owned()],
            ]
        );
    }

    // A blank variable is what an unset one usually looks like in a shell, so both
    // have to leave xdg-open alone in the list.
    #[test]
    fn a_blank_or_unset_visual_leaves_only_xdg_open() {
        assert_eq!(programs(None), vec![vec!["xdg-open".to_owned()]]);
        assert_eq!(programs(Some("   ")), vec![vec!["xdg-open".to_owned()]]);
    }

    // This is the "neither available" arm the status bar exists for, and the message
    // is what it prints.
    #[test]
    fn nothing_that_starts_refuses_and_names_what_was_tried() {
        let error = start(
            &[vec!["watershed-no-such-program".to_owned()]],
            Path::new("/tmp/watershed-test.wesl"),
        )
        .unwrap_err();
        assert!(error.contains("watershed-no-such-program"), "{error}");
        assert!(error.contains("/tmp/watershed-test.wesl"), "{error}");
    }

    // A missing $VISUAL must fall through to xdg-open rather than refuse outright.
    #[test]
    fn a_missing_program_falls_through_to_one_that_exists() {
        let started = start(
            &[
                vec!["watershed-no-such-program".to_owned()],
                vec!["true".to_owned()],
            ],
            Path::new("/tmp/watershed-test.wesl"),
        );
        assert_eq!(started, Ok("true".to_owned()));
    }
}
