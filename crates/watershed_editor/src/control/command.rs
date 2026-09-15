//! What a client can ask for, and how far along it is.
//!
//! Every command is polled once a frame until it reports [`Poll::Done`], so "wait for
//! something" and "start a job and see it finish" are the same mechanism rather than two.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::terrain::SaveOptions;
use bevy::{
    prelude::*,
    render::view::screenshot::{Screenshot, save_to_disk},
    time::TimeUpdateStrategy,
};
use serde_json::{Value, json};

use super::observe::{self, Topic};
use crate::document::Document;
use crate::edit::Edit;
use crate::preset::Preset;
use crate::ui::report;
use crate::view::{EditorCamera, FreeView, fit_camera, look_at_cell, set_cells_across};

const DEFAULT_WAIT_FRAMES: u32 = 36_000;

/// How far along a command is, as of this frame.
pub(super) enum Poll {
    /// Not finished. Ask again next frame.
    Running,
    /// Finished, with the fields to send back.
    Done(Value),
    /// Finished badly. The message is sent instead of the fields.
    Failed(String),
}

/// One thing a client can ask the editor for.
///
/// A verb that starts a job also waits for it — there is no separate "did that
/// finish". That is the whole protocol: the reply is held until the effect has
/// actually happened, so a caller never sleeps and hopes, and a job that failed
/// reports the error rather than a success the caller would have to notice was empty.
///
/// Several variants carry state of their own, because a command is polled once a frame
/// and has to remember whether it has already started what it is waiting for.
pub(super) enum Command {
    /// Answers immediately. Proves the socket.
    Ping,
    /// Lets that many frames pass.
    Step(u32),
    /// Waits for a condition, or fails when the timeout runs out.
    Wait {
        /// What is being waited for.
        condition: Condition,
        /// Frames to wait before giving up.
        timeout: u32,
    },
    /// Builds a preset and waits for its bake.
    New {
        /// Extent of the terrain to build.
        size: UVec2,
        /// Seed the preset is built from.
        seed: u32,
        /// Which preset.
        preset: Preset,
        /// Whether the job has been asked for yet.
        started: bool,
    },
    /// Puts a field on screen. `field add <name>` is a different thing — an
    /// [`Edit`] — and parses to [`Command::Edit`].
    Field(String),
    /// An edit and the re-bake that answers it, held together: the reply says the
    /// effect has happened, and for an edit the effect is the bake rather than the
    /// changed number. A document that no longer bakes reports that error here rather
    /// than answering with a success nothing followed.
    /// A refused edit also reaches the status bar and the log panel, so a person at the
    /// editor sees why a command driven from elsewhere changed nothing.
    Edit {
        /// The change to make.
        edit: Edit,
        /// The reply from the edit itself, once it has been applied.
        applied: Option<Value>,
    },
    /// Bakes the whole document and waits for it.
    Bake {
        /// Whether the job has been asked for yet.
        started: bool,
    },
    /// Solves the water and waits for it, baking the document first if it is not whole
    /// — the same call the button makes, so the verb proves the button rather than a
    /// narrower thing beside it.
    SolveWater {
        /// Whether the job has been asked for yet.
        started: bool,
    },
    /// Drops the water and its spec. Synchronous.
    ResetWater,
    /// Writes the document and waits for it.
    Save {
        /// Where to write.
        path: PathBuf,
        /// What to put in the file.
        options: SaveOptions,
        /// Whether the job has been asked for yet.
        started: bool,
    },
    /// Reads a document and waits for it.
    Load {
        /// Where to read from.
        path: PathBuf,
        /// Whether the job has been asked for yet.
        started: bool,
    },
    /// Moves the camera by a number of cells.
    Pan(Vec2),
    /// Changes the zoom.
    Zoom(ZoomTo),
    /// Writes a PNG of the window and waits until it is on disk.
    Capture {
        /// Where to write the PNG.
        path: PathBuf,
        /// The screenshot entity, once spawned. Its *absence from the world* is what
        /// says the PNG is on disk, so the reply never races a half-written file.
        entity: Option<Entity>,
    },
    /// Takes the last change back, or replays the last one taken back, and the
    /// re-bake that answers it, on the same terms as [`Command::Edit`]: the reply
    /// says how far the history reaches each way once the document is settled.
    History {
        /// Redo rather than undo.
        redo: bool,
        /// The reply from the history itself, once it has moved.
        applied: Option<Value>,
    },
    /// Answers a question about the editor's state. See [`Topic`].
    Observe(Topic),
    /// Pins the frame delta, so a run is reproducible.
    FixedDelta(Duration),
    /// Undoes [`Command::FixedDelta`].
    Realtime,
    /// Ends the process.
    Quit,
}

/// What a `zoom` is aiming at.
pub(super) enum ZoomTo {
    /// The whole document, in the space the panels have left.
    Fit,
    /// That many document cells across the view.
    CellsAcross(f32),
}

/// What a `wait` is waiting for.
///
/// All three require the document to be *settled* rather than merely idle: an edit is
/// answered by a bake that a system opens on a later frame, so a document between the
/// two has nothing in flight and is not finished either.
pub(super) enum Condition {
    /// Every field carries a baked raster.
    Bake,
    /// The document carries a solved water state.
    Water,
    /// Nothing more than settled.
    Idle,
}

impl Command {
    /// The word this command was parsed from, for the reply. Every edit that is not a
    /// `set` answers `"field"`.
    pub(super) fn verb(&self) -> &'static str {
        match self {
            Self::Ping => "ping",
            Self::Step(_) => "step",
            Self::Wait { .. } => "wait",
            Self::New { .. } => "new",
            Self::Field(_) => "field",
            Self::Edit { edit, .. } => match edit {
                Edit::Set { .. } => "set",
                _ => "field",
            },
            Self::Bake { .. } => "bake",
            Self::SolveWater { .. } => "solve-water",
            Self::ResetWater => "reset-water",
            Self::Save { .. } => "save",
            Self::Load { .. } => "load",
            Self::Pan(_) => "pan",
            Self::Zoom(_) => "zoom",
            Self::Capture { .. } => "capture",
            Self::History { redo: false, .. } => "undo",
            Self::History { redo: true, .. } => "redo",
            Self::Observe(_) => "observe",
            Self::FixedDelta(_) => "fixed-delta",
            Self::Realtime => "realtime",
            Self::Quit => "quit",
        }
    }

    /// Reads one line as a command. Refused, with a message naming what was wrong,
    /// for an unknown verb, a missing argument or an unreadable value.
    pub(super) fn parse(line: &str) -> Result<Self, String> {
        let mut words = line.split_whitespace();
        let verb = words.next().ok_or("empty command")?;
        let rest: Vec<&str> = words.collect();

        match verb {
            "ping" => Ok(Self::Ping),
            "step" => Ok(Self::Step(optional_number(rest.first(), 1)?)),
            "wait" => {
                let what = rest.first().ok_or("wait needs something to wait for")?;
                Ok(Self::Wait {
                    condition: Condition::parse(what)?,
                    timeout: optional_number(rest.get(1), DEFAULT_WAIT_FRAMES)?,
                })
            }
            "new" => {
                let width = number(rest.first().ok_or("new needs a width")?)?;
                let height = number(rest.get(1).ok_or("new needs a height")?)?;
                let seed = optional_number(rest.get(2), 1)?;
                let preset = match rest.get(3) {
                    Some(word) => Preset::parse(word).ok_or(format!("no preset named `{word}`"))?,
                    None => Preset::default(),
                };
                Ok(Self::New {
                    size: UVec2::new(width, height),
                    seed,
                    preset,
                    started: false,
                })
            }
            "field" => match rest.as_slice() {
                ["add", name, ..] => Ok(Self::Edit {
                    edit: Edit::AddField {
                        name: (*name).to_owned(),
                    },
                    applied: None,
                }),
                ["rm", name, ..] => Ok(Self::Edit {
                    edit: Edit::RemoveField {
                        name: (*name).to_owned(),
                    },
                    applied: None,
                }),
                ["rename", ..] => {
                    Err("field rename is gone: rename the file in shaders/ instead".to_owned())
                }
                ["rm"] => Err("field rm needs a name".to_owned()),
                [name, ..] => Ok(Self::Field((*name).to_owned())),
                [] => Err("field needs a name".to_owned()),
            },
            "set" => {
                let path = rest.first().ok_or("set needs a path")?;
                Ok(Self::Edit {
                    edit: Edit::Set {
                        path: (*path).to_owned(),
                        words: owned(&rest[1..]),
                    },
                    applied: None,
                })
            }
            "bake" => Ok(Self::Bake { started: false }),
            "solve-water" => Ok(Self::SolveWater { started: false }),
            "reset-water" => Ok(Self::ResetWater),
            "save" => Ok(Self::Save {
                path: PathBuf::from(rest.first().ok_or("save needs a path")?),
                options: match rest.get(1) {
                    Some(&"document") | None => SaveOptions::document(),
                    Some(&"export") => SaveOptions::export(),
                    Some(word) => return Err(format!("no save option named `{word}`")),
                },
                started: false,
            }),
            "load" => Ok(Self::Load {
                path: PathBuf::from(rest.first().ok_or("load needs a path")?),
                started: false,
            }),
            "pan" => {
                let x = number(rest.first().ok_or("pan needs a cell x")?)?;
                let y = number(rest.get(1).ok_or("pan needs a cell y")?)?;
                Ok(Self::Pan(Vec2::new(x, y)))
            }
            "zoom" => match rest.first() {
                Some(&"fit") => Ok(Self::Zoom(ZoomTo::Fit)),
                Some(word) => Ok(Self::Zoom(ZoomTo::CellsAcross(number(word)?))),
                None => Err("zoom needs `fit` or a cell count".to_owned()),
            },
            "capture" => Ok(Self::Capture {
                path: PathBuf::from(rest.first().ok_or("capture needs a path")?),
                entity: None,
            }),
            "undo" => Ok(Self::History {
                redo: false,
                applied: None,
            }),
            "redo" => Ok(Self::History {
                redo: true,
                applied: None,
            }),
            "observe" => Ok(Self::Observe(Topic::parse(
                rest.first().ok_or("observe needs a topic")?,
            )?)),
            "fixed-delta" => Ok(Self::FixedDelta(delta(
                rest.first().ok_or("fixed-delta needs a duration")?,
            )?)),
            "realtime" => Ok(Self::Realtime),
            "quit" => Ok(Self::Quit),
            other => Err(format!("no such command: {other}")),
        }
    }

    /// Advances the command by one frame and says how far along it is.
    ///
    /// Called once a frame with the frames since the command arrived, until it answers
    /// something other than [`Poll::Running`]. May act on the world — a job-starting
    /// command starts its job on the first call and waits on the rest.
    pub(super) fn poll(&mut self, world: &mut World, elapsed: u32) -> Poll {
        match self {
            Self::Ping => Poll::Done(json!({})),

            Self::Step(frames) => {
                if elapsed >= *frames {
                    Poll::Done(json!({}))
                } else {
                    Poll::Running
                }
            }

            Self::Wait { condition, timeout } => {
                if condition.met(world) {
                    Poll::Done(json!({}))
                } else if elapsed >= *timeout {
                    Poll::Failed(format!("timed out after {elapsed} frames"))
                } else {
                    Poll::Running
                }
            }

            Self::New {
                size,
                seed,
                preset,
                started,
            } => {
                if !*started {
                    *started = true;
                    let mut document = world.resource_mut::<Document>();
                    if let Err(error) = document.start_new(*size, *seed, *preset) {
                        return Poll::Failed(error);
                    }
                    return Poll::Running;
                }
                finished(world, |document| {
                    json!({
                        "size": [document.size.x, document.size.y],
                        "fields": document.field_names(),
                    })
                })
            }

            Self::Field(name) => {
                let mut document = world.resource_mut::<Document>();
                match document.set_active(name) {
                    Ok(()) => Poll::Done(json!({ "field": name })),
                    Err(error) => Poll::Failed(error),
                }
            }

            Self::Edit { edit, applied } => {
                if applied.is_none() {
                    let mut document = world.resource_mut::<Document>();
                    match document.apply(edit) {
                        Ok(value) => *applied = Some(value),
                        Err(error) => {
                            report(&mut document, Err(error.clone()));
                            return Poll::Failed(error);
                        }
                    }
                    return Poll::Running;
                }
                answered(world, applied)
            }

            Self::History { redo, applied } => {
                if applied.is_none() {
                    let mut document = world.resource_mut::<Document>();
                    let moved = if *redo {
                        document.redo()
                    } else {
                        document.undo()
                    };
                    if let Err(error) = moved {
                        return Poll::Failed(error);
                    }
                    let depth = document.history();
                    *applied = Some(json!({ "undo": depth.undo, "redo": depth.redo }));
                    return Poll::Running;
                }
                answered(world, applied)
            }

            Self::Bake { started } => {
                if !*started {
                    *started = true;
                    let mut document = world.resource_mut::<Document>();
                    if let Err(error) = document.start_bake() {
                        return Poll::Failed(error);
                    }
                    return Poll::Running;
                }
                finished(
                    world,
                    |document| json!({ "baked": document.baked().name() }),
                )
            }

            Self::SolveWater { started } => {
                if !*started {
                    *started = true;
                    let mut document = world.resource_mut::<Document>();
                    if let Err(error) = document.solve_with_bake() {
                        return Poll::Failed(error);
                    }
                    return Poll::Running;
                }
                finished(world, |document| {
                    let lakes = document
                        .terrain()
                        .and_then(|terrain| terrain.water())
                        .map(|water| water.lakes())
                        .unwrap_or(0);
                    json!({ "lakes": lakes })
                })
            }

            Self::ResetWater => {
                let mut document = world.resource_mut::<Document>();
                match document.reset_water() {
                    Ok(()) => Poll::Done(json!({})),
                    Err(error) => Poll::Failed(error),
                }
            }

            Self::Save {
                path,
                options,
                started,
            } => {
                if !*started {
                    *started = true;
                    let mut document = world.resource_mut::<Document>();
                    if let Err(error) = document.start_save(path.clone(), *options) {
                        return Poll::Failed(error);
                    }
                    return Poll::Running;
                }
                let path = path.clone();
                finished(world, move |_| {
                    let bytes = directory_bytes(&path);
                    json!({ "path": path.display().to_string(), "bytes": bytes })
                })
            }

            Self::Load { path, started } => {
                if !*started {
                    *started = true;
                    let mut document = world.resource_mut::<Document>();
                    if let Err(error) = document.start_load(path.clone()) {
                        return Poll::Failed(error);
                    }
                    return Poll::Running;
                }
                finished(world, |document| {
                    json!({
                        "size": [document.size.x, document.size.y],
                        "fields": document.field_names(),
                        "water": document
                            .terrain()
                            .is_some_and(|terrain| terrain.water().is_some()),
                    })
                })
            }

            Self::Pan(cell) => {
                let size = world.resource::<Document>().size;
                let cell = *cell;
                match world
                    .query_filtered::<&mut Transform, With<EditorCamera>>()
                    .single_mut(world)
                {
                    Ok(mut transform) => {
                        look_at_cell(&mut transform, size, cell);
                        Poll::Done(json!({ "centre": [cell.x, cell.y] }))
                    }
                    Err(error) => Poll::Failed(error.to_string()),
                }
            }

            Self::Zoom(to) => {
                let terrain_size = world
                    .resource::<Document>()
                    .terrain()
                    .map(|terrain| terrain.size);
                let free = *world.resource::<FreeView>();
                let mut query =
                    world.query_filtered::<(&mut Transform, &mut Projection), With<EditorCamera>>();
                let Ok((mut transform, mut projection)) = query.single_mut(world) else {
                    return Poll::Failed("there is no editor camera".to_owned());
                };

                match to {
                    ZoomTo::Fit => match terrain_size {
                        Some(size) => {
                            fit_camera(&mut transform, &mut projection, size, free);
                            Poll::Done(json!({ "fit": [size.x, size.y] }))
                        }
                        None => Poll::Failed("there is no document to fit".to_owned()),
                    },
                    ZoomTo::CellsAcross(cells) => {
                        set_cells_across(&mut projection, *cells);
                        Poll::Done(json!({ "cells_across": cells }))
                    }
                }
            }

            Self::Capture { path, entity } => match entity {
                None => {
                    if let Some(parent) = path.parent()
                        && let Err(error) = std::fs::create_dir_all(parent)
                    {
                        return Poll::Failed(format!(
                            "cannot create {}: {error}",
                            parent.display()
                        ));
                    }
                    *entity = Some(
                        world
                            .spawn(Screenshot::primary_window())
                            .observe(save_to_disk(path.clone()))
                            .id(),
                    );
                    Poll::Running
                }
                Some(id) => {
                    if world.entities().contains(*id) {
                        Poll::Running
                    } else {
                        Poll::Done(json!({ "path": path.display().to_string() }))
                    }
                }
            },

            Self::Observe(topic) => Poll::Done(observe::run(world, topic)),

            Self::FixedDelta(delta) => {
                world.insert_resource(TimeUpdateStrategy::ManualDuration(*delta));
                Poll::Done(json!({ "seconds": delta.as_secs_f32() }))
            }

            Self::Realtime => {
                world.insert_resource(TimeUpdateStrategy::Automatic);
                Poll::Done(json!({}))
            }

            Self::Quit => {
                world.write_message(AppExit::Success);
                Poll::Done(json!({}))
            }
        }
    }
}

fn directory_bytes(path: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| entry.metadata().ok())
        .filter(|meta| meta.is_file())
        .map(|meta| meta.len())
        .sum()
}

fn answered(world: &World, applied: &mut Option<Value>) -> Poll {
    let document = world.resource::<Document>();
    if !document.is_settled() {
        return Poll::Running;
    }
    match document.error() {
        Some(error) => Poll::Failed(error.to_owned()),
        None => Poll::Done(applied.take().unwrap_or_else(|| json!({}))),
    }
}

fn finished(world: &mut World, fields: impl FnOnce(&Document) -> Value) -> Poll {
    let document = world.resource::<Document>();
    if document.is_busy() {
        return Poll::Running;
    }
    match document.error() {
        Some(error) => Poll::Failed(error.to_owned()),
        None => Poll::Done(fields(document)),
    }
}

impl Condition {
    fn parse(word: &str) -> Result<Self, String> {
        match word {
            "bake" | "terrain" => Ok(Self::Bake),
            "water" => Ok(Self::Water),
            "idle" => Ok(Self::Idle),
            other => Err(format!("nothing to wait for called {other}")),
        }
    }

    fn met(&self, world: &World) -> bool {
        let document = world.resource::<Document>();
        if !document.is_settled() {
            return false;
        }
        match self {
            Self::Idle => true,
            Self::Bake => document.terrain().is_some_and(|terrain| {
                let faults = terrain.field_faults();
                terrain
                    .fields
                    .iter()
                    .all(|f| !f.baked().is_empty() || faults.iter().any(|(id, _)| *id == f.id))
            }),
            Self::Water => document
                .terrain()
                .is_some_and(|terrain| terrain.water().is_some()),
        }
    }
}

fn owned(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| (*word).to_owned()).collect()
}

fn number<T: std::str::FromStr>(word: &str) -> Result<T, String> {
    word.parse().map_err(|_| format!("not a number: {word}"))
}

fn optional_number<T: std::str::FromStr>(word: Option<&&str>, fallback: T) -> Result<T, String> {
    match word {
        Some(word) => number(word),
        None => Ok(fallback),
    }
}

fn delta(word: &str) -> Result<Duration, String> {
    let seconds = match word.split_once('/') {
        Some((numerator, denominator)) => {
            let numerator: f32 = number(numerator)?;
            let denominator: f32 = number(denominator)?;
            if denominator == 0.0 {
                return Err("a delta cannot be divided by zero".to_owned());
            }
            numerator / denominator
        }
        None => number(word)?,
    };
    if seconds < 0.0 {
        return Err(format!("a delta cannot be negative: {seconds}"));
    }
    Ok(Duration::from_secs_f32(seconds))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A frame budget is written as a fraction everywhere else, so `1/60` has to be
    // accepted as well as `0.016`. The two refusals are the ones a fraction opens up: a
    // division by zero, and a negative that `Duration::from_secs_f32` would panic on.
    #[test]
    fn a_delta_is_read_as_a_fraction_or_a_decimal() {
        assert_eq!(delta("1/60").unwrap(), Duration::from_secs_f32(1.0 / 60.0));
        assert_eq!(delta("0").unwrap(), Duration::ZERO);
        assert!(delta("1/0").is_err());
        assert!(delta("-1").is_err());
    }

    // Every verb, not just the awkward ones: a verb the control client cannot parse is
    // a feature nothing outside the window can reach, which is exactly the drift the
    // standing rule about the two surfaces exists to catch.
    #[test]
    fn every_verb_parses_to_the_verb_it_names() {
        let lines = [
            ("ping", "ping"),
            ("step 4", "step"),
            ("wait bake", "wait"),
            ("wait water 600", "wait"),
            ("new 256 256 7 ridges", "new"),
            ("field height", "field"),
            ("field add biomes", "field"),
            ("field rm base", "field"),
            ("set height.scale 0.004", "set"),
            ("set height.offset 0.5 0.25", "set"),
            ("set height.shift 2", "set"),
            ("bake", "bake"),
            ("solve-water", "solve-water"),
            ("reset-water", "reset-water"),
            ("save /tmp/a-terrain", "save"),
            ("save /tmp/a-terrain export", "save"),
            ("load /tmp/a-terrain", "load"),
            ("pan 100 200", "pan"),
            ("zoom fit", "zoom"),
            ("zoom 512", "zoom"),
            ("capture /tmp/a.png", "capture"),
            ("undo", "undo"),
            ("redo", "redo"),
            ("observe water", "observe"),
            ("fixed-delta 1/60", "fixed-delta"),
            ("realtime", "realtime"),
            ("quit", "quit"),
        ];

        for (line, verb) in lines {
            let command = Command::parse(line).unwrap_or_else(|error| panic!("{line}: {error}"));
            assert_eq!(command.verb(), verb, "{line}");
        }
    }

    // A caller writes these by hand, so a mistyped word or a missing argument has to
    // come back as a message rather than as a command that half happened.
    #[test]
    fn a_command_that_is_not_a_verb_is_refused() {
        assert!(Command::parse("wander about").is_err());
        assert!(Command::parse("").is_err());
        assert!(Command::parse("new 256").is_err());
        assert!(Command::parse("new 256 256 1 nothing-like-this").is_err());
        assert!(Command::parse("zoom").is_err());
        assert!(Command::parse("save /tmp/a-terrain sideways").is_err());
        assert!(Command::parse("field").is_err());
        assert!(Command::parse("field rm").is_err());
        assert!(Command::parse("layer sideways height 1").is_err());
        assert!(Command::parse("set").is_err());
    }

    // The brush is gone rather than hidden: a script still written against it has to be
    // told the verb, the topic or the preset does not exist, naming the word it used,
    // rather than have any of it half happen.
    #[test]
    fn the_brush_verbs_its_topic_and_a_removed_preset_are_refused_by_name() {
        for (line, refusal) in [
            ("brush radius 24", "no such command: brush"),
            ("stroke 10,20", "no such command: stroke"),
            ("observe brush", "nothing to observe called brush"),
        ] {
            let Err(error) = Command::parse(line) else {
                panic!("`{line}` parsed");
            };
            assert_eq!(error, refusal);
        }
        let Err(error) = Command::parse("new 256 256 7 regions") else {
            panic!("the regions preset parsed");
        };
        assert!(error.contains("regions"), "{error}");
    }

    // A field is one shader file now, so a script still driving the node graph, adopting
    // a stock shader or renaming a field in place has to be refused with a message that
    // names the word it used, rather than have a half-edited document behind it.
    #[test]
    fn the_node_graph_verbs_its_topic_and_field_rename_are_refused_by_name() {
        for (line, refusal) in [
            ("node add height shader:ridged", "no such command: node"),
            ("shader adopt ridged", "no such command: shader"),
            ("observe nodes", "nothing to observe called nodes"),
            (
                "field rename base continent",
                "field rename is gone: rename the file in shaders/ instead",
            ),
        ] {
            let Err(error) = Command::parse(line) else {
                panic!("`{line}` parsed");
            };
            assert_eq!(error, refusal);
        }
    }

    // A field whose file names no field is left unbaked on purpose, so a scenario
    // waiting for the bake after such an edit has to be released rather than wait for a
    // raster that never comes.
    #[test]
    fn waiting_for_the_bake_is_met_by_a_field_left_unbaked_by_its_fault() {
        use crate::terrain::{Field, TerrainSpec};
        let mut terrain = TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("base").held(0.25))
            .with_field(Field::new("height").reading(&["nowhere"]));
        terrain.bake_in_place().unwrap();
        assert!(terrain.field("height").unwrap().baked().is_empty());

        let mut document = Document::default();
        document.adopt(terrain);
        let mut world = World::new();
        world.insert_resource(document);
        assert!(Condition::Bake.met(&world));
    }
}
