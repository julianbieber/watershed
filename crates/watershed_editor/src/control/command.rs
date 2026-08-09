//! What a client can ask for, and how far along it is.
//!
//! Every command is polled once a frame until it reports [`Poll::Done`], so "wait for
//! something" and "start a job and see it finish" are the same mechanism rather than two.

use std::path::PathBuf;
use std::time::Duration;

use bevy::{
    prelude::*,
    render::view::screenshot::{Screenshot, save_to_disk},
    time::TimeUpdateStrategy,
};
use serde_json::{Value, json};
use watershed::SaveOptions;

use super::observe::{self, Topic};
use crate::document::Document;
use crate::edit::{Edit, parse_op};
use crate::preset::Preset;
use crate::view::{EditorCamera, FreeView, fit_camera, look_at_cell, set_cells_across};

/// Ten minutes at 60 Hz. Long enough for a 4096-square solve on a slow machine, short
/// enough that a stuck scenario reports rather than hangs.
const DEFAULT_WAIT_FRAMES: u32 = 36_000;

pub(super) enum Poll {
    Running,
    Done(Value),
    Failed(String),
}

pub(super) enum Command {
    Ping,
    Step(u32),
    Wait {
        condition: Condition,
        timeout: u32,
    },
    /// TODO(jb-doc): why starting a job and waiting for it are one verb rather than two —
    /// that the reply is held until the effect has happened, which is the whole protocol.
    New {
        size: UVec2,
        seed: u32,
        preset: Preset,
        started: bool,
    },
    Field(String),
    /// An edit and the re-bake that answers it, held together for the reason every other
    /// job-starting verb is: the reply says the effect has happened, and for an edit the
    /// effect is the bake rather than the changed number.
    Edit {
        edit: Edit,
        applied: Option<Value>,
    },
    Bake {
        started: bool,
    },
    SolveWater {
        started: bool,
    },
    ResetWater,
    Save {
        path: PathBuf,
        options: SaveOptions,
        started: bool,
    },
    Load {
        path: PathBuf,
        started: bool,
    },
    Pan(Vec2),
    Zoom(ZoomTo),
    Capture {
        path: PathBuf,
        /// The screenshot entity, once spawned. Its *absence from the world* is what says
        /// the PNG is on disk — see the poll arm.
        entity: Option<Entity>,
    },
    Observe(Topic),
    FixedDelta(Duration),
    Realtime,
    Quit,
}

pub(super) enum ZoomTo {
    Fit,
    CellsAcross(f32),
}

pub(super) enum Condition {
    /// The document is idle and carries a baked terrain.
    Bake,
    /// The document is idle and carries a solved water state.
    Water,
    /// Nothing is in flight, whatever that job was.
    Idle,
}

impl Command {
    pub(super) fn verb(&self) -> &'static str {
        match self {
            Self::Ping => "ping",
            Self::Step(_) => "step",
            Self::Wait { .. } => "wait",
            Self::New { .. } => "new",
            Self::Field(_) => "field",
            Self::Edit { edit, .. } => match edit {
                Edit::Set { .. } => "set",
                _ => "layer",
            },
            Self::Bake { .. } => "bake",
            Self::SolveWater { .. } => "solve-water",
            Self::ResetWater => "reset-water",
            Self::Save { .. } => "save",
            Self::Load { .. } => "load",
            Self::Pan(_) => "pan",
            Self::Zoom(_) => "zoom",
            Self::Capture { .. } => "capture",
            Self::Observe(_) => "observe",
            Self::FixedDelta(_) => "fixed-delta",
            Self::Realtime => "realtime",
            Self::Quit => "quit",
        }
    }

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
            "field" => Ok(Self::Field(
                rest.first().ok_or("field needs a name")?.to_string(),
            )),
            "layer" => Ok(Self::Edit {
                edit: layer_edit(&rest)?,
                applied: None,
            }),
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
                    Some(&"full") => SaveOptions::full(),
                    Some(&"bakes-only") => SaveOptions::bakes_only(),
                    Some(&"layers-only") => SaveOptions::layers_only(),
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
                        Err(error) => return Poll::Failed(error),
                    }
                    return Poll::Running;
                }
                // Held until the re-bake the edit provoked has landed, so a scenario that
                // observes the field on the next line is reading the world the edit made.
                // A stack that no longer bakes — a cycle a toggle uncovered — reports the
                // error here rather than answering with a success nothing followed.
                let document = world.resource::<Document>();
                if !document.is_settled() {
                    return Poll::Running;
                }
                match document.error() {
                    Some(error) => Poll::Failed(error.to_owned()),
                    None => Poll::Done(applied.take().unwrap_or_else(|| json!({}))),
                }
            }

            Self::Bake { started } => {
                if !*started {
                    *started = true;
                    let mut document = world.resource_mut::<Document>();
                    if let Err(error) = document.start_bake(None) {
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
                    // The same call the button makes, so the verb proves the button rather
                    // than a narrower thing beside it: a solve needs the whole document
                    // baked, and this bakes it when it is not.
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
                    let bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
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
                // The fit needs the terrain and the projection at once, so the size is
                // read out before the camera is borrowed.
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
                // `clear_screenshots` despawns the entity in `First`, which runs strictly
                // after the `ScreenshotCaptured` observer has written the file. So the
                // entity being gone is the signal that the PNG is on disk — no polling the
                // filesystem, and no racing a half-written file.
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

/// A job's reply is held until the document goes idle, and a job that failed reports the
/// error rather than a success the caller would have to notice was empty.
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

    /// Settled rather than merely idle: an edit is answered by a bake a system opens, so a
    /// document between the two has nothing in flight and is not finished either.
    fn met(&self, world: &World) -> bool {
        let document = world.resource::<Document>();
        if !document.is_settled() {
            return false;
        }
        match self {
            Self::Idle => true,
            Self::Bake => document
                .terrain()
                .is_some_and(|terrain| terrain.fields.iter().all(|f| !f.baked().is_empty())),
            Self::Water => document
                .terrain()
                .is_some_and(|terrain| terrain.water().is_some()),
        }
    }
}

/// TODO(jb-doc): why the four structural edits are one verb with a sub-word rather than
/// four verbs, where `set` is a verb of its own.
fn layer_edit(rest: &[&str]) -> Result<Edit, String> {
    let what = *rest.first().ok_or("layer needs add, rm, move or toggle")?;
    let field = (*rest.get(1).ok_or("layer needs a field name")?).to_owned();
    match what {
        "add" => Ok(Edit::Add {
            field,
            op: parse_op(&owned(&rest[2..]))?,
        }),
        "rm" => Ok(Edit::Remove {
            field,
            index: number(rest.get(2).ok_or("layer rm needs an index")?)?,
        }),
        "move" => Ok(Edit::Move {
            field,
            index: number(rest.get(2).ok_or("layer move needs an index")?)?,
            to: number(rest.get(3).ok_or("layer move needs somewhere to go")?)?,
        }),
        "toggle" => Ok(Edit::Toggle {
            field,
            index: number(rest.get(2).ok_or("layer toggle needs an index")?)?,
            // No word at all flips it, which is what a keyboard-less caller wants; a word
            // states it, which is what a scenario wants so a re-run cannot drift.
            enabled: match rest.get(3) {
                None => None,
                Some(&"on") => Some(true),
                Some(&"off") => Some(false),
                Some(word) => return Err(format!("a toggle is on or off, not `{word}`")),
            },
        }),
        other => Err(format!("no layer edit called `{other}`")),
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

/// Accepts `1/60` as well as `0.016`, because a frame budget is the thing a scenario
/// actually means and writing it as a fraction is how it is written everywhere else.
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

    #[test]
    fn a_delta_is_read_as_a_fraction_or_a_decimal() {
        assert_eq!(delta("1/60").unwrap(), Duration::from_secs_f32(1.0 / 60.0));
        assert_eq!(delta("0").unwrap(), Duration::ZERO);
        assert!(delta("1/0").is_err());
        assert!(delta("-1").is_err());
    }

    /// TODO(jb-comment): why every verb is parsed here rather than only the awkward ones —
    /// that a verb the ctl cannot parse is a feature nothing can reach, which is exactly
    /// the drift the standing rule exists to catch.
    #[test]
    fn every_verb_parses_to_the_verb_it_names() {
        let lines = [
            ("ping", "ping"),
            ("step 4", "step"),
            ("wait bake", "wait"),
            ("wait water 600", "wait"),
            ("new 256 256 7 ridges", "new"),
            ("field height", "field"),
            ("layer add height noise fbm 0.01", "layer"),
            ("layer add height constant 0.25", "layer"),
            ("layer add height slope base 4", "layer"),
            ("layer rm height 2", "layer"),
            ("layer move height 2 0", "layer"),
            ("layer toggle height 1", "layer"),
            ("layer toggle height 1 off", "layer"),
            ("set height.1.amplitude 0.5", "set"),
            ("set height.1.blend mul", "set"),
            ("set height.1.mask field moisture 0.4 0.6 0 1", "set"),
            ("set height.1.op.scale 0.004", "set"),
            ("set height.shift 2", "set"),
            ("bake", "bake"),
            ("solve-water", "solve-water"),
            ("reset-water", "reset-water"),
            ("save /tmp/a.watershed", "save"),
            ("save /tmp/a.watershed full", "save"),
            ("load /tmp/a.watershed", "load"),
            ("pan 100 200", "pan"),
            ("zoom fit", "zoom"),
            ("zoom 512", "zoom"),
            ("capture /tmp/a.png", "capture"),
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

    #[test]
    fn a_command_that_is_not_a_verb_is_refused() {
        assert!(Command::parse("wander about").is_err());
        assert!(Command::parse("").is_err());
        assert!(Command::parse("new 256").is_err());
        assert!(Command::parse("new 256 256 1 nothing-like-this").is_err());
        assert!(Command::parse("zoom").is_err());
        assert!(Command::parse("save /tmp/a.watershed sideways").is_err());
        assert!(Command::parse("layer").is_err());
        assert!(Command::parse("layer add height").is_err());
        assert!(Command::parse("layer sideways height 1").is_err());
        assert!(Command::parse("layer toggle height 1 maybe").is_err());
        assert!(Command::parse("layer rm height").is_err());
        assert!(Command::parse("set").is_err());
    }
}
