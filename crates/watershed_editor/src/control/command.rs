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
use crate::brush::{BrushSettings, apply_stroke};
use crate::document::Document;
use crate::edit::{BrushChange, Edit, brush_summary, parse_op};
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
    /// changed number. A stack that no longer bakes — a cycle a toggle uncovered —
    /// reports that error here rather than answering with a success nothing followed.
    /// A refused edit also reaches the status bar and the log panel, so a person at the
    /// editor sees why a command driven from elsewhere changed nothing.
    Edit {
        /// The change to make.
        edit: Edit,
        /// The reply from the edit itself, once it has been applied.
        applied: Option<Value>,
    },
    /// Several of the brush's settings at once, where a node's are set one at a time:
    /// a brush is one tool with a handful of knobs, and a caller usually means to state
    /// a whole configuration rather than nudge a number.
    Brush(Vec<BrushChange>),
    /// A stroke and the re-bake that answers it, on the same terms as
    /// [`Command::Edit`] — except that what it waits for is the rectangle the stroke
    /// made stale rather than the whole view.
    Stroke {
        /// The polyline, in document cells.
        points: Vec<Vec2>,
        /// The reply from the stroke itself, once it has been applied.
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
    /// Copies a stock shader into the document's shader directory. Synchronous, and
    /// answers the name the copy was given — which is what a `layer add` names.
    AdoptShader(String),
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
    /// The word this command was parsed from, for the reply. Every structural graph
    /// edit answers `"node"`, whatever it did.
    pub(super) fn verb(&self) -> &'static str {
        match self {
            Self::Ping => "ping",
            Self::Step(_) => "step",
            Self::Wait { .. } => "wait",
            Self::New { .. } => "new",
            Self::Field(_) => "field",
            Self::Edit { edit, .. } => match edit {
                Edit::Set { .. } => "set",
                Edit::AddField { .. } | Edit::RenameField { .. } | Edit::RemoveField { .. } => {
                    "field"
                }
                _ => "node",
            },
            Self::Brush(_) => "brush",
            Self::Stroke { .. } => "stroke",
            Self::Bake { .. } => "bake",
            Self::SolveWater { .. } => "solve-water",
            Self::ResetWater => "reset-water",
            Self::AdoptShader(_) => "shader",
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
                ["rename", from, to, ..] => Ok(Self::Edit {
                    edit: Edit::RenameField {
                        from: (*from).to_owned(),
                        to: (*to).to_owned(),
                    },
                    applied: None,
                }),
                ["rm", name, ..] => Ok(Self::Edit {
                    edit: Edit::RemoveField {
                        name: (*name).to_owned(),
                    },
                    applied: None,
                }),
                ["rename", ..] => Err("field rename needs the old name and the new one".to_owned()),
                ["rm"] => Err("field rm needs a name".to_owned()),
                [name, ..] => Ok(Self::Field((*name).to_owned())),
                [] => Err("field needs a name".to_owned()),
            },
            "node" => Ok(Self::Edit {
                edit: node_edit(&rest)?,
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
            "brush" => {
                if rest.is_empty() || !rest.len().is_multiple_of(2) {
                    return Err(
                        "brush takes a name and a value, and as many pairs as you like".to_owned(),
                    );
                }
                rest.chunks(2)
                    .map(|pair| BrushChange::parse(pair[0], pair[1]))
                    .collect::<Result<Vec<_>, _>>()
                    .map(Self::Brush)
            }
            "stroke" => {
                let points = rest
                    .iter()
                    .map(|word| point(word))
                    .collect::<Result<Vec<_>, _>>()?;
                if points.is_empty() {
                    return Err("stroke needs at least one cell to paint at".to_owned());
                }
                Ok(Self::Stroke {
                    points,
                    applied: None,
                })
            }
            "bake" => Ok(Self::Bake { started: false }),
            "solve-water" => Ok(Self::SolveWater { started: false }),
            "reset-water" => Ok(Self::ResetWater),
            "shader" => match rest.first() {
                Some(&"adopt") => Ok(Self::AdoptShader(
                    (*rest.get(1).ok_or("shader adopt needs a stock name")?).to_owned(),
                )),
                _ => Err("shader needs adopt".to_owned()),
            },
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

            Self::Brush(changes) => {
                let mut settings = world.resource_mut::<BrushSettings>();
                for change in changes.iter() {
                    change.apply(&mut settings.0);
                }
                Poll::Done(brush_summary(&settings.0))
            }

            Self::Stroke { points, applied } => {
                if applied.is_none() {
                    let brush = world.resource::<BrushSettings>().0;
                    let mut document = world.resource_mut::<Document>();
                    match apply_stroke(&mut document, &brush, points, false) {
                        Ok(value) => *applied = Some(value),
                        Err(error) => return Poll::Failed(error),
                    }
                    return Poll::Running;
                }
                answered(world, applied)
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

            Self::AdoptShader(stock) => {
                let mut library = world.resource_mut::<crate::gpu::ShaderLibrary>();
                match library.adopt(stock) {
                    Ok(file) => Poll::Done(json!({ "file": file })),
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
            Self::Bake => document
                .terrain()
                .is_some_and(|terrain| terrain.fields.iter().all(|f| !f.baked().is_empty())),
            Self::Water => document
                .terrain()
                .is_some_and(|terrain| terrain.water().is_some()),
        }
    }
}

fn node_edit(rest: &[&str]) -> Result<Edit, String> {
    let what = *rest
        .first()
        .ok_or("node needs add, rm, connect, disconnect, bypass, place, name or output")?;
    let field = (*rest.get(1).ok_or("node needs a field name")?).to_owned();
    let named = |at: usize, what: &str| -> Result<String, String> {
        rest.get(at)
            .map(|word| (*word).to_owned())
            .ok_or_else(|| format!("node {what} needs a node"))
    };
    match what {
        "add" => Ok(Edit::AddNode {
            field,
            op: parse_op(&owned(&rest[2..]))?,
            position: None,
        }),
        "rm" => Ok(Edit::RemoveNode {
            field,
            node: named(2, "rm")?,
        }),
        "connect" => Ok(Edit::Connect {
            field,
            from: named(2, "connect")?,
            to: named(3, "connect")?,
            pin: number(rest.get(4).ok_or("node connect needs a pin")?)?,
        }),
        "disconnect" => Ok(Edit::Disconnect {
            field,
            node: named(2, "disconnect")?,
            pin: number(rest.get(3).ok_or("node disconnect needs a pin")?)?,
        }),
        "bypass" => Ok(Edit::Bypass {
            field,
            node: named(2, "bypass")?,
            bypassed: match rest.get(3) {
                None => None,
                Some(&"on") => Some(true),
                Some(&"off") => Some(false),
                Some(word) => return Err(format!("a bypass is on or off, not `{word}`")),
            },
        }),
        "place" => Ok(Edit::PlaceNode {
            field,
            node: named(2, "place")?,
            position: [
                number(rest.get(3).ok_or("node place needs an x")?)?,
                number(rest.get(4).ok_or("node place needs a y")?)?,
            ],
        }),
        "name" => {
            let name = named(3, "name")?;
            Ok(Edit::RenameNode {
                field,
                node: named(2, "name")?,
                name: (name != "none").then_some(name),
            })
        }
        "output" => {
            let node = named(2, "output")?;
            Ok(Edit::SetOutput {
                field,
                node: (node != "none").then_some(node),
            })
        }
        other => Err(format!("no node edit called `{other}`")),
    }
}

fn owned(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| (*word).to_owned()).collect()
}

fn number<T: std::str::FromStr>(word: &str) -> Result<T, String> {
    word.parse().map_err(|_| format!("not a number: {word}"))
}

fn point(word: &str) -> Result<Vec2, String> {
    let (x, y) = word
        .split_once(',')
        .ok_or_else(|| format!("a point is written x,y, not `{word}`"))?;
    Ok(Vec2::new(number(x)?, number(y)?))
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
            ("field rename base continent", "field"),
            ("field rm base", "field"),
            ("node add height noise fbm 0.01", "node"),
            ("node add height constant 0.25", "node"),
            ("node add height slope 4", "node"),
            ("node rm height n2", "node"),
            ("node connect height n2 n0 0", "node"),
            ("node disconnect height n0 0", "node"),
            ("node bypass height n1", "node"),
            ("node bypass height n1 off", "node"),
            ("node place height n1 20 40", "node"),
            ("node name height n1 ridge", "node"),
            ("node output height n1", "node"),
            ("set height.1.amplitude 0.5", "set"),
            ("set height.1.blend mul", "set"),
            ("set height.1.mask field moisture 0.4 0.6 0 1", "set"),
            ("set height.1.op.scale 0.004", "set"),
            ("set height.shift 2", "set"),
            ("node add height paint", "node"),
            ("brush radius 24", "brush"),
            (
                "brush mode smooth radius 8 falloff 0.2 strength 0.5 value 0.3",
                "brush",
            ),
            ("stroke 10,20", "stroke"),
            ("stroke 10,20 30,40 50,60", "stroke"),
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
        assert!(Command::parse("node").is_err());
        assert!(Command::parse("node add height").is_err());
        assert!(Command::parse("layer sideways height 1").is_err());
        assert!(Command::parse("node bypass height n1 maybe").is_err());
        assert!(Command::parse("node rm height").is_err());
        assert!(Command::parse("set").is_err());
        assert!(Command::parse("brush").is_err());
        assert!(Command::parse("brush radius").is_err());
        assert!(Command::parse("brush sideways 2").is_err());
        assert!(Command::parse("brush mode sideways").is_err());
        assert!(Command::parse("brush radius wide").is_err());
        assert!(Command::parse("stroke").is_err());
        assert!(Command::parse("stroke 10").is_err());
        assert!(Command::parse("stroke 10,20 sideways").is_err());
    }

    // Points are written `x,y` so a stroke's arguments cannot be miscounted into pairs:
    // a flat list with a number dropped would still parse and would paint a different
    // line. Also pins that a coordinate keeps its fraction.
    #[test]
    fn a_stroke_reads_its_points_as_cells_rather_than_as_a_flat_list_of_numbers() {
        let Ok(Command::Stroke { points, .. }) = Command::parse("stroke 10,20 30.5,40") else {
            panic!("a stroke did not parse to a stroke");
        };
        assert_eq!(points, vec![Vec2::new(10.0, 20.0), Vec2::new(30.5, 40.0)]);
    }
}
