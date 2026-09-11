//! What a caller can ask the running editor about itself.
//!
//! Every answer is what the editor *acted on*, not something the caller could work out
//! for itself: the fitted colour range, the rectangle a live re-bake covers, the layer
//! a stroke would land in. A second derivation on the caller's side would part company
//! with the editor the moment the camera moved.
//!
//! Adding a topic is a change here; asking a new question of an existing one is not.
//! That asymmetry is the point — it is what keeps a scenario per feature cheap enough
//! to bother with.

use crate::terrain::WaterState;
use bevy::prelude::*;
use serde_json::{Value, json};

use super::log::LogBuffer;
use crate::brush::{BrushSettings, target_of};
use crate::document::{Baked, Document};
use crate::edit::{brush_summary, op_name, op_summary};
use crate::gpu::{STOCK, ShaderLibrary};
use crate::view::{
    CHANNEL_THRESHOLD, EditorCamera, FreeView, ViewRange, VisibleCells, cells_across,
    view_centre_cell,
};

/// What an `observe` command is asking about.
pub(super) enum Topic {
    /// The document's state: what is running, what failed, how much is baked.
    Document,
    /// A summary of the active field's baked values.
    Field,
    /// Every field's whole stack, not just the active one's — an edit names a field,
    /// so a caller has to be able to see the stack it is about to address without
    /// switching the view to it first.
    Nodes,
    /// The brush's settings and where a stroke would land.
    Brush,
    /// The solved water, counted.
    Water,
    /// Where the camera is and what the ramp is fitted to.
    View,
    /// Warnings and errors since the last time this was asked. Draining.
    Log,
    /// The shaders the document carries, what each declares, and why one did not
    /// parse.
    Shaders,
}

impl Topic {
    /// The topic of that exact name, or a message naming what was asked for.
    pub(super) fn parse(word: &str) -> Result<Self, String> {
        match word {
            "document" => Ok(Self::Document),
            "field" => Ok(Self::Field),
            "nodes" => Ok(Self::Nodes),
            "brush" => Ok(Self::Brush),
            "water" => Ok(Self::Water),
            "view" => Ok(Self::View),
            "log" => Ok(Self::Log),
            "shaders" => Ok(Self::Shaders),
            other => Err(format!("nothing to observe called {other}")),
        }
    }
}

/// Answers the topic. Every answer carries `available`, or is a shape whose fields
/// are always there — a topic asked of a document that has none says so rather than
/// failing.
pub(super) fn run(world: &mut World, topic: &Topic) -> Value {
    match topic {
        Topic::Document => document(world),
        Topic::Field => field(world),
        Topic::Nodes => nodes(world),
        Topic::Brush => brush(world),
        Topic::Water => water(world),
        Topic::View => view(world),
        Topic::Log => log(world),
        Topic::Shaders => shaders(world),
    }
}

fn document(world: &World) -> Value {
    let document = world.resource::<Document>();
    json!({
        "busy": document.is_busy(),
        "settled": document.is_settled(),
        "job": document.job().map(|kind| kind.name()),
        "error": document.error(),
        "baked": document.baked().name(),
        "baked_rect": match document.baked() {
            Baked::Rect(rect) => json!([rect.min.x, rect.min.y, rect.max.x, rect.max.y]),
            _ => Value::Null,
        },
        "size": [document.size.x, document.size.y],
        "undo": document.history().undo,
        "redo": document.history().redo,
        "seed": document.seed,
        "preset": document.preset.name(),
        "path": document.path.as_ref().map(|path| path.display().to_string()),
        "active": document.active(),
        "fields": document.field_names(),
        "water": document
            .terrain()
            .is_some_and(|terrain| terrain.water().is_some()),
    })
}

fn field(world: &World) -> Value {
    let document = world.resource::<Document>();
    let Some(terrain) = document.terrain() else {
        return json!({ "available": false });
    };
    let Some(field) = terrain.field(document.active()) else {
        return json!({ "available": false });
    };

    let baked = field.baked();
    if baked.is_empty() {
        return json!({ "available": false, "reason": "not baked" });
    }

    let mut values: Vec<f32> = baked
        .data()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    values.sort_by(f32::total_cmp);
    let at = |q: f32| values[((values.len() - 1) as f32 * q) as usize];

    json!({
        "available": true,
        "name": field.id.to_string(),
        "shift": field.shift,
        "resolution": [baked.width(), baked.height()],
        "cells": values.len(),
        "min": at(0.0),
        "p10": at(0.10),
        "median": at(0.50),
        "p90": at(0.90),
        "max": at(1.0),
    })
}

fn nodes(world: &World) -> Value {
    let document = world.resource::<Document>();
    let Some(terrain) = document.terrain() else {
        return json!({ "available": false });
    };

    let fields: Vec<Value> = terrain
        .fields
        .iter()
        .map(|field| {
            let nodes: Vec<Value> = field
                .graph
                .nodes
                .iter()
                .map(|node| {
                    json!({
                        "node": node.id.to_string(),
                        "name": node.name,
                        "op": op_name(&node.op),
                        "summary": op_summary(&node.op),
                        "bypassed": node.bypassed,
                        "inputs": node
                            .inputs
                            .iter()
                            .map(|pin| match pin {
                                Some(source) => json!(source.to_string()),
                                None => Value::Null,
                            })
                            .collect::<Vec<Value>>(),
                        "position": node.position,
                    })
                })
                .collect();
            json!({
                "field": field.id.to_string(),
                "shift": field.shift,
                "range": [field.range.0, field.range.1],
                "hillshade": field.hillshade,
                "light_azimuth": field.light_azimuth,
                "categorical": field.is_categorical(),
                "output": field.graph.output.map(|id| id.to_string()),
                "nodes": nodes,
            })
        })
        .collect();

    json!({ "available": true, "active": document.active(), "fields": fields })
}

fn brush(world: &World) -> Value {
    let settings = world.resource::<BrushSettings>();
    let document = world.resource::<Document>();
    let target = target_of(document);
    let mut value = brush_summary(&settings.0);
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "field".to_owned(),
            match &target {
                Some((field, _)) => json!(field),
                None => Value::Null,
            },
        );
        object.insert(
            "node".to_owned(),
            match &target {
                Some((_, id)) => json!(id.to_string()),
                None => Value::Null,
            },
        );
    }
    value
}

fn water(world: &World) -> Value {
    let document = world.resource::<Document>();
    let Some(state) = document.terrain().and_then(|terrain| terrain.water()) else {
        return json!({ "available": false });
    };

    let size = state.size();
    let cells = (size.x as u64) * (size.y as u64);
    let (water_cells, channel_cells, sinks) = counts(state);

    json!({
        "available": true,
        "size": [size.x, size.y],
        "cells": cells,
        "lakes": state.lakes(),
        "water_cells": water_cells,
        "water_fraction": water_cells as f64 / cells.max(1) as f64,
        "channel_threshold": CHANNEL_THRESHOLD,
        "channel_cells": channel_cells,
        "channel_fraction": channel_cells as f64 / cells.max(1) as f64,
        "sinks": sinks,
    })
}

fn counts(state: &WaterState) -> (u64, u64, u64) {
    let size = state.size();
    let mut water = 0;
    let mut channel = 0;
    let mut sinks = 0;
    for y in 0..size.y {
        for x in 0..size.x {
            if state.is_water(x, y) {
                water += 1;
            }
            if state.channel(x, y, CHANNEL_THRESHOLD) {
                channel += 1;
            }
            if state.downstream(x, y).is_none() {
                sinks += 1;
            }
        }
    }
    (water, channel, sinks)
}

fn view(world: &mut World) -> Value {
    let size = world.resource::<Document>().size;
    let range = *world.resource::<ViewRange>();
    let visible = world.resource::<VisibleCells>().0;
    let free = *world.resource::<FreeView>();

    let mut query = world.query_filtered::<(&Transform, &Projection), With<EditorCamera>>();
    let Ok((transform, projection)) = query.single(world) else {
        return json!({ "available": false });
    };

    let centre = view_centre_cell(transform, size);
    json!({
        "available": true,
        "centre": [centre.x, centre.y],
        "cells_across": cells_across(projection),
        "free_size": [free.size.x, free.size.y],
        "free_centre": [free.centre.x, free.centre.y],
        "cells": [visible.min.x, visible.min.y, visible.max.x, visible.max.y],
        "range": [range.low, range.high],
        "diverging": range.diverging,
    })
}

fn log(world: &World) -> Value {
    match world.get_resource::<LogBuffer>() {
        Some(buffer) => buffer.since_last_read(),
        None => json!({ "available": false }),
    }
}

/// Every shader the document's directory holds, in name order: what it declares, and
/// why it did not parse.
///
/// The parameters are named rather than counted, because a caller setting one has to
/// know what it is called.
fn shaders(world: &mut World) -> Value {
    let library = world.resource::<ShaderLibrary>();
    let files: Vec<Value> = library
        .files()
        .map(|file| {
            let entry = library.entry(file).expect("a listed file has an entry");
            json!({
                "file": file,
                "params": entry
                    .layout
                    .fields
                    .iter()
                    .map(|param| json!({
                        "name": param.name,
                        "type": param.ty.as_str(),
                        "group": param.group,
                    }))
                    .collect::<Vec<_>>(),
                "error": entry.error,
            })
        })
        .collect();
    json!({
        "root": library.root().to_string_lossy(),
        "shaders": files,
        "stock": STOCK.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
    })
}
