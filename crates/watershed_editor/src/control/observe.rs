//! What a scenario can ask the editor about itself.
//!
//! Adding a topic is a Rust change here; adding a *scenario* is a data file. That
//! asymmetry is the point — it is what keeps a scenario per feature cheap enough to
//! bother with.

use bevy::prelude::*;
use serde_json::{Value, json};
use watershed::WaterState;

use super::log::LogBuffer;
use crate::document::{Baked, Document};
use crate::edit::{blend_name, mask_summary, op_name, op_summary};
use crate::view::{
    CHANNEL_THRESHOLD, EditorCamera, FreeView, ViewRange, VisibleCells, cells_across,
    view_centre_cell,
};

pub(super) enum Topic {
    Document,
    Field,
    Layers,
    Water,
    View,
    Log,
}

impl Topic {
    pub(super) fn parse(word: &str) -> Result<Self, String> {
        match word {
            "document" => Ok(Self::Document),
            "field" => Ok(Self::Field),
            "layers" => Ok(Self::Layers),
            "water" => Ok(Self::Water),
            "view" => Ok(Self::View),
            "log" => Ok(Self::Log),
            other => Err(format!("nothing to observe called {other}")),
        }
    }
}

pub(super) fn run(world: &mut World, topic: &Topic) -> Value {
    match topic {
        Topic::Document => document(world),
        Topic::Field => field(world),
        Topic::Layers => layers(world),
        Topic::Water => water(world),
        Topic::View => view(world),
        Topic::Log => log(world),
    }
}

fn document(world: &World) -> Value {
    let document = world.resource::<Document>();
    json!({
        "busy": document.is_busy(),
        "settled": document.is_settled(),
        "job": document.job().map(|kind| kind.name()),
        "error": document.error(),
        // How much of the bake matches the layers, which after an edit is the visible
        // rectangle rather than the document — and is what a solve is refused against.
        "baked": document.baked().name(),
        "baked_rect": match document.baked() {
            Baked::Rect(rect) => json!([rect.min.x, rect.min.y, rect.max.x, rect.max.y]),
            _ => Value::Null,
        },
        "size": [document.size.x, document.size.y],
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

/// TODO(jb-doc): why the summary is quantiles rather than a mean — that a field is judged
/// by whether it *varies*, and a mean says nothing about that.
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

/// Every field's whole stack, not just the active one's — an edit names a field, so a
/// scenario has to be able to see the stack it is about to address without switching the
/// view to it first.
///
/// TODO(jb-doc): why the summary is the same string the panel puts on a layer's header,
/// and what a second phrasing here would let drift.
fn layers(world: &World) -> Value {
    let document = world.resource::<Document>();
    let Some(terrain) = document.terrain() else {
        return json!({ "available": false });
    };

    let fields: Vec<Value> = terrain
        .fields
        .iter()
        .map(|field| {
            let layers: Vec<Value> = field
                .layers
                .iter()
                .enumerate()
                .map(|(index, layer)| {
                    json!({
                        "index": index,
                        "op": op_name(&layer.op),
                        "summary": op_summary(&layer.op),
                        "blend": blend_name(layer.blend),
                        "amplitude": layer.amplitude,
                        "mask": mask_summary(&layer.mask),
                        "enabled": layer.enabled,
                    })
                })
                .collect();
            json!({
                "field": field.id.to_string(),
                "shift": field.shift,
                "range": [field.range.0, field.range.1],
                // Whether this field's bake is read at its nearest texel rather than
                // between them, which follows an op parameter rather than being set.
                "categorical": field.is_categorical(),
                "layers": layers,
            })
        })
        .collect();

    json!({ "available": true, "active": document.active(), "fields": fields })
}

/// The channel threshold is the view's, not a second one: a scenario asserting on channels
/// has to be asking about the ones it can see.
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

/// TODO(jb-doc): why the fitted range is reported here rather than derived by the caller —
/// that it is what the screen is actually showing, and a second derivation would part
/// company with it the moment the camera moved.
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
        // How much of the window the panels have left the world, which is what a fit aims
        // at. Reported rather than derived because a panel's width is egui's to decide.
        "free_size": [free.size.x, free.size.y],
        "free_centre": [free.centre.x, free.centre.y],
        // The rectangle a live re-bake covers, reported here rather than derived by the
        // caller for the reason the fitted range is: it is what the editor acted on.
        "cells": [visible.min.x, visible.min.y, visible.max.x, visible.max.y],
        "range": [range.low, range.high],
        "diverging": range.diverging,
    })
}

fn log(world: &World) -> Value {
    match world.get_resource::<LogBuffer>() {
        Some(buffer) => buffer.drain(),
        None => json!({ "available": false }),
    }
}
