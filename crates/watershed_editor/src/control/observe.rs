//! What a scenario can ask the editor about itself.
//!
//! Adding a topic is a Rust change here; adding a *scenario* is a data file. That
//! asymmetry is the point — it is what keeps a scenario per feature cheap enough to
//! bother with.

use bevy::prelude::*;
use serde_json::{Value, json};
use watershed::WaterState;

use super::log::LogBuffer;
use crate::document::Document;
use crate::view::{CHANNEL_THRESHOLD, EditorCamera, ViewRange, cells_across, view_centre_cell};

pub(super) enum Topic {
    Document,
    Field,
    Water,
    View,
    Log,
}

impl Topic {
    pub(super) fn parse(word: &str) -> Result<Self, String> {
        match word {
            "document" => Ok(Self::Document),
            "field" => Ok(Self::Field),
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
        Topic::Water => water(world),
        Topic::View => view(world),
        Topic::Log => log(world),
    }
}

fn document(world: &World) -> Value {
    let document = world.resource::<Document>();
    json!({
        "busy": document.is_busy(),
        "job": document.job().map(|kind| kind.name()),
        "error": document.error(),
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

    let mut query = world.query_filtered::<(&Transform, &Projection), With<EditorCamera>>();
    let Ok((transform, projection)) = query.single(world) else {
        return json!({ "available": false });
    };

    let centre = view_centre_cell(transform, size);
    json!({
        "available": true,
        "centre": [centre.x, centre.y],
        "cells_across": cells_across(projection),
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
