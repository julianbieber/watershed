//! What a caller can ask the running editor about itself.
//!
//! Every answer is what the editor *acted on*, not something the caller could work out
//! for itself: the fitted colour range, the fault that left a field unbaked. A second
//! derivation on the caller's side would part company with the editor the moment the
//! camera moved.
//!
//! Adding a topic is a change here; asking a new question of an existing one is not.
//! That asymmetry is the point — it is what keeps a scenario per feature cheap enough
//! to bother with.

use crate::terrain::WaterState;
use bevy::prelude::*;
use serde_json::{Value, json};

use super::log::LogBuffer;
use crate::document::Document;
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
    /// The solved water, counted, and the fields the water spec is over.
    Water,
    /// Where the camera is and what the ramp is fitted to.
    View,
    /// Warnings and errors since the last time this was asked. Draining.
    Log,
    /// The shaders the document carries, what each declares, and why one did not
    /// parse.
    Shaders,
    /// Every field of the document in bake order, what each one reads, and the cycle
    /// that stopped the order from being computed.
    Fields,
}

impl Topic {
    /// The topic of that exact name, or a message naming what was asked for.
    pub(super) fn parse(word: &str) -> Result<Self, String> {
        match word {
            "document" => Ok(Self::Document),
            "field" => Ok(Self::Field),
            "water" => Ok(Self::Water),
            "view" => Ok(Self::View),
            "log" => Ok(Self::Log),
            "shaders" => Ok(Self::Shaders),
            "fields" => Ok(Self::Fields),
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
        Topic::Water => water(world),
        Topic::View => view(world),
        Topic::Log => log(world),
        Topic::Shaders => shaders(world),
        Topic::Fields => fields(world),
    }
}

fn document(world: &World) -> Value {
    let document = world.resource::<Document>();
    json!({
        "busy": document.is_busy(),
        "settled": document.is_settled(),
        "job": document.job().map(|kind| kind.name()),
        "error": document.error(),
        "bake_failed": document.bake_failed(),
        "baked": document.baked().name(),
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

    let reads = crate::edit::reads_of(field);
    let read_by = crate::edit::readers_of(terrain, document.active());

    let file = field.file();
    let params = &field.shader.params;

    let baked = field.baked();
    if baked.is_empty() {
        return json!({
            "available": false,
            "reason": "not baked",
            "file": file,
            "params": params,
            "reads": reads,
            "read_by": read_by,
        });
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
        "file": file,
        "params": params,
        "reads": reads,
        "read_by": read_by,
    })
}

fn fields(world: &World) -> Value {
    let document = world.resource::<Document>();
    let Some(terrain) = document.terrain() else {
        return json!({ "available": false });
    };
    let (names, order, cycle) = match terrain.bake_order() {
        Ok(order) => (
            order
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<String>>(),
            "bake",
            Value::Null,
        ),
        Err(error) => (
            terrain
                .fields
                .iter()
                .map(|field| field.id.to_string())
                .collect(),
            "declaration",
            json!(error.to_string()),
        ),
    };
    let faults = terrain.field_faults();
    let library = world.get_resource::<ShaderLibrary>();
    let fields: Vec<Value> = names
        .iter()
        .filter_map(|name| terrain.field(name))
        .map(|field| {
            json!({
                "name": field.id.to_string(),
                "role": field.role.as_str(),
                "shift": field.shift,
                "reads": crate::edit::reads_of(field),
                "fault": faults
                    .iter()
                    .find(|(id, _)| *id == field.id)
                    .map(|(_, fault)| fault.clone())
                    .or_else(|| {
                        library
                            .and_then(|library| library.entry(&field.file()))
                            .and_then(|entry| entry.error.clone())
                    }),
            })
        })
        .collect();

    json!({
        "available": true,
        "active": document.active(),
        "order": order,
        "cycle": cycle,
        "fields": fields,
    })
}

fn water(world: &World) -> Value {
    let document = world.resource::<Document>();
    let Some(terrain) = document.terrain() else {
        return json!({ "available": false });
    };
    let height = terrain
        .water_spec
        .as_ref()
        .map(|spec| spec.height.to_string());
    let moisture = terrain
        .water_spec
        .as_ref()
        .and_then(|spec| spec.moisture.as_ref())
        .map(|id| id.to_string());
    let Some(state) = terrain.water() else {
        return json!({ "available": false, "height": height, "moisture": moisture });
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
        "height": height,
        "moisture": moisture,
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
                "layers": entry
                    .layers
                    .iter()
                    .map(|read| json!({
                        "name": read.name,
                        "layer": read.layer,
                        "binding": read.binding,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::{Field, TerrainSpec};

    // Acceptance criterion five, which is the socket's half of the whole task: a caller
    // driving the editor over the control port can read the dependency both ways round
    // without opening every field's file itself.
    #[test]
    fn observing_a_field_reports_both_ends_of_the_relation() {
        let terrain = TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("base").held(0.25))
            .with_field(Field::new("height").reading(&["base"]));

        let mut document = Document::default();
        document.adopt(terrain);
        document.set_active("height").unwrap();
        let mut world = World::new();
        world.insert_resource(document);

        let height = field(&world);
        assert_eq!(height["reads"], json!(["base"]));
        assert_eq!(height["read_by"], json!([]));

        world.resource_mut::<Document>().set_active("base").unwrap();
        let base = field(&world);
        assert_eq!(base["reads"], json!([]));
        assert_eq!(base["read_by"], json!(["height"]));
    }

    // A field is one shader file and the numbers set on it, so the socket has to name
    // both — a caller about to `set <field>.<param>` has to see what is there.
    #[test]
    fn observing_a_field_reports_its_file_and_its_parameters() {
        let terrain = TerrainSpec::new(UVec2::splat(16)).with_field(Field::new("base").held(0.25));

        let mut document = Document::default();
        document.adopt(terrain);
        document.set_active("base").unwrap();
        let mut world = World::new();
        world.insert_resource(document);

        let base = field(&world);
        assert_eq!(base["file"], json!("base.wgsl"));
        assert_eq!(base["params"], json!({ "value": [0.25] }));
    }

    // Acceptance criterion seven: the whole document's shape over the socket, in the
    // order the bake visits the fields in, so a caller sees the same picture the
    // overview draws without opening every file itself.
    #[test]
    fn observing_the_fields_reports_them_in_bake_order_with_what_each_reads() {
        let terrain = TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("height").reading(&["base"]))
            .with_field(Field::new("base").held(0.25));

        let mut document = Document::default();
        document.adopt(terrain);
        let mut world = World::new();
        world.insert_resource(document);

        let answer = fields(&world);
        assert_eq!(answer["order"], json!("bake"));
        assert_eq!(answer["cycle"], Value::Null);
        let listed = answer["fields"].as_array().expect("an array of fields");
        assert_eq!(listed[0]["name"], json!("base"));
        assert_eq!(listed[0]["reads"], json!([]));
        assert_eq!(listed[0]["role"], json!("custom"));
        assert_eq!(listed[0]["shift"], json!(0));
        assert_eq!(listed[1]["name"], json!("height"));
        assert_eq!(listed[1]["reads"], json!(["base"]));
    }

    // Acceptance criterion eight, read from the socket rather than from the status bar:
    // a document whose fields cannot be ordered still reports every one of them, says
    // the order is the declared one, and names the cycle.
    #[test]
    fn observing_the_fields_of_a_cyclic_document_names_the_cycle() {
        let terrain = TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("here").reading(&["there"]))
            .with_field(Field::new("there").reading(&["here"]));

        let mut document = Document::default();
        document.adopt(terrain);
        let mut world = World::new();
        world.insert_resource(document);

        let answer = fields(&world);
        assert_eq!(answer["order"], json!("declaration"));
        let cycle = answer["cycle"].as_str().expect("the cycle, as text");
        assert!(cycle.contains("cycle"), "{cycle:?} does not name the cycle");
        let listed = answer["fields"].as_array().expect("an array of fields");
        assert_eq!(listed[0]["name"], json!("here"));
        assert_eq!(listed[1]["name"], json!("there"));
    }

    // The socket's half of a read by name: a caller sees a file's `@layer` as a read of
    // the field it names, and a name that is no field as the reader's fault, without
    // opening the file.
    #[test]
    fn observing_the_fields_reports_a_layer_read_and_the_fault_of_a_name_that_is_no_field() {
        let terrain = TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("base").held(0.25))
            .with_field(Field::new("height").reading(&["base"]))
            .with_field(Field::new("lost").reading(&["nowhere"]));

        let mut document = Document::default();
        document.adopt(terrain);
        let mut world = World::new();
        world.insert_resource(document);

        let answer = fields(&world);
        let listed = answer["fields"].as_array().expect("an array of fields");
        let named = |name: &str| {
            listed
                .iter()
                .find(|field| field["name"] == json!(name))
                .unwrap_or_else(|| panic!("{name} is not listed"))
        };
        assert_eq!(named("height")["reads"], json!(["base"]));
        assert_eq!(named("height")["fault"], Value::Null);
        let fault = named("lost")["fault"].as_str().expect("the fault, as text");
        assert!(fault.contains("nowhere"), "{fault:?}");
    }

    // A field whose own file does not parse has no fault in the document's reads, so
    // the socket has to fall back to the file's error — otherwise a caller sees a field
    // that silently never bakes.
    #[test]
    fn observing_the_fields_reports_the_error_of_a_file_that_did_not_parse() {
        let terrain = TerrainSpec::new(UVec2::splat(16)).with_field(Field::new("height"));

        let mut document = Document::default();
        document.adopt(terrain);
        let mut world = World::new();
        world.insert_resource(document);
        world.insert_resource(ShaderLibrary::with_fault(
            "height.wgsl",
            "line 3: expected `;`",
        ));

        let answer = fields(&world);
        assert_eq!(answer["fields"][0]["fault"], json!("line 3: expected `;`"));
    }
}
