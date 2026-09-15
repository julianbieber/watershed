//! What a caller can ask the running editor about itself.
//!
//! Every answer is what the editor *acted on*, not something the caller could work out
//! for itself: the fitted colour range, the rectangle a live re-bake covers. A second
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
use crate::document::{Baked, Document};
use crate::edit::{op_name, op_summary};
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
    /// Every field's whole graph, not just the active one's — an edit names a field,
    /// so a caller has to be able to see the graph it is about to address without
    /// switching the view to it first.
    Nodes,
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
            "nodes" => Ok(Self::Nodes),
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
        Topic::Nodes => nodes(world),
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

    let reads = crate::edit::reads_of(field);
    let read_by = crate::edit::readers_of(terrain, document.active());

    let baked = field.baked();
    if baked.is_empty() {
        return json!({
            "available": false,
            "reason": "not baked",
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
        "reads": reads,
        "read_by": read_by,
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
                "contours": field.contours,
                "contour_interval": field.contour_interval,
                "output": field.graph.output.map(|id| id.to_string()),
                "nodes": nodes,
            })
        })
        .collect();

    json!({ "available": true, "active": document.active(), "fields": fields })
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
                    .map(|(_, fault)| fault),
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
                "inputs": entry
                    .inputs
                    .iter()
                    .map(|input| json!({
                        "name": input.name,
                        "label": input.label,
                        "binding": input.binding,
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
                "reach": entry.reach,
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
    use crate::terrain::graph::NodeOp;
    use crate::terrain::{Field, TerrainSpec};
    use watershed::FieldId;

    // Acceptance criterion five, which is the socket's half of the whole task: a caller
    // driving the editor over the control port can read the dependency both ways round
    // without walking every field's graph itself.
    #[test]
    fn observing_a_field_reports_both_ends_of_the_relation() {
        let mut terrain = TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("base").with_op(NodeOp::held(0.25)))
            .with_field(Field::new("height").with_op(NodeOp::FieldRef(FieldId::from("base"))));
        terrain.bake_in_place().expect("a bake");

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

    // Acceptance criterion seven: the whole document's shape over the socket, in the
    // order the bake visits the fields in, so a caller sees the same picture the
    // overview draws without walking every graph itself.
    #[test]
    fn observing_the_fields_reports_them_in_bake_order_with_what_each_reads() {
        let terrain = TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("height").with_op(NodeOp::FieldRef(FieldId::from("base"))))
            .with_field(Field::new("base").with_op(NodeOp::held(0.25)));

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
            .with_field(Field::new("here").with_op(NodeOp::FieldRef(FieldId::from("there"))))
            .with_field(Field::new("there").with_op(NodeOp::FieldRef(FieldId::from("here"))));

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
        let mut reads_base = crate::terrain::shader::ShaderLayer::new("reader.wgsl");
        reads_base.layers = vec![FieldId::from("base")];
        let mut reads_nowhere = crate::terrain::shader::ShaderLayer::new("lost.wgsl");
        reads_nowhere.layers = vec![FieldId::from("nowhere")];
        let terrain = TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("base").with_op(NodeOp::held(0.25)))
            .with_field(Field::new("height").with_op(NodeOp::Shader(reads_base)))
            .with_field(Field::new("lost").with_op(NodeOp::Shader(reads_nowhere)));

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
}
