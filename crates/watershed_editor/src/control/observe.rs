//! What a caller can ask the running editor about itself.
//!
//! Every answer is what the editor *acted on*, not something the caller could work out
//! for itself: the fitted colour range, the fault that left a layer unbaked. A second
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
use crate::project::Project;
use crate::view::{
    CHANNEL_THRESHOLD, EditorCamera, FreeView, ViewRange, VisibleCells, cells_across,
    view_centre_cell,
};

/// What an `observe` command is asking about.
pub(super) enum Topic {
    /// The document's state: what is running, what failed, how much is baked.
    Document,
    /// A summary of the active layer's baked values.
    Layer,
    /// The solved water, counted, and the layers the water spec is over.
    Water,
    /// Where the camera is and what the ramp is fitted to.
    View,
    /// Warnings and errors since the last time this was asked. Draining.
    Log,
    /// The shaders the document carries, what each declares, and why one did not
    /// parse.
    Shaders,
    /// Every layer of the document in bake order, what each one reads, and the cycle
    /// that stopped the order from being computed.
    Layers,
}

impl Topic {
    /// The topic of that exact name, or a message naming what was asked for.
    pub(super) fn parse(word: &str) -> Result<Self, String> {
        match word {
            "document" => Ok(Self::Document),
            "layer" => Ok(Self::Layer),
            "water" => Ok(Self::Water),
            "view" => Ok(Self::View),
            "log" => Ok(Self::Log),
            "shaders" => Ok(Self::Shaders),
            "layers" => Ok(Self::Layers),
            other => Err(format!("nothing to observe called {other}")),
        }
    }
}

/// Answers the topic. Every answer carries `available`, or is a shape whose layers
/// are always there — a topic asked of a document that has none says so rather than
/// failing.
pub(super) fn run(world: &mut World, topic: &Topic) -> Value {
    match topic {
        Topic::Document => document(world),
        Topic::Layer => layer(world),
        Topic::Water => water(world),
        Topic::View => view(world),
        Topic::Log => log(world),
        Topic::Shaders => shaders(world),
        Topic::Layers => layers(world),
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
        "project": world.resource::<Project>().dir().display().to_string(),
        "active": document.active(),
        "layers": document.layer_names(),
        "water": document
            .terrain()
            .is_some_and(|terrain| terrain.water().is_some()),
    })
}

fn layer(world: &World) -> Value {
    let document = world.resource::<Document>();
    let Some(terrain) = document.terrain() else {
        return json!({ "available": false });
    };
    let Some(layer) = terrain.layer(document.active()) else {
        return json!({ "available": false });
    };

    let reads = crate::edit::reads_of(layer);
    let read_by = crate::edit::readers_of(terrain, document.active());

    let file = layer.file();
    let path = document.shader_root().join(&file);
    let params = &layer.shader.params;

    let (low, high) = layer.bounds();

    let baked = layer.baked();
    if baked.is_empty() {
        return json!({
            "available": false,
            "reason": "not baked",
            "role": layer.role.as_str(),
            "shift": layer.shift,
            "range": [low, high],
            "categorical": layer.categorical,
            "file": file,
            "path": path.display().to_string(),
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
        "name": layer.id.to_string(),
        "role": layer.role.as_str(),
        "shift": layer.shift,
        "range": [low, high],
        "categorical": layer.categorical,
        "resolution": [baked.width(), baked.height()],
        "cells": values.len(),
        "min": at(0.0),
        "p10": at(0.10),
        "median": at(0.50),
        "p90": at(0.90),
        "max": at(1.0),
        "file": file,
        "path": path.display().to_string(),
        "params": params,
        "reads": reads,
        "read_by": read_by,
    })
}

fn layers(world: &World) -> Value {
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
                .layers
                .iter()
                .map(|layer| layer.id.to_string())
                .collect(),
            "declaration",
            json!(error.to_string()),
        ),
    };
    let faults = terrain.layer_faults();
    let library = world.get_resource::<ShaderLibrary>();
    let layers: Vec<Value> = names
        .iter()
        .filter_map(|name| terrain.layer(name))
        .map(|layer| {
            json!({
                "name": layer.id.to_string(),
                "role": layer.role.as_str(),
                "shift": layer.shift,
                "categorical": layer.categorical,
                "reads": crate::edit::reads_of(layer),
                "fault": faults
                    .iter()
                    .find(|(id, _)| *id == layer.id)
                    .map(|(_, fault)| fault.clone())
                    .or_else(|| {
                        library
                            .and_then(|library| library.entry(&layer.file()))
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
        "layers": layers,
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
                "imports": entry
                    .imports
                    .iter()
                    .map(|import| json!({
                        "path": import.path,
                        "resolved": import.resolved,
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
    use crate::terrain::{Layer, TerrainSpec};

    // Acceptance criterion five, which is the socket's half of the whole task: a caller
    // driving the editor over the control port can read the dependency both ways round
    // without opening every layer's file itself.
    #[test]
    fn observing_a_layer_reports_both_ends_of_the_relation() {
        let terrain = TerrainSpec::new(UVec2::splat(16))
            .with_layer(Layer::new("base").held(0.25))
            .with_layer(Layer::new("height").reading(&["base"]));

        let mut document = Document::default();
        document.adopt(terrain);
        document.set_active("height").unwrap();
        let mut world = World::new();
        world.insert_resource(document);

        let height = layer(&world);
        assert_eq!(height["reads"], json!(["base"]));
        assert_eq!(height["read_by"], json!([]));

        world.resource_mut::<Document>().set_active("base").unwrap();
        let base = layer(&world);
        assert_eq!(base["reads"], json!([]));
        assert_eq!(base["read_by"], json!(["height"]));
    }

    // A layer is one shader file and the numbers set on it, so the socket has to name
    // both — a caller about to `set <layer>.<param>` has to see what is there.
    #[test]
    fn observing_a_layer_reports_its_file_and_its_parameters() {
        let terrain = TerrainSpec::new(UVec2::splat(16)).with_layer(Layer::new("base").held(0.25));

        let mut document = Document::default();
        document.adopt(terrain);
        document.set_active("base").unwrap();
        let mut world = World::new();
        world.insert_resource(document);

        let base = layer(&world);
        assert_eq!(base["file"], json!("base.wesl"));
        assert_eq!(base["params"], json!({ "value": [0.25] }));
    }

    // The panel's Open button and the socket have to name the same file, and only an
    // absolute path means anything to a program outside the process.
    #[test]
    fn observing_a_layer_reports_an_absolute_path_to_its_file() {
        let terrain =
            TerrainSpec::new(UVec2::splat(16)).with_layer(Layer::new("height").held(0.25));

        let mut document = Document::default();
        document.adopt(terrain);
        document.set_active("height").unwrap();
        let mut world = World::new();
        world.insert_resource(document);

        let height = layer(&world);
        let path = std::path::PathBuf::from(height["path"].as_str().unwrap());
        assert!(path.is_absolute(), "{}", path.display());
        assert!(path.ends_with("height.wesl"), "{}", path.display());
    }

    // Editing a header line and reading the result back is how every property this task
    // moved into the file is checked, and an unbaked layer is exactly the state a file
    // that has just been saved is in — so the four have to be on both replies.
    #[test]
    fn observing_a_layer_reports_the_four_its_file_declares_baked_or_not() {
        let mut moisture = Layer::new("moisture").with_shift(2).with_range((0.0, 2.0));
        moisture.categorical = true;
        let terrain = TerrainSpec::new(UVec2::splat(16)).with_layer(moisture);

        let mut document = Document::default();
        document.adopt(terrain);
        document.set_active("moisture").unwrap();
        let mut world = World::new();
        world.insert_resource(document);

        let unbaked = layer(&world);
        assert_eq!(unbaked["reason"], json!("not baked"));
        assert_eq!(unbaked["shift"], json!(2));
        assert_eq!(unbaked["range"], json!([0.0, 2.0]));
        assert_eq!(unbaked["categorical"], json!(true));

        world
            .resource_mut::<Document>()
            .terrain_mut()
            .expect("a document")
            .bake_in_place()
            .unwrap();

        let baked = layer(&world);
        assert_eq!(baked["available"], json!(true));
        assert_eq!(baked["role"], json!("custom"));
        assert_eq!(baked["shift"], json!(2));
        assert_eq!(baked["range"], json!([0.0, 2.0]));
        assert_eq!(baked["categorical"], json!(true));
        assert_eq!(baked["resolution"], json!([4, 4]));
    }

    // Acceptance criterion seven: the whole document's shape over the socket, in the
    // order the bake visits the layers in, so a caller sees the same picture the
    // overview draws without opening every file itself.
    #[test]
    fn observing_the_layers_reports_them_in_bake_order_with_what_each_reads() {
        let terrain = TerrainSpec::new(UVec2::splat(16))
            .with_layer(Layer::new("height").reading(&["base"]))
            .with_layer(Layer::new("base").held(0.25));

        let mut document = Document::default();
        document.adopt(terrain);
        let mut world = World::new();
        world.insert_resource(document);

        let answer = layers(&world);
        assert_eq!(answer["order"], json!("bake"));
        assert_eq!(answer["cycle"], Value::Null);
        let listed = answer["layers"].as_array().expect("an array of layers");
        assert_eq!(listed[0]["name"], json!("base"));
        assert_eq!(listed[0]["reads"], json!([]));
        assert_eq!(listed[0]["role"], json!("custom"));
        assert_eq!(listed[0]["shift"], json!(0));
        assert_eq!(listed[0]["categorical"], json!(false));
        assert_eq!(listed[1]["name"], json!("height"));
        assert_eq!(listed[1]["reads"], json!(["base"]));
    }

    // Acceptance criterion eight, read from the socket rather than from the status bar:
    // a document whose layers cannot be ordered still reports every one of them, says
    // the order is the declared one, and names the cycle.
    #[test]
    fn observing_the_layers_of_a_cyclic_document_names_the_cycle() {
        let terrain = TerrainSpec::new(UVec2::splat(16))
            .with_layer(Layer::new("here").reading(&["there"]))
            .with_layer(Layer::new("there").reading(&["here"]));

        let mut document = Document::default();
        document.adopt(terrain);
        let mut world = World::new();
        world.insert_resource(document);

        let answer = layers(&world);
        assert_eq!(answer["order"], json!("declaration"));
        let cycle = answer["cycle"].as_str().expect("the cycle, as text");
        assert!(cycle.contains("cycle"), "{cycle:?} does not name the cycle");
        let listed = answer["layers"].as_array().expect("an array of layers");
        assert_eq!(listed[0]["name"], json!("here"));
        assert_eq!(listed[1]["name"], json!("there"));
    }

    // The socket's half of a read by name: a caller sees a file's `@layer` as a read of
    // the layer it names, and a name that is no layer as the reader's fault, without
    // opening the file.
    #[test]
    fn observing_the_layers_reports_a_layer_read_and_the_fault_of_a_name_that_is_no_layer() {
        let terrain = TerrainSpec::new(UVec2::splat(16))
            .with_layer(Layer::new("base").held(0.25))
            .with_layer(Layer::new("height").reading(&["base"]))
            .with_layer(Layer::new("lost").reading(&["nowhere"]));

        let mut document = Document::default();
        document.adopt(terrain);
        let mut world = World::new();
        world.insert_resource(document);

        let answer = layers(&world);
        let listed = answer["layers"].as_array().expect("an array of layers");
        let named = |name: &str| {
            listed
                .iter()
                .find(|layer| layer["name"] == json!(name))
                .unwrap_or_else(|| panic!("{name} is not listed"))
        };
        assert_eq!(named("height")["reads"], json!(["base"]));
        assert_eq!(named("height")["fault"], Value::Null);
        let fault = named("lost")["fault"].as_str().expect("the fault, as text");
        assert!(fault.contains("nowhere"), "{fault:?}");
    }

    // A layer whose own file does not parse has no fault in the document's reads, so
    // the socket has to fall back to the file's error — otherwise a caller sees a layer
    // that silently never bakes.
    #[test]
    fn observing_the_layers_reports_the_error_of_a_file_that_did_not_parse() {
        let terrain = TerrainSpec::new(UVec2::splat(16)).with_layer(Layer::new("height"));

        let mut document = Document::default();
        document.adopt(terrain);
        let mut world = World::new();
        world.insert_resource(document);
        world.insert_resource(ShaderLibrary::with_fault(
            "height.wesl",
            "line 3: expected `;`",
        ));

        let answer = layers(&world);
        assert_eq!(answer["layers"][0]["fault"], json!("line 3: expected `;`"));
    }

    // `observe shaders` is how a caller checks a layer's imports without an IDE, so each
    // name imported has to be listed with whether it reached the library.
    #[test]
    fn observing_the_shaders_reports_each_import_and_whether_it_resolved() {
        let mut world = World::new();
        world.insert_resource(ShaderLibrary::reading(
            "height.wesl",
            "import package::lib::{fbm_unit, seed_offset};\n\nfn value(p: vec2<f32>) -> f32 {\n    return fbm_unit(p + seed_offset(1u, 2u), 4u, 0.5, 2.0);\n}\n",
        ));

        let answer = shaders(&mut world);
        let file = &answer["shaders"][0];
        assert_eq!(
            file["imports"],
            json!([
                { "path": "package::lib::fbm_unit", "resolved": true },
                { "path": "package::lib::seed_offset", "resolved": true },
            ])
        );
        assert_eq!(file["error"], Value::Null);
    }
}
