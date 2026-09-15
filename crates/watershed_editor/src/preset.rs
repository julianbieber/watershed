//! Documents to start from: a small fixed set of worked examples, each field of them
//! produced by a shader node.
//!
//! A preset is a starting point for editing and a fixture for testing, not a terrain
//! anyone is meant to ship. The set stays small for that reason — it is chosen to
//! cover the ways a graph can be put together, not to be a library of landscapes.

use crate::gpu;
use crate::terrain::graph::{FieldGraph, NodeOp};
use crate::terrain::shader::{ShaderLayer, parse_inputs, parse_layers, parse_params, parse_reach};
use crate::terrain::{Field, TerrainSpec, WaterSpec};
use bevy::prelude::*;
use watershed::FieldRole;

/// Which starting document to build.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Preset {
    /// Land masses at one scale with bumps at another, from one shader node. The
    /// simplest document, and the default.
    #[default]
    Continents,
    /// Ridged relief laid over a continent, which needs the continent to be a field of
    /// its own that the shader lifting it reads by name.
    Ridges,
}

impl Preset {
    /// Every preset. The fixed-size array is what keeps this from drifting: adding a
    /// variant will not compile until the length and the list are both updated.
    pub const ALL: [Self; 2] = [Self::Continents, Self::Ridges];

    /// The word this preset is named by on the command line, and the only spelling
    /// [`Preset::parse`] accepts.
    pub fn name(self) -> &'static str {
        match self {
            Self::Continents => "continents",
            Self::Ridges => "ridges",
        }
    }

    /// The preset of that exact name, or `None`. Case-sensitive, and does not trim.
    pub fn parse(word: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|preset| preset.name() == word)
    }

    /// The stock shader files the nodes of this preset name, every one of them in
    /// [`gpu::STOCK`]. A document built from the preset reads `0.0` from any of them
    /// its shader directory does not hold.
    pub fn stock_files(self) -> &'static [&'static str] {
        match self {
            Self::Continents => &["fbm.wgsl", "continents.wgsl"],
            Self::Ridges => &["fbm.wgsl", "continents.wgsl", "mountains_over_base.wgsl"],
        }
    }

    /// The document, unbaked and unsolved.
    ///
    /// Whatever the preset, the result has a field named `height` holding
    /// [`FieldRole::Height`], one named `moisture` holding [`FieldRole::Moisture`],
    /// and a water spec over the two — so every preset exercises the water overlay
    /// and the role lookups, and none of them opens on an editor with half its
    /// display inert.
    ///
    /// Every shader node carries a value for every parameter its file declares, its
    /// hidden `seed` drawn from `seed`. Two calls with the same arguments give equal
    /// documents.
    pub fn build(self, size: UVec2, seed: u32) -> TerrainSpec {
        let mut terrain = match self {
            Self::Continents => continents(size, seed),
            Self::Ridges => ridges(size, seed),
        };
        for field in &mut terrain.fields {
            field.role = match field.id.as_str() {
                "height" => FieldRole::Height,
                "moisture" => FieldRole::Moisture,
                _ => FieldRole::Custom,
            };
        }
        terrain.water_spec = Some(WaterSpec::new("height").with_moisture("moisture"));
        terrain
    }
}

fn at(column: i32, row: i32) -> [f32; 2] {
    [column as f32 * 260.0, row as f32 * 150.0]
}

fn salted(seed: u32, salt: u32) -> u32 {
    let mut hash = seed.wrapping_mul(0x9e37_79b9) ^ salt.wrapping_mul(0x85eb_ca6b);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x2c1b_3c6d);
    hash ^= hash >> 12;
    hash & 0x00ff_ffff
}

fn layer(file: &str, seed: u32, values: &[(&str, f32)]) -> ShaderLayer {
    let source = gpu::stock_source(file).expect("a preset names only shaders this build ships");
    let mut layer = ShaderLayer::new(file);
    layer.reconcile(&parse_params(source).expect("a stock shader declares readable parameters"));
    layer.reconcile_inputs(&parse_inputs(source).expect("a stock shader declares readable inputs"));
    layer.reconcile_layers(&parse_layers(source).expect("a stock shader declares readable layers"));
    layer.reconcile_reach(parse_reach(source).expect("a stock shader declares a readable reach"));
    layer.params.insert("seed".to_owned(), vec![seed as f32]);
    for (name, value) in values {
        layer.params.insert((*name).to_owned(), vec![*value]);
    }
    layer
}

fn single(layer: ShaderLayer) -> FieldGraph {
    let mut graph = FieldGraph::new();
    graph.add_node(NodeOp::Shader(layer), at(0, 0));
    graph
}

fn moisture(seed: u32) -> Field {
    Field::new("moisture")
        .with_shift(4)
        .with_graph(single(layer(
            "fbm.wgsl",
            salted(seed, 11),
            &[("scale", 0.004), ("octaves", 4.0)],
        )))
}

fn continents(size: UVec2, seed: u32) -> TerrainSpec {
    TerrainSpec::new(size)
        .with_field(moisture(seed))
        .with_field(Field::new("height").with_graph(single(layer(
            "continents.wgsl",
            salted(seed, 1),
            &[],
        ))))
}

fn ridges(size: UVec2, seed: u32) -> TerrainSpec {
    TerrainSpec::new(size)
        .with_field(moisture(seed))
        .with_field(Field::new("base").with_graph(single(layer(
            "continents.wgsl",
            salted(seed, 1),
            &[("land_scale", 0.0015), ("relief", 0.0)],
        ))))
        .with_field(Field::new("height").with_graph(single(layer(
            "mountains_over_base.wgsl",
            salted(seed, 3),
            &[],
        ))))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: UVec2 = UVec2::new(96, 96);

    fn shaders(terrain: &TerrainSpec) -> Vec<&ShaderLayer> {
        terrain
            .fields
            .iter()
            .flat_map(|field| &field.graph.nodes)
            .filter_map(|node| match &node.op {
                NodeOp::Shader(shader) => Some(shader),
                NodeOp::FieldRef(_) => None,
            })
            .collect()
    }

    // Every preset opens with the height and water displays live, and a water spec
    // naming a field the document does not carry plans fine and fails at the water
    // step, which is a long way from where the mistake is.
    #[test]
    fn every_preset_names_a_height_field_and_a_water_spec_over_fields_it_has() {
        for preset in Preset::ALL {
            let terrain = preset.build(SIZE, 7);
            assert!(
                terrain.field("height").is_some(),
                "{} has no height field",
                preset.name()
            );
            let spec = terrain
                .water_spec
                .clone()
                .unwrap_or_else(|| panic!("{} carries no water spec", preset.name()));
            assert!(terrain.field(spec.height.as_str()).is_some());
            if let Some(moisture) = &spec.moisture {
                assert!(terrain.field(moisture.as_str()).is_some());
            }
        }
    }

    // A `new` writes the preset's stock files before it bakes, so a node naming a file
    // outside that list reads zero, and one this build does not ship cannot be written
    // at all.
    #[test]
    fn every_shader_a_preset_names_is_in_its_stock_files_and_shipped() {
        for preset in Preset::ALL {
            for file in preset.stock_files() {
                assert!(gpu::stock_source(file).is_some(), "{file} is not shipped");
            }
            for shader in shaders(&preset.build(SIZE, 7)) {
                assert!(
                    preset.stock_files().contains(&shader.file.as_str()),
                    "{} names {}, which it does not write",
                    preset.name(),
                    shader.file
                );
            }
        }
    }

    // The shape `observe nodes height` reports for `ridges`: one shader node and no
    // reference, its file naming `base`, which is what makes `base` a read of `height`.
    #[test]
    fn ridges_height_is_one_shader_node_that_reads_base_by_name() {
        let ridges = Preset::Ridges.build(SIZE, 7);
        let height = ridges.field("height").unwrap();
        assert_eq!(height.graph.nodes.len(), 1);
        let NodeOp::Shader(shader) = &height.graph.nodes[0].op else {
            panic!("ridges' height is not a shader node");
        };
        assert_eq!(shader.file, "mountains_over_base.wgsl");
        assert_eq!(shader.layers, vec![watershed::FieldId::from("base")]);
        assert_eq!(crate::edit::reads_of(height), ["base"]);
    }

    // Editing one field's file is how a person changes that field, so no two fields of
    // `ridges` may share a file — an edit to `base`'s would otherwise move `moisture`
    // with it.
    #[test]
    fn every_field_of_ridges_names_a_file_of_its_own() {
        let ridges = Preset::Ridges.build(SIZE, 7);
        let mut files: Vec<&str> = shaders(&ridges)
            .iter()
            .map(|shader| shader.file.as_str())
            .collect();
        let count = files.len();
        files.sort_unstable();
        files.dedup();
        assert_eq!(files.len(), count, "{files:?}");
    }

    // `seed` is the whole of what varies a preset, so the same arguments have to build
    // the same document and another seed a different one.
    #[test]
    fn equal_arguments_build_equal_documents_and_another_seed_builds_another() {
        for preset in Preset::ALL {
            assert_eq!(preset.build(SIZE, 7), preset.build(SIZE, 7));
            assert_ne!(preset.build(SIZE, 7), preset.build(SIZE, 8));
        }
    }

    // A parameter left for the first sweep to fill would make that sweep reconcile it
    // and ask for a second bake of a document that has only just baked.
    #[test]
    fn every_node_already_carries_every_parameter_its_file_declares() {
        for preset in Preset::ALL {
            for shader in shaders(&preset.build(SIZE, 7)) {
                let source = gpu::stock_source(&shader.file).unwrap();
                let mut copy = shader.clone();
                assert!(
                    !copy.reconcile(&parse_params(source).unwrap()),
                    "{} leaves {} to be reconciled",
                    preset.name(),
                    shader.file
                );
            }
        }
    }

    // The names are the command-line surface of the editor, so the two directions have
    // to agree — a name that does not parse back makes a preset unreachable from the
    // control client, and a preset that is gone must not parse at all.
    #[test]
    fn a_preset_is_named_by_the_word_that_parses_back_to_it() {
        for preset in Preset::ALL {
            assert_eq!(Preset::parse(preset.name()), Some(preset));
        }
        assert_eq!(Preset::parse("regions"), None);
        assert_eq!(Preset::parse("nothing-like-this"), None);
    }
}
