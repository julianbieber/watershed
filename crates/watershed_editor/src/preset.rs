//! Documents to start from: a small fixed set of worked examples, each layer of them
//! one stock shader file.
//!
//! A preset is a starting point for editing and a fixture for testing, not a terrain
//! anyone is meant to ship. The set stays small for that reason — it is chosen to
//! cover the ways layers can read each other, not to be a library of landscapes.

use crate::gpu;
use crate::terrain::shader::{ShaderLayer, parse_header, parse_layers, parse_params};
use crate::terrain::{Layer, TerrainSpec, WaterSpec};
use bevy::prelude::*;

/// Which starting document to build.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Preset {
    /// Land masses at one scale with bumps at another, from one shader. The simplest
    /// document, and the default.
    #[default]
    Continents,
    /// Ridged relief laid over a continent, which needs the continent to be a layer of
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
    /// Each layer of this preset as `(layer, stock file)`: the layer's shader file is
    /// `<layer>.wesl`, a copy of that stock file, which is always in [`gpu::STOCK`].
    pub fn files(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Continents => &[("moisture", "fbm.wesl"), ("height", "continents.wesl")],
            Self::Ridges => &[
                ("moisture", "fbm.wesl"),
                ("base", "base.wesl"),
                ("height", "mountains_over_base.wesl"),
            ],
        }
    }

    /// The document, unbaked and unsolved.
    ///
    /// Whatever the preset, the result has a layer named `height` holding
    /// `Height`, one named `moisture` holding `Moisture`, and a water spec over the
    /// two — so every preset exercises the water overlay and the role lookups, and
    /// none of them opens on an editor with half its display inert. The roles come
    /// from the stock files the preset copies, as every layer's do.
    ///
    /// Every layer carries a value for every parameter its file declares and the four
    /// properties its header declares, its hidden `seed` drawn from `seed`, and the
    /// document's seed is `seed`. Two calls with the same arguments give equal
    /// documents.
    pub fn build(self, size: UVec2, seed: u32) -> TerrainSpec {
        let mut terrain = match self {
            Self::Continents => continents(size, seed),
            Self::Ridges => ridges(size, seed),
        };
        terrain.water_spec = Some(WaterSpec::new("height").with_moisture("moisture"));
        terrain.seed = seed;
        terrain
    }
}

fn salted(seed: u32, salt: u32) -> u32 {
    let mut hash = seed.wrapping_mul(0x9e37_79b9) ^ salt.wrapping_mul(0x85eb_ca6b);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x2c1b_3c6d);
    hash ^= hash >> 12;
    hash & 0x00ff_ffff
}
fn shader(file: &str, seed: u32, values: &[(&str, f32)]) -> ShaderLayer {
    let source = gpu::stock_source(file).expect("a preset names only shaders this build ships");
    let mut layer = ShaderLayer::default();
    layer.reconcile(&parse_params(source).expect("a stock shader declares readable parameters"));
    layer.reconcile_layers(&parse_layers(source).expect("a stock shader declares readable layers"));
    layer.params.insert("seed".to_owned(), vec![seed as f32]);
    for (name, value) in values {
        layer.params.insert((*name).to_owned(), vec![*value]);
    }
    layer
}

fn built(name: &str, file: &str, seed: u32, values: &[(&str, f32)]) -> Layer {
    let source = gpu::stock_source(file).expect("a preset names only shaders this build ships");
    let mut layer = Layer::new(name);
    layer.reconcile_header(
        &parse_header(source).expect("a stock shader declares a readable header"),
    );
    layer.shader = shader(file, seed, values);
    layer
}

fn moisture(seed: u32) -> Layer {
    built(
        "moisture",
        "fbm.wesl",
        salted(seed, 11),
        &[("scale", 0.004), ("octaves", 4.0)],
    )
}

fn continents(size: UVec2, seed: u32) -> TerrainSpec {
    TerrainSpec::new(size)
        .with_layer(moisture(seed))
        .with_layer(built("height", "continents.wesl", salted(seed, 1), &[]))
}

fn ridges(size: UVec2, seed: u32) -> TerrainSpec {
    TerrainSpec::new(size)
        .with_layer(moisture(seed))
        .with_layer(built("base", "base.wesl", salted(seed, 1), &[]))
        .with_layer(built(
            "height",
            "mountains_over_base.wesl",
            salted(seed, 3),
            &[],
        ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: UVec2 = UVec2::new(96, 96);

    fn stock_of(preset: Preset, layer: &str) -> &'static str {
        preset
            .files()
            .iter()
            .find(|(name, _)| *name == layer)
            .map(|(_, stock)| *stock)
            .unwrap_or_else(|| panic!("{} writes no file for {layer}", preset.name()))
    }

    // Every preset opens with the height and water displays live, and a water spec
    // naming a layer the document does not carry plans fine and fails at the water
    // step, which is a long way from where the mistake is.
    #[test]
    fn every_preset_names_a_height_layer_and_a_water_spec_over_layers_it_has() {
        for preset in Preset::ALL {
            let terrain = preset.build(SIZE, 7);
            assert!(
                terrain.layer("height").is_some(),
                "{} has no height layer",
                preset.name()
            );
            let spec = terrain
                .water_spec
                .clone()
                .unwrap_or_else(|| panic!("{} carries no water spec", preset.name()));
            assert!(terrain.layer(spec.height.as_str()).is_some());
            if let Some(moisture) = &spec.moisture {
                assert!(terrain.layer(moisture.as_str()).is_some());
            }
        }
    }
    // A `new` writes one file per entry of `files` before it bakes, so a layer with no
    // entry would read zero, and a stock file this build does not ship cannot be
    // written at all.
    #[test]
    fn every_layer_of_a_preset_has_a_shipped_file_and_every_file_a_layer() {
        for preset in Preset::ALL {
            let terrain = preset.build(SIZE, 7);
            for (layer, stock) in preset.files() {
                assert!(gpu::stock_source(stock).is_some(), "{stock} is not shipped");
                assert!(terrain.layer(layer).is_some(), "{layer} is not built");
            }
            for layer in &terrain.layers {
                stock_of(preset, layer.id.as_str());
            }
        }
    }

    // What makes `base` a read of `height` in `ridges`: the file `height` is copied
    // from names `base`, and nothing else declares the dependency.
    #[test]
    fn ridges_height_reads_base_by_name() {
        let ridges = Preset::Ridges.build(SIZE, 7);
        let height = ridges.layer("height").unwrap();
        assert_eq!(
            height.shader.layers,
            vec![crate::terrain::LayerId::from("base")]
        );
        assert_eq!(crate::edit::reads_of(height), ["base"]);
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
    fn every_layer_already_carries_every_parameter_its_file_declares() {
        for preset in Preset::ALL {
            for layer in &preset.build(SIZE, 7).layers {
                let source = gpu::stock_source(stock_of(preset, layer.id.as_str())).unwrap();
                let mut copy = layer.shader.clone();
                assert!(
                    !copy.reconcile(&parse_params(source).unwrap()),
                    "{} leaves {} to be reconciled",
                    preset.name(),
                    layer.id
                );
            }
        }
    }

    // A preset's layers are written to disk as copies of their stock files, so a preset
    // whose layer disagreed with the file beside it would be corrected by the first
    // sweep — a document that moves under the person who has just opened it.
    #[test]
    fn every_preset_layer_carries_what_its_stock_file_declares() {
        for preset in Preset::ALL {
            for layer in &preset.build(SIZE, 7).layers {
                let source = gpu::stock_source(stock_of(preset, layer.id.as_str())).unwrap();
                let header = parse_header(source).unwrap();
                let mut copy = layer.clone();
                assert!(
                    !copy.reconcile_header(&header),
                    "{} leaves {} to be reconciled",
                    preset.name(),
                    layer.id
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
