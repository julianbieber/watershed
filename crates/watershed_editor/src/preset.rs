//! Documents to start from: a small fixed set of worked examples, each exercising a
//! different way of building a height field.
//!
//! A preset is a starting point for editing and a fixture for testing, not a terrain
//! anyone is meant to ship. The set stays small for that reason — it is chosen to
//! cover the ways a stack can be put together, not to be a library of landscapes.

use crate::terrain::layer::{Blend, Layer, LayerOp, Mask, Remap};
use crate::terrain::noise::{NoiseKind, NoiseSpec, WarpSpec, sub_seed};
use crate::terrain::regions::{Region, RegionOutput, RegionSpec};
use crate::terrain::{Field, TerrainSpec, WaterSpec};
use bevy::prelude::*;
use watershed::FieldRole;

/// Which starting document to build.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Preset {
    /// Land masses at one scale with bumps at another, both in one field. The
    /// simplest stack, and the default.
    #[default]
    Continents,
    /// A ridged field masked by the continent it sits on, which needs the continent
    /// to be a field of its own.
    Ridges,
    /// A region tiling feeding two blended columns, one used as a height and one as a
    /// mask on the relief laid over it.
    Regions,
}

impl Preset {
    /// Every preset. The fixed-size array is what keeps this from drifting: adding a
    /// variant will not compile until the length and the list are both updated.
    pub const ALL: [Self; 3] = [Self::Continents, Self::Ridges, Self::Regions];

    /// The word this preset is named by on the command line, and the only spelling
    /// [`Preset::parse`] accepts.
    pub fn name(self) -> &'static str {
        match self {
            Self::Continents => "continents",
            Self::Ridges => "ridges",
            Self::Regions => "regions",
        }
    }

    /// The preset of that exact name, or `None`. Case-sensitive, and does not trim.
    pub fn parse(word: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|preset| preset.name() == word)
    }

    /// The document, unbaked and unsolved.
    ///
    /// Whatever the preset, the result has a field named `height` holding
    /// [`FieldRole::Height`], one named `moisture` holding [`FieldRole::Moisture`],
    /// and a water spec over the two — so every preset exercises the water overlay
    /// and the role lookups, and none of them opens on an editor with half its
    /// display inert.
    ///
    /// `seed` is the document's whole source of variation: two calls with the same
    /// arguments give equal documents.
    pub fn build(self, size: UVec2, seed: u32) -> TerrainSpec {
        let mut terrain = match self {
            Self::Continents => continents(size, seed),
            Self::Ridges => ridges(size, seed),
            Self::Regions => regions(size, seed),
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

fn moisture(seed: u32) -> Field {
    Field::new("moisture").with_shift(4).with_layer(
        Layer::new(LayerOp::Noise(
            NoiseSpec::new(sub_seed(seed, 11), NoiseKind::Fbm, 0.004).with_octaves(4),
        ))
        .with_blend(Blend::Replace),
    )
}

const CONTINENT_SCALE: f32 = 0.0015;
const RELIEF_SCALE: f32 = 0.04;

fn continents(size: UVec2, seed: u32) -> TerrainSpec {
    TerrainSpec::new(size)
        .with_field(moisture(seed))
        .with_field(
            Field::new("height")
                .with_layer(
                    Layer::new(LayerOp::Noise(
                        NoiseSpec::new(sub_seed(seed, 1), NoiseKind::Fbm, CONTINENT_SCALE)
                            .with_octaves(4),
                    ))
                    .with_blend(Blend::Replace),
                )
                .with_layer(
                    Layer::new(LayerOp::Noise(
                        NoiseSpec::new(sub_seed(seed, 2), NoiseKind::Fbm, RELIEF_SCALE)
                            .with_octaves(4),
                    ))
                    .with_blend(Blend::Add)
                    .with_amplitude(0.18),
                ),
        )
}

fn ridges(size: UVec2, seed: u32) -> TerrainSpec {
    TerrainSpec::new(size)
        .with_field(moisture(seed))
        .with_field(
            Field::new("base").with_layer(
                Layer::new(LayerOp::Noise(
                    NoiseSpec::new(sub_seed(seed, 1), NoiseKind::Fbm, CONTINENT_SCALE)
                        .with_octaves(4),
                ))
                .with_blend(Blend::Replace),
            ),
        )
        .with_field(
            Field::new("height")
                .with_layer(Layer::new(LayerOp::FieldRef("base".into())).with_blend(Blend::Replace))
                .with_layer(
                    Layer::new(LayerOp::Noise(
                        NoiseSpec::new(sub_seed(seed, 3), NoiseKind::Ridged, 0.006).with_octaves(5),
                    ))
                    .with_blend(Blend::Add)
                    .with_amplitude(0.55)
                    .with_mask(Mask::Field(
                        "base".into(),
                        Remap::new((0.45, 0.75), (0.0, 1.0)),
                    )),
                )
                .with_layer(
                    Layer::new(LayerOp::Noise(
                        NoiseSpec::new(sub_seed(seed, 4), NoiseKind::Fbm, RELIEF_SCALE)
                            .with_octaves(3),
                    ))
                    .with_blend(Blend::Add)
                    .with_amplitude(0.08),
                ),
        )
}

const REGION_CELL_TILES: u32 = 384;
const REGION_BLEND_TILES: u32 = 48;

fn regions(size: UVec2, seed: u32) -> TerrainSpec {
    let spec = RegionSpec::new(
        sub_seed(seed, 7),
        REGION_CELL_TILES,
        REGION_BLEND_TILES,
        vec!["base".to_owned(), "relief".to_owned()],
    )
    .with_warp(WarpSpec {
        seed: sub_seed(seed, 8),
        amplitude: 160.0,
        scale: 1.0 / 288.0,
        octaves: 3,
        salts: None,
    })
    .with_region(Region::new(3, [0.22, 0.04]))
    .with_region(Region::new(3, [0.52, 0.10]))
    .with_region(Region::new(2, [0.58, 0.16]))
    .with_region(Region::new(2, [0.74, 0.42]))
    .with_region(Region::new(2, [0.48, 0.06]));

    TerrainSpec::new(size)
        .with_field(moisture(seed))
        .with_field(
            Field::new("base").with_shift(2).with_layer(
                Layer::new(LayerOp::Regions {
                    spec: spec.clone(),
                    output: RegionOutput::Blended("base".to_owned()),
                })
                .with_blend(Blend::Replace),
            ),
        )
        .with_field(
            Field::new("relief").with_shift(2).with_layer(
                Layer::new(LayerOp::Regions {
                    spec,
                    output: RegionOutput::Blended("relief".to_owned()),
                })
                .with_blend(Blend::Replace),
            ),
        )
        .with_field(
            Field::new("height")
                .with_layer(Layer::new(LayerOp::FieldRef("base".into())).with_blend(Blend::Replace))
                .with_layer(
                    Layer::new(LayerOp::Noise(
                        NoiseSpec::new(sub_seed(seed, 9), NoiseKind::Fbm, RELIEF_SCALE)
                            .with_octaves(4),
                    ))
                    .with_blend(Blend::Add)
                    .with_amplitude(1.0)
                    .with_mask(Mask::Field("relief".into(), Remap::IDENTITY)),
                ),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: UVec2 = UVec2::new(96, 96);

    // Presets are the fixtures everything else in the editor is exercised against, so
    // one that does not bake takes the whole editor's test coverage with it.
    #[test]
    fn every_preset_bakes_and_names_a_height_field() {
        for preset in Preset::ALL {
            let mut terrain = preset.build(SIZE, 7);
            terrain
                .bake_in_place()
                .unwrap_or_else(|error| panic!("{} did not bake: {error}", preset.name()));
            assert!(
                terrain.field("height").is_some(),
                "{} has no height field",
                preset.name()
            );
        }
    }

    // Asserted per preset rather than once, because a flat height is a silent failure:
    // it bakes, it solves, and it draws as a single colour that looks like a rendering
    // fault rather than a stack that cancelled itself out.
    #[test]
    fn every_preset_produces_a_height_that_varies() {
        for preset in Preset::ALL {
            let mut terrain = preset.build(SIZE, 7);
            terrain.bake_in_place().unwrap();

            let baked = terrain.field("height").unwrap().baked();
            let (low, high) = baked
                .data()
                .iter()
                .fold((f32::MAX, f32::MIN), |(low, high), &value| {
                    (low.min(value), high.max(value))
                });

            assert!(
                high - low > 0.05,
                "{} spans only {low}..{high}",
                preset.name()
            );
        }
    }

    // A water spec naming a field the document does not carry plans fine and fails at
    // the water step, which is a long way from where the mistake is.
    #[test]
    fn every_preset_names_a_water_spec_over_fields_it_has() {
        for preset in Preset::ALL {
            let terrain = preset.build(SIZE, 7);
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

    // The water overlay is only exercised if the presets actually pond; a height with
    // no depressions would leave that whole display path untested.
    #[test]
    fn every_preset_solves_water_that_ponds_somewhere() {
        for preset in Preset::ALL {
            let mut terrain = preset.build(SIZE, 7);
            terrain.bake_in_place().unwrap();
            let spec = terrain.water_spec.clone().unwrap();
            terrain
                .solve_water(&spec)
                .unwrap_or_else(|error| panic!("{} did not solve: {error}", preset.name()));

            assert!(terrain.water().is_some(), "{}", preset.name());
        }
    }

    // The names are the command-line surface of the editor, so the two directions have
    // to agree — a name that does not parse back makes a preset unreachable from the
    // control client.
    #[test]
    fn a_preset_is_named_by_the_word_that_parses_back_to_it() {
        for preset in Preset::ALL {
            assert_eq!(Preset::parse(preset.name()), Some(preset));
        }
        assert_eq!(Preset::parse("nothing-like-this"), None);
    }
}
