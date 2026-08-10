// TODO(jb-doc): module docs — what a preset is for, and why the set is deliberately
// small rather than a library of terrains.

use bevy::prelude::*;
use watershed::layer::{Blend, Layer, LayerOp, Mask, Remap};
use watershed::noise::{NoiseKind, NoiseSpec, WarpSpec, sub_seed};
use watershed::regions::{Region, RegionOutput, RegionSpec};
use watershed::{Field, Terrain, WaterSpec};

/// TODO(jb-doc): why the list is an array over the enum rather than a registry, and what
/// stops it drifting from [`Preset::ALL`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Preset {
    #[default]
    Continents,
    Ridges,
    Regions,
}

impl Preset {
    pub const ALL: [Self; 3] = [Self::Continents, Self::Ridges, Self::Regions];

    pub fn name(self) -> &'static str {
        match self {
            Self::Continents => "continents",
            Self::Ridges => "ridges",
            Self::Regions => "regions",
        }
    }

    pub fn parse(word: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|preset| preset.name() == word)
    }

    /// TODO(jb-doc): why every preset carries a moisture field and a water spec, and what
    /// a preset that carried neither would leave the editor unable to show.
    pub fn build(self, size: UVec2, seed: u32) -> Terrain {
        let mut terrain = match self {
            Self::Continents => continents(size, seed),
            Self::Ridges => ridges(size, seed),
            Self::Regions => regions(size, seed),
        };
        terrain.water_spec = Some(WaterSpec::new("height").with_moisture("moisture"));
        terrain
    }
}

/// TODO(jb-comment): why moisture is baked four shifts down from the height rather than
/// at the document's own resolution, and what that costs the water solve that weights by
/// it.
fn moisture(seed: u32) -> Field {
    Field::new("moisture").with_shift(4).with_layer(
        Layer::new(LayerOp::Noise(
            NoiseSpec::new(sub_seed(seed, 11), NoiseKind::Fbm, 0.004).with_octaves(4),
        ))
        .with_blend(Blend::Replace),
    )
}

// TODO(jb-comment): where these two scales come from — the wavelengths they put a land
// mass and its bumps at, and why no threshold can cut regions out of the second alone.
const CONTINENT_SCALE: f32 = 0.0015;
const RELIEF_SCALE: f32 = 0.04;

fn continents(size: UVec2, seed: u32) -> Terrain {
    Terrain::new(size).with_field(moisture(seed)).with_field(
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
                    NoiseSpec::new(sub_seed(seed, 2), NoiseKind::Fbm, RELIEF_SCALE).with_octaves(4),
                ))
                .with_blend(Blend::Add)
                .with_amplitude(0.18),
            ),
    )
}

fn ridges(size: UVec2, seed: u32) -> Terrain {
    // The continent is a field of its own rather than the height's first layer, because
    // the ridge is masked by it — and a field masked by itself is a cycle the bake
    // refuses, not a read of the layers underneath.
    Terrain::new(size)
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
                // TODO(jb-comment): why the ridge is masked by the continent it sits on
                // rather than laid flat across the document, and what an unmasked one does
                // to the coast.
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

// TODO(jb-comment): why the region cell and blend are set in tiles against the document's
// own size rather than as a fraction of it, and what a fraction would do to a 512-cell
// preview of a 4096-cell document.
const REGION_CELL_TILES: u32 = 384;
const REGION_BLEND_TILES: u32 = 48;

/// TODO(jb-doc): what the column means to the layer that reads it — that a region carries
/// numbers only, and the blend at a cell is the distance-weighted mix of the nearby
/// sites'.
fn regions(size: UVec2, seed: u32) -> Terrain {
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

    Terrain::new(size)
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
                // TODO(jb-comment): why the relief column reaches the height as a *mask*
                // on an ordinary noise layer rather than as an amplitude, and what that
                // buys a boundary between two regions of unlike relief.
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

    #[test]
    fn every_preset_bakes_and_names_a_height_field() {
        for preset in Preset::ALL {
            let mut terrain = preset.build(SIZE, 7);
            terrain
                .bake()
                .unwrap_or_else(|error| panic!("{} did not bake: {error}", preset.name()));
            assert!(
                terrain.field("height").is_some(),
                "{} has no height field",
                preset.name()
            );
        }
    }

    /// TODO(jb-comment): why this is asserted per preset rather than once — that a preset
    /// whose height came out flat would still bake, still solve, and draw as one colour.
    #[test]
    fn every_preset_produces_a_height_that_varies() {
        for preset in Preset::ALL {
            let mut terrain = preset.build(SIZE, 7);
            terrain.bake().unwrap();

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

    #[test]
    fn every_preset_solves_water_that_ponds_somewhere() {
        for preset in Preset::ALL {
            let mut terrain = preset.build(SIZE, 7);
            terrain.bake().unwrap();
            let spec = terrain.water_spec.clone().unwrap();
            terrain
                .solve_water(&spec)
                .unwrap_or_else(|error| panic!("{} did not solve: {error}", preset.name()));

            assert!(terrain.water().is_some(), "{}", preset.name());
        }
    }

    #[test]
    fn a_preset_is_named_by_the_word_that_parses_back_to_it() {
        for preset in Preset::ALL {
            assert_eq!(Preset::parse(preset.name()), Some(preset));
        }
        assert_eq!(Preset::parse("nothing-like-this"), None);
    }
}
