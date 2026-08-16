// TODO(jb-doc): module docs — that one enum is the whole two-way binding between a number
// field and the thing it edits, and why the read and the write have to name the same place.

use bevy::feathers::controls::{NumberFormat, NumberInputValue, UpdateNumberInput};
use bevy::prelude::*;
use bevy::ui_widgets::ValueChange;
use watershed::brush::Brush;
use watershed::layer::{Layer, LayerOp, Mask};

use crate::brush::BrushSettings;
use crate::document::Document;
use crate::edit::Edit;
use crate::ui::{NewDialog, report};

/// Which number a number field stands for. Layer-indexed variants carry the index within
/// the *active* field's stack, which is what the panel is built from and what a rebuild
/// re-derives.
#[derive(Component, Clone, Copy, Default, PartialEq, Eq, Hash, Debug)]
pub enum NumberBinding {
    /// A field naming nothing. Never built by the panel — it is what the scene system
    /// needs a binding to be able to be before one is written over it.
    #[default]
    Unbound,
    Shift,
    RangeLow,
    RangeHigh,
    BrushRadius,
    BrushFalloff,
    BrushStrength,
    BrushValue,
    Amplitude(usize),
    MaskConstant(usize),
    MaskFromLow(usize),
    MaskFromHigh(usize),
    MaskToLow(usize),
    MaskToHigh(usize),
    Constant(usize),
    NoiseSeed(usize),
    NoiseScale(usize),
    NoiseOctaves(usize),
    NoiseStrike(usize),
    NoiseAspect(usize),
    WarpAmplitude(usize),
    WarpScale(usize),
    WarpOctaves(usize),
    SlopeSampleTiles(usize),
    RegionSeed(usize),
    RegionCellTiles(usize),
    RegionBlendTiles(usize),
    RegionWeight(usize, usize),
    RegionValue(usize, usize, usize),
    DialogWidth,
    DialogHeight,
    DialogSeed,
}

impl NumberBinding {
    /// Whole numbers are edited as whole numbers, so the field cannot offer a fraction the
    /// document has nowhere to put.
    pub fn format(self) -> NumberFormat {
        if self.is_integer() {
            NumberFormat::I32
        } else {
            NumberFormat::F32
        }
    }

    fn is_integer(self) -> bool {
        matches!(
            self,
            Self::Shift
                | Self::NoiseSeed(_)
                | Self::NoiseOctaves(_)
                | Self::WarpOctaves(_)
                | Self::RegionSeed(_)
                | Self::RegionCellTiles(_)
                | Self::RegionBlendTiles(_)
                | Self::RegionWeight(..)
                | Self::DialogWidth
                | Self::DialogHeight
                | Self::DialogSeed
        )
    }

    /// The range the ctl's own verbs enforce. A field with no range answers `None`.
    ///
    /// TODO(jb-comment): why the clamp lives here rather than in the widget, and what the
    /// a spinner did for the panel that a text field has to do for itself.
    fn range(self) -> Option<(f32, f32)> {
        match self {
            Self::Shift => Some((0.0, 8.0)),
            Self::BrushRadius => Some((0.0, 512.0)),
            Self::BrushFalloff => Some((0.0, 1.0)),
            Self::MaskConstant(_) => Some((0.0, 1.0)),
            Self::NoiseScale(_) | Self::WarpScale(_) => Some((0.0, 1.0)),
            Self::NoiseOctaves(_) => Some((1.0, 10.0)),
            Self::WarpOctaves(_) => Some((1.0, 6.0)),
            Self::NoiseStrike(_) => Some((-180.0, 180.0)),
            Self::NoiseAspect(_) => Some((1.0, 32.0)),
            Self::SlopeSampleTiles(_) => Some((0.5, 64.0)),
            Self::RegionCellTiles(_) => Some((8.0, 4096.0)),
            Self::RegionBlendTiles(_) => Some((0.0, 1024.0)),
            Self::RegionWeight(..) => Some((0.0, u32::MAX as f32)),
            Self::NoiseSeed(_) | Self::RegionSeed(_) | Self::DialogSeed => {
                Some((0.0, u32::MAX as f32))
            }
            Self::DialogWidth | Self::DialogHeight => Some((16.0, 8192.0)),
            _ => None,
        }
    }

    fn clamp(self, value: f32) -> f32 {
        match self.range() {
            Some((low, high)) => value.clamp(low, high),
            None => value,
        }
    }

    /// What the field should be showing. `None` where the binding names something the
    /// document no longer has — a stack that shrank under a panel waiting to be rebuilt.
    pub fn read(
        self,
        document: &Document,
        brush: &Brush,
        dialog: &NewDialog,
    ) -> Option<NumberInputValue> {
        let value = match self {
            Self::Unbound => return None,
            Self::Shift => field(document)?.shift as f32,
            Self::RangeLow => field(document)?.range.0,
            Self::RangeHigh => field(document)?.range.1,
            Self::BrushRadius => brush.radius_cells,
            Self::BrushFalloff => brush.falloff,
            Self::BrushStrength => brush.strength,
            Self::BrushValue => brush.value,
            Self::DialogWidth => dialog.width as f32,
            Self::DialogHeight => dialog.height as f32,
            Self::DialogSeed => dialog.seed as f32,
            Self::Amplitude(index) => layer(document, index)?.amplitude,
            Self::MaskConstant(index) => match layer(document, index)?.mask {
                Mask::Constant(value) => value,
                _ => return None,
            },
            Self::MaskFromLow(index) => remap(document, index)?.from.0,
            Self::MaskFromHigh(index) => remap(document, index)?.from.1,
            Self::MaskToLow(index) => remap(document, index)?.to.0,
            Self::MaskToHigh(index) => remap(document, index)?.to.1,
            Self::Constant(index) => match &layer(document, index)?.op {
                LayerOp::Constant(value) => *value,
                _ => return None,
            },
            Self::NoiseSeed(index) => noise(document, index)?.seed as f32,
            Self::NoiseScale(index) => noise(document, index)?.scale,
            Self::NoiseOctaves(index) => noise(document, index)?.octaves as f32,
            Self::NoiseStrike(index) => noise(document, index)?.transform.strike_degrees,
            Self::NoiseAspect(index) => noise(document, index)?.transform.aspect,
            Self::WarpAmplitude(index) => noise(document, index)?.warp.as_ref()?.amplitude,
            Self::WarpScale(index) => noise(document, index)?.warp.as_ref()?.scale,
            Self::WarpOctaves(index) => noise(document, index)?.warp.as_ref()?.octaves as f32,
            Self::SlopeSampleTiles(index) => match &layer(document, index)?.op {
                LayerOp::Slope { sample_tiles, .. } => *sample_tiles,
                _ => return None,
            },
            Self::RegionSeed(index) => regions(document, index)?.seed as f32,
            Self::RegionCellTiles(index) => regions(document, index)?.cell_tiles as f32,
            Self::RegionBlendTiles(index) => regions(document, index)?.blend_tiles as f32,
            Self::RegionWeight(index, region) => {
                regions(document, index)?.regions.get(region)?.weight as f32
            }
            Self::RegionValue(index, region, column) => *regions(document, index)?
                .regions
                .get(region)?
                .values
                .get(column)?,
        };
        Some(if self.is_integer() {
            NumberInputValue::I32(value as i32)
        } else {
            NumberInputValue::F32(value)
        })
    }

    /// Writes the value where it belongs, answering whether the bake the document holds is
    /// no longer the bake the document describes.
    ///
    /// The shift goes through [`Edit::Set`] rather than being written here, so the rule
    /// about the water spec's height field lives in one place.
    fn write(
        self,
        value: f32,
        document: &mut Document,
        brush: &mut Brush,
        dialog: &mut NewDialog,
    ) -> Result<bool, String> {
        let value = self.clamp(value);
        match self {
            Self::Shift => {
                let active = document.active().to_owned();
                document
                    .apply(&Edit::Set {
                        path: format!("{active}.shift"),
                        words: vec![(value as u8).to_string()],
                    })
                    .map(|_| false)
            }
            Self::BrushRadius => {
                brush.radius_cells = value;
                Ok(false)
            }
            Self::BrushFalloff => {
                brush.falloff = value;
                Ok(false)
            }
            Self::BrushStrength => {
                brush.strength = value;
                Ok(false)
            }
            Self::BrushValue => {
                brush.value = value;
                Ok(false)
            }
            Self::DialogWidth => {
                dialog.width = value as u32;
                Ok(false)
            }
            Self::DialogHeight => {
                dialog.height = value as u32;
                Ok(false)
            }
            Self::DialogSeed => {
                dialog.seed = value as u32;
                Ok(false)
            }
            _ => {
                let written = self.write_document(value, document);
                Ok(written)
            }
        }
    }

    fn write_document(self, value: f32, document: &mut Document) -> bool {
        let active = document.active().to_owned();
        let Some(terrain) = document.terrain_mut() else {
            return false;
        };
        let Some(field) = terrain.field_mut(&active) else {
            return false;
        };

        match self {
            Self::RangeLow => field.range.0 = value,
            Self::RangeHigh => field.range.1 = value,
            _ => {
                let index = match self.layer_index() {
                    Some(index) => index,
                    None => return false,
                };
                let Some(layer) = field.layers.get_mut(index) else {
                    return false;
                };
                return self.write_layer(value, layer);
            }
        }
        true
    }

    fn write_layer(self, value: f32, layer: &mut Layer) -> bool {
        match self {
            Self::Amplitude(_) => layer.amplitude = value,
            Self::MaskConstant(_) => match &mut layer.mask {
                Mask::Constant(held) => *held = value,
                _ => return false,
            },
            Self::MaskFromLow(_)
            | Self::MaskFromHigh(_)
            | Self::MaskToLow(_)
            | Self::MaskToHigh(_) => {
                let Mask::Field(_, remap) = &mut layer.mask else {
                    return false;
                };
                match self {
                    Self::MaskFromLow(_) => remap.from.0 = value,
                    Self::MaskFromHigh(_) => remap.from.1 = value,
                    Self::MaskToLow(_) => remap.to.0 = value,
                    _ => remap.to.1 = value,
                }
            }
            Self::Constant(_) => match &mut layer.op {
                LayerOp::Constant(held) => *held = value,
                _ => return false,
            },
            Self::NoiseSeed(_)
            | Self::NoiseScale(_)
            | Self::NoiseOctaves(_)
            | Self::NoiseStrike(_)
            | Self::NoiseAspect(_)
            | Self::WarpAmplitude(_)
            | Self::WarpScale(_)
            | Self::WarpOctaves(_) => {
                let LayerOp::Noise(spec) = &mut layer.op else {
                    return false;
                };
                match self {
                    Self::NoiseSeed(_) => spec.seed = value as u32,
                    Self::NoiseScale(_) => spec.scale = value,
                    Self::NoiseOctaves(_) => spec.octaves = value as u32,
                    Self::NoiseStrike(_) => spec.transform.strike_degrees = value,
                    Self::NoiseAspect(_) => spec.transform.aspect = value,
                    _ => {
                        let Some(warp) = &mut spec.warp else {
                            return false;
                        };
                        match self {
                            Self::WarpAmplitude(_) => warp.amplitude = value,
                            Self::WarpScale(_) => warp.scale = value,
                            _ => warp.octaves = value as u32,
                        }
                    }
                }
            }
            Self::SlopeSampleTiles(_) => {
                let LayerOp::Slope { sample_tiles, .. } = &mut layer.op else {
                    return false;
                };
                *sample_tiles = value;
            }
            Self::RegionSeed(_)
            | Self::RegionCellTiles(_)
            | Self::RegionBlendTiles(_)
            | Self::RegionWeight(..)
            | Self::RegionValue(..) => {
                let LayerOp::Regions { spec, .. } = &mut layer.op else {
                    return false;
                };
                match self {
                    Self::RegionSeed(_) => spec.seed = value as u32,
                    Self::RegionCellTiles(_) => spec.cell_tiles = value as u32,
                    Self::RegionBlendTiles(_) => spec.blend_tiles = value as u32,
                    Self::RegionWeight(_, region) => {
                        let Some(region) = spec.regions.get_mut(region) else {
                            return false;
                        };
                        region.weight = value as u32;
                    }
                    _ => {
                        let Self::RegionValue(_, region, column) = self else {
                            return false;
                        };
                        let Some(region) = spec.regions.get_mut(region) else {
                            return false;
                        };
                        let Some(held) = region.values.get_mut(column) else {
                            return false;
                        };
                        *held = value;
                    }
                }
            }
            _ => return false,
        }
        true
    }

    fn layer_index(self) -> Option<usize> {
        match self {
            Self::Amplitude(index)
            | Self::MaskConstant(index)
            | Self::MaskFromLow(index)
            | Self::MaskFromHigh(index)
            | Self::MaskToLow(index)
            | Self::MaskToHigh(index)
            | Self::Constant(index)
            | Self::NoiseSeed(index)
            | Self::NoiseScale(index)
            | Self::NoiseOctaves(index)
            | Self::NoiseStrike(index)
            | Self::NoiseAspect(index)
            | Self::WarpAmplitude(index)
            | Self::WarpScale(index)
            | Self::WarpOctaves(index)
            | Self::SlopeSampleTiles(index)
            | Self::RegionSeed(index)
            | Self::RegionCellTiles(index)
            | Self::RegionBlendTiles(index)
            | Self::RegionWeight(index, _)
            | Self::RegionValue(index, _, _) => Some(index),
            _ => None,
        }
    }
}

fn field(document: &Document) -> Option<&watershed::Field> {
    document.terrain()?.field(document.active())
}

fn layer(document: &Document, index: usize) -> Option<&Layer> {
    field(document)?.layers.get(index)
}

fn remap(document: &Document, index: usize) -> Option<&watershed::layer::Remap> {
    match &layer(document, index)?.mask {
        Mask::Field(_, remap) => Some(remap),
        _ => None,
    }
}

fn noise(document: &Document, index: usize) -> Option<&watershed::noise::NoiseSpec> {
    match &layer(document, index)?.op {
        LayerOp::Noise(spec) => Some(spec),
        _ => None,
    }
}

fn regions(document: &Document, index: usize) -> Option<&watershed::regions::RegionSpec> {
    match &layer(document, index)?.op {
        LayerOp::Regions { spec, .. } => Some(spec),
        _ => None,
    }
}

/// A number is taken when the entry is finished rather than as it is typed: every one of
/// these provokes a re-bake, and a field being typed into is a half-written number for as
/// long as it takes to write the whole one.
pub fn on_f32(
    change: On<ValueChange<f32>>,
    bindings: Query<&NumberBinding>,
    document: ResMut<Document>,
    brush: ResMut<BrushSettings>,
    dialog: ResMut<NewDialog>,
) {
    if !change.is_final {
        return;
    }
    apply(
        change.source,
        change.value,
        &bindings,
        document,
        brush,
        dialog,
    );
}

pub fn on_i32(
    change: On<ValueChange<i32>>,
    bindings: Query<&NumberBinding>,
    document: ResMut<Document>,
    brush: ResMut<BrushSettings>,
    dialog: ResMut<NewDialog>,
) {
    if !change.is_final {
        return;
    }
    apply(
        change.source,
        change.value as f32,
        &bindings,
        document,
        brush,
        dialog,
    );
}

fn apply(
    source: Entity,
    value: f32,
    bindings: &Query<&NumberBinding>,
    mut document: ResMut<Document>,
    mut brush: ResMut<BrushSettings>,
    mut dialog: ResMut<NewDialog>,
) {
    let Ok(binding) = bindings.get(source) else {
        return;
    };
    match binding.write(value, &mut document, &mut brush.0, &mut dialog) {
        Ok(true) => document.note_edit(),
        Ok(false) => {}
        Err(error) => report(&mut document, Err(error)),
    }
}

/// The other direction: whatever the document holds is what the fields show. A field with
/// the keyboard in it is left alone by the widget itself, which is what keeps this from
/// overwriting a number halfway through being typed.
pub fn push(
    document: Res<Document>,
    brush: Res<BrushSettings>,
    dialog: Res<NewDialog>,
    inputs: Query<(Entity, &NumberBinding)>,
    mut commands: Commands,
) {
    for (entity, binding) in inputs.iter() {
        let Some(value) = binding.read(&document, &brush.0, &dialog) else {
            continue;
        };
        commands.trigger(UpdateNumberInput { entity, value });
    }
}
