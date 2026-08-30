//! Two-way binding between a number field on screen and the number it edits.
//!
//! One enum names every editable number in the editor, and both directions are
//! written against it: [`NumberBinding::read`] says what a field should show and
//! `write` puts a typed value back. Both have to resolve the same name to the same
//! place, so they live side by side — a binding that read one number and wrote
//! another would look like a field that will not take an edit.

use crate::terrain::brush::Brush;
use crate::terrain::graph::{NodeId, NodeOp};
use bevy::feathers::controls::{NumberFormat, NumberInputValue, UpdateNumberInput};
use bevy::prelude::*;
use bevy::ui_widgets::ValueChange;

use crate::brush::BrushSettings;
use crate::document::Document;
use crate::edit::Edit;
use crate::ui::{NewDialog, report};

/// Which number a number field stands for.
///
/// Node-keyed variants carry a [`NodeId`] of the *active* field's graph. An id names
/// its node for the life of the document, so a binding stays pointed at what it was
/// built for across every edit that does not delete that node — and answers "not
/// there" rather than guessing when one does.
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
    ScaleFactor(NodeId),
    RemapFromLow(NodeId),
    RemapFromHigh(NodeId),
    RemapToLow(NodeId),
    RemapToHigh(NodeId),
    Constant(NodeId),
    NoiseSeed(NodeId),
    NoiseScale(NodeId),
    NoiseOctaves(NodeId),
    NoiseStrike(NodeId),
    NoiseAspect(NodeId),
    WarpAmplitude(NodeId),
    WarpScale(NodeId),
    WarpOctaves(NodeId),
    SlopeSampleTiles(NodeId),
    RegionSeed(NodeId),
    RegionCellTiles(NodeId),
    RegionBlendTiles(NodeId),
    RegionWeight(NodeId, usize),
    RegionValue(NodeId, usize, usize),
    /// One component of one parameter of a shader node: the node, the parameter's
    /// position in the node's own key order, and which component of it.
    ///
    /// The parameter is positional because a binding has to be `Copy`, and safe to be
    /// positional because the panel is rebuilt whenever the shader's parameters change
    /// — a binding left over from before answers "not there" rather than writing into
    /// whatever moved into that position.
    ShaderParam(NodeId, usize, usize),
    DialogWidth,
    DialogHeight,
    DialogSeed,
}

impl NumberBinding {
    /// How the field is to be edited. Whole-numbered bindings edit as integers, so the
    /// field cannot offer a fraction the document has nowhere to put.
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

    fn range(self) -> Option<(f32, f32)> {
        match self {
            Self::Shift => Some((0.0, 8.0)),
            Self::BrushRadius => Some((0.0, 512.0)),
            Self::BrushFalloff => Some((0.0, 1.0)),
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

    /// What the field should be showing.
    ///
    /// `None` where the binding names something the document no longer has — a stack
    /// that shrank under a panel waiting to be rebuilt, or a node whose op has
    /// changed to one with no such number.
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
            Self::ScaleFactor(id) => match op(document, id)? {
                NodeOp::Scale(factor) => *factor,
                _ => return None,
            },
            Self::RemapFromLow(id) => remap(document, id)?.from.0,
            Self::RemapFromHigh(id) => remap(document, id)?.from.1,
            Self::RemapToLow(id) => remap(document, id)?.to.0,
            Self::RemapToHigh(id) => remap(document, id)?.to.1,
            Self::Constant(id) => match op(document, id)? {
                NodeOp::Constant(value) => *value,
                _ => return None,
            },
            Self::NoiseSeed(id) => noise(document, id)?.seed as f32,
            Self::NoiseScale(id) => noise(document, id)?.scale,
            Self::NoiseOctaves(id) => noise(document, id)?.octaves as f32,
            Self::NoiseStrike(id) => noise(document, id)?.transform.strike_degrees,
            Self::NoiseAspect(id) => noise(document, id)?.transform.aspect,
            Self::WarpAmplitude(id) => noise(document, id)?.warp.as_ref()?.amplitude,
            Self::WarpScale(id) => noise(document, id)?.warp.as_ref()?.scale,
            Self::WarpOctaves(id) => noise(document, id)?.warp.as_ref()?.octaves as f32,
            Self::SlopeSampleTiles(id) => match op(document, id)? {
                NodeOp::Slope { sample_tiles, .. } => *sample_tiles,
                _ => return None,
            },
            Self::RegionSeed(id) => regions(document, id)?.seed as f32,
            Self::RegionCellTiles(id) => regions(document, id)?.cell_tiles as f32,
            Self::RegionBlendTiles(id) => regions(document, id)?.blend_tiles as f32,
            Self::RegionWeight(id, region) => {
                regions(document, id)?.regions.get(region)?.weight as f32
            }
            Self::RegionValue(id, region, column) => *regions(document, id)?
                .regions
                .get(region)?
                .values
                .get(column)?,
            Self::ShaderParam(id, param, component) => {
                let NodeOp::Shader(shader) = op(document, id)? else {
                    return None;
                };
                *shader.params.values().nth(param)?.get(component)?
            }
        };
        Some(if self.is_integer() {
            NumberInputValue::I32(value as i32)
        } else {
            NumberInputValue::F32(value)
        })
    }

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
                let Some(id) = self.node() else {
                    return false;
                };
                let Some(node) = field.graph.node_mut(id) else {
                    return false;
                };
                return self.write_op(value, &mut node.op);
            }
        }
        true
    }

    fn write_op(self, value: f32, op: &mut NodeOp) -> bool {
        match self {
            Self::ScaleFactor(_) => match op {
                NodeOp::Scale(held) => *held = value,
                _ => return false,
            },
            Self::RemapFromLow(_)
            | Self::RemapFromHigh(_)
            | Self::RemapToLow(_)
            | Self::RemapToHigh(_) => {
                let NodeOp::Remap(remap) = op else {
                    return false;
                };
                match self {
                    Self::RemapFromLow(_) => remap.from.0 = value,
                    Self::RemapFromHigh(_) => remap.from.1 = value,
                    Self::RemapToLow(_) => remap.to.0 = value,
                    _ => remap.to.1 = value,
                }
            }
            Self::Constant(_) => match op {
                NodeOp::Constant(held) => *held = value,
                _ => return false,
            },
            Self::ShaderParam(_, param, component) => {
                let NodeOp::Shader(shader) = op else {
                    return false;
                };
                let Some(slot) = shader
                    .params
                    .values_mut()
                    .nth(param)
                    .and_then(|value| value.get_mut(component))
                else {
                    return false;
                };
                *slot = value;
            }
            Self::NoiseSeed(_)
            | Self::NoiseScale(_)
            | Self::NoiseOctaves(_)
            | Self::NoiseStrike(_)
            | Self::NoiseAspect(_)
            | Self::WarpAmplitude(_)
            | Self::WarpScale(_)
            | Self::WarpOctaves(_) => {
                let NodeOp::Noise(spec) = op else {
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
                let NodeOp::Slope { sample_tiles, .. } = op else {
                    return false;
                };
                *sample_tiles = value;
            }
            Self::RegionSeed(_)
            | Self::RegionCellTiles(_)
            | Self::RegionBlendTiles(_)
            | Self::RegionWeight(..)
            | Self::RegionValue(..) => {
                let NodeOp::Regions { spec, .. } = op else {
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

    fn node(self) -> Option<NodeId> {
        match self {
            Self::ScaleFactor(id)
            | Self::RemapFromLow(id)
            | Self::RemapFromHigh(id)
            | Self::RemapToLow(id)
            | Self::RemapToHigh(id)
            | Self::Constant(id)
            | Self::NoiseSeed(id)
            | Self::NoiseScale(id)
            | Self::NoiseOctaves(id)
            | Self::NoiseStrike(id)
            | Self::NoiseAspect(id)
            | Self::WarpAmplitude(id)
            | Self::WarpScale(id)
            | Self::WarpOctaves(id)
            | Self::SlopeSampleTiles(id)
            | Self::RegionSeed(id)
            | Self::RegionCellTiles(id)
            | Self::RegionBlendTiles(id)
            | Self::RegionWeight(id, _)
            | Self::RegionValue(id, _, _)
            | Self::ShaderParam(id, _, _) => Some(id),
            _ => None,
        }
    }
}

fn field(document: &Document) -> Option<&crate::terrain::Field> {
    document.terrain()?.field(document.active())
}

fn op(document: &Document, id: NodeId) -> Option<&NodeOp> {
    field(document)?.graph.node(id).map(|node| &node.op)
}

fn remap(document: &Document, id: NodeId) -> Option<&crate::terrain::graph::Remap> {
    match op(document, id)? {
        NodeOp::Remap(remap) => Some(remap),
        _ => None,
    }
}

fn noise(document: &Document, id: NodeId) -> Option<&crate::terrain::noise::NoiseSpec> {
    match op(document, id)? {
        NodeOp::Noise(spec) => Some(spec),
        _ => None,
    }
}

fn regions(document: &Document, id: NodeId) -> Option<&crate::terrain::regions::RegionSpec> {
    match op(document, id)? {
        NodeOp::Regions { spec, .. } => Some(spec),
        _ => None,
    }
}

/// Takes a finished float entry and writes it through its binding.
///
/// Only a finished entry: every one of these can provoke a re-bake, and a field part
/// way through being typed holds a number nobody meant.
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

/// As [`on_f32`], for the bindings that edit as integers.
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

/// The other direction: writes what the document holds into every bound field.
///
/// Runs every frame. A field with the keyboard in it is left alone by the widget
/// itself, so this cannot overwrite a number part way through being typed. Bindings
/// that resolve to nothing are skipped, leaving the field as it was.
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
