//! Two-way binding between a number field on screen and the number it edits.
//!
//! One enum names every editable number in the editor, and both directions are
//! written against it: [`NumberBinding::read`] says what a field should show and
//! `write` puts a typed value back. Both have to resolve the same name to the same
//! place, so they live side by side — a binding that read one number and wrote
//! another would look like a field that will not take an edit.

use crate::terrain::graph::{NodeId, NodeOp};
use bevy::feathers::controls::{NumberFormat, NumberInputValue, UpdateNumberInput};
use bevy::prelude::*;
use bevy::ui_widgets::ValueChange;

use crate::document::Document;
use crate::edit::{Edit, Slot};
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
    LightAzimuth,
    ContourInterval,
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
            Self::Shift | Self::DialogWidth | Self::DialogHeight | Self::DialogSeed
        )
    }

    fn range(self) -> Option<(f32, f32)> {
        match self {
            Self::Shift => Some((0.0, 8.0)),
            Self::LightAzimuth => Some((0.0, 360.0)),
            Self::ContourInterval => Some((crate::edit::MIN_CONTOUR_INTERVAL, f32::MAX)),
            Self::DialogSeed => Some((0.0, u32::MAX as f32)),
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
    pub fn read(self, document: &Document, dialog: &NewDialog) -> Option<NumberInputValue> {
        let value = match self {
            Self::Unbound => return None,
            Self::Shift => field(document)?.shift as f32,
            Self::RangeLow => field(document)?.range.0,
            Self::RangeHigh => field(document)?.range.1,
            Self::LightAzimuth => field(document)?.light_azimuth,
            Self::ContourInterval => field(document)?.contour_interval,
            Self::DialogWidth => dialog.width as f32,
            Self::DialogHeight => dialog.height as f32,
            Self::DialogSeed => dialog.seed as f32,
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
        dialog: &mut NewDialog,
    ) -> Result<(), String> {
        let value = self.clamp(value);
        match self {
            Self::Shift => {
                let active = document.active().to_owned();
                document
                    .apply(&Edit::Set {
                        path: format!("{active}.shift"),
                        words: vec![(value as u8).to_string()],
                    })
                    .map(|_| ())
            }
            Self::LightAzimuth => {
                let active = document.active().to_owned();
                document
                    .apply(&Edit::Set {
                        path: format!("{active}.light_azimuth"),
                        words: vec![value.to_string()],
                    })
                    .map(|_| ())
            }
            Self::ContourInterval => {
                let active = document.active().to_owned();
                document
                    .apply(&Edit::Set {
                        path: format!("{active}.contour_interval"),
                        words: vec![value.to_string()],
                    })
                    .map(|_| ())
            }
            Self::DialogWidth => {
                dialog.width = value as u32;
                Ok(())
            }
            Self::DialogHeight => {
                dialog.height = value as u32;
                Ok(())
            }
            Self::DialogSeed => {
                dialog.seed = value as u32;
                Ok(())
            }
            _ => {
                self.write_document(value, document);
                Ok(())
            }
        }
    }

    fn slot(self) -> Slot {
        let property = match self {
            Self::RangeLow => "range.low",
            Self::RangeHigh => "range.high",
            Self::ShaderParam(..) => "shader.param",
            _ => return Slot::Once,
        };
        let index = match self {
            Self::ShaderParam(_, param, component) => [param, component],
            _ => [0, 0],
        };
        Slot::Control {
            property,
            node: self.node(),
            index,
        }
    }

    fn write_document(self, value: f32, document: &mut Document) {
        let active = document.active().to_owned();
        document.write(&active, self.slot(), move |field| match self {
            Self::RangeLow => field.range.0 = value,
            Self::RangeHigh => field.range.1 = value,
            _ => {
                if let Some(node) = self.node().and_then(|id| field.graph.node_mut(id)) {
                    self.write_op(value, &mut node.op);
                }
            }
        });
    }

    fn write_op(self, value: f32, op: &mut NodeOp) -> bool {
        let Self::ShaderParam(_, param, component) = self else {
            return false;
        };
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
        true
    }

    fn node(self) -> Option<NodeId> {
        match self {
            Self::ShaderParam(id, _, _) => Some(id),
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

/// Takes a finished float entry and writes it through its binding.
///
/// Only a finished entry: every one of these can provoke a re-bake, and a field part
/// way through being typed holds a number nobody meant.
pub fn on_f32(
    change: On<ValueChange<f32>>,
    bindings: Query<&NumberBinding>,
    document: ResMut<Document>,
    dialog: ResMut<NewDialog>,
) {
    if !change.is_final {
        return;
    }
    apply(change.source, change.value, &bindings, document, dialog);
}

/// As [`on_f32`], for the bindings that edit as integers.
pub fn on_i32(
    change: On<ValueChange<i32>>,
    bindings: Query<&NumberBinding>,
    document: ResMut<Document>,
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
        dialog,
    );
}

fn apply(
    source: Entity,
    value: f32,
    bindings: &Query<&NumberBinding>,
    mut document: ResMut<Document>,
    mut dialog: ResMut<NewDialog>,
) {
    let Ok(binding) = bindings.get(source) else {
        return;
    };
    let result = binding.write(value, &mut document, &mut dialog);
    report(&mut document, result);
}

/// The other direction: writes what the document holds into every bound field.
///
/// Runs every frame. A field with the keyboard in it is left alone by the widget
/// itself, so this cannot overwrite a number part way through being typed. Bindings
/// that resolve to nothing are skipped, leaving the field as it was.
pub fn push(
    document: Res<Document>,
    dialog: Res<NewDialog>,
    inputs: Query<(Entity, &NumberBinding)>,
    mut commands: Commands,
) {
    for (entity, binding) in inputs.iter() {
        let Some(value) = binding.read(&document, &dialog) else {
            continue;
        };
        commands.trigger(UpdateNumberInput { entity, value });
    }
}
