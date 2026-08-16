// TODO(jb-doc): module docs — the rule the whole panel is built on: a choice is shape and
// a number is a value, so choosing rebuilds and typing does not.

use bevy::feathers::containers::{group, group_body, group_header};
use bevy::feathers::controls::{
    ButtonVariant, FeathersButton, FeathersCheckbox, FeathersDisclosureToggle, FeathersToolButton,
};
use bevy::feathers::theme::{ThemeBackgroundColor, ThemedText};
use bevy::feathers::tokens;
use bevy::prelude::*;
use bevy::ui::{Checked, InteractionDisabled};
use bevy::ui_widgets::{Activate, ValueChange};
use watershed::layer::{Layer, LayerOp, Mask, Remap, SlopeMode};
use watershed::noise::{NoiseKind, NoiseSpec};
use watershed::raster::Raster;
use watershed::{FieldId, FieldRole};

use crate::brush::{BrushSettings, target_of};
use crate::document::{Baked, Document};
use crate::edit::{
    BLENDS, BRUSH_MODES, Edit, NOISE_KINDS, SLOPE_MODES, blend_name, brush_mode_name,
    noise_kind_name, op_name, op_summary, parse_region_output, region_output_name, slope_mode_name,
};
use crate::ui::bind::NumberBinding;
use crate::ui::widgets::{self, one};
use crate::ui::{ADDABLE, AddLayer, Expanded, PANEL_WIDTH, report};

/// The panel's children are thrown away and rebuilt whenever this changes. Everything a
/// caption prints and everything that decides how many widgets there are is in it;
/// nothing a number field holds is.
#[derive(Resource, Default, PartialEq, Eq)]
pub struct Shape(String);

#[derive(Component, Default, Clone)]
pub struct StackBody;

#[derive(Component, Default, Clone)]
pub struct PreviewTag;

pub fn panel() -> impl Scene {
    bsn! {
        Node {
            width: {px(PANEL_WIDTH)},
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Stretch,
            row_gap: px(6),
            padding: px(8),
            overflow: {Overflow::scroll_y()},
        }
        ThemeBackgroundColor(tokens::WINDOW_BG)
        StackBody
    }
}

/// The stack of the field the toolbar has selected, which is what makes moisture and
/// temperature editable by this panel with nothing about them written here.
pub fn rebuild(
    document: Res<Document>,
    brush: Res<BrushSettings>,
    expanded: Res<Expanded>,
    add: Res<AddLayer>,
    mut shape: ResMut<Shape>,
    body: Single<Entity, With<StackBody>>,
    mut commands: Commands,
) {
    let current = Shape(fingerprint(&document, &brush, &expanded, &add));
    if *shape == current {
        return;
    }
    *shape = current;

    commands
        .entity(*body)
        .despawn_related::<Children>()
        .queue_spawn_related_scenes::<Children>(contents(&document, &brush, &expanded, &add));
}

/// What the panel prints that is neither a choice nor a number: whether the bake on screen
/// is the whole document, and whether the shift may be moved at all.
pub fn sync(document: Res<Document>, mut preview: Query<&mut Text, With<PreviewTag>>) {
    let previewing = document.baked() != Baked::Whole || document.is_dirty();
    for mut text in preview.iter_mut() {
        widgets::set_text(&mut text, if previewing { "preview" } else { "" });
    }
}

/// Whether the shift is one the panel refuses to move: the water solve reads its height
/// field one texel per cell, so a coarse one is a document that can never solve.
///
/// Only while the field is *already* right, so a document that arrived at a coarse height
/// some other way can be put back rather than being locked out of the panel that would
/// repair it. The guard proper is [`Edit::Set`] refusing — this is what stops a person
/// reaching for it in the first place.
fn shift_is_pinned(document: &Document) -> bool {
    document
        .terrain()
        .zip(field_of(document))
        .is_some_and(|(terrain, field)| {
            crate::edit::is_solve_height(terrain, document.active()) && field.shift == 0
        })
}

fn field_of(document: &Document) -> Option<&watershed::Field> {
    document.terrain()?.field(document.active())
}

fn fingerprint(
    document: &Document,
    brush: &BrushSettings,
    expanded: &Expanded,
    add: &AddLayer,
) -> String {
    let mut key = String::new();
    key.push_str(document.active());
    key.push('|');
    key.push_str(&document.field_names().join(","));
    key.push('|');
    key.push_str(&add.0);
    key.push('|');
    key.push_str(brush_mode_name(brush.0.mode));
    key.push('|');
    key.push_str(if expanded.brush { "open" } else { "shut" });
    key.push('|');
    key.push_str(if shift_is_pinned(document) {
        "pinned"
    } else {
        "free"
    });
    key.push('|');
    match target_of(document) {
        Some((field, index)) => key.push_str(&format!("{field}:{index}")),
        None => key.push_str("none"),
    }
    key.push('|');

    let Some(field) = field_of(document) else {
        return key + "empty";
    };
    key.push_str(field.role.as_str());
    for (index, layer) in field.layers.iter().enumerate() {
        key.push_str(&format!(
            "|{index}:{}:{}:{}:{}:{}",
            op_name(&layer.op),
            blend_name(layer.blend),
            layer.enabled,
            mask_kind(&layer.mask),
            expanded.has(index),
        ));
        match &layer.op {
            LayerOp::Noise(spec) => {
                key.push_str(noise_kind_name(spec.kind));
                key.push_str(if spec.warp.is_some() { ":warp" } else { "" });
            }
            LayerOp::Slope { of, mode, .. } => {
                key.push_str(&format!("{of}:{}", slope_mode_name(*mode)));
            }
            LayerOp::FieldRef(id) => key.push_str(id.as_ref()),
            LayerOp::Regions { spec, output } => {
                key.push_str(&region_output_name(output));
                key.push_str(&spec.columns.join(","));
                key.push_str(&format!(":{}", spec.regions.len()));
            }
            _ => {}
        }
        if let Mask::Field(id, _) = &layer.mask {
            key.push_str(&format!(":mask{id}"));
        }
    }
    key
}

fn mask_kind(mask: &Mask) -> &'static str {
    match mask {
        Mask::Constant(_) => "constant",
        Mask::Painted(_) => "painted",
        Mask::Field(..) => "field",
    }
}

fn contents(
    document: &Document,
    brush: &BrushSettings,
    expanded: &Expanded,
    add: &AddLayer,
) -> Vec<Box<dyn SceneList>> {
    let active = document.active().to_owned();
    let names = document.field_names();
    let Some(field) = field_of(document) else {
        return vec![one(widgets::text("no document"))];
    };

    let mut children: Vec<Box<dyn SceneList>> = vec![
        one(widgets::row(vec![
            one(widgets::text(active.clone())),
            one(bsn! { widgets::small("") PreviewTag }),
        ])),
        one(brush_section(document, brush, expanded)),
        one(properties(&active, field, shift_is_pinned(document))),
    ];

    for (index, layer) in field.layers.iter().enumerate() {
        children.push(one(layer_entry(
            &active,
            index,
            field.layers.len(),
            layer,
            &names,
            expanded.has(index),
        )));
    }

    children.push(one(add_row(&active, &names, add)));
    children
}

/// A shift is the one edit that discards the bake rather than patching it, since it
/// changes the raster the field is written onto.
fn properties(active: &str, field: &watershed::Field, pinned: bool) -> impl Scene {
    let active = active.to_owned();
    let role = field.role;
    let role_items: Vec<Box<dyn SceneList>> = FieldRole::ALL
        .into_iter()
        .map(|choice| {
            let active = active.clone();
            one(bsn! {
                widgets::item_caption(choice.as_str())
                on(move |_: On<Activate>, mut document: ResMut<Document>| {
                    let result = document
                        .apply(&Edit::Set {
                            path: format!("{active}.role"),
                            words: vec![choice.as_str().to_owned()],
                        })
                        .map(|_| ());
                    report(&mut document, result);
                })
            })
        })
        .collect();

    widgets::column(vec![
        one(widgets::captioned(
            "shift",
            if pinned {
                one(widgets::small(field.shift.to_string()))
            } else {
                one(widgets::number(NumberBinding::Shift))
            },
        )),
        one(widgets::captioned(
            "role",
            one(widgets::menu(role.as_str(), role_items)),
        )),
        one(widgets::captioned(
            "range",
            one(widgets::row(vec![
                one(widgets::number(NumberBinding::RangeLow)),
                one(widgets::number(NumberBinding::RangeHigh)),
            ])),
        )),
    ])
}

/// The brush, and where a drag over the world would land.
///
/// TODO(jb-doc): why the target is shown even though nothing here can choose it, and what a
/// panel that only offered the numbers would leave a person guessing about.
fn brush_section(document: &Document, brush: &BrushSettings, expanded: &Expanded) -> impl Scene {
    let active = document.active().to_owned();
    let target = target_of(document);
    let open = expanded.brush;

    let mode_items: Vec<Box<dyn SceneList>> = BRUSH_MODES
        .into_iter()
        .map(|mode| {
            one(bsn! {
                widgets::item_caption(brush_mode_name(mode))
                on(move |_: On<Activate>, mut brush: ResMut<BrushSettings>| {
                    brush.0.mode = mode;
                })
            })
        })
        .collect();

    let mut body: Vec<Box<dyn SceneList>> = vec![
        one(widgets::captioned(
            "mode",
            one(widgets::menu(brush_mode_name(brush.0.mode), mode_items)),
        )),
        one(widgets::number_row("radius", NumberBinding::BrushRadius)),
        one(widgets::number_row("falloff", NumberBinding::BrushFalloff)),
        one(widgets::number_row(
            "strength",
            NumberBinding::BrushStrength,
        )),
        one(widgets::number_row("value", NumberBinding::BrushValue)),
    ];
    match &target {
        Some((field, index)) => {
            body.push(one(widgets::small(format!(
                "drag paints {field} layer {index}"
            ))));
        }
        None => {
            body.push(one(widgets::small(format!("{active} has no paint layer"))));
            body.push(one(bsn! {
                @FeathersButton {
                    @caption: bsn! { Text("Add paint layer") ThemedText },
                }
                on(move |_: On<Activate>, mut document: ResMut<Document>| {
                    let active = document.active().to_owned();
                    let result = document
                        .apply(&Edit::Add {
                            field: active,
                            op: LayerOp::Paint(Raster::default()),
                        })
                        .map(|_| ());
                    report(&mut document, result);
                })
            }));
        }
    }

    section("brush", open, body, move |open, expanded: &mut Expanded| {
        expanded.brush = open;
    })
}

fn layer_entry(
    active: &str,
    index: usize,
    count: usize,
    layer: &Layer,
    names: &[String],
    open: bool,
) -> impl Scene {
    let active = active.to_owned();
    let title = format!("{index}  {}", op_name(&layer.op));

    let header = widgets::row(vec![
        one(enabled_checkbox(&active, index, layer.enabled)),
        one(move_button(&active, index, index.checked_sub(1), "^")),
        one(move_button(
            &active,
            index,
            (index + 1 < count).then_some(index + 1),
            "v",
        )),
        one(remove_button(&active, index)),
        one(widgets::text(title)),
    ]);

    let body: Vec<Box<dyn SceneList>> = vec![
        one(blend_row(&active, index, layer)),
        one(widgets::number_row(
            "amplitude",
            NumberBinding::Amplitude(index),
        )),
        one(mask_editor(index, &layer.mask, names)),
        one(op_editor(index, &layer.op, names)),
    ];

    widgets::column(vec![
        one(header),
        one(section(
            op_summary(&layer.op),
            open,
            body,
            move |open, expanded: &mut Expanded| expanded.set(index, open),
        )),
    ])
}

fn enabled_checkbox(active: &str, index: usize, enabled: bool) -> impl Scene {
    let active = active.to_owned();
    bsn! {
        {widgets::when(enabled, Checked)}
        @FeathersCheckbox
        on(move |change: On<ValueChange<bool>>, mut document: ResMut<Document>| {
            let result = document
                .apply(&Edit::Toggle {
                    field: active.clone(),
                    index,
                    enabled: Some(change.value),
                })
                .map(|_| ());
            report(&mut document, result);
        })
    }
}

fn move_button(active: &str, index: usize, to: Option<usize>, caption: &'static str) -> impl Scene {
    let active = active.to_owned();
    bsn! {
        @FeathersToolButton {
            @caption: bsn! { Text(caption) ThemedText },
            @variant: ButtonVariant::Plain,
        }
        {widgets::when(to.is_none(), InteractionDisabled)}
        on(move |_: On<Activate>, mut document: ResMut<Document>| {
            let Some(to) = to else {
                return;
            };
            let result = document
                .apply(&Edit::Move {
                    field: active.clone(),
                    index,
                    to,
                })
                .map(|_| ());
            report(&mut document, result);
        })
    }
}

fn remove_button(active: &str, index: usize) -> impl Scene {
    let active = active.to_owned();
    bsn! {
        @FeathersToolButton {
            @caption: bsn! { Text("x") ThemedText },
            @variant: ButtonVariant::Plain,
        }
        on(move |_: On<Activate>, mut document: ResMut<Document>| {
            let result = document
                .apply(&Edit::Remove {
                    field: active.clone(),
                    index,
                })
                .map(|_| ());
            report(&mut document, result);
        })
    }
}

fn blend_row(active: &str, index: usize, layer: &Layer) -> impl Scene {
    let active = active.to_owned();
    let items: Vec<Box<dyn SceneList>> = BLENDS
        .into_iter()
        .map(|blend| {
            let active = active.clone();
            one(bsn! {
                widgets::item_caption(blend_name(blend))
                on(move |_: On<Activate>, mut document: ResMut<Document>| {
                    let result = document
                        .apply(&Edit::Set {
                            path: format!("{active}.{index}.blend"),
                            words: vec![blend_name(blend).to_owned()],
                        })
                        .map(|_| ());
                    report(&mut document, result);
                })
            })
        })
        .collect();

    widgets::captioned("blend", one(widgets::menu(blend_name(layer.blend), items)))
}

/// A painted mask is not offered: there is nothing here that paints one, and a shape that
/// cannot be filled in is worse than one that cannot be chosen.
fn mask_editor(index: usize, mask: &Mask, names: &[String]) -> impl Scene {
    let current = mask_kind(mask);
    let first = first_field(names);

    let items: Vec<Box<dyn SceneList>> = ["constant", "field"]
        .into_iter()
        .map(|kind| {
            let first = first.clone();
            one(bsn! {
                widgets::item_caption(kind)
                on(move |_: On<Activate>, mut document: ResMut<Document>| {
                    let replacement = if kind == "constant" {
                        Mask::Constant(1.0)
                    } else {
                        Mask::Field(first.clone(), Remap::IDENTITY)
                    };
                    let changed = with_layer(&mut document, index, |layer| {
                        if mask_kind(&layer.mask) == kind {
                            return false;
                        }
                        layer.mask = replacement;
                        true
                    });
                    if changed == Some(true) {
                        document.note_edit();
                    }
                })
            })
        })
        .collect();

    let mut rows: Vec<Box<dyn SceneList>> = vec![one(widgets::captioned(
        "mask",
        one(widgets::menu(current, items)),
    ))];
    match mask {
        Mask::Constant(_) => {
            rows.push(one(widgets::number_row(
                "value",
                NumberBinding::MaskConstant(index),
            )));
        }
        Mask::Painted(raster) => {
            rows.push(one(widgets::small(format!(
                "{}x{} painted",
                raster.width(),
                raster.height()
            ))));
        }
        Mask::Field(id, _) => {
            rows.push(one(widgets::captioned(
                "of",
                one(field_menu(id, names, move |document, chosen| {
                    with_layer(document, index, |layer| {
                        let Mask::Field(id, _) = &mut layer.mask else {
                            return false;
                        };
                        *id = chosen;
                        true
                    })
                })),
            )));
            rows.push(one(widgets::captioned(
                "from",
                one(widgets::row(vec![
                    one(widgets::number(NumberBinding::MaskFromLow(index))),
                    one(widgets::number(NumberBinding::MaskFromHigh(index))),
                ])),
            )));
            rows.push(one(widgets::captioned(
                "to",
                one(widgets::row(vec![
                    one(widgets::number(NumberBinding::MaskToLow(index))),
                    one(widgets::number(NumberBinding::MaskToHigh(index))),
                ])),
            )));
        }
    }
    widgets::column(rows)
}

fn op_editor(index: usize, op: &LayerOp, names: &[String]) -> impl Scene {
    let mut rows: Vec<Box<dyn SceneList>> = vec![one(widgets::small(op_name(op)))];

    match op {
        LayerOp::Constant(_) => {
            rows.push(one(widgets::number_row(
                "value",
                NumberBinding::Constant(index),
            )));
        }

        LayerOp::Noise(spec) => {
            let kind_items: Vec<Box<dyn SceneList>> = NOISE_KINDS
                .into_iter()
                .map(|kind| {
                    one(bsn! {
                        widgets::item_caption(noise_kind_name(kind))
                        on(move |_: On<Activate>, mut document: ResMut<Document>| {
                            let changed = with_layer(&mut document, index, |layer| {
                                let LayerOp::Noise(spec) = &mut layer.op else {
                                    return false;
                                };
                                spec.kind = kind;
                                true
                            });
                            if changed == Some(true) {
                                document.note_edit();
                            }
                        })
                    })
                })
                .collect();

            rows.push(one(widgets::captioned(
                "kind",
                one(widgets::menu(noise_kind_name(spec.kind), kind_items)),
            )));
            rows.push(one(widgets::number_row(
                "seed",
                NumberBinding::NoiseSeed(index),
            )));
            // A scale is a reciprocal wavelength, so the useful range spans three decades.
            rows.push(one(widgets::number_row(
                "scale",
                NumberBinding::NoiseScale(index),
            )));
            rows.push(one(widgets::number_row(
                "octaves",
                NumberBinding::NoiseOctaves(index),
            )));
            rows.push(one(widgets::number_row(
                "strike",
                NumberBinding::NoiseStrike(index),
            )));
            rows.push(one(widgets::number_row(
                "aspect",
                NumberBinding::NoiseAspect(index),
            )));
            if spec.warp.is_some() {
                rows.push(one(widgets::captioned(
                    "warp",
                    one(widgets::row(vec![
                        one(widgets::number(NumberBinding::WarpAmplitude(index))),
                        one(widgets::number(NumberBinding::WarpScale(index))),
                        one(widgets::number(NumberBinding::WarpOctaves(index))),
                    ])),
                )));
            }
        }

        LayerOp::Slope { of, mode, .. } => {
            rows.push(one(widgets::captioned(
                "of",
                one(field_menu(of, names, move |document, chosen| {
                    with_layer(document, index, |layer| {
                        let LayerOp::Slope { of, .. } = &mut layer.op else {
                            return false;
                        };
                        *of = chosen;
                        true
                    })
                })),
            )));
            rows.push(one(widgets::number_row(
                "sample tiles",
                NumberBinding::SlopeSampleTiles(index),
            )));

            let mode_items: Vec<Box<dyn SceneList>> = SLOPE_MODES
                .into_iter()
                .map(|candidate| {
                    one(bsn! {
                        widgets::item_caption(slope_mode_name(candidate))
                        on(move |_: On<Activate>, mut document: ResMut<Document>| {
                            let changed = with_layer(&mut document, index, |layer| {
                                let LayerOp::Slope { mode, .. } = &mut layer.op else {
                                    return false;
                                };
                                *mode = candidate;
                                true
                            });
                            if changed == Some(true) {
                                document.note_edit();
                            }
                        })
                    })
                })
                .collect();
            rows.push(one(widgets::captioned(
                "mode",
                one(widgets::menu(slope_mode_name(*mode), mode_items)),
            )));
        }

        LayerOp::FieldRef(id) => {
            rows.push(one(widgets::captioned(
                "of",
                one(field_menu(id, names, move |document, chosen| {
                    with_layer(document, index, |layer| {
                        let LayerOp::FieldRef(id) = &mut layer.op else {
                            return false;
                        };
                        *id = chosen;
                        true
                    })
                })),
            )));
        }

        LayerOp::Regions { spec, output } => {
            let current = region_output_name(output);
            // The two categorical outputs sit under the columns because they are read
            // differently rather than because they are a different kind of thing to choose.
            let output_items: Vec<Box<dyn SceneList>> = spec
                .columns
                .iter()
                .cloned()
                .chain(["region_id".to_owned(), "cover_class".to_owned()])
                .map(|name| {
                    let chosen = name.clone();
                    one(bsn! {
                        widgets::item_caption(name)
                        on(move |_: On<Activate>, mut document: ResMut<Document>| {
                            let picked = parse_region_output(&chosen);
                            let changed = with_layer(&mut document, index, |layer| {
                                let LayerOp::Regions { output, .. } = &mut layer.op else {
                                    return false;
                                };
                                *output = picked;
                                true
                            });
                            if changed == Some(true) {
                                document.note_edit();
                            }
                        })
                    })
                })
                .collect();

            rows.push(one(widgets::captioned(
                "output",
                one(widgets::menu(current, output_items)),
            )));
            rows.push(one(widgets::number_row(
                "seed",
                NumberBinding::RegionSeed(index),
            )));
            rows.push(one(widgets::number_row(
                "cell tiles",
                NumberBinding::RegionCellTiles(index),
            )));
            rows.push(one(widgets::number_row(
                "blend tiles",
                NumberBinding::RegionBlendTiles(index),
            )));

            let mut heading: Vec<Box<dyn SceneList>> = vec![one(widgets::small("weight"))];
            for column in &spec.columns {
                heading.push(one(widgets::small(column.clone())));
            }
            rows.push(one(widgets::row(heading)));

            for (region, values) in spec.regions.iter().enumerate() {
                let mut cells: Vec<Box<dyn SceneList>> = vec![one(widgets::number(
                    NumberBinding::RegionWeight(index, region),
                ))];
                for column in 0..values.values.len() {
                    cells.push(one(widgets::number(NumberBinding::RegionValue(
                        index, region, column,
                    ))));
                }
                rows.push(one(widgets::row(cells)));
            }
        }

        LayerOp::Paint(raster) => {
            rows.push(one(widgets::small(if raster.is_empty() {
                "unpainted".to_owned()
            } else {
                format!("{}x{} painted", raster.width(), raster.height())
            })));
        }

        LayerOp::External(raster) => {
            rows.push(one(widgets::small(format!(
                "{}x{} raster",
                raster.width(),
                raster.height()
            ))));
        }
    }

    widgets::column(rows)
}

fn add_row(active: &str, names: &[String], add: &AddLayer) -> impl Scene {
    let active = active.to_owned();
    let names = names.to_vec();
    let items: Vec<Box<dyn SceneList>> = ADDABLE
        .into_iter()
        .map(|kind| {
            one(bsn! {
                widgets::item_caption(kind)
                on(move |_: On<Activate>, mut add: ResMut<AddLayer>| {
                    add.0 = kind.to_owned();
                })
            })
        })
        .collect();

    let chosen = add.0.clone();
    widgets::row(vec![
        one(widgets::menu(add.0.clone(), items)),
        one(bsn! {
            @FeathersButton {
                @caption: bsn! { Text("Add layer") ThemedText },
            }
            on(move |_: On<Activate>, mut document: ResMut<Document>| {
                let result = document
                    .apply(&Edit::Add {
                        field: active.clone(),
                        op: default_op(&chosen, &names),
                    })
                    .map(|_| ());
                report(&mut document, result);
            })
        }),
    ])
}

/// A collapsible block: a chevron and a caption, and a body that is only built when it is
/// open. Closed sections cost nothing because the shape they are part of says they are shut.
fn section(
    caption: impl Into<String>,
    open: bool,
    body: Vec<Box<dyn SceneList>>,
    toggle: impl Fn(bool, &mut Expanded) + Clone + Send + Sync + 'static,
) -> impl Scene {
    let caption = caption.into();
    let body: Vec<Box<dyn SceneList>> = if open { body } else { Vec::new() };

    bsn! {
        group()
        Children [
            (
                group_header()
                Children [
                    (
                        Node {
                            display: Display::Flex,
                            flex_direction: FlexDirection::Row,
                            align_items: AlignItems::Center,
                            column_gap: px(6),
                        }
                        Children [
                            (
                                {widgets::when(open, Checked)}
                                @FeathersDisclosureToggle
                                on(move |
                                    change: On<ValueChange<bool>>,
                                    mut expanded: ResMut<Expanded>,
                                | {
                                    toggle(change.value, &mut expanded);
                                })
                            ),
                            widgets::small(caption),
                        ]
                    ),
                ]
            ),
            (
                group_body()
                Children [ {body} ]
            ),
        ]
    }
}

fn field_menu(
    current: &FieldId,
    names: &[String],
    write: impl Fn(&mut Document, FieldId) -> Option<bool> + Clone + Send + Sync + 'static,
) -> impl Scene {
    let current = current.to_string();
    let items: Vec<Box<dyn SceneList>> = names
        .iter()
        .map(|name| {
            let chosen = FieldId::from(name.as_str());
            let write = write.clone();
            one(bsn! {
                widgets::item_caption(name.clone())
                on(move |_: On<Activate>, mut document: ResMut<Document>| {
                    if write(&mut document, chosen.clone()) == Some(true) {
                        document.note_edit();
                    }
                })
            })
        })
        .collect();
    widgets::menu(current, items)
}

fn with_layer<R>(
    document: &mut Document,
    index: usize,
    write: impl FnOnce(&mut Layer) -> R,
) -> Option<R> {
    let active = document.active().to_owned();
    let terrain = document.terrain_mut()?;
    let field = terrain.field_mut(&active)?;
    field.layers.get_mut(index).map(write)
}

/// What the add button makes, which has to be something that bakes on its own — a field
/// reference to nothing would put the document in an error state on the frame it was
/// added, and the panel would look like it had refused.
fn default_op(kind: &str, names: &[String]) -> LayerOp {
    match kind {
        "constant" => LayerOp::Constant(0.5),
        "fieldref" => LayerOp::FieldRef(first_field(names)),
        "slope" => LayerOp::Slope {
            of: first_field(names),
            sample_tiles: 4.0,
            mode: SlopeMode::default(),
        },
        // Empty, and sized by the first stroke — see `edit::parse_op`, which is the same
        // decision reached from the other end.
        "paint" => LayerOp::Paint(Raster::default()),
        _ => LayerOp::Noise(NoiseSpec::new(1, NoiseKind::Fbm, 0.02)),
    }
}

fn first_field(names: &[String]) -> FieldId {
    names
        .first()
        .map(|name| FieldId::from(name.as_str()))
        .unwrap_or_else(|| FieldId::from("height"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use watershed::TerrainSpec;
    use watershed::layer::Blend;

    fn document_with(ops: Vec<LayerOp>) -> Document {
        let mut document = Document::default();
        let field = ops
            .into_iter()
            .fold(watershed::Field::new("height"), |field, op| {
                field.with_layer(Layer::new(op))
            });
        document.adopt(TerrainSpec::new(UVec2::splat(64)).with_field(field));
        document
    }

    fn key(document: &Document) -> String {
        fingerprint(
            document,
            &BrushSettings::default(),
            &Expanded::default(),
            &AddLayer::default(),
        )
    }

    /// The rule the panel is built on, from the side that would break it quietly: a choice
    /// that did not change the shape would leave the menu showing the option it used to be
    /// on, with the document already changed underneath it.
    #[test]
    fn a_choice_changes_the_shape_the_panel_is_built_from() {
        let mut document = document_with(vec![LayerOp::Constant(0.5)]);
        let before = key(&document);

        document
            .terrain_mut()
            .unwrap()
            .field_mut("height")
            .unwrap()
            .layers[0]
            .blend = Blend::Mul;
        assert_ne!(key(&document), before, "a blend is a choice");

        document
            .terrain_mut()
            .unwrap()
            .field_mut("height")
            .unwrap()
            .layers[0]
            .enabled = false;
        assert_ne!(key(&document), before, "so is being switched off");
    }

    /// And from the other side: a number in the shape would rebuild the panel on the frame
    /// it was typed into, which throws away the field the keyboard is in.
    #[test]
    fn a_number_does_not_change_the_shape() {
        let mut document = document_with(vec![LayerOp::Noise(NoiseSpec::new(
            1,
            NoiseKind::Fbm,
            0.02,
        ))]);
        let before = key(&document);

        {
            let field = document.terrain_mut().unwrap().field_mut("height").unwrap();
            field.layers[0].amplitude = 0.25;
            field.range = (-1.0, 2.0);
            let LayerOp::Noise(spec) = &mut field.layers[0].op else {
                panic!("the layer stopped being noise");
            };
            spec.seed = 99;
            spec.scale = 0.5;
        }
        assert_eq!(key(&document), before);
    }

    /// A layer added or taken away changes how many widgets there are, which is the one
    /// thing a standing panel cannot absorb.
    #[test]
    fn the_number_of_layers_is_part_of_the_shape() {
        let one = document_with(vec![LayerOp::Constant(0.5)]);
        let two = document_with(vec![LayerOp::Constant(0.5), LayerOp::Constant(0.25)]);
        assert_ne!(key(&one), key(&two));
    }

    /// The shift is a number field or a plain label depending on this, so it decides how
    /// many widgets there are and belongs in the shape however numeric it looks.
    #[test]
    fn whether_the_shift_is_pinned_is_part_of_the_shape() {
        let mut document = document_with(vec![LayerOp::Constant(0.5)]);
        document
            .terrain_mut()
            .unwrap()
            .field_mut("height")
            .unwrap()
            .role = watershed::FieldRole::Height;
        assert!(
            shift_is_pinned(&document),
            "the solve height sits at shift 0"
        );
        let pinned = key(&document);

        document
            .terrain_mut()
            .unwrap()
            .field_mut("height")
            .unwrap()
            .shift = 2;
        assert!(!shift_is_pinned(&document), "a coarse height is not pinned");
        assert_ne!(key(&document), pinned);
    }

    /// An open section holds widgets a shut one does not, so which sections are open is
    /// shape rather than decoration.
    #[test]
    fn opening_a_section_changes_the_shape() {
        let document = document_with(vec![LayerOp::Constant(0.5)]);
        let shut = key(&document);
        let open = fingerprint(
            &document,
            &BrushSettings::default(),
            &Expanded {
                brush: false,
                layers: vec![0],
            },
            &AddLayer::default(),
        );
        assert_ne!(shut, open);
    }
}
