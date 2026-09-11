//! The panel beside the map: the field's own properties, the brush, and the node the
//! canvas has selected.
//!
//! Everything here follows from one rule. **A choice is shape and a number is a
//! value**: choosing an op or a mode changes which widgets exist, so the panel
//! is thrown away and rebuilt; typing a number changes only what a widget holds, so
//! the panel stands and the value is pushed into it. Getting a number into the shape
//! would rebuild the panel under the keyboard on the frame it was typed into, and
//! leaving a choice out of it would leave a menu showing what it used to say over a
//! document that had already changed.

use crate::terrain::graph::{Remap, SlopeMode};
use crate::terrain::noise::{NoiseKind, NoiseSpec};
use crate::terrain::shader::{ParamsLayout, ShaderLayer, Widget};
use bevy::feathers::containers::{group, group_body, group_header};
use bevy::feathers::controls::{
    ButtonVariant, FeathersButton, FeathersCheckbox, FeathersDisclosureToggle, FeathersToolButton,
};
use bevy::feathers::theme::{ThemeBackgroundColor, ThemedText};
use bevy::feathers::tokens;
use bevy::prelude::*;
use bevy::ui::{Checked, InteractionDisabled};
use bevy::ui_widgets::{Activate, ValueChange};
use watershed::raster::Raster;
use watershed::{FieldId, FieldRole};

use crate::brush::{BrushSettings, target_of};
use crate::canvas::Selection;
use crate::document::{Baked, Document};
use crate::edit::{
    BINARIES, BRUSH_MODES, Edit, NOISE_KINDS, SLOPE_MODES, Slot, binary_name, brush_mode_name,
    noise_kind_name, op_name, op_summary, parse_region_output, region_output_name, slope_mode_name,
};
use crate::gpu::{STOCK, ShaderLibrary};
use crate::terrain::graph::{Binary, Curve, GraphNode, NodeId, NodeOp};
use crate::ui::bind::NumberBinding;
use crate::ui::widgets::{self, one};
use crate::ui::{ADDABLE, AddLayer, Expanded, PANEL_WIDTH, report};

/// What the panel was last built from, and which rebuild that was.
///
/// The key covers everything a caption prints and everything that decides how many
/// widgets there are, and nothing a number field holds. The panel is rebuilt exactly
/// when the key changes.
#[derive(Resource, Default)]
pub struct Shape {
    key: String,
    generation: u64,
}

/// Which rebuild a child of the panel belongs to.
///
/// A rebuild queues its children rather than spawning them, so a despawn cannot reach
/// a spawn that has not happened yet and a child from an earlier rebuild can arrive
/// after a later one has already replaced the panel. [`prune`] uses this to take such
/// a child out again.
#[derive(Component, Default, Clone)]
pub struct StackEntry(u64);

/// The panel's scroll container, whose children are the panel's contents.
#[derive(Component, Default, Clone)]
pub struct StackBody;

/// The label that says whether what is on screen is the whole bake or a preview.
#[derive(Component, Default, Clone)]
pub struct PreviewTag;

/// The panel's outer scene: a scrolling column, empty until [`rebuild`] fills it.
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

/// Rebuilds the panel when the shape changes, and does nothing otherwise.
///
/// Always for the field the toolbar has selected, whichever that is — nothing here
/// names a field, so a document's own fields are editable by this panel without it
/// knowing anything about them.
pub fn rebuild(
    document: Res<Document>,
    brush: Res<BrushSettings>,
    expanded: Res<Expanded>,
    add: Res<AddLayer>,
    library: Res<ShaderLibrary>,
    selection: Res<Selection>,
    mut shape: ResMut<Shape>,
    body: Single<Entity, With<StackBody>>,
    mut commands: Commands,
) {
    let key = fingerprint(&document, &brush, &expanded, &add, &library, &selection);
    if shape.key == key {
        return;
    }
    shape.key = key;
    shape.generation += 1;
    let generation = shape.generation;

    let entries: Vec<Box<dyn SceneList>> =
        contents(&document, &brush, &expanded, &add, &library, &selection)
            .into_iter()
            .map(|scene| one(bsn! { {scene} StackEntry({generation}) }))
            .collect();

    commands
        .entity(*body)
        .despawn_related::<Children>()
        .queue_spawn_related_scenes::<Children>(entries);
}

/// Removes children left over from an earlier rebuild — the ones a rebuild could not
/// despawn because they had not been spawned yet. See [`StackEntry`].
pub fn prune(
    shape: Res<Shape>,
    body: Single<&Children, With<StackBody>>,
    entries: Query<&StackEntry>,
    mut commands: Commands,
) {
    for child in body.iter() {
        if entries
            .get(child)
            .is_ok_and(|entry| entry.0 != shape.generation)
        {
            commands.entity(child).despawn();
        }
    }
}

/// Writes the one thing the panel says that is neither a choice nor a number: whether
/// what is on screen is the whole bake or a preview of part of it.
pub fn sync(document: Res<Document>, mut preview: Query<&mut Text, With<PreviewTag>>) {
    let previewing = document.baked() != Baked::Whole || document.is_dirty();
    for mut text in preview.iter_mut() {
        widgets::set_text(&mut text, if previewing { "preview" } else { "" });
    }
}

fn shift_is_pinned(document: &Document) -> bool {
    document
        .terrain()
        .zip(field_of(document))
        .is_some_and(|(terrain, field)| {
            crate::edit::is_solve_height(terrain, document.active()) && field.shift == 0
        })
}

fn field_of(document: &Document) -> Option<&crate::terrain::Field> {
    document.terrain()?.field(document.active())
}

fn fingerprint(
    document: &Document,
    brush: &BrushSettings,
    expanded: &Expanded,
    add: &AddLayer,
    library: &ShaderLibrary,
    selection: &Selection,
) -> String {
    let mut key = String::new();
    key.push_str(document.active());
    key.push('|');
    key.push_str(
        &selection
            .node
            .map_or_else(|| "none".to_owned(), |node| node.to_string()),
    );
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
    key.push_str(&format!("|hillshade:{}", field.hillshade));
    key.push_str(&format!("|contours:{}", field.contours));
    key.push_str(&format!(
        "|out:{}",
        field
            .graph
            .output
            .map_or_else(|| "none".to_owned(), |id| id.to_string())
    ));
    for node in &field.graph.nodes {
        key.push_str(&format!(
            "|{}:{}:{}:{}:{:?}",
            node.id,
            op_name(&node.op),
            node.bypassed,
            expanded.has(node.id),
            node.inputs,
        ));
        match &node.op {
            NodeOp::Noise(spec) => {
                key.push_str(noise_kind_name(spec.kind));
                key.push_str(if spec.warp.is_some() { ":warp" } else { "" });
            }
            NodeOp::Slope { mode, .. } => key.push_str(slope_mode_name(*mode)),
            NodeOp::FieldRef(id) => key.push_str(id.as_ref()),
            NodeOp::Binary(binary) => key.push_str(binary_name(*binary)),
            NodeOp::Curve(curve) => key.push_str(&format!("{}", curve.points.len())),
            NodeOp::Shader(shader) => {
                key.push_str(&shader.file);
                match library.entry(&shader.file) {
                    Some(entry) => {
                        for param in &entry.layout.fields {
                            key.push(':');
                            key.push_str(&param.group);
                            key.push('/');
                            key.push_str(&param.label);
                        }
                        key.push_str(if entry.error.is_some() {
                            ":broken"
                        } else {
                            ":good"
                        });
                    }
                    None => key.push_str(":missing"),
                }
            }
            NodeOp::Regions { spec, output } => {
                key.push_str(&region_output_name(output));
                key.push_str(&spec.columns.join(","));
                key.push_str(&format!(":{}", spec.regions.len()));
            }
            _ => {}
        }
    }
    key
}

fn contents(
    document: &Document,
    brush: &BrushSettings,
    expanded: &Expanded,
    add: &AddLayer,
    library: &ShaderLibrary,
    selection: &Selection,
) -> Vec<Box<dyn Scene>> {
    let active = document.active().to_owned();
    let names = document.field_names();
    let Some(field) = field_of(document) else {
        return vec![widgets::boxed(widgets::text("no document"))];
    };
    let selected = selection
        .node
        .filter(|node| field.graph.node(*node).is_some());

    let mut children: Vec<Box<dyn Scene>> = vec![
        widgets::boxed(widgets::row(vec![
            one(widgets::text(active.clone())),
            one(bsn! { widgets::small("") PreviewTag }),
        ])),
        widgets::boxed(brush_section(document, brush, expanded)),
    ];

    // The field's own properties belong with the node its value is read from: that is
    // where a person looks for what the field is, and with nothing selected there is
    // nothing else the panel could be about.
    if selected.is_none_or(|node| field.graph.output == Some(node)) {
        children.push(widgets::boxed(properties(
            &active,
            field,
            shift_is_pinned(document),
        )));
    }

    match selected.and_then(|node| field.graph.node(node)) {
        Some(node) => children.push(widgets::boxed(node_entry(
            &active,
            node,
            field.graph.output == Some(node.id),
            &names,
            expanded.has(node.id),
            library,
        ))),
        None => children.push(widgets::boxed(widgets::small(
            "select a node on the canvas",
        ))),
    }

    children.push(widgets::boxed(add_row(&active, &names, add)));
    children
}

fn properties(active: &str, field: &crate::terrain::Field, pinned: bool) -> impl Scene {
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
        one(toggle_row(
            &active,
            "hillshade",
            field.hillshade,
            vec![
                one(widgets::small("azimuth")),
                one(widgets::number(NumberBinding::LightAzimuth)),
            ],
        )),
        one(toggle_row(
            &active,
            "contours",
            field.contours,
            vec![
                one(widgets::small("interval")),
                one(widgets::number(NumberBinding::ContourInterval)),
            ],
        )),
    ])
}

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
                    @caption: bsn! { Text("Add paint node") ThemedText },
                }
                on(move |_: On<Activate>, mut document: ResMut<Document>, selection: Res<Selection>| {
                    let active = document.active().to_owned();
                    let at = field_of(&document)
                        .map(|field| field.graph.free_position_beside(selection.node));
                    let result = document
                        .apply(&Edit::AddNode {
                            field: active,
                            op: NodeOp::Paint(Raster::default()),
                            position: at,
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

fn node_entry(
    active: &str,
    node: &GraphNode,
    is_output: bool,
    names: &[String],
    open: bool,
    library: &ShaderLibrary,
) -> impl Scene {
    let active = active.to_owned();
    let id = node.id;
    let title = match &node.name {
        Some(name) => format!("{id}  {name}"),
        None => format!("{id}  {}", op_name(&node.op)),
    };

    let header = widgets::row(vec![
        one(bypass_checkbox(&active, id, node.bypassed)),
        one(output_button(&active, id, is_output)),
        one(remove_button(&active, id)),
        one(widgets::text(title)),
    ]);

    let mut body: Vec<Box<dyn SceneList>> = Vec::new();
    if !node.inputs.is_empty() {
        body.push(one(widgets::small(inputs_caption(node))));
    }
    body.push(one(op_editor(id, &node.op, names, library)));

    widgets::column(vec![
        one(header),
        one(section(
            op_summary(&node.op),
            open,
            body,
            move |open, expanded: &mut Expanded| expanded.set(id, open),
        )),
    ])
}

/// What is wired into a node, as one line. Read-only: an edge is drawn on the canvas
/// or written by a control verb, not picked from a menu here.
fn inputs_caption(node: &GraphNode) -> String {
    let pins: Vec<String> = node
        .inputs
        .iter()
        .map(|pin| match pin {
            Some(source) => source.to_string(),
            None => "-".to_owned(),
        })
        .collect();
    format!("inputs {}", pins.join(" "))
}

fn bypass_checkbox(active: &str, id: NodeId, bypassed: bool) -> impl Scene {
    let active = active.to_owned();
    bsn! {
        {widgets::when(!bypassed, Checked)}
        @FeathersCheckbox
        on(move |change: On<ValueChange<bool>>, mut document: ResMut<Document>| {
            let result = document
                .apply(&Edit::Bypass {
                    field: active.clone(),
                    node: id.to_string(),
                    bypassed: Some(!change.value),
                })
                .map(|_| ());
            report(&mut document, result);
        })
    }
}

fn toggle_row(
    active: &str,
    property: &'static str,
    checked: bool,
    beside: Vec<Box<dyn SceneList>>,
) -> impl Scene {
    let active = active.to_owned();
    let mut children: Vec<Box<dyn SceneList>> = vec![
        one(bsn! {
            {widgets::when(checked, Checked)}
            @FeathersCheckbox
            on(move |change: On<ValueChange<bool>>, mut document: ResMut<Document>| {
                let result = document
                    .apply(&Edit::Set {
                        path: format!("{active}.{property}"),
                        words: vec![if change.value { "on" } else { "off" }.to_owned()],
                    })
                    .map(|_| ());
                report(&mut document, result);
            })
        }),
        one(widgets::small(property)),
    ];
    children.extend(beside);
    widgets::row(children)
}

fn output_button(active: &str, id: NodeId, is_output: bool) -> impl Scene {
    let active = active.to_owned();
    bsn! {
        @FeathersToolButton {
            @caption: bsn! { Text("=") ThemedText },
            @variant: ButtonVariant::Plain,
        }
        {widgets::when(is_output, InteractionDisabled)}
        on(move |_: On<Activate>, mut document: ResMut<Document>| {
            let result = document
                .apply(&Edit::SetOutput {
                    field: active.clone(),
                    node: Some(id.to_string()),
                })
                .map(|_| ());
            report(&mut document, result);
        })
    }
}

fn remove_button(active: &str, id: NodeId) -> impl Scene {
    let active = active.to_owned();
    bsn! {
        @FeathersToolButton {
            @caption: bsn! { Text("x") ThemedText },
            @variant: ButtonVariant::Plain,
        }
        on(move |_: On<Activate>, mut document: ResMut<Document>| {
            let result = document
                .apply(&Edit::RemoveNode {
                    field: active.clone(),
                    node: id.to_string(),
                })
                .map(|_| ());
            report(&mut document, result);
        })
    }
}

fn op_editor(id: NodeId, op: &NodeOp, names: &[String], library: &ShaderLibrary) -> impl Scene {
    let mut rows: Vec<Box<dyn SceneList>> = vec![one(widgets::small(op_name(op)))];

    match op {
        NodeOp::Constant(_) => {
            rows.push(one(widgets::number_row(
                "value",
                NumberBinding::Constant(id),
            )));
        }

        NodeOp::Noise(spec) => {
            let kind_items: Vec<Box<dyn SceneList>> = NOISE_KINDS
                .into_iter()
                .map(|kind| {
                    one(bsn! {
                        widgets::item_caption(noise_kind_name(kind))
                        on(move |_: On<Activate>, mut document: ResMut<Document>| {
                            with_op(&mut document, id, move |op| {
                                let NodeOp::Noise(spec) = op else {
                                    return;
                                };                                spec.kind = kind;
                            });
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
                NumberBinding::NoiseSeed(id),
            )));
            rows.push(one(widgets::number_row(
                "scale",
                NumberBinding::NoiseScale(id),
            )));
            rows.push(one(widgets::number_row(
                "octaves",
                NumberBinding::NoiseOctaves(id),
            )));
            rows.push(one(widgets::number_row(
                "strike",
                NumberBinding::NoiseStrike(id),
            )));
            rows.push(one(widgets::number_row(
                "aspect",
                NumberBinding::NoiseAspect(id),
            )));
            if spec.warp.is_some() {
                rows.push(one(widgets::captioned(
                    "warp",
                    one(widgets::row(vec![
                        one(widgets::number(NumberBinding::WarpAmplitude(id))),
                        one(widgets::number(NumberBinding::WarpScale(id))),
                        one(widgets::number(NumberBinding::WarpOctaves(id))),
                    ])),
                )));
            }
        }

        NodeOp::Slope { mode, .. } => {
            rows.push(one(widgets::number_row(
                "sample tiles",
                NumberBinding::SlopeSampleTiles(id),
            )));

            let mode_items: Vec<Box<dyn SceneList>> = SLOPE_MODES
                .into_iter()
                .map(|candidate| {
                    one(bsn! {
                        widgets::item_caption(slope_mode_name(candidate))
                        on(move |_: On<Activate>, mut document: ResMut<Document>| {
                            with_op(&mut document, id, move |op| {
                                let NodeOp::Slope { mode, .. } = op else {
                                    return;
                                };                                *mode = candidate;
                            });
                        })
                    })
                })
                .collect();
            rows.push(one(widgets::captioned(
                "mode",
                one(widgets::menu(slope_mode_name(*mode), mode_items)),
            )));
        }

        NodeOp::FieldRef(read) => {
            rows.push(one(widgets::captioned(
                "of",
                one(field_menu(read, names, move |document, chosen| {
                    with_op(document, id, move |op| {
                        if let NodeOp::FieldRef(held) = op {
                            *held = chosen;
                        }
                    });
                })),
            )));
        }

        NodeOp::Regions { spec, output } => {
            let current = region_output_name(output);
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
                            with_op(&mut document, id, move |op| {
                                let NodeOp::Regions { output, .. } = op else {
                                    return;
                                };                                *output = picked;
                            });
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
                NumberBinding::RegionSeed(id),
            )));
            rows.push(one(widgets::number_row(
                "cell tiles",
                NumberBinding::RegionCellTiles(id),
            )));
            rows.push(one(widgets::number_row(
                "blend tiles",
                NumberBinding::RegionBlendTiles(id),
            )));

            let mut heading: Vec<Box<dyn SceneList>> = vec![one(widgets::small("weight"))];
            for column in &spec.columns {
                heading.push(one(widgets::small(column.clone())));
            }
            rows.push(one(widgets::row(heading)));

            for (region, values) in spec.regions.iter().enumerate() {
                let mut cells: Vec<Box<dyn SceneList>> = vec![one(widgets::number(
                    NumberBinding::RegionWeight(id, region),
                ))];
                for column in 0..values.values.len() {
                    cells.push(one(widgets::number(NumberBinding::RegionValue(
                        id, region, column,
                    ))));
                }
                rows.push(one(widgets::row(cells)));
            }
        }

        NodeOp::Binary(binary) => {
            let items: Vec<Box<dyn SceneList>> = BINARIES
                .into_iter()
                .map(|candidate| {
                    one(bsn! {
                        widgets::item_caption(binary_name(candidate))
                        on(move |_: On<Activate>, mut document: ResMut<Document>| {
                            with_op(&mut document, id, move |op| {
                                let NodeOp::Binary(held) = op else {
                                    return;
                                };                                *held = candidate;
                            });
                        })
                    })
                })
                .collect();
            rows.push(one(widgets::captioned(
                "mode",
                one(widgets::menu(binary_name(*binary), items)),
            )));
        }

        NodeOp::Lerp => {}

        NodeOp::Scale(_) => {
            rows.push(one(widgets::number_row(
                "factor",
                NumberBinding::ScaleFactor(id),
            )));
        }

        NodeOp::Remap(_) => {
            rows.push(one(widgets::captioned(
                "from",
                one(widgets::row(vec![
                    one(widgets::number(NumberBinding::RemapFromLow(id))),
                    one(widgets::number(NumberBinding::RemapFromHigh(id))),
                ])),
            )));
            rows.push(one(widgets::captioned(
                "to",
                one(widgets::row(vec![
                    one(widgets::number(NumberBinding::RemapToLow(id))),
                    one(widgets::number(NumberBinding::RemapToHigh(id))),
                ])),
            )));
        }

        NodeOp::Curve(curve) => {
            rows.push(one(widgets::small(format!(
                "{} points",
                curve.points.len()
            ))));
        }

        NodeOp::Paint(raster) => {
            rows.push(one(widgets::small(if raster.is_empty() {
                "unpainted".to_owned()
            } else {
                format!("{}x{} painted", raster.width(), raster.height())
            })));
        }

        NodeOp::External(raster) => {
            rows.push(one(widgets::small(format!(
                "{}x{} raster",
                raster.width(),
                raster.height()
            ))));
        }

        NodeOp::Shader(shader) => {
            rows.push(one(widgets::small(shader.file.clone())));
            match library.entry(&shader.file) {
                None => rows.push(one(widgets::small("no such shader in this document"))),
                Some(entry) => {
                    if let Some(error) = &entry.error {
                        rows.push(one(widgets::small(error.clone())));
                    }
                    rows.extend(param_rows(id, shader, &entry.layout));
                }
            }
        }
    }

    widgets::column(rows)
}

/// One row per parameter the shader declares, in declaration order, with a heading
/// wherever the `@group` changes.
///
/// A parameter is addressed by its position in the layer's own key order rather than
/// by name, which is what lets a binding stay `Copy`; the two orders are put back
/// together here, where the layout is in hand.
fn param_rows(id: NodeId, shader: &ShaderLayer, layout: &ParamsLayout) -> Vec<Box<dyn SceneList>> {
    let mut rows: Vec<Box<dyn SceneList>> = Vec::new();
    let mut group = String::new();
    for field in &layout.fields {
        if matches!(field.widget, Widget::Hidden) {
            continue;
        }
        if field.group != group {
            group = field.group.clone();
            if !group.is_empty() {
                rows.push(one(widgets::small(group.clone())));
            }
        }
        let Some(param) = shader.params.keys().position(|name| *name == field.name) else {
            continue;
        };
        let components = field.ty.components();
        for component in 0..components {
            let caption = if components == 1 {
                field.label.clone()
            } else {
                format!("{} {}", field.label, "xyzw".as_bytes()[component] as char)
            };
            rows.push(one(widgets::number_row(
                caption,
                NumberBinding::ShaderParam(id, param, component),
            )));
        }
    }
    rows
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
                @caption: bsn! { Text("Add node") ThemedText },
            }
            on(move |_: On<Activate>, mut document: ResMut<Document>, mut library: ResMut<ShaderLibrary>, selection: Res<Selection>| {
                let at = field_of(&document)
                    .map(|field| field.graph.free_position_beside(selection.node));
                let op = match stock_of(&chosen) {
                    Some(stock) => match library.adopt(stock) {
                        Ok(file) => NodeOp::Shader(ShaderLayer::new(file)),
                        Err(error) => {
                            report(&mut document, Err(error));
                            return;
                        }
                    },
                    None => default_op(&chosen, &names),
                };
                let result = document
                    .apply(&Edit::AddNode {
                        field: active.clone(),
                        op,
                        position: at,
                    })
                    .map(|_| ());
                report(&mut document, result);
            })
        }),
    ])
}

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
    write: impl Fn(&mut Document, FieldId) + Clone + Send + Sync + 'static,
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
                    write(&mut document, chosen.clone());
                })
            })
        })
        .collect();
    widgets::menu(current, items)
}

fn with_op(
    document: &mut Document,
    id: NodeId,
    write: impl FnOnce(&mut NodeOp) + Send + Sync + 'static,
) {
    let active = document.active().to_owned();
    document.write(&active, Slot::Once, move |field| {
        if let Some(node) = field.graph.node_mut(id) {
            write(&mut node.op);
        }
    });
}

/// The stock shader a `shader:` entry of [`ADDABLE`] names, or `None` for an entry
/// that is an op word rather than a shader.
fn stock_of(chosen: &str) -> Option<&'static str> {
    let name = chosen.strip_prefix("shader:")?;
    let file = if name == "blank" {
        "_template.wgsl".to_owned()
    } else {
        format!("{name}.wgsl")
    };
    STOCK
        .iter()
        .find(|(stock, _)| *stock == file)
        .map(|(stock, _)| *stock)
}

fn default_op(kind: &str, names: &[String]) -> NodeOp {
    match kind {
        "constant" => NodeOp::Constant(0.5),
        "fieldref" => NodeOp::FieldRef(first_field(names)),
        "slope" => NodeOp::Slope {
            sample_tiles: 4.0,
            mode: SlopeMode::default(),
        },
        "paint" => NodeOp::Paint(Raster::default()),
        "binary" => NodeOp::Binary(Binary::default()),
        "lerp" => NodeOp::Lerp,
        "scale" => NodeOp::Scale(1.0),
        "remap" => NodeOp::Remap(Remap::IDENTITY),
        "curve" => NodeOp::Curve(Curve::default()),
        _ => NodeOp::Noise(NoiseSpec::new(1, NoiseKind::Fbm, 0.02)),
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
    use crate::terrain::TerrainSpec;
    use crate::terrain::graph::Binary;

    fn document_with(ops: Vec<NodeOp>) -> Document {
        let mut document = Document::default();
        let field = ops
            .into_iter()
            .fold(crate::terrain::Field::new("height"), |field, op| {
                field.with_op(op)
            });
        document.adopt(TerrainSpec::new(UVec2::splat(64)).with_field(field));
        document
    }

    fn selecting(document: &Document) -> Selection {
        let mut selection = Selection::default();
        selection.select(
            document
                .terrain()
                .and_then(|terrain| terrain.field("height"))
                .and_then(|field| field.graph.nodes.first())
                .map(|node| node.id),
        );
        selection
    }

    fn key(document: &Document) -> String {
        fingerprint(
            document,
            &BrushSettings::default(),
            &Expanded::default(),
            &AddLayer::default(),
            &ShaderLibrary::default(),
            &selecting(document),
        )
    }

    // The rule the panel is built on, from the side that would break it quietly: a
    // choice that did not change the shape would leave the menu showing the option it
    // used to be on, with the document already changed underneath it.
    #[test]
    fn a_choice_changes_the_shape_the_panel_is_built_from() {
        let mut document = document_with(vec![NodeOp::Constant(0.5)]);
        let before = key(&document);

        let node = {
            let field = document.terrain_mut().unwrap().field_mut("height").unwrap();
            let node = field.graph.nodes[0].id;
            field.graph.node_mut(node).unwrap().op = NodeOp::Binary(Binary::Mul);
            node
        };
        assert_ne!(key(&document), before, "the op a node carries is a choice");

        document
            .terrain_mut()
            .unwrap()
            .field_mut("height")
            .unwrap()
            .graph
            .set_bypassed(node, true)
            .unwrap();
        assert_ne!(key(&document), before, "so is being bypassed");
    }

    // And from the other side: a number in the shape would rebuild the panel on the
    // frame it was typed into, which throws away the field the keyboard is in.
    #[test]
    fn a_number_does_not_change_the_shape() {
        let mut document =
            document_with(vec![NodeOp::Noise(NoiseSpec::new(1, NoiseKind::Fbm, 0.02))]);
        let before = key(&document);

        {
            let field = document.terrain_mut().unwrap().field_mut("height").unwrap();
            field.range = (-1.0, 2.0);
            let node = field.graph.nodes[0].id;
            let NodeOp::Noise(spec) = &mut field.graph.node_mut(node).unwrap().op else {
                panic!("the node stopped being noise");
            };
            spec.seed = 99;
            spec.scale = 0.5;
        }
        assert_eq!(key(&document), before);
    }

    // A node added or taken away changes how many widgets there are, which is the one
    // thing a standing panel cannot absorb.
    #[test]
    fn the_number_of_nodes_is_part_of_the_shape() {
        let one = document_with(vec![NodeOp::Constant(0.5)]);
        let two = document_with(vec![NodeOp::Constant(0.5), NodeOp::Constant(0.25)]);
        assert_ne!(key(&one), key(&two));
    }

    // The shift is a number field or a plain label depending on this, so it decides how
    // many widgets there are and belongs in the shape however numeric it looks. It is
    // pinned only while the height field is already at shift 0, so a document that
    // arrived coarse some other way can still be repaired from the panel.
    #[test]
    fn whether_the_shift_is_pinned_is_part_of_the_shape() {
        let mut document = document_with(vec![NodeOp::Constant(0.5)]);
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

    // An open section holds widgets a shut one does not, so which sections are open is
    // shape rather than decoration.
    #[test]
    fn opening_a_section_changes_the_shape() {
        let document = document_with(vec![NodeOp::Constant(0.5)]);
        let opened = document
            .terrain()
            .unwrap()
            .field("height")
            .unwrap()
            .graph
            .nodes[0]
            .id;
        let shut = key(&document);
        let open = fingerprint(
            &document,
            &BrushSettings::default(),
            &Expanded {
                brush: false,
                nodes: vec![opened],
            },
            &AddLayer::default(),
            &ShaderLibrary::default(),
            &selecting(&document),
        );
        assert_ne!(shut, open);
    }

    // The panel is an inspector for whatever the canvas has selected, so the selection
    // decides which widgets exist and therefore belongs in the shape. Deselecting is
    // the case that matters: the panel becomes the field's own properties, which is a
    // different set of widgets rather than the same ones showing nothing.
    #[test]
    fn what_is_selected_is_part_of_the_shape() {
        let document = document_with(vec![NodeOp::Constant(0.5)]);
        let selected = key(&document);
        let deselected = fingerprint(
            &document,
            &BrushSettings::default(),
            &Expanded::default(),
            &AddLayer::default(),
            &ShaderLibrary::default(),
            &Selection::default(),
        );
        assert_ne!(selected, deselected);
    }
}
