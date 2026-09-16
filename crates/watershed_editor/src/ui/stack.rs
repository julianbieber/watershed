//! The panel beside the map: what the editor shows about the document, and what it
//! lets a person change anywhere but on the canvas itself.
//!
//! Everything here follows from one rule. **A choice is shape and a number is a
//! value**: choosing a role or a mode changes which widgets exist, so the panel
//! is thrown away and rebuilt; typing a number changes only what a widget holds, so
//! the panel stands and the value is pushed into it. Getting a number into the shape
//! would rebuild the panel under the keyboard on the frame it was typed into, and
//! leaving a choice out of it would leave a menu showing what it used to say over a
//! document that had already changed.

use bevy::feathers::containers::{group, group_body, group_header};
use bevy::feathers::controls::{
    FeathersButton, FeathersCheckbox, FeathersDisclosureToggle, FeathersTextInput,
    FeathersTextInputContainer,
};
use bevy::feathers::theme::{ThemeBackgroundColor, ThemedText};
use bevy::feathers::tokens;
use bevy::prelude::*;
use bevy::text::{EditableText, TextEdit, TextEditChange};
use bevy::ui::Checked;
use bevy::ui_widgets::{Activate, ValueChange};

use std::path::Path;

use crate::canvas::OpenLayer;
use crate::document::{Baked, Document};
use crate::edit::Edit;
use crate::gpu::{ShaderLibrary, shader_reference};
use crate::terrain::Layer;
use crate::terrain::shader::{ParamsLayout, ShaderLayer, Widget};
use crate::ui::bind::NumberBinding;
use crate::ui::widgets::{self, one};
use crate::ui::{Expanded, NewLayer, PANEL_WIDTH, report};

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

/// The text field holding the name the "Add layer" button will use.
#[derive(Component, Default, Clone)]
pub struct NewLayerInput;

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
/// Always for the layer the toolbar has selected, whichever that is — nothing here
/// names a layer, so a document's own layers are editable by this panel without it
/// knowing anything about them.
pub fn rebuild(
    document: Res<Document>,
    expanded: Res<Expanded>,
    library: Res<ShaderLibrary>,
    mut shape: ResMut<Shape>,
    body: Single<Entity, With<StackBody>>,
    mut commands: Commands,
) {
    let key = fingerprint(&document, &expanded, &library);
    if shape.key == key {
        return;
    }
    shape.key = key;
    shape.generation += 1;
    let generation = shape.generation;

    let entries: Vec<Box<dyn SceneList>> = contents(&document, &expanded, &library)
        .into_iter()
        .map(|scene| one(bsn! { @{scene} StackEntry({generation}) }))
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

fn layer_of(document: &Document) -> Option<&Layer> {
    document.terrain()?.layer(document.active())
}

fn fingerprint(document: &Document, expanded: &Expanded, library: &ShaderLibrary) -> String {
    let mut key = String::new();
    key.push_str(document.active());
    key.push('|');
    key.push_str(&document.layer_names().join(","));
    key.push('|');
    key.push_str(if expanded.reference { "ref" } else { "noref" });
    key.push('|');
    key.push_str(
        &document
            .path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
    );
    key.push('|');

    if let Some(terrain) = document.terrain() {
        key.push_str(&crate::edit::readers_of(terrain, document.active()).join(","));
    }
    key.push('|');

    let Some(layer) = layer_of(document) else {
        return key + "empty";
    };
    key.push_str(layer.role.as_str());
    key.push_str(&format!("|shift:{}", layer.shift));
    let (low, high) = layer.bounds();
    key.push_str(&format!("|range:{low}..{high}"));
    key.push_str(&format!("|categorical:{}", layer.categorical));
    key.push_str(&format!("|hillshade:{}", layer.hillshade));
    key.push_str(&format!("|contours:{}", layer.contours));
    key.push('|');
    match library.entry(&layer.file()) {
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
    key
}

fn contents(
    document: &Document,
    expanded: &Expanded,
    library: &ShaderLibrary,
) -> Vec<Box<dyn Scene>> {
    let active = document.active().to_owned();
    let Some(layer) = layer_of(document) else {
        return vec![widgets::boxed(widgets::text("no document"))];
    };

    let mut children: Vec<Box<dyn Scene>> = vec![widgets::boxed(widgets::row(vec![
        one(widgets::text(active.clone())),
        one(bsn! { @widgets::small("") PreviewTag }),
    ]))];

    let reads = crate::edit::reads_of(layer);
    let read_by = document
        .terrain()
        .map(|terrain| crate::edit::readers_of(terrain, &active))
        .unwrap_or_default();
    children.push(widgets::boxed(properties(&active, layer, &reads, &read_by)));
    let root = document.shader_root();
    children.push(widgets::boxed(shader_section(layer, library, &root)));
    children.push(widgets::boxed(reference_section(expanded.reference)));
    children.push(widgets::boxed(layer_row(&active)));
    children
}

fn layer_row(active: &str) -> impl Scene {
    let remove = active.to_owned();
    widgets::column(vec![
        one(widgets::row(vec![
            one(bsn! {
                @FeathersTextInputContainer
                Node { width: px(120) }
                Children [
                    @FeathersTextInput
                    NewLayerInput
                    on(|change: On<TextEditChange>,
                        texts: Query<&EditableText>,
                        mut name: ResMut<NewLayer>| {
                        if let Ok(text) = texts.get(change.event_target()) {
                            name.0 = text.value().to_string();
                        }
                    })
                ]
            }),
            one(bsn! {
                @FeathersButton {
                    @caption: bsn! { Text("Add layer") ThemedText },
                }
                on(|_: On<Activate>, mut document: ResMut<Document>, mut name: ResMut<NewLayer>| {
                    let result = document
                        .apply(&Edit::AddLayer { name: name.0.clone() })
                        .map(|_| ());
                    if result.is_ok() {
                        name.0.clear();
                    }
                    report(&mut document, result);
                })
            }),
        ])),
        one(bsn! {
            @FeathersButton {
                @caption: bsn! { Text("Remove layer") ThemedText },
            }
            on(move |_: On<Activate>, mut document: ResMut<Document>| {
                let result = document
                    .apply(&Edit::RemoveLayer {
                        name: remove.clone(),
                    })
                    .map(|_| ());
                report(&mut document, result);
            })
        }),
    ])
}

/// Puts the name held in [`NewLayer`] into the text field the frame it appears, so a
/// half-typed name survives the panel being rebuilt under it.
pub fn seed_layer_name(
    name: Res<NewLayer>,
    mut inputs: Query<&mut EditableText, Added<NewLayerInput>>,
) {
    for mut text in inputs.iter_mut() {
        text.queue_edit(TextEdit::SelectAll);
        text.queue_edit(TextEdit::Insert(name.0.clone().into()));
    }
}

fn layer_links(caption: &str, names: &[String]) -> impl Scene {
    let mut children: Vec<Box<dyn SceneList>> = vec![one(widgets::small(caption.to_owned()))];
    if names.is_empty() {
        children.push(one(widgets::small("none")));
    }
    for name in names {
        let layer = name.clone();
        children.push(one(bsn! {
            @FeathersButton {
                @caption: bsn! { Text({name.clone()}) ThemedText },
            }
            on(move |_: On<Activate>, mut open: MessageWriter<OpenLayer>| {
                open.write(OpenLayer {
                    layer: layer.clone(),
                });
            })
        }));
    }
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            flex_wrap: FlexWrap::Wrap,
            column_gap: px(4),
            row_gap: px(4),
        }
        Children [ {children} ]
    }
}

fn properties(active: &str, layer: &Layer, reads: &[String], read_by: &[String]) -> impl Scene {
    let active = active.to_owned();
    let (low, high) = layer.bounds();

    widgets::column(vec![
        one(widgets::captioned(
            "role",
            one(widgets::small(layer.role.as_str())),
        )),
        one(widgets::captioned(
            "shift",
            one(widgets::small(layer.shift.to_string())),
        )),
        one(widgets::captioned(
            "range",
            one(widgets::small(format!("{low} {high}"))),
        )),
        one(widgets::captioned(
            "class",
            one(widgets::small(if layer.categorical {
                "categorical"
            } else {
                "quantity"
            })),
        )),
        one(toggle_row(
            &active,
            "hillshade",
            layer.hillshade,
            vec![
                one(widgets::small("azimuth")),
                one(widgets::number(NumberBinding::LightAzimuth)),
            ],
        )),
        one(toggle_row(
            &active,
            "contours",
            layer.contours,
            vec![
                one(widgets::small("interval")),
                one(widgets::number(NumberBinding::ContourInterval)),
            ],
        )),
        one(layer_links("reads", reads)),
        one(layer_links("read by", read_by)),
    ])
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
            @{widgets::when(checked, Checked)}
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

fn shader_section(layer: &Layer, library: &ShaderLibrary, root: &Path) -> impl Scene {
    let file = layer.file();
    let path = root.join(&file);
    let open = path.clone();
    let mut rows: Vec<Box<dyn SceneList>> = vec![
        one(widgets::small(path.display().to_string())),
        one(bsn! {
            @FeathersButton {
                @caption: bsn! { Text("Open") ThemedText },
            }
            on(move |_: On<Activate>, mut document: ResMut<Document>| {
                let result = crate::open::open(&open).map(|_| ());
                report(&mut document, result);
            })
        }),
    ];
    match library.entry(&file) {
        None => rows.push(one(widgets::small("no such file in shaders/"))),
        Some(entry) => {
            if let Some(error) = &entry.error {
                rows.push(one(widgets::small(error.clone())));
            }
            rows.extend(param_rows(&layer.shader, &entry.layout));
        }
    }
    widgets::column(rows)
}

fn param_rows(shader: &ShaderLayer, layout: &ParamsLayout) -> Vec<Box<dyn SceneList>> {
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
                NumberBinding::ShaderParam(param, component),
            )));
        }
    }
    rows
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
        @group()
        Children [
            @group_header()
            Children [
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(6),
                }
                Children [
                    @{widgets::when(open, Checked)}
                    @FeathersDisclosureToggle
                    on(move |
                        change: On<ValueChange<bool>>,
                        mut expanded: ResMut<Expanded>,
                    | {
                        toggle(change.value, &mut expanded);
                    })
                    --
                    @widgets::small(caption)
                ]
            ]
            --
            @group_body()
            Children [ {body} ]
        ]
    }
}

const REFERENCE_HEIGHT: f32 = 260.0;

fn reference_section(open: bool) -> impl Scene {
    section(
        "Reference",
        open,
        vec![one(reference_body())],
        |open, expanded: &mut Expanded| expanded.reference = open,
    )
}

fn reference_body() -> impl Scene {
    let reference = shader_reference();
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Stretch,
            max_height: {px(REFERENCE_HEIGHT)},
            overflow: {Overflow::scroll_y()},
        }
        Children [ @widgets::small(reference) ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::LayerId;
    use crate::terrain::LayerRole;
    use crate::terrain::TerrainSpec;

    fn document_with(value: f32) -> Document {
        let mut document = Document::default();
        document
            .adopt(TerrainSpec::new(UVec2::splat(64)).with_layer(Layer::new("height").held(value)));
        document
    }

    fn key(document: &Document) -> String {
        fingerprint(document, &Expanded::default(), &ShaderLibrary::default())
    }

    fn height(document: &mut Document) -> &mut Layer {
        document.terrain_mut().unwrap().layer_mut("height").unwrap()
    }

    // The `read by` row is derived from every *other* layer's shader, so retargeting a
    // reader elsewhere changes neither the layer names nor the layer on screen. Without
    // the readers in the key the row would keep naming the old reader.
    #[test]
    fn retargeting_another_layers_read_rebuilds_the_panel() {
        let mut document = Document::default();
        document.adopt(
            TerrainSpec::new(UVec2::splat(64))
                .with_layer(Layer::new("height").held(0.5))
                .with_layer(Layer::new("other").held(0.25))
                .with_layer(Layer::new("reader").held(0.0).reading(&["height"])),
        );
        document.set_active("height").unwrap();
        let before = key(&document);

        document
            .terrain_mut()
            .unwrap()
            .layer_mut("reader")
            .unwrap()
            .shader
            .layers = vec![LayerId::from("other")];
        assert_ne!(key(&document), before, "`read by` went stale");
    }

    // The path is a caption now, so a save that moves the shader directory has to
    // rebuild the panel — nothing else about the document would ever redraw it.
    #[test]
    fn saving_the_document_elsewhere_rebuilds_the_panel() {
        let mut document = document_with(0.5);
        let before = key(&document);

        document.path = Some(std::path::PathBuf::from("/tmp/watershed-test"));
        assert_ne!(key(&document), before, "the path went stale");
    }

    // The rule the panel is built on, from the side that would break it quietly: a
    // choice that did not change the shape would leave the menu showing the option it
    // used to be on, with the document already changed underneath it.
    #[test]
    fn a_choice_changes_the_shape_the_panel_is_built_from() {
        let mut document = document_with(0.5);
        let before = key(&document);

        let layer = height(&mut document);
        layer.role = LayerRole::ALL
            .into_iter()
            .find(|role| *role != layer.role)
            .unwrap();
        assert_ne!(key(&document), before, "the role is a choice");

        let after_role = key(&document);
        let layer = height(&mut document);
        layer.hillshade = !layer.hillshade;
        assert_ne!(key(&document), after_role, "so is hillshading");
    }

    // And from the other side: a number in the shape would rebuild the panel on the
    // frame it was typed into, which throws away the layer the keyboard is in.
    #[test]
    fn a_number_does_not_change_the_shape() {
        let mut document = document_with(0.5);
        let before = key(&document);

        height(&mut document)
            .shader
            .params
            .get_mut("value")
            .expect("a held value")[0] = 0.9;
        assert_eq!(key(&document), before);
    }

    // The four the file declares are drawn as text, so a file edited under the panel
    // has to rebuild it: nothing else would ever redraw the label.
    #[test]
    fn the_properties_the_file_declares_are_part_of_the_shape() {
        let mut document = document_with(0.5);
        let before = key(&document);

        height(&mut document).shift = 2;
        let after_shift = key(&document);
        assert_ne!(after_shift, before, "the shift is a label");

        height(&mut document).range = (0.0, 2.0);
        let after_range = key(&document);
        assert_ne!(after_range, after_shift, "so is the range");

        height(&mut document).categorical = true;
        assert_ne!(key(&document), after_range, "so is the class");
    }

    // The toggle is a choice, and a choice is shape: a Reference flag outside the key
    // would leave the panel showing what it showed before the toggle was pressed.
    #[test]
    fn opening_the_reference_changes_the_shape() {
        let document = document_with(0.5);
        let shut = key(&document);
        let open = fingerprint(
            &document,
            &Expanded { reference: true },
            &ShaderLibrary::default(),
        );
        assert_ne!(shut, open);
    }

    // The acceptance, as far as it can be asserted without a window: the text the
    // section renders is what carries the signature, so the panel is empty of it the
    // day the header drops it.
    #[test]
    fn the_reference_carries_the_ridged_fbm_signature() {
        assert!(
            shader_reference()
                .contains("ridged_fbm(p, octaves: u32, persistence: f32, lacunarity: f32) -> f32")
        );
    }
}
