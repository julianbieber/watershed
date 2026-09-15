//! The strip along the top: the actions that apply to a document as a whole, the path
//! it is read from and written to, which field is on screen, and what the run is
//! doing.
//!
//! The toolbar's shape does not depend on the document, so it is built once and only
//! dressed afterwards — captions rewritten, buttons enabled and disabled. The one
//! part that does depend on the document, the field menu, is rebuilt when the list of
//! fields changes and not otherwise.

use crate::terrain::SaveOptions;
use bevy::feathers::controls::{
    ButtonVariant, FeathersButton, FeathersMenu, FeathersMenuButton, FeathersMenuPopup,
    FeathersTextInput, FeathersTextInputContainer,
};
use bevy::feathers::theme::{ThemeBackgroundColor, ThemedText};
use bevy::feathers::tokens;
use bevy::prelude::*;
use bevy::text::{EditableText, TextEdit, TextEditChange};
use bevy::ui::InteractionDisabled;
use bevy::ui_widgets::Activate;

use crate::document::{Baked, Document};
use crate::ui::widgets::{self, one, set_text};
use crate::ui::{FilePath, NewDialog, report};
use crate::view::{EditorCamera, FreeView, fit_camera};

/// Opens the new-terrain dialog. Disabled while a job is running.
#[derive(Component, Default, Clone)]
pub struct NewButton;

/// Loads the document at the path in the field. Disabled while a job is running.
#[derive(Component, Default, Clone)]
pub struct OpenButton;

/// Writes the document to the path in the field. Disabled while a job is running or
/// there is no document.
#[derive(Component, Default, Clone)]
pub struct SaveButton;

/// Bakes the whole document. Disabled unless there is something left to bake — an
/// edit only re-bakes what is on screen, and this is how the rest catches up.
#[derive(Component, Default, Clone)]
pub struct BakeAllButton;

/// Solves the water, baking first if it has to. Enabled whenever there is a document
/// — a button that went dead after every edit is indistinguishable from a broken one.
#[derive(Component, Default, Clone)]
pub struct SolveButton;

/// Discards the solved water. Disabled unless there is some.
#[derive(Component, Default, Clone)]
pub struct ResetWaterButton;

/// The line at the right-hand end that says what the run is doing.
#[derive(Component, Default, Clone)]
pub struct StatusLabel;

/// The text field holding the path to load from and save to.
#[derive(Component, Default, Clone)]
pub struct PathInput;

/// The field menu's button caption, which names the field on screen.
#[derive(Component, Default, Clone)]
pub struct FieldMenuCaption;

/// The field menu's popup, whose children [`rebuild_field_menu`] replaces.
#[derive(Component, Default, Clone)]
pub struct FieldMenuPopup;

/// The field list the popup was last built from, so it is rebuilt when the document's
/// fields change and not every frame. Nothing else records what the popup holds.
#[derive(Resource, Default)]
pub struct FieldChoices(Vec<String>);

/// The toolbar's scene. Built once; everything document-dependent about it is filled
/// in afterwards by [`sync`], [`rebuild_field_menu`] and [`seed_path`].
pub fn toolbar() -> impl Scene {
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            flex_wrap: FlexWrap::Wrap,
            align_items: AlignItems::Center,
            column_gap: px(6),
            row_gap: px(4),
            padding: px(6),
        }
        ThemeBackgroundColor(tokens::WINDOW_BG)
        Children [
            (
                @FeathersButton {
                    @caption: bsn! { Text("New…") ThemedText },
                }
                NewButton
                on(|_: On<Activate>, mut dialog: ResMut<NewDialog>| {
                    dialog.open = true;
                })
            ),
            separator(),
            (
                @FeathersTextInputContainer
                Node { width: px(220) }
                Children [
                    (
                        @FeathersTextInput
                        PathInput
                        on(|change: On<TextEditChange>,
                            texts: Query<&EditableText>,
                            mut path: ResMut<FilePath>| {
                            if let Ok(text) = texts.get(change.event_target()) {
                                path.0 = text.value().to_string();
                            }
                        })
                    )
                ]
            ),
            (
                @FeathersButton {
                    @caption: bsn! { Text("Open") ThemedText },
                }
                OpenButton
                on(|_: On<Activate>, mut document: ResMut<Document>, path: Res<FilePath>| {
                    let result = document.start_load(path.0.clone().into());
                    report(&mut document, result);
                })
            ),
            (
                @FeathersButton {
                    @caption: bsn! { Text("Save") ThemedText },
                }
                SaveButton
                on(|_: On<Activate>, mut document: ResMut<Document>, path: Res<FilePath>| {
                    let result =
                        document.start_save(path.0.clone().into(), SaveOptions::document());
                    report(&mut document, result);
                })
            ),
            separator(),
            (
                @FeathersMenu
                Node { min_width: px(120) }
                Children [
                    (
                        @FeathersMenuButton {
                            @caption: bsn! { Text("field") ThemedText FieldMenuCaption },
                        }
                        Node { flex_grow: 1.0 }
                    ),
                    (@FeathersMenuPopup FieldMenuPopup),
                ]
            ),
            separator(),
            (
                @FeathersButton {
                    @caption: bsn! { Text("Bake all") ThemedText },
                }
                BakeAllButton
                on(|_: On<Activate>, mut document: ResMut<Document>| {
                    let result = document.start_bake();
                    report(&mut document, result);
                })
            ),
            (
                @FeathersButton {
                    @caption: bsn! { Text("Solve water") ThemedText },
                }
                SolveButton
                on(|_: On<Activate>, mut document: ResMut<Document>| {
                    let result = document.solve_with_bake();
                    report(&mut document, result);
                })
            ),
            (
                @FeathersButton {
                    @caption: bsn! { Text("Reset water") ThemedText },
                }
                ResetWaterButton
                on(|_: On<Activate>, mut document: ResMut<Document>| {
                    let result = document.reset_water();
                    report(&mut document, result);
                })
            ),
            separator(),
            (
                @FeathersButton {
                    @caption: bsn! { Text("Fit") ThemedText },
                    @variant: ButtonVariant::Plain,
                }
                on(|_: On<Activate>,
                    document: Res<Document>,
                    free: Res<FreeView>,
                    camera: Single<(&mut Transform, &mut Projection), With<EditorCamera>>| {
                    let Some(terrain) = document.terrain() else {
                        return;
                    };
                    let size = terrain.size;
                    let (mut transform, mut projection) = camera.into_inner();
                    fit_camera(&mut transform, &mut projection, size, *free);
                })
            ),
            separator(),
            (widgets::text("no document") StatusLabel),
        ]
    }
}

fn separator() -> impl Scene {
    bsn! {
        Node {
            width: px(1),
            height: px(18),
            margin: UiRect::horizontal(px(2)),
        }
        ThemeBackgroundColor(tokens::GROUP_HEADER_BORDER)
    }
}

/// Enables and disables the buttons for what the document can currently take, names
/// the field on screen, and writes the status line.
///
/// The status line prefers the running job, then the last refusal, then a description
/// of the document — a job in flight is the answer to most of "why has nothing
/// changed", so it takes precedence over everything else the line could say.
pub fn sync(
    document: Res<Document>,
    mut commands: Commands,
    mut status: Single<&mut Text, With<StatusLabel>>,
    mut caption: Single<&mut Text, (With<FieldMenuCaption>, Without<StatusLabel>)>,
    disabled: Query<(), With<InteractionDisabled>>,
    new_button: Single<Entity, With<NewButton>>,
    open_button: Single<Entity, With<OpenButton>>,
    save_button: Single<Entity, With<SaveButton>>,
    bake_button: Single<Entity, With<BakeAllButton>>,
    solve_button: Single<Entity, With<SolveButton>>,
    reset_button: Single<Entity, With<ResetWaterButton>>,
) {
    let busy = document.is_busy();
    let has_document = document.terrain().is_some();
    let has_water = document
        .terrain()
        .is_some_and(|terrain| terrain.water().is_some());
    let whole = document.baked() == Baked::Whole && !document.is_dirty();

    enable(&mut commands, &disabled, *new_button, !busy);
    enable(&mut commands, &disabled, *open_button, !busy);
    enable(
        &mut commands,
        &disabled,
        *save_button,
        !busy && has_document,
    );
    enable(
        &mut commands,
        &disabled,
        *bake_button,
        !busy && !whole && has_document,
    );
    enable(
        &mut commands,
        &disabled,
        *solve_button,
        !busy && has_document,
    );
    enable(&mut commands, &disabled, *reset_button, !busy && has_water);

    set_text(&mut caption, document.active());
    let report = match (document.job(), document.error()) {
        (Some(kind), _) => format!("{}…", kind.name()),
        (None, Some(error)) => error.to_owned(),
        (None, None) => match document.terrain() {
            Some(terrain) => format!(
                "{}x{}  {} field(s){}{}",
                terrain.size.x,
                terrain.size.y,
                terrain.fields.len(),
                if has_water { "  water" } else { "" },
                if whole { "" } else { "  preview" },
            ),
            None => "no document".to_owned(),
        },
    };
    set_text(&mut status, &report);
}

fn enable(
    commands: &mut Commands,
    disabled: &Query<(), With<InteractionDisabled>>,
    entity: Entity,
    enabled: bool,
) {
    let already = !disabled.contains(entity);
    if already == enabled {
        return;
    }
    if enabled {
        commands.entity(entity).remove::<InteractionDisabled>();
    } else {
        commands.entity(entity).insert(InteractionDisabled);
    }
}

/// Replaces the field menu's items when the document's field names change, and does
/// nothing otherwise.
pub fn rebuild_field_menu(
    document: Res<Document>,
    mut choices: ResMut<FieldChoices>,
    popup: Single<Entity, With<FieldMenuPopup>>,
    mut commands: Commands,
) {
    let names = document.field_names();
    if names == choices.0 {
        return;
    }
    choices.0 = names.clone();

    let items: Vec<Box<dyn SceneList>> = names
        .into_iter()
        .map(|name| {
            let chosen = name.clone();
            one(bsn! {
                widgets::item_caption(name)
                on(move |_: On<Activate>, mut document: ResMut<Document>| {
                    let result = document.set_active(&chosen);
                    report(&mut document, result);
                })
            })
        })
        .collect();

    commands
        .entity(*popup)
        .despawn_related::<Children>()
        .queue_spawn_related_scenes::<Children>(items);
}

/// Puts the starting path into the text field the frame it appears. A text input owns
/// a buffer that cannot be given a value at spawn time, so it is written once here.
pub fn seed_path(path: Res<FilePath>, mut inputs: Query<&mut EditableText, Added<PathInput>>) {
    for mut text in inputs.iter_mut() {
        text.queue_edit(TextEdit::SelectAll);
        text.queue_edit(TextEdit::Insert(path.0.clone().into()));
    }
}
