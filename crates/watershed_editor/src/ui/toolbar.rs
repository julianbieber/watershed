// TODO(jb-doc): module docs — why the toolbar is built once and only dressed afterwards,
// where the layer panel below it is thrown away and rebuilt.

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
use watershed::SaveOptions;

use crate::document::{Baked, Document};
use crate::ui::widgets::{self, one, set_text};
use crate::ui::{FilePath, NewDialog, report};
use crate::view::{EditorCamera, FreeView, fit_camera};

#[derive(Component, Default, Clone)]
pub struct NewButton;

#[derive(Component, Default, Clone)]
pub struct OpenButton;

#[derive(Component, Default, Clone)]
pub struct SaveButton;

#[derive(Component, Default, Clone)]
pub struct BakeAllButton;

#[derive(Component, Default, Clone)]
pub struct SolveButton;

#[derive(Component, Default, Clone)]
pub struct ResetWaterButton;

#[derive(Component, Default, Clone)]
pub struct StatusLabel;

#[derive(Component, Default, Clone)]
pub struct PathInput;

#[derive(Component, Default, Clone)]
pub struct FieldMenuCaption;

#[derive(Component, Default, Clone)]
pub struct FieldMenuPopup;

/// The field list the menu was last built from. A document arrives with fields the popup
/// has never heard of, and there is nowhere else the popup's children could come from.
#[derive(Resource, Default)]
pub struct FieldChoices(Vec<String>);

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
                    let result = document.start_bake(None);
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

/// Whatever the run is doing takes the right-hand end, because a job in flight is the
/// answer to every "why has nothing changed".
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
    // The whole-document bake. An edit only re-bakes what is on screen, so this is how the
    // rest of the document catches up without solving anything.
    enable(
        &mut commands,
        &disabled,
        *bake_button,
        !busy && !whole && has_document,
    );
    // Enabled whenever there is a document, and it bakes first if it has to. Gating it on
    // the document being wholly baked is what it did first, and a disabled button is
    // indistinguishable from a broken one: after any edit it went quietly dead with nothing
    // on screen saying why.
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

/// TODO(jb-comment): why the popup is emptied and refilled rather than being built with
/// every field a document could ever have.
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

/// The path the toolbar starts with, put into the field once it exists — a text input is a
/// buffer, and there is no spawn-time value to give it.
pub fn seed_path(path: Res<FilePath>, mut inputs: Query<&mut EditableText, Added<PathInput>>) {
    for mut text in inputs.iter_mut() {
        text.queue_edit(TextEdit::SelectAll);
        text.queue_edit(TextEdit::Insert(path.0.clone().into()));
    }
}
