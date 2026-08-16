// TODO(jb-doc): module docs — why a modal is a node that is switched off rather than a
// window that is closed, and what that costs the first frame after Create.

use bevy::feathers::controls::{ButtonVariant, FeathersButton};
use bevy::feathers::theme::{ThemeBackgroundColor, ThemeBorderColor, ThemedText};
use bevy::feathers::tokens;
use bevy::prelude::*;
use bevy::ui::InteractionDisabled;
use bevy::ui_widgets::Activate;

use crate::document::Document;
use crate::preset::Preset;
use crate::ui::bind::NumberBinding;
use crate::ui::widgets::{self, one};
use crate::ui::{NewDialog, report};

#[derive(Component, Default, Clone)]
pub struct DialogRoot;

#[derive(Component, Default, Clone)]
pub struct PresetCaption;

#[derive(Component, Default, Clone)]
pub struct CreateButton;

pub fn dialog() -> impl Scene {
    let presets: Vec<Box<dyn SceneList>> = Preset::ALL
        .into_iter()
        .map(|preset| {
            one(bsn! {
                widgets::item_caption(preset.name())
                on(move |_: On<Activate>, mut dialog: ResMut<NewDialog>| {
                    dialog.preset = preset;
                })
            })
        })
        .collect();

    bsn! {
        Node {
            position_type: PositionType::Absolute,
            left: percent(0),
            top: percent(0),
            width: percent(100),
            height: percent(100),
            display: Display::None,
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
        }
        // Over the panels, and under a menu popup — which sits at 100, and one of which is
        // inside this dialog.
        GlobalZIndex(50)
        DialogRoot
        Children [
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Stretch,
                    row_gap: px(8),
                    padding: px(12),
                    min_width: px(300),
                    border: px(1),
                }
                ThemeBackgroundColor(tokens::WINDOW_BG)
                ThemeBorderColor(tokens::GROUP_HEADER_BORDER)
                Children [
                    widgets::text("New terrain"),
                    widgets::number_row("width", NumberBinding::DialogWidth),
                    widgets::number_row("height", NumberBinding::DialogHeight),
                    widgets::number_row("seed", NumberBinding::DialogSeed),
                    (
                        Node {
                            display: Display::Flex,
                            flex_direction: FlexDirection::Row,
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::SpaceBetween,
                            column_gap: px(6),
                        }
                        Children [
                            (Node { min_width: px(84) } Children [ widgets::small("preset") ]),
                            (
                                Node { flex_grow: 1.0, max_width: px(150) }
                                Children [
                                    (
                                        @bevy::feathers::controls::FeathersMenu
                                        Node { flex_grow: 1.0 }
                                        Children [
                                            (
                                                @bevy::feathers::controls::FeathersMenuButton {
                                                    @caption: bsn! {
                                                        Text("continents")
                                                        ThemedText
                                                        PresetCaption
                                                    },
                                                }
                                                Node { flex_grow: 1.0 }
                                            ),
                                            (
                                                @bevy::feathers::controls::FeathersMenuPopup
                                                Children [ {presets} ]
                                            ),
                                        ]
                                    )
                                ]
                            ),
                        ]
                    ),
                    (
                        Node {
                            display: Display::Flex,
                            flex_direction: FlexDirection::Row,
                            justify_content: JustifyContent::End,
                            column_gap: px(6),
                        }
                        Children [
                            (
                                @FeathersButton {
                                    @caption: bsn! { Text("Create") ThemedText },
                                    @variant: ButtonVariant::Primary,
                                }
                                CreateButton
                                on(|_: On<Activate>,
                                    mut document: ResMut<Document>,
                                    mut dialog: ResMut<NewDialog>| {
                                    let result = document.start_new(
                                        UVec2::new(dialog.width, dialog.height),
                                        dialog.seed,
                                        dialog.preset,
                                    );
                                    report(&mut document, result);
                                    dialog.open = false;
                                })
                            ),
                            (
                                @FeathersButton {
                                    @caption: bsn! { Text("Cancel") ThemedText },
                                }
                                on(|_: On<Activate>, mut dialog: ResMut<NewDialog>| {
                                    dialog.open = false;
                                })
                            ),
                        ]
                    ),
                ]
            )
        ]
    }
}

pub fn sync(
    dialog: Res<NewDialog>,
    document: Res<Document>,
    mut root: Single<&mut Node, With<DialogRoot>>,
    mut caption: Single<&mut Text, With<PresetCaption>>,
    create: Single<Entity, With<CreateButton>>,
    mut commands: Commands,
) {
    let display = if dialog.open {
        Display::Flex
    } else {
        Display::None
    };
    if root.display != display {
        root.display = display;
    }

    let name = dialog.preset.name();
    if caption.0 != name {
        caption.0 = name.to_owned();
    }

    if document.is_busy() {
        commands.entity(*create).insert(InteractionDisabled);
    } else {
        commands.entity(*create).remove::<InteractionDisabled>();
    }
}
