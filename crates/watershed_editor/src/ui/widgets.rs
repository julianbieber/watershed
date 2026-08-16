// TODO(jb-doc): module docs — what belongs here rather than beside the panel that uses it,
// and why a helper takes a boxed scene list rather than a closure.

use bevy::feathers::controls::{
    FeathersMenu, FeathersMenuButton, FeathersMenuItem, FeathersMenuPopup, FeathersNumberInput,
};
use bevy::feathers::display::{label, label_small};
use bevy::feathers::theme::ThemedText;
use bevy::prelude::*;
use bevy::scene::EntityScene;

use crate::ui::bind::{self, NumberBinding};

/// One scene as the one-entity list a child slot takes. Every list built here is a
/// `Vec` of these, because a panel's children are counted at run time.
pub fn one(scene: impl Scene) -> Box<dyn SceneList> {
    Box::new(EntityScene(scene))
}

/// A component the scene only carries when the condition holds — a checked checkbox, a
/// button that cannot be pressed. `None` is a scene that patches nothing.
pub fn when<C: Component + Clone + Default + Unpin>(on: bool, component: C) -> Option<impl Scene> {
    on.then_some(template_value(component))
}

/// Written only when it changed. A `Text` touched every frame is a text layout redone
/// every frame, and a panel says the same thing for most of them.
pub fn set_text(text: &mut Text, body: &str) {
    if text.0 != body {
        text.0 = body.to_owned();
    }
}

/// A horizontal strip of controls.
pub fn row(children: Vec<Box<dyn SceneList>>) -> impl Scene {
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(4),
        }
        Children [ {children} ]
    }
}

/// A vertical stack of controls.
pub fn column(children: Vec<Box<dyn SceneList>>) -> impl Scene {
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Stretch,
            row_gap: px(4),
        }
        Children [ {children} ]
    }
}

/// A caption on the left and its control on the right, a caption and the thing it names.
pub fn captioned(caption: impl Into<String>, control: Box<dyn SceneList>) -> impl Scene {
    let caption = caption.into();
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            justify_content: JustifyContent::SpaceBetween,
            column_gap: px(6),
        }
        Children [
            (
                Node { min_width: px(84) }
                Children [ label_small(caption) ]
            ),
            (
                Node { flex_grow: 1.0, max_width: px(150) }
                Children [ {control} ]
            ),
        ]
    }
}

/// A number field bound to one place in the document, the brush or the dialog.
pub fn number(binding: NumberBinding) -> impl Scene {
    bsn! {
        @FeathersNumberInput {
            @number_format: {binding.format()},
        }
        template_value(binding)
        Node { flex_grow: 1.0 }
        on(bind::on_f32)
        on(bind::on_i32)
    }
}

/// A caption and the number field it names.
pub fn number_row(caption: impl Into<String>, binding: NumberBinding) -> impl Scene {
    captioned(caption, one(number(binding)))
}

/// A menu standing in for a combo box: the button shows the choice, the popup offers the
/// rest. The caption is not synced — a choice is part of the panel's shape, so making one
/// rebuilds the panel that shows it.
pub fn menu(caption: impl Into<String>, items: Vec<Box<dyn SceneList>>) -> impl Scene {
    let caption = caption.into();
    bsn! {
        @FeathersMenu
        Node { flex_grow: 1.0 }
        Children [
            (
                @FeathersMenuButton {
                    @caption: bsn! { Text({caption}) ThemedText },
                }
                Node { flex_grow: 1.0 }
            ),
            (
                @FeathersMenuPopup
                Children [ {items} ]
            ),
        ]
    }
}

/// Plain text, for the things a panel says rather than the things it offers.
pub fn text(body: impl Into<String>) -> impl Scene {
    label(body.into())
}

pub fn small(body: impl Into<String>) -> impl Scene {
    label_small(body.into())
}

/// A menu item is a caption and what choosing it does. Callers build the observer, because
/// what a choice means is the one thing no two of them share.
pub fn item_caption(caption: impl Into<String>) -> impl Scene {
    let caption = caption.into();
    bsn! {
        @FeathersMenuItem {
            @caption: bsn! { Text({caption}) ThemedText },
        }
    }
}
