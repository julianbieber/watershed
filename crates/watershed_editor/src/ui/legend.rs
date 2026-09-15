//! The key to what the viewport is showing: which layer, which colour ramp, and what
//! its two ends are worth.
//!
//! The numbers are the ends the ramp is *currently* fitted to, not the range the
//! layer declares. The viewport refits to what is on screen as the camera moves, so
//! a declared range would say nothing about the colours actually in front of the
//! reader.

use bevy::feathers::theme::{ThemeBackgroundColor, ThemeBorderColor};
use bevy::feathers::tokens;
use bevy::prelude::*;
use bevy::ui::{BackgroundGradient, ColorStop, Gradient, LinearGradient};

use crate::document::Document;
use crate::material;
use crate::ui::widgets::{self, set_text};
use crate::view::ViewRange;

const STEPS: usize = 64;

const RAMP_WIDTH: f32 = 200.0;

/// The legend as a whole. [`sync`] shows and hides the legend through this.
#[derive(Component, Default, Clone)]
pub struct LegendRoot;

/// The line naming the layer on show.
#[derive(Component, Default, Clone)]
pub struct LegendTitle;

/// The colour strip. Its gradient is replaced when the ramp's polarity changes.
#[derive(Component, Default, Clone)]
pub struct LegendRamp;

/// The number under the left end of the strip.
#[derive(Component, Default, Clone)]
pub struct LegendLow;

/// The number under the right end of the strip.
#[derive(Component, Default, Clone)]
pub struct LegendHigh;

/// The line saying how the ends were arrived at.
#[derive(Component, Default, Clone)]
pub struct LegendCaption;

/// The legend's scene, built hidden and with every text empty — [`sync`] fills it in
/// and shows it once a document is loaded.
pub fn legend() -> impl Scene {
    bsn! {
        Node {
            display: Display::None,
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Stretch,
            row_gap: px(4),
            padding: px(8),
            width: {px(RAMP_WIDTH + 16.0)},
            border: px(1),
        }
        ThemeBackgroundColor(tokens::WINDOW_BG)
        ThemeBorderColor(tokens::GROUP_HEADER_BORDER)
        LegendRoot
        Children [
            (widgets::text("") LegendTitle),
            (
                Node {
                    width: {px(RAMP_WIDTH)},
                    height: px(14),
                    border: px(1),
                }
                BorderColor::all(Color::srgb(0.35, 0.35, 0.35))
                LegendRamp
            ),
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::SpaceBetween,
                }
                Children [
                    (widgets::small("") LegendLow),
                    (widgets::small("") LegendHigh),
                ]
            ),
            (widgets::small("") LegendCaption),
        ]
    }
}

/// Brings the legend up to date with the document and the fitted view range, and
/// hides it entirely while no terrain is loaded.
///
/// Texts are written only where they changed. The colour strip is rebuilt only when
/// the ramp's polarity changes: the numbers move with the camera every frame, and the
/// colours do not.
pub fn sync(
    document: Res<Document>,
    range: Res<ViewRange>,
    mut root: Single<&mut Node, With<LegendRoot>>,
    mut title: Single<&mut Text, With<LegendTitle>>,
    mut low: Single<&mut Text, (With<LegendLow>, Without<LegendTitle>)>,
    mut high: Single<&mut Text, (With<LegendHigh>, Without<LegendTitle>, Without<LegendLow>)>,
    mut caption: Single<
        &mut Text,
        (
            With<LegendCaption>,
            Without<LegendTitle>,
            Without<LegendLow>,
            Without<LegendHigh>,
        ),
    >,
    ramp: Single<Entity, With<LegendRamp>>,
    mut drawn: Local<Option<bool>>,
    mut commands: Commands,
) {
    let display = if document.terrain().is_some() {
        Display::Flex
    } else {
        Display::None
    };
    if root.display != display {
        root.display = display;
    }
    if display == Display::None {
        return;
    }

    set_text(&mut title, document.active());
    set_text(&mut low, &format!("{:.4}", range.low));
    set_text(&mut high, &format!("{:.4}", range.high));
    set_text(
        &mut caption,
        if range.diverging {
            "diverging about 0"
        } else {
            "fitted to the view"
        },
    );

    if *drawn != Some(range.diverging) {
        *drawn = Some(range.diverging);
        commands
            .entity(*ramp)
            .insert(ramp_gradient(range.diverging));
    }
}

fn ramp_gradient(diverging: bool) -> BackgroundGradient {
    let stops = (0..STEPS)
        .map(|step| {
            let t = step as f32 / (STEPS - 1) as f32;
            let colour = if diverging {
                material::diverging(t * 2.0 - 1.0)
            } else {
                material::sequential(t)
            };
            ColorStop::percent(Color::srgb(colour.x, colour.y, colour.z), t * 100.0)
        })
        .collect();
    BackgroundGradient(vec![Gradient::Linear(LinearGradient::to_right(stops))])
}
