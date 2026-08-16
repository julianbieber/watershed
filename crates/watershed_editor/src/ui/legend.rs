// TODO(jb-doc): why the legend prints the live ends rather than the field's declared
// range, and what a sub-3:1 ramp needs from it besides the colour.

use bevy::feathers::theme::{ThemeBackgroundColor, ThemeBorderColor};
use bevy::feathers::tokens;
use bevy::prelude::*;
use bevy::ui::{BackgroundGradient, ColorStop, Gradient, LinearGradient};

use crate::document::Document;
use crate::material;
use crate::ui::widgets::{self, set_text};
use crate::view::ViewRange;

/// Steps the ramp is drawn with. The strip is a gradient rather than a row of rectangles,
/// so this is only how finely the ramp's own curve is sampled.
const STEPS: usize = 64;

/// Points the strip is drawn across, which is what the numbers under it have to line up
/// with.
const RAMP_WIDTH: f32 = 200.0;

#[derive(Component, Default, Clone)]
pub struct LegendRoot;

#[derive(Component, Default, Clone)]
pub struct LegendTitle;

#[derive(Component, Default, Clone)]
pub struct LegendRamp;

#[derive(Component, Default, Clone)]
pub struct LegendLow;

#[derive(Component, Default, Clone)]
pub struct LegendHigh;

#[derive(Component, Default, Clone)]
pub struct LegendCaption;

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
                // A hairline ring, because the ramp's dark end is under 2:1 against the
                // panel and without it the strip has no visible edge.
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

    // The ramp only has to be rebuilt when its polarity changes: the numbers move with the
    // camera, and the colours the strip is made of do not.
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
