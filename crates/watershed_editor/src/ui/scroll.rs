// TODO(jb-doc): module docs — that the wheel belongs to whatever is under the pointer, and
// why that is one rule rather than a list of which panels take it.

use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use bevy::ui::{OverflowAxis, ScrollPosition};

/// Points one line of wheel moves a panel. A line is not a length, so it has to be given
/// one somewhere, and a row of the layer stack is about this tall.
const LINE_HEIGHT: f32 = 21.0;

/// A wheel turn offered to a node. It bubbles, so the widget the pointer is actually over
/// gets first refusal and whatever encloses it takes what is left — which is what makes a
/// number field inside the layer panel scroll the panel rather than nothing at all.
#[derive(EntityEvent, Debug)]
#[entity_event(propagate, auto_propagate)]
pub struct Scroll {
    entity: Entity,
    delta: Vec2,
}

pub fn send(mut wheel: MessageReader<MouseWheel>, hover: Res<HoverMap>, mut commands: Commands) {
    for message in wheel.read() {
        let mut delta = -Vec2::new(message.x, message.y);
        if message.unit == MouseScrollUnit::Line {
            delta *= LINE_HEIGHT;
        }
        for hits in hover.values() {
            for entity in hits.keys().copied() {
                commands.trigger(Scroll { entity, delta });
            }
        }
    }
}

/// Takes as much of the turn as this node can scroll and passes the rest on. A node that is
/// already at its end keeps nothing, so the wheel carries through to whatever encloses it.
pub fn apply(
    mut scroll: On<Scroll>,
    mut nodes: Query<(&mut ScrollPosition, &Node, &ComputedNode)>,
) {
    let Ok((mut position, node, computed)) = nodes.get_mut(scroll.entity) else {
        return;
    };
    let limit = (computed.content_size() - computed.size()) * computed.inverse_scale_factor();

    let delta = &mut scroll.delta;
    if node.overflow.x == OverflowAxis::Scroll && delta.x != 0.0 {
        let ended = if delta.x > 0.0 {
            position.x >= limit.x
        } else {
            position.x <= 0.0
        };
        if !ended {
            position.x += delta.x;
            delta.x = 0.0;
        }
    }
    if node.overflow.y == OverflowAxis::Scroll && delta.y != 0.0 {
        let ended = if delta.y > 0.0 {
            position.y >= limit.y
        } else {
            position.y <= 0.0
        };
        if !ended {
            position.y += delta.y;
            delta.y = 0.0;
        }
    }
}
