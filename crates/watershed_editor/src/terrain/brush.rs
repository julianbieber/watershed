//! Painting a raster by hand: the shape of a stroke, and what it does to the values
//! it covers.
//!
//! Nothing here knows about input devices, documents or fields. A stroke is a
//! polyline and a handful of numbers, applied to a raster the caller passes in, so
//! the same code serves an editor drag, a scripted edit and a test.

use glam::{UVec2, Vec2};
use serde::{Deserialize, Serialize};

use watershed::raster::{CellRect, Raster, Texel};

/// What a stroke does to the value already under it.
///
/// The first two move a value by an amount and are unbounded; the last two move it
/// a fraction of the way towards a target and cannot overshoot. Which pair a mode
/// belongs to is what decides how [`Brush::strength`] is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BrushMode {
    /// Adds `strength * weight`. Nothing clamps the result.
    #[default]
    Add,
    /// Subtracts `strength * weight` — `Add` with the sign reversed, so `strength`
    /// stays a magnitude in every mode.
    Subtract,
    /// Moves the value towards [`Brush::value`] by the stroke's rate, so it
    /// approaches without passing it.
    Set,
    /// Moves the value towards the mean of its eight neighbours and itself, by the
    /// stroke's rate.
    Smooth,
}

/// The shape and force of a stroke.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Brush {
    /// Reach, in document cells — never in texels, so the same brush covers the
    /// same ground on a field at any shift. At or below zero the stroke does
    /// nothing.
    pub radius_cells: f32,
    /// The fraction of the radius the weight falls off over, clamped to
    /// `0.0..=1.0`. At `0.0` the brush is a hard disc; at `1.0` it falls from the
    /// centre all the way to the rim.
    pub falloff: f32,
    /// How hard the stroke pushes. In the raster's own units for
    /// [`BrushMode::Add`] and [`BrushMode::Subtract`]; a rate — the fraction of the
    /// way to the target, clamped to `0.0..=1.0` — for [`BrushMode::Set`] and
    /// [`BrushMode::Smooth`].
    pub strength: f32,
    /// What [`BrushMode::Set`] moves a cell toward, and what no other mode reads.
    pub value: f32,
    /// What the stroke does to what it covers.
    pub mode: BrushMode,
}

impl Default for Brush {
    fn default() -> Self {
        Self {
            radius_cells: 24.0,
            falloff: 0.5,
            strength: 0.25,
            value: 1.0,
            mode: BrushMode::Add,
        }
    }
}

impl Brush {
    /// How much of the stroke lands at a distance from it, in `0.0..=1.0`.
    ///
    /// `1.0` out to `radius * (1 - falloff)`, then smoothstep down to `0.0` at the
    /// rim, and exactly `0.0` at and beyond it — so a falloff of `0.0` is a hard
    /// disc at full strength rather than a brush with no effect. Monotone
    /// decreasing, which is what makes a stroke a ridge and not a rim.
    ///
    /// `0.0` for a non-finite distance or a brush with no radius.
    pub fn weight_at(&self, distance_cells: f32) -> f32 {
        if self.radius_cells <= 0.0 || !distance_cells.is_finite() {
            return 0.0;
        }
        if distance_cells >= self.radius_cells {
            return 0.0;
        }
        let inner = self.radius_cells * (1.0 - self.falloff.clamp(0.0, 1.0));
        if distance_cells <= inner {
            return 1.0;
        }
        let t = (self.radius_cells - distance_cells) / (self.radius_cells - inner);
        t * t * (3.0 - 2.0 * t)
    }

    /// Applies the stroke to `raster` and returns the cells it may have moved.
    ///
    /// `points` are in document cells, and `size` is the document those cells belong
    /// to, so the raster is addressed through the document rather than at its own
    /// resolution: a stroke lands on the same ground whatever shift the field is at.
    /// A texel is weighted once, by its distance to the whole polyline — not by a
    /// stamp summed at each point, which would make the result depend on how finely
    /// the caller sampled the pointer.
    ///
    /// [`BrushMode::Smooth`] reads every neighbour as it was before the stroke, so
    /// the result does not depend on the order the texels are visited.
    ///
    /// The returned rectangle is in document cells, clipped to `size`, and rounded
    /// outwards: every cell whose value changed is inside it, and it may name cells
    /// that did not change. Empty when the stroke touched nothing — no points, no
    /// radius, or entirely off the document.
    pub fn stroke(&self, raster: &mut Raster<u8>, size: UVec2, points: &[Vec2]) -> CellRect {
        let Some(texels) = self.touched_texels(raster.size(), size, points) else {
            return CellRect::EMPTY;
        };
        let per_texel = cells_per_texel(raster.size(), size);

        let window = Window::of(raster, texels, self.mode == BrushMode::Smooth);

        for j in texels.min.y..texels.max.y {
            for i in texels.min.x..texels.max.x {
                let cell = Vec2::new(
                    (i as f32 + 0.5) * per_texel.x,
                    (j as f32 + 0.5) * per_texel.y,
                );
                let weight = self.weight_at(distance_to_polyline(points, cell));
                if weight <= 0.0 {
                    continue;
                }
                let Some(current) = raster.get(i, j).copied().map(Texel::to_f32) else {
                    continue;
                };
                let rate = (self.strength * weight).clamp(0.0, 1.0);
                let value = match self.mode {
                    BrushMode::Add => current + self.strength * weight,
                    BrushMode::Subtract => current - self.strength * weight,
                    BrushMode::Set => current + (self.value - current) * rate,
                    BrushMode::Smooth => current + (window.mean_around(i, j) - current) * rate,
                };
                raster.set(i, j, to_byte(value));
            }
        }

        CellRect::new(
            UVec2::new(
                (texels.min.x as f32 * per_texel.x) as u32,
                (texels.min.y as f32 * per_texel.y) as u32,
            ),
            UVec2::new(
                (texels.max.x as f32 * per_texel.x).ceil() as u32,
                (texels.max.y as f32 * per_texel.y).ceil() as u32,
            ),
        )
        .intersect(CellRect::from_size(size))
    }

    fn touched_texels(&self, resolution: UVec2, size: UVec2, points: &[Vec2]) -> Option<CellRect> {
        if points.is_empty()
            || resolution.x == 0
            || resolution.y == 0
            || size.x == 0
            || size.y == 0
            || self.radius_cells <= 0.0
        {
            return None;
        }
        let per_texel = cells_per_texel(resolution, size);

        let mut low = points[0];
        let mut high = points[0];
        for point in points {
            low = low.min(*point);
            high = high.max(*point);
        }
        let low = (low - Vec2::splat(self.radius_cells)) / per_texel - Vec2::splat(0.5);
        let high = (high + Vec2::splat(self.radius_cells)) / per_texel - Vec2::splat(0.5);

        let bounds = resolution.as_vec2();
        let min = low.floor().clamp(Vec2::ZERO, bounds).as_uvec2();
        let max = (high.floor() + Vec2::ONE)
            .clamp(Vec2::ZERO, bounds)
            .as_uvec2();
        if max.x <= min.x || max.y <= min.y {
            return None;
        }
        Some(CellRect::new(min, max))
    }
}

fn cells_per_texel(resolution: UVec2, size: UVec2) -> Vec2 {
    Vec2::new(
        size.x as f32 / resolution.x as f32,
        size.y as f32 / resolution.y as f32,
    )
}

fn distance_to_polyline(points: &[Vec2], position: Vec2) -> f32 {
    match points.len() {
        0 => f32::INFINITY,
        1 => points[0].distance(position),
        _ => points
            .windows(2)
            .map(|segment| distance_to_segment(segment[0], segment[1], position))
            .fold(f32::INFINITY, f32::min),
    }
}

fn distance_to_segment(from: Vec2, to: Vec2, position: Vec2) -> f32 {
    let along = to - from;
    let length_squared = along.length_squared();
    if length_squared <= f32::EPSILON {
        return from.distance(position);
    }
    let t = ((position - from).dot(along) / length_squared).clamp(0.0, 1.0);
    (from + along * t).distance(position)
}

fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

struct Window {
    rect: CellRect,
    values: Vec<f32>,
}

impl Window {
    fn of(raster: &Raster<u8>, texels: CellRect, wanted: bool) -> Self {
        if !wanted {
            return Self {
                rect: CellRect::EMPTY,
                values: Vec::new(),
            };
        }
        let rect = texels
            .expand(1)
            .intersect(CellRect::from_size(raster.size()));
        let mut values = Vec::with_capacity((rect.width() * rect.height()) as usize);
        for y in rect.min.y..rect.max.y {
            for x in rect.min.x..rect.max.x {
                values.push(raster.get(x, y).copied().map_or(0.0, Texel::to_f32));
            }
        }
        Self { rect, values }
    }

    fn at(&self, x: u32, y: u32) -> Option<f32> {
        if !self.rect.contains(x, y) {
            return None;
        }
        let index = (y - self.rect.min.y) * self.rect.width() + (x - self.rect.min.x);
        self.values.get(index as usize).copied()
    }

    fn mean_around(&self, x: u32, y: u32) -> f32 {
        let mut total = 0.0;
        let mut count = 0.0;
        for j in y.saturating_sub(1)..=y + 1 {
            for i in x.saturating_sub(1)..=x + 1 {
                if let Some(value) = self.at(i, j) {
                    total += value;
                    count += 1.0;
                }
            }
        }
        if count == 0.0 { 0.0 } else { total / count }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raster(size: u32) -> Raster<u8> {
        Raster::new(UVec2::splat(size), 0u8)
    }

    /// Half of the band a stroke is kept in, as the byte holding it.
    const MIDDLE: f32 = 128.0 / 255.0;

    /// A raster a stroke can move in either direction from, which zero is not: a byte
    /// clamps at the bottom of the band, so a subtract from zero is a no-op.
    fn middling(size: u32) -> Raster<u8> {
        Raster::new(UVec2::splat(size), 128u8)
    }

    /// A texel as the float everything downstream reads it as.
    fn at(raster: &Raster<u8>, x: u32, y: u32) -> f32 {
        raster.get(x, y).copied().map_or(0.0, Texel::to_f32)
    }

    fn document(size: u32) -> UVec2 {
        UVec2::splat(size)
    }

    // Pins the three points that make the falloff a brush rather than a step: full
    // at the centre, nothing at the rim, and monotone in between — a non-monotone
    // curve would lay down a rim instead of a ridge.
    #[test]
    fn a_brush_is_full_strength_at_its_centre_and_nothing_at_its_rim() {
        let brush = Brush {
            radius_cells: 10.0,
            falloff: 1.0,
            ..Brush::default()
        };
        assert_eq!(brush.weight_at(0.0), 1.0);
        assert_eq!(brush.weight_at(10.0), 0.0);
        assert_eq!(brush.weight_at(11.0), 0.0);
        assert!(brush.weight_at(5.0) > 0.0 && brush.weight_at(5.0) < 1.0);
        assert!(brush.weight_at(2.0) > brush.weight_at(8.0));
    }

    // A falloff of zero divides by a zero-width band; the guard has to resolve it to
    // full strength, because reading it as "no effect" would make the least soft
    // brush the weakest one.
    #[test]
    fn a_brush_with_no_falloff_is_a_hard_disc() {
        let brush = Brush {
            radius_cells: 10.0,
            falloff: 0.0,
            ..Brush::default()
        };
        assert_eq!(brush.weight_at(0.0), 1.0);
        assert_eq!(brush.weight_at(9.9), 1.0);
        assert_eq!(brush.weight_at(10.0), 0.0);
    }

    // The basic stroke, and the containment of it: a brush that wrote outside its
    // radius would corrupt ground the user never dragged over.
    #[test]
    fn a_stroke_raises_what_it_covers_and_leaves_the_rest_of_the_raster_alone() {
        let mut raster = raster(64);
        let brush = Brush {
            radius_cells: 8.0,
            falloff: 0.5,
            strength: 0.5,
            mode: BrushMode::Add,
            ..Brush::default()
        };
        let touched = brush.stroke(&mut raster, document(64), &[Vec2::new(32.0, 32.0)]);

        assert!(at(&raster, 32, 32) > 0.0);
        assert_eq!(at(&raster, 0, 0), 0.0);
        assert_eq!(at(&raster, 63, 63), 0.0);
        assert!(touched.contains(32, 32));
        assert!(!touched.contains(0, 0));
    }

    // The rectangle a stroke reports is what the caller re-bakes, so a cell that moved and
    // is not in it is a stale patch left on the screen.
    #[test]
    fn every_cell_a_stroke_moved_is_inside_the_rectangle_it_reports() {
        let mut raster = raster(64);
        let brush = Brush {
            radius_cells: 6.0,
            strength: 0.5,
            ..Brush::default()
        };
        let touched = brush.stroke(
            &mut raster,
            document(64),
            &[Vec2::new(10.0, 12.0), Vec2::new(48.0, 40.0)],
        );

        for y in 0..64 {
            for x in 0..64 {
                if at(&raster, x, y) != 0.0 {
                    assert!(
                        touched.contains(x, y),
                        "{x},{y} moved outside the rectangle"
                    );
                }
            }
        }
    }

    // A stroke is one polyline rather than a stamp per point, so subdividing it must not
    // change the answer — otherwise how fast the mouse was moving would decide how high a
    // ridge came out.
    #[test]
    fn subdividing_a_stroke_does_not_change_what_it_lays_down() {
        let brush = Brush {
            radius_cells: 8.0,
            strength: 0.5,
            ..Brush::default()
        };
        let ends = [Vec2::new(8.0, 32.0), Vec2::new(56.0, 32.0)];
        let subdivided = [
            Vec2::new(8.0, 32.0),
            Vec2::new(20.0, 32.0),
            Vec2::new(32.0, 32.0),
            Vec2::new(44.0, 32.0),
            Vec2::new(56.0, 32.0),
        ];

        let mut coarse = raster(64);
        let mut fine = raster(64);
        brush.stroke(&mut coarse, document(64), &ends);
        brush.stroke(&mut fine, document(64), &subdivided);

        for (a, b) in coarse.data().iter().zip(fine.data()) {
            assert_eq!(a, b, "{a} against {b}");
        }
    }

    // `Subtract` exists so `strength` can stay a magnitude; that only holds if the
    // two modes are exact mirrors, including through the falloff.
    #[test]
    fn subtracting_is_adding_in_the_other_direction() {
        let mut raised = middling(32);
        let mut lowered = middling(32);
        let brush = Brush {
            radius_cells: 6.0,
            strength: 0.4,
            mode: BrushMode::Add,
            ..Brush::default()
        };
        let points = [Vec2::new(16.0, 16.0)];
        brush.stroke(&mut raised, document(32), &points);
        Brush {
            mode: BrushMode::Subtract,
            ..brush
        }
        .stroke(&mut lowered, document(32), &points);

        for (index, (up, down)) in raised.data().iter().zip(lowered.data()).enumerate() {
            let up = Texel::to_f32(*up) - MIDDLE;
            let down = Texel::to_f32(*down) - MIDDLE;
            assert!((up + down).abs() < 1.0 / 255.0, "texel {index}: {up} against {down}");
        }
    }

    // `Set` is a rate towards a target, so even at full strength it must not
    // overshoot: an unclamped rate would push cells past the value the user asked
    // for and, under the falloff, past it by a different amount at every radius.
    #[test]
    fn setting_moves_toward_the_value_and_never_past_it() {
        let mut raster = raster(32);
        let brush = Brush {
            radius_cells: 6.0,
            falloff: 0.5,
            strength: 1.0,
            value: 0.75,
            mode: BrushMode::Set,
        };
        brush.stroke(&mut raster, document(32), &[Vec2::new(16.0, 16.0)]);

        assert!((at(&raster, 16, 16) - 0.75).abs() < 1.0 / 255.0);
        for value in raster.data() {
            let value = Texel::to_f32(*value);
            assert!((0.0..=0.75 + 1.0 / 255.0).contains(&value), "{value} left the band");
        }
    }

    // Smoothing reads its neighbours from a snapshot; this pins that it both lowers
    // the spike and raises the ring beside it, which a pass reading already-written
    // texels would do unevenly, and that it stays inside its radius.
    #[test]
    fn smoothing_pulls_a_spike_down_without_moving_the_ground_around_it() {
        let mut raster = raster(32);
        raster.set(16, 16, 255u8);
        let before = raster.data().to_vec();

        Brush {
            radius_cells: 3.0,
            falloff: 0.0,
            strength: 1.0,
            mode: BrushMode::Smooth,
            ..Brush::default()
        }
        .stroke(&mut raster, document(32), &[Vec2::new(16.5, 16.5)]);

        assert!(at(&raster, 16, 16) < 0.5);
        assert!(at(&raster, 15, 16) > 0.0);
        assert_eq!(raster.data()[0], before[0]);
    }

    // A coarse layer is stretched over the document rather than matching it, so a
    // stroke in cell coordinates has to land in the same place whatever resolution
    // it is painted at: cell (16, 16) is texel 16 of the fine raster and texel 4 of
    // the coarse one, and both have to come out raised.
    #[test]
    fn a_stroke_lands_in_the_same_place_on_a_coarse_raster_as_on_a_fine_one() {
        let brush = Brush {
            radius_cells: 16.0,
            falloff: 0.5,
            strength: 1.0,
            ..Brush::default()
        };
        let mut fine = Raster::new(UVec2::splat(64), 0u8);
        let mut coarse = Raster::new(UVec2::splat(16), 0u8);
        let points = [Vec2::new(16.0, 16.0)];
        brush.stroke(&mut fine, document(64), &points);
        brush.stroke(&mut coarse, document(64), &points);

        assert!(at(&fine, 16, 16) > 0.5);
        assert!(at(&coarse, 4, 4) > 0.5);
        assert_eq!(at(&coarse, 15, 15), 0.0);
    }

    // The texel rectangle is computed in floats and cast to unsigned; without the
    // clamp on the float side a stroke off the left edge lands on the first texel
    // instead of nowhere.
    #[test]
    fn a_stroke_outside_the_document_writes_nothing_rather_than_wrapping() {
        let mut raster = raster(32);
        let brush = Brush {
            radius_cells: 4.0,
            strength: 1.0,
            ..Brush::default()
        };
        let touched = brush.stroke(&mut raster, document(32), &[Vec2::new(-100.0, -100.0)]);
        assert!(touched.is_empty());
        assert!(raster.data().iter().all(|byte| *byte == 0));
    }

    // Both are states the editor passes through on the way to a real stroke — a
    // press before any motion, and a radius dragged to zero — so they have to be
    // empty results rather than panics or full-document rectangles.
    #[test]
    fn a_stroke_with_no_points_and_a_brush_with_no_radius_do_nothing() {
        let mut raster = raster(32);
        let brush = Brush {
            strength: 1.0,
            ..Brush::default()
        };
        assert!(brush.stroke(&mut raster, document(32), &[]).is_empty());
        assert!(
            Brush {
                radius_cells: 0.0,
                ..brush
            }
            .stroke(&mut raster, document(32), &[Vec2::splat(16.0)])
            .is_empty()
        );
        assert!(raster.data().iter().all(|byte| *byte == 0));
    }
}
