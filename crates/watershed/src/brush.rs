// TODO(jb-doc): module docs — what a brush is here (numbers and a polyline, no input and
// no ECS), and why it writes into a raster the caller owns rather than into a document.

use glam::{UVec2, Vec2};
use serde::{Deserialize, Serialize};

use crate::raster::{CellRect, Raster};

/// TODO(jb-doc): what each mode does to the value under it, and why `Subtract` is a mode of
/// its own rather than an `Add` carrying a negative strength.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BrushMode {
    #[default]
    Add,
    Subtract,
    Set,
    Smooth,
}

/// TODO(jb-doc): the unit every field here is in — that a radius is document cells, and
/// that a strength is the field's own units for `Add` and `Subtract` where it is a fraction
/// of the way there for `Set` and `Smooth`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Brush {
    pub radius_cells: f32,
    pub falloff: f32,
    pub strength: f32,
    /// What [`BrushMode::Set`] moves a cell toward, and what no other mode reads.
    pub value: f32,
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
    /// TODO(jb-doc): the shape of the falloff, and why a falloff of zero is a hard disc
    /// rather than a brush with no effect.
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

    /// TODO(jb-doc): why a stroke is weighted by its distance to the *whole* polyline rather
    /// than by a stamp summed at each point, and what a sum would make the result depend on.
    ///
    /// TODO(jb-doc): the coordinate space the points are in, and why the raster is addressed
    /// through the document's size rather than at its own resolution.
    pub fn stroke(&self, raster: &mut Raster<f32>, size: UVec2, points: &[Vec2]) -> CellRect {
        let Some(texels) = self.touched_texels(raster.size(), size, points) else {
            return CellRect::EMPTY;
        };
        let per_texel = cells_per_texel(raster.size(), size);

        // A smoothed texel reads its neighbours as they were before the stroke, or the
        // result would depend on which corner of the rectangle the walk started at.
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
                let Some(current) = raster.get(i, j).copied() else {
                    continue;
                };
                let rate = (self.strength * weight).clamp(0.0, 1.0);
                let value = match self.mode {
                    BrushMode::Add => current + self.strength * weight,
                    BrushMode::Subtract => current - self.strength * weight,
                    BrushMode::Set => current + (self.value - current) * rate,
                    BrushMode::Smooth => current + (window.mean_around(i, j) - current) * rate,
                };
                raster.set(i, j, value);
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

    /// TODO(jb-comment): why the rectangle is rounded outwards, given a texel stands for the
    /// centre of the block it covers.
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

        // Clamped as floats before the cast, or a rectangle entirely to the left of the
        // document saturates at zero on both ends and comes back as the first texel.
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

/// The texels a smoothing pass reads, copied out before any of them is written.
struct Window {
    rect: CellRect,
    values: Vec<f32>,
}

impl Window {
    fn of(raster: &Raster<f32>, texels: CellRect, wanted: bool) -> Self {
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
                values.push(raster.get(x, y).copied().unwrap_or(0.0));
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

    fn raster(size: u32) -> Raster<f32> {
        Raster::new(UVec2::splat(size), 0.0)
    }

    fn document(size: u32) -> UVec2 {
        UVec2::splat(size)
    }

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
        // Monotone, which is what makes a stroke a ridge rather than a rim.
        assert!(brush.weight_at(2.0) > brush.weight_at(8.0));
    }

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

        assert!(*raster.get(32, 32).unwrap() > 0.0);
        assert_eq!(*raster.get(0, 0).unwrap(), 0.0);
        assert_eq!(*raster.get(63, 63).unwrap(), 0.0);
        assert!(touched.contains(32, 32));
        assert!(!touched.contains(0, 0));
    }

    /// The rectangle a stroke reports is what the caller re-bakes, so a cell that moved and
    /// is not in it is a stale patch left on the screen.
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
                if *raster.get(x, y).unwrap() != 0.0 {
                    assert!(
                        touched.contains(x, y),
                        "{x},{y} moved outside the rectangle"
                    );
                }
            }
        }
    }

    /// A stroke is one polyline rather than a stamp per point, so subdividing it must not
    /// change the answer — otherwise how fast the mouse was moving would decide how high a
    /// ridge came out.
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
            assert!((a - b).abs() < 1e-5, "{a} against {b}");
        }
    }

    #[test]
    fn subtracting_is_adding_in_the_other_direction() {
        let mut raised = raster(32);
        let mut lowered = raster(32);
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

        for (up, down) in raised.data().iter().zip(lowered.data()) {
            assert!((up + down).abs() < 1e-6);
        }
    }

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

        assert!((*raster.get(16, 16).unwrap() - 0.75).abs() < 1e-6);
        for value in raster.data() {
            assert!((0.0..=0.75).contains(value), "{value} left the band");
        }
    }

    #[test]
    fn smoothing_pulls_a_spike_down_without_moving_the_ground_around_it() {
        let mut raster = raster(32);
        raster.set(16, 16, 1.0);
        let before = raster.data().to_vec();

        Brush {
            radius_cells: 3.0,
            falloff: 0.0,
            strength: 1.0,
            mode: BrushMode::Smooth,
            ..Brush::default()
        }
        .stroke(&mut raster, document(32), &[Vec2::new(16.5, 16.5)]);

        assert!(*raster.get(16, 16).unwrap() < 0.5);
        assert!(*raster.get(15, 16).unwrap() > 0.0);
        assert_eq!(*raster.get(0, 0).unwrap(), before[0]);
    }

    /// A coarse layer is stretched over the document rather than matching it, so a stroke in
    /// cell coordinates has to land in the same place whatever resolution it is painted at.
    #[test]
    fn a_stroke_lands_in_the_same_place_on_a_coarse_raster_as_on_a_fine_one() {
        let brush = Brush {
            radius_cells: 16.0,
            falloff: 0.5,
            strength: 1.0,
            ..Brush::default()
        };
        let mut fine = Raster::new(UVec2::splat(64), 0.0);
        let mut coarse = Raster::new(UVec2::splat(16), 0.0);
        let points = [Vec2::new(16.0, 16.0)];
        brush.stroke(&mut fine, document(64), &points);
        brush.stroke(&mut coarse, document(64), &points);

        // Cell (16, 16) is texel 16 of the fine raster and texel 4 of the coarse one.
        assert!(*fine.get(16, 16).unwrap() > 0.5);
        assert!(*coarse.get(4, 4).unwrap() > 0.5);
        assert_eq!(*coarse.get(15, 15).unwrap(), 0.0);
    }

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
        assert!(raster.data().iter().all(|value| *value == 0.0));
    }

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
        assert!(raster.data().iter().all(|value| *value == 0.0));
    }
}
