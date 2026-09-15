//! The curve between two field cards, and the ribbon it is drawn as.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

/// How thick an edge is on screen, in pixels, at every zoom.
pub(super) const EDGE_PIXELS: f32 = 3.0;
const EDGE_SAMPLES: usize = 48;
const MIN_REACH: f32 = 70.0;

/// The mesh a ribbon between two canvas points is drawn as, `half_width` canvas units
/// either side of the curve joining them.
///
/// The curve leaves `start` and arrives at `end` sideways, so a ribbon meets the
/// right edge of the card it leaves and the left edge of the card it reaches square on.
pub(super) fn ribbon_between(start: Vec2, end: Vec2, half_width: f32) -> Mesh {
    ribbon(&curve(start, end), half_width)
}

fn curve(start: Vec2, end: Vec2) -> Vec<Vec2> {
    let reach = ((end.x - start.x).abs() * 0.5).max(MIN_REACH);
    let first = start + Vec2::X * reach;
    let second = end - Vec2::X * reach;
    (0..=EDGE_SAMPLES)
        .map(|step| {
            let t = step as f32 / EDGE_SAMPLES as f32;
            let u = 1.0 - t;
            start * (u * u * u)
                + first * (3.0 * u * u * t)
                + second * (3.0 * u * t * t)
                + end * (t * t * t)
        })
        .collect()
}

/// An empty ribbon, which is what a ribbon is spawned holding until it is first routed.
pub fn blank_ribbon() -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new())
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, Vec::<[f32; 2]>::new())
    .with_inserted_indices(Indices::U32(Vec::new()))
}

fn ribbon(points: &[Vec2], half_width: f32) -> Mesh {
    if points.len() < 2 {
        return blank_ribbon();
    }
    let last = points.len() - 1;
    let mut positions = Vec::with_capacity(points.len() * 2);
    let mut uvs = Vec::with_capacity(points.len() * 2);
    let mut indices = Vec::with_capacity(last * 6);

    for (index, point) in points.iter().enumerate() {
        let previous = points[index.saturating_sub(1)];
        let next = points[(index + 1).min(last)];
        let tangent = (next - previous).normalize_or_zero();
        let normal = Vec2::new(-tangent.y, tangent.x) * half_width;
        let left = *point + normal;
        let right = *point - normal;
        positions.push([left.x, left.y, 0.0]);
        positions.push([right.x, right.y, 0.0]);
        let along = index as f32 / last as f32;
        uvs.push([along, 0.0]);
        uvs.push([along, 1.0]);
    }
    for index in 0..last as u32 {
        let base = index * 2;
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(Indices::U32(indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Both ends have to land exactly on the card edges the ribbon joins, or a ribbon and
    // the card it leaves separate visibly at the one place a person is looking.
    #[test]
    fn a_curve_starts_and_ends_on_the_points_it_joins() {
        let start = Vec2::new(-40.0, 12.0);
        let end = Vec2::new(220.0, -60.0);
        let points = curve(start, end);
        assert_eq!(points.first().copied(), Some(start));
        assert_eq!(points.last().copied(), Some(end));
    }

    // A ribbon is two vertices per sample and two triangles per span; a mesh that
    // disagreed with its own index buffer draws nothing rather than drawing wrong.
    #[test]
    fn a_ribbon_carries_two_vertices_a_sample_and_two_triangles_a_span() {
        let points = curve(Vec2::ZERO, Vec2::new(100.0, 0.0));
        let mesh = ribbon(&points, 1.5);
        let positions = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().len();
        assert_eq!(positions, points.len() * 2);
        let Some(Indices::U32(indices)) = mesh.indices() else {
            panic!("a ribbon is indexed");
        };
        assert_eq!(indices.len(), (points.len() - 1) * 6);
        assert!(indices.iter().all(|index| (*index as usize) < positions));
    }

    // One point has no span to widen, and a mesh with no vertices and no indices is
    // what keeps a degenerate sampling from panicking.
    #[test]
    fn a_ribbon_of_one_point_is_empty_rather_than_a_panic() {
        let mesh = ribbon(&[Vec2::ZERO], 1.5);
        assert_eq!(mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().len(), 0);
    }
}
