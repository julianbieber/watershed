//! Storage and addressing for the grids a field's baked values live in, and the
//! rectangle arithmetic that names sub-areas of a document in cell coordinates.

use glam::UVec2;
use serde::{Deserialize, Serialize};

/// The largest shift [`step`] will honour. A larger shift is silently treated as
/// this one, which keeps `1 << shift` inside a `u32` for every `u8` a caller can
/// pass; there is no error path for an out-of-range shift.
pub const MAX_SHIFT: u8 = 16;

/// Cells covered by one texel of a field at `shift`: one at shift 0, `2^shift`
/// above that, saturating at `2^MAX_SHIFT`.
///
/// The field holding [`FieldRole::Height`](crate::field::FieldRole::Height) is
/// rejected at plan time unless its shift is 0; every other field is free.
pub fn step(shift: u8) -> u32 {
    1u32 << shift.min(MAX_SHIFT)
}

/// Texel dimensions of a `size`-cell document rasterised at `shift`.
///
/// Rounds up, so the raster always covers the whole document, and never returns a
/// zero component — a shift coarser than the document still yields one texel.
pub fn resolution(size: UVec2, shift: u8) -> UVec2 {
    let step = step(shift);
    UVec2::new(size.x.div_ceil(step).max(1), size.y.div_ceil(step).max(1))
}

/// Cell position a texel stands for: the centre of the block of cells it covers,
/// not the block's corner. Inverse of [`raster_coord`].
pub fn texel_center(index: u32, shift: u8) -> f32 {
    (index as f32 + 0.5) * step(shift) as f32
}

/// Texel coordinate a cell position reads at, in the convention
/// [`Raster::sample_bilinear`] expects: texel centres sit on the integers.
///
/// Inverse of [`texel_center`]. At shift 0 a cell centre (`x + 0.5`) therefore
/// lands exactly on texel `x`.
pub fn raster_coord(position: f32, shift: u8) -> f32 {
    position / step(shift) as f32 - 0.5
}

/// A value a [`Raster`] can be sampled through.
///
/// `to_f32` is what every sample returns. An implementation covering a fixed range
/// maps it onto the unit interval, so a sample is comparable whatever the storage
/// type behind the raster is.
pub trait Texel: Copy {
    fn to_f32(self) -> f32;
}

impl Texel for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}

impl Texel for u8 {
    fn to_f32(self) -> f32 {
        self as f32 / 255.0
    }
}

/// A row-major grid of texels with its own dimensions and nothing else: a raster
/// does not know which field it belongs to, where it sits in a document, or at
/// what shift it was produced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Raster<T> {
    size: UVec2,
    data: Vec<T>,
}

impl<T> Default for Raster<T> {
    fn default() -> Self {
        Self {
            size: UVec2::ZERO,
            data: Vec::new(),
        }
    }
}

impl<T> Raster<T> {
    /// Texel dimensions. Not the size of the document the raster covers.
    pub fn size(&self) -> UVec2 {
        self.size
    }

    /// Texel columns.
    pub fn width(&self) -> u32 {
        self.size.x
    }

    /// Texel rows.
    pub fn height(&self) -> u32 {
        self.size.y
    }

    /// Total texels, `width * height`.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the raster holds no texels. A default raster is empty, and stays so
    /// until it is replaced wholesale — there is no growing operation.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The texels in row-major order: index `y * width + x`.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// The texels in row-major order, writable. The length is fixed; writing
    /// through this slice cannot desynchronise the data from [`Raster::size`].
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Wraps an existing row-major buffer, or `None` if its length is not exactly
    /// `size.x * size.y`. A mismatch is never resolved by padding or truncating —
    /// the caller has the wrong buffer and gets it back.
    pub fn from_vec(size: UVec2, data: Vec<T>) -> Option<Self> {
        if data.len() as u64 != size.x as u64 * size.y as u64 {
            return None;
        }
        Some(Self { size, data })
    }

    fn index(&self, x: u32, y: u32) -> Option<usize> {
        if x >= self.size.x || y >= self.size.y {
            return None;
        }
        Some((y as usize) * (self.size.x as usize) + (x as usize))
    }

    /// The texel at `(x, y)`, or `None` outside the raster. Out of bounds is never
    /// a panic and never a clamp — for a clamping read use the sampling methods.
    pub fn get(&self, x: u32, y: u32) -> Option<&T> {
        self.index(x, y).map(|i| &self.data[i])
    }

    /// As [`Raster::get`], writable.
    pub fn get_mut(&mut self, x: u32, y: u32) -> Option<&mut T> {
        self.index(x, y).map(|i| &mut self.data[i])
    }

    /// Writes `value` at `(x, y)`. Returns whether the coordinate was inside the
    /// raster; a write outside it is dropped, not clamped.
    pub fn set(&mut self, x: u32, y: u32, value: T) -> bool {
        match self.index(x, y) {
            Some(i) => {
                self.data[i] = value;
                true
            }
            None => false,
        }
    }
}

impl<T: Clone> Raster<T> {
    /// A `size`-texel raster with every texel set to `value`. A zero component
    /// gives an empty raster rather than an error.
    pub fn new(size: UVec2, value: T) -> Self {
        let cells = size.x as usize * size.y as usize;
        Self {
            size,
            data: vec![value; cells],
        }
    }

    /// Overwrites every texel with `value`, leaving the dimensions alone.
    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }
}

impl<T: Texel> Raster<T> {
    /// Bilinear read at a texel coordinate — texel centres are the integers, so
    /// `(0.0, 0.0)` is the first texel exactly.
    ///
    /// Coordinates outside the raster clamp to its edge; an empty raster reads as
    /// `0.0`. Neither is an error, so a caller cannot distinguish an edge read
    /// from an in-bounds one by the return value.
    pub fn sample_bilinear(&self, u: f32, v: f32) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let max_x = self.size.x - 1;
        let max_y = self.size.y - 1;
        let u = u.clamp(0.0, max_x as f32);
        let v = v.clamp(0.0, max_y as f32);
        let u_floor = u.floor();
        let v_floor = v.floor();
        let fx = u - u_floor;
        let fy = v - v_floor;
        let x0 = (u_floor as u32).min(max_x);
        let y0 = (v_floor as u32).min(max_y);
        let x1 = (x0 + 1).min(max_x);
        let y1 = (y0 + 1).min(max_y);

        let at = |x: u32, y: u32| {
            self.data[(y as usize) * (self.size.x as usize) + (x as usize)].to_f32()
        };
        let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
        let top = a + (b - a) * fx;
        let bottom = c + (d - c) * fx;
        top + (bottom - top) * fy
    }

    /// Nearest-texel read, in the same coordinate convention as
    /// [`Raster::sample_bilinear`].
    ///
    /// For a raster whose values stand for a *class* rather than a quantity: two
    /// classes have no midpoint, so a read between them must return one of the two
    /// and never their average. Clamps outside the raster, reads `0.0` when empty,
    /// and reads the texel at the origin for a NaN coordinate.
    pub fn sample_nearest(&self, u: f32, v: f32) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let max_x = self.size.x - 1;
        let max_y = self.size.y - 1;
        let x = u.round().clamp(0.0, max_x as f32) as u32;
        let y = v.round().clamp(0.0, max_y as f32) as u32;
        self.data[(y as usize) * (self.size.x as usize) + (x as usize)].to_f32()
    }

    /// Bilinear read at a *cell* position in a `size`-cell document, stretching
    /// the raster over the whole document whatever its own dimensions are.
    ///
    /// The raster is therefore not obliged to match the document's resolution: one
    /// at exactly `size` reads cell centres (`x + 0.5`) as texel centres. Reads
    /// `0.0` for an empty raster or a `size` with a zero component.
    pub fn sample_over(&self, size: UVec2, x: f32, y: f32) -> f32 {
        if self.data.is_empty() || size.x == 0 || size.y == 0 {
            return 0.0;
        }
        let u = x * (self.size.x as f32 / size.x as f32) - 0.5;
        let v = y * (self.size.y as f32 / size.y as f32) - 0.5;
        self.sample_bilinear(u, v)
    }
}

/// A half-open rectangle of cells: `min` is included, `max` is not. Nothing here
/// bounds it — a rectangle may name cells outside the document, and is clipped
/// where it is used rather than where it is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellRect {
    /// First cell included, on each axis.
    pub min: UVec2,
    /// First cell *past* the rectangle, on each axis.
    pub max: UVec2,
}

impl CellRect {
    /// The rectangle covering no cells. Absorbing under [`CellRect::union`] and
    /// [`CellRect::expand`], annihilating under [`CellRect::intersect`].
    pub const EMPTY: Self = Self {
        min: UVec2::ZERO,
        max: UVec2::ZERO,
    };

    /// The half-open rectangle from `min` to `max`. A `max` at or below `min` on
    /// either axis is accepted and reads as empty.
    pub const fn new(min: UVec2, max: UVec2) -> Self {
        Self { min, max }
    }

    /// The rectangle covering a whole `size`-cell document.
    pub const fn from_size(size: UVec2) -> Self {
        Self {
            min: UVec2::ZERO,
            max: size,
        }
    }

    /// Whether the rectangle covers no cells, by either axis being degenerate.
    /// True for more values than [`CellRect::EMPTY`].
    pub fn is_empty(&self) -> bool {
        self.max.x <= self.min.x || self.max.y <= self.min.y
    }

    /// Cells spanned on the x axis; `0` when empty.
    pub fn width(&self) -> u32 {
        self.max.x.saturating_sub(self.min.x)
    }

    /// Cells spanned on the y axis; `0` when empty.
    pub fn height(&self) -> u32 {
        self.max.y.saturating_sub(self.min.y)
    }

    /// Whether the cell is inside, under the half-open convention: a cell at `max`
    /// is not.
    pub fn contains(&self, x: u32, y: u32) -> bool {
        x >= self.min.x && x < self.max.x && y >= self.min.y && y < self.max.y
    }

    /// The smallest rectangle covering both. An empty operand is ignored rather
    /// than dragging the result back to the origin.
    pub fn union(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// The cells in both. Disjoint rectangles give exactly [`CellRect::EMPTY`], so
    /// the result is never a degenerate rectangle at some arbitrary position.
    pub fn intersect(self, other: Self) -> Self {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        if max.x <= min.x || max.y <= min.y {
            Self::EMPTY
        } else {
            Self { min, max }
        }
    }

    /// Grows the rectangle by `by` cells on every side, saturating at `0` and at
    /// `u32::MAX` rather than wrapping. An empty rectangle stays empty.
    pub fn expand(self, by: u32) -> Self {
        if self.is_empty() {
            return self;
        }
        Self {
            min: UVec2::new(self.min.x.saturating_sub(by), self.min.y.saturating_sub(by)),
            max: UVec2::new(self.max.x.saturating_add(by), self.max.y.saturating_add(by)),
        }
    }

    /// The texels at `shift` covering this rectangle of cells, clipped to
    /// `resolution`.
    ///
    /// Rounds outwards, so the result covers every cell the rectangle names and
    /// may cover cells outside it. An empty rectangle gives [`CellRect::EMPTY`].
    pub fn to_texels(self, shift: u8, resolution: UVec2) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        let step = step(shift);
        let min = UVec2::new(self.min.x / step, self.min.y / step);
        let max = UVec2::new(self.max.x.div_ceil(step), self.max.y.div_ceil(step));
        Self {
            min: min.min(resolution),
            max: max.min(resolution),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pins the row-major index and the out-of-bounds contract together: a wrong
    // stride reads a neighbouring row instead of failing, so the round trip has to
    // cover every cell, and the reject has to be a `None`/`false` rather than a panic.
    #[test]
    fn a_raster_reads_back_what_was_written_at_every_cell() {
        let mut raster = Raster::new(UVec2::new(3, 2), 0.0f32);
        for y in 0..2 {
            for x in 0..3 {
                assert!(raster.set(x, y, (y * 3 + x) as f32));
            }
        }
        for y in 0..2 {
            for x in 0..3 {
                assert_eq!(*raster.get(x, y).unwrap(), (y * 3 + x) as f32);
            }
        }
        assert!(raster.get(3, 0).is_none());
        assert!(raster.get(0, 2).is_none());
        assert!(!raster.set(3, 0, 1.0));
    }

    // Fixes the coordinate convention the rest of the crate is written against: a
    // half-texel error anywhere in it still interpolates smoothly, so only an exact
    // read at a centre catches it.
    #[test]
    fn a_bilinear_read_at_a_texel_centre_is_that_texel_exactly() {
        let raster = Raster::from_vec(UVec2::new(2, 2), vec![0.25f32, 0.5, 0.75, 1.0]).unwrap();
        assert_eq!(raster.sample_bilinear(0.0, 0.0), 0.25);
        assert_eq!(raster.sample_bilinear(1.0, 0.0), 0.5);
        assert_eq!(raster.sample_bilinear(0.0, 1.0), 0.75);
        assert_eq!(raster.sample_bilinear(1.0, 1.0), 1.0);
    }

    // Baking reads a neighbourhood around every texel, so edge texels are sampled
    // out of bounds as a matter of course; this pins that as edge clamping rather
    // than a wrap, a zero, or a panic.
    #[test]
    fn a_bilinear_read_outside_the_raster_clamps_to_its_edge() {
        let raster = Raster::from_vec(UVec2::new(2, 2), vec![0.0f32, 1.0, 2.0, 3.0]).unwrap();
        assert_eq!(raster.sample_bilinear(-10.0, -10.0), 0.0);
        assert_eq!(raster.sample_bilinear(10.0, 10.0), 3.0);
        assert_eq!(raster.sample_bilinear(0.5, 0.0), 0.5);
    }

    // A released or not-yet-baked field is an empty raster that callers still
    // sample, so the empty case is a live path and not a defensive branch.
    #[test]
    fn an_empty_raster_reads_as_zero_rather_than_panicking() {
        let raster = Raster::<f32>::default();
        assert_eq!(raster.sample_bilinear(0.0, 0.0), 0.0);
        assert_eq!(raster.sample_over(UVec2::new(8, 8), 4.0, 4.0), 0.0);
    }

    // Pins the `Texel for u8` mapping: byte rasters have to be comparable with f32
    // ones at the sampling boundary, which only holds if the full byte range maps
    // onto the whole unit interval.
    #[test]
    fn a_byte_texel_reads_as_the_unit_interval() {
        let raster = Raster::from_vec(UVec2::new(2, 1), vec![0u8, 255]).unwrap();
        assert_eq!(raster.sample_bilinear(0.0, 0.0), 0.0);
        assert_eq!(raster.sample_bilinear(1.0, 0.0), 1.0);
    }

    // `sample_over` composes two half-texel offsets, and the composition is only
    // right if they cancel at the document's own resolution — the case every
    // shift-0 field takes.
    #[test]
    fn a_raster_at_the_documents_own_size_reads_cell_centres_exactly() {
        let raster = Raster::from_vec(UVec2::new(2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let size = UVec2::new(2, 2);
        assert_eq!(raster.sample_over(size, 0.5, 0.5), 1.0);
        assert_eq!(raster.sample_over(size, 1.5, 0.5), 2.0);
        assert_eq!(raster.sample_over(size, 1.5, 1.5), 4.0);
    }

    // Both guards in `resolution` are load-bearing for allocation: rounding down
    // would drop the last partial texel, and a zero would make an empty raster out
    // of a document that has cells. The last case pins the `MAX_SHIFT` saturation.
    #[test]
    fn a_shift_divides_the_resolution_and_never_falls_below_one_texel() {
        let size = UVec2::new(4096, 4096);
        assert_eq!(resolution(size, 0), UVec2::new(4096, 4096));
        assert_eq!(resolution(size, 4), UVec2::new(256, 256));
        assert_eq!(resolution(UVec2::new(3, 3), 4), UVec2::new(1, 1));
        assert_eq!(resolution(UVec2::new(5, 5), 1), UVec2::new(3, 3));
        assert_eq!(resolution(size, 200), UVec2::new(1, 1));
    }

    // The two halves of the shift convention are written independently, so nothing
    // but this forces them to agree; a sign error in the half-texel offset survives
    // either one read alone.
    #[test]
    fn a_texel_centre_and_a_raster_coordinate_are_inverses() {
        for shift in [0u8, 1, 4, 8] {
            for index in [0u32, 1, 7, 100] {
                let position = texel_center(index, shift);
                assert_eq!(raster_coord(position, shift), index as f32);
            }
        }
    }

    // Dirty rectangles drive which texels get rebaked, so rounding inwards on
    // either edge leaves stale texels in a baked field. Pins the outward rounding
    // against a rectangle unaligned to the shift on both edges.
    #[test]
    fn a_texel_rectangle_covers_every_cell_the_rectangle_names() {
        let rect = CellRect::new(UVec2::new(3, 3), UVec2::new(9, 9));
        let texels = rect.to_texels(2, UVec2::new(4, 4));
        assert_eq!(texels.min, UVec2::new(0, 0));
        assert_eq!(texels.max, UVec2::new(3, 3));
    }

    // Dirty rectangles accumulate by union from an empty start, so treating empty
    // as a rectangle at the origin would silently extend every accumulation to
    // cover the corner of the document.
    #[test]
    fn a_union_ignores_an_empty_rectangle() {
        let rect = CellRect::new(UVec2::new(1, 1), UVec2::new(2, 2));
        assert_eq!(rect.union(CellRect::EMPTY), rect);
        assert_eq!(CellRect::EMPTY.union(rect), rect);
        assert!(rect.intersect(CellRect::EMPTY).is_empty());
    }

    // `expand` grows a dirty rectangle by a filter's reach, which routinely runs
    // off the origin; wrapping there would turn a small rectangle into one covering
    // the far edge of the document.
    #[test]
    fn expanding_a_rectangle_at_the_origin_saturates_rather_than_wrapping() {
        let rect = CellRect::new(UVec2::ZERO, UVec2::new(2, 2)).expand(10);
        assert_eq!(rect.min, UVec2::ZERO);
        assert_eq!(rect.max, UVec2::new(12, 12));
    }
}
