use glam::UVec2;
use serde::{Deserialize, Serialize};

/// TODO(jb-doc): why a shift is capped at all, and what a cap of 16 means against the
/// largest document the format admits.
pub const MAX_SHIFT: u8 = 16;

/// TODO(jb-doc): what a shift is — one texel per cell at 0, one per 2^shift at n — and
/// which fields are obliged to stay at 0.
pub fn step(shift: u8) -> u32 {
    1u32 << shift.min(MAX_SHIFT)
}

/// TODO(jb-doc): why the resolution rounds up and floors at one texel.
pub fn resolution(size: UVec2, shift: u8) -> UVec2 {
    let step = step(shift);
    UVec2::new(size.x.div_ceil(step).max(1), size.y.div_ceil(step).max(1))
}

/// TODO(jb-doc): the position a texel stands for, and why it is the centre of the block
/// it covers rather than the block's corner.
pub fn texel_center(index: u32, shift: u8) -> f32 {
    (index as f32 + 0.5) * step(shift) as f32
}

/// TODO(jb-doc): the inverse of [`texel_center`], and why the half-texel offset is what
/// makes a shift-0 read of a cell centre land exactly on a texel.
pub fn raster_coord(position: f32, shift: u8) -> f32 {
    position / step(shift) as f32 - 0.5
}

/// TODO(jb-doc): what a texel is allowed to be, and why the conversion to f32 is on the
/// type rather than at every call site.
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

/// TODO(jb-doc): what a raster owns, what it deliberately does not know (its position in
/// the document, the field it belongs to), and the row-major order its data is in.
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
    pub fn size(&self) -> UVec2 {
        self.size
    }

    pub fn width(&self) -> u32 {
        self.size.x
    }

    pub fn height(&self) -> u32 {
        self.size.y
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// TODO(jb-doc): why a mismatched length is `None` rather than a panic or a resize.
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

    pub fn get(&self, x: u32, y: u32) -> Option<&T> {
        self.index(x, y).map(|i| &self.data[i])
    }

    pub fn get_mut(&mut self, x: u32, y: u32) -> Option<&mut T> {
        self.index(x, y).map(|i| &mut self.data[i])
    }

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
    pub fn new(size: UVec2, value: T) -> Self {
        let cells = size.x as usize * size.y as usize;
        Self {
            size,
            data: vec![value; cells],
        }
    }

    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }
}

impl<T: Texel> Raster<T> {
    /// TODO(jb-doc): the coordinate convention this takes (texel centres on the integers),
    /// what happens outside the raster, and why an empty raster reads as zero.
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

    /// TODO(jb-doc): the one reason a raster is ever read this way — that a value standing
    /// for a *class* has no midpoint, so the texel between two of them is one of the two
    /// rather than their average.
    pub fn sample_nearest(&self, u: f32, v: f32) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let max_x = self.size.x - 1;
        let max_y = self.size.y - 1;
        // NaN survives `clamp` and lands on zero through the cast rather than on an index
        // out of bounds, which is the same fallback an empty raster reads as.
        let x = u.round().clamp(0.0, max_x as f32) as u32;
        let y = v.round().clamp(0.0, max_y as f32) as u32;
        self.data[(y as usize) * (self.size.x as usize) + (x as usize)].to_f32()
    }

    /// TODO(jb-doc): why a raster carried by a layer is stretched over the whole document
    /// rather than being obliged to match its resolution.
    pub fn sample_over(&self, size: UVec2, x: f32, y: f32) -> f32 {
        if self.data.is_empty() || size.x == 0 || size.y == 0 {
            return 0.0;
        }
        let u = x * (self.size.x as f32 / size.x as f32) - 0.5;
        let v = y * (self.size.y as f32 / size.y as f32) - 0.5;
        self.sample_bilinear(u, v)
    }
}

/// TODO(jb-doc): the half-open convention, and why the document's own rectangle is the
/// only thing that bounds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellRect {
    pub min: UVec2,
    pub max: UVec2,
}

impl CellRect {
    pub const EMPTY: Self = Self {
        min: UVec2::ZERO,
        max: UVec2::ZERO,
    };

    pub const fn new(min: UVec2, max: UVec2) -> Self {
        Self { min, max }
    }

    pub const fn from_size(size: UVec2) -> Self {
        Self {
            min: UVec2::ZERO,
            max: size,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.max.x <= self.min.x || self.max.y <= self.min.y
    }

    pub fn width(&self) -> u32 {
        self.max.x.saturating_sub(self.min.x)
    }

    pub fn height(&self) -> u32 {
        self.max.y.saturating_sub(self.min.y)
    }

    pub fn contains(&self, x: u32, y: u32) -> bool {
        x >= self.min.x && x < self.max.x && y >= self.min.y && y < self.max.y
    }

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

    pub fn intersect(self, other: Self) -> Self {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        if max.x <= min.x || max.y <= min.y {
            Self::EMPTY
        } else {
            Self { min, max }
        }
    }

    pub fn expand(self, by: u32) -> Self {
        if self.is_empty() {
            return self;
        }
        Self {
            min: UVec2::new(self.min.x.saturating_sub(by), self.min.y.saturating_sub(by)),
            max: UVec2::new(self.max.x.saturating_add(by), self.max.y.saturating_add(by)),
        }
    }

    /// TODO(jb-doc): why the conversion rounds outwards, and what an over-inclusive texel
    /// rectangle costs against what an under-inclusive one would break.
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

    #[test]
    fn a_bilinear_read_at_a_texel_centre_is_that_texel_exactly() {
        let raster = Raster::from_vec(UVec2::new(2, 2), vec![0.25f32, 0.5, 0.75, 1.0]).unwrap();
        assert_eq!(raster.sample_bilinear(0.0, 0.0), 0.25);
        assert_eq!(raster.sample_bilinear(1.0, 0.0), 0.5);
        assert_eq!(raster.sample_bilinear(0.0, 1.0), 0.75);
        assert_eq!(raster.sample_bilinear(1.0, 1.0), 1.0);
    }

    #[test]
    fn a_bilinear_read_outside_the_raster_clamps_to_its_edge() {
        let raster = Raster::from_vec(UVec2::new(2, 2), vec![0.0f32, 1.0, 2.0, 3.0]).unwrap();
        assert_eq!(raster.sample_bilinear(-10.0, -10.0), 0.0);
        assert_eq!(raster.sample_bilinear(10.0, 10.0), 3.0);
        assert_eq!(raster.sample_bilinear(0.5, 0.0), 0.5);
    }

    #[test]
    fn an_empty_raster_reads_as_zero_rather_than_panicking() {
        let raster = Raster::<f32>::default();
        assert_eq!(raster.sample_bilinear(0.0, 0.0), 0.0);
        assert_eq!(raster.sample_over(UVec2::new(8, 8), 4.0, 4.0), 0.0);
    }

    #[test]
    fn a_byte_texel_reads_as_the_unit_interval() {
        let raster = Raster::from_vec(UVec2::new(2, 1), vec![0u8, 255]).unwrap();
        assert_eq!(raster.sample_bilinear(0.0, 0.0), 0.0);
        assert_eq!(raster.sample_bilinear(1.0, 0.0), 1.0);
    }

    #[test]
    fn a_raster_at_the_documents_own_size_reads_cell_centres_exactly() {
        let raster = Raster::from_vec(UVec2::new(2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let size = UVec2::new(2, 2);
        assert_eq!(raster.sample_over(size, 0.5, 0.5), 1.0);
        assert_eq!(raster.sample_over(size, 1.5, 0.5), 2.0);
        assert_eq!(raster.sample_over(size, 1.5, 1.5), 4.0);
    }

    #[test]
    fn a_shift_divides_the_resolution_and_never_falls_below_one_texel() {
        let size = UVec2::new(4096, 4096);
        assert_eq!(resolution(size, 0), UVec2::new(4096, 4096));
        assert_eq!(resolution(size, 4), UVec2::new(256, 256));
        assert_eq!(resolution(UVec2::new(3, 3), 4), UVec2::new(1, 1));
        assert_eq!(resolution(UVec2::new(5, 5), 1), UVec2::new(3, 3));
        assert_eq!(resolution(size, 200), UVec2::new(1, 1));
    }

    #[test]
    fn a_texel_centre_and_a_raster_coordinate_are_inverses() {
        for shift in [0u8, 1, 4, 8] {
            for index in [0u32, 1, 7, 100] {
                let position = texel_center(index, shift);
                assert_eq!(raster_coord(position, shift), index as f32);
            }
        }
    }

    #[test]
    fn a_texel_rectangle_covers_every_cell_the_rectangle_names() {
        let rect = CellRect::new(UVec2::new(3, 3), UVec2::new(9, 9));
        let texels = rect.to_texels(2, UVec2::new(4, 4));
        assert_eq!(texels.min, UVec2::new(0, 0));
        assert_eq!(texels.max, UVec2::new(3, 3));
    }

    #[test]
    fn a_union_ignores_an_empty_rectangle() {
        let rect = CellRect::new(UVec2::new(1, 1), UVec2::new(2, 2));
        assert_eq!(rect.union(CellRect::EMPTY), rect);
        assert_eq!(CellRect::EMPTY.union(rect), rect);
        assert!(rect.intersect(CellRect::EMPTY).is_empty());
    }

    #[test]
    fn expanding_a_rectangle_at_the_origin_saturates_rather_than_wrapping() {
        let rect = CellRect::new(UVec2::ZERO, UVec2::new(2, 2)).expand(10);
        assert_eq!(rect.min, UVec2::ZERO);
        assert_eq!(rect.max, UVec2::new(12, 12));
    }
}
