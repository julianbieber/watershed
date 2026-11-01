//! The layout of a terrain as a grid of equal chunks, and where a cell position
//! falls in it.

use glam::UVec2;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::raster::CellRect;

/// Why a chunk size and a chunk count do not make a grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum ChunkError {
    /// A chunk with no cells on some axis.
    #[error("the chunk size has a zero component: {0} by {1}")]
    ZeroChunkSize(u32, u32),
    /// No chunks on some axis.
    #[error("the chunk count has a zero component: {0} by {1}")]
    ZeroChunks(u32, u32),
    /// The extent, chunk size times count, does not fit a `u32` on some axis.
    #[error("{0} chunks of {1} cells overflow the extent on one axis")]
    ExtentOverflow(u32, u32),
}

/// A terrain as `chunks` chunks on each axis, each `chunk_size` cells — and, since
/// every layer holds one texel per cell, `chunk_size` texels.
///
/// Every chunk has the same extent, so the document's extent is always a whole
/// number of chunks. Only [`ChunkGrid::new`] builds one, so a grid in hand is a
/// valid one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ChunkGrid {
    chunk_size: UVec2,
    chunks: UVec2,
}

/// Where a cell position falls: the chunk holding it, and the texel inside that
/// chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkTexel {
    /// The chunk's index on each axis.
    pub chunk: UVec2,
    /// The texel within the chunk, counted from its first cell.
    pub texel: UVec2,
}

impl ChunkGrid {
    /// A grid of `chunks` chunks of `chunk_size` cells each.
    ///
    /// Refused for a zero component in either, and for an extent that does not
    /// fit a `u32` on either axis.
    pub fn new(chunk_size: UVec2, chunks: UVec2) -> Result<Self, ChunkError> {
        if chunk_size.x == 0 || chunk_size.y == 0 {
            return Err(ChunkError::ZeroChunkSize(chunk_size.x, chunk_size.y));
        }
        if chunks.x == 0 || chunks.y == 0 {
            return Err(ChunkError::ZeroChunks(chunks.x, chunks.y));
        }
        for (count, size) in [(chunks.x, chunk_size.x), (chunks.y, chunk_size.y)] {
            if count.checked_mul(size).is_none() {
                return Err(ChunkError::ExtentOverflow(count, size));
            }
        }
        Ok(Self { chunk_size, chunks })
    }

    /// One chunk covering `size`. Refused exactly when `size` has a zero component.
    pub fn single(size: UVec2) -> Result<Self, ChunkError> {
        Self::new(size, UVec2::ONE)
    }

    /// Cells, and so texels, of one chunk on each axis.
    pub fn chunk_size(&self) -> UVec2 {
        self.chunk_size
    }

    /// Chunks on each axis.
    pub fn chunks(&self) -> UVec2 {
        self.chunks
    }

    /// The whole document in cells: chunk size times chunk count.
    pub fn extent(&self) -> UVec2 {
        self.chunk_size * self.chunks
    }

    /// Where a cell position falls, or `None` outside the half-open extent and for
    /// a position that is negative or not finite.
    ///
    /// A cell's centre is at `x + 0.5`, and every position in `x..x + 1` falls in
    /// cell `x`. Exact while the extent stays below `2^24` cells on an axis, which
    /// is where an `f32` stops naming every whole cell.
    pub fn locate(&self, x: f32, y: f32) -> Option<ChunkTexel> {
        let extent = self.extent();
        if !(x.is_finite() && y.is_finite()) || x < 0.0 || y < 0.0 {
            return None;
        }
        if x >= extent.x as f32 || y >= extent.y as f32 {
            return None;
        }
        let cell = UVec2::new(
            (x.floor() as u32).min(extent.x - 1),
            (y.floor() as u32).min(extent.y - 1),
        );
        Some(ChunkTexel {
            chunk: cell / self.chunk_size,
            texel: cell % self.chunk_size,
        })
    }

    /// The document cell of a chunk's first texel, or `None` for a chunk outside
    /// the grid.
    pub fn origin(&self, chunk: UVec2) -> Option<UVec2> {
        self.contains(chunk).then(|| chunk * self.chunk_size)
    }

    /// A chunk's cells as a rectangle of the document, or `None` for a chunk
    /// outside the grid.
    pub fn rect(&self, chunk: UVec2) -> Option<CellRect> {
        let origin = self.origin(chunk)?;
        Some(CellRect::new(origin, origin + self.chunk_size))
    }

    /// Whether `chunk` names a chunk of this grid.
    pub fn contains(&self, chunk: UVec2) -> bool {
        chunk.x < self.chunks.x && chunk.y < self.chunks.y
    }

    /// Every chunk index, row by row.
    pub fn iter(&self) -> impl Iterator<Item = UVec2> + use<> {
        let chunks = self.chunks;
        (0..chunks.y).flat_map(move |y| (0..chunks.x).map(move |x| UVec2::new(x, y)))
    }

    /// How many chunks the grid holds.
    pub fn count(&self) -> u64 {
        self.chunks.x as u64 * self.chunks.y as u64
    }
}

#[derive(Deserialize)]
struct GridFields {
    chunk_size: UVec2,
    chunks: UVec2,
}

impl<'de> Deserialize<'de> for ChunkGrid {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let fields = GridFields::deserialize(deserializer)?;
        ChunkGrid::new(fields.chunk_size, fields.chunks).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pins `locate` and `origin` as inverses over every cell of a small grid,
    // including the last cell, where an off-by-one in the clamp would land.
    #[test]
    fn locating_a_cell_and_adding_its_chunk_origin_gives_the_cell_back() {
        let grid = ChunkGrid::new(UVec2::new(3, 2), UVec2::new(2, 3)).unwrap();
        for y in 0..grid.extent().y {
            for x in 0..grid.extent().x {
                let found = grid.locate(x as f32 + 0.5, y as f32 + 0.5).unwrap();
                let origin = grid.origin(found.chunk).unwrap();
                assert_eq!(origin + found.texel, UVec2::new(x, y));
            }
        }
    }

    // The extent is half-open: a consumer asking at the far edge, a negative or a
    // NaN position must get `None` rather than a clamped neighbour.
    #[test]
    fn a_position_outside_the_extent_is_located_nowhere() {
        let grid = ChunkGrid::new(UVec2::new(4, 4), UVec2::new(2, 2)).unwrap();
        assert!(grid.locate(8.0, 1.0).is_none());
        assert!(grid.locate(1.0, 8.0).is_none());
        assert!(grid.locate(-0.5, 1.0).is_none());
        assert!(grid.locate(f32::NAN, 1.0).is_none());
        assert!(grid.locate(7.99, 7.99).is_some());
    }

    // A grid comes from untrusted files, so a zero or overflowing one has to be a
    // refusal at construction rather than a division by zero later.
    #[test]
    fn a_zero_or_overflowing_grid_is_refused() {
        assert!(ChunkGrid::new(UVec2::new(0, 4), UVec2::ONE).is_err());
        assert!(ChunkGrid::new(UVec2::new(4, 4), UVec2::new(1, 0)).is_err());
        assert!(ChunkGrid::new(UVec2::new(u32::MAX, 1), UVec2::new(2, 1)).is_err());
        assert!(ron::from_str::<ChunkGrid>("(chunk_size: (0, 1), chunks: (1, 1))").is_err());
    }

    // `rect` is what bake and save slice whole-document rasters with, so it has to
    // tile the extent exactly: every chunk in, none overlapping.
    #[test]
    fn the_chunk_rectangles_tile_the_extent() {
        let grid = ChunkGrid::new(UVec2::new(2, 3), UVec2::new(3, 2)).unwrap();
        let cells: u64 = grid
            .iter()
            .map(|chunk| {
                let rect = grid.rect(chunk).unwrap();
                rect.width() as u64 * rect.height() as u64
            })
            .sum();
        assert_eq!(cells, grid.extent().x as u64 * grid.extent().y as u64);
        assert!(grid.rect(UVec2::new(3, 0)).is_none());
    }
}
