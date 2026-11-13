//! What `terrain.ron` says: the whole of a terrain's values except the numbers,
//! which are in the images it names.
//!
//! Everything here is the serialized form and nothing more — it is what a person
//! editing a terrain by hand reads, so it holds names and indices rather than data,
//! and it never carries a texel.

use glam::UVec2;
use serde::{Deserialize, Serialize};

use crate::channel::{ChannelMeta, TexelFormat};
use crate::terrain::FieldInfo;

/// The format this build writes, and the only one it reads.
///
/// A terrain carrying any other version is refused outright; there is no migration
/// path.
pub const VERSION: u32 = 4;

/// One image in a terrain: what it is called, how big each chunk of it is, and how
/// to read the channels in it.
///
/// An image is stored as one file per chunk, named from the chunk's index, this
/// stem and the image's [format](LayerMeta::format): `chunk_<x>_<y>_<file>.png`
/// for a byte image, and `chunk_<x>_<y>_<file>.f32.zz` for a float one: its
/// interleaved little-endian texels split into byte planes — byte `k` of every
/// texel, in row-major order, then byte `k + 1` — as one zlib stream.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerMeta {
    /// The stem every chunk file of this image is named from: `layer_<n>`. A
    /// reader refuses any other shape.
    pub file: String,
    /// Columns in each chunk's image, which is the chunk size.
    pub width: u32,
    /// Rows in each chunk's image, which is the chunk size.
    pub height: u32,
    /// One per channel, in the order they appear in a texel. Between one and
    /// [`MAX_CHANNELS`](crate::channel::MAX_CHANNELS) of them.
    pub channels: Vec<ChannelMeta>,
}

impl LayerMeta {
    /// The width every channel of this image shares, or `None` if it has no
    /// channel or its channels disagree — an image a reader refuses.
    pub fn format(&self) -> Option<TexelFormat> {
        let first = self.channels.first()?.format();
        self.channels
            .iter()
            .all(|channel| channel.format() == first)
            .then_some(first)
    }
}

/// A solved water, as a terrain carries it.
///
/// All of it is one four-channel image at the document's extent, chunked like every
/// other, in a fixed channel order:
/// depth, the two components of the flow direction, then accumulation. One image
/// rather than three, and one index to check rather than five.
///
/// The solver's own output cannot be rebuilt from this — lake ids are not carried
/// and the direction codes became a vector — so it stays with the document that
/// produced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WaterInfo {
    /// How many lakes the solve found. The ids themselves are not carried.
    pub lakes: u32,
    /// The image holding the four channels.
    pub layer: u8,
}

impl WaterInfo {
    /// Channel holding the depth at a cell.
    pub const DEPTH: usize = 0;
    /// Channel holding the x component of the flow direction.
    pub const FLOW_X: usize = 1;
    /// Channel holding the y component of the flow direction.
    pub const FLOW_Y: usize = 2;
    /// Channel holding the accumulation reaching a cell.
    pub const ACCUM: usize = 3;
}

/// `terrain.ron` itself.
///
/// The values and nothing that produced them. A recipe, where a terrain carries
/// one, is a second file beside this one and is never named from here — so a
/// reader of values need not know what a layer stack is.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TerrainMeta {
    /// [`VERSION`]. Checked before anything else is read.
    pub version: u32,
    /// Cells of one chunk on each axis.
    pub chunk_size: UVec2,
    /// Chunks on each axis. The document's extent is `chunk_size * chunks`.
    pub chunks: UVec2,
    /// Every image, in the order a reader loads them.
    pub layers: Vec<LayerMeta>,
    /// Every readable field, in the order the document declared them.
    pub fields: Vec<FieldInfo>,
    /// The solved water, if the terrain carries one.
    pub water: Option<WaterInfo>,
}
