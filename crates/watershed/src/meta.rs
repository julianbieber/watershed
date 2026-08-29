//! What `terrain.ron` says: the whole of a terrain's values except the numbers,
//! which are in the images it names.
//!
//! Everything here is the serialized form and nothing more — it is what a person
//! editing a terrain by hand reads, so it holds names and indices rather than data,
//! and it never carries a texel.

use serde::{Deserialize, Serialize};

use crate::channel::ChannelMeta;
use crate::terrain::FieldInfo;

/// The format this build writes, and the only one it reads.
///
/// A terrain carrying any other version is refused outright; there is no migration
/// path, exactly as there was none for the format this replaced.
pub const VERSION: u32 = 2;

/// One image in a terrain: what it is called, how big it is, and how to read the
/// channels in it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerMeta {
    /// The file, as a plain name inside the terrain's own directory. Never a path:
    /// a reader refuses anything with a separator in it.
    pub file: String,
    /// Columns in the image.
    pub width: u32,
    /// Rows in the image.
    pub height: u32,
    /// The [`raster`](crate::raster) shift this image stands at, for one whose
    /// extent is derived from the document — a bake, or the water.
    ///
    /// `None` for a painted raster, which is stretched over the document from
    /// whatever size it was authored at and so has an extent of its own.
    pub shift: Option<u8>,
    /// One per channel, in the order they appear in a texel. Between one and
    /// [`MAX_CHANNELS`](crate::channel::MAX_CHANNELS) of them.
    pub channels: Vec<ChannelMeta>,
}

/// A solved water, as a terrain carries it.
///
/// All of it is one image at the document's extent, in a fixed channel order:
/// depth, the two components of the flow direction, then accumulation. One image
/// rather than three, and one index to check rather than five.
///
/// The solver's own output cannot be rebuilt from this — lake ids do not survive
/// quantisation and the direction codes became a vector — so it stays with the
/// document that produced it.
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
    /// Columns of the document, in cells.
    pub size_x: u32,
    /// Rows of the document, in cells.
    pub size_y: u32,
    /// Every image, in the order a reader loads them.
    pub layers: Vec<LayerMeta>,
    /// Every readable field, in the order the document declared them.
    pub fields: Vec<FieldInfo>,
    /// The solved water, if the terrain carries one.
    pub water: Option<WaterInfo>,
}
