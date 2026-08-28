//! What `terrain.ron` says: the whole of a terrain except the numbers, which are in
//! the images it names.
//!
//! Everything here is the serialized form and nothing more — it is what a person
//! editing a terrain by hand reads, so it holds names and indices rather than data,
//! and it never carries a texel.

use serde::{Deserialize, Serialize};

use crate::channel::ChannelMeta;
use crate::field::FieldId;
use crate::layer::Layer;
use crate::terrain::FieldInfo;
use crate::water::WaterSpec;

/// The format this build writes, and the only one it reads.
///
/// A terrain carrying any other version is refused outright; there is no migration
/// path, exactly as there was none for the format this replaced.
pub const VERSION: u32 = 1;

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

/// Which raster of a layer stack a painted image stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaintSlot {
    /// The layer's own raster — a [`LayerOp::Paint`](crate::layer::LayerOp::Paint)
    /// or [`LayerOp::External`](crate::layer::LayerOp::External).
    Op,
    /// The layer's [`Mask::Painted`](crate::layer::Mask::Painted).
    Mask,
}

/// A painted raster's home, stated rather than implied by position.
///
/// The old format re-attached paint by walking the stacks in the same order twice.
/// This file is hand-editable, so an order both halves have to agree on without
/// saying so is the one thing left that a reader would have to derive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PaintRef {
    /// Position in the field's stack.
    pub stack_index: u32,
    /// Which of that layer's two rasters this is.
    pub slot: PaintSlot,
    /// The image holding it.
    pub layer: u8,
}

/// The recipe for one field: everything needed to bake it again.
///
/// Absent from a terrain saved without its recipe, which is readable but can no
/// longer be edited.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldStack {
    /// The field this stack belongs to.
    pub field: FieldId,
    /// The interval a bake clamps into. Not the same number as the range of the
    /// channel the field is stored in, which is what the byte spreads over.
    pub range: (f32, f32),
    /// Carried through and never read by this crate.
    pub export: bool,
    /// The layers, in evaluation order, with every raster in them emptied out.
    pub stack: Vec<Layer>,
    /// Where each emptied raster went.
    pub paint: Vec<PaintRef>,
}

/// A solved water, as a terrain carries it.
///
/// All of it is one image at the document's extent, in a fixed channel order:
/// depth, the two components of the flow direction, then accumulation. One image
/// rather than three, and one index to check rather than five.
///
/// A solved [`WaterState`](crate::water::WaterState) cannot be rebuilt from this —
/// lake ids do not survive quantisation and the direction codes became a vector —
/// so the solver's own output stays with the document that produced it.
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
    /// The spec the water was solved from, kept so a document that carries no
    /// solved water can still be re-solved rather than losing it.
    pub water_spec: Option<WaterSpec>,
    /// The recipe, one entry per field. Empty in a terrain saved without one.
    pub stacks: Vec<FieldStack>,
}
