//! Authoring, baking and reading a terrain.
//!
//! A terrain is an extent in cells and a set of named fields over it. Each field is
//! a stack of layers — noise, paint, a slope of another field, a region tiling —
//! evaluated onto a grid of its own resolution, and optionally a solved water state
//! derived from whichever field holds the height.
//!
//! Two types carry that, and which one a caller holds says what it is doing:
//!
//! - [`TerrainSpec`] is the authored document. It holds the layers, it is what is
//!   saved and loaded, and it is what an editor mutates.
//! - [`Terrain`] is the result of baking one. It holds the values and nothing that
//!   produced them, so a consuming application can read a document it could not
//!   author.
//!
//! Everything derived is the caller's to ask for: a loaded document arrives unbaked
//! and every field samples as `0.0` until [`TerrainSpec::bake`] or the staged
//! [`TerrainSpec::begin_bake`] has run over it.

pub mod bake;
pub mod brush;
pub mod channel;
pub mod error;
pub mod field;
pub mod io;
pub mod layer;
pub mod meta;
pub mod noise;
pub mod raster;
pub mod regions;
pub mod terrain;
pub mod water;

pub use bake::{
    Bake, BakeError, BakePlan, BakeProgress, BakeReport, BakeStep, PlanError, StepKind, TerrainSpec,
};
pub use brush::{Brush, BrushMode};
pub use channel::{ChannelEncoding, ChannelMeta, ChannelTable, MAX_CHANNELS};
pub use error::Error;
pub use field::{Field, FieldId, FieldRole};
pub use io::{IoError, SaveOptions};
pub use layer::{Blend, Layer, LayerOp, Mask, Remap, SlopeMode};
pub use meta::{FieldStack, LayerMeta, PaintRef, PaintSlot, TerrainMeta, WaterInfo};
pub use raster::{CellRect, Raster};
pub use regions::{Region, RegionMap, RegionOutput, RegionSpec};
pub use terrain::{
    ChannelView, FieldInfo, FieldView, LayerTexels, LayerView, Terrain, TerrainLayer, WaterView,
};
pub use water::{WaterError, WaterSpec, WaterState};
