//! Reading a terrain.
//!
//! A terrain is an extent in cells and a set of named fields over it, each held as
//! a band of an image at its own resolution, together with a solved water state
//! where the terrain that was written carried one.
//!
//! [`Terrain`] is what this crate is for: the values and nothing that produced
//! them. Nothing here bakes, solves or evaluates anything — a terrain directory was
//! settled when it was written, and [`Terrain::load_from_dir`] is the only way to
//! obtain one. What authored it is the editor's business, and lives there.

pub mod channel;
pub mod field;
pub mod io;
pub mod meta;
pub mod raster;
pub mod terrain;

pub use channel::{ChannelEncoding, ChannelMeta, ChannelTable, MAX_CHANNELS};
pub use field::{FieldId, FieldRole};
pub use io::IoError;
pub use meta::{LayerMeta, TerrainMeta, WaterInfo};
pub use raster::{CellRect, Raster};
pub use terrain::{
    ChannelView, FieldInfo, FieldView, LayerTexels, LayerView, Terrain, TerrainLayer, WaterView,
};
