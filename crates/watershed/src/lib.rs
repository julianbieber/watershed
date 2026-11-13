//! Reading a terrain.
//!
//! A terrain is an extent in cells laid out as a grid of equal chunks, and a set of
//! named fields over it, each held as a band of an image at one texel per cell,
//! together with a solved water state where the terrain that was written carried
//! one.
//!
//! [`Terrain`] is what this crate is for: the values and nothing that produced
//! them. Nothing here bakes, solves or evaluates anything — a terrain directory was
//! settled when it was written. A consumer opens one with [`TerrainDir::open`],
//! locates a position in its [`ChunkGrid`], and loads the chunk that holds it with
//! [`TerrainDir::load_chunk`]. What authored it is the editor's business, and lives
//! there.

pub mod channel;
pub mod chunk;
pub mod field;
pub mod io;
pub mod meta;
pub mod raster;
pub mod terrain;

pub use channel::{ChannelEncoding, ChannelMeta, ChannelTable, MAX_CHANNELS, TexelFormat};
pub use chunk::{ChunkError, ChunkGrid, ChunkTexel};
pub use field::{FieldId, FieldRole};
pub use io::{IoError, TerrainDir};
pub use meta::{LayerMeta, TerrainMeta, WaterInfo};
pub use raster::{CellRect, Raster};
pub use terrain::{
    ChannelView, FieldInfo, FieldView, LayerTexels, LayerView, Terrain, TerrainLayer, WaterView,
};
