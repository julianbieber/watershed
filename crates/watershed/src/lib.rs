// TODO(jb-doc): crate-level docs — what a terrain is here (size, named fields, layer
// stacks, a derived water state), and which of those the caller owns.

// TODO(jb-comment): stage 5 adds `io`.

pub mod bake;
pub mod field;
pub mod layer;
pub mod noise;
pub mod raster;
pub mod regions;
pub mod water;

pub use bake::{BakeError, Terrain};
pub use field::{Field, FieldId};
pub use layer::{Blend, Layer, LayerOp, Mask, Remap};
pub use raster::{CellRect, Raster};
pub use regions::{Region, RegionMap, RegionOutput, RegionSpec};
pub use water::{WaterError, WaterSpec, WaterState};
