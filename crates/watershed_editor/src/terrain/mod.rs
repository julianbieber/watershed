//! Authoring a terrain: the document an editor mutates, and everything that turns
//! it into the values a `watershed::Terrain` carries.
//!
//! Nothing outside the editor holds any of this. A consuming project reads a
//! terrain directory, and a directory was settled when it was written — so the
//! layer shaders, the water solve and the bake are all authoring-time machinery and
//! live here rather than in the library.

#![allow(dead_code)]

pub mod bake;
pub mod layer;
pub mod recipe;
pub mod shader;
pub mod water;

pub use bake::TerrainSpec;
pub use layer::{Layer, LayerId, LayerRole};
pub use water::{WaterSpec, WaterState};
