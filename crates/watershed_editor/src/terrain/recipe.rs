//! The on-disk form of a document's recipe: `recipe.ron`, beside the values a
//! `watershed::Terrain` was written from.
//!
//! The values and the recipe are two files because they are two audiences. A
//! consuming project reads `terrain.ron` and never learns what a field graph is; the
//! editor reads both, and only the editor can. Nothing here is needed to read a
//! terrain, and a directory with this half deleted is still a terrain.
//!
//! A recipe read here is untrusted input, exactly as the values are: its size is
//! checked before it is read, and its version before anything in it is used.

use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use watershed::field::FieldId;
use watershed::io::IoError;
use watershed::terrain::Terrain;

use crate::gpu::ShaderRuntime;
use crate::terrain::bake::{BakeError, PlanError, TerrainSpec};
use crate::terrain::field::Field;
use crate::terrain::graph::{FieldGraph, NodeOp};
use crate::terrain::water::{WaterError, WaterSpec};

/// The recipe file a terrain directory carries when it was saved as a document, and
/// the only name this reader knows without being told it.
pub const RECIPE_FILE: &str = "recipe.ron";

/// The recipe format this build writes, and the only one it reads.
///
/// Separate from [`watershed::meta::VERSION`], which versions the `terrain.ron` a
/// consuming project reads: the two files have two audiences and change for
/// different reasons, and a recipe that gains a field must not invalidate every
/// terrain already exported.
///
/// A recipe carrying any other version is refused outright; there is no migration
/// path.
pub const RECIPE_VERSION: u32 = 3;

/// The largest `recipe.ron` this build will read, checked before the file is read.
pub const MAX_RECIPE_BYTES: u64 = 64 * 1024 * 1024;

/// Why a document could not be written or read.
#[derive(Debug, Error)]
pub enum RecipeError {
    /// The values half. See [`IoError`].
    #[error(transparent)]
    Io(#[from] IoError),
    /// The underlying reader or writer.
    #[error("io: {0}")]
    File(#[from] std::io::Error),
    /// `recipe.ron` is not readable as a recipe — including one that names a node op
    /// this build does not have.
    #[error("{RECIPE_FILE} is not readable: {0}")]
    Meta(String),
    /// `recipe.ron` is larger than [`MAX_RECIPE_BYTES`]. Checked before it is read.
    #[error("{RECIPE_FILE} is {0} bytes, over the {MAX_RECIPE_BYTES} byte limit")]
    TooLarge(u64),
    /// A recipe this build cannot read. There is no migration path.
    #[error("recipe version {0} is not supported; this build reads {RECIPE_VERSION}")]
    UnsupportedVersion(u32),
    /// The document the recipe describes cannot be planned.
    #[error("plan: {0}")]
    Plan(#[from] PlanError),
    /// The document the recipe describes could not be baked.
    #[error("bake: {0}")]
    Bake(#[from] BakeError),
    /// The water could not be solved again.
    #[error("water: {0}")]
    Water(#[from] WaterError),
}

/// What a save writes, beyond the values every terrain carries.
///
/// The values are always written — a terrain without them is one no consumer can
/// read — so the only choice left is whether the recipe goes with them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaveOptions {
    /// Write `recipe.ron`. Without it the terrain has values but no recipe, and cannot
    /// be opened for editing again.
    pub recipe: bool,
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self::document()
    }
}

impl SaveOptions {
    /// Everything: the values and the recipe that produced them. What an editor
    /// saves, and the only form that can be opened for editing again.
    pub fn document() -> Self {
        Self { recipe: true }
    }

    /// Values without the recipe, for a consumer that reads a terrain and never
    /// authors one.
    ///
    /// **Removes a recipe already in the directory**, because one left behind would
    /// claim to describe values it no longer produced.
    pub fn export() -> Self {
        Self { recipe: false }
    }
}

/// The recipe for one field: everything needed to bake it again.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldStack {
    /// The field this graph belongs to.
    pub field: FieldId,
    /// The interval a bake clamps into. Not the same number as the range of the
    /// channel the field's values are stored in, which is what the byte spreads
    /// over.
    pub range: (f32, f32),
    /// Carried through and never read by the bake.
    pub export: bool,
    /// The graph, with every shader node's values dropped.
    pub graph: FieldGraph,
}

/// `recipe.ron` itself.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecipeMeta {
    /// The recipe format version, [`RECIPE_VERSION`] — not the version of the
    /// `terrain.ron` beside it. Checked before anything else is read.
    pub version: u32,
    /// The spec the water was solved from, kept so a document that carries no
    /// solved water can still be re-solved rather than losing it.
    pub water_spec: Option<WaterSpec>,
    /// One entry per field that has a graph.
    pub stacks: Vec<FieldStack>,
}

fn stripped_op(op: &NodeOp) -> NodeOp {
    match op {
        NodeOp::Shader(shader) => {
            let mut shader = shader.clone();
            shader.clear();
            NodeOp::Shader(shader)
        }
        other => other.clone(),
    }
}

fn recipe_of(spec: &TerrainSpec) -> RecipeMeta {
    let stacks = spec
        .fields
        .iter()
        .map(|field| {
            let mut graph = field.graph.clone();
            for node in &mut graph.nodes {
                node.op = stripped_op(&node.op);
            }
            FieldStack {
                field: field.id.clone(),
                range: field.range,
                export: field.export,
                graph,
            }
        })
        .collect();
    RecipeMeta {
        version: RECIPE_VERSION,
        water_spec: spec.water_spec.clone(),
        stacks,
    }
}

impl TerrainSpec {
    /// Writes the document to a directory: the values as a terrain, and beside them
    /// `recipe.ron`.
    ///
    /// The document is baked first if it is not already, because a terrain that
    /// carries no values is one no consumer can open.
    ///
    /// **Deletes files.** The values half removes any `layer_<n>.png` it no longer
    /// names, and [`SaveOptions::export`] removes the recipe itself. Nothing else is
    /// touched.
    ///
    /// Nothing about the document is changed by saving it.
    pub fn save_to_dir(
        &self,
        path: impl AsRef<Path>,
        options: SaveOptions,
    ) -> Result<(), RecipeError> {
        let root = path.as_ref().to_path_buf();
        let mut baked = self.clone();
        if baked
            .fields
            .iter()
            .any(|field| field.baked().size() != field.resolution(baked.size))
        {
            baked.bake_in_place()?;
        }
        let terrain = baked.clone().bake()?;
        terrain.save_to_dir(&root)?;

        let recipe = root.join(RECIPE_FILE);
        if options.recipe {
            let text =
                ron::ser::to_string_pretty(&recipe_of(&baked), ron::ser::PrettyConfig::default())
                    .map_err(|error| RecipeError::Meta(error.to_string()))?;
            std::fs::write(recipe, text.as_bytes())?;
        } else if recipe.exists() {
            std::fs::remove_file(recipe)?;
        }
        Ok(())
    }

    /// Reads a document from a directory and re-bakes it from its recipe, dispatching
    /// its shader nodes through `runtime`, then solves the water again when the recipe
    /// carries a water spec.
    ///
    /// The bake is redone rather than taken from the values: the carried values were
    /// quantised, and a document whose bakes came from them would disagree with itself
    /// the first time part of it was re-baked. Under a runtime holding no device every
    /// shader node reads `0.0`.
    ///
    /// A directory carrying no `recipe.ron` loads as a document with no nodes, which
    /// cannot be edited or saved again without losing what is left of it.
    pub fn load_from_dir(
        path: impl AsRef<Path>,
        runtime: ShaderRuntime,
    ) -> Result<Self, RecipeError> {
        let root = path.as_ref().to_path_buf();
        let terrain = Terrain::load_from_dir(&root)?;
        let recipe = read_recipe(&root)?;

        let mut spec = TerrainSpec::new(terrain.size());
        spec.set_shader_runtime(runtime);
        spec.water_spec = recipe.as_ref().and_then(|meta| meta.water_spec.clone());

        for view in terrain.fields() {
            let mut field = Field::new(view.name().to_owned())
                .with_role(view.role())
                .with_shift(view.shift());
            if let Some(stack) = recipe.as_ref().and_then(|meta| {
                meta.stacks
                    .iter()
                    .find(|entry| entry.field.as_str() == view.name())
            }) {
                field.range = stack.range;
                field.export = stack.export;
                field.graph = stack.graph.clone();
            }
            spec.fields.push(field);
        }

        if spec
            .fields
            .iter()
            .any(|field| !field.graph.nodes.is_empty())
        {
            spec.bake_in_place()?;
            if let Some(water) = spec.water_spec.clone() {
                spec.solve_water(&water)?;
            }
        }
        Ok(spec)
    }
}

fn read_recipe(root: &Path) -> Result<Option<RecipeMeta>, RecipeError> {
    let path = root.join(RECIPE_FILE);
    if !path.is_file() {
        return Ok(None);
    }
    let size = std::fs::metadata(&path)?.len();
    if size > MAX_RECIPE_BYTES {
        return Err(RecipeError::TooLarge(size));
    }
    let text = std::fs::read_to_string(&path)?;
    let meta: RecipeMeta =
        ron::from_str(&text).map_err(|error| RecipeError::Meta(error.to_string()))?;
    if meta.version != RECIPE_VERSION {
        return Err(RecipeError::UnsupportedVersion(meta.version));
    }
    Ok(Some(meta))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use glam::UVec2;
    use watershed::field::{FieldId, FieldRole};
    use watershed::meta::TerrainMeta;
    use watershed::raster::Raster;

    use crate::terrain::shader::ShaderLayer;

    const SIZE: UVec2 = UVec2::new(48, 32);

    fn ramp() -> Raster<f32> {
        let data = (0..(SIZE.x * SIZE.y) as usize)
            .map(|i| (i % 17) as f32 / 17.0)
            .collect();
        Raster::from_vec(SIZE, data).unwrap()
    }

    fn moisture() -> Field {
        Field::new("moisture")
            .with_shift(2)
            .with_op(NodeOp::holding(Raster::new(UVec2::new(12, 8), 0.5)))
    }

    fn graph_document() -> TerrainSpec {
        TerrainSpec::new(SIZE).with_field(moisture()).with_field(
            Field::new("height")
                .with_role(FieldRole::Height)
                .with_graph({
                    let NodeOp::Shader(mut layer) = NodeOp::holding(ramp()) else {
                        unreachable!("a held node is a shader node");
                    };
                    layer.inputs = vec!["in0".to_owned()];
                    let mut graph = FieldGraph::new();
                    let read = graph.node_with(NodeOp::FieldRef(FieldId::from("moisture")), &[]);
                    let shaded = graph.node_with(NodeOp::Shader(layer), &[read]);
                    graph.set_output(Some(shaded)).unwrap();
                    graph
                }),
        )
    }

    fn baked_document() -> TerrainSpec {
        let mut terrain = TerrainSpec::new(SIZE).with_field(moisture()).with_field(
            Field::new("height")
                .with_role(FieldRole::Height)
                .with_op(NodeOp::holding(ramp())),
        );
        terrain.bake_in_place().unwrap();
        terrain
    }

    fn load(root: &Path) -> Result<TerrainSpec, RecipeError> {
        TerrainSpec::load_from_dir(root, ShaderRuntime::default())
    }

    fn scratch(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "watershed-recipe-{}-{}-{name}",
            std::process::id(),
            RECIPE_VERSION
        ));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    fn saved(spec: &TerrainSpec, options: SaveOptions, name: &str) -> PathBuf {
        let root = scratch(name);
        spec.save_to_dir(&root, options).unwrap();
        root
    }

    fn values_meta(root: &Path) -> TerrainMeta {
        let text = std::fs::read_to_string(root.join("terrain.ron")).unwrap();
        ron::from_str(&text).unwrap()
    }

    // The shape of the split, and the first thing a person opening the directory sees:
    // the values name no stack, and the recipe is a file of its own beside them.
    #[test]
    fn the_values_carry_no_recipe_and_the_recipe_is_a_file_beside_them() {
        let root = saved(&graph_document(), SaveOptions::document(), "split");
        let text = std::fs::read_to_string(root.join("terrain.ron")).unwrap();

        assert!(root.join(RECIPE_FILE).is_file());
        assert!(!text.contains("stack"));
        std::fs::remove_dir_all(&root).unwrap();
    }

    // The recipe is what separates a document from an export, and an export has to be
    // readable while being unable to pose as something that can be edited again.
    #[test]
    fn an_export_carries_the_values_but_no_recipe() {
        let root = saved(&baked_document(), SaveOptions::export(), "export");

        assert!(!root.join(RECIPE_FILE).exists());
        assert!(Terrain::load_from_dir(&root).is_ok());
        let loaded = load(&root).unwrap();
        assert_eq!(loaded.fields.len(), 2);
        assert!(
            loaded
                .fields
                .iter()
                .all(|field| field.graph.nodes.is_empty())
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    // Exporting over a document has to take the recipe with it: one left behind would
    // claim to describe values it no longer produced.
    #[test]
    fn exporting_over_a_document_removes_the_recipe_it_replaces() {
        let root = saved(&graph_document(), SaveOptions::document(), "replace");
        assert!(root.join(RECIPE_FILE).is_file());

        baked_document()
            .save_to_dir(&root, SaveOptions::export())
            .unwrap();

        assert!(!root.join(RECIPE_FILE).exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    // A saved value has to land within the quantisation bound of the value that was
    // baked, or the format is losing more than the eight bits it admits to.
    #[test]
    fn a_baked_value_survives_the_round_trip_within_a_step() {
        let spec = baked_document();
        let before = spec.clone().bake().unwrap();
        let root = saved(&spec, SaveOptions::export(), "quantise");
        let after = Terrain::load_from_dir(&root).unwrap();

        for view in before.fields() {
            let mirror = after.field(view.name()).unwrap();
            let step = (view.range_high() - view.range_low()) / 255.0;
            for (x, y) in [(0, 0), (17, 9), (47, 31)] {
                let left = view.value_at(x, y).unwrap();
                let right = mirror.value_at(x, y).unwrap();
                assert!(
                    (left - right).abs() <= step / 2.0 + 1e-4,
                    "{} at {x},{y}: {left} became {right}",
                    view.name()
                );
            }
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    // Fields at the same shift share an image and one at another shift starts its own;
    // this is the packing rule as the format actually applies it.
    #[test]
    fn fields_sharing_a_shift_share_an_image() {
        let root = saved(&graph_document(), SaveOptions::document(), "packing");
        let meta = values_meta(&root);

        assert_eq!(meta.layers.len(), 2);
        assert!(meta.layers.iter().all(|layer| layer.shift.is_some()));
        assert!(
            meta.layers
                .iter()
                .all(|layer| layer.file.starts_with("layer_"))
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    // The water spec is carried even though the solved state is quantised, which is
    // what lets a document that was saved without water still grow it back.
    #[test]
    fn a_document_keeps_its_water_spec_through_a_save() {
        let mut spec = baked_document();
        spec.solve_water(&WaterSpec::new("height")).unwrap();
        let root = saved(&spec, SaveOptions::document(), "water");

        let loaded = load(&root).unwrap();
        assert_eq!(loaded.water_spec, Some(WaterSpec::new("height")));
        assert!(Terrain::load_from_dir(&root).unwrap().water().is_some());
        std::fs::remove_dir_all(&root).unwrap();
    }

    // No spec has to mean no water rather than a default solve, or every document that
    // never wanted water would acquire some on its first load.
    #[test]
    fn a_document_with_no_water_spec_loads_with_no_water() {
        let root = saved(&baked_document(), SaveOptions::document(), "dry");
        assert!(load(&root).unwrap().water().is_none());
        std::fs::remove_dir_all(&root).unwrap();
    }

    // A document is saved, loaded and saved again in the editor's own loop, so the loss
    // has to settle rather than compound: the second trip must move nothing the first
    // one did not already move.
    #[test]
    fn a_second_round_trip_moves_nothing_further() {
        let first_root = saved(&graph_document(), SaveOptions::document(), "trip-one");
        let first = load(&first_root).unwrap();
        let second_root = saved(&first, SaveOptions::document(), "trip-two");
        let second = load(&second_root).unwrap();

        for (left, right) in first.fields.iter().zip(&second.fields) {
            assert_eq!(left.baked().size(), right.baked().size());
            for (a, b) in left.baked().data().iter().zip(right.baked().data()) {
                assert_eq!(a.to_bits(), b.to_bits(), "a second trip moved {a} to {b}");
            }
        }
        std::fs::remove_dir_all(&first_root).unwrap();
        std::fs::remove_dir_all(&second_root).unwrap();
    }

    // A shader layer's recipe is the file it names and the values for its parameters —
    // and not the raster or the declared reach, both of which are derived from the
    // file and are re-read on the way in rather than saved.
    #[test]
    fn a_shader_layer_carries_its_file_and_its_parameters_and_no_values() {
        let mut shader = ShaderLayer::new("ridged.wgsl");
        shader.params.insert("scale".to_owned(), vec![0.03]);
        shader.reach = Some(2);
        shader.put_values(Raster::new(SIZE, 0.5));
        let spec = TerrainSpec::new(SIZE).with_field(
            Field::new("height")
                .with_role(FieldRole::Height)
                .with_op(NodeOp::Shader(shader)),
        );
        let root = saved(&spec, SaveOptions::document(), "shader");

        let loaded = load(&root).unwrap();
        match &loaded.fields[0].graph.nodes[0].op {
            NodeOp::Shader(shader) => {
                assert_eq!(shader.file, "ridged.wgsl");
                assert_eq!(shader.params.get("scale"), Some(&vec![0.03]));
                assert!(shader.values().is_empty(), "the values were serialized");
                assert_eq!(shader.reach, None, "the reach was serialized");
            }
            other => panic!("{other:?}"),
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    // There is no migration path for the recipe either, so a version this build does
    // not read is refused rather than parsed as far as it happens to agree.
    #[test]
    fn a_recipe_from_another_version_is_refused() {
        let root = saved(&baked_document(), SaveOptions::document(), "version");
        let text = std::fs::read_to_string(root.join(RECIPE_FILE)).unwrap();
        let mut recipe: RecipeMeta = ron::from_str(&text).unwrap();
        recipe.version = 99;
        let text = ron::ser::to_string_pretty(&recipe, ron::ser::PrettyConfig::default()).unwrap();
        std::fs::write(root.join(RECIPE_FILE), text).unwrap();

        assert!(matches!(
            load(&root),
            Err(RecipeError::UnsupportedVersion(99))
        ));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
