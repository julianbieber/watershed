//! The on-disk form of a document's recipe: `recipe.ron`, beside the values a
//! `watershed::Terrain` was written from.
//!
//! The values and the recipe are two files because they are two audiences. A
//! consuming project reads `terrain.ron` and never learns what a layer's shader is;
//! the editor reads both, and only the editor can. Nothing here is needed to read a
//! terrain, and a directory with this half deleted is still a terrain.
//!
//! A recipe read here is untrusted input, exactly as the values are: its size is
//! checked before it is read, and its version before anything in it is used.

use std::collections::BTreeMap;
use std::path::Path;

use glam::UVec2;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use watershed::io::IoError;
use watershed::terrain::Terrain;

use crate::gpu::ShaderRuntime;
use crate::terrain::LayerId;
use crate::terrain::bake::{BakeError, PlanError, TerrainSpec};
use crate::terrain::layer::Layer;
use crate::terrain::shader::SHADER_DIR;
use crate::terrain::water::{WaterError, WaterSpec};

/// The recipe file a terrain directory carries when it was saved as a document, and
/// the only name this reader knows without being told it.
pub const RECIPE_FILE: &str = "recipe.ron";

/// The recipe format this build writes, and the only one it reads.
///
/// Separate from [`watershed::meta::VERSION`], which versions the `terrain.ron` a
/// consuming project reads: the two files have two audiences and change for
/// different reasons, and a recipe that gains a layer must not invalidate every
/// terrain already exported.
///
/// A recipe carrying any other version is refused outright; there is no migration
/// path.
pub const RECIPE_VERSION: u32 = 5;

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
    /// `recipe.ron` is not readable as a recipe.
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
/// The recipe for one layer: everything needed to bake it again beside its shader
/// file, which is the layer's `shaders/<name>.wesl`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerRecipe {
    /// The layer's name, and the stem of its shader file.
    pub name: LayerId,
    /// Carried through and never read by the bake.
    #[serde(default)]
    pub export: bool,
    /// A value per parameter the layer's shader declares.
    pub params: BTreeMap<String, Vec<f32>>,
}

/// `recipe.ron` itself.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecipeMeta {
    /// The recipe format version, [`RECIPE_VERSION`] — not the version of the
    /// `terrain.ron` beside it. Checked before anything else is read.
    pub version: u32,
    /// The document's extent in cells.
    pub size: UVec2,
    /// The seed every shader of the document is told about.
    pub seed: u32,
    /// The spec the water was solved from, kept so a document that carries no
    /// solved water can still be re-solved rather than losing it.
    pub water_spec: Option<WaterSpec>,
    /// One entry per layer, in declaration order. Written as `fields` in `recipe.ron`.
    #[serde(rename = "fields")]
    pub layers: Vec<LayerRecipe>,
}

#[derive(Deserialize)]
struct RecipeVersion {
    version: u32,
}

fn recipe_of(spec: &TerrainSpec) -> RecipeMeta {
    RecipeMeta {
        version: RECIPE_VERSION,
        size: spec.size,
        seed: spec.seed,
        water_spec: spec.water_spec.clone(),
        layers: spec
            .layers
            .iter()
            .map(|layer| LayerRecipe {
                name: layer.id.clone(),
                export: layer.export,
                params: layer.shader.params.clone(),
            })
            .collect(),
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
            .layers
            .iter()
            .any(|layer| layer.baked().size() != layer.resolution(baked.size))
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
    /// each layer's shader from the directory's `shaders` through `base`'s device at
    /// the recipe's seed, then solves the water again when the recipe carries a water
    /// spec.
    ///
    /// The bake is redone rather than taken from the values: the carried values were
    /// quantised, and a document whose bakes came from them would disagree with itself
    /// the first time part of it was re-baked. Under a runtime holding no device every
    /// layer reads `0.0`.
    ///
    /// A layer's role, shift, range and class come from its shader file either way —
    /// the recipe carries none of the four — so a directory whose shaders cannot be
    /// read loads its layers at their defaults.
    ///
    /// A directory carrying no `recipe.ron` loads its layers' names from the values,
    /// at seed 0 and with no parameter values.
    pub fn load_from_dir(path: impl AsRef<Path>, base: ShaderRuntime) -> Result<Self, RecipeError> {
        let root = path.as_ref().to_path_buf();
        let terrain = Terrain::load_from_dir(&root)?;
        let recipe = read_recipe(&root)?;

        let mut spec = TerrainSpec::new(terrain.size());
        match &recipe {
            Some(meta) => {
                spec.size = meta.size;
                spec.seed = meta.seed;
                spec.water_spec = meta.water_spec.clone();
                for entry in &meta.layers {
                    let mut layer = Layer::new(entry.name.clone()).with_export(entry.export);
                    layer.shader.params = entry.params.clone();
                    spec.layers.push(layer);
                }
            }
            None => {
                for view in terrain.fields() {
                    spec.layers.push(Layer::new(view.name().to_owned()));
                }
            }
        }

        let runtime = base.with_directory(spec.seed, &root.join(SHADER_DIR));
        for layer in &mut spec.layers {
            if let Some(program) = runtime.program(&layer.file()) {
                layer.shader.reconcile(&program.layout);
                layer.shader.reconcile_layers(&program.layers);
                layer.reconcile_header(&program.header);
            }
        }
        spec.set_shader_runtime(runtime);

        if !spec.layers.is_empty() {
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
    let version: RecipeVersion =
        ron::from_str(&text).map_err(|error| RecipeError::Meta(error.to_string()))?;
    if version.version != RECIPE_VERSION {
        return Err(RecipeError::UnsupportedVersion(version.version));
    }
    let meta: RecipeMeta =
        ron::from_str(&text).map_err(|error| RecipeError::Meta(error.to_string()))?;
    Ok(Some(meta))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::terrain::LayerRole;
    use watershed::meta::TerrainMeta;
    use watershed::raster::Raster;

    const SIZE: UVec2 = UVec2::new(48, 32);

    fn ramp() -> Raster<f32> {
        let data = (0..(SIZE.x * SIZE.y) as usize)
            .map(|i| (i % 17) as f32 / 17.0)
            .collect();
        Raster::from_vec(SIZE, data).unwrap()
    }
    fn moisture() -> Layer {
        Layer::new("moisture")
            .with_shift(2)
            .holding(Raster::new(UVec2::new(12, 8), 0.5))
    }

    fn reading_document() -> TerrainSpec {
        TerrainSpec::new(SIZE).with_layer(moisture()).with_layer(
            Layer::new("height")
                .with_role(LayerRole::Height)
                .holding(ramp())
                .reading(&["moisture"]),
        )
    }

    fn baked_document() -> TerrainSpec {
        let mut terrain = TerrainSpec::new(SIZE).with_layer(moisture()).with_layer(
            Layer::new("height")
                .with_role(LayerRole::Height)
                .holding(ramp()),
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
    // the values name no parameters, and the recipe is a file of its own beside them.
    #[test]
    fn the_values_carry_no_recipe_and_the_recipe_is_a_file_beside_them() {
        let root = saved(&reading_document(), SaveOptions::document(), "split");
        let text = std::fs::read_to_string(root.join("terrain.ron")).unwrap();

        assert!(root.join(RECIPE_FILE).is_file());
        assert!(!text.contains("params"));
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
        assert_eq!(loaded.layers.len(), 2);
        assert!(
            loaded
                .layers
                .iter()
                .all(|layer| layer.shader.params.is_empty())
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    // Exporting over a document has to take the recipe with it: one left behind would
    // claim to describe values it no longer produced.
    #[test]
    fn exporting_over_a_document_removes_the_recipe_it_replaces() {
        let root = saved(&reading_document(), SaveOptions::document(), "replace");
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

    // Layers at the same shift share an image and one at another shift starts its own;
    // this is the packing rule as the format actually applies it.
    #[test]
    fn layers_sharing_a_shift_share_an_image() {
        let root = saved(&reading_document(), SaveOptions::document(), "packing");
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
        let first_root = saved(&reading_document(), SaveOptions::document(), "trip-one");
        let first = load(&first_root).unwrap();
        let second_root = saved(&first, SaveOptions::document(), "trip-two");
        let second = load(&second_root).unwrap();

        for (left, right) in first.layers.iter().zip(&second.layers) {
            assert_eq!(left.baked().size(), right.baked().size());
            for (a, b) in left.baked().data().iter().zip(right.baked().data()) {
                assert_eq!(a.to_bits(), b.to_bits(), "a second trip moved {a} to {b}");
            }
        }
        std::fs::remove_dir_all(&first_root).unwrap();
        std::fs::remove_dir_all(&second_root).unwrap();
    }
    // A layer's recipe is its export flag, the document's seed and the values for its
    // parameters — not the raster, which is derived from the file and re-made on the
    // way in, and not the four properties the shader file now declares, which would
    // be a second copy for the file to disagree with.
    #[test]
    fn a_layer_carries_its_parameters_and_seed_and_no_values_or_declared_properties() {
        let mut height = Layer::new("height")
            .with_role(LayerRole::Height)
            .with_range((-2.0, 3.0))
            .holding(Raster::new(SIZE, 0.5));
        height.shader.params.insert("scale".to_owned(), vec![0.03]);
        let mut spec = TerrainSpec::new(SIZE).with_layer(height);
        spec.seed = 42;
        let root = saved(&spec, SaveOptions::document(), "shader");

        let written = std::fs::read_to_string(root.join(RECIPE_FILE)).unwrap();
        for key in ["role", "shift", "range", "categorical"] {
            assert!(
                !written.contains(key),
                "the recipe carries `{key}`:\n{written}"
            );
        }

        let loaded = load(&root).unwrap();
        assert_eq!(loaded.seed, 42);
        let layer = &loaded.layers[0];
        assert_eq!(layer.shader.params.get("scale"), Some(&vec![0.03]));
        assert!(
            layer.shader.values().is_empty(),
            "the values were serialized"
        );
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
    // A document saved while a layer carried its own role, shift and range has no
    // reading here, so it is refused by its version before the disagreeing shape of
    // the rest is parsed, and the message says which version it was.
    #[test]
    fn a_version_four_recipe_is_refused_naming_its_version() {
        let root = saved(&baked_document(), SaveOptions::document(), "version-four");
        std::fs::write(
            root.join(RECIPE_FILE),
            "(version: 4, size: (64, 64), seed: 0, water_spec: None, \
             fields: [(name: (\"height\"), role: Height, shift: 0, range: (0.0, 1.0), params: {})])",
        )
        .unwrap();

        let error = load(&root).unwrap_err();
        assert!(
            matches!(error, RecipeError::UnsupportedVersion(4)),
            "{error}"
        );
        assert!(error.to_string().contains('4'), "{error}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    // Pins the on-disk key the editor's rename to layer must not move: a recipe saved
    // before it still loads, and one saved after it still says `fields`.
    #[test]
    fn a_recipe_keeps_its_fields_key_on_disk() {
        let written = ron::ser::to_string_pretty(
            &recipe_of(&baked_document()),
            ron::ser::PrettyConfig::default(),
        )
        .unwrap();
        assert!(written.contains("fields:"), "{written}");
        assert!(!written.contains("layers:"), "{written}");

        let text = format!(
            "(version: {RECIPE_VERSION}, size: (64, 64), seed: 0, water_spec: None, \
             fields: [(name: (\"height\"), params: {{}})])"
        );
        let meta: RecipeMeta = ron::from_str(&text).unwrap();
        assert_eq!(meta.layers.len(), 1);
        assert_eq!(meta.layers[0].name, LayerId::from("height"));
    }
}
