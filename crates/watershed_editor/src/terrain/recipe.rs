//! The on-disk form of a document's recipe: `recipe.ron` and the painted images it
//! names, beside the values a `watershed::Terrain` was written from.
//!
//! The values and the recipe are two files because they are two audiences. A
//! consuming project reads `terrain.ron` and never learns what a field graph is; the
//! editor reads both, and only the editor can. Nothing here is needed to read a
//! terrain, and a directory with this half deleted is still a terrain.
//!
//! A recipe read here is untrusted input, exactly as the values are: every name is
//! checked to be a name rather than a path, and every index is checked against what
//! was actually loaded.

use std::io::Cursor;
use std::path::{Component, Path};

use glam::UVec2;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use watershed::channel::ChannelMeta;
use watershed::field::FieldId;
use watershed::io::IoError;
use watershed::meta::LayerMeta;
use watershed::raster::Raster;
use watershed::terrain::Terrain;

use crate::terrain::bake::{BakeError, PlanError, TerrainSpec};
use crate::terrain::field::Field;
use crate::terrain::graph::{FieldGraph, NodeId, NodeOp};
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

/// The most painted images one recipe may name.
pub const MAX_PAINT_IMAGES: usize = 1024;

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
    /// A file the recipe names is not a plain name inside the directory.
    #[error("`{0}` is not a file name inside the terrain")]
    BadFileName(String),
    /// A painted image could not be encoded or decoded.
    #[error("`{file}`: {reason}")]
    Image {
        /// The file that failed.
        file: String,
        /// Why.
        reason: String,
    },
    /// The recipe is well-formed but does not hang together — a stack position past
    /// the layers it declares, an image index past the ones it names, a slot that
    /// does not take paint.
    #[error("the recipe does not hang together: {0}")]
    Inconsistent(String),
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
    /// Write `recipe.ron` and the painted images it names. Without this the terrain
    /// has values but no recipe, and cannot be opened for editing again.
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

/// A painted raster's home, stated rather than implied by position.
///
/// The recipe is hand-editable, so an order both halves would have to agree on
/// without saying so is the one thing left that a reader would have to derive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PaintRef {
    /// The node holding it, by [`NodeId`].
    pub node: u32,
    /// The image holding it, as an index into [`RecipeMeta::images`].
    pub image: u32,
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
    /// The graph, with every raster in it emptied out.
    pub graph: FieldGraph,
    /// Where each emptied raster went.
    pub paint: Vec<PaintRef>,
}

/// `recipe.ron` itself.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecipeMeta {
    /// The recipe format version, [`RECIPE_VERSION`] — not the version of the
    /// `terrain.ron` beside it. Checked before anything else is read.
    pub version: u32,
    /// Every painted image, in the order [`PaintRef::image`] indexes them.
    pub images: Vec<LayerMeta>,
    /// The spec the water was solved from, kept so a document that carries no
    /// solved water can still be re-solved rather than losing it.
    pub water_spec: Option<WaterSpec>,
    /// One entry per field that has a graph.
    pub stacks: Vec<FieldStack>,
}

fn paint_file(index: usize) -> String {
    format!("paint_{index:03}.png")
}

fn is_paint_file(name: &str) -> bool {
    name.strip_prefix("paint_")
        .and_then(|rest| rest.strip_suffix(".png"))
        .is_some_and(|digits| {
            !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
        })
}

fn check_file_name(name: &str) -> Result<(), RecipeError> {
    let path = Path::new(name);
    let mut components = path.components();
    let single =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    if !single || name.contains('\0') {
        return Err(RecipeError::BadFileName(name.to_owned()));
    }
    Ok(())
}

fn encode_gray(size: UVec2, bytes: &[u8], file: &str) -> Result<Vec<u8>, RecipeError> {
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(Cursor::new(&mut out), size.x, size.y);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|error| RecipeError::Image {
        file: file.to_owned(),
        reason: error.to_string(),
    })?;
    writer
        .write_image_data(bytes)
        .map_err(|error| RecipeError::Image {
            file: file.to_owned(),
            reason: error.to_string(),
        })?;
    writer.finish().map_err(|error| RecipeError::Image {
        file: file.to_owned(),
        reason: error.to_string(),
    })?;
    Ok(out)
}

fn decode_gray(file: &str, bytes: &[u8], size: UVec2) -> Result<Vec<u8>, RecipeError> {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder.read_info().map_err(|error| RecipeError::Image {
        file: file.to_owned(),
        reason: error.to_string(),
    })?;
    let info = reader.info();
    if info.width != size.x || info.height != size.y {
        return Err(RecipeError::Image {
            file: file.to_owned(),
            reason: format!(
                "is {}x{}, where the recipe declares {}x{}",
                info.width, info.height, size.x, size.y
            ),
        });
    }
    if info.color_type != png::ColorType::Grayscale || info.bit_depth != png::BitDepth::Eight {
        return Err(RecipeError::Image {
            file: file.to_owned(),
            reason: "is not eight-bit greyscale".to_owned(),
        });
    }
    let mut out = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let frame = reader
        .next_frame(&mut out)
        .map_err(|error| RecipeError::Image {
            file: file.to_owned(),
            reason: error.to_string(),
        })?;
    out.truncate(frame.buffer_size());
    Ok(out)
}

fn stripped_op(op: &NodeOp) -> NodeOp {
    match op {
        NodeOp::Paint(_) => NodeOp::Paint(Raster::default()),
        NodeOp::External(_) => NodeOp::External(Raster::default()),
        NodeOp::Shader(shader) => {
            let mut shader = shader.clone();
            shader.clear();
            NodeOp::Shader(shader)
        }
        other => other.clone(),
    }
}

struct PaintedImage {
    meta: LayerMeta,
    bytes: Vec<u8>,
}

fn external_image(index: usize, raster: &Raster<f32>) -> PaintedImage {
    let finite: Vec<f32> = raster
        .data()
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let low = finite.iter().copied().fold(f32::INFINITY, f32::min);
    let high = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let meta = if finite.is_empty() {
        ChannelMeta::linear(0.0, 0.0)
    } else {
        ChannelMeta::linear(low, high)
    };
    let bytes = raster
        .data()
        .iter()
        .map(|value| meta.encode(*value))
        .collect();
    PaintedImage {
        meta: LayerMeta {
            file: paint_file(index),
            width: raster.width(),
            height: raster.height(),
            shift: None,
            channels: vec![meta],
        },
        bytes,
    }
}

fn painted_image(index: usize, raster: &Raster<u8>) -> PaintedImage {
    PaintedImage {
        meta: LayerMeta {
            file: paint_file(index),
            width: raster.width(),
            height: raster.height(),
            shift: None,
            channels: vec![ChannelMeta::raw()],
        },
        bytes: raster.data().to_vec(),
    }
}

/// The recipe of a document, with every painted raster lifted out into its own
/// image.
fn recipe_of(spec: &TerrainSpec) -> (RecipeMeta, Vec<PaintedImage>) {
    let mut images = Vec::new();
    let mut stacks = Vec::with_capacity(spec.fields.len());
    for field in &spec.fields {
        let mut paint = Vec::new();
        let mut graph = field.graph.clone();
        for node in &mut graph.nodes {
            match &node.op {
                NodeOp::Paint(raster) if !raster.is_empty() => {
                    images.push(painted_image(images.len(), raster));
                }
                NodeOp::External(raster) if !raster.is_empty() => {
                    images.push(external_image(images.len(), raster));
                }
                _ => {
                    node.op = stripped_op(&node.op);
                    continue;
                }
            }
            paint.push(PaintRef {
                node: node.id.0,
                image: images.len() as u32 - 1,
            });
            node.op = stripped_op(&node.op);
        }
        stacks.push(FieldStack {
            field: field.id.clone(),
            range: field.range,
            export: field.export,
            graph,
            paint,
        });
    }
    let meta = RecipeMeta {
        version: RECIPE_VERSION,
        images: images.iter().map(|image| image.meta.clone()).collect(),
        water_spec: spec.water_spec.clone(),
        stacks,
    };
    (meta, images)
}

impl TerrainSpec {
    /// Writes the document to a directory: the values as a terrain, and beside them
    /// `recipe.ron` and one image per painted raster.
    ///
    /// The document is baked first if it is not already, because a terrain that
    /// carries no values is one no consumer can open.
    ///
    /// **Deletes files.** The values half removes any `layer_<n>.png` it no longer
    /// names, and this half removes any `paint_<n>.png` the recipe no longer names —
    /// and, for [`SaveOptions::export`], the recipe itself. Nothing outside those two
    /// naming schemes is touched.
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

        let named = if options.recipe {
            let (meta, images) = recipe_of(&baked);
            for image in &images {
                let size = UVec2::new(image.meta.width, image.meta.height);
                let bytes = encode_gray(size, &image.bytes, &image.meta.file)?;
                std::fs::write(root.join(&image.meta.file), bytes)?;
            }
            let text = ron::ser::to_string_pretty(&meta, ron::ser::PrettyConfig::default())
                .map_err(|error| RecipeError::Meta(error.to_string()))?;
            std::fs::write(root.join(RECIPE_FILE), text.as_bytes())?;
            meta.images
                .iter()
                .map(|image| image.file.clone())
                .collect::<Vec<_>>()
        } else {
            let recipe = root.join(RECIPE_FILE);
            if recipe.exists() {
                std::fs::remove_file(recipe)?;
            }
            Vec::new()
        };

        for entry in std::fs::read_dir(&root)? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            if is_paint_file(&name) && !named.contains(&name) {
                std::fs::remove_file(root.join(&name))?;
            }
        }
        Ok(())
    }

    /// Reads a document from a directory, re-attaching the painted rasters its
    /// recipe names and re-baking from them.
    ///
    /// The bake is redone rather than taken from the images: the carried values were
    /// quantised, and a document whose paint is the decoded paint but whose bakes
    /// came from the original would disagree with itself the first time part of it
    /// was re-baked.
    ///
    /// A directory carrying no `recipe.ron` loads as a document with no layers, which
    /// cannot be edited or saved again without losing what is left of it.
    pub fn load_from_dir(path: impl AsRef<Path>) -> Result<Self, RecipeError> {
        let root = path.as_ref().to_path_buf();
        let terrain = Terrain::load_from_dir(&root)?;
        let recipe = read_recipe(&root)?;

        let mut spec = TerrainSpec::new(terrain.size());
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
                let meta = recipe.as_ref().expect("a stack came from a recipe");
                field.range = stack.range;
                field.export = stack.export;
                field.graph = stack.graph.clone();
                attach_paint(&root, meta, stack, &mut field)?;
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
    if meta.images.len() > MAX_PAINT_IMAGES {
        return Err(RecipeError::Inconsistent(format!(
            "{} painted images is over the {MAX_PAINT_IMAGES} limit",
            meta.images.len()
        )));
    }
    for image in &meta.images {
        check_file_name(&image.file)?;
    }
    Ok(Some(meta))
}

fn attach_paint(
    root: &Path,
    meta: &RecipeMeta,
    stack: &FieldStack,
    field: &mut Field,
) -> Result<(), RecipeError> {
    for reference in &stack.paint {
        let declared = meta.images.get(reference.image as usize).ok_or_else(|| {
            RecipeError::Inconsistent(format!(
                "`{}` names image {}, of {}",
                field.id,
                reference.image,
                meta.images.len()
            ))
        })?;
        let target = field
            .graph
            .node_mut(NodeId(reference.node))
            .ok_or_else(|| {
                RecipeError::Inconsistent(format!(
                    "`{}` names node n{}, which its graph does not have",
                    field.id, reference.node
                ))
            })?;
        let size = UVec2::new(declared.width, declared.height);
        let bytes = std::fs::read(root.join(&declared.file))?;
        let pixels = decode_gray(&declared.file, &bytes, size)?;
        let channel = declared.channels.first().copied().ok_or_else(|| {
            RecipeError::Inconsistent(format!("`{}` declares no channel", declared.file))
        })?;
        let short = || RecipeError::Inconsistent(format!("`{}` is short", declared.file));
        match &mut target.op {
            NodeOp::Paint(slot) => {
                *slot = Raster::from_vec(size, pixels).ok_or_else(short)?;
            }
            NodeOp::External(slot) => {
                let table = channel.table();
                let data: Vec<f32> = pixels.iter().map(|byte| table.get(*byte)).collect();
                *slot = Raster::from_vec(size, data).ok_or_else(short)?;
            }
            _ => {
                return Err(RecipeError::Inconsistent(format!(
                    "`{}` node n{} does not take paint",
                    field.id, reference.node
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use watershed::field::{FieldId, FieldRole};
    use watershed::meta::TerrainMeta;

    use crate::terrain::graph::Remap;
    use crate::terrain::noise::{NoiseKind, NoiseSpec};
    use crate::terrain::shader::ShaderLayer;

    const SIZE: UVec2 = UVec2::new(48, 32);

    fn cells() -> usize {
        (SIZE.x * SIZE.y) as usize
    }

    fn ramp() -> Raster<f32> {
        let data = (0..cells()).map(|i| (i % 17) as f32 / 17.0).collect();
        Raster::from_vec(SIZE, data).unwrap()
    }

    fn awkward_ramp() -> Raster<f32> {
        let mut data: Vec<f32> = (0..cells()).map(|i| (i % 17) as f32 / 17.0).collect();
        data[5] = f32::NAN;
        data[6] = -0.0;
        data[7] = f32::INFINITY;
        data[8] = f32::NEG_INFINITY;
        data[9] = f32::MIN_POSITIVE / 3.0;
        Raster::from_vec(SIZE, data).unwrap()
    }

    /// The same awkward extent as [`awkward_ramp`], as the bytes a paint node keeps.
    fn awkward_bytes() -> Raster<u8> {
        let size = UVec2::new(SIZE.x + 3, SIZE.y - 1);
        let count = (size.x * size.y) as usize;
        let data = (0..count).map(|i| (i % 251) as u8).collect();
        Raster::from_vec(size, data).unwrap()
    }

    /// The same ramp as [`ramp`], as the bytes a paint node keeps.
    fn byte_ramp() -> Raster<u8> {
        let data = (0..cells())
            .map(|i| ((i % SIZE.x as usize) as f32 / SIZE.x as f32 * 255.0) as u8)
            .collect();
        Raster::from_vec(SIZE, data).unwrap()
    }

    fn byte_mask() -> Raster<u8> {
        let data = (0..cells()).map(|i| (i % 251) as u8).collect();
        Raster::from_vec(SIZE, data).unwrap()
    }

    fn noise_op(seed: u32) -> NodeOp {
        NodeOp::Noise(NoiseSpec::new(seed, NoiseKind::Fbm, 0.05))
    }

    /// A graph carrying every kind of raster the recipe has to lift out and put back:
    /// a painted one, an imported one, and a painted weight.
    fn painted_document() -> TerrainSpec {
        TerrainSpec::new(SIZE)
            .with_field(Field::new("moisture").with_shift(2).with_op(noise_op(3)))
            .with_field(
                Field::new("height")
                    .with_role(FieldRole::Height)
                    .with_graph({
                        let mut graph = FieldGraph::new();
                        let painted = graph.node_with(NodeOp::Paint(awkward_bytes()), &[]);
                        let imported = graph.node_with(NodeOp::External(ramp()), &[]);
                        let weight = graph.node_with(NodeOp::Paint(byte_mask()), &[]);
                        let over = graph.node_with(NodeOp::Lerp, &[painted, imported, weight]);
                        let detail = graph.node_with(noise_op(9), &[]);
                        let read =
                            graph.node_with(NodeOp::FieldRef(FieldId::from("moisture")), &[]);
                        let band = graph
                            .node_with(NodeOp::Remap(Remap::new((0.3, 0.7), (0.0, 1.0))), &[read]);
                        let total = graph.node_with(NodeOp::Lerp, &[over, detail, band]);
                        graph.set_output(Some(total)).unwrap();
                        graph
                    }),
            )
    }

    fn baked_document() -> TerrainSpec {
        let mut terrain = TerrainSpec::new(SIZE)
            .with_field(Field::new("moisture").with_shift(2).with_op(noise_op(3)))
            .with_field(
                Field::new("height")
                    .with_role(FieldRole::Height)
                    .with_sum([NodeOp::Paint(byte_ramp()), noise_op(9)]),
            );
        terrain.bake_in_place().unwrap();
        terrain
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
        let root = saved(&painted_document(), SaveOptions::document(), "split");
        let text = std::fs::read_to_string(root.join("terrain.ron")).unwrap();

        assert!(root.join(RECIPE_FILE).is_file());
        assert!(!text.contains("stack"));
        assert!(!text.contains("paint_"));
        std::fs::remove_dir_all(&root).unwrap();
    }

    // The recipe is what separates a document from an export, and an export has to be
    // readable while being unable to pose as something that can be edited again.
    #[test]
    fn an_export_carries_the_values_but_no_recipe() {
        let root = saved(&baked_document(), SaveOptions::export(), "export");

        assert!(!root.join(RECIPE_FILE).exists());
        assert!(Terrain::load_from_dir(&root).is_ok());
        let loaded = TerrainSpec::load_from_dir(&root).unwrap();
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
        let root = saved(&painted_document(), SaveOptions::document(), "replace");
        assert!(root.join(RECIPE_FILE).is_file());

        baked_document()
            .save_to_dir(&root, SaveOptions::export())
            .unwrap();

        assert!(!root.join(RECIPE_FILE).exists());
        assert!(
            !std::fs::read_dir(&root).unwrap().any(|entry| entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("paint_")),
            "an export left a painted image behind"
        );
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

    // A painted raster is stretched over the document from whatever size it was
    // authored at, so applying the bake layers' extent rule to it would make every
    // paint layer unloadable.
    #[test]
    fn a_painted_image_carries_its_own_extent_and_no_shift() {
        let odd = Raster::from_vec(UVec2::new(7, 5), vec![128u8; 35]).unwrap();
        let spec =
            TerrainSpec::new(SIZE).with_field(Field::new("height").with_op(NodeOp::Paint(odd)));
        let root = saved(&spec, SaveOptions::document(), "odd-paint");

        let text = std::fs::read_to_string(root.join(RECIPE_FILE)).unwrap();
        let recipe: RecipeMeta = ron::from_str(&text).unwrap();
        assert_eq!((recipe.images[0].width, recipe.images[0].height), (7, 5));
        assert_eq!(recipe.images[0].shift, None);

        let loaded = TerrainSpec::load_from_dir(&root).unwrap();
        match &loaded.fields[0].graph.nodes[0].op {
            NodeOp::Paint(raster) => assert_eq!(raster.size(), UVec2::new(7, 5)),
            other => panic!("{other:?}"),
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    // A mask is already bytes, so it is the one thing in the format that comes back
    // exactly — and it does so because of its encoding, not by luck.
    #[test]
    fn a_painted_mask_round_trips_byte_for_byte() {
        let root = saved(&painted_document(), SaveOptions::document(), "mask");
        let loaded = TerrainSpec::load_from_dir(&root).unwrap();

        let mask = loaded
            .fields
            .iter()
            .flat_map(|field| &field.graph.nodes)
            .find_map(|node| match &node.op {
                NodeOp::Paint(raster) if raster == &byte_mask() => Some(raster),
                _ => None,
            })
            .expect("the painted weight came back");
        assert_eq!(mask, &byte_mask());
        std::fs::remove_dir_all(&root).unwrap();
    }

    // Fields at the same shift share an image and one at another shift starts its own;
    // this is the packing rule as the format actually applies it, and the recipe's own
    // images must not join that pool.
    #[test]
    fn fields_sharing_a_shift_share_an_image_and_paint_is_not_one_of_them() {
        let root = saved(&painted_document(), SaveOptions::document(), "packing");
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

        let loaded = TerrainSpec::load_from_dir(&root).unwrap();
        assert_eq!(loaded.water_spec, Some(WaterSpec::new("height")));
        assert!(Terrain::load_from_dir(&root).unwrap().water().is_some());
        std::fs::remove_dir_all(&root).unwrap();
    }

    // No spec has to mean no water rather than a default solve, or every document that
    // never wanted water would acquire some on its first load.
    #[test]
    fn a_document_with_no_water_spec_loads_with_no_water() {
        let root = saved(&baked_document(), SaveOptions::document(), "dry");
        assert!(TerrainSpec::load_from_dir(&root).unwrap().water().is_none());
        std::fs::remove_dir_all(&root).unwrap();
    }

    // A document is saved, loaded and saved again in the editor's own loop, so the loss
    // has to settle rather than compound: the second trip must move nothing the first
    // one did not already move.
    #[test]
    fn a_second_round_trip_moves_nothing_further() {
        let first_root = saved(&painted_document(), SaveOptions::document(), "trip-one");
        let first = TerrainSpec::load_from_dir(&first_root).unwrap();
        let second_root = saved(&first, SaveOptions::document(), "trip-two");
        let second = TerrainSpec::load_from_dir(&second_root).unwrap();

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

        let loaded = TerrainSpec::load_from_dir(&root).unwrap();
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

    // The recipe is untrusted input like everything else: an index out of a hand-edited
    // file has to be refused rather than panicking.
    #[test]
    fn a_recipe_naming_an_image_it_does_not_have_is_refused() {
        let root = saved(&painted_document(), SaveOptions::document(), "bad-index");
        let text = std::fs::read_to_string(root.join(RECIPE_FILE)).unwrap();
        let mut recipe: RecipeMeta = ron::from_str(&text).unwrap();
        for stack in &mut recipe.stacks {
            for reference in &mut stack.paint {
                reference.image = 99;
            }
        }
        let text = ron::ser::to_string_pretty(&recipe, ron::ser::PrettyConfig::default()).unwrap();
        std::fs::write(root.join(RECIPE_FILE), text).unwrap();

        assert!(matches!(
            TerrainSpec::load_from_dir(&root),
            Err(RecipeError::Inconsistent(_))
        ));
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
            TerrainSpec::load_from_dir(&root),
            Err(RecipeError::UnsupportedVersion(99))
        ));
        std::fs::remove_dir_all(&root).unwrap();
    }

    // A name with a separator in it would be joined onto the directory and escape it,
    // so it is refused before it is ever turned into a path.
    #[test]
    fn a_recipe_naming_a_path_rather_than_a_file_is_refused() {
        let root = saved(&painted_document(), SaveOptions::document(), "path-name");
        let text = std::fs::read_to_string(root.join(RECIPE_FILE)).unwrap();
        let mut recipe: RecipeMeta = ron::from_str(&text).unwrap();
        recipe.images[0].file = "../escaped.png".to_owned();
        let text = ron::ser::to_string_pretty(&recipe, ron::ser::PrettyConfig::default()).unwrap();
        std::fs::write(root.join(RECIPE_FILE), text).unwrap();

        assert!(matches!(
            TerrainSpec::load_from_dir(&root),
            Err(RecipeError::BadFileName(_))
        ));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
