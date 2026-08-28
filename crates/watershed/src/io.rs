//! The on-disk form of a terrain: a directory holding `terrain.ron` and the images
//! it names, what is left to be re-derived on load, and what a reader is allowed to
//! assume about a directory it did not write.
//!
//! A terrain read here is untrusted input — it may have been written by hand, or by
//! someone else — so every size a file declares is checked against what the
//! metadata already said before it is allocated against, and every name is checked
//! to be a name rather than a path.

use std::collections::BTreeSet;
use std::io::Cursor;
use std::path::{Component, Path, PathBuf};

use glam::UVec2;
use thiserror::Error;

use crate::bake::{BakeError, PlanError, TerrainSpec};
use crate::channel::{ChannelError, ChannelMeta, MAX_CHANNELS};
use crate::field::Field;
use crate::layer::{LayerOp, Mask};
use crate::meta::{FieldStack, LayerMeta, PaintRef, PaintSlot, TerrainMeta, VERSION};
use crate::raster::{MAX_SHIFT, Raster, resolution};
use crate::terrain::{LayerTexels, Terrain, TerrainLayer};
use crate::water::WaterError;

/// The metadata file every terrain directory holds, and the only name a reader
/// knows without being told it.
pub const META_FILE: &str = "terrain.ron";

/// The largest `terrain.ron` this build will read, checked before the file is read
/// rather than after.
pub const MAX_META_BYTES: u64 = 64 * 1024 * 1024;

/// The most images one terrain may hold.
pub const MAX_LAYERS: usize = 1024;

/// The most cells a terrain may declare, which bounds every allocation derived from
/// its extent.
pub const MAX_CELLS: u64 = 1 << 30;

/// The most image data one load will decode, across every image together.
pub const MAX_DECODED_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Why a terrain could not be written or read.
///
/// A load fails on the first of these it meets, in the order the metadata lists
/// things, and produces no terrain — so a caller never gets a partially read one.
#[derive(Debug, Error)]
pub enum IoError {
    /// The underlying reader or writer. A missing directory arrives here.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// The metadata is not readable as a terrain.
    #[error("{META_FILE} is not readable: {0}")]
    Meta(String),
    /// `terrain.ron` is larger than [`MAX_META_BYTES`]. Checked before it is read.
    #[error("{META_FILE} is {0} bytes, over the {MAX_META_BYTES} byte limit")]
    MetaTooLarge(u64),
    /// A terrain this build cannot read. There is no migration path.
    #[error("terrain version {0} is not supported; this build reads {VERSION}")]
    UnsupportedVersion(u32),
    /// A file named by the metadata is not a plain name inside the directory, or is
    /// not a regular file. A name is never joined onto the directory before this
    /// has passed: an absolute one would discard the directory entirely.
    #[error("`{0}` is not a file name inside the terrain")]
    BadFileName(String),
    /// An image declares a size, a channel count or a bit depth that is not what
    /// the metadata said it would. Checked against the image's header, before its
    /// pixels are allocated.
    #[error("`{file}` is {found}, where the metadata declares {expected}")]
    ImageShape {
        /// The file that disagreed.
        file: String,
        /// What its header says.
        found: String,
        /// What the metadata requires.
        expected: String,
    },
    /// An image could not be decoded.
    #[error("`{file}` could not be decoded: {source}")]
    Decode {
        /// The file that failed.
        file: String,
        /// Why.
        source: png::DecodingError,
    },
    /// The metadata is well-formed but does not hang together — an index past the
    /// layers that were loaded, a field whose shift is not its layer's, a water
    /// image that is not four channels, an extent or a shift outside what this
    /// build will carry.
    #[error("the terrain does not hang together: {0}")]
    Inconsistent(String),
    /// A terrain with no readable field. Nothing re-derives a bake on the way in,
    /// so this is refused rather than loaded as a terrain answering nothing.
    #[error("the terrain carries no readable field")]
    NoFields,
    /// A channel could not be read. See [`ChannelError`].
    #[error(transparent)]
    Channel(#[from] ChannelError),
    /// The document a directory describes cannot be planned.
    #[error("plan: {0}")]
    Plan(#[from] PlanError),
    /// The document a directory describes could not be baked.
    #[error("bake: {0}")]
    Bake(#[from] BakeError),
    /// The terrain carried no water and the spec it carries cannot be solved.
    #[error("water: {0}")]
    Water(#[from] WaterError),
}

/// What a save writes, beyond the values every terrain carries.
///
/// The bakes and the water are always written — a terrain without them is one no
/// consumer can read, since nothing is re-derived on the way in any more — so the
/// only choice left is whether the recipe goes with them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaveOptions {
    /// Write the layer stacks and the painted images inside them. Without this the
    /// terrain has values but no recipe, and cannot be edited or re-baked.
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
    /// What loads from it cannot be edited or re-baked — the fields come back with
    /// empty stacks — so saving it again would lose the document.
    pub fn export() -> Self {
        Self { recipe: false }
    }
}

trait Store {
    fn read(&self, name: &str) -> Result<Vec<u8>, IoError>;
    fn write(&mut self, name: &str, bytes: &[u8]) -> Result<(), IoError>;
    fn remove(&mut self, name: &str) -> Result<(), IoError>;
    fn list(&self) -> Result<Vec<String>, IoError>;
    fn exists(&self, name: &str) -> bool;
}

struct DirStore {
    root: PathBuf,
}

impl DirStore {
    fn path(&self, name: &str) -> Result<PathBuf, IoError> {
        check_file_name(name)?;
        Ok(self.root.join(name))
    }
}

impl Store for DirStore {
    fn read(&self, name: &str) -> Result<Vec<u8>, IoError> {
        let path = self.path(name)?;
        let meta = std::fs::symlink_metadata(&path)?;
        if !meta.is_file() {
            return Err(IoError::BadFileName(name.to_owned()));
        }
        if meta.len() > MAX_META_BYTES && name == META_FILE {
            return Err(IoError::MetaTooLarge(meta.len()));
        }
        Ok(std::fs::read(&path)?)
    }

    fn write(&mut self, name: &str, bytes: &[u8]) -> Result<(), IoError> {
        let path = self.path(name)?;
        let temporary = path.with_extension("tmp");
        std::fs::write(&temporary, bytes)?;
        std::fs::rename(&temporary, &path)?;
        Ok(())
    }

    fn remove(&mut self, name: &str) -> Result<(), IoError> {
        let path = self.path(name)?;
        std::fs::remove_file(path)?;
        Ok(())
    }

    fn list(&self) -> Result<Vec<String>, IoError> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            if let Some(name) = entry.file_name().to_str() {
                names.push(name.to_owned());
            }
        }
        Ok(names)
    }

    fn exists(&self, name: &str) -> bool {
        self.path(name)
            .is_ok_and(|path| std::fs::symlink_metadata(path).is_ok_and(|meta| meta.is_file()))
    }
}

fn check_file_name(name: &str) -> Result<(), IoError> {
    let bad = name.is_empty()
        || name.contains('/')
        || name.contains('\\')
        || name.contains('\0')
        || name.contains(':')
        || name.ends_with('.')
        || name.ends_with(' ')
        || Path::new(name).components().count() != 1
        || !matches!(
            Path::new(name).components().next(),
            Some(Component::Normal(_))
        );
    if bad {
        return Err(IoError::BadFileName(name.to_owned()));
    }
    Ok(())
}

fn layer_file(index: usize) -> String {
    format!("layer_{index:03}.png")
}

fn png_color(channels: usize) -> Option<png::ColorType> {
    match channels {
        1 => Some(png::ColorType::Grayscale),
        2 => Some(png::ColorType::GrayscaleAlpha),
        3 => Some(png::ColorType::Rgb),
        4 => Some(png::ColorType::Rgba),
        _ => None,
    }
}

fn encode_png(size: UVec2, channels: usize, bytes: &[u8]) -> Result<Vec<u8>, IoError> {
    let color = png_color(channels)
        .ok_or_else(|| IoError::Inconsistent(format!("a layer cannot hold {channels} channels")))?;
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, size.x, size.y);
    encoder.set_color(color);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    encoder.set_filter(png::Filter::Up);
    let mut writer = encoder
        .write_header()
        .map_err(|error| IoError::Meta(error.to_string()))?;
    writer
        .write_image_data(bytes)
        .map_err(|error| IoError::Meta(error.to_string()))?;
    writer
        .finish()
        .map_err(|error| IoError::Meta(error.to_string()))?;
    Ok(out)
}

fn decode_png(file: &str, bytes: &[u8], size: UVec2, channels: usize) -> Result<Vec<u8>, IoError> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_limits(png::Limits {
        bytes: MAX_DECODED_BYTES as usize,
    });
    let mut reader = decoder.read_info().map_err(|source| IoError::Decode {
        file: file.to_owned(),
        source,
    })?;

    let info = reader.info();
    let expected = png_color(channels)
        .ok_or_else(|| IoError::Inconsistent(format!("a layer cannot hold {channels} channels")))?;
    if info.width != size.x
        || info.height != size.y
        || info.color_type != expected
        || info.bit_depth != png::BitDepth::Eight
        || info.interlaced
    {
        return Err(IoError::ImageShape {
            file: file.to_owned(),
            found: format!(
                "{}x{} {:?} {:?}{}",
                info.width,
                info.height,
                info.color_type,
                info.bit_depth,
                if info.interlaced { " interlaced" } else { "" }
            ),
            expected: format!("{}x{} {expected:?} Eight", size.x, size.y),
        });
    }

    let wanted = reader
        .output_buffer_size()
        .ok_or_else(|| IoError::Inconsistent(format!("`{file}` declares an unusable size")))?;
    let mut out = vec![0u8; wanted];
    let frame = reader
        .next_frame(&mut out)
        .map_err(|source| IoError::Decode {
            file: file.to_owned(),
            source,
        })?;
    out.truncate(frame.buffer_size());
    Ok(out)
}

fn stripped_op(op: &LayerOp) -> LayerOp {
    match op {
        LayerOp::Paint(_) => LayerOp::Paint(Raster::default()),
        LayerOp::External(_) => LayerOp::External(Raster::default()),
        other => other.clone(),
    }
}

fn stripped_mask(mask: &Mask) -> Mask {
    match mask {
        Mask::Painted(_) => Mask::Painted(Raster::default()),
        other => other.clone(),
    }
}

struct PaintedImage {
    meta: LayerMeta,
    bytes: Vec<u8>,
}

fn paint_image(index: usize, raster: &Raster<f32>) -> PaintedImage {
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
            file: layer_file(index),
            width: raster.width(),
            height: raster.height(),
            shift: None,
            channels: vec![meta],
        },
        bytes,
    }
}

fn mask_image(index: usize, raster: &Raster<u8>) -> PaintedImage {
    PaintedImage {
        meta: LayerMeta {
            file: layer_file(index),
            width: raster.width(),
            height: raster.height(),
            shift: None,
            channels: vec![ChannelMeta::raw()],
        },
        bytes: raster.data().to_vec(),
    }
}

fn decoded_raster(bytes: &[u8], size: UVec2, meta: &ChannelMeta) -> Option<Raster<f32>> {
    let table = meta.table();
    let data: Vec<f32> = bytes.iter().map(|byte| table.get(*byte)).collect();
    Raster::from_vec(size, data)
}

impl Terrain {
    /// Reads a terrain from a directory: `terrain.ron` and the images it names.
    ///
    /// Runs no bake and no water solve — everything a consumer reads was settled
    /// when the directory was written, which is the point of the format. A
    /// directory carrying no readable field is [refused](IoError::NoFields) rather
    /// than loaded as a terrain that answers nothing.
    ///
    /// Fails on the first fault, in the order the metadata lists things, and
    /// returns no terrain.
    pub fn load_from_dir(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let store = DirStore {
            root: path.as_ref().to_path_buf(),
        };
        let (terrain, _) = read(&store)?;
        Ok(terrain)
    }

    /// Writes the terrain's values to a directory, without a recipe.
    ///
    /// **Deletes images.** Once `terrain.ron` is written, any file matching this
    /// writer's own naming scheme that the new metadata does not name is removed,
    /// so a save over a larger terrain leaves nothing stale behind. That sweep runs
    /// only if this call created the directory or found a `terrain.ron` already in
    /// it; otherwise the files are written and the stale ones are left alone.
    ///
    /// A failure part way through leaves the directory partly written.
    pub fn save_to_dir(&self, path: impl AsRef<Path>) -> Result<(), IoError> {
        let root = path.as_ref().to_path_buf();
        let fresh = !root.exists();
        std::fs::create_dir_all(&root)?;
        let mut store = DirStore { root };
        let pruning = fresh || store.exists(META_FILE);
        write(&mut store, self, None, pruning)
    }
}

fn meta_of(
    terrain: &Terrain,
    stacks: Vec<FieldStack>,
    water_spec: Option<crate::water::WaterSpec>,
) -> TerrainMeta {
    TerrainMeta {
        version: VERSION,
        size_x: terrain.size().x,
        size_y: terrain.size().y,
        layers: Vec::new(),
        fields: terrain.fields.clone(),
        water: terrain.water,
        water_spec,
        stacks,
    }
}

fn write(
    store: &mut impl Store,
    terrain: &Terrain,
    recipe: Option<(&TerrainSpec, Vec<FieldStack>, Vec<PaintedImage>)>,
    pruning: bool,
) -> Result<(), IoError> {
    let (spec, stacks, painted) = match recipe {
        Some((spec, stacks, painted)) => (Some(spec), stacks, painted),
        None => (None, Vec::new(), Vec::new()),
    };

    let mut meta = meta_of(
        terrain,
        stacks,
        spec.and_then(|spec| spec.water_spec.clone()),
    );

    for (index, layer) in terrain.layers.iter().enumerate() {
        let size = layer.texels().size();
        let channels = layer.texels().channels();
        let file = layer_file(index);
        let bytes = encode_png(size, channels, layer.texels().bytes())?;
        store.write(&file, &bytes)?;
        meta.layers.push(LayerMeta {
            file,
            width: size.x,
            height: size.y,
            shift: layer.shift(),
            channels: layer.channels().to_vec(),
        });
    }

    for image in &painted {
        let bytes = encode_png(
            UVec2::new(image.meta.width, image.meta.height),
            image.meta.channels.len(),
            &image.bytes,
        )?;
        store.write(&image.meta.file, &bytes)?;
        meta.layers.push(image.meta.clone());
    }

    let text = ron::ser::to_string_pretty(&meta, ron::ser::PrettyConfig::default())
        .map_err(|error| IoError::Meta(error.to_string()))?;
    store.write(META_FILE, text.as_bytes())?;

    if pruning {
        let named: BTreeSet<&str> = meta
            .layers
            .iter()
            .map(|layer| layer.file.as_str())
            .chain(std::iter::once(META_FILE))
            .collect();
        for name in store.list()? {
            if !named.contains(name.as_str()) && is_layer_file(&name) {
                store.remove(&name)?;
            }
        }
    }

    Ok(())
}

fn is_layer_file(name: &str) -> bool {
    name.strip_prefix("layer_")
        .and_then(|rest| rest.strip_suffix(".png"))
        .is_some_and(|digits| {
            !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
        })
}

fn read(store: &impl Store) -> Result<(Terrain, TerrainMeta), IoError> {
    let bytes = store.read(META_FILE)?;
    if bytes.len() as u64 > MAX_META_BYTES {
        return Err(IoError::MetaTooLarge(bytes.len() as u64));
    }
    let text = std::str::from_utf8(&bytes).map_err(|error| IoError::Meta(error.to_string()))?;
    let meta: TerrainMeta =
        ron::from_str(text).map_err(|error| IoError::Meta(error.to_string()))?;

    if meta.version != VERSION {
        return Err(IoError::UnsupportedVersion(meta.version));
    }
    let size = UVec2::new(meta.size_x, meta.size_y);
    if size.x == 0 || size.y == 0 {
        return Err(IoError::Inconsistent(format!(
            "the extent is {}x{}",
            size.x, size.y
        )));
    }
    if (size.x as u64) * (size.y as u64) > MAX_CELLS {
        return Err(IoError::Inconsistent(format!(
            "the extent {}x{} is over the {MAX_CELLS} cell limit",
            size.x, size.y
        )));
    }
    if meta.layers.len() > MAX_LAYERS {
        return Err(IoError::Inconsistent(format!(
            "{} layers is over the {MAX_LAYERS} limit",
            meta.layers.len()
        )));
    }

    let mut budget = MAX_DECODED_BYTES;
    let mut layers = Vec::with_capacity(meta.layers.len());
    for declared in &meta.layers {
        let channels = declared.channels.len();
        if channels == 0 || channels > MAX_CHANNELS {
            return Err(IoError::Inconsistent(format!(
                "`{}` declares {channels} channels",
                declared.file
            )));
        }
        for channel in &declared.channels {
            channel.validate()?;
        }
        if let Some(shift) = declared.shift {
            if shift > MAX_SHIFT {
                return Err(IoError::Inconsistent(format!(
                    "`{}` declares shift {shift}, over the {MAX_SHIFT} limit",
                    declared.file
                )));
            }
            let expected = resolution(size, shift);
            if UVec2::new(declared.width, declared.height) != expected {
                return Err(IoError::ImageShape {
                    file: declared.file.clone(),
                    found: format!("{}x{}", declared.width, declared.height),
                    expected: format!("{}x{} for shift {shift}", expected.x, expected.y),
                });
            }
        }

        let declared_size = UVec2::new(declared.width, declared.height);
        let cost = (declared.width as u64)
            .saturating_mul(declared.height as u64)
            .saturating_mul(channels as u64);
        budget = budget.checked_sub(cost).ok_or_else(|| {
            IoError::Inconsistent(format!(
                "the terrain's images are over the {MAX_DECODED_BYTES} byte limit"
            ))
        })?;

        let bytes = store.read(&declared.file)?;
        let pixels = decode_png(&declared.file, &bytes, declared_size, channels)?;
        let texels = LayerTexels::from_bytes(declared_size, channels, pixels).ok_or_else(|| {
            IoError::ImageShape {
                file: declared.file.clone(),
                found: "a different number of texels".to_owned(),
                expected: format!("{}x{} of {channels}", declared_size.x, declared_size.y),
            }
        })?;
        layers.push(TerrainLayer::new(
            declared.shift,
            declared.channels.clone(),
            texels,
        ));
    }

    for field in &meta.fields {
        let layer = layers.get(field.layer as usize).ok_or_else(|| {
            IoError::Inconsistent(format!(
                "field `{}` names layer {}, of {}",
                field.name,
                field.layer,
                layers.len()
            ))
        })?;
        if field.channel as usize >= layer.channels().len() {
            return Err(IoError::Inconsistent(format!(
                "field `{}` names channel {} of {}",
                field.name,
                field.channel,
                layer.channels().len()
            )));
        }
        if layer.shift() != Some(field.shift) {
            return Err(IoError::Inconsistent(format!(
                "field `{}` is at shift {} but its layer is at {:?}",
                field.name,
                field.shift,
                layer.shift()
            )));
        }
    }

    if let Some(water) = meta.water {
        let layer = layers.get(water.layer as usize).ok_or_else(|| {
            IoError::Inconsistent(format!(
                "the water names layer {}, of {}",
                water.layer,
                layers.len()
            ))
        })?;
        if layer.channels().len() != MAX_CHANNELS {
            return Err(IoError::Inconsistent(format!(
                "the water layer holds {} channels, not {MAX_CHANNELS}",
                layer.channels().len()
            )));
        }
        if layer.texels().size() != size {
            return Err(IoError::Inconsistent(
                "the water layer is not at the document's extent".to_owned(),
            ));
        }
    }

    if meta.fields.is_empty() {
        return Err(IoError::NoFields);
    }

    let terrain = Terrain {
        size,
        fields: meta.fields.clone(),
        layers,
        water: meta.water,
    };
    Ok((terrain, meta))
}

impl TerrainSpec {
    /// Writes the document to a directory: `terrain.ron`, one image per group of
    /// fields, one for the water, and one per painted raster.
    ///
    /// The document is baked first if it is not already, because a terrain that
    /// carries no values is one no consumer can open.
    ///
    /// **Deletes images**, under the same rule as [`Terrain::save_to_dir`]: after
    /// `terrain.ron` is written, files matching this writer's naming scheme that it
    /// no longer names are removed, and only in a directory this call created or
    /// that already held a `terrain.ron`.
    ///
    /// Nothing about the document is changed by saving it.
    pub fn save_to_dir(&self, path: impl AsRef<Path>, options: SaveOptions) -> Result<(), IoError> {
        let root = path.as_ref().to_path_buf();
        let fresh = !root.exists();
        std::fs::create_dir_all(&root)?;
        let mut store = DirStore { root };
        let pruning = fresh || store.exists(META_FILE);
        write_spec(&mut store, self, options, pruning)
    }
}

fn write_spec(
    store: &mut impl Store,
    spec: &TerrainSpec,
    options: SaveOptions,
    pruning: bool,
) -> Result<(), IoError> {
    {
        let mut baked = spec.clone();
        if baked
            .fields
            .iter()
            .any(|field| field.baked().size() != field.resolution(baked.size))
        {
            baked.bake_in_place()?;
        }
        let terrain = baked.clone().bake()?;

        if !options.recipe {
            return write(store, &terrain, None, pruning);
        }

        let mut painted = Vec::new();
        let mut stacks = Vec::with_capacity(baked.fields.len());
        let mut next = terrain.layers.len();
        for field in &baked.fields {
            let mut paint = Vec::new();
            let mut stack = Vec::with_capacity(field.layers.len());
            for (index, layer) in field.layers.iter().enumerate() {
                match &layer.op {
                    LayerOp::Paint(raster) | LayerOp::External(raster) => {
                        painted.push(paint_image(next, raster));
                        paint.push(PaintRef {
                            stack_index: index as u32,
                            slot: PaintSlot::Op,
                            layer: next as u8,
                        });
                        next += 1;
                    }
                    _ => {}
                }
                if let Mask::Painted(raster) = &layer.mask {
                    painted.push(mask_image(next, raster));
                    paint.push(PaintRef {
                        stack_index: index as u32,
                        slot: PaintSlot::Mask,
                        layer: next as u8,
                    });
                    next += 1;
                }
                let mut stripped = layer.clone();
                stripped.op = stripped_op(&layer.op);
                stripped.mask = stripped_mask(&layer.mask);
                stack.push(stripped);
            }
            stacks.push(FieldStack {
                field: field.id.clone(),
                range: field.range,
                export: field.export,
                stack,
                paint,
            });
        }

        write(store, &terrain, Some((&baked, stacks, painted)), pruning)
    }
}

impl TerrainSpec {
    /// Reads a document from a directory, re-attaching the painted rasters its
    /// recipe names and re-baking from them.
    ///
    /// The bake is redone rather than taken from the images: the carried values were
    /// quantised, and a document whose paint is the decoded paint but whose bakes
    /// came from the original would disagree with itself the first time part of it
    /// was re-baked.
    ///
    /// A directory saved without a recipe loads as a document with no layers, which
    /// cannot be edited or saved again without losing it.
    pub fn load_from_dir(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let store = DirStore {
            root: path.as_ref().to_path_buf(),
        };
        read_spec(&store)
    }
}

fn read_spec(store: &impl Store) -> Result<TerrainSpec, IoError> {
    {
        let (terrain, meta) = read(store)?;

        let mut spec = TerrainSpec::new(terrain.size());
        spec.water_spec = meta.water_spec.clone();

        for info in &meta.fields {
            let stack = meta
                .stacks
                .iter()
                .find(|entry| entry.field.as_str() == info.name);
            let mut field = Field::new(info.name.clone())
                .with_role(info.role)
                .with_shift(info.shift);
            if let Some(stack) = stack {
                field.range = stack.range;
                field.export = stack.export;
                field.layers = stack.stack.clone();
                for reference in &stack.paint {
                    let declared = meta.layers.get(reference.layer as usize).ok_or_else(|| {
                        IoError::Inconsistent(format!(
                            "`{}` names paint layer {}, of {}",
                            info.name,
                            reference.layer,
                            meta.layers.len()
                        ))
                    })?;
                    let target = field
                        .layers
                        .get_mut(reference.stack_index as usize)
                        .ok_or_else(|| {
                            IoError::Inconsistent(format!(
                                "`{}` names stack position {}, of {}",
                                info.name,
                                reference.stack_index,
                                stack.stack.len()
                            ))
                        })?;
                    let size = UVec2::new(declared.width, declared.height);
                    let bytes = store.read(&declared.file)?;
                    let pixels = decode_png(&declared.file, &bytes, size, 1)?;
                    let channel = declared.channels.first().copied().ok_or_else(|| {
                        IoError::Inconsistent(format!("`{}` declares no channel", declared.file))
                    })?;
                    match reference.slot {
                        PaintSlot::Op => {
                            let raster =
                                decoded_raster(&pixels, size, &channel).ok_or_else(|| {
                                    IoError::Inconsistent(format!("`{}` is short", declared.file))
                                })?;
                            match &mut target.op {
                                LayerOp::Paint(slot) | LayerOp::External(slot) => *slot = raster,
                                _ => {
                                    return Err(IoError::Inconsistent(format!(
                                        "`{}` position {} does not take paint",
                                        info.name, reference.stack_index
                                    )));
                                }
                            }
                        }
                        PaintSlot::Mask => {
                            let raster = Raster::from_vec(size, pixels).ok_or_else(|| {
                                IoError::Inconsistent(format!("`{}` is short", declared.file))
                            })?;
                            match &mut target.mask {
                                Mask::Painted(slot) => *slot = raster,
                                _ => {
                                    return Err(IoError::Inconsistent(format!(
                                        "`{}` position {} does not take a mask",
                                        info.name, reference.stack_index
                                    )));
                                }
                            }
                        }
                    }
                }
            }
            spec.fields.push(field);
        }

        if spec.fields.iter().any(|field| !field.layers.is_empty()) {
            spec.bake_in_place()?;
            if let Some(water) = spec.water_spec.clone() {
                spec.solve_water(&water)?;
            }
        }

        Ok(spec)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::channel::ChannelEncoding;
    use crate::field::{FieldId, FieldRole};
    use crate::layer::{Blend, Layer, Remap};
    use crate::meta::WaterInfo;
    use crate::noise::{NoiseKind, NoiseSpec};
    use crate::water::WaterSpec;

    const SIZE: UVec2 = UVec2::new(48, 32);

    #[derive(Default)]
    struct MemStore {
        files: BTreeMap<String, Vec<u8>>,
    }

    impl Store for MemStore {
        fn read(&self, name: &str) -> Result<Vec<u8>, IoError> {
            check_file_name(name)?;
            self.files.get(name).cloned().ok_or_else(|| {
                IoError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    name.to_owned(),
                ))
            })
        }

        fn write(&mut self, name: &str, bytes: &[u8]) -> Result<(), IoError> {
            check_file_name(name)?;
            self.files.insert(name.to_owned(), bytes.to_vec());
            Ok(())
        }

        fn remove(&mut self, name: &str) -> Result<(), IoError> {
            self.files.remove(name);
            Ok(())
        }

        fn list(&self) -> Result<Vec<String>, IoError> {
            Ok(self.files.keys().cloned().collect())
        }

        fn exists(&self, name: &str) -> bool {
            self.files.contains_key(name)
        }
    }

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

    fn byte_mask() -> Raster<u8> {
        let data = (0..cells()).map(|i| (i % 251) as u8).collect();
        Raster::from_vec(SIZE, data).unwrap()
    }

    fn noise_layer(seed: u32) -> Layer {
        Layer::new(LayerOp::Noise(NoiseSpec::new(seed, NoiseKind::Fbm, 0.05)))
            .with_blend(Blend::Replace)
    }

    fn painted_document() -> TerrainSpec {
        TerrainSpec::new(SIZE)
            .with_field(
                Field::new("moisture")
                    .with_shift(2)
                    .with_layer(noise_layer(3)),
            )
            .with_field(
                Field::new("height")
                    .with_role(FieldRole::Height)
                    .with_layer(
                        Layer::new(LayerOp::Paint(awkward_ramp())).with_blend(Blend::Replace),
                    )
                    .with_layer(
                        Layer::new(LayerOp::External(ramp())).with_mask(Mask::Painted(byte_mask())),
                    )
                    .with_layer(noise_layer(9).with_mask(Mask::Field(
                        FieldId::from("moisture"),
                        Remap::new((0.3, 0.7), (0.0, 1.0)),
                    ))),
            )
    }

    fn baked_document() -> TerrainSpec {
        let mut terrain = TerrainSpec::new(SIZE)
            .with_field(
                Field::new("moisture")
                    .with_shift(2)
                    .with_layer(noise_layer(3)),
            )
            .with_field(
                Field::new("height")
                    .with_role(FieldRole::Height)
                    .with_layer(Layer::new(LayerOp::Paint(ramp())).with_blend(Blend::Replace))
                    .with_layer(noise_layer(9)),
            );
        terrain.bake_in_place().unwrap();
        terrain
    }

    fn saved(spec: &TerrainSpec, options: SaveOptions) -> MemStore {
        let mut store = MemStore::default();
        write_spec(&mut store, spec, options, true).unwrap();
        store
    }

    fn meta_of(store: &MemStore) -> TerrainMeta {
        ron::from_str(std::str::from_utf8(&store.files[META_FILE]).unwrap()).unwrap()
    }

    fn put_meta(store: &mut MemStore, meta: &TerrainMeta) {
        let text = ron::ser::to_string_pretty(meta, ron::ser::PrettyConfig::default()).unwrap();
        store.files.insert(META_FILE.to_owned(), text.into_bytes());
    }

    // The shape of the format, and the one thing a person opening the directory sees
    // first: metadata plus the images it names, and nothing that has to be guessed at.
    #[test]
    fn a_terrain_is_one_metadata_file_and_the_images_it_names() {
        let store = saved(&baked_document(), SaveOptions::export());
        let meta = meta_of(&store);

        assert!(store.files.contains_key(META_FILE));
        assert!(!meta.layers.is_empty());
        for layer in &meta.layers {
            assert!(store.files.contains_key(&layer.file), "{}", layer.file);
        }
        let named: BTreeSet<&str> = meta
            .layers
            .iter()
            .map(|layer| layer.file.as_str())
            .chain(std::iter::once(META_FILE))
            .collect();
        for name in store.files.keys() {
            assert!(named.contains(name.as_str()), "{name} is not named");
        }
    }

    // The metadata is text so a terrain can be inspected and hand-edited without this
    // crate. The negative half is the load-bearing one: texels must stay in the
    // images, or a 4096-square document's metadata would be hundreds of megabytes.
    #[test]
    fn the_metadata_is_readable_ron_that_carries_no_texels() {
        let store = saved(&painted_document(), SaveOptions::document());
        let text = std::str::from_utf8(&store.files[META_FILE]).unwrap();

        assert!(text.contains("moisture"), "{text}");
        assert!(text.contains("height"), "{text}");
        assert!(
            !text.contains("0.058823"),
            "painted data reached the metadata"
        );
        assert!(text.len() < 16 * 1024, "metadata is {} bytes", text.len());
    }

    // The whole point of the change: a consumer opens the directory and reads values
    // out of it, with no bake and no solve on the way in.
    #[test]
    fn a_consumer_reads_values_without_baking() {
        let mut spec = baked_document();
        spec.solve_water(&WaterSpec::new("height")).unwrap();
        let store = saved(&spec, SaveOptions::export());

        let (terrain, _) = read(&store).unwrap();
        assert_eq!(terrain.size(), SIZE);
        let height = terrain.field("height").unwrap();
        assert!(height.value_at(0, 0).is_some());
        assert!(terrain.water().is_some());
        assert!(terrain.fields().count() == 2);
    }

    // A saved value has to land within the quantisation bound of the value that was
    // baked, or the format is losing more than the eight bits it admits to.
    #[test]
    fn a_baked_value_survives_the_round_trip_within_a_step() {
        let spec = baked_document();
        let before = spec.clone().bake().unwrap();
        let store = saved(&spec, SaveOptions::export());
        let (after, _) = read(&store).unwrap();

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
    }

    // Fields at the same shift share an image and one at another shift starts its
    // own; this is the packing rule as the format actually applies it.
    #[test]
    fn fields_sharing_a_shift_share_an_image() {
        let spec = TerrainSpec::new(SIZE)
            .with_field(Field::new("a").with_layer(noise_layer(1)))
            .with_field(Field::new("b").with_layer(noise_layer(2)))
            .with_field(
                Field::new("coarse")
                    .with_shift(2)
                    .with_layer(noise_layer(3)),
            );
        let store = saved(&spec, SaveOptions::export());
        let meta = meta_of(&store);

        assert_eq!(meta.layers.len(), 2);
        assert_eq!(meta.layers[0].channels.len(), 2);
        assert_eq!(meta.layers[0].shift, Some(0));
        assert_eq!(meta.layers[1].channels.len(), 1);
        assert_eq!(meta.layers[1].shift, Some(2));
    }

    // A painted raster is stretched over the document from whatever size it was
    // authored at, so applying the bake layers' extent rule to it would make every
    // paint layer unloadable.
    #[test]
    fn a_painted_image_carries_its_own_extent_and_no_shift() {
        let odd = Raster::from_vec(UVec2::new(7, 5), vec![0.5f32; 35]).unwrap();
        let spec = TerrainSpec::new(SIZE).with_field(
            Field::new("height")
                .with_layer(Layer::new(LayerOp::Paint(odd)).with_blend(Blend::Replace)),
        );
        let store = saved(&spec, SaveOptions::document());
        let meta = meta_of(&store);

        let paint = meta
            .layers
            .iter()
            .find(|layer| layer.shift.is_none())
            .unwrap();
        assert_eq!((paint.width, paint.height), (7, 5));

        let loaded = read_spec(&store).unwrap();
        match &loaded.fields[0].layers[0].op {
            LayerOp::Paint(raster) => assert_eq!(raster.size(), UVec2::new(7, 5)),
            other => panic!("{other:?}"),
        }
    }

    // A mask is already bytes, so it is the one thing left in the format that comes
    // back exactly — and it does so because of its encoding, not by luck.
    #[test]
    fn a_painted_mask_round_trips_byte_for_byte() {
        let store = saved(&painted_document(), SaveOptions::document());
        let loaded = read_spec(&store).unwrap();

        let mask = loaded
            .fields
            .iter()
            .flat_map(|field| &field.layers)
            .find_map(|layer| match &layer.mask {
                Mask::Painted(raster) => Some(raster),
                _ => None,
            })
            .unwrap();
        assert_eq!(mask, &byte_mask());
    }

    // The recipe is what separates a document from an export, and an export has to be
    // readable while being unable to pose as something that can be edited again.
    #[test]
    fn an_export_carries_the_values_but_no_recipe() {
        let store = saved(&baked_document(), SaveOptions::export());
        let meta = meta_of(&store);
        assert!(meta.stacks.is_empty());

        let loaded = read_spec(&store).unwrap();
        assert!(loaded.fields.iter().all(|field| field.layers.is_empty()));
        assert_eq!(loaded.fields.len(), 2);
    }

    // Saving over a terrain that had more images has to remove the ones it no longer
    // names, or a later load would read a stale image the metadata never mentions.
    #[test]
    fn a_save_removes_an_image_it_no_longer_names() {
        let mut store = saved(&painted_document(), SaveOptions::document());
        let before = store.files.len();
        store.files.insert(layer_file(99), b"stale".to_vec());

        write_spec(&mut store, &baked_document(), SaveOptions::export(), true).unwrap();

        assert!(!store.files.contains_key(&layer_file(99)));
        assert!(store.files.len() < before);
        read(&store).unwrap();
    }

    // The sweep must never touch a file this writer would not itself have produced:
    // a terrain directory a person also keeps notes or references in stays intact.
    #[test]
    fn a_save_leaves_a_file_it_did_not_write() {
        let mut store = saved(&baked_document(), SaveOptions::export());
        store.files.insert("notes.txt".to_owned(), b"mine".to_vec());
        store
            .files
            .insert("heightmap.png".to_owned(), b"mine".to_vec());

        write_spec(&mut store, &baked_document(), SaveOptions::export(), true).unwrap();

        assert_eq!(store.files.get("notes.txt").unwrap(), b"mine");
        assert_eq!(store.files.get("heightmap.png").unwrap(), b"mine");
    }

    // Not pruning is the answer for a directory that is not already a terrain, which
    // is what stops a mistyped save path deleting someone's files.
    #[test]
    fn a_save_that_is_not_pruning_removes_nothing() {
        let mut store = saved(&painted_document(), SaveOptions::document());
        store.files.insert(layer_file(99), b"stale".to_vec());

        write_spec(&mut store, &baked_document(), SaveOptions::export(), false).unwrap();

        assert_eq!(store.files.get(&layer_file(99)).unwrap(), b"stale");
    }

    // There is no migration path, so a newer terrain has to be refused outright
    // rather than read as far as it happens to agree.
    #[test]
    fn a_terrain_from_a_later_version_is_refused() {
        let mut store = saved(&baked_document(), SaveOptions::export());
        let mut meta = meta_of(&store);
        meta.version = 99;
        put_meta(&mut store, &meta);

        assert!(matches!(read(&store), Err(IoError::UnsupportedVersion(99))));
    }

    // A directory that is not a terrain has to be refused rather than read as one
    // whose metadata happens to be missing.
    #[test]
    fn a_directory_that_is_not_a_terrain_is_refused() {
        let store = MemStore::default();
        assert!(matches!(read(&store), Err(IoError::Io(_))));
    }

    // Nothing re-derives a bake on the way in any more, so a terrain with no fields
    // would otherwise load as one that answers nothing at every cell.
    #[test]
    fn a_terrain_carrying_no_field_is_refused() {
        let mut store = saved(&baked_document(), SaveOptions::export());
        let mut meta = meta_of(&store);
        meta.fields.clear();
        meta.water = None;
        put_meta(&mut store, &meta);

        assert!(matches!(read(&store), Err(IoError::NoFields)));
    }

    // Joining a name like this onto the directory would discard the directory
    // entirely, so it has to be refused before it is ever turned into a path.
    #[test]
    fn a_file_name_that_is_a_path_is_refused() {
        for name in [
            "/etc/passwd",
            "../../secret.png",
            "sub/dir.png",
            "..",
            ".",
            "",
            "bad\0name",
        ] {
            assert!(check_file_name(name).is_err(), "{name:?} was accepted");
        }
        assert!(check_file_name("layer_000.png").is_ok());
    }

    // The metadata names every file the loader opens, so a name it cannot open has
    // to be an error rather than a layer quietly missing from the terrain.
    #[test]
    fn an_image_the_metadata_names_but_the_directory_lacks_is_refused() {
        let mut store = saved(&baked_document(), SaveOptions::export());
        let meta = meta_of(&store);
        store.files.remove(&meta.layers[0].file);

        assert!(matches!(read(&store), Err(IoError::Io(_))));
    }

    // The extent is checked against the image's own header before its pixels are
    // allocated, which is what stops a small file claiming an enormous one.
    #[test]
    fn an_image_at_the_wrong_extent_is_refused() {
        let mut store = saved(&baked_document(), SaveOptions::export());
        let mut meta = meta_of(&store);
        meta.layers[0].width += 1;
        put_meta(&mut store, &meta);

        assert!(matches!(read(&store), Err(IoError::ImageShape { .. })));
    }

    // A channel count the image does not have would otherwise be read as a different
    // number of texels at the same size.
    #[test]
    fn an_image_with_the_wrong_channel_count_is_refused() {
        let mut store = saved(&baked_document(), SaveOptions::export());
        let mut meta = meta_of(&store);
        meta.layers[0].channels.push(ChannelMeta::linear(0.0, 1.0));
        put_meta(&mut store, &meta);

        assert!(matches!(read(&store), Err(IoError::ImageShape { .. })));
    }

    // Two shifts can imply the same image extent, so a shift out of range is not
    // caught by the extent check and needs refusing on its own.
    #[test]
    fn a_shift_past_what_the_grid_honours_is_refused() {
        let mut store = saved(&baked_document(), SaveOptions::export());
        let mut meta = meta_of(&store);
        meta.layers[0].shift = Some(200);
        put_meta(&mut store, &meta);

        assert!(matches!(read(&store), Err(IoError::Inconsistent(_))));
    }

    // A field's shift decides how a cell maps to a texel; if it disagrees with the
    // image it points at, the read is bounds-checked against one grid and performed
    // against another.
    #[test]
    fn a_field_whose_shift_is_not_its_layers_is_refused() {
        let mut store = saved(&baked_document(), SaveOptions::export());
        let mut meta = meta_of(&store);
        meta.fields[0].shift = meta.fields[0].shift.wrapping_add(1);
        put_meta(&mut store, &meta);

        assert!(matches!(read(&store), Err(IoError::Inconsistent(_))));
    }

    // An index out of a hand-edited file has to be refused rather than panicking or
    // silently dropping the field.
    #[test]
    fn a_field_pointing_past_the_images_is_refused() {
        let mut store = saved(&baked_document(), SaveOptions::export());
        let mut meta = meta_of(&store);
        meta.fields[0].layer = 200;
        put_meta(&mut store, &meta);
        assert!(matches!(read(&store), Err(IoError::Inconsistent(_))));

        let mut store = saved(&baked_document(), SaveOptions::export());
        let mut meta = meta_of(&store);
        meta.fields[0].channel = 200;
        put_meta(&mut store, &meta);
        assert!(matches!(read(&store), Err(IoError::Inconsistent(_))));
    }

    // A range that is not finite and ordered makes every read through it a NaN, so
    // it is refused at load rather than producing plausible nonsense.
    #[test]
    fn a_channel_range_that_cannot_be_read_is_refused() {
        for broken in [
            ChannelMeta {
                low: f32::NAN,
                high: 1.0,
                encoding: ChannelEncoding::Linear,
            },
            ChannelMeta {
                low: 1.0,
                high: 0.0,
                encoding: ChannelEncoding::Linear,
            },
        ] {
            let mut store = saved(&baked_document(), SaveOptions::export());
            let mut meta = meta_of(&store);
            meta.layers[0].channels[0] = broken;
            put_meta(&mut store, &meta);

            assert!(
                matches!(read(&store), Err(IoError::Channel(_))),
                "{broken:?}"
            );
        }
    }

    // Half an image is still a file the metadata names, so only the decode catches
    // it; a short read that produced a terrain would hand back silent zeroes.
    #[test]
    fn a_truncated_image_is_refused_rather_than_read_short() {
        let mut store = saved(&baked_document(), SaveOptions::export());
        let meta = meta_of(&store);
        let file = meta.layers[0].file.clone();
        let bytes = store.files[&file].clone();
        store.files.insert(file, bytes[..bytes.len() / 2].to_vec());

        assert!(read(&store).is_err());
    }

    // A truncated metadata file fails to parse rather than being read as a terrain
    // that simply declares less than it did.
    #[test]
    fn truncated_metadata_is_refused() {
        let mut store = saved(&baked_document(), SaveOptions::export());
        let bytes = store.files[META_FILE].clone();
        store
            .files
            .insert(META_FILE.to_owned(), bytes[..bytes.len() / 2].to_vec());

        assert!(matches!(read(&store), Err(IoError::Meta(_))));
    }

    // The water spec is carried even though the solved state is quantised, which is
    // what lets a document that was saved without water still grow it back.
    #[test]
    fn a_document_keeps_its_water_spec_through_a_save() {
        let mut spec = baked_document();
        spec.solve_water(&WaterSpec::new("height")).unwrap();
        let store = saved(&spec, SaveOptions::document());

        assert_eq!(meta_of(&store).water_spec, Some(WaterSpec::new("height")));
        assert_eq!(
            read_spec(&store).unwrap().water_spec,
            Some(WaterSpec::new("height"))
        );
    }

    // All of the water is one image, which is what four channels bought; a reader
    // validates one index rather than five.
    #[test]
    fn the_water_is_a_single_four_channel_image() {
        let mut spec = baked_document();
        spec.solve_water(&WaterSpec::new("height")).unwrap();
        let store = saved(&spec, SaveOptions::export());
        let meta = meta_of(&store);

        let water = meta.water.unwrap();
        let layer = &meta.layers[water.layer as usize];
        assert_eq!(layer.channels.len(), MAX_CHANNELS);
        assert_eq!((layer.width, layer.height), (SIZE.x, SIZE.y));
    }

    // The water indices come out of the same untrusted file as everything else.
    #[test]
    fn a_water_image_that_is_not_four_channels_is_refused() {
        let mut spec = baked_document();
        spec.solve_water(&WaterSpec::new("height")).unwrap();
        let mut store = saved(&spec, SaveOptions::export());
        let mut meta = meta_of(&store);
        meta.water = Some(WaterInfo { lakes: 1, layer: 0 });
        put_meta(&mut store, &meta);

        assert!(matches!(read(&store), Err(IoError::Inconsistent(_))));
    }

    // No spec has to mean no water rather than a default solve, or every document
    // that never wanted water would acquire some on its first load.
    #[test]
    fn a_document_with_no_water_spec_loads_with_no_water() {
        let store = saved(&baked_document(), SaveOptions::document());
        assert!(read_spec(&store).unwrap().water().is_none());
        assert!(read(&store).unwrap().0.water().is_none());
    }

    fn scratch(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "watershed-{}-{}-{name}",
            std::process::id(),
            VERSION
        ));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    // Every other test here runs over the in-memory store, so this is the one that
    // exercises the real thing: creating the directory, the temporary-name rename,
    // and reading it back off a disk.
    #[test]
    fn a_terrain_round_trips_through_a_real_directory() {
        let root = scratch("round-trip");
        let mut spec = baked_document();
        spec.solve_water(&WaterSpec::new("height")).unwrap();
        spec.save_to_dir(&root, SaveOptions::document()).unwrap();

        assert!(root.join(META_FILE).is_file());
        let terrain = Terrain::load_from_dir(&root).unwrap();
        assert_eq!(terrain.size(), SIZE);
        assert!(terrain.field("height").unwrap().value_at(1, 1).is_some());
        assert!(terrain.water().is_some());

        let document = TerrainSpec::load_from_dir(&root).unwrap();
        assert!(document.fields.iter().all(|field| !field.layers.is_empty()));

        assert!(
            !std::fs::read_dir(&root).unwrap().any(|entry| entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".tmp")),
            "a finished save left a temporary file behind"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    // The destructive half, against a real directory: a stale image goes, and a file
    // this writer would never have produced stays.
    #[test]
    fn a_real_save_prunes_its_own_stale_images_only() {
        let root = scratch("prune");
        let spec = painted_document();
        spec.save_to_dir(&root, SaveOptions::document()).unwrap();
        std::fs::write(root.join(layer_file(99)), b"stale").unwrap();
        std::fs::write(root.join("notes.txt"), b"mine").unwrap();

        baked_document()
            .save_to_dir(&root, SaveOptions::export())
            .unwrap();

        assert!(!root.join(layer_file(99)).exists());
        assert_eq!(std::fs::read(root.join("notes.txt")).unwrap(), b"mine");
        Terrain::load_from_dir(&root).unwrap();
        std::fs::remove_dir_all(&root).unwrap();
    }

    // Saving into a directory that is not already a terrain must not delete what is
    // in it — the guard against a mistyped save path.
    #[test]
    fn a_save_into_a_foreign_directory_deletes_nothing() {
        let root = scratch("foreign");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join(layer_file(7)), b"someone else's").unwrap();

        baked_document()
            .save_to_dir(&root, SaveOptions::export())
            .unwrap();

        assert_eq!(
            std::fs::read(root.join(layer_file(7))).unwrap(),
            b"someone else's"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    // A document is saved, loaded and saved again in the editor's own loop, so the
    // loss has to settle rather than compound: the second trip must move nothing the
    // first one did not already move.
    #[test]
    fn a_second_round_trip_moves_nothing_further() {
        let first = read_spec(&saved(&painted_document(), SaveOptions::document())).unwrap();
        let second = read_spec(&saved(&first, SaveOptions::document())).unwrap();

        for (left, right) in first.fields.iter().zip(&second.fields) {
            assert_eq!(left.baked().size(), right.baked().size());
            for (a, b) in left.baked().data().iter().zip(right.baked().data()) {
                assert_eq!(a.to_bits(), b.to_bits(), "a second trip moved {a} to {b}");
            }
        }
    }
}
