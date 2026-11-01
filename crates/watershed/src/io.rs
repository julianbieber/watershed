//! The on-disk form of a terrain's values: a directory holding `terrain.ron` and
//! the images it names, and what a reader is allowed to assume about a directory it
//! did not write.
//!
//! Only the values are here. The recipe that produced them, where a terrain carries
//! one, is a second file this half of the format never opens.
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

use crate::channel::{ChannelError, MAX_CHANNELS};
use crate::chunk::ChunkGrid;
use crate::meta::{LayerMeta, TerrainMeta, VERSION};
use crate::terrain::{FieldInfo, LayerTexels, Terrain, TerrainLayer};

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
    /// An image file is larger than its declared shape could encode to. Checked
    /// before the file is read.
    #[error("`{file}` is {found} bytes, over the {limit} bytes its image could encode to")]
    FileTooLarge {
        /// The file that was refused.
        file: String,
        /// Its size on disk.
        found: u64,
        /// The most its declared image could take.
        limit: u64,
    },
    /// The metadata is well-formed but does not hang together — an index past the
    /// images it declares, a water image that is not four channels, a chunk grid
    /// that is zero or overflows, a chunk index outside the grid, or a chunk
    /// outside what this build will carry.
    #[error("the terrain does not hang together: {0}")]
    Inconsistent(String),
    /// A terrain with no readable field. Nothing re-derives a bake on the way in,
    /// so this is refused rather than loaded as a terrain answering nothing.
    #[error("the terrain carries no readable field")]
    NoFields,
    /// A channel could not be read. See [`ChannelError`].
    #[error(transparent)]
    Channel(#[from] ChannelError),
}

trait Store {
    fn read(&self, name: &str, limit: u64) -> Result<Vec<u8>, IoError>;
    fn write(&mut self, name: &str, bytes: &[u8]) -> Result<(), IoError>;
    fn remove(&mut self, name: &str) -> Result<(), IoError>;
    fn list(&self) -> Result<Vec<String>, IoError>;
    fn exists(&self, name: &str) -> bool;
}

#[derive(Debug)]
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
    fn read(&self, name: &str, limit: u64) -> Result<Vec<u8>, IoError> {
        let path = self.path(name)?;
        let meta = std::fs::symlink_metadata(&path)?;
        if !meta.is_file() {
            return Err(IoError::BadFileName(name.to_owned()));
        }
        if meta.len() > limit {
            return Err(too_large(name, meta.len(), limit));
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

fn too_large(name: &str, found: u64, limit: u64) -> IoError {
    if name == META_FILE {
        IoError::MetaTooLarge(found)
    } else {
        IoError::FileTooLarge {
            file: name.to_owned(),
            found,
            limit,
        }
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

/// A terrain directory opened for reading: `terrain.ron` read and checked, and no
/// image read yet.
///
/// What a consumer holds to load a large terrain a chunk at a time: the grid says
/// where a position falls, and [`TerrainDir::load_chunk`] reads only that chunk's
/// files.
#[derive(Debug)]
pub struct TerrainDir {
    store: DirStore,
    grid: ChunkGrid,
    meta: TerrainMeta,
}

impl TerrainDir {
    /// Opens a terrain directory: reads `terrain.ron` and checks it hangs together,
    /// reading no image.
    ///
    /// Refused for metadata over [`MAX_META_BYTES`], at any version but
    /// [`VERSION`], with a zero or overflowing chunk grid, a chunk over
    /// [`MAX_CELLS`] cells, more than [`MAX_LAYERS`] images, one chunk's images
    /// decoding to more than [`MAX_DECODED_BYTES`], a field or water naming an
    /// image or channel that is not there, or no field at all. Fails on the first
    /// fault, in the order the metadata lists things.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let store = DirStore {
            root: path.as_ref().to_path_buf(),
        };
        let (grid, meta) = read_meta(&store)?;
        Ok(Self { store, grid, meta })
    }

    /// How the terrain is laid out in chunks, for a consumer to
    /// [locate](ChunkGrid::locate) a position with.
    pub fn grid(&self) -> ChunkGrid {
        self.grid
    }

    /// Every field the terrain carries, in the order the document declared them.
    pub fn fields(&self) -> &[FieldInfo] {
        &self.meta.fields
    }

    /// Reads one chunk: a [`Terrain`] over that chunk's extent, read in the chunk's
    /// own cells, carrying the same fields, value ranges and water as every other
    /// chunk.
    ///
    /// Reads that chunk's files and no other. Refused for a chunk outside the grid,
    /// a file larger than its image could encode to, or an image whose header is not
    /// the chunk's extent and the declared channels; returns no terrain on any
    /// fault.
    pub fn load_chunk(&self, chunk: UVec2) -> Result<Terrain, IoError> {
        load_chunk(&self.store, &self.grid, &self.meta, chunk)
    }
}

impl Terrain {
    /// Writes the terrain's values to a directory, one set of images per chunk of
    /// `grid`.
    ///
    /// Refused unless `grid`'s extent is the terrain's.
    ///
    /// **Deletes images.** Once `terrain.ron` is written, any image file the new
    /// metadata does not name — of this format or of the one before it — is
    /// removed, so a save over a larger terrain leaves nothing stale behind. That
    /// sweep runs only if this call created the directory or found a `terrain.ron`
    /// already in it; otherwise the files are written and the stale ones are left
    /// alone. Nothing outside those naming schemes is touched, so a recipe beside
    /// the values survives a save.
    ///
    /// A failure part way through leaves the directory partly written.
    pub fn save_to_dir(&self, path: impl AsRef<Path>, grid: &ChunkGrid) -> Result<(), IoError> {
        let root = path.as_ref().to_path_buf();
        let fresh = !root.exists();
        std::fs::create_dir_all(&root)?;
        let mut store = DirStore { root };
        let pruning = fresh || store.exists(META_FILE);
        write(&mut store, self, grid, pruning)
    }
}

fn write(
    store: &mut impl Store,
    terrain: &Terrain,
    grid: &ChunkGrid,
    pruning: bool,
) -> Result<(), IoError> {
    if grid.extent() != terrain.size() {
        return Err(IoError::Inconsistent(format!(
            "a {}x{} grid of {}x{} chunks does not cover a {}x{} terrain",
            grid.chunks().x,
            grid.chunks().y,
            grid.chunk_size().x,
            grid.chunk_size().y,
            terrain.size().x,
            terrain.size().y
        )));
    }
    let chunk_size = grid.chunk_size();
    let meta = TerrainMeta {
        version: VERSION,
        chunk_size,
        chunks: grid.chunks(),
        layers: terrain
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| LayerMeta {
                file: layer_stem(index),
                width: chunk_size.x,
                height: chunk_size.y,
                channels: layer.channels().to_vec(),
            })
            .collect(),
        fields: terrain.fields.clone(),
        water: terrain.water,
    };

    let mut named: BTreeSet<String> = BTreeSet::new();
    for chunk in grid.iter() {
        let part = terrain
            .chunk(grid, chunk)
            .ok_or_else(|| IoError::Inconsistent("a chunk could not be cut".to_owned()))?;
        for (layer, declared) in part.layers.iter().zip(&meta.layers) {
            let file = chunk_file(chunk, &declared.file);
            let bytes = encode_png(
                chunk_size,
                layer.texels().channels(),
                layer.texels().bytes(),
            )?;
            store.write(&file, &bytes)?;
            named.insert(file);
        }
    }

    let text = ron::ser::to_string_pretty(&meta, ron::ser::PrettyConfig::default())
        .map_err(|error| IoError::Meta(error.to_string()))?;
    store.write(META_FILE, text.as_bytes())?;

    if pruning {
        for name in store.list()? {
            if is_layer_file(&name) && !named.contains(&name) {
                store.remove(&name)?;
            }
        }
    }

    Ok(())
}

fn layer_stem(index: usize) -> String {
    format!("layer_{index:03}")
}

fn chunk_file(chunk: UVec2, stem: &str) -> String {
    format!("chunk_{}_{}_{stem}.png", chunk.x, chunk.y)
}

fn digits(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit())
}

fn is_layer_stem(stem: &str) -> bool {
    stem.strip_prefix("layer_").is_some_and(digits)
}

/// Whether `name` is one of the images a terrain's values are written as:
/// `chunk_<x>_<y>_layer_<n>.png`, or the `layer_<n>.png` of the format before
/// chunks, each number at least one digit.
pub fn is_layer_file(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".png") else {
        return false;
    };
    if is_layer_stem(stem) {
        return true;
    }
    let Some(rest) = stem.strip_prefix("chunk_") else {
        return false;
    };
    let mut parts = rest.splitn(3, '_');
    let (Some(x), Some(y), Some(layer)) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    digits(x) && digits(y) && is_layer_stem(layer)
}

fn read_meta(store: &impl Store) -> Result<(ChunkGrid, TerrainMeta), IoError> {
    let bytes = store.read(META_FILE, MAX_META_BYTES)?;
    let text = std::str::from_utf8(&bytes).map_err(|error| IoError::Meta(error.to_string()))?;
    let meta: TerrainMeta =
        ron::from_str(text).map_err(|error| IoError::Meta(error.to_string()))?;

    if meta.version != VERSION {
        return Err(IoError::UnsupportedVersion(meta.version));
    }
    let grid = ChunkGrid::new(meta.chunk_size, meta.chunks)
        .map_err(|error| IoError::Inconsistent(error.to_string()))?;
    let chunk_size = grid.chunk_size();
    let chunk_cells = chunk_size.x as u64 * chunk_size.y as u64;
    if chunk_cells > MAX_CELLS {
        return Err(IoError::Inconsistent(format!(
            "a chunk of {}x{} is over the {MAX_CELLS} cell limit",
            chunk_size.x, chunk_size.y
        )));
    }
    if meta.layers.len() > MAX_LAYERS {
        return Err(IoError::Inconsistent(format!(
            "{} layers is over the {MAX_LAYERS} limit",
            meta.layers.len()
        )));
    }

    let mut budget = MAX_DECODED_BYTES;
    for declared in &meta.layers {
        if !is_layer_stem(&declared.file) {
            return Err(IoError::BadFileName(declared.file.clone()));
        }
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
        if UVec2::new(declared.width, declared.height) != chunk_size {
            return Err(IoError::ImageShape {
                file: declared.file.clone(),
                found: format!("{}x{}", declared.width, declared.height),
                expected: format!("{}x{}, the chunk size", chunk_size.x, chunk_size.y),
            });
        }
        budget = budget
            .checked_sub(chunk_cells.saturating_mul(channels as u64))
            .ok_or_else(|| {
                IoError::Inconsistent(format!(
                    "one chunk's images are over the {MAX_DECODED_BYTES} byte limit"
                ))
            })?;
    }

    for field in &meta.fields {
        let layer = meta.layers.get(field.layer as usize).ok_or_else(|| {
            IoError::Inconsistent(format!(
                "field `{}` names layer {}, of {}",
                field.name,
                field.layer,
                meta.layers.len()
            ))
        })?;
        if field.channel as usize >= layer.channels.len() {
            return Err(IoError::Inconsistent(format!(
                "field `{}` names channel {} of {}",
                field.name,
                field.channel,
                layer.channels.len()
            )));
        }
    }

    if let Some(water) = meta.water {
        let layer = meta.layers.get(water.layer as usize).ok_or_else(|| {
            IoError::Inconsistent(format!(
                "the water names layer {}, of {}",
                water.layer,
                meta.layers.len()
            ))
        })?;
        if layer.channels.len() != MAX_CHANNELS {
            return Err(IoError::Inconsistent(format!(
                "the water layer holds {} channels, not {MAX_CHANNELS}",
                layer.channels.len()
            )));
        }
    }

    if meta.fields.is_empty() {
        return Err(IoError::NoFields);
    }
    Ok((grid, meta))
}

fn load_chunk(
    store: &impl Store,
    grid: &ChunkGrid,
    meta: &TerrainMeta,
    chunk: UVec2,
) -> Result<Terrain, IoError> {
    if !grid.contains(chunk) {
        return Err(IoError::Inconsistent(format!(
            "chunk ({}, {}) is outside the {}x{} grid",
            chunk.x,
            chunk.y,
            grid.chunks().x,
            grid.chunks().y
        )));
    }
    let size = grid.chunk_size();
    let mut layers = Vec::with_capacity(meta.layers.len());
    for declared in &meta.layers {
        let channels = declared.channels.len();
        let file = chunk_file(chunk, &declared.file);
        let bytes = store.read(&file, encoded_bound(size, channels))?;
        let pixels = decode_png(&file, &bytes, size, channels)?;
        let texels =
            LayerTexels::from_bytes(size, channels, pixels).ok_or_else(|| IoError::ImageShape {
                file: file.clone(),
                found: "a different number of texels".to_owned(),
                expected: format!("{}x{} of {channels}", size.x, size.y),
            })?;
        layers.push(TerrainLayer::new(declared.channels.clone(), texels));
    }
    Ok(Terrain::new(size, meta.fields.clone(), layers, meta.water))
}

fn encoded_bound(size: UVec2, channels: usize) -> u64 {
    let raw = (size.x as u64 * channels as u64 + 1) * size.y as u64;
    raw.saturating_mul(2).saturating_add(64 * 1024)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::channel::{ChannelEncoding, ChannelMeta};
    use crate::field::FieldRole;
    use crate::meta::WaterInfo;
    use crate::terrain::FieldInfo;

    const SIZE: UVec2 = UVec2::new(48, 32);

    #[derive(Default)]
    struct MemStore {
        files: BTreeMap<String, Vec<u8>>,
    }

    impl Store for MemStore {
        fn read(&self, name: &str, limit: u64) -> Result<Vec<u8>, IoError> {
            check_file_name(name)?;
            let bytes = self.files.get(name).cloned().ok_or_else(|| {
                IoError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    name.to_owned(),
                ))
            })?;
            if bytes.len() as u64 > limit {
                return Err(too_large(name, bytes.len() as u64, limit));
            }
            Ok(bytes)
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

    fn layer_of(channels: usize) -> TerrainLayer {
        let meta: Vec<ChannelMeta> = (0..channels)
            .map(|channel| ChannelMeta::linear(0.0, 1.0 + channel as f32))
            .collect();
        let mut bytes = Vec::with_capacity((SIZE.x * SIZE.y) as usize * channels);
        for index in 0..SIZE.x * SIZE.y {
            for channel in &meta {
                bytes.push(channel.encode((index % 17) as f32 / 17.0));
            }
        }
        let texels = LayerTexels::from_bytes(SIZE, channels, bytes).unwrap();
        TerrainLayer::new(meta, texels)
    }

    fn info(name: &str, role: FieldRole, layer: u8) -> FieldInfo {
        FieldInfo {
            name: name.to_owned(),
            role,
            categorical: false,
            layer,
            channel: 0,
        }
    }

    fn grid() -> ChunkGrid {
        ChunkGrid::new(UVec2::new(24, 16), UVec2::new(2, 2)).unwrap()
    }

    fn terrain() -> Terrain {
        Terrain {
            size: SIZE,
            fields: vec![
                info("height", FieldRole::Height, 0),
                info("moisture", FieldRole::Moisture, 1),
            ],
            layers: vec![layer_of(1), layer_of(1)],
            water: None,
        }
    }

    fn watered() -> Terrain {
        let mut terrain = terrain();
        terrain.layers.push(layer_of(MAX_CHANNELS));
        terrain.water = Some(WaterInfo { lakes: 3, layer: 2 });
        terrain
    }

    fn saved(terrain: &Terrain) -> MemStore {
        let mut store = MemStore::default();
        write(&mut store, terrain, &grid(), true).unwrap();
        store
    }

    fn load(store: &MemStore, chunk: UVec2) -> Result<Terrain, IoError> {
        let (grid, meta) = read_meta(store)?;
        load_chunk(store, &grid, &meta, chunk)
    }

    fn load_all(store: &MemStore) -> Result<Vec<Terrain>, IoError> {
        let (grid, meta) = read_meta(store)?;
        grid.iter()
            .map(|chunk| load_chunk(store, &grid, &meta, chunk))
            .collect()
    }

    fn meta_of_store(store: &MemStore) -> TerrainMeta {
        ron::from_str(std::str::from_utf8(&store.files[META_FILE]).unwrap()).unwrap()
    }

    fn first_file(meta: &TerrainMeta) -> String {
        chunk_file(UVec2::ZERO, &meta.layers[0].file)
    }

    fn stale() -> String {
        chunk_file(UVec2::new(9, 9), &layer_stem(0))
    }

    fn put_meta(store: &mut MemStore, meta: &TerrainMeta) {
        let text = ron::ser::to_string_pretty(meta, ron::ser::PrettyConfig::default()).unwrap();
        store.files.insert(META_FILE.to_owned(), text.into_bytes());
    }

    // The shape of the format, and the one thing a person opening the directory sees
    // first: metadata plus one image per chunk per layer, and nothing to guess at.
    #[test]
    fn a_terrain_is_one_metadata_file_and_one_image_per_chunk_per_layer() {
        let store = saved(&terrain());
        let meta = meta_of_store(&store);

        assert!(store.files.contains_key(META_FILE));
        assert_eq!(
            (meta.chunk_size, meta.chunks),
            (grid().chunk_size(), grid().chunks())
        );
        for chunk in grid().iter() {
            for layer in &meta.layers {
                let file = chunk_file(chunk, &layer.file);
                assert!(store.files.contains_key(&file), "{file}");
            }
        }
        assert_eq!(store.files.len(), meta.layers.len() * 4 + 1);
    }

    // The metadata is meant to be read and edited by hand, so it must stay a
    // description of the images rather than growing a copy of what is in them.
    #[test]
    fn the_metadata_is_readable_ron_that_carries_no_texels() {
        let store = saved(&terrain());
        let text = std::str::from_utf8(&store.files[META_FILE]).unwrap();

        assert!(text.contains("version"));
        assert!(text.len() < 4096, "the metadata is {} bytes", text.len());
    }

    // The consumer contract the chunk-consumer context asked for: one chunk loads
    // on its own, in its own cells, holding the same values the document had there.
    #[test]
    fn a_chunk_reads_the_documents_values_in_its_own_cells() {
        let store = saved(&terrain());
        let chunk = UVec2::new(1, 1);
        let loaded = load(&store, chunk).unwrap();
        let origin = grid().origin(chunk).unwrap();

        assert_eq!(loaded.size(), grid().chunk_size());
        for (x, y) in [(0, 0), (7, 11), (23, 15)] {
            assert_eq!(
                loaded.field("height").unwrap().value_at(x, y),
                terrain()
                    .field("height")
                    .unwrap()
                    .value_at(origin.x + x, origin.y + y)
            );
        }
    }

    // Loading a chunk must not depend on any other chunk's files, or a consumer
    // streaming chunks pays for the whole terrain.
    #[test]
    fn a_chunk_loads_with_every_other_chunks_files_gone() {
        let mut store = saved(&terrain());
        store
            .files
            .retain(|name, _| name == META_FILE || name.starts_with("chunk_0_1_"));

        assert!(load(&store, UVec2::new(0, 1)).is_ok());
        assert!(load(&store, UVec2::new(1, 1)).is_err());
    }

    // A chunk index comes from the consumer, so one past the grid has to be a
    // refusal rather than a file name that happens not to exist.
    #[test]
    fn a_chunk_outside_the_grid_is_refused() {
        let store = saved(&terrain());
        assert!(matches!(
            load(&store, UVec2::new(2, 0)),
            Err(IoError::Inconsistent(_))
        ));
    }

    // A png is lossless over the bytes it carries, so a value that survives the
    // quantisation has to survive the file exactly rather than approximately.
    #[test]
    fn a_written_value_reads_back_byte_for_byte() {
        let before = terrain();
        let store = saved(&before);
        for chunk in grid().iter() {
            let expected = before.chunk(&grid(), chunk).unwrap();
            let after = load(&store, chunk).unwrap();
            for index in 0..before.layer_count() {
                assert_eq!(
                    expected.layer(index).unwrap().bytes(),
                    after.layer(index).unwrap().bytes(),
                    "chunk {chunk} layer {index}"
                );
            }
        }
    }

    // A grid that does not cover the terrain would write chunks that slice past its
    // edge, so it is refused before anything is written.
    #[test]
    fn a_grid_that_does_not_cover_the_terrain_is_refused() {
        let mut store = MemStore::default();
        let wrong = ChunkGrid::new(UVec2::new(16, 16), UVec2::new(2, 2)).unwrap();
        assert!(matches!(
            write(&mut store, &terrain(), &wrong, true),
            Err(IoError::Inconsistent(_))
        ));
        assert!(store.files.is_empty());
    }

    // Saving over a terrain that had more images has to remove the ones it no longer
    // names, or a later load would read a stale image the metadata never mentions.
    #[test]
    fn a_save_removes_an_image_it_no_longer_names() {
        let mut store = saved(&watered());
        let before = store.files.len();
        store.files.insert(stale(), b"stale".to_vec());

        write(&mut store, &terrain(), &grid(), true).unwrap();

        assert!(!store.files.contains_key(&stale()));
        assert!(store.files.len() < before);
        load_all(&store).unwrap();
    }

    // No migration is offered (format-replace), so a save over a directory of the
    // old format has to sweep its images without being able to read its metadata.
    #[test]
    fn a_save_over_the_old_format_removes_its_images() {
        let mut store = MemStore::default();
        store
            .files
            .insert(META_FILE.to_owned(), b"(version: 2, size_x: 4)".to_vec());
        store
            .files
            .insert("layer_000.png".to_owned(), b"old".to_vec());

        write(&mut store, &terrain(), &grid(), true).unwrap();

        assert!(!store.files.contains_key("layer_000.png"));
        load_all(&store).unwrap();
    }

    // The sweep must never touch a file this writer would not itself have produced:
    // a terrain directory that also holds a recipe, or a person's notes, stays intact.
    #[test]
    fn a_save_leaves_a_file_it_did_not_write() {
        let mut store = saved(&terrain());
        store.files.insert("notes.txt".to_owned(), b"mine".to_vec());
        store
            .files
            .insert("recipe.ron".to_owned(), b"mine".to_vec());
        store
            .files
            .insert("paint_000.png".to_owned(), b"mine".to_vec());

        write(&mut store, &terrain(), &grid(), true).unwrap();

        assert_eq!(store.files.get("notes.txt").unwrap(), b"mine");
        assert_eq!(store.files.get("recipe.ron").unwrap(), b"mine");
        assert_eq!(store.files.get("paint_000.png").unwrap(), b"mine");
    }

    // Not pruning is the answer for a directory that is not already a terrain, which
    // is what stops a mistyped save path deleting someone's files.
    #[test]
    fn a_save_that_is_not_pruning_removes_nothing() {
        let mut store = saved(&watered());
        store.files.insert(stale(), b"stale".to_vec());

        write(&mut store, &terrain(), &grid(), false).unwrap();

        assert_eq!(store.files.get(&stale()).unwrap(), b"stale");
    }

    // The sweep decides deletion by name alone, so the name test has to accept
    // exactly what a writer of either format produces.
    #[test]
    fn only_names_a_writer_produces_are_layer_files() {
        for name in [
            "chunk_0_0_layer_000.png",
            "chunk_12_3_layer_7.png",
            "layer_000.png",
        ] {
            assert!(is_layer_file(name), "{name}");
        }
        for name in [
            "chunk_0_layer_000.png",
            "chunk_a_0_layer_000.png",
            "chunk_0_0_paint_000.png",
            "chunk_0_0_layer_.png",
            "layer_000.ron",
            "notes.txt",
        ] {
            assert!(!is_layer_file(name), "{name}");
        }
    }

    // There is no migration path, so another version has to be refused outright
    // rather than read as far as it happens to agree.
    #[test]
    fn a_terrain_from_another_version_is_refused() {
        let mut store = saved(&terrain());
        let mut meta = meta_of_store(&store);
        meta.version = 99;
        put_meta(&mut store, &meta);

        assert!(matches!(
            read_meta(&store),
            Err(IoError::UnsupportedVersion(99))
        ));
    }

    // A directory that is not a terrain has to be refused rather than read as one
    // whose metadata happens to be missing.
    #[test]
    fn a_directory_that_is_not_a_terrain_is_refused() {
        let store = MemStore::default();
        assert!(matches!(read_meta(&store), Err(IoError::Io(_))));
    }

    // Nothing re-derives a bake on the way in, so a terrain with no fields would
    // otherwise load as one that answers nothing at every cell.
    #[test]
    fn a_terrain_carrying_no_field_is_refused() {
        let mut store = saved(&terrain());
        let mut meta = meta_of_store(&store);
        meta.fields.clear();
        meta.water = None;
        put_meta(&mut store, &meta);

        assert!(matches!(read_meta(&store), Err(IoError::NoFields)));
    }

    // The grid comes out of an untrusted file, and a zero in it is a division by
    // zero for every consumer that locates a position.
    #[test]
    fn a_zero_chunk_grid_is_refused() {
        for (chunk_size, chunks) in [(UVec2::new(0, 16), UVec2::ONE), (UVec2::ONE, UVec2::ZERO)] {
            let mut store = saved(&terrain());
            let mut meta = meta_of_store(&store);
            meta.chunk_size = chunk_size;
            meta.chunks = chunks;
            put_meta(&mut store, &meta);

            assert!(matches!(read_meta(&store), Err(IoError::Inconsistent(_))));
        }
    }

    // The cell limit bounds one chunk's allocation now that the document as a whole
    // has none, so a chunk past it is refused before any image is read.
    #[test]
    fn a_chunk_over_the_cell_limit_is_refused() {
        let mut store = saved(&terrain());
        let mut meta = meta_of_store(&store);
        meta.chunk_size = UVec2::new(1 << 16, 1 << 15);
        for layer in &mut meta.layers {
            layer.width = meta.chunk_size.x;
            layer.height = meta.chunk_size.y;
        }
        put_meta(&mut store, &meta);

        assert!(matches!(read_meta(&store), Err(IoError::Inconsistent(_))));
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
        assert!(check_file_name("chunk_0_0_layer_000.png").is_ok());
    }

    // The stem is spliced into every chunk file name, so anything but the shape a
    // writer produces is refused before a name is built from it.
    #[test]
    fn a_layer_stem_that_is_not_one_a_writer_produces_is_refused() {
        let mut store = saved(&terrain());
        let mut meta = meta_of_store(&store);
        meta.layers[0].file = "../escape".to_owned();
        put_meta(&mut store, &meta);

        assert!(matches!(read_meta(&store), Err(IoError::BadFileName(_))));
    }

    // The metadata names every file the loader opens, so a name it cannot open has
    // to be an error rather than a layer quietly missing from the terrain.
    #[test]
    fn an_image_the_metadata_names_but_the_directory_lacks_is_refused() {
        let mut store = saved(&terrain());
        let meta = meta_of_store(&store);
        store.files.remove(&first_file(&meta));

        assert!(matches!(load(&store, UVec2::ZERO), Err(IoError::Io(_))));
    }

    // Every image is the chunk's extent, and the metadata saying otherwise is refused
    // before the image's pixels are allocated.
    #[test]
    fn an_image_not_at_the_chunk_size_is_refused() {
        let mut store = saved(&terrain());
        let mut meta = meta_of_store(&store);
        meta.layers[0].width += 1;
        put_meta(&mut store, &meta);

        assert!(matches!(read_meta(&store), Err(IoError::ImageShape { .. })));
    }

    // A channel count the image does not have would otherwise be read as a different
    // number of texels at the same size.
    #[test]
    fn an_image_with_the_wrong_channel_count_is_refused() {
        let mut store = saved(&terrain());
        let mut meta = meta_of_store(&store);
        meta.layers[0].channels.push(ChannelMeta::linear(0.0, 1.0));
        put_meta(&mut store, &meta);

        assert!(matches!(
            load(&store, UVec2::ZERO),
            Err(IoError::ImageShape { .. })
        ));
    }

    // A file is checked against what its image could encode to before it is read
    // into memory, so a huge file named by a small chunk is never read whole.
    #[test]
    fn an_image_file_larger_than_its_chunk_could_encode_to_is_refused() {
        let mut store = saved(&terrain());
        let meta = meta_of_store(&store);
        store
            .files
            .insert(first_file(&meta), vec![0; 4 * 1024 * 1024]);

        assert!(matches!(
            load(&store, UVec2::ZERO),
            Err(IoError::FileTooLarge { .. })
        ));
    }

    // An index out of a hand-edited file has to be refused rather than panicking or
    // silently dropping the field.
    #[test]
    fn a_field_pointing_past_the_images_is_refused() {
        let mut store = saved(&terrain());
        let mut meta = meta_of_store(&store);
        meta.fields[0].layer = 200;
        put_meta(&mut store, &meta);
        assert!(matches!(read_meta(&store), Err(IoError::Inconsistent(_))));

        let mut store = saved(&terrain());
        let mut meta = meta_of_store(&store);
        meta.fields[0].channel = 200;
        put_meta(&mut store, &meta);
        assert!(matches!(read_meta(&store), Err(IoError::Inconsistent(_))));
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
            let mut store = saved(&terrain());
            let mut meta = meta_of_store(&store);
            meta.layers[0].channels[0] = broken;
            put_meta(&mut store, &meta);

            assert!(
                matches!(read_meta(&store), Err(IoError::Channel(_))),
                "{broken:?}"
            );
        }
    }

    // Half an image is still a file the metadata names, so only the decode catches
    // it; a short read that produced a terrain would hand back silent zeroes.
    #[test]
    fn a_truncated_image_is_refused_rather_than_read_short() {
        let mut store = saved(&terrain());
        let meta = meta_of_store(&store);
        let file = first_file(&meta);
        let bytes = store.files[&file].clone();
        store.files.insert(file, bytes[..bytes.len() / 2].to_vec());

        assert!(load(&store, UVec2::ZERO).is_err());
    }

    // A truncated metadata file fails to parse rather than being read as a terrain
    // that simply declares less than it did.
    #[test]
    fn truncated_metadata_is_refused() {
        let mut store = saved(&terrain());
        let bytes = store.files[META_FILE].clone();
        store
            .files
            .insert(META_FILE.to_owned(), bytes[..bytes.len() / 2].to_vec());

        assert!(matches!(read_meta(&store), Err(IoError::Meta(_))));
    }

    // All of the water is one image, chunked like every other, which is what four
    // channels bought; a reader validates one index rather than five.
    #[test]
    fn the_water_is_a_single_four_channel_image_in_every_chunk() {
        let store = saved(&watered());
        let meta = meta_of_store(&store);

        let water = meta.water.unwrap();
        let layer = &meta.layers[water.layer as usize];
        assert_eq!(layer.channels.len(), MAX_CHANNELS);
        assert_eq!(UVec2::new(layer.width, layer.height), grid().chunk_size());
        for chunk in load_all(&store).unwrap() {
            assert!(chunk.water().is_some());
        }
    }

    // The water indices come out of the same untrusted file as everything else.
    #[test]
    fn a_water_image_that_is_not_four_channels_is_refused() {
        let mut store = saved(&watered());
        let mut meta = meta_of_store(&store);
        meta.water = Some(WaterInfo { lakes: 1, layer: 0 });
        put_meta(&mut store, &meta);

        assert!(matches!(read_meta(&store), Err(IoError::Inconsistent(_))));
    }

    // A terrain that carries no water must load as one that has none, not as one
    // that acquires an empty solve.
    #[test]
    fn a_terrain_with_no_water_image_loads_with_no_water() {
        assert!(
            load(&saved(&terrain()), UVec2::ZERO)
                .unwrap()
                .water()
                .is_none()
        );
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
    // exercises the real thing: creating the directory and reading a chunk back off a
    // disk through the consumer's own path.
    #[test]
    fn a_terrain_round_trips_through_a_real_directory() {
        let root = scratch("round-trip");
        watered().save_to_dir(&root, &grid()).unwrap();

        assert!(root.join(META_FILE).is_file());
        let dir = TerrainDir::open(&root).unwrap();
        assert_eq!(dir.grid(), grid());
        let at = dir.grid().locate(30.5, 20.5).unwrap();
        assert_eq!(at.chunk, UVec2::new(1, 1));
        let loaded = dir.load_chunk(at.chunk).unwrap();
        assert!(
            loaded
                .field("height")
                .unwrap()
                .value_at(at.texel.x, at.texel.y)
                .is_some()
        );
        assert!(loaded.water().is_some());

        std::fs::remove_dir_all(&root).unwrap();
    }

    // The destructive half, against a real directory: a stale image goes, and a file
    // this writer would never have produced stays.
    #[test]
    fn a_real_save_prunes_its_own_stale_images_only() {
        let root = scratch("prune");
        watered().save_to_dir(&root, &grid()).unwrap();
        std::fs::write(root.join(stale()), b"stale").unwrap();
        std::fs::write(root.join("notes.txt"), b"mine").unwrap();

        terrain().save_to_dir(&root, &grid()).unwrap();

        assert!(!root.join(stale()).exists());
        assert_eq!(std::fs::read(root.join("notes.txt")).unwrap(), b"mine");
        TerrainDir::open(&root)
            .unwrap()
            .load_chunk(UVec2::ZERO)
            .unwrap();
        std::fs::remove_dir_all(&root).unwrap();
    }

    // Saving into a directory that is not already a terrain must not delete what is
    // in it — the guard against a mistyped save path.
    #[test]
    fn a_save_into_a_foreign_directory_deletes_nothing() {
        let root = scratch("foreign");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join(stale()), b"someone else's").unwrap();

        terrain().save_to_dir(&root, &grid()).unwrap();

        assert_eq!(
            std::fs::read(root.join(stale())).unwrap(),
            b"someone else's"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }
}
