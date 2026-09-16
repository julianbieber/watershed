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
use crate::meta::{LayerMeta, TerrainMeta, VERSION};
use crate::raster::{MAX_SHIFT, resolution};
use crate::terrain::{LayerTexels, Terrain, TerrainLayer};

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

    /// Writes the terrain's values to a directory.
    ///
    /// **Deletes images.** Once `terrain.ron` is written, any `layer_<n>.png` the
    /// new metadata does not name is removed, so a save over a larger terrain
    /// leaves nothing stale behind. That sweep runs only if this call created the
    /// directory or found a `terrain.ron` already in it; otherwise the files are
    /// written and the stale ones are left alone. Nothing outside that naming
    /// scheme is touched, so a recipe beside the values survives a save.
    ///
    /// A failure part way through leaves the directory partly written.
    pub fn save_to_dir(&self, path: impl AsRef<Path>) -> Result<(), IoError> {
        let root = path.as_ref().to_path_buf();
        let fresh = !root.exists();
        std::fs::create_dir_all(&root)?;
        let mut store = DirStore { root };
        let pruning = fresh || store.exists(META_FILE);
        write(&mut store, self, pruning)
    }
}

fn meta_of(terrain: &Terrain) -> TerrainMeta {
    TerrainMeta {
        version: VERSION,
        size_x: terrain.size().x,
        size_y: terrain.size().y,
        layers: Vec::new(),
        fields: terrain.fields.clone(),
        water: terrain.water,
    }
}

fn write(store: &mut impl Store, terrain: &Terrain, pruning: bool) -> Result<(), IoError> {
    let mut meta = meta_of(terrain);

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

/// Whether `name` is one of the images a terrain's values are written as:
/// `layer_<n>.png` with at least one digit.
pub fn is_layer_file(name: &str) -> bool {
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

    fn layer_of(shift: u8, channels: usize) -> TerrainLayer {
        let texels = resolution(SIZE, shift);
        let meta: Vec<ChannelMeta> = (0..channels)
            .map(|channel| ChannelMeta::linear(0.0, 1.0 + channel as f32))
            .collect();
        let mut bytes = Vec::with_capacity((texels.x * texels.y) as usize * channels);
        for index in 0..texels.x * texels.y {
            for channel in &meta {
                bytes.push(channel.encode((index % 17) as f32 / 17.0));
            }
        }
        let texels = LayerTexels::from_bytes(texels, channels, bytes).unwrap();
        TerrainLayer::new(Some(shift), meta, texels)
    }

    fn info(name: &str, role: FieldRole, shift: u8, layer: u8) -> FieldInfo {
        FieldInfo {
            name: name.to_owned(),
            role,
            shift,
            categorical: false,
            layer,
            channel: 0,
        }
    }

    fn terrain() -> Terrain {
        Terrain {
            size: SIZE,
            fields: vec![
                info("height", FieldRole::Height, 0, 0),
                info("moisture", FieldRole::Moisture, 2, 1),
            ],
            layers: vec![layer_of(0, 1), layer_of(2, 1)],
            water: None,
        }
    }

    fn watered() -> Terrain {
        let mut terrain = terrain();
        terrain.layers.push(layer_of(0, MAX_CHANNELS));
        terrain.water = Some(WaterInfo { lakes: 3, layer: 2 });
        terrain
    }

    fn saved(terrain: &Terrain) -> MemStore {
        let mut store = MemStore::default();
        write(&mut store, terrain, true).unwrap();
        store
    }

    fn meta_of_store(store: &MemStore) -> TerrainMeta {
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
        let store = saved(&terrain());
        let meta = meta_of_store(&store);

        assert!(store.files.contains_key(META_FILE));
        assert!(!meta.layers.is_empty());
        for layer in &meta.layers {
            assert!(store.files.contains_key(&layer.file), "{}", layer.file);
        }
        assert_eq!(store.files.len(), meta.layers.len() + 1);
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

    // The whole point of the format: what is written is what is read, with no bake
    // and no solve on the way in.
    #[test]
    fn a_consumer_reads_values_without_baking() {
        let store = saved(&terrain());
        let (loaded, _) = read(&store).unwrap();

        assert_eq!(loaded.size(), SIZE);
        assert_eq!(loaded.field("height").unwrap().shift(), 0);
        assert_eq!(loaded.field("moisture").unwrap().shift(), 2);
        for (x, y) in [(0, 0), (7, 11), (47, 31)] {
            assert_eq!(
                loaded.field("height").unwrap().value_at(x, y),
                terrain().field("height").unwrap().value_at(x, y)
            );
        }
    }

    // A png is lossless over the bytes it carries, so a value that survives the
    // quantisation has to survive the file exactly rather than approximately.
    #[test]
    fn a_written_value_reads_back_byte_for_byte() {
        let before = terrain();
        let (after, _) = read(&saved(&before)).unwrap();

        for index in 0..before.layer_count() {
            assert_eq!(
                before.layer(index).unwrap().bytes(),
                after.layer(index).unwrap().bytes(),
                "layer {index}"
            );
        }
    }

    // Saving over a terrain that had more images has to remove the ones it no longer
    // names, or a later load would read a stale image the metadata never mentions.
    #[test]
    fn a_save_removes_an_image_it_no_longer_names() {
        let mut store = saved(&watered());
        let before = store.files.len();
        store.files.insert(layer_file(99), b"stale".to_vec());

        write(&mut store, &terrain(), true).unwrap();

        assert!(!store.files.contains_key(&layer_file(99)));
        assert!(store.files.len() < before);
        read(&store).unwrap();
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

        write(&mut store, &terrain(), true).unwrap();

        assert_eq!(store.files.get("notes.txt").unwrap(), b"mine");
        assert_eq!(store.files.get("recipe.ron").unwrap(), b"mine");
        assert_eq!(store.files.get("paint_000.png").unwrap(), b"mine");
    }

    // Not pruning is the answer for a directory that is not already a terrain, which
    // is what stops a mistyped save path deleting someone's files.
    #[test]
    fn a_save_that_is_not_pruning_removes_nothing() {
        let mut store = saved(&watered());
        store.files.insert(layer_file(99), b"stale".to_vec());

        write(&mut store, &terrain(), false).unwrap();

        assert_eq!(store.files.get(&layer_file(99)).unwrap(), b"stale");
    }

    // There is no migration path, so a newer terrain has to be refused outright
    // rather than read as far as it happens to agree.
    #[test]
    fn a_terrain_from_a_later_version_is_refused() {
        let mut store = saved(&terrain());
        let mut meta = meta_of_store(&store);
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

    // Nothing re-derives a bake on the way in, so a terrain with no fields would
    // otherwise load as one that answers nothing at every cell.
    #[test]
    fn a_terrain_carrying_no_field_is_refused() {
        let mut store = saved(&terrain());
        let mut meta = meta_of_store(&store);
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
        let mut store = saved(&terrain());
        let meta = meta_of_store(&store);
        store.files.remove(&meta.layers[0].file);

        assert!(matches!(read(&store), Err(IoError::Io(_))));
    }

    // The extent is checked against the image's own header before its pixels are
    // allocated, which is what stops a small file claiming an enormous one.
    #[test]
    fn an_image_at_the_wrong_extent_is_refused() {
        let mut store = saved(&terrain());
        let mut meta = meta_of_store(&store);
        meta.layers[0].width += 1;
        put_meta(&mut store, &meta);

        assert!(matches!(read(&store), Err(IoError::ImageShape { .. })));
    }

    // A channel count the image does not have would otherwise be read as a different
    // number of texels at the same size.
    #[test]
    fn an_image_with_the_wrong_channel_count_is_refused() {
        let mut store = saved(&terrain());
        let mut meta = meta_of_store(&store);
        meta.layers[0].channels.push(ChannelMeta::linear(0.0, 1.0));
        put_meta(&mut store, &meta);

        assert!(matches!(read(&store), Err(IoError::ImageShape { .. })));
    }

    // Two shifts can imply the same image extent, so a shift out of range is not
    // caught by the extent check and needs refusing on its own.
    #[test]
    fn a_shift_past_what_the_grid_honours_is_refused() {
        let mut store = saved(&terrain());
        let mut meta = meta_of_store(&store);
        meta.layers[0].shift = Some(200);
        put_meta(&mut store, &meta);

        assert!(matches!(read(&store), Err(IoError::Inconsistent(_))));
    }

    // A field's shift decides how a cell maps to a texel; if it disagrees with the
    // image it points at, the read is bounds-checked against one grid and performed
    // against another.
    #[test]
    fn a_field_whose_shift_is_not_its_layers_is_refused() {
        let mut store = saved(&terrain());
        let mut meta = meta_of_store(&store);
        meta.fields[0].shift = meta.fields[0].shift.wrapping_add(1);
        put_meta(&mut store, &meta);

        assert!(matches!(read(&store), Err(IoError::Inconsistent(_))));
    }

    // An index out of a hand-edited file has to be refused rather than panicking or
    // silently dropping the field.
    #[test]
    fn a_field_pointing_past_the_images_is_refused() {
        let mut store = saved(&terrain());
        let mut meta = meta_of_store(&store);
        meta.fields[0].layer = 200;
        put_meta(&mut store, &meta);
        assert!(matches!(read(&store), Err(IoError::Inconsistent(_))));

        let mut store = saved(&terrain());
        let mut meta = meta_of_store(&store);
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
            let mut store = saved(&terrain());
            let mut meta = meta_of_store(&store);
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
        let mut store = saved(&terrain());
        let meta = meta_of_store(&store);
        let file = meta.layers[0].file.clone();
        let bytes = store.files[&file].clone();
        store.files.insert(file, bytes[..bytes.len() / 2].to_vec());

        assert!(read(&store).is_err());
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

        assert!(matches!(read(&store), Err(IoError::Meta(_))));
    }

    // All of the water is one image, which is what four channels bought; a reader
    // validates one index rather than five.
    #[test]
    fn the_water_is_a_single_four_channel_image() {
        let store = saved(&watered());
        let meta = meta_of_store(&store);

        let water = meta.water.unwrap();
        let layer = &meta.layers[water.layer as usize];
        assert_eq!(layer.channels.len(), MAX_CHANNELS);
        assert_eq!((layer.width, layer.height), (SIZE.x, SIZE.y));
        assert!(read(&store).unwrap().0.water().is_some());
    }

    // The water indices come out of the same untrusted file as everything else.
    #[test]
    fn a_water_image_that_is_not_four_channels_is_refused() {
        let mut store = saved(&watered());
        let mut meta = meta_of_store(&store);
        meta.water = Some(WaterInfo { lakes: 1, layer: 0 });
        put_meta(&mut store, &meta);

        assert!(matches!(read(&store), Err(IoError::Inconsistent(_))));
    }

    // A terrain that carries no water must load as one that has none, not as one
    // that acquires an empty solve.
    #[test]
    fn a_terrain_with_no_water_image_loads_with_no_water() {
        assert!(read(&saved(&terrain())).unwrap().0.water().is_none());
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
    // exercises the real thing: creating the directory and reading it back off a disk.
    #[test]
    fn a_terrain_round_trips_through_a_real_directory() {
        let root = scratch("round-trip");
        watered().save_to_dir(&root).unwrap();

        assert!(root.join(META_FILE).is_file());
        let loaded = Terrain::load_from_dir(&root).unwrap();
        assert_eq!(loaded.size(), SIZE);
        assert!(loaded.field("height").unwrap().value_at(1, 1).is_some());
        assert!(loaded.water().is_some());

        std::fs::remove_dir_all(&root).unwrap();
    }

    // The destructive half, against a real directory: a stale image goes, and a file
    // this writer would never have produced stays.
    #[test]
    fn a_real_save_prunes_its_own_stale_images_only() {
        let root = scratch("prune");
        watered().save_to_dir(&root).unwrap();
        std::fs::write(root.join(layer_file(99)), b"stale").unwrap();
        std::fs::write(root.join("notes.txt"), b"mine").unwrap();

        terrain().save_to_dir(&root).unwrap();

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

        terrain().save_to_dir(&root).unwrap();

        assert_eq!(
            std::fs::read(root.join(layer_file(7))).unwrap(),
            b"someone else's"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }
}
