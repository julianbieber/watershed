//! The on-disk form of a document: what is written, what is left to be re-derived on
//! load, and what a reader is allowed to assume about a file it did not write.
//!
//! Every number in a file is little-endian whatever the host, and every raster is
//! zstd-compressed, so a document written on one machine loads bit-identically on
//! another.

use std::io::{Read, Write};
use std::path::Path;

use glam::UVec2;
use thiserror::Error;

use crate::bake::{PlanError, TerrainSpec};
use crate::field::Field;
use crate::layer::{LayerOp, Mask};
use crate::raster::Raster;
use crate::water::{WaterError, WaterState};

/// The first four bytes of every file this module writes, whatever [`SaveOptions`]
/// produced it — a bake-only export is the same format with different flags, not a
/// second one, so a reader identifies a watershed file before it knows what is in
/// it.
pub const MAGIC: [u8; 4] = *b"WSHD";

/// The format version this build writes, and the only one it reads. A file carrying
/// any other version is refused outright; there is no migration path.
pub const VERSION: u16 = 1;

const FLAG_BAKES: u16 = 1 << 0;
const FLAG_WATER: u16 = 1 << 1;

const MAX_HEADER_BYTES: u32 = 64 * 1024 * 1024;

const COMPRESSION_LEVEL: i32 = 3;

/// Why a document could not be written or read.
///
/// A load fails on the first of these it meets and produces no document, so a
/// caller never gets a partially read one. The last two are not about the file at
/// all: they are what a well-formed file that left work to be re-derived on load
/// can fail at.
#[derive(Debug, Error)]
pub enum IoError {
    /// The underlying reader or writer. A truncated file arrives here.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// The first four bytes are not [`MAGIC`] — not a watershed file.
    #[error("not a watershed file: magic is {0:?}")]
    BadMagic([u8; 4]),
    /// A watershed file this build cannot read. There is no fallback.
    #[error("file version {0} is not supported; this build reads {VERSION}")]
    UnsupportedVersion(u16),
    /// The header length is read before anything has been validated, so it is
    /// bounded before it is allocated: this is what a corrupt or hostile file hits
    /// instead of an allocation the size of the number it claimed.
    #[error("header is {0} bytes, over the {MAX_HEADER_BYTES} byte limit")]
    HeaderTooLarge(u32),
    /// The header is not UTF-8, or is not a document.
    #[error("header is not readable: {0}")]
    Header(String),
    /// A raster block's byte count does not match the dimensions declared with it,
    /// or its compressed length is larger than anything the compressor could have
    /// produced for that many bytes. Bounded before it is allocated, for the same
    /// reason as the header.
    #[error("a {kind} block is {found} bytes where {expected} were expected")]
    BlockSize {
        /// The element type of the block that failed, for the message.
        kind: &'static str,
        /// Bytes the header implies, or the ceiling a compressed block may not pass.
        expected: usize,
        /// Bytes the file claims.
        found: usize,
    },
    /// A block is well-formed but the wrong shape for the document it arrived with —
    /// a bake at a resolution its field does not have, or water not at the
    /// document's size.
    #[error("a {kind} block is {width} by {height}, which is not the document's {expected}")]
    BlockShape {
        /// `"bake"` or `"water"`.
        kind: &'static str,
        /// Columns the block declares.
        width: u32,
        /// Rows the block declares.
        height: u32,
        /// What the document requires.
        expected: UVec2,
    },
    /// The file carried no bakes and the document it describes cannot be planned.
    #[error("bake: {0}")]
    Bake(#[from] PlanError),
    /// The file carried no water and the spec it carries cannot be solved.
    #[error("water: {0}")]
    Water(#[from] WaterError),
}

/// What a save writes, beyond the field metadata every file carries.
///
/// Each of the three is a trade of file size against work on load. Anything left
/// out is re-derived when the file is read, so every combination loads to the same
/// document — the constructors name the four that are worth having.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaveOptions {
    /// Write the layer stacks, and the painted rasters inside them. Without this
    /// the loaded document has values but no recipe, and cannot be edited or
    /// re-baked.
    pub layers: bool,
    /// Write the baked rasters. Skipped silently if any field is unbaked or baked
    /// at the wrong resolution, in which case the load re-bakes.
    pub bakes: bool,
    /// Write the solved water. Skipped silently if it is not at the document's
    /// size, in which case the load re-solves from the spec the header carries.
    pub water: bool,
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self::document()
    }
}

impl SaveOptions {
    /// Layers and water, no bakes. The default, and what an editor saves.
    ///
    /// The two derived things cost about the same on disk and are very different to
    /// re-derive: on a 4096-square document with two fields the bakes are around
    /// 55 MB and take about two seconds to redo, while the water is around 50 MB and
    /// takes about five. So the expensive one is carried and the cheap one is paid
    /// for on every load.
    pub fn document() -> Self {
        Self {
            layers: true,
            bakes: false,
            water: true,
        }
    }

    /// The recipe and nothing derived from it — the smallest file, and the slowest
    /// to load, since both the bake and the water solve run on the way in. On the
    /// document measured above that is around 56 MB and about eight seconds.
    ///
    /// This is the shape a document had before the water was carried, so it is also
    /// what proves a file with neither block still loads to the same document.
    pub fn layers_only() -> Self {
        Self {
            layers: true,
            bakes: false,
            water: false,
        }
    }

    /// Everything. The largest file and the fastest load, and the only one that
    /// preserves a baked raster exactly as it stands — including one a caller wrote
    /// into by hand, which a re-bake would overwrite.
    pub fn full() -> Self {
        Self {
            layers: true,
            bakes: true,
            water: true,
        }
    }

    /// Values without the recipe: the field metadata, the baked rasters and the
    /// water, but no layers.
    ///
    /// For a consumer that reads a terrain and never authors one. What loads from it
    /// cannot be edited or re-baked — the fields come back with empty stacks — so
    /// saving one again would lose the document.
    pub fn bakes_only() -> Self {
        Self {
            layers: false,
            bakes: true,
            water: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Op(usize, usize),
    MaskPainted(usize, usize),
}

fn painted_slots(terrain: &TerrainSpec) -> Vec<Slot> {
    let mut slots = Vec::new();
    for (field_index, field) in terrain.fields.iter().enumerate() {
        for (layer_index, layer) in field.layers.iter().enumerate() {
            if matches!(layer.op, LayerOp::Paint(_) | LayerOp::External(_)) {
                slots.push(Slot::Op(field_index, layer_index));
            }
            if matches!(layer.mask, Mask::Painted(_)) {
                slots.push(Slot::MaskPainted(field_index, layer_index));
            }
        }
    }
    slots
}

trait Element: Copy + Default {
    const WIDTH: usize;
    const KIND: &'static str;
    fn write_le(self, out: &mut Vec<u8>);
    fn read_le(bytes: &[u8]) -> Self;
}

macro_rules! element {
    ($type:ty, $kind:literal) => {
        impl Element for $type {
            const WIDTH: usize = size_of::<$type>();
            const KIND: &'static str = $kind;

            fn write_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }

            fn read_le(bytes: &[u8]) -> Self {
                let mut buffer = [0u8; size_of::<$type>()];
                buffer.copy_from_slice(bytes);
                <$type>::from_le_bytes(buffer)
            }
        }
    };
}

element!(f32, "f32");
element!(u8, "u8");
element!(u16, "u16");
element!(u32, "u32");

fn write_u16(writer: &mut impl Write, value: u16) -> Result<(), IoError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u32(writer: &mut impl Write, value: u32) -> Result<(), IoError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_u16(reader: &mut impl Read) -> Result<u16, IoError> {
    let mut buffer = [0u8; 2];
    reader.read_exact(&mut buffer)?;
    Ok(u16::from_le_bytes(buffer))
}

fn read_u32(reader: &mut impl Read) -> Result<u32, IoError> {
    let mut buffer = [0u8; 4];
    reader.read_exact(&mut buffer)?;
    Ok(u32::from_le_bytes(buffer))
}

fn write_raster<T: Element>(writer: &mut impl Write, raster: &Raster<T>) -> Result<(), IoError> {
    let mut raw = Vec::with_capacity(raster.len() * T::WIDTH);
    for texel in raster.data() {
        texel.write_le(&mut raw);
    }
    let packed = zstd::encode_all(&raw[..], COMPRESSION_LEVEL)?;

    write_u32(writer, raster.width())?;
    write_u32(writer, raster.height())?;
    write_u32(writer, raw.len() as u32)?;
    write_u32(writer, packed.len() as u32)?;
    writer.write_all(&packed)?;
    Ok(())
}

fn read_raster<T: Element>(reader: &mut impl Read) -> Result<Raster<T>, IoError> {
    let width = read_u32(reader)?;
    let height = read_u32(reader)?;
    let raw_len = read_u32(reader)? as usize;
    let packed_len = read_u32(reader)? as usize;

    let expected = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(T::WIDTH);
    if raw_len != expected {
        return Err(IoError::BlockSize {
            kind: T::KIND,
            expected,
            found: raw_len,
        });
    }

    let ceiling = raw_len + raw_len / 8 + 1024;
    if packed_len > ceiling {
        return Err(IoError::BlockSize {
            kind: T::KIND,
            expected: ceiling,
            found: packed_len,
        });
    }

    let mut packed = vec![0u8; packed_len];
    reader.read_exact(&mut packed)?;
    let raw = zstd::decode_all(&packed[..])?;
    if raw.len() != raw_len {
        return Err(IoError::BlockSize {
            kind: T::KIND,
            expected: raw_len,
            found: raw.len(),
        });
    }

    let data = raw.chunks_exact(T::WIDTH).map(T::read_le).collect();
    Raster::from_vec(UVec2::new(width, height), data).ok_or(IoError::BlockSize {
        kind: T::KIND,
        expected,
        found: raw_len,
    })
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

fn header_document(terrain: &TerrainSpec, layers: bool) -> TerrainSpec {
    let mut header = TerrainSpec::new(terrain.size);
    header.water_spec = terrain.water_spec.clone();
    for field in &terrain.fields {
        let mut stripped = Field::new(field.id.clone())
            .with_shift(field.shift)
            .with_range(field.range);
        if layers {
            stripped.layers = field
                .layers
                .iter()
                .map(|layer| {
                    let mut layer = layer.clone();
                    layer.op = stripped_op(&layer.op);
                    layer.mask = stripped_mask(&layer.mask);
                    layer
                })
                .collect();
        }
        header.fields.push(stripped);
    }
    header
}

impl TerrainSpec {
    /// Writes the document: [`MAGIC`], [`VERSION`], a flag word, a RON header, then
    /// one zstd-compressed block per raster in a fixed order — painted rasters,
    /// then bakes, then the water's four grids.
    ///
    /// The header is text and holds the field metadata and the layer stacks with
    /// their rasters emptied out, so it stays small on a document whose paint is
    /// megabytes. The flag word, not `options`, records what actually got written:
    /// asking for bakes the document has not taken, or water at the wrong size,
    /// silently writes the file without them rather than failing.
    ///
    /// Nothing about the document is changed by saving it.
    pub fn save(&self, writer: &mut impl Write, options: SaveOptions) -> Result<(), IoError> {
        let header = header_document(self, options.layers);
        let text = ron::ser::to_string_pretty(&header, ron::ser::PrettyConfig::default())
            .map_err(|error| IoError::Header(error.to_string()))?;
        let bytes = text.as_bytes();

        let bakes = options.bakes
            && self
                .fields
                .iter()
                .all(|field| field.baked().size() == field.resolution(self.size));
        let water = self
            .water
            .as_ref()
            .filter(|state| options.water && state.size() == self.size);

        let mut flags = 0u16;
        if bakes {
            flags |= FLAG_BAKES;
        }
        if water.is_some() {
            flags |= FLAG_WATER;
        }

        writer.write_all(&MAGIC)?;
        write_u16(writer, VERSION)?;
        write_u16(writer, flags)?;
        write_u32(writer, bytes.len() as u32)?;
        writer.write_all(bytes)?;

        if options.layers {
            for slot in painted_slots(self) {
                match slot {
                    Slot::Op(field, layer) => match &self.fields[field].layers[layer].op {
                        LayerOp::Paint(raster) | LayerOp::External(raster) => {
                            write_raster(writer, raster)?
                        }
                        _ => unreachable!(),
                    },
                    Slot::MaskPainted(field, layer) => {
                        match &self.fields[field].layers[layer].mask {
                            Mask::Painted(raster) => write_raster(writer, raster)?,
                            _ => unreachable!(),
                        }
                    }
                }
            }
        }

        if bakes {
            for field in &self.fields {
                write_raster(writer, field.baked())?;
            }
        }

        if let Some(state) = water {
            write_u32(writer, state.lakes())?;
            write_raster(writer, state.depth())?;
            write_raster(writer, state.flow_dir())?;
            write_raster(writer, state.flow_accum())?;
            write_raster(writer, state.lake_id())?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Reads a document, re-deriving whatever the file left out.
    ///
    /// A file with no bakes is baked on the way in; one with no water is re-solved
    /// from the spec the header carries, and if it carries none the document simply
    /// has no water. What the file holds is read off its own flag word, never off
    /// [`SaveOptions`], so a file written by any of them is read on its own terms —
    /// and a carried bake is installed as it stands rather than being recomputed,
    /// even where a re-bake would give something else.
    ///
    /// Fails on the first thing it cannot read and returns no document.
    pub fn load(reader: &mut impl Read) -> Result<Self, IoError> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(IoError::BadMagic(magic));
        }

        let version = read_u16(reader)?;
        if version != VERSION {
            return Err(IoError::UnsupportedVersion(version));
        }

        let flags = read_u16(reader)?;
        let header_len = read_u32(reader)?;
        if header_len > MAX_HEADER_BYTES {
            return Err(IoError::HeaderTooLarge(header_len));
        }

        let mut bytes = vec![0u8; header_len as usize];
        reader.read_exact(&mut bytes)?;
        let text =
            std::str::from_utf8(&bytes).map_err(|error| IoError::Header(error.to_string()))?;
        let mut terrain: TerrainSpec =
            ron::from_str(text).map_err(|error| IoError::Header(error.to_string()))?;

        for slot in painted_slots(&terrain) {
            match slot {
                Slot::Op(field, layer) => {
                    let raster = read_raster(reader)?;
                    match &mut terrain.fields[field].layers[layer].op {
                        LayerOp::Paint(target) | LayerOp::External(target) => *target = raster,
                        _ => unreachable!(),
                    }
                }
                Slot::MaskPainted(field, layer) => {
                    let raster = read_raster(reader)?;
                    match &mut terrain.fields[field].layers[layer].mask {
                        Mask::Painted(target) => *target = raster,
                        _ => unreachable!(),
                    }
                }
            }
        }

        if flags & FLAG_BAKES != 0 {
            for index in 0..terrain.fields.len() {
                let raster = read_raster(reader)?;
                let expected = terrain.fields[index].resolution(terrain.size);
                if raster.size() != expected {
                    return Err(IoError::BlockShape {
                        kind: "bake",
                        width: raster.width(),
                        height: raster.height(),
                        expected,
                    });
                }
                *terrain.fields[index].baked_mut() = raster;
            }
        } else {
            terrain.bake_in_place()?;
        }

        if flags & FLAG_WATER != 0 {
            let lakes = read_u32(reader)?;
            let depth = read_raster(reader)?;
            let flow_dir = read_raster(reader)?;
            let flow_accum = read_raster(reader)?;
            let lake_id = read_raster(reader)?;
            if depth.size() != terrain.size {
                return Err(IoError::BlockShape {
                    kind: "water",
                    width: depth.width(),
                    height: depth.height(),
                    expected: terrain.size,
                });
            }
            terrain.water = Some(WaterState::from_parts(
                depth, flow_dir, flow_accum, lake_id, lakes,
            ));
        } else if let Some(spec) = terrain.water_spec.clone() {
            terrain.solve_water(&spec)?;
        }

        Ok(terrain)
    }

    /// [`TerrainSpec::save`] to a file, creating or truncating it. A failure part
    /// way through leaves the file partly written.
    pub fn save_to_path(
        &self,
        path: impl AsRef<Path>,
        options: SaveOptions,
    ) -> Result<(), IoError> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        self.save(&mut writer, options)
    }

    /// [`TerrainSpec::load`] from a file.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        Self::load(&mut reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::FieldId;
    use crate::layer::{Blend, Layer, Remap};
    use crate::noise::{NoiseKind, NoiseSpec};
    use crate::water::WaterSpec;

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
                    .with_layer(Layer::new(LayerOp::Paint(ramp())).with_blend(Blend::Replace))
                    .with_layer(noise_layer(9)),
            );
        terrain.bake_in_place().unwrap();
        terrain
    }

    fn round_trip(terrain: &TerrainSpec, options: SaveOptions) -> TerrainSpec {
        let mut bytes = Vec::new();
        terrain.save(&mut bytes, options).unwrap();
        TerrainSpec::load(&mut &bytes[..]).unwrap()
    }

    // Asking for bakes a document has not taken is an ordinary thing for a caller to
    // do, so it has to demote the file to a header rather than refuse the save or
    // write empty blocks the loader would then read as real ones.
    #[test]
    fn asking_to_save_a_bake_the_document_never_took_still_loads() {
        let terrain = painted_document();
        assert!(terrain.fields.iter().all(|field| field.baked().is_empty()));

        let loaded = round_trip(&terrain, SaveOptions::full());

        for field in &loaded.fields {
            assert_eq!(field.baked().size(), field.resolution(loaded.size));
        }
    }

    fn same_bits(left: &Raster<f32>, right: &Raster<f32>) -> bool {
        left.size() == right.size()
            && left
                .data()
                .iter()
                .zip(right.data())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    fn paint_of(terrain: &TerrainSpec, field: usize, layer: usize) -> &Raster<f32> {
        match &terrain.fields[field].layers[layer].op {
            LayerOp::Paint(raster) | LayerOp::External(raster) => raster,
            other => panic!("layer {layer} is {other:?}"),
        }
    }

    fn mask_of(terrain: &TerrainSpec, field: usize, layer: usize) -> &Raster<u8> {
        match &terrain.fields[field].layers[layer].mask {
            Mask::Painted(raster) => raster,
            other => panic!("layer {layer} mask is {other:?}"),
        }
    }

    // Painted data is the only thing in a file that cannot be re-derived, so it is
    // compared bit for bit rather than approximately. The fixture deliberately holds
    // a NaN, a negative zero, both infinities and a subnormal: those are exactly the
    // values a float encoding that goes through anything but its raw bits loses, and
    // an ordinary ramp would round-trip through almost any encoding.
    #[test]
    fn a_painted_document_round_trips_bit_identically() {
        let terrain = painted_document();
        let loaded = round_trip(&terrain, SaveOptions::document());

        assert_eq!(loaded.size, terrain.size);
        assert_eq!(loaded.fields.len(), terrain.fields.len());
        for (left, right) in loaded.fields.iter().zip(&terrain.fields) {
            assert_eq!(left.id, right.id);
            assert_eq!(left.shift, right.shift);
            assert_eq!(left.range, right.range);
            assert_eq!(left.layers.len(), right.layers.len());
        }

        assert!(same_bits(paint_of(&loaded, 1, 0), paint_of(&terrain, 1, 0)));
        assert!(same_bits(paint_of(&loaded, 1, 1), paint_of(&terrain, 1, 1)));
        assert_eq!(mask_of(&loaded, 1, 1), mask_of(&terrain, 1, 1));
    }

    // The premise of leaving bakes out of the default file: re-deriving has to give
    // the same answer as carrying, or the two save options would produce different
    // documents rather than the same one at different prices.
    #[test]
    fn a_header_only_file_loads_to_the_same_bake_as_one_that_carried_it() {
        let terrain = baked_document();

        let carried = round_trip(&terrain, SaveOptions::full());
        let header_only = round_trip(&terrain, SaveOptions::document());

        for (left, right) in header_only.fields.iter().zip(&carried.fields) {
            assert_eq!(left.id, right.id);
            assert!(
                same_bits(left.baked(), right.baked()),
                "field `{}` differs between a re-evaluated bake and a carried one",
                left.id
            );
        }

        for (loaded, original) in carried.fields.iter().zip(&terrain.fields) {
            assert!(same_bits(loaded.baked(), original.baked()));
        }
    }

    // The converse, and the reason `full` exists: a carried block is installed as it
    // stands. Scribbling on the bake first is the only way to tell "loaded the
    // block" from "re-baked and got the same thing".
    #[test]
    fn a_carried_bake_is_not_re_evaluated_on_load() {
        let mut terrain = baked_document();
        let scribble = {
            let baked = terrain.fields[1].baked_mut();
            baked.data_mut()[11] = 12.5;
            baked.clone()
        };

        let carried = round_trip(&terrain, SaveOptions::full());
        assert!(same_bits(carried.fields[1].baked(), &scribble));
        assert_eq!(carried.fields[1].baked().data()[11], 12.5);
    }

    // The water spec is carried even when the solved state is not, which is what
    // makes a layers-only file a complete document; without it the water would
    // vanish on the first save that left it out.
    #[test]
    fn a_document_re_solves_its_water_when_the_file_carries_none() {
        let mut terrain = baked_document();
        terrain.solve_water(&WaterSpec::new("height")).unwrap();
        let solved = terrain.water().unwrap().clone();

        let header_only = round_trip(&terrain, SaveOptions::layers_only());
        let carried = round_trip(&terrain, SaveOptions::full());

        assert_eq!(header_only.water_spec, Some(WaterSpec::new("height")));
        assert_eq!(carried.water().unwrap().lakes(), solved.lakes());
        assert!(same_bits(
            header_only.water().unwrap().depth(),
            solved.depth()
        ));
        assert!(same_bits(carried.water().unwrap().depth(), solved.depth()));
        assert_eq!(carried.water().unwrap().flow_accum(), solved.flow_accum());
        assert_eq!(carried.water().unwrap().lake_id(), solved.lake_id());
    }

    // A carried block and a re-solve agree on every ordinary document, so neither
    // can be told from the other there. Swapping the spec for one that would solve to
    // a different answer is what separates them: the carried file keeps the old
    // state, the layers-only file solves the new spec.
    #[test]
    fn a_document_carries_its_water_rather_than_re_solving_it() {
        let mut terrain = baked_document();
        terrain.solve_water(&WaterSpec::new("height")).unwrap();
        let solved = terrain.water().unwrap().clone();
        assert!(solved.lakes() > 0, "the fixture has to pond somewhere");

        terrain.water_spec = Some(WaterSpec::new("height").with_lake_min_cells(u32::MAX));

        let loaded = round_trip(&terrain, SaveOptions::document());
        let re_solved = round_trip(&terrain, SaveOptions::layers_only());

        assert!(loaded.fields.iter().all(|field| !field.layers.is_empty()));
        assert_eq!(loaded.water().unwrap().lakes(), solved.lakes());
        assert!(same_bits(loaded.water().unwrap().depth(), solved.depth()));
        assert_ne!(
            re_solved.water().unwrap().lakes(),
            solved.lakes(),
            "the swapped spec has to change the solve, or this proves nothing"
        );
    }

    // No spec has to mean no water rather than a default solve, or every document
    // that never wanted water would acquire some on its first load.
    #[test]
    fn a_document_with_no_water_spec_loads_with_no_water() {
        let terrain = baked_document();
        let loaded = round_trip(&terrain, SaveOptions::document());
        assert!(loaded.water().is_none());
    }

    // `clear_water` drops the spec as well as the state, and this is where that
    // matters: dropping only the state would let the next load put the water back.
    #[test]
    fn clearing_the_water_stops_a_load_from_re_solving_it() {
        let mut terrain = baked_document();
        terrain.solve_water(&WaterSpec::new("height")).unwrap();
        terrain.clear_water();

        let loaded = round_trip(&terrain, SaveOptions::document());
        assert!(loaded.water_spec.is_none());
        assert!(loaded.water().is_none());
    }

    // Pins what a consumer-facing export contains and what it deliberately does not:
    // the values are readable and the recipe is gone, so the file cannot be edited
    // back into a document.
    #[test]
    fn a_bake_only_export_carries_the_fields_and_the_water_but_no_layers() {
        let mut terrain = baked_document();
        terrain.solve_water(&WaterSpec::new("height")).unwrap();

        let exported = round_trip(&terrain, SaveOptions::bakes_only());

        assert!(exported.fields.iter().all(|field| field.layers.is_empty()));
        assert!(exported.water().is_some());
        for (loaded, original) in exported.fields.iter().zip(&terrain.fields) {
            assert_eq!(loaded.id, original.id);
            assert_eq!(loaded.shift, original.shift);
            assert!(same_bits(loaded.baked(), original.baked()));
        }
    }

    // The header is text so a document can be inspected and hand-edited without this
    // crate. The negative assertion is the load-bearing half: painted texels must
    // stay in their compressed blocks, or a 4096-square document's header would be
    // hundreds of megabytes of RON.
    #[test]
    fn the_header_is_readable_ron_naming_the_fields() {
        let terrain = painted_document();
        let mut bytes = Vec::new();
        terrain.save(&mut bytes, SaveOptions::document()).unwrap();

        let header_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let header = std::str::from_utf8(&bytes[12..12 + header_len]).unwrap();

        assert!(header.contains("moisture"), "{header}");
        assert!(header.contains("height"), "{header}");
        assert!(
            !header.contains("0.058823"),
            "painted data reached the header"
        );
    }

    // Not a pass/fail test: it prints size and load time for each save option on a
    // full-size document, which is where the figures on `SaveOptions::document` come
    // from and how the choice of what to carry is argued. Ignored because it is a
    // measurement.
    // Run with `cargo test --release -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn the_default_format_measures_what_a_document_costs() {
        let side = 4096;
        let size = UVec2::new(side, side);
        let cells = (side * side) as usize;

        let painted: Vec<f32> = (0..cells)
            .map(|index| {
                let x = (index % side as usize) as f32;
                let y = (index / side as usize) as f32;
                ((x * 0.01).sin() * (y * 0.013).cos()) * 0.5 + 0.5
            })
            .collect();

        let mut terrain = TerrainSpec::new(size)
            .with_field(
                Field::new("moisture")
                    .with_shift(4)
                    .with_layer(noise_layer(3)),
            )
            .with_field(
                Field::new("height").with_layer(noise_layer(9)).with_layer(
                    Layer::new(LayerOp::Paint(Raster::from_vec(size, painted).unwrap()))
                        .with_blend(Blend::Add)
                        .with_amplitude(0.25),
                ),
            );

        let started = std::time::Instant::now();
        terrain.bake_in_place().unwrap();
        let bake = started.elapsed();

        terrain.solve_water(&WaterSpec::new("height")).unwrap();

        let report = |label: &str, options: SaveOptions| {
            let started = std::time::Instant::now();
            let mut bytes = Vec::new();
            terrain.save(&mut bytes, options).unwrap();
            let wrote = started.elapsed();

            let started = std::time::Instant::now();
            let loaded = TerrainSpec::load(&mut &bytes[..]).unwrap();
            let read = started.elapsed();

            println!(
                "{label:11} {:>9.2} MB  save {:>7.0?}  load {:>7.0?}  water {}",
                bytes.len() as f64 / (1024.0 * 1024.0),
                wrote,
                read,
                loaded.water().is_some(),
            );
        };

        println!("{side}x{side}, two fields, full bake {bake:.0?}");
        report("layers-only", SaveOptions::layers_only());
        report("document", SaveOptions::document());
        report("full", SaveOptions::full());
        report("bakes-only", SaveOptions::bakes_only());
    }

    // The magic is checked before the length fields it precedes, so an unrelated file
    // is rejected rather than read as a header of whatever size its bytes happen to
    // spell.
    #[test]
    fn a_file_that_is_not_a_watershed_file_is_refused() {
        let bytes = b"NOPE\x01\x00\x00\x00\x00\x00\x00\x00".to_vec();
        assert!(matches!(
            TerrainSpec::load(&mut &bytes[..]),
            Err(IoError::BadMagic(_))
        ));
    }

    // There is no migration path, so a newer file has to be refused outright rather
    // than parsed as far as it happens to agree.
    #[test]
    fn a_file_from_a_later_version_is_refused() {
        let terrain = baked_document();
        let mut bytes = Vec::new();
        terrain.save(&mut bytes, SaveOptions::document()).unwrap();
        bytes[4] = 99;

        assert!(matches!(
            TerrainSpec::load(&mut &bytes[..]),
            Err(IoError::UnsupportedVersion(99))
        ));
    }

    // Half a file still has a valid magic, version and header; only the block reads
    // catch it. A short read that produced a document would hand back one silently
    // missing its paint.
    #[test]
    fn a_truncated_file_is_refused_rather_than_loaded_short() {
        let terrain = painted_document();
        let mut bytes = Vec::new();
        terrain.save(&mut bytes, SaveOptions::full()).unwrap();
        bytes.truncate(bytes.len() / 2);

        assert!(TerrainSpec::load(&mut &bytes[..]).is_err());
    }
}
