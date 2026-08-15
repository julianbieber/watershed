use std::io::{Read, Write};
use std::path::Path;

use glam::UVec2;
use thiserror::Error;

use crate::bake::{PlanError, TerrainSpec};
use crate::field::Field;
use crate::layer::{LayerOp, Mask};
use crate::raster::Raster;
use crate::water::{WaterError, WaterState};

/// TODO(jb-doc): what the magic identifies, and why a bake-only export carries the same
/// one rather than a second.
pub const MAGIC: [u8; 4] = *b"WSHD";

pub const VERSION: u16 = 1;

const FLAG_BAKES: u16 = 1 << 0;
const FLAG_WATER: u16 = 1 << 1;

/// TODO(jb-doc): what bounds this is defending, given the header length is read before
/// anything has been validated.
const MAX_HEADER_BYTES: u32 = 64 * 1024 * 1024;

/// TODO(jb-doc): why this level and not the default, measured against a painted document.
const COMPRESSION_LEVEL: i32 = 3;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a watershed file: magic is {0:?}")]
    BadMagic([u8; 4]),
    #[error("file version {0} is not supported; this build reads {VERSION}")]
    UnsupportedVersion(u16),
    #[error("header is {0} bytes, over the {MAX_HEADER_BYTES} byte limit")]
    HeaderTooLarge(u32),
    #[error("header is not readable: {0}")]
    Header(String),
    #[error("a {kind} block is {found} bytes where {expected} were expected")]
    BlockSize {
        kind: &'static str,
        expected: usize,
        found: usize,
    },
    #[error("a {kind} block is {width} by {height}, which is not the document's {expected}")]
    BlockShape {
        kind: &'static str,
        width: u32,
        height: u32,
        expected: UVec2,
    },
    #[error("bake: {0}")]
    Bake(#[from] PlanError),
    #[error("water: {0}")]
    Water(#[from] WaterError),
}

/// TODO(jb-doc): the three things a caller chooses when writing, and which combination is
/// the plain document, which the bake-only export.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaveOptions {
    pub layers: bool,
    pub bakes: bool,
    pub water: bool,
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self::document()
    }
}

impl SaveOptions {
    /// TODO(jb-doc): why the plain document carries the water but not the bakes — the
    /// measured cost of re-deriving each on load, and which of the two that leaves a
    /// reader paying. Figures in `the_default_format_measures_what_a_document_costs`.
    pub fn document() -> Self {
        Self {
            layers: true,
            bakes: false,
            water: true,
        }
    }

    /// TODO(jb-doc): what this is for, given no caller ships it — that it is the shape a
    /// document had before the water was carried, and what it therefore costs to load.
    pub fn layers_only() -> Self {
        Self {
            layers: true,
            bakes: false,
            water: false,
        }
    }

    pub fn full() -> Self {
        Self {
            layers: true,
            bakes: true,
            water: true,
        }
    }

    /// TODO(jb-doc): what a consumer of this gets and what it deliberately cannot do with
    /// it.
    pub fn bakes_only() -> Self {
        Self {
            layers: false,
            bakes: true,
            water: true,
        }
    }
}

/// TODO(jb-comment): why a slot is a coordinate into the document rather than a borrow,
/// and what that buys the writer and the reader together.
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

/// TODO(jb-doc): what a texel is written as, and why the encoding is little-endian
/// regardless of the host.
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

    // TODO(jb-comment): why a compressed block is bounded by its own uncompressed length,
    // and what a file claiming otherwise would cost before the first byte is read.
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

/// TODO(jb-comment): why the header is built by hand rather than by cloning the document,
/// and what a clone would have cost at 4096 squared.
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
    /// TODO(jb-doc): the shape of what this writes — magic, version, flags, a RON header,
    /// then one compressed block per raster in a fixed order.
    pub fn save(&self, writer: &mut impl Write, options: SaveOptions) -> Result<(), IoError> {
        let header = header_document(self, options.layers);
        let text = ron::ser::to_string_pretty(&header, ron::ser::PrettyConfig::default())
            .map_err(|error| IoError::Header(error.to_string()))?;
        let bytes = text.as_bytes();

        // TODO(jb-comment): why an unbaked field demotes the whole file to a header rather
        // than writing an empty block or refusing the save.
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

    /// TODO(jb-doc): what "re-evaluate whatever is absent" means precisely — a missing
    /// bake is baked, a missing water is re-solved from the spec the document carries,
    /// and a missing spec means the document has no water. Note the flags decide this,
    /// not [`SaveOptions`], so a file written by any writer is read on its own terms.
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

    pub fn save_to_path(
        &self,
        path: impl AsRef<Path>,
        options: SaveOptions,
    ) -> Result<(), IoError> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        self.save(&mut writer, options)
    }

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

    /// TODO(jb-comment): why the awkward floats are in the painted raster specifically,
    /// and what a round trip that only carried ordinary ones would fail to prove.
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

    /// TODO(jb-comment): why the spec is swapped out from under the solved state rather
    /// than the water being scribbled on — that a re-solve and a carried block agree on
    /// every ordinary document, so only a spec the state *disagrees* with can tell them
    /// apart.
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

    #[test]
    fn a_document_with_no_water_spec_loads_with_no_water() {
        let terrain = baked_document();
        let loaded = round_trip(&terrain, SaveOptions::document());
        assert!(loaded.water().is_none());
    }

    #[test]
    fn clearing_the_water_stops_a_load_from_re_solving_it() {
        let mut terrain = baked_document();
        terrain.solve_water(&WaterSpec::new("height")).unwrap();
        terrain.clear_water();

        let loaded = round_trip(&terrain, SaveOptions::document());
        assert!(loaded.water_spec.is_none());
        assert!(loaded.water().is_none());
    }

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

    /// TODO(jb-doc): what these figures are for — the doc comments on [`SaveOptions`] and
    /// [`COMPRESSION_LEVEL`] carry them, and this is how they were taken.
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

    #[test]
    fn a_file_that_is_not_a_watershed_file_is_refused() {
        let bytes = b"NOPE\x01\x00\x00\x00\x00\x00\x00\x00".to_vec();
        assert!(matches!(
            TerrainSpec::load(&mut &bytes[..]),
            Err(IoError::BadMagic(_))
        ));
    }

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

    #[test]
    fn a_truncated_file_is_refused_rather_than_loaded_short() {
        let terrain = painted_document();
        let mut bytes = Vec::new();
        terrain.save(&mut bytes, SaveOptions::full()).unwrap();
        bytes.truncate(bytes.len() / 2);

        assert!(TerrainSpec::load(&mut &bytes[..]).is_err());
    }
}
