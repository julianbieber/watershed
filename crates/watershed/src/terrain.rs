//! The read side of a terrain: what a consuming project holds once a document has
//! been baked, and how it addresses and reads it.

use glam::{UVec2, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::channel::{ChannelMeta, ChannelTable, MAX_CHANNELS, TexelFormat};
use crate::chunk::ChunkGrid;
use crate::field::FieldRole;
use crate::meta::WaterInfo;
use crate::raster::CellRect;

/// The texels of one image: every channel of every texel, interleaved, in
/// row-major order, each channel [`TexelFormat::size`] bytes wide.
///
/// The buffer is exactly what an upload wants — a float image is little-endian
/// `f32`s with nothing between them — and its length is always a whole number of
/// texels at the image's size.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerTexels {
    size: UVec2,
    channels: usize,
    format: TexelFormat,
    bytes: Vec<u8>,
}

impl LayerTexels {
    /// Channels a texel of this image carries: one, two, three or four.
    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn format(&self) -> TexelFormat {
        self.format
    }

    /// Texel dimensions of the image.
    pub fn size(&self) -> UVec2 {
        self.size
    }

    /// Bytes one texel takes: every channel of it.
    pub fn texel_bytes(&self) -> usize {
        self.channels * self.format.size()
    }

    /// The whole buffer, channels interleaved, exactly as the image holds it.
    ///
    /// `texel_bytes() * width * height` bytes. This is what an upload or a
    /// re-encode wants; a point read should go through [`LayerView`] instead.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The bytes of one texel, or `None` outside the image.
    pub fn texel(&self, x: u32, y: u32) -> Option<&[u8]> {
        if x >= self.size.x || y >= self.size.y {
            return None;
        }
        let stride = self.texel_bytes();
        let at = (y as usize * self.size.x as usize + x as usize) * stride;
        self.bytes.get(at..at + stride)
    }

    /// Builds the texels for `channels` channels of `format` from an interleaved
    /// buffer.
    ///
    /// `None` if `channels` is not between one and four, or if the buffer is not
    /// exactly one texel's bytes per cell of `size` — which is what makes a decoded
    /// image's extent agree with the metadata that declared it.
    pub fn from_bytes(
        size: UVec2,
        channels: usize,
        format: TexelFormat,
        bytes: Vec<u8>,
    ) -> Option<Self> {
        if !(1..=MAX_CHANNELS).contains(&channels) {
            return None;
        }
        let cells = (size.x as usize).checked_mul(size.y as usize)?;
        if bytes.len() != cells.checked_mul(channels * format.size())? {
            return None;
        }
        Some(Self {
            size,
            channels,
            format,
            bytes,
        })
    }

    fn slice(&self, rect: CellRect) -> Option<Self> {
        if rect.max.x > self.size.x || rect.max.y > self.size.y {
            return None;
        }
        let stride = self.texel_bytes();
        let row = self.size.x as usize * stride;
        let mut bytes = Vec::with_capacity(rect.width() as usize * rect.height() as usize * stride);
        for y in rect.min.y..rect.max.y {
            let start = y as usize * row + rect.min.x as usize * stride;
            bytes.extend_from_slice(&self.bytes[start..start + rect.width() as usize * stride]);
        }
        Self::from_bytes(
            UVec2::new(rect.width(), rect.height()),
            self.channels,
            self.format,
            bytes,
        )
    }
}

/// One image of a terrain, with everything needed to read a number out of it.
#[derive(Clone, Debug)]
pub struct TerrainLayer {
    pub(crate) channels: Vec<ChannelMeta>,
    pub(crate) tables: Vec<Option<ChannelTable>>,
    pub(crate) texels: LayerTexels,
}

impl TerrainLayer {
    /// Builds a layer and the decode table each of its byte channels is read
    /// through.
    ///
    /// Every channel's [format](ChannelMeta::format) must be the texels' format;
    /// a channel that disagrees reads as nothing.
    pub fn new(channels: Vec<ChannelMeta>, texels: LayerTexels) -> Self {
        let tables = channels.iter().map(ChannelMeta::table).collect();
        Self {
            channels,
            tables,
            texels,
        }
    }

    /// The interleaved bytes, as the image holds them.
    pub fn texels(&self) -> &LayerTexels {
        &self.texels
    }

    /// How each channel of this image is read.
    pub fn channels(&self) -> &[ChannelMeta] {
        &self.channels
    }

    fn value(&self, x: u32, y: u32, channel: usize) -> Option<f32> {
        let meta = self.channels.get(channel)?;
        if meta.format() != self.texels.format {
            return None;
        }
        let texel = self.texels.texel(x, y)?;
        let at = channel * self.texels.format.size();
        match &self.tables[channel] {
            Some(table) => texel.get(at).map(|byte| table.get(*byte)),
            None => meta.decode(texel.get(at..)?),
        }
    }
}

/// Everything about a readable field except its values: what the bake settled once
/// so that reading a texel needs no further reference to the document.
///
/// Also the serialized form in `terrain.ron`, so the metadata and what a loaded
/// terrain holds cannot be two shapes that drift apart.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldInfo {
    /// The [`FieldId`](crate::field::FieldId) the field was declared under, and the
    /// key [`Terrain::field`] looks up.
    pub name: String,
    /// The role the field held in the spec.
    pub role: FieldRole,
    /// Whether values name a class rather than measure a quantity, which decides
    /// how [`FieldView::sample`] interpolates. Settled from the spec's layers at
    /// bake time, so a read does not have to re-derive it.
    pub categorical: bool,
    /// The image this field's values are in.
    pub layer: u8,
    /// The channel of that image holding them.
    pub channel: u8,
}

/// A baked terrain: an extent, a set of named fields, and optionally a solved water
/// state.
///
/// Nothing that produced it survives here — no layers, no noise specs, no plan — so
/// a consuming project cannot re-bake from a `Terrain` and does not have to carry
/// the machinery that would let it. What it may assume instead is that every field
/// [`Terrain::fields`] names is readable at every cell inside the extent.
///
/// Values are held as the images carry them, one texel per cell — a quantity as
/// its own `f32`, a class as a byte — and are decoded as they are read.
///
/// A terrain covers its own extent and every read is in its own cells: a terrain
/// loaded as one chunk of a directory reads that chunk's first cell at `(0, 0)`,
/// and a consumer turns a document position into a chunk and a texel with
/// [`ChunkGrid::locate`].
#[derive(Clone, Debug, Default)]
pub struct Terrain {
    pub(crate) size: UVec2,
    pub(crate) fields: Vec<FieldInfo>,
    pub(crate) layers: Vec<TerrainLayer>,
    pub(crate) water: Option<WaterInfo>,
}

impl Terrain {
    /// A terrain from the parts a writer already holds: the extent, the fields in
    /// the order they are to be read back, the images they sit in, and the water.
    ///
    /// Nothing is checked. Every [`FieldInfo`] must name a layer that is there and
    /// a channel that layer has, every layer must be at the extent, and a
    /// [`WaterInfo`] must name a four-channel layer — the same conditions
    /// [`TerrainDir::open`](crate::io::TerrainDir::open) refuses a directory for. A
    /// terrain built outside them reads wrong values or none.
    pub fn new(
        size: UVec2,
        fields: Vec<FieldInfo>,
        layers: Vec<TerrainLayer>,
        water: Option<WaterInfo>,
    ) -> Self {
        Self {
            size,
            fields,
            layers,
            water,
        }
    }

    /// Cells on the x axis.
    pub fn width(&self) -> u32 {
        self.size.x
    }

    /// Cells on the y axis.
    pub fn height(&self) -> u32 {
        self.size.y
    }

    /// The extent in cells. Every field covers all of it, one texel per cell.
    pub fn size(&self) -> UVec2 {
        self.size
    }

    /// Every readable field, in the order the spec declared them — not bake order
    /// and not hash order, so a project may index by position and get the same field
    /// back across runs.
    ///
    /// A declared field whose layer is missing is skipped, so the listing is exactly
    /// what can be read.
    pub fn fields(&self) -> impl Iterator<Item = FieldView<'_>> {
        self.fields.iter().filter_map(|info| self.view_of(info))
    }

    /// The field of that exact name, or `None` if the terrain does not carry one.
    /// Case-sensitive.
    pub fn field(&self, name: &str) -> Option<FieldView<'_>> {
        self.fields
            .iter()
            .find(|info| info.name == name)
            .and_then(|info| self.view_of(info))
    }

    /// The field holding `role`, for a project that wants the height of a document
    /// it did not author.
    ///
    /// At most one field can hold [`FieldRole::Height`] or [`FieldRole::Moisture`] —
    /// a document holding two is rejected before it is ever baked — so the answer is
    /// unambiguous. [`FieldRole::Custom`] is carried by any number of fields and
    /// always resolves to `None`.
    pub fn field_with_role(&self, role: FieldRole) -> Option<FieldView<'_>> {
        if role == FieldRole::Custom {
            return None;
        }
        self.fields
            .iter()
            .find(|info| info.role == role)
            .and_then(|info| self.view_of(info))
    }

    /// How many images the terrain holds.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// The image at that index, for a caller reading whole texels rather than one
    /// field at a time — the two components of a flow direction, say.
    pub fn layer(&self, index: usize) -> Option<LayerView<'_>> {
        self.layers.get(index).map(|layer| LayerView {
            layer,
            size: self.size,
        })
    }

    /// The solved water, present only if the spec declared some and the water step
    /// of the bake ran.
    pub fn water(&self) -> Option<WaterView<'_>> {
        let info = self.water?;
        let layer = self.layer(info.layer as usize)?;
        Some(WaterView { info, layer })
    }

    /// The part of this terrain one chunk of `grid` covers, as a terrain of its own
    /// over the chunk's extent, carrying the same fields, channel encodings and
    /// water.
    ///
    /// `None` for a chunk outside the grid, or a grid whose extent is not this
    /// terrain's.
    pub fn chunk(&self, grid: &ChunkGrid, chunk: UVec2) -> Option<Terrain> {
        if grid.extent() != self.size {
            return None;
        }
        let rect = grid.rect(chunk)?;
        let layers = self
            .layers
            .iter()
            .map(|layer| layer.slice(rect))
            .collect::<Option<Vec<_>>>()?;
        Some(Terrain {
            size: grid.chunk_size(),
            fields: self.fields.clone(),
            layers,
            water: self.water,
        })
    }

    fn view_of<'a>(&'a self, info: &'a FieldInfo) -> Option<FieldView<'a>> {
        let layer = self.layers.get(info.layer as usize)?;
        if info.channel as usize >= layer.channels.len() {
            return None;
        }
        Some(FieldView {
            info,
            layer,
            size: self.size,
        })
    }
}

impl TerrainLayer {
    fn slice(&self, rect: CellRect) -> Option<TerrainLayer> {
        Some(TerrainLayer {
            channels: self.channels.clone(),
            tables: self.tables.clone(),
            texels: self.texels.slice(rect)?,
        })
    }
}

/// A resolved handle to one image of a [`Terrain`].
///
/// `Copy` and small, so resolving once outside a loop and reading through it copies
/// no part of the grid.
#[derive(Clone, Copy, Debug)]
pub struct LayerView<'a> {
    layer: &'a TerrainLayer,
    size: UVec2,
}

impl<'a> LayerView<'a> {
    /// Channels a texel of this image carries.
    pub fn channels(&self) -> usize {
        self.layer.texels.channels()
    }

    /// The interleaved bytes, as the image holds them — for an upload or a
    /// re-encode, not for point reads.
    pub fn bytes(&self) -> &'a [u8] {
        self.layer.texels.bytes()
    }

    /// How wide each channel of a texel of [`LayerView::bytes`] is.
    pub fn format(&self) -> TexelFormat {
        self.layer.texels.format()
    }

    /// Every channel of a cell, decoded, written into `out` and returned as the
    /// part of it that was filled.
    ///
    /// Empty outside the terrain's extent.
    pub fn channels_at<'o>(&self, x: u32, y: u32, out: &'o mut [f32; 4]) -> &'o [f32] {
        if x >= self.size.x || y >= self.size.y {
            return &out[..0];
        }
        let count = self.channels().min(self.layer.channels.len());
        for (channel, slot) in out.iter_mut().enumerate().take(count) {
            let Some(value) = self.layer.value(x, y, channel) else {
                return &out[..0];
            };
            *slot = value;
        }
        &out[..count]
    }

    /// The single channel of a one-channel image at a cell, or `None` outside the
    /// extent or if the image has more than one channel.
    pub fn f32_at(&self, x: u32, y: u32) -> Option<f32> {
        self.vector_at::<1>(x, y).map(|values| values[0])
    }

    /// The two channels of a two-channel image at a cell.
    pub fn vec2_at(&self, x: u32, y: u32) -> Option<Vec2> {
        self.vector_at::<2>(x, y).map(Vec2::from_array)
    }

    /// The three channels of a three-channel image at a cell.
    pub fn vec3_at(&self, x: u32, y: u32) -> Option<Vec3> {
        self.vector_at::<3>(x, y).map(Vec3::from_array)
    }

    /// The four channels of a four-channel image at a cell.
    pub fn vec4_at(&self, x: u32, y: u32) -> Option<Vec4> {
        self.vector_at::<4>(x, y).map(Vec4::from_array)
    }

    fn vector_at<const N: usize>(&self, x: u32, y: u32) -> Option<[f32; N]> {
        if self.channels() != N {
            return None;
        }
        let mut buffer = [0.0f32; 4];
        let values = self.channels_at(x, y, &mut buffer);
        if values.len() != N {
            return None;
        }
        let mut out = [0.0f32; N];
        out.copy_from_slice(values);
        Some(out)
    }

    /// One channel of this image, read as a field would be.
    pub fn channel(&self, index: usize) -> Option<ChannelView<'a>> {
        if index >= self.layer.channels.len() {
            return None;
        }
        Some(ChannelView {
            layer: self.layer,
            channel: index,
            size: self.size,
            categorical: false,
        })
    }
}

/// One channel of one image, and the reads over it.
#[derive(Clone, Copy, Debug)]
pub struct ChannelView<'a> {
    layer: &'a TerrainLayer,
    channel: usize,
    size: UVec2,
    categorical: bool,
}

impl ChannelView<'_> {
    /// How this channel's bytes are read.
    pub fn meta(&self) -> ChannelMeta {
        self.layer.channels[self.channel]
    }

    /// The value at an integer *cell* of the terrain.
    ///
    /// `None` outside the extent — this is the read that refuses rather than
    /// clamping. Use [`ChannelView::sample`] for the clamping one.
    pub fn value_at(&self, x: u32, y: u32) -> Option<f32> {
        if x >= self.size.x || y >= self.size.y {
            return None;
        }
        self.layer.value(x, y, self.channel)
    }

    /// The value at a continuous position in cells, where a cell centre is at
    /// `x + 0.5`.
    ///
    /// Clamps to the extent instead of failing, so a position outside the terrain
    /// reads its nearest edge. Every texel is decoded before it is interpolated, so
    /// a channel whose encoding is not a straight line — a logarithmic one — reads
    /// the same way a linear one does.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let u = x - 0.5;
        let v = y - 0.5;
        if self.categorical {
            return self.nearest(u, v);
        }

        let size = self.layer.texels.size();
        let (x0, tx) = split(u, size.x);
        let (y0, ty) = split(v, size.y);
        let x1 = (x0 + 1).min(size.x.saturating_sub(1));
        let y1 = (y0 + 1).min(size.y.saturating_sub(1));

        let c00 = self.texel_value(x0, y0);
        let c10 = self.texel_value(x1, y0);
        let c01 = self.texel_value(x0, y1);
        let c11 = self.texel_value(x1, y1);
        let top = c00 + (c10 - c00) * tx;
        let bottom = c01 + (c11 - c01) * tx;
        top + (bottom - top) * ty
    }

    fn nearest(&self, u: f32, v: f32) -> f32 {
        let size = self.layer.texels.size();
        let x = clamp_index(u.round(), size.x);
        let y = clamp_index(v.round(), size.y);
        self.texel_value(x, y)
    }

    fn texel_value(&self, x: u32, y: u32) -> f32 {
        self.layer.value(x, y, self.channel).unwrap_or(0.0)
    }
}

fn clamp_index(value: f32, extent: u32) -> u32 {
    if !value.is_finite() || value < 0.0 {
        return 0;
    }
    (value as u32).min(extent.saturating_sub(1))
}

fn split(coord: f32, extent: u32) -> (u32, f32) {
    if !coord.is_finite() || coord <= 0.0 {
        return (0, 0.0);
    }
    let last = extent.saturating_sub(1);
    let floor = coord.floor();
    if floor >= last as f32 {
        return (last, 0.0);
    }
    (floor as u32, coord - floor)
}

/// A resolved handle to one field of a [`Terrain`], borrowing its metadata and the
/// channel its values sit in.
///
/// `Copy` and small — two borrows and the extent — so resolving once outside a loop
/// and reading through it copies no part of the grid. It borrows the terrain, which
/// therefore cannot be modified while any view of it is alive.
#[derive(Clone, Copy, Debug)]
pub struct FieldView<'a> {
    info: &'a FieldInfo,
    layer: &'a TerrainLayer,
    size: UVec2,
}

impl<'a> FieldView<'a> {
    /// The field's name in the document.
    pub fn name(&self) -> &'a str {
        &self.info.name
    }

    /// The role the field was declared with.
    pub fn role(&self) -> FieldRole {
        self.info.role
    }

    /// Whether [`FieldView::sample`] reads the nearest texel rather than
    /// interpolating. See [`FieldInfo::categorical`].
    pub fn is_categorical(&self) -> bool {
        self.info.categorical
    }

    /// Low end of the interval every value is inside.
    ///
    /// The extent of the values actually stored, which is tighter than the range
    /// the field declared. The declared range is part of the recipe, not of the
    /// values, and is not carried here.
    pub fn range_low(&self) -> f32 {
        self.channel().meta().low
    }

    /// High end of the interval every value is inside. See [`FieldView::range_low`].
    pub fn range_high(&self) -> f32 {
        self.channel().meta().high
    }

    /// Columns of the underlying image, which is the terrain's width.
    pub fn texel_width(&self) -> u32 {
        self.layer.texels.size().x
    }

    /// Rows of the underlying image, which is the terrain's height.
    pub fn texel_height(&self) -> u32 {
        self.layer.texels.size().y
    }

    /// The value at an integer *cell* of the terrain.
    ///
    /// `None` outside the extent — this is the read that refuses rather than
    /// clamping. Use [`FieldView::sample`] for the clamping one.
    pub fn value_at(&self, x: u32, y: u32) -> Option<f32> {
        self.channel().value_at(x, y)
    }

    /// The value at a continuous position in cells, where a cell centre is at
    /// `x + 0.5`.
    ///
    /// Clamps to the extent instead of failing, so a position outside the terrain
    /// reads its nearest edge. Interpolated between texels, or read to the nearest
    /// one for a [categorical](FieldView::is_categorical) field.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        self.channel().sample(x, y)
    }

    /// The bytes of the image this field sits in, channels interleaved.
    ///
    /// The field's own values start at [`FieldView::offset`] and repeat every
    /// [`FieldView::stride`] bytes, each [`FieldView::format`] wide. For an upload
    /// or a re-encode; a point read should go through [`FieldView::value_at`].
    pub fn bytes(&self) -> &'a [u8] {
        self.layer.texels.bytes()
    }

    /// Bytes between one texel of [`FieldView::bytes`] and the next.
    pub fn stride(&self) -> usize {
        self.layer.texels.texel_bytes()
    }

    /// Bytes from the start of a texel of [`FieldView::bytes`] to this field's
    /// channel.
    pub fn offset(&self) -> usize {
        self.info.channel as usize * self.layer.texels.format().size()
    }

    /// How wide each of this field's values is in [`FieldView::bytes`].
    pub fn format(&self) -> TexelFormat {
        self.layer.texels.format()
    }

    fn channel(&self) -> ChannelView<'a> {
        ChannelView {
            layer: self.layer,
            channel: self.info.channel as usize,
            size: self.size,
            categorical: self.info.categorical,
        }
    }
}

/// The solved water of a loaded terrain.
///
/// Everything is read from the one image the water occupies. What the solver's
/// own output has and this does not: lake ids, which are not carried, and the raw
/// direction codes, which became a vector.
#[derive(Clone, Copy, Debug)]
pub struct WaterView<'a> {
    info: WaterInfo,
    layer: LayerView<'a>,
}

impl WaterView<'_> {
    /// How many lakes the solve found.
    pub fn lakes(&self) -> u32 {
        self.info.lakes
    }

    /// How deep the water stands at a cell. `None` outside the extent.
    pub fn depth_at(&self, x: u32, y: u32) -> Option<f32> {
        self.layer.channel(WaterInfo::DEPTH)?.value_at(x, y)
    }

    /// Whether any water stands at a cell, at any depth. `false` outside the extent.
    pub fn is_water(&self, x: u32, y: u32) -> bool {
        self.depth_at(x, y).is_some_and(|depth| depth > 0.0)
    }

    /// How much water reaches a cell: its own weight plus everything draining
    /// through it. `0.0` outside the extent.
    pub fn accumulation(&self, x: u32, y: u32) -> f32 {
        self.layer
            .channel(WaterInfo::ACCUM)
            .and_then(|channel| channel.value_at(x, y))
            .unwrap_or(0.0)
    }

    /// Whether at least `threshold` reaches the cell — the test that turns an
    /// accumulation field into a river network. `false` outside the extent.
    pub fn channel_at(&self, x: u32, y: u32, threshold: f32) -> bool {
        self.accumulation(x, y) >= threshold
    }

    /// The way water leaves a cell, as a vector in cells.
    ///
    /// `None` outside the extent and where there is no outflow — a border cell or
    /// the bottom of a lake. A cell with no outflow is stored as the exact zero
    /// vector, which no flowing cell can produce.
    pub fn flow_at(&self, x: u32, y: u32) -> Option<Vec2> {
        let flow = Vec2::new(
            self.layer.channel(WaterInfo::FLOW_X)?.value_at(x, y)?,
            self.layer.channel(WaterInfo::FLOW_Y)?.value_at(x, y)?,
        );
        (flow != Vec2::ZERO).then_some(flow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer_of(size: UVec2, constants: &[f32]) -> TerrainLayer {
        let channels: Vec<ChannelMeta> = constants
            .iter()
            .map(|value| ChannelMeta::float(*value, *value))
            .collect();
        let mut bytes = Vec::new();
        for _ in 0..size.x * size.y {
            for (channel, value) in channels.iter().zip(constants) {
                channel.encode(*value, &mut bytes);
            }
        }
        let texels =
            LayerTexels::from_bytes(size, channels.len(), TexelFormat::F32, bytes).unwrap();
        TerrainLayer::new(channels, texels)
    }

    fn info(name: &str, role: FieldRole, layer: u8, channel: u8) -> FieldInfo {
        FieldInfo {
            name: name.to_owned(),
            role,
            categorical: false,
            layer,
            channel,
        }
    }

    fn baked() -> Terrain {
        let size = UVec2::new(64, 32);
        Terrain {
            size,
            fields: vec![
                info("height", FieldRole::Height, 0, 0),
                info("moisture", FieldRole::Moisture, 1, 0),
            ],
            layers: vec![layer_of(size, &[0.25]), layer_of(size, &[0.5])],
            water: None,
        }
    }

    fn shared_image(size: UVec2) -> Terrain {
        Terrain {
            size,
            fields: vec![
                info("a", FieldRole::Custom, 0, 0),
                info("b", FieldRole::Custom, 0, 1),
            ],
            layers: vec![layer_of(size, &[0.25, 0.75])],
            water: None,
        }
    }

    // The extent is what every cell read is bounds-checked against.
    #[test]
    fn a_terrain_answers_the_extent_it_was_written_with() {
        let terrain = baked();
        assert_eq!((terrain.width(), terrain.height()), (64, 32));
    }

    // The order is what a consuming project reads its fields back in, so it is part
    // of the contract rather than an artefact of how they were packed into images.
    #[test]
    fn fields_come_back_in_the_order_the_metadata_declared_them() {
        let terrain = baked();
        let names: Vec<_> = terrain
            .fields()
            .map(|view| view.name().to_owned())
            .collect();
        assert_eq!(names, vec!["height", "moisture"]);
    }

    // Role lookup is how a project finds the height of a document it did not author,
    // so it has to work off the declared role rather than off a conventional name.
    #[test]
    fn a_role_resolves_to_the_one_field_holding_it() {
        let terrain = baked();
        let height = terrain.field_with_role(FieldRole::Height).unwrap();
        assert_eq!(height.name(), "height");
        assert_eq!(
            terrain.field_with_role(FieldRole::Moisture).unwrap().name(),
            "moisture"
        );
    }

    // Custom is held by any number of fields, so it is the one role a lookup cannot
    // answer with a single view.
    #[test]
    fn the_custom_role_resolves_to_no_field() {
        assert!(baked().field_with_role(FieldRole::Custom).is_none());
    }

    // A chunk is read in its own cells, so slicing has to carry the texels of the
    // chunk's rectangle and not the document's first ones.
    #[test]
    fn a_chunk_of_a_terrain_reads_its_own_rectangle() {
        let size = UVec2::new(4, 2);
        let channels = vec![ChannelMeta::categorical()];
        let bytes: Vec<u8> = (0..8).collect();
        let texels = LayerTexels::from_bytes(size, 1, TexelFormat::U8, bytes).unwrap();
        let terrain = Terrain {
            size,
            fields: vec![info("v", FieldRole::Custom, 0, 0)],
            layers: vec![TerrainLayer::new(channels, texels)],
            water: None,
        };
        let grid = ChunkGrid::new(UVec2::new(2, 2), UVec2::new(2, 1)).unwrap();
        let right = terrain.chunk(&grid, UVec2::new(1, 0)).unwrap();
        assert_eq!(right.size(), UVec2::new(2, 2));
        assert_eq!(right.layer(0).unwrap().bytes(), &[2, 3, 6, 7]);
        assert!(terrain.chunk(&grid, UVec2::new(2, 0)).is_none());
    }

    // `value_at` refuses past the extent rather than clamping, which is the half of
    // the contract a consumer walking a chunk's edge relies on.
    #[test]
    fn a_cell_outside_the_extent_reads_nothing() {
        let terrain = baked();
        let height = terrain.field("height").unwrap();
        assert_eq!(height.value_at(63, 31), Some(0.25));
        assert_eq!(height.value_at(64, 0), None);
        assert_eq!(height.value_at(0, 32), None);
    }

    // A position read clamps where a cell read refuses — the one place the two
    // spellings deliberately differ.
    #[test]
    fn a_position_read_clamps_to_the_extent() {
        let terrain = baked();
        let height = terrain.field("height").unwrap();
        assert_eq!(height.sample(-40.0, -40.0), 0.25);
        assert_eq!(height.sample(4000.0, 4000.0), 0.25);
    }

    // Lookup is by exact name and there is no fallback, so a misspelling has to be a
    // `None` rather than a neighbouring field.
    #[test]
    fn a_name_the_terrain_does_not_carry_resolves_to_nothing() {
        assert!(baked().field("elevation").is_none());
    }

    // A view carries the metadata the write settled, so a project reading through one
    // never needs the document that produced the terrain.
    #[test]
    fn a_view_reports_the_role_its_field_declared() {
        let terrain = baked();
        let moisture = terrain.field("moisture").unwrap();
        assert_eq!(moisture.role(), FieldRole::Moisture);
    }

    // Fields are packed into shared images, so a field read has to pick
    // its own channel out of an interleaved texel rather than assume it is alone.
    #[test]
    fn two_fields_sharing_an_image_read_their_own_channel() {
        let terrain = shared_image(UVec2::new(16, 16));

        assert_eq!(terrain.layer_count(), 1);
        assert_eq!(terrain.layer(0).unwrap().channels(), 2);
        assert_eq!(terrain.field("a").unwrap().value_at(3, 3), Some(0.25));
        assert_eq!(terrain.field("b").unwrap().value_at(3, 3), Some(0.75));
        assert_eq!(terrain.field("a").unwrap().offset(), 0);
        assert_eq!(terrain.field("b").unwrap().offset(), 4);
        assert_eq!(terrain.field("a").unwrap().stride(), 8);
    }

    // The whole-texel read is what the brief asked for, and the arm has to follow the
    // channel count rather than a caller's guess at it.
    #[test]
    fn a_layer_reads_as_the_vector_its_channel_count_implies() {
        let terrain = shared_image(UVec2::new(8, 8));

        let layer = terrain.layer(0).unwrap();
        assert_eq!(layer.vec2_at(1, 1), Some(Vec2::new(0.25, 0.75)));
        assert!(layer.f32_at(1, 1).is_none());
        assert!(layer.vec3_at(1, 1).is_none());
        assert!(layer.vec2_at(8, 0).is_none());
    }

    // The raw bytes are what an upload wants, so their length has to be the texel
    // count times the channel count with nothing in between.
    #[test]
    fn a_layer_hands_back_its_bytes_interleaved() {
        let terrain = baked();
        let layer = terrain.layer(0).unwrap();
        let size = layer.bytes().len();
        assert_eq!(size, 64 * 32 * layer.channels() * 4);
    }

    #[test]
    fn a_float_layer_is_little_endian_f32_with_nothing_between_texels() {
        let terrain = baked();
        let height = terrain.field("height").unwrap();
        assert_eq!(height.format(), TexelFormat::F32);
        assert_eq!(
            &height.bytes()[..8],
            [0.25f32.to_le_bytes(), 0.25f32.to_le_bytes()].concat()
        );
    }

    #[test]
    fn a_channel_whose_format_is_not_the_images_reads_nothing() {
        let size = UVec2::new(2, 2);
        let texels = LayerTexels::from_bytes(size, 1, TexelFormat::U8, vec![1; 4]).unwrap();
        let terrain = Terrain {
            size,
            fields: vec![info("v", FieldRole::Custom, 0, 0)],
            layers: vec![TerrainLayer::new(
                vec![ChannelMeta::float(0.0, 1.0)],
                texels,
            )],
            water: None,
        };
        assert_eq!(terrain.field("v").unwrap().value_at(0, 0), None);
    }
}
