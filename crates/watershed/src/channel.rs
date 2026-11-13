//! Turning a field's floats into the bytes an image carries, and back.
//!
//! A channel either holds its value itself, as an `f32`, or one of 256 levels a byte
//! can name. Both halves of that trade live here — what a bake writes and how a
//! read turns it back into a number — because a writer and a reader that disagree
//! about it produce wrong numbers rather than an error. The rule deciding which
//! fields share an image lives here too, for the same reason.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Channels one image may carry, and so fields that may share one.
///
/// Four because that is the widest ordinary colour type, and because it lets the
/// whole of a solved water — depth, the two components of its flow, and
/// accumulation — sit in a single image.
pub const MAX_CHANNELS: usize = 4;

/// The largest value a byte holds, and so the largest class index a
/// [`ChannelEncoding::Categorical`] channel can carry.
pub const MAX_CLASS: f32 = 255.0;

/// Why a channel could not be encoded or read.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ChannelError {
    /// A range endpoint is a NaN or an infinity. Reachable from a hand-edited
    /// file, and from a field declared with one.
    #[error("channel range ({low}, {high}) is not finite")]
    RangeNotFinite {
        /// Low end as it was read.
        low: f32,
        /// High end as it was read.
        high: f32,
    },
    /// The endpoints are the wrong way round. A field is
    /// allowed to declare its range in either order and is sorted on the way in,
    /// so this can only come from a file.
    #[error("channel range ({low}, {high}) is not ordered")]
    RangeUnordered {
        /// Low end as it was read.
        low: f32,
        /// High end as it was read.
        high: f32,
    },
    /// A categorical field carried a value that is not a class index: not a whole
    /// number, negative, or past [`MAX_CLASS`]. Refused when the terrain is
    /// written rather than silently rounded, because a smeared class is a wrong
    /// answer a reader cannot detect.
    #[error("`{field}` is categorical but holds {value}, which is not a class index")]
    StrayClass {
        /// The field the value was found in.
        field: String,
        /// The offending value.
        value: f32,
    },
}

/// How wide one channel of a texel is, which every channel of an image shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TexelFormat {
    /// One byte per channel, read through a [`ChannelTable`].
    U8,
    /// One little-endian `f32` per channel: the value itself.
    F32,
}

impl TexelFormat {
    /// Bytes one channel of a texel takes.
    pub fn size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::F32 => 4,
        }
    }
}

/// How a channel's bytes turn back into numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelEncoding {
    /// The value itself, as a little-endian `f32`, so a read returns exactly what
    /// the bake produced.
    Float,
    /// The class index itself, stored in a byte and returned unchanged, so a class
    /// never lands between two of its neighbours.
    Categorical,
}

impl ChannelEncoding {
    pub fn format(self) -> TexelFormat {
        match self {
            Self::Float => TexelFormat::F32,
            Self::Categorical => TexelFormat::U8,
        }
    }
}

/// A channel's interval and how the bytes in it are read.
///
/// For a [`ChannelEncoding::Float`] channel the interval is the extent of the
/// values it holds, which is not the same number as the range a field declares:
/// a field's range is the clamp its bake applies, and this is what the values then
/// actually span.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelMeta {
    /// Low end of the interval. A value that cannot be stored — a NaN or an
    /// infinity — reads back as this.
    pub low: f32,
    /// High end of the interval.
    pub high: f32,
    /// How the bytes in this channel are read.
    pub encoding: ChannelEncoding,
}

impl ChannelMeta {
    /// A channel holding values inside `(low, high)` as themselves. The pair is
    /// sorted, so a caller may pass it in either order.
    pub fn float(low: f32, high: f32) -> Self {
        let (low, high) = if low <= high {
            (low, high)
        } else {
            (high, low)
        };
        Self {
            low,
            high,
            encoding: ChannelEncoding::Float,
        }
    }

    /// A channel carrying class indices, which run from zero to [`MAX_CLASS`].
    pub fn categorical() -> Self {
        Self {
            low: 0.0,
            high: MAX_CLASS,
            encoding: ChannelEncoding::Categorical,
        }
    }

    pub fn format(&self) -> TexelFormat {
        self.encoding.format()
    }

    /// Whether this channel can be read at all.
    ///
    /// Checked when a terrain is loaded, because every field here comes out of a
    /// file that may have been written by hand.
    pub fn validate(&self) -> Result<(), ChannelError> {
        if !self.low.is_finite() || !self.high.is_finite() {
            return Err(ChannelError::RangeNotFinite {
                low: self.low,
                high: self.high,
            });
        }
        if self.low > self.high {
            return Err(ChannelError::RangeUnordered {
                low: self.low,
                high: self.high,
            });
        }
        Ok(())
    }

    /// Appends the bytes `value` is stored as: [`TexelFormat::size`] of them.
    ///
    /// Total: every `f32` encodes to something. A value that is not finite is
    /// stored as [`ChannelMeta::low`] — the format carries no NaN. A float channel
    /// stores every finite value exactly; a categorical one rounds to the nearest
    /// class and saturates at zero and [`MAX_CLASS`].
    pub fn encode(&self, value: f32, out: &mut Vec<u8>) {
        match self.encoding {
            ChannelEncoding::Float => {
                let stored = if value.is_finite() { value } else { self.low };
                out.extend_from_slice(&stored.to_le_bytes());
            }
            ChannelEncoding::Categorical => out.push(class_byte(value)),
        }
    }

    /// The value the first [`TexelFormat::size`] bytes of `bytes` stand for, or
    /// `None` if there are fewer. Inverse of [`ChannelMeta::encode`], exactly, for
    /// a finite value and for a class index.
    pub fn decode(&self, bytes: &[u8]) -> Option<f32> {
        match self.encoding {
            ChannelEncoding::Float => Some(f32::from_le_bytes(bytes.get(..4)?.try_into().ok()?)),
            ChannelEncoding::Categorical => bytes.first().map(|byte| *byte as f32),
        }
    }

    /// Every value a byte channel can return, indexed by the byte that returns it,
    /// so a read is a lookup rather than arithmetic. `None` for a float channel,
    /// which is read as itself.
    pub fn table(&self) -> Option<ChannelTable> {
        if self.format() != TexelFormat::U8 {
            return None;
        }
        let mut values = [0.0f32; 256];
        for (byte, slot) in values.iter_mut().enumerate() {
            *slot = byte as f32;
        }
        Some(ChannelTable(values))
    }
}

fn class_byte(value: f32) -> u8 {
    if !value.is_finite() {
        return 0;
    }
    value.round().clamp(0.0, MAX_CLASS) as u8
}

/// The 256 values one byte channel can return, in byte order.
#[derive(Clone, Copy, Debug)]
pub struct ChannelTable([f32; 256]);

impl ChannelTable {
    /// The value a byte stands for.
    pub fn get(&self, byte: u8) -> f32 {
        self.0[byte as usize]
    }
}

/// The first value that could not be stored as a class index, if there is one.
///
/// A field is categorical because its shader file declares it so, but what reaches a
/// texel has been through that shader's arithmetic and the field's clamp range, so a
/// class index can arrive scaled or part-way between two others. Storing that would
/// smear one class into its neighbour, so a terrain carrying it is refused when it is
/// written.
pub fn stray_class(values: &[f32]) -> Option<f32> {
    values.iter().copied().find(|value| {
        !value.is_finite() || value.fract() != 0.0 || !(0.0..=MAX_CLASS).contains(value)
    })
}

/// Where each field's values sit once the fields have been grouped into images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    /// Index of the image the field landed in.
    pub layer: u8,
    /// Index of the channel within that image.
    pub channel: u8,
}

/// Groups fields of the given formats into images: where each field lands, and
/// the format of every image, in image order.
///
/// An image holds fields of one format only, up to [`MAX_CHANNELS`] of them. Fields
/// are placed in declaration order, each into the newest image of its format that
/// has room, so the same document always packs the same way and a saved terrain
/// can be compared against another save of it.
pub fn plan_layers(fields: &[TexelFormat]) -> (Vec<Placement>, Vec<TexelFormat>) {
    let mut images: Vec<(TexelFormat, usize)> = Vec::new();
    let placements = fields
        .iter()
        .map(|format| {
            let open = images
                .iter()
                .rposition(|(kind, used)| kind == format && *used < MAX_CHANNELS);
            let layer = open.unwrap_or_else(|| {
                images.push((*format, 0));
                images.len() - 1
            });
            let channel = images[layer].1;
            images[layer].1 += 1;
            Placement {
                layer: layer as u8,
                channel: channel as u8,
            }
        })
        .collect();
    (
        placements,
        images.into_iter().map(|(kind, _)| kind).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(meta: &ChannelMeta, value: f32) -> f32 {
        let mut bytes = Vec::new();
        meta.encode(value, &mut bytes);
        assert_eq!(bytes.len(), meta.format().size());
        meta.decode(&bytes).unwrap()
    }

    #[test]
    fn a_float_channel_returns_every_finite_value_exactly() {
        let meta = ChannelMeta::float(-20.0, 8000.0);
        for value in [
            -20.0,
            0.0,
            0.1,
            1234.5678,
            7999.9,
            8000.0,
            -0.0,
            f32::MIN_POSITIVE / 3.0,
        ] {
            assert_eq!(
                round_trip(&meta, value).to_bits(),
                value.to_bits(),
                "{value}"
            );
        }
    }

    #[test]
    fn a_float_channel_resolves_a_tenth_of_a_metre_over_eight_kilometres() {
        let meta = ChannelMeta::float(0.0, 8000.0);
        let levels: std::collections::BTreeSet<u32> = (0..80_000)
            .map(|step| round_trip(&meta, step as f32 * 0.1).to_bits())
            .collect();
        assert_eq!(levels.len(), 80_000);
    }

    #[test]
    fn a_float_channel_is_little_endian() {
        let mut bytes = Vec::new();
        ChannelMeta::float(0.0, 2.0).encode(1.5, &mut bytes);
        assert_eq!(bytes, 1.5f32.to_le_bytes());
    }

    #[test]
    fn a_value_that_is_not_finite_is_stored_as_the_low_end() {
        let meta = ChannelMeta::float(-1.0, 1.0);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(round_trip(&meta, value), meta.low, "{value}");
        }
        let class = ChannelMeta::categorical();
        assert_eq!(round_trip(&class, f32::NAN), class.low);
    }

    #[test]
    fn a_float_channel_has_no_table_and_a_byte_channel_has_one() {
        assert!(ChannelMeta::float(0.0, 1.0).table().is_none());
        assert!(ChannelMeta::categorical().table().is_some());
    }

    #[test]
    fn a_short_buffer_decodes_to_nothing() {
        assert_eq!(ChannelMeta::float(0.0, 1.0).decode(&[0, 0, 0]), None);
        assert_eq!(ChannelMeta::categorical().decode(&[]), None);
    }

    #[test]
    fn a_class_index_comes_back_exactly() {
        let meta = ChannelMeta::categorical();
        for class in [0.0, 1.0, 7.0, 128.0, 255.0] {
            assert_eq!(round_trip(&meta, class), class);
        }
    }

    #[test]
    fn a_value_that_is_not_a_class_index_is_found() {
        assert_eq!(stray_class(&[0.0, 3.0, 255.0]), None);
        assert_eq!(stray_class(&[0.0, 3.4]), Some(3.4));
        assert_eq!(stray_class(&[-1.0]), Some(-1.0));
        assert_eq!(stray_class(&[256.0]), Some(256.0));
        assert_eq!(stray_class(&[f32::NAN]).map(f32::is_nan), Some(true));
    }

    #[test]
    fn the_table_agrees_with_decoding_a_byte() {
        let meta = ChannelMeta::categorical();
        let table = meta.table().unwrap();
        for byte in 0..=255u8 {
            assert_eq!(Some(table.get(byte)), meta.decode(&[byte]));
        }
    }

    #[test]
    fn an_unreadable_range_is_refused() {
        assert!(ChannelMeta::float(0.0, 1.0).validate().is_ok());
        let cases = [
            ChannelMeta {
                low: f32::NAN,
                high: 1.0,
                encoding: ChannelEncoding::Float,
            },
            ChannelMeta {
                low: 0.0,
                high: f32::INFINITY,
                encoding: ChannelEncoding::Float,
            },
            ChannelMeta {
                low: 1.0,
                high: 0.0,
                encoding: ChannelEncoding::Float,
            },
        ];
        for meta in cases {
            assert!(meta.validate().is_err(), "{meta:?}");
        }
    }

    #[test]
    fn fields_of_one_format_pack_four_to_an_image_in_declaration_order() {
        let (places, images) = plan_layers(&[TexelFormat::F32; 6]);
        assert_eq!(images, vec![TexelFormat::F32, TexelFormat::F32]);
        let layers: Vec<u8> = places.iter().map(|place| place.layer).collect();
        let channels: Vec<u8> = places.iter().map(|place| place.channel).collect();
        assert_eq!(layers, vec![0, 0, 0, 0, 1, 1]);
        assert_eq!(channels, vec![0, 1, 2, 3, 0, 1]);
    }

    #[test]
    fn fields_of_different_formats_never_share_an_image() {
        use TexelFormat::{F32, U8};
        let (places, images) = plan_layers(&[F32, U8, F32, U8]);
        assert_eq!(images, vec![F32, U8]);
        let placed: Vec<(u8, u8)> = places.iter().map(|p| (p.layer, p.channel)).collect();
        assert_eq!(placed, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
    }

    #[test]
    fn a_document_with_no_fields_plans_no_layers() {
        let (places, images) = plan_layers(&[]);
        assert!(places.is_empty());
        assert!(images.is_empty());
    }
}
