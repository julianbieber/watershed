//! Turning a field's floats into the bytes an image carries, and back.
//!
//! A channel is eight bits, so a stored value is one of 256 levels of a declared
//! interval. Both halves of that trade live here — the encoding a bake writes and
//! the table a read goes through — because a writer and a reader that disagree
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

/// The largest value a byte holds: the top of the scale every spread encoding
/// divides by, and so also the largest class index a
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
    /// A logarithmic channel needs an interval with room in it, and one that does
    /// not reach below zero.
    #[error("a log channel cannot span ({low}, {high})")]
    RangeNotLoggable {
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

/// How a channel's bytes turn back into numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelEncoding {
    /// The declared interval spread evenly over the byte.
    Linear,
    /// The class index itself, stored and returned unchanged, so a class never
    /// lands between two of its neighbours.
    Categorical,
    /// The interval spread over the byte through `log1p`, for a quantity whose
    /// interesting values crowd the bottom of its range.
    Log,
    /// One signed component of a direction, centred rather than spread: the
    /// midpoint byte is exactly zero and the ends are exactly minus one and one.
    ///
    /// That exactness is load-bearing. It makes a zero-length vector
    /// representable, which is what lets a cell with no outflow be told from one
    /// flowing somewhere without spending a channel on a flag.
    Unit,
    /// The byte itself. For data that was already bytes, which therefore comes
    /// back exactly rather than approximately.
    Raw,
}

/// A channel's interval and how the bytes in it are read.
///
/// The interval is the *quantisation* range, which is not the same number as the
/// range a field declares: a field's range is the clamp
/// its bake applies, and this is what the byte then spreads over. They coincide
/// for an ordinary field and deliberately do not for water.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelMeta {
    /// Low end of the interval. A value that cannot be represented — a NaN, an
    /// infinity, or anything at all in a degenerate interval — reads back as this.
    pub low: f32,
    /// High end of the interval.
    pub high: f32,
    /// How a byte in this channel is read.
    pub encoding: ChannelEncoding,
}

impl ChannelMeta {
    /// A channel spreading `(low, high)` evenly over the byte. The pair is sorted,
    /// so a caller may pass it in either order.
    pub fn linear(low: f32, high: f32) -> Self {
        let (low, high) = if low <= high {
            (low, high)
        } else {
            (high, low)
        };
        Self {
            low,
            high,
            encoding: ChannelEncoding::Linear,
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

    /// A channel carrying `0..=high` through `log1p`.
    pub fn log(high: f32) -> Self {
        Self {
            low: 0.0,
            high,
            encoding: ChannelEncoding::Log,
        }
    }

    /// A channel carrying one signed component of a direction.
    pub fn unit() -> Self {
        Self {
            low: -1.0,
            high: 1.0,
            encoding: ChannelEncoding::Unit,
        }
    }

    /// A channel carrying bytes unchanged.
    pub fn raw() -> Self {
        Self {
            low: 0.0,
            high: MAX_CLASS,
            encoding: ChannelEncoding::Raw,
        }
    }

    /// Whether the interval has no width, in which case every value in it encodes
    /// to zero and reads back as [`ChannelMeta::low`].
    ///
    /// Reachable rather than theoretical: a constant field, a document that ponds
    /// nowhere, and a solve whose accumulation is everywhere zero all produce one.
    pub fn is_degenerate(&self) -> bool {
        self.high <= self.low
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
        if self.encoding == ChannelEncoding::Log && (self.low < 0.0 || self.high <= self.low) {
            return Err(ChannelError::RangeNotLoggable {
                low: self.low,
                high: self.high,
            });
        }
        Ok(())
    }

    /// The byte a value is stored as.
    ///
    /// Total: every `f32` encodes to something. A value outside the interval
    /// saturates at the nearer end, and one that is not finite encodes to zero,
    /// which reads back as [`ChannelMeta::low`] — the format carries no NaN, and a
    /// caller that needs one cannot use this to move it.
    pub fn encode(&self, value: f32) -> u8 {
        if !value.is_finite() {
            return 0;
        }
        match self.encoding {
            ChannelEncoding::Raw | ChannelEncoding::Categorical => {
                value.round().clamp(0.0, MAX_CLASS) as u8
            }
            ChannelEncoding::Unit => {
                let scaled = value.clamp(-1.0, 1.0) * UNIT_SCALE + UNIT_ZERO;
                scaled.round().clamp(0.0, MAX_CLASS) as u8
            }
            ChannelEncoding::Linear => {
                if self.is_degenerate() {
                    return 0;
                }
                let t = (value - self.low) / (self.high - self.low);
                (t.clamp(0.0, 1.0) * MAX_CLASS).round() as u8
            }
            ChannelEncoding::Log => {
                if self.is_degenerate() {
                    return 0;
                }
                let span = (self.high - self.low).ln_1p();
                let t = (value - self.low).max(0.0).ln_1p() / span;
                (t.clamp(0.0, 1.0) * MAX_CLASS).round() as u8
            }
        }
    }

    /// The value a byte stands for. Inverse of [`ChannelMeta::encode`] up to the
    /// rounding, which costs at most half a step of whatever scale the encoding
    /// spreads over the byte — half of `(high - low) / 255` for a linear channel,
    /// and a fixed *proportion* of the value rather than a fixed amount for a
    /// logarithmic one.
    pub fn decode(&self, byte: u8) -> f32 {
        match self.encoding {
            ChannelEncoding::Raw | ChannelEncoding::Categorical => byte as f32,
            ChannelEncoding::Unit => ((byte as f32 - UNIT_ZERO) / UNIT_SCALE).clamp(-1.0, 1.0),
            ChannelEncoding::Linear => {
                if self.is_degenerate() {
                    return self.low;
                }
                self.low + (byte as f32 / MAX_CLASS) * (self.high - self.low)
            }
            ChannelEncoding::Log => {
                if self.is_degenerate() {
                    return self.low;
                }
                let span = (self.high - self.low).ln_1p();
                self.low + (byte as f32 / MAX_CLASS * span).exp_m1()
            }
        }
    }

    /// Every value this channel can return, indexed by the byte that returns it.
    ///
    /// A channel has 256 possible bytes, so a read is a lookup rather than
    /// arithmetic, and every encoding costs the same to read however expensive it
    /// was to write. Built when a terrain is loaded; it is not part of the file.
    pub fn table(&self) -> ChannelTable {
        let mut values = [0.0f32; 256];
        for (byte, slot) in values.iter_mut().enumerate() {
            *slot = self.decode(byte as u8);
        }
        ChannelTable(values)
    }
}

const UNIT_SCALE: f32 = 127.0;
const UNIT_ZERO: f32 = 128.0;

/// The 256 values one channel can return, in byte order.
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
/// A field is categorical because of the layers in it, but what reaches a texel has
/// been through an amplitude, a blend and a mask weight, so a class index can
/// arrive scaled or part-way between two others. Storing that would smear one class
/// into its neighbour, so a terrain carrying it is refused when it is written.
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

/// Groups fields into images: which image each one lands in, and each image's shift.
///
/// Fields sharing a shift may share an image, because an image has one extent and a
/// shift is what decides it. Buckets come in order of first appearance and are
/// filled in declaration order, so the same document always packs the same way and
/// a saved terrain can be compared against another save of it.
pub fn plan_layers(shifts: &[u8]) -> (Vec<Placement>, Vec<u8>) {
    let mut layer_shifts: Vec<u8> = Vec::new();
    let mut open: Vec<(u8, usize, u8)> = Vec::new();
    let mut placements = Vec::with_capacity(shifts.len());

    for &shift in shifts {
        let slot = open
            .iter_mut()
            .find(|(candidate, _, used)| *candidate == shift && (*used as usize) < MAX_CHANNELS);
        let (layer, channel) = match slot {
            Some((_, layer, used)) => {
                let channel = *used;
                *used += 1;
                (*layer, channel)
            }
            None => {
                let layer = layer_shifts.len();
                layer_shifts.push(shift);
                open.push((shift, layer, 1));
                (layer, 0)
            }
        };
        placements.push(Placement {
            layer: layer as u8,
            channel,
        });
    }

    (placements, layer_shifts)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The round-trip bound is the whole contract of an eight-bit channel, and it is
    // what replaces the old format's promise of bit-exact floats.
    #[test]
    fn a_finite_value_comes_back_within_half_a_step() {
        let meta = ChannelMeta::linear(-20.0, 1200.0);
        let step = (meta.high - meta.low) / MAX_CLASS;
        for i in 0..=1000 {
            let value = meta.low + (meta.high - meta.low) * (i as f32 / 1000.0);
            let found = meta.decode(meta.encode(value));
            assert!(
                (found - value).abs() <= step / 2.0 + 1e-3,
                "{value} came back as {found}"
            );
        }
    }

    // A range with no width is reachable from an ordinary document — a constant
    // field is the common one — so it has to be a defined answer rather than a
    // division by zero.
    #[test]
    fn a_degenerate_range_reads_back_as_its_own_endpoint() {
        let meta = ChannelMeta::linear(0.25, 0.25);
        assert!(meta.is_degenerate());
        assert_eq!(meta.encode(0.25), 0);
        assert_eq!(meta.decode(0), 0.25);
        assert_eq!(meta.decode(200), 0.25);
    }

    // A bake clamps into the field's range but `f32::clamp` passes NaN through, so
    // these reach the encoder from an ordinary painted document rather than only
    // from a hostile file.
    #[test]
    fn a_value_that_is_not_finite_encodes_to_the_low_end() {
        let meta = ChannelMeta::linear(-1.0, 1.0);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(meta.encode(value), 0, "{value}");
            assert_eq!(meta.decode(meta.encode(value)), meta.low);
        }
    }

    // The subnormal and the negative zero from the old format's fixture: they no
    // longer survive as bits, but they must still encode to something rather than
    // panicking or landing outside the range.
    #[test]
    fn awkward_finite_values_still_encode_inside_the_range() {
        let meta = ChannelMeta::linear(0.0, 1.0);
        for value in [-0.0f32, f32::MIN_POSITIVE / 3.0] {
            let found = meta.decode(meta.encode(value));
            assert!((0.0..=1.0).contains(&found), "{value} came back as {found}");
        }
    }

    // The reason `Categorical` exists: a class index is an identity, so it has to
    // come back as itself rather than within a tolerance of itself.
    #[test]
    fn a_class_index_comes_back_exactly() {
        let meta = ChannelMeta::categorical();
        for class in [0.0, 1.0, 7.0, 128.0, 255.0] {
            assert_eq!(meta.decode(meta.encode(class)), class);
        }
    }

    // Values that cannot be a class have to be caught when the terrain is written;
    // by read time the damage is a plausible wrong answer.
    #[test]
    fn a_value_that_is_not_a_class_index_is_found() {
        assert_eq!(stray_class(&[0.0, 3.0, 255.0]), None);
        assert_eq!(stray_class(&[0.0, 3.4]), Some(3.4));
        assert_eq!(stray_class(&[-1.0]), Some(-1.0));
        assert_eq!(stray_class(&[256.0]), Some(256.0));
        assert_eq!(stray_class(&[f32::NAN]).map(f32::is_nan), Some(true));
    }

    // This is what lets the flow texture drop its flag channel: a sink is the exact
    // zero vector, and a spread range has no byte that lands on zero.
    #[test]
    fn a_unit_channel_represents_zero_exactly() {
        let meta = ChannelMeta::unit();
        assert_eq!(meta.decode(meta.encode(0.0)), 0.0);
        assert_eq!(meta.decode(meta.encode(1.0)), 1.0);
        assert_eq!(meta.decode(meta.encode(-1.0)), -1.0);
        assert!(
            (0..=255).filter(|byte| meta.decode(*byte) == 0.0).count() == 1,
            "exactly one byte may mean no flow"
        );
    }

    // The eight compass directions are what actually gets stored, so the bound that
    // matters is the one over them rather than over the whole interval.
    #[test]
    fn the_diagonal_directions_survive_the_unit_channel() {
        let meta = ChannelMeta::unit();
        let diagonal = std::f32::consts::FRAC_1_SQRT_2;
        for value in [diagonal, -diagonal, 1.0, -1.0, 0.0] {
            let found = meta.decode(meta.encode(value));
            assert!(
                (found - value).abs() < 0.005,
                "{value} came back as {found}"
            );
        }
    }

    // A mask is already bytes, so it is the one thing in the new format that still
    // round-trips exactly — by encoding, not by arithmetic coincidence.
    #[test]
    fn a_raw_channel_round_trips_every_byte() {
        let meta = ChannelMeta::raw();
        for byte in 0..=255u8 {
            assert_eq!(meta.encode(meta.decode(byte)), byte);
        }
    }

    // Log exists because accumulation crowds the bottom of its range; the property
    // worth pinning is that small values stay distinguishable where a linear spread
    // would put them all in one bucket.
    #[test]
    fn a_log_channel_keeps_small_values_apart() {
        let meta = ChannelMeta::log(10_000_000.0);
        let small: Vec<u8> = [0.0, 1.0, 2.0, 5.0, 20.0]
            .iter()
            .map(|value| meta.encode(*value))
            .collect();
        let mut sorted = small.clone();
        sorted.dedup();
        assert_eq!(small.len(), sorted.len(), "{small:?} collapsed");
        assert_eq!(meta.encode(0.0), 0);
        assert_eq!(meta.decode(0), 0.0);
    }

    // The table is what every read actually goes through, so it has to agree with
    // the arithmetic it replaces rather than merely resemble it.
    #[test]
    fn the_table_agrees_with_decoding_a_byte() {
        for meta in [
            ChannelMeta::linear(-3.0, 9.0),
            ChannelMeta::categorical(),
            ChannelMeta::log(1000.0),
            ChannelMeta::unit(),
            ChannelMeta::raw(),
        ] {
            let table = meta.table();
            for byte in 0..=255u8 {
                assert_eq!(table.get(byte), meta.decode(byte));
            }
        }
    }

    // A range out of a file is untrusted, and each of these produces a channel
    // whose reads would be NaN rather than an error.
    #[test]
    fn an_unreadable_range_is_refused() {
        assert!(ChannelMeta::linear(0.0, 1.0).validate().is_ok());
        let cases = [
            ChannelMeta {
                low: f32::NAN,
                high: 1.0,
                encoding: ChannelEncoding::Linear,
            },
            ChannelMeta {
                low: 0.0,
                high: f32::INFINITY,
                encoding: ChannelEncoding::Linear,
            },
            ChannelMeta {
                low: 1.0,
                high: 0.0,
                encoding: ChannelEncoding::Linear,
            },
            ChannelMeta {
                low: -1.0,
                high: 5.0,
                encoding: ChannelEncoding::Log,
            },
        ];
        for meta in cases {
            assert!(meta.validate().is_err(), "{meta:?}");
        }
    }

    // The packing rule is the new logic in the format, and "stable across saves" is
    // only a claim if the grouping is pinned against a document that interleaves
    // shifts rather than one that happens to be sorted.
    #[test]
    fn fields_pack_by_shift_in_declaration_order() {
        let (places, shifts) = plan_layers(&[0, 4, 0, 4, 0, 0, 0]);
        assert_eq!(shifts, vec![0, 4, 0]);
        let layers: Vec<u8> = places.iter().map(|place| place.layer).collect();
        let channels: Vec<u8> = places.iter().map(|place| place.channel).collect();
        assert_eq!(layers, vec![0, 1, 0, 1, 0, 0, 2]);
        assert_eq!(channels, vec![0, 0, 1, 1, 2, 3, 0]);
    }

    // A layer that ran past four channels would be an image with no colour type,
    // so the cap is a property of the grouping rather than a later check.
    #[test]
    fn no_layer_is_given_more_channels_than_an_image_has() {
        let (places, shifts) = plan_layers(&[2; 9]);
        assert_eq!(shifts.len(), 3);
        for layer in 0..shifts.len() as u8 {
            let used = places.iter().filter(|place| place.layer == layer).count();
            assert!(used <= MAX_CHANNELS, "layer {layer} holds {used}");
        }
    }

    // An empty document is a legitimate thing to save, and the grouping is where it
    // would otherwise divide by zero or index an empty list.
    #[test]
    fn a_document_with_no_fields_plans_no_layers() {
        let (places, shifts) = plan_layers(&[]);
        assert!(places.is_empty());
        assert!(shifts.is_empty());
    }
}
