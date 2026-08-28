//! The editor's whole vocabulary for naming and changing a document: the paths that
//! address a field, a layer and a property, the words every enum is spelled with, and
//! the edits themselves.
//!
//! The panel and the control client both go through here rather than each writing
//! their own. Two spellings of "multiply" would be two things to keep in step, and a
//! path that worked from one and not the other would make the two disagree about what
//! a document even contains.

use serde_json::{Value, json};
use watershed::brush::{Brush, BrushMode};
use watershed::layer::{Blend, Layer, LayerOp, Mask, Remap, SlopeMode};
use watershed::noise::{NoiseKind, NoiseSpec, WarpSpec};
use watershed::raster::Raster;
use watershed::regions::RegionOutput;
use watershed::{Field, FieldRole, TerrainSpec};

/// A structural change to a document, as a value rather than a method.
///
/// Being a value is the point: a button builds one and a socket parses one, and both
/// then take the identical path through [`Edit::apply`]. Neither side can acquire a
/// shortcut the other lacks, and neither can change a document in a way the other
/// could not have.
pub enum Edit {
    /// Appends a layer to the top of a field's stack.
    Add {
        /// Field to add to. Must exist.
        field: String,
        /// The op the new layer carries, at full amplitude, added and unmasked.
        op: LayerOp,
    },
    /// Takes a layer out, renumbering every layer above it.
    Remove {
        /// Field to remove from. Must exist.
        field: String,
        /// Position in the stack. Must be in bounds.
        index: usize,
    },
    /// Moves a layer to another position in its stack.
    Move {
        /// Field to move within. Must exist.
        field: String,
        /// Position to move from. Must be in bounds.
        index: usize,
        /// Position to move to, counted in the stack *after* the layer has been lifted
        /// out of it — so moving layer 0 to 2 in a stack of three puts it on top.
        /// Clamped to the end rather than refused.
        to: usize,
    },
    /// Enables or disables a layer, which changes both what the bake evaluates and
    /// what the field depends on.
    Toggle {
        /// Field to toggle within. Must exist.
        field: String,
        /// Position in the stack. Must be in bounds.
        index: usize,
        /// The state to set, or `None` to flip whatever it is.
        enabled: Option<bool>,
    },
    /// Writes one property, named by a dotted path. See the module's grammar.
    Set {
        /// `field.property`, or `field.index.property`, or `field.index.op.property`.
        path: String,
        /// The value, as words. Most properties take one; a mask or a warp takes
        /// several.
        words: Vec<String>,
    },
}

impl Edit {
    /// Applies the edit and describes what it did, as the reply the control client
    /// sends back.
    ///
    /// Refused, with a message fit to show, if the edit names a field or a layer the
    /// document does not have or a value it cannot read. A refusal leaves the document
    /// exactly as it was.
    ///
    /// Nothing here notices that the bake is now stale — that is
    /// [`Document::apply`](crate::document::Document::apply)'s job, and why edits go
    /// through the document rather than through the terrain directly.
    pub fn apply(&self, terrain: &mut TerrainSpec) -> Result<Value, String> {
        match self {
            Self::Add { field, op } => {
                let name = op_name(op);
                let field = field_mut(terrain, field)?;
                field.layers.push(Layer::new(op.clone()));
                Ok(json!({
                    "added": name,
                    "index": field.layers.len() - 1,
                    "layers": field.layers.len(),
                }))
            }

            Self::Remove { field, index } => {
                let field = field_mut(terrain, field)?;
                bounds(&field.layers, *index)?;
                let removed = field.layers.remove(*index);
                Ok(json!({
                    "removed": op_name(&removed.op),
                    "layers": field.layers.len(),
                }))
            }

            Self::Move { field, index, to } => {
                let field = field_mut(terrain, field)?;
                bounds(&field.layers, *index)?;
                let to = (*to).min(field.layers.len().saturating_sub(1));
                let layer = field.layers.remove(*index);
                field.layers.insert(to, layer);
                Ok(json!({ "from": index, "to": to, "layers": field.layers.len() }))
            }

            Self::Toggle {
                field,
                index,
                enabled,
            } => {
                let field = field_mut(terrain, field)?;
                bounds(&field.layers, *index)?;
                let layer = &mut field.layers[*index];
                layer.enabled = enabled.unwrap_or(!layer.enabled);
                Ok(json!({ "index": index, "enabled": layer.enabled }))
            }

            Self::Set { path, words } => set(terrain, path, words),
        }
    }
}

fn field_mut<'a>(terrain: &'a mut TerrainSpec, name: &str) -> Result<&'a mut Field, String> {
    terrain
        .field_mut(name)
        .ok_or_else(|| format!("no field named `{name}`"))
}

fn bounds(layers: &[Layer], index: usize) -> Result<(), String> {
    if index < layers.len() {
        Ok(())
    } else {
        Err(format!(
            "layer {index} is past the end of a stack of {}",
            layers.len()
        ))
    }
}

fn set(terrain: &mut TerrainSpec, path: &str, words: &[String]) -> Result<Value, String> {
    let parts: Vec<&str> = path.split('.').collect();
    let name = *parts.first().ok_or("a path needs a field name")?;

    let Some(index) = parts.get(1) else {
        return Err(format!("`{path}` names a field and nothing on it"));
    };
    let Ok(index) = index.parse::<usize>() else {
        return set_field(terrain, name, &parts[1..], words);
    };

    let field = field_mut(terrain, name)?;
    bounds(&field.layers, index)?;
    let layer = &mut field.layers[index];
    let property = *parts
        .get(2)
        .ok_or_else(|| format!("`{path}` names a layer and nothing on it"))?;

    match property {
        "enabled" => {
            layer.enabled = boolean(first(words)?)?;
            Ok(json!({ "enabled": layer.enabled }))
        }
        "blend" => {
            layer.blend = parse_blend(first(words)?)?;
            Ok(json!({ "blend": blend_name(layer.blend) }))
        }
        "amplitude" => {
            layer.amplitude = number(first(words)?)?;
            Ok(json!({ "amplitude": layer.amplitude }))
        }
        "mask" => {
            layer.mask = parse_mask(words)?;
            Ok(json!({ "mask": mask_summary(&layer.mask) }))
        }
        "op" => match parts.get(3) {
            None => {
                layer.op = parse_op(words)?;
                Ok(json!({ "op": op_summary(&layer.op) }))
            }
            Some(property) => {
                set_op(&mut layer.op, property, words)?;
                Ok(json!({ "op": op_summary(&layer.op) }))
            }
        },
        other => Err(format!("a layer has nothing called `{other}`")),
    }
}

fn set_field(
    terrain: &mut TerrainSpec,
    name: &str,
    parts: &[&str],
    words: &[String],
) -> Result<Value, String> {
    match parts.first().copied() {
        Some("shift") => {
            let shift: u8 = number(first(words)?)?;
            if shift != 0 && is_solve_height(terrain, name) {
                return Err(format!(
                    "`{name}` is the water spec's height field and has to stay at shift 0"
                ));
            }
            let field = field_mut(terrain, name)?;
            field.shift = shift;
            Ok(json!({ "shift": field.shift }))
        }
        Some("role") => set_field_role(terrain, name, words),
        _ => set_other_field_property(terrain, name, parts, words),
    }
}

fn set_field_role(
    terrain: &mut TerrainSpec,
    name: &str,
    words: &[String],
) -> Result<Value, String> {
    let word = first(words)?;
    let role =
        FieldRole::parse(word).ok_or_else(|| format!("a field has no role called `{word}`"))?;

    let field = field_mut(terrain, name)?;
    let previous = field.role;
    if previous == role {
        return Ok(json!({ "role": role.as_str() }));
    }
    if role == FieldRole::Height && field.shift != 0 {
        return Err(format!(
            "`{name}` is at shift {} and a height field has to stay at shift 0",
            field.shift
        ));
    }

    let displaced: Vec<String> = if role == FieldRole::Custom {
        Vec::new()
    } else {
        terrain
            .fields
            .iter_mut()
            .filter(|field| field.role == role && field.id.as_str() != name)
            .map(|field| {
                field.role = FieldRole::Custom;
                field.id.to_string()
            })
            .collect()
    };

    field_mut(terrain, name)?.role = role;

    if terrain.water_spec.is_some() && terrain.field_with_role(FieldRole::Height).is_none() {
        field_mut(terrain, name)?.role = previous;
        for id in &displaced {
            field_mut(terrain, id)?.role = role;
        }
        return Err(format!(
            "`{name}` is the height field of a terrain that declares water — reset the water first"
        ));
    }

    Ok(json!({ "role": role.as_str(), "displaced": displaced }))
}

/// Whether the water solve would read this field as its height.
///
/// The one thing that pins a field's resolution: such a field is refused a non-zero
/// shift, because the solve reads its height one texel per cell and will not resample.
pub fn is_solve_height(terrain: &TerrainSpec, name: &str) -> bool {
    terrain
        .field_with_role(FieldRole::Height)
        .is_some_and(|field| field.id.as_str() == name)
}

fn set_other_field_property(
    terrain: &mut TerrainSpec,
    name: &str,
    parts: &[&str],
    words: &[String],
) -> Result<Value, String> {
    let field = field_mut(terrain, name)?;
    match parts.first().copied() {
        Some("range") => {
            let low: f32 = number(first(words)?)?;
            let high: f32 = number(words.get(1).ok_or("a range needs two numbers")?)?;
            field.range = (low, high);
            Ok(json!({ "range": [low, high] }))
        }
        Some(other) => Err(format!("a field has nothing called `{other}`")),
        None => Err("a path needs something after the field name".to_owned()),
    }
}

fn set_op(op: &mut LayerOp, property: &str, words: &[String]) -> Result<(), String> {
    match (op, property) {
        (LayerOp::Constant(value), "value") => *value = number(first(words)?)?,

        (LayerOp::Noise(spec), "kind") => spec.kind = parse_noise_kind(first(words)?)?,
        (LayerOp::Noise(spec), "scale") => spec.scale = number(first(words)?)?,
        (LayerOp::Noise(spec), "octaves") => spec.octaves = number(first(words)?)?,
        (LayerOp::Noise(spec), "seed") => spec.seed = number(first(words)?)?,
        (LayerOp::Noise(spec), "strike") => {
            spec.transform.strike_degrees = number(first(words)?)?;
        }
        (LayerOp::Noise(spec), "aspect") => spec.transform.aspect = number(first(words)?)?,
        (LayerOp::Noise(spec), "warp") => spec.warp = parse_warp(words)?,

        (LayerOp::Slope { of, .. }, "of") => *of = first(words)?.as_str().into(),
        (LayerOp::Slope { sample_tiles, .. }, "sample_tiles") => {
            *sample_tiles = number(first(words)?)?;
        }
        (LayerOp::Slope { mode, .. }, "mode") => *mode = parse_slope_mode(first(words)?)?,

        (LayerOp::FieldRef(id), "field") => *id = first(words)?.as_str().into(),

        (LayerOp::Regions { output, .. }, "output") => *output = parse_region_output(first(words)?),
        (LayerOp::Regions { spec, .. }, "seed") => spec.seed = number(first(words)?)?,
        (LayerOp::Regions { spec, .. }, "cell_tiles") => spec.cell_tiles = number(first(words)?)?,
        (LayerOp::Regions { spec, .. }, "blend_tiles") => spec.blend_tiles = number(first(words)?)?,
        (LayerOp::Regions { spec, .. }, "warp") => spec.warp = parse_warp(words)?,

        (op, other) => {
            return Err(format!("a {} op has nothing called `{other}`", op_name(op)));
        }
    }
    Ok(())
}

/// A region table is not a command line, so `regions` is deliberately absent: an existing
/// one is edited through `op.output` and the rest of `op.*`, and a new one comes from a
/// preset or a file.
pub fn parse_op(words: &[String]) -> Result<LayerOp, String> {
    let kind = first(words)?;
    let rest = &words[1..];
    match kind.as_str() {
        "constant" => Ok(LayerOp::Constant(number(first(rest)?)?)),
        "noise" => {
            let kind = parse_noise_kind(first(rest)?)?;
            let scale = number(rest.get(1).ok_or("a noise op needs a scale")?)?;
            let mut spec = NoiseSpec::new(0, kind, scale);
            if let Some(octaves) = rest.get(2) {
                spec.octaves = number(octaves)?;
            }
            if let Some(seed) = rest.get(3) {
                spec.seed = number(seed)?;
            }
            Ok(LayerOp::Noise(spec))
        }
        "fieldref" => Ok(LayerOp::FieldRef(first(rest)?.as_str().into())),
        "slope" => Ok(LayerOp::Slope {
            of: first(rest)?.as_str().into(),
            sample_tiles: number(rest.get(1).ok_or("a slope op needs a sample distance")?)?,
            mode: match rest.get(2) {
                Some(mode) => parse_slope_mode(mode)?,
                None => SlopeMode::default(),
            },
        }),
        "paint" => Ok(LayerOp::Paint(Raster::default())),
        other => Err(format!("no layer op called `{other}`")),
    }
}

/// Every brush mode, in the order the panel offers them.
pub const BRUSH_MODES: [BrushMode; 4] = [
    BrushMode::Add,
    BrushMode::Subtract,
    BrushMode::Set,
    BrushMode::Smooth,
];

/// The word this brush mode is named by, in the panel and on the command line.
pub fn brush_mode_name(mode: BrushMode) -> &'static str {
    match mode {
        BrushMode::Add => "add",
        BrushMode::Subtract => "sub",
        BrushMode::Set => "set",
        BrushMode::Smooth => "smooth",
    }
}

fn parse_brush_mode(word: &str) -> Result<BrushMode, String> {
    BRUSH_MODES
        .into_iter()
        .find(|mode| brush_mode_name(*mode) == word)
        .ok_or_else(|| format!("no brush mode called `{word}`"))
}

/// A change to one of the brush's settings, named and read the way a layer's
/// properties are — so the panel's controls and the control client's words cannot come
/// to mean different things.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BrushChange {
    /// Reach in document cells. Floored at zero on apply.
    Radius(f32),
    /// Fraction of the radius the weight falls off over. Clamped to `0.0..=1.0`.
    Falloff(f32),
    /// How hard the stroke pushes. Not clamped — what it means depends on the mode.
    Strength(f32),
    /// What `set` moves towards. Not clamped.
    Value(f32),
    /// What the stroke does to what it covers.
    Mode(BrushMode),
}

impl BrushChange {
    /// Reads `name` as one of the brush's settings and `word` as its value. Refused,
    /// with a message fit to show, for an unknown name or an unreadable value.
    pub fn parse(name: &str, word: &str) -> Result<Self, String> {
        match name {
            "radius" => Ok(Self::Radius(number(word)?)),
            "falloff" => Ok(Self::Falloff(number(word)?)),
            "strength" => Ok(Self::Strength(number(word)?)),
            "value" => Ok(Self::Value(number(word)?)),
            "mode" => Ok(Self::Mode(parse_brush_mode(word)?)),
            other => Err(format!("a brush has nothing called `{other}`")),
        }
    }

    /// Writes the change into `brush`, clamping where the setting has a range. Cannot
    /// fail.
    pub fn apply(self, brush: &mut Brush) {
        match self {
            Self::Radius(radius) => brush.radius_cells = radius.max(0.0),
            Self::Falloff(falloff) => brush.falloff = falloff.clamp(0.0, 1.0),
            Self::Strength(strength) => brush.strength = strength,
            Self::Value(value) => brush.value = value,
            Self::Mode(mode) => brush.mode = mode,
        }
    }
}

/// The brush as the control client reports it, keyed by the same names
/// [`BrushChange::parse`] takes.
pub fn brush_summary(brush: &Brush) -> Value {
    json!({
        "mode": brush_mode_name(brush.mode),
        "radius": brush.radius_cells,
        "falloff": brush.falloff,
        "strength": brush.strength,
        "value": brush.value,
    })
}

fn parse_mask(words: &[String]) -> Result<Mask, String> {
    match first(words)?.as_str() {
        "none" | "constant" if words.len() == 1 => Ok(Mask::Constant(1.0)),
        "constant" => Ok(Mask::Constant(number(&words[1])?)),
        "field" => {
            let id = words.get(1).ok_or("a field mask needs a field name")?;
            let remap = if words.len() >= 6 {
                Remap::new(
                    (number(&words[2])?, number(&words[3])?),
                    (number(&words[4])?, number(&words[5])?),
                )
            } else {
                Remap::IDENTITY
            };
            Ok(Mask::Field(id.as_str().into(), remap))
        }
        other => Err(format!("no mask called `{other}`")),
    }
}

fn parse_warp(words: &[String]) -> Result<Option<WarpSpec>, String> {
    if first(words)? == "none" {
        return Ok(None);
    }
    if words.len() < 4 {
        return Err("a warp needs a seed, an amplitude, a scale and an octave count".to_owned());
    }
    Ok(Some(WarpSpec {
        seed: number(&words[0])?,
        amplitude: number(&words[1])?,
        scale: number(&words[2])?,
        octaves: number(&words[3])?,
        salts: None,
    }))
}

/// Every blend mode, in the order the panel offers them.
pub const BLENDS: [Blend; 5] = [
    Blend::Add,
    Blend::Mul,
    Blend::Replace,
    Blend::Max,
    Blend::Min,
];

/// The word this blend mode is named by, in the panel and on the command line.
pub fn blend_name(blend: Blend) -> &'static str {
    match blend {
        Blend::Add => "add",
        Blend::Mul => "mul",
        Blend::Replace => "replace",
        Blend::Max => "max",
        Blend::Min => "min",
    }
}

fn parse_blend(word: &str) -> Result<Blend, String> {
    BLENDS
        .into_iter()
        .find(|blend| blend_name(*blend) == word)
        .ok_or_else(|| format!("no blend mode called `{word}`"))
}

/// Every noise kind, in the order the panel offers them.
pub const NOISE_KINDS: [NoiseKind; 3] = [NoiseKind::Fbm, NoiseKind::Signed, NoiseKind::Ridged];

/// The word this noise kind is named by, in the panel and on the command line.
pub fn noise_kind_name(kind: NoiseKind) -> &'static str {
    match kind {
        NoiseKind::Fbm => "fbm",
        NoiseKind::Signed => "signed",
        NoiseKind::Ridged => "ridged",
    }
}

fn parse_noise_kind(word: &str) -> Result<NoiseKind, String> {
    NOISE_KINDS
        .into_iter()
        .find(|kind| noise_kind_name(*kind) == word)
        .ok_or_else(|| format!("no noise kind called `{word}`"))
}

/// The slope mode of that exact name. The error names both spellings, since there are
/// only two and a caller that got it wrong wants to see them.
pub fn parse_slope_mode(word: &str) -> Result<SlopeMode, String> {
    match word {
        "gradient" => Ok(SlopeMode::Gradient),
        "steepest_axis" => Ok(SlopeMode::SteepestAxis),
        other => Err(format!(
            "no slope mode called `{other}` — it is `gradient` or `steepest_axis`"
        )),
    }
}

/// The word this slope mode is named by, and the only spelling
/// [`parse_slope_mode`] accepts.
pub fn slope_mode_name(mode: SlopeMode) -> &'static str {
    match mode {
        SlopeMode::Gradient => "gradient",
        SlopeMode::SteepestAxis => "steepest_axis",
    }
}

/// Every slope mode, in the order the panel offers them.
pub const SLOPE_MODES: [SlopeMode; 2] = [SlopeMode::Gradient, SlopeMode::SteepestAxis];

/// Reads a region output. Cannot fail: a bare word is taken as a column name, so the
/// two categorical outputs take names no column would, and a column the table does not
/// carry is caught at plan time where the table is known.
pub fn parse_region_output(word: &str) -> RegionOutput {
    match word {
        "region_id" => RegionOutput::RegionId,
        "cover_class" => RegionOutput::CoverClass,
        column => RegionOutput::Blended(column.to_owned()),
    }
}

/// The word this region output is named by, and the spelling
/// [`parse_region_output`] reads back.
pub fn region_output_name(output: &RegionOutput) -> String {
    match output {
        RegionOutput::Blended(column) => column.clone(),
        RegionOutput::RegionId => "region_id".to_owned(),
        RegionOutput::CoverClass => "cover_class".to_owned(),
    }
}

/// The word this op is named by, and the spelling `parse_op` takes — except
/// `regions` and `external`, which nothing builds from words.
pub fn op_name(op: &LayerOp) -> &'static str {
    match op {
        LayerOp::Constant(_) => "constant",
        LayerOp::Noise(_) => "noise",
        LayerOp::Paint(_) => "paint",
        LayerOp::Slope { .. } => "slope",
        LayerOp::FieldRef(_) => "fieldref",
        LayerOp::Regions { .. } => "regions",
        LayerOp::External(_) => "external",
    }
}

/// One line describing an op and its parameters, for the layer panel's collapsed row
/// and the control client's listing. Not a path, and nothing reads it back.
pub fn op_summary(op: &LayerOp) -> String {
    match op {
        LayerOp::Constant(value) => format!("constant {value}"),
        LayerOp::Noise(spec) => format!(
            "{} scale {} x{}",
            noise_kind_name(spec.kind),
            spec.scale,
            spec.octaves
        ),
        LayerOp::Paint(raster) => format!("paint {}x{}", raster.width(), raster.height()),
        LayerOp::Slope {
            of,
            sample_tiles,
            mode,
        } => format!(
            "slope of {of} over {sample_tiles} by {}",
            slope_mode_name(*mode)
        ),
        LayerOp::FieldRef(id) => format!("fieldref {id}"),
        LayerOp::Regions { output, .. } => format!("regions {}", region_output_name(output)),
        LayerOp::External(raster) => format!("external {}x{}", raster.width(), raster.height()),
    }
}

/// One line describing a mask, on the same terms as [`op_summary`].
pub fn mask_summary(mask: &Mask) -> String {
    match mask {
        Mask::Constant(value) => format!("constant {value}"),
        Mask::Painted(raster) => format!("painted {}x{}", raster.width(), raster.height()),
        Mask::Field(id, remap) => format!(
            "field {id} {}..{} -> {}..{}",
            remap.from.0, remap.from.1, remap.to.0, remap.to.1
        ),
    }
}

fn first(words: &[String]) -> Result<&String, String> {
    words.first().ok_or_else(|| "a value is missing".to_owned())
}

fn number<T: std::str::FromStr>(word: &str) -> Result<T, String> {
    word.parse().map_err(|_| format!("not a number: {word}"))
}

fn boolean(word: &str) -> Result<bool, String> {
    match word {
        "true" | "on" | "yes" | "1" => Ok(true),
        "false" | "off" | "no" | "0" => Ok(false),
        other => Err(format!("not a yes or a no: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::UVec2;
    use watershed::{FieldId, WaterSpec};

    fn document() -> TerrainSpec {
        TerrainSpec::new(UVec2::new(64, 64))
            .with_field(
                Field::new("base")
                    .with_layer(Layer::new(LayerOp::Constant(0.25)).with_blend(Blend::Replace)),
            )
            .with_field(
                Field::new("height")
                    .with_role(FieldRole::Height)
                    .with_layer(Layer::new(LayerOp::FieldRef(FieldId::from("base"))))
                    .with_layer(Layer::new(LayerOp::Noise(NoiseSpec::new(
                        1,
                        NoiseKind::Fbm,
                        0.02,
                    )))),
            )
    }

    fn words(line: &str) -> Vec<String> {
        line.split_whitespace().map(str::to_owned).collect()
    }

    fn set_line(terrain: &mut TerrainSpec, line: &str) -> Result<Value, String> {
        let words = words(line);
        Edit::Set {
            path: words[0].clone(),
            words: words[1..].to_vec(),
        }
        .apply(terrain)
    }

    // The four structural verbs against one stack, in sequence, because each one
    // renumbers the layers the next is addressed by — a reply that reported the wrong
    // index would send the panel's next edit to a different layer.
    #[test]
    fn a_layer_can_be_added_removed_reordered_and_switched_off() {
        let mut terrain = document();

        Edit::Add {
            field: "height".to_owned(),
            op: LayerOp::Constant(0.5),
        }
        .apply(&mut terrain)
        .unwrap();
        assert_eq!(terrain.field("height").unwrap().layers.len(), 3);

        Edit::Move {
            field: "height".to_owned(),
            index: 2,
            to: 0,
        }
        .apply(&mut terrain)
        .unwrap();
        assert!(matches!(
            terrain.field("height").unwrap().layers[0].op,
            LayerOp::Constant(_)
        ));

        Edit::Toggle {
            field: "height".to_owned(),
            index: 0,
            enabled: None,
        }
        .apply(&mut terrain)
        .unwrap();
        assert!(!terrain.field("height").unwrap().layers[0].enabled);

        Edit::Remove {
            field: "height".to_owned(),
            index: 0,
        }
        .apply(&mut terrain)
        .unwrap();
        assert_eq!(terrain.field("height").unwrap().layers.len(), 2);
    }

    // The assertion the scenario exists for, made where it can be made numerically: the
    // editor's whole claim is that editing the stack changes the field it bakes.
    #[test]
    fn a_layer_added_to_a_stack_moves_the_bake_it_produces() {
        let mut terrain = document();
        terrain.bake_in_place().unwrap();
        let before = terrain.field("height").unwrap().baked().data().to_vec();

        Edit::Add {
            field: "height".to_owned(),
            op: LayerOp::Constant(0.25),
        }
        .apply(&mut terrain)
        .unwrap();
        terrain.bake_in_place().unwrap();
        let after = terrain.field("height").unwrap().baked().data().to_vec();

        assert_ne!(before, after);
        assert!(
            before
                .iter()
                .zip(&after)
                .all(|(before, after)| after >= before),
            "a texel fell after a layer was added"
        );

        Edit::Toggle {
            field: "height".to_owned(),
            index: 2,
            enabled: Some(false),
        }
        .apply(&mut terrain)
        .unwrap();
        terrain.bake_in_place().unwrap();
        assert_eq!(terrain.field("height").unwrap().baked().data(), &before[..]);
    }

    // "Move to the top" reaches this as a number past the end, from the panel's button
    // and from a script alike; refusing it would make the commonest move the one that
    // fails.
    #[test]
    fn a_move_past_the_end_lands_on_the_end_rather_than_being_refused() {
        let mut terrain = document();
        Edit::Move {
            field: "height".to_owned(),
            index: 0,
            to: 99,
        }
        .apply(&mut terrain)
        .unwrap();
        assert!(matches!(
            terrain.field("height").unwrap().layers[1].op,
            LayerOp::FieldRef(_)
        ));
    }

    // Every one of these arrives from a caller working against a document that has
    // changed under it, so each has to be a message rather than a panic or a silent
    // no-op that looks like the edit was applied.
    #[test]
    fn an_edit_naming_something_the_document_does_not_have_is_refused() {
        let mut terrain = document();
        assert!(
            Edit::Add {
                field: "nowhere".to_owned(),
                op: LayerOp::Constant(0.5),
            }
            .apply(&mut terrain)
            .is_err()
        );
        assert!(
            Edit::Remove {
                field: "height".to_owned(),
                index: 9,
            }
            .apply(&mut terrain)
            .is_err()
        );
        assert!(set_line(&mut terrain, "height.0.sideways 1").is_err());
        assert!(set_line(&mut terrain, "height.0.op.scale 1").is_err());
        assert!(set_line(&mut terrain, "height").is_err());
    }

    // The path grammar is the whole surface the control client edits through, so a
    // property nothing can address is a control the panel has and a script cannot use.
    #[test]
    fn every_layer_property_is_reachable_by_its_path() {
        let mut terrain = document();

        set_line(&mut terrain, "height.1.amplitude 0.35").unwrap();
        set_line(&mut terrain, "height.1.blend mul").unwrap();
        set_line(&mut terrain, "height.1.enabled off").unwrap();
        set_line(&mut terrain, "height.1.mask field base 0.4 0.6 0 1").unwrap();

        let layer = &terrain.field("height").unwrap().layers[1];
        assert_eq!(layer.amplitude, 0.35);
        assert_eq!(layer.blend, Blend::Mul);
        assert!(!layer.enabled);
        assert_eq!(
            layer.mask,
            Mask::Field(FieldId::from("base"), Remap::new((0.4, 0.6), (0.0, 1.0)))
        );
    }

    // The reason op parameters are addressable one at a time: the seed here comes from
    // the document rather than from any of the three edits, where rewriting the op
    // wholesale would have to restate every parameter and would silently reset the ones
    // it forgot.
    #[test]
    fn an_op_parameter_can_be_moved_without_rewriting_the_op_around_it() {
        let mut terrain = document();
        set_line(&mut terrain, "height.1.op.scale 0.004").unwrap();
        set_line(&mut terrain, "height.1.op.octaves 6").unwrap();
        set_line(&mut terrain, "height.1.op.kind ridged").unwrap();

        let LayerOp::Noise(spec) = &terrain.field("height").unwrap().layers[1].op else {
            panic!("the op stopped being noise");
        };
        assert_eq!(spec.scale, 0.004);
        assert_eq!(spec.octaves, 6);
        assert_eq!(spec.kind, NoiseKind::Ridged);
        assert_eq!(spec.seed, 1);
    }

    // `height.shift` and `height.1.blend` are one grammar with no marker segment
    // between them, so the only thing separating a field property from a layer index is
    // whether the segment parses as a number.
    #[test]
    fn a_field_property_is_told_apart_from_a_layer_index_by_being_unreadable_as_a_number() {
        let mut terrain = document();
        set_line(&mut terrain, "base.shift 2").unwrap();
        set_line(&mut terrain, "base.range -1 1").unwrap();
        let field = terrain.field("base").unwrap();
        assert_eq!(field.shift, 2);
        assert_eq!(field.range, (-1.0, 1.0));
    }

    // The defect this guards was reachable from the panel in one drag: `solve_water`
    // reads its height one texel per cell and refuses to resample, so a coarse height
    // field is a document that can never solve — and the refusal names the shift rather
    // than the edit that set it. The moisture field is checked too, in the other
    // direction: the solve samples it rather than indexing it, so it is free to be
    // coarse, which is what every preset does with it.
    #[test]
    fn the_water_specs_height_field_cannot_be_made_coarse() {
        let mut terrain = document();
        terrain.water_spec = Some(WaterSpec::new("height").with_moisture("base"));

        let refused = set_line(&mut terrain, "height.shift 2").unwrap_err();
        assert!(refused.contains("shift 0"), "{refused}");
        assert_eq!(terrain.field("height").unwrap().shift, 0);

        set_line(&mut terrain, "base.shift 4").unwrap();
        assert_eq!(terrain.field("base").unwrap().shift, 4);
    }

    // The defect this guards was reported from the running editor as "solve water does
    // nothing; it only works on a fresh document". An edit invalidates the *state* the
    // solve produced; it must not take away the *spec* the solve is run from, or the
    // first edit after the first solve makes the document permanently unsolvable. The
    // edit and the invalidation in the body are what `Document::note_edit` does to a
    // terrain, spelled out because a test has no app to do it through.
    #[test]
    fn an_edit_after_a_solve_leaves_the_document_solvable() {
        let mut terrain = document();
        terrain.water_spec = Some(WaterSpec::new("height"));
        terrain.bake_in_place().unwrap();
        let spec = terrain.water_spec.clone().unwrap();
        terrain.solve_water(&spec).unwrap();

        Edit::Set {
            path: "height.1.op.scale".to_owned(),
            words: vec!["0.05".to_owned()],
        }
        .apply(&mut terrain)
        .unwrap();
        terrain.invalidate_water();

        assert!(terrain.water().is_none(), "the stale answer is dropped");
        assert!(
            terrain.water_spec.is_some(),
            "the recipe the next solve needs is not"
        );
    }

    // Zero has to stay reachable, or a document that arrived at a coarse height some other
    // way — a file written before the guard existed — could never be put back.
    #[test]
    fn a_height_field_can_always_be_returned_to_one_texel_per_cell() {
        let mut terrain = document();
        terrain.field_mut("height").unwrap().shift = 3;
        terrain.water_spec = Some(WaterSpec::new("height"));

        set_line(&mut terrain, "height.shift 0").unwrap();
        assert_eq!(terrain.field("height").unwrap().shift, 0);
    }

    // A role is what the bake reads, so the panel cannot be allowed to leave two fields
    // claiming one: taking it takes it from whoever held it.
    #[test]
    fn taking_a_role_takes_it_from_the_field_that_held_it() {
        let mut terrain = document();
        set_line(&mut terrain, "base.role height").unwrap();

        assert_eq!(terrain.field("base").unwrap().role, FieldRole::Height);
        assert_eq!(terrain.field("height").unwrap().role, FieldRole::Custom);
    }

    // The same rule the shift control carries, arrived at from the other side: a coarse
    // field cannot become the height field either.
    #[test]
    fn a_coarse_field_cannot_take_the_height_role() {
        let mut terrain = document();
        terrain.field_mut("base").unwrap().shift = 4;

        let refused = set_line(&mut terrain, "base.role height").unwrap_err();
        assert!(refused.contains("shift 0"), "{refused}");
        assert_eq!(terrain.field("base").unwrap().role, FieldRole::Custom);
    }

    // Resetting the water is how a terrain stops having a height field. An edit that took
    // the last one away would leave a document that can never solve.
    #[test]
    fn a_terrain_that_declares_water_cannot_be_left_without_a_height_field() {
        let mut terrain = document();
        terrain.water_spec = Some(WaterSpec::new("height"));

        let refused = set_line(&mut terrain, "height.role custom").unwrap_err();
        assert!(refused.contains("reset the water"), "{refused}");
        assert_eq!(terrain.field("height").unwrap().role, FieldRole::Height);
    }

    // The refusal has to put back everything it moved, or a rejected edit leaves the
    // document holding a role the panel never showed being taken.
    #[test]
    fn a_refused_role_change_leaves_every_other_field_as_it_was() {
        let mut terrain = document();
        terrain.field_mut("base").unwrap().role = FieldRole::Moisture;
        terrain.water_spec = Some(WaterSpec::new("height"));

        set_line(&mut terrain, "height.role moisture").unwrap_err();

        assert_eq!(terrain.field("height").unwrap().role, FieldRole::Height);
        assert_eq!(terrain.field("base").unwrap().role, FieldRole::Moisture);
    }

    // Roles are spelled the same way everywhere, so an unknown word has to be refused
    // rather than fall back to `custom` — which would silently take a document's height
    // away.
    #[test]
    fn a_role_the_vocabulary_does_not_have_is_refused() {
        let mut terrain = document();
        let refused = set_line(&mut terrain, "height.role elevation").unwrap_err();
        assert!(refused.contains("elevation"), "{refused}");
    }

    // The add button and the control client build ops from the same words, so a
    // spelling that parsed to the wrong op would give the two different documents from
    // the same instruction.
    #[test]
    fn every_op_a_command_line_can_write_parses_to_the_op_it_names() {
        for (line, name) in [
            ("constant 0.5", "constant"),
            ("noise fbm 0.01", "noise"),
            ("noise ridged 0.01 5 42", "noise"),
            ("fieldref base", "fieldref"),
            ("slope base 4", "slope"),
        ] {
            let op = parse_op(&words(line)).unwrap_or_else(|error| panic!("{line}: {error}"));
            assert_eq!(op_name(&op), name, "{line}");
        }
        assert!(parse_op(&words("regions")).is_err());
        assert!(parse_op(&words("noise sideways 0.01")).is_err());
        assert!(parse_op(&words("noise fbm")).is_err());
    }

    // Naming and parsing are written out separately for each enum, so nothing but this
    // forces them to agree; a name that does not parse back makes a value the panel can
    // display and no script can set.
    #[test]
    fn every_blend_mode_and_noise_kind_parses_back_from_the_name_it_prints() {
        for blend in BLENDS {
            assert_eq!(parse_blend(blend_name(blend)).unwrap(), blend);
        }
        for kind in NOISE_KINDS {
            assert_eq!(parse_noise_kind(noise_kind_name(kind)).unwrap(), kind);
        }
        assert!(parse_blend("sideways").is_err());
    }

    // The two categorical outputs have to be unreachable as column names, or a table with
    // a column called `region_id` would make one of them unsayable.
    #[test]
    fn a_region_output_parses_back_from_the_name_it_prints() {
        for output in [
            RegionOutput::RegionId,
            RegionOutput::CoverClass,
            RegionOutput::Blended("base".to_owned()),
        ] {
            let name = region_output_name(&output);
            assert_eq!(parse_region_output(&name), output);
        }
    }
}
