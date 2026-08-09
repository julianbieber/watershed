// TODO(jb-doc): module docs — that this owns the whole vocabulary for naming and
// addressing a layer, and why the panel and the ctl are obliged to share it rather than
// each spelling a blend mode its own way.

use serde_json::{Value, json};
use watershed::brush::{Brush, BrushMode};
use watershed::layer::{Blend, Layer, LayerOp, Mask, Remap};
use watershed::noise::{NoiseKind, NoiseSpec, WarpSpec};
use watershed::raster::Raster;
use watershed::regions::RegionOutput;
use watershed::{Field, Terrain};

/// TODO(jb-doc): why a structural change is a value applied to a terrain rather than a
/// method on the document — that the same four verbs reach it from a button and from a
/// socket, and neither may take a shortcut the other cannot.
pub enum Edit {
    Add {
        field: String,
        op: LayerOp,
    },
    Remove {
        field: String,
        index: usize,
    },
    /// TODO(jb-comment): why the destination is an index in the list *after* the layer has
    /// been lifted out of it.
    Move {
        field: String,
        index: usize,
        to: usize,
    },
    Toggle {
        field: String,
        index: usize,
        enabled: Option<bool>,
    },
    Set {
        path: String,
        words: Vec<String>,
    },
}

impl Edit {
    pub fn apply(&self, terrain: &mut Terrain) -> Result<Value, String> {
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
                // Clamped rather than refused: a scenario saying "to the top" writes a
                // number past the end, and the last position is what it meant.
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

fn field_mut<'a>(terrain: &'a mut Terrain, name: &str) -> Result<&'a mut Field, String> {
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

/// TODO(jb-doc): the grammar — that a path names a *place* and the words that follow are
/// the value, and why a mask is addressed as one place taking several words rather than as
/// a place per shape it can take.
fn set(terrain: &mut Terrain, path: &str, words: &[String]) -> Result<Value, String> {
    let parts: Vec<&str> = path.split('.').collect();
    let name = *parts.first().ok_or("a path needs a field name")?;

    // A second segment that is a number is a layer index; anything else is a property of
    // the field itself, which is what keeps `height.shift` and `height.1.blend` in one
    // grammar without a marker segment between them.
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
    terrain: &mut Terrain,
    name: &str,
    parts: &[&str],
    words: &[String],
) -> Result<Value, String> {
    match parts.first().copied() {
        // A shift changes the field's resolution, and the bake is discarded rather than
        // resampled — so this is the one edit after which even the visible rectangle is
        // rebuilt from nothing rather than patched.
        Some("shift") => {
            let shift: u8 = number(first(words)?)?;
            // Refused here rather than discovered at the solve: `solve_water` reads the
            // height one texel per cell and will not resample, so a coarse one leaves a
            // document that can never solve — and the error it raises names the shift
            // rather than the edit that set it. Zero is always allowed, or the field could
            // not be put back.
            if shift != 0 && is_solve_height(terrain, name) {
                return Err(format!(
                    "`{name}` is the water spec's height field and has to stay at shift 0"
                ));
            }
            let field = field_mut(terrain, name)?;
            field.shift = shift;
            Ok(json!({ "shift": field.shift }))
        }
        _ => set_other_field_property(terrain, name, parts, words),
    }
}

/// Whether the water solve would read this field as its height, which is the one thing
/// that pins a field's resolution.
pub fn is_solve_height(terrain: &Terrain, name: &str) -> bool {
    terrain
        .water_spec
        .as_ref()
        .is_some_and(|spec| spec.height.as_str() == name)
}

fn set_other_field_property(
    terrain: &mut Terrain,
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

/// TODO(jb-doc): why an op's parameters are reachable one at a time as well as wholesale,
/// and what editing `op.scale` preserves that rewriting the op would throw away.
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
        }),
        // Sized by the first stroke rather than here, which is what lets it be written
        // without a document to measure against — and what makes an unpainted one inert,
        // since an empty raster reads as zero everywhere.
        "paint" => Ok(LayerOp::Paint(Raster::default())),
        other => Err(format!("no layer op called `{other}`")),
    }
}

pub const BRUSH_MODES: [BrushMode; 4] = [
    BrushMode::Add,
    BrushMode::Subtract,
    BrushMode::Set,
    BrushMode::Smooth,
];

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

/// One of the brush's numbers, named and read the way a layer's properties are — so the
/// panel's controls and the ctl's words cannot come to mean different things.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BrushChange {
    Radius(f32),
    Falloff(f32),
    Strength(f32),
    Value(f32),
    Mode(BrushMode),
}

impl BrushChange {
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
    }))
}

pub const BLENDS: [Blend; 5] = [
    Blend::Add,
    Blend::Mul,
    Blend::Replace,
    Blend::Max,
    Blend::Min,
];

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

pub const NOISE_KINDS: [NoiseKind; 3] = [NoiseKind::Fbm, NoiseKind::Signed, NoiseKind::Ridged];

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

/// A bare word is a column name, so the two categorical outputs take names no column
/// would: an unknown column is caught by the bake, which is where the table is known.
pub fn parse_region_output(word: &str) -> RegionOutput {
    match word {
        "region_id" => RegionOutput::RegionId,
        "cover_class" => RegionOutput::CoverClass,
        column => RegionOutput::Blended(column.to_owned()),
    }
}

pub fn region_output_name(output: &RegionOutput) -> String {
    match output {
        RegionOutput::Blended(column) => column.clone(),
        RegionOutput::RegionId => "region_id".to_owned(),
        RegionOutput::CoverClass => "cover_class".to_owned(),
    }
}

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
        LayerOp::Slope { of, sample_tiles } => format!("slope of {of} over {sample_tiles}"),
        LayerOp::FieldRef(id) => format!("fieldref {id}"),
        LayerOp::Regions { output, .. } => format!("regions {}", region_output_name(output)),
        LayerOp::External(raster) => format!("external {}x{}", raster.width(), raster.height()),
    }
}

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

    fn document() -> Terrain {
        Terrain::new(UVec2::new(64, 64))
            .with_field(
                Field::new("base")
                    .with_layer(Layer::new(LayerOp::Constant(0.25)).with_blend(Blend::Replace)),
            )
            .with_field(
                Field::new("height")
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

    fn set_line(terrain: &mut Terrain, line: &str) -> Result<Value, String> {
        let words = words(line);
        Edit::Set {
            path: words[0].clone(),
            words: words[1..].to_vec(),
        }
        .apply(terrain)
    }

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

    /// The assertion the scenario exists for, made where it can be made numerically: the
    /// editor's whole claim is that editing the stack changes the field it bakes.
    #[test]
    fn a_layer_added_to_a_stack_moves_the_bake_it_produces() {
        let mut terrain = document();
        terrain.bake().unwrap();
        let before = terrain.field("height").unwrap().baked().data().to_vec();

        Edit::Add {
            field: "height".to_owned(),
            op: LayerOp::Constant(0.25),
        }
        .apply(&mut terrain)
        .unwrap();
        terrain.bake().unwrap();
        let after = terrain.field("height").unwrap().baked().data().to_vec();

        assert_ne!(before, after);
        // The added layer adds a constant, so no texel may come out lower than it was —
        // a difference in the other direction would mean the stack was reordered.
        assert!(
            before
                .iter()
                .zip(&after)
                .all(|(before, after)| after >= before),
            "a texel fell after a layer was added"
        );

        // And switching it off puts every one of them back, which is what makes `enabled`
        // data rather than the caller keeping a copy of the stack.
        Edit::Toggle {
            field: "height".to_owned(),
            index: 2,
            enabled: Some(false),
        }
        .apply(&mut terrain)
        .unwrap();
        terrain.bake().unwrap();
        assert_eq!(terrain.field("height").unwrap().baked().data(), &before[..]);
    }

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
        // The seed came from the document rather than from any of the three edits, which
        // is the whole difference between this and writing the op again.
        assert_eq!(spec.seed, 1);
    }

    #[test]
    fn a_field_property_is_told_apart_from_a_layer_index_by_being_unreadable_as_a_number() {
        let mut terrain = document();
        set_line(&mut terrain, "height.shift 2").unwrap();
        set_line(&mut terrain, "height.range -1 1").unwrap();
        let field = terrain.field("height").unwrap();
        assert_eq!(field.shift, 2);
        assert_eq!(field.range, (-1.0, 1.0));
    }

    /// The defect this guards was reachable from the panel in one drag: `solve_water`
    /// reads its height one texel per cell and refuses to resample, so a coarse height
    /// field is a document that can never solve — and the refusal names the shift rather
    /// than the edit that set it.
    #[test]
    fn the_water_specs_height_field_cannot_be_made_coarse() {
        let mut terrain = document();
        terrain.water_spec = Some(WaterSpec::new("height").with_moisture("base"));

        let refused = set_line(&mut terrain, "height.shift 2").unwrap_err();
        assert!(refused.contains("shift 0"), "{refused}");
        assert_eq!(terrain.field("height").unwrap().shift, 0);

        // The field the spec reads as *moisture* is sampled rather than indexed, so it is
        // free to be coarse — which is what every preset does with it.
        set_line(&mut terrain, "base.shift 4").unwrap();
        assert_eq!(terrain.field("base").unwrap().shift, 4);
    }

    /// The defect this guards was reported from the running editor as "solve water does
    /// nothing; it only works on a fresh document". An edit invalidates the *state* the
    /// solve produced; it must not take away the *spec* the solve is run from, or the
    /// first edit after the first solve makes the document permanently unsolvable.
    #[test]
    fn an_edit_after_a_solve_leaves_the_document_solvable() {
        let mut terrain = document();
        terrain.water_spec = Some(WaterSpec::new("height"));
        terrain.bake().unwrap();
        let spec = terrain.water_spec.clone().unwrap();
        terrain.solve_water(&spec).unwrap();

        // What `Document::note_edit` does to the terrain, made here without an app.
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

    /// Zero has to stay reachable, or a document that got into the coarse state before the
    /// guard existed could never be put back.
    #[test]
    fn a_height_field_can_always_be_returned_to_one_texel_per_cell() {
        let mut terrain = document();
        set_line(&mut terrain, "height.shift 3").unwrap();
        terrain.water_spec = Some(WaterSpec::new("height"));

        set_line(&mut terrain, "height.shift 0").unwrap();
        assert_eq!(terrain.field("height").unwrap().shift, 0);
    }

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

    /// The two categorical outputs have to be unreachable as column names, or a table with
    /// a column called `region_id` would make one of them unsayable.
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
