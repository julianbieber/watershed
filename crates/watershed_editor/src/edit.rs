//! The editor's whole vocabulary for naming and changing a document: the paths that
//! address a field, a node and a property, the words every enum is spelled with, and
//! the edits themselves.
//!
//! The panel and the control client both go through here rather than each writing
//! their own. Two spellings of "multiply" would be two things to keep in step, and a
//! path that worked from one and not the other would make the two disagree about what
//! a document even contains.

use crate::terrain::brush::{Brush, BrushMode};
use crate::terrain::graph::{Binary, Curve, CurvePoint, FieldGraph, NodeId, NodeOp};
use crate::terrain::graph::{Remap, SlopeMode};
use crate::terrain::noise::{NoiseKind, NoiseSpec, WarpSpec};
use crate::terrain::regions::RegionOutput;
use crate::terrain::shader::ShaderLayer;
use crate::terrain::{Field, TerrainSpec};
use serde_json::{Value, json};
use watershed::FieldRole;
use watershed::raster::Raster;

/// A structural change to a document, as a value rather than a method.
///
/// Being a value is the point: a button builds one and a socket parses one, and both
/// then take the identical path through [`Edit::apply`]. Neither side can acquire a
/// shortcut the other lacks, and neither can change a document in a way the other
/// could not have.
#[derive(Clone)]
pub enum Edit {
    /// Adds an unconnected node to a field's graph.
    AddNode {
        /// Field to add to. Must exist.
        field: String,
        /// What the node produces.
        op: NodeOp,
        /// Where its card sits on the canvas, or `None` to put it somewhere free —
        /// which is what a caller that is not pointing at the canvas means.
        position: Option<[f32; 2]>,
    },
    /// Takes a node out along with every edge touching it.
    RemoveNode {
        /// Field to remove from. Must exist.
        field: String,
        /// The node, as `n<id>` or its name.
        node: String,
    },
    /// Writes one input pin, replacing whatever was on it.
    Connect {
        /// Field the two nodes are in. Must exist.
        field: String,
        /// The node the value comes from.
        from: String,
        /// The node it is written into.
        to: String,
        /// Which of that node's input pins.
        pin: usize,
    },
    /// Clears one input pin, which then reads `0.0`.
    Disconnect {
        /// Field the node is in. Must exist.
        field: String,
        /// The node whose pin is cleared.
        node: String,
        /// Which pin.
        pin: usize,
    },
    /// Sets or flips whether a node is bypassed.
    Bypass {
        /// Field the node is in. Must exist.
        field: String,
        /// The node.
        node: String,
        /// The state to set, or `None` to flip whatever it is.
        bypassed: Option<bool>,
    },
    /// Writes a node's canvas position and nothing else, so it never re-bakes.
    PlaceNode {
        /// Field the node is in. Must exist.
        field: String,
        /// The node.
        node: String,
        /// Where to put it.
        position: [f32; 2],
    },
    /// Names a node, or clears its name.
    RenameNode {
        /// Field the node is in. Must exist.
        field: String,
        /// The node.
        node: String,
        /// The new name, or `None` to clear it.
        name: Option<String>,
    },
    /// Names the node the field's value is read from.
    SetOutput {
        /// Field to write. Must exist.
        field: String,
        /// The node, or `None` to leave the field with no output — it then bakes
        /// zero rather than refusing the document.
        node: Option<String>,
    },
    /// Writes one property, named by a dotted path. See the module's grammar.
    Set {
        /// `field.property`, or `field.node.property`, where node is `n<id>` or the
        /// node's name.
        path: String,
        /// The value, as words. Most properties take one; a remap or a warp takes
        /// several.
        words: Vec<String>,
    },
}

impl Edit {
    /// Whether what this edit changes is read by a bake.
    ///
    /// A node's position and its name are authoring data: they are written to the
    /// document and saved with it, but nothing that evaluates a texel reads either. An
    /// edit that touches only those must not make the bake stale, or dragging a card
    /// would throw away the whole field — and the solved water with it.
    pub fn reaches_the_bake(&self) -> bool {
        !matches!(self, Self::PlaceNode { .. } | Self::RenameNode { .. })
    }

    /// Applies the edit and describes what it did, as the reply the control client
    /// sends back.
    ///
    /// Refused, with a message fit to show, if the edit names a field or a node the
    /// document does not have, a value it cannot read, or an edge that would close a
    /// cycle. A refusal leaves the document exactly as it was.
    ///
    /// Nothing here notices that the bake is now stale — that is
    /// [`Document::apply`](crate::document::Document::apply)'s job, and why edits go
    /// through the document rather than through the terrain directly.
    pub fn apply(&self, terrain: &mut TerrainSpec) -> Result<Value, String> {
        match self {
            Self::AddNode {
                field,
                op,
                position,
            } => {
                let name = op_name(op);
                let field = field_mut(terrain, field)?;
                let at = position.unwrap_or_else(|| field.graph.free_position());
                let id = field.graph.add_node(op.clone(), at);
                Ok(json!({
                    "added": name,
                    "node": node_path(id),
                    "nodes": field.graph.nodes.len(),
                }))
            }

            Self::RemoveNode { field, node } => {
                let field = field_mut(terrain, field)?;
                let id = node_id(&field.graph, node)?;
                let removed = op_name(&field.graph.node(id).expect("resolved above").op);
                field.graph.remove_node(id).map_err(refusal)?;
                Ok(json!({ "removed": removed, "nodes": field.graph.nodes.len() }))
            }

            Self::Connect {
                field,
                from,
                to,
                pin,
            } => {
                let field = field_mut(terrain, field)?;
                let from = node_id(&field.graph, from)?;
                let to = node_id(&field.graph, to)?;
                field.graph.connect(from, to, *pin).map_err(refusal)?;
                Ok(json!({ "from": node_path(from), "to": node_path(to), "pin": pin }))
            }

            Self::Disconnect { field, node, pin } => {
                let field = field_mut(terrain, field)?;
                let id = node_id(&field.graph, node)?;
                field.graph.disconnect(id, *pin).map_err(refusal)?;
                Ok(json!({ "node": node_path(id), "pin": pin }))
            }

            Self::Bypass {
                field,
                node,
                bypassed,
            } => {
                let field = field_mut(terrain, field)?;
                let id = node_id(&field.graph, node)?;
                let now = bypassed
                    .unwrap_or_else(|| !field.graph.node(id).expect("resolved above").bypassed);
                field.graph.set_bypassed(id, now).map_err(refusal)?;
                Ok(json!({ "node": node_path(id), "bypassed": now }))
            }

            Self::PlaceNode {
                field,
                node,
                position,
            } => {
                let field = field_mut(terrain, field)?;
                let id = node_id(&field.graph, node)?;
                field.graph.place(id, *position).map_err(refusal)?;
                Ok(json!({ "node": node_path(id), "position": position }))
            }

            Self::RenameNode { field, node, name } => {
                let field = field_mut(terrain, field)?;
                let id = node_id(&field.graph, node)?;
                field.graph.rename(id, name.clone()).map_err(refusal)?;
                Ok(json!({ "node": node_path(id), "name": name }))
            }

            Self::SetOutput { field, node } => {
                let field = field_mut(terrain, field)?;
                let id = match node {
                    Some(node) => Some(node_id(&field.graph, node)?),
                    None => None,
                };
                field.graph.set_output(id).map_err(refusal)?;
                Ok(json!({ "output": id.map(node_path) }))
            }

            Self::Set { path, words } => set(terrain, path, words),
        }
    }
}

fn refusal(error: crate::terrain::graph::GraphError) -> String {
    error.to_string()
}

/// How a control path names this node when it has no name of its own.
pub fn node_path(id: NodeId) -> String {
    format!("n{}", id.0)
}

fn node_id(graph: &FieldGraph, word: &str) -> Result<NodeId, String> {
    if let Some(digits) = word.strip_prefix('n')
        && let Ok(value) = digits.parse::<u32>()
    {
        let id = NodeId(value);
        return match graph.node(id) {
            Some(_) => Ok(id),
            None => Err(format!("this field has no node `{word}`")),
        };
    }
    graph
        .nodes
        .iter()
        .find(|node| node.name.as_deref() == Some(word))
        .map(|node| node.id)
        .ok_or_else(|| format!("this field has no node called `{word}`"))
}

fn field_mut<'a>(terrain: &'a mut TerrainSpec, name: &str) -> Result<&'a mut Field, String> {
    terrain
        .field_mut(name)
        .ok_or_else(|| format!("no field named `{name}`"))
}

fn set(terrain: &mut TerrainSpec, path: &str, words: &[String]) -> Result<Value, String> {
    let parts: Vec<&str> = path.split('.').collect();
    let name = *parts.first().ok_or("a path needs a field name")?;
    if parts.len() < 2 {
        return Err(format!("`{path}` names a field and nothing on it"));
    }
    if parts.len() == 2 {
        return set_field(terrain, name, &parts[1..], words);
    }

    let field = field_mut(terrain, name)?;
    let id = node_id(&field.graph, parts[1])?;
    let property = parts[2];
    if property == "op" {
        return match parts.get(3) {
            None => {
                let op = parse_op(words)?;
                let node = field.graph.node_mut(id).expect("resolved above");
                node.inputs.resize(op.arity(), None);
                node.op = op;
                Ok(json!({ "op": op_summary(&node.op) }))
            }
            Some(property) => {
                let node = field.graph.node_mut(id).expect("resolved above");
                set_op(&mut node.op, property, words)?;
                Ok(json!({ "op": op_summary(&node.op) }))
            }
        };
    }
    match property {
        "bypassed" => {
            let bypassed = boolean(first(words)?)?;
            field.graph.set_bypassed(id, bypassed).map_err(refusal)?;
            Ok(json!({ "bypassed": bypassed }))
        }
        "name" => {
            let word = first(words)?;
            let name = (word != "none").then(|| word.clone());
            field.graph.rename(id, name.clone()).map_err(refusal)?;
            Ok(json!({ "name": name }))
        }
        property => {
            let node = field.graph.node_mut(id).expect("resolved above");
            set_op(&mut node.op, property, words)?;
            Ok(json!({ "op": op_summary(&node.op) }))
        }
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
        Some("output") => {
            let word = first(words)?;
            let id = (word != "none")
                .then(|| node_id(&field.graph, word))
                .transpose()?;
            field.graph.set_output(id).map_err(refusal)?;
            Ok(json!({ "output": id.map(node_path) }))
        }
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

fn set_op(op: &mut NodeOp, property: &str, words: &[String]) -> Result<(), String> {
    match (op, property) {
        (NodeOp::Constant(value), "value") => *value = number(first(words)?)?,

        (NodeOp::Noise(spec), "kind") => spec.kind = parse_noise_kind(first(words)?)?,
        (NodeOp::Noise(spec), "scale") => spec.scale = number(first(words)?)?,
        (NodeOp::Noise(spec), "octaves") => spec.octaves = number(first(words)?)?,
        (NodeOp::Noise(spec), "seed") => spec.seed = number(first(words)?)?,
        (NodeOp::Noise(spec), "strike") => {
            spec.transform.strike_degrees = number(first(words)?)?;
        }
        (NodeOp::Noise(spec), "aspect") => spec.transform.aspect = number(first(words)?)?,
        (NodeOp::Noise(spec), "warp") => spec.warp = parse_warp(words)?,

        (NodeOp::Slope { sample_tiles, .. }, "sample_tiles") => {
            *sample_tiles = number(first(words)?)?;
        }
        (NodeOp::Slope { mode, .. }, "mode") => *mode = parse_slope_mode(first(words)?)?,

        (NodeOp::FieldRef(id), "field") => *id = first(words)?.as_str().into(),

        (NodeOp::Regions { output, .. }, "output") => *output = parse_region_output(first(words)?),
        (NodeOp::Regions { spec, .. }, "seed") => spec.seed = number(first(words)?)?,
        (NodeOp::Regions { spec, .. }, "cell_tiles") => spec.cell_tiles = number(first(words)?)?,
        (NodeOp::Regions { spec, .. }, "blend_tiles") => spec.blend_tiles = number(first(words)?)?,
        (NodeOp::Regions { spec, .. }, "warp") => spec.warp = parse_warp(words)?,

        (NodeOp::Binary(binary), "mode") => *binary = parse_binary(first(words)?)?,
        (NodeOp::Scale(factor), "factor") => *factor = number(first(words)?)?,

        (NodeOp::Remap(remap), "from") => {
            remap.from = (
                number(first(words)?)?,
                number(words.get(1).ok_or("a band needs two numbers")?)?,
            );
        }
        (NodeOp::Remap(remap), "to") => {
            remap.to = (
                number(first(words)?)?,
                number(words.get(1).ok_or("a band needs two numbers")?)?,
            );
        }

        (NodeOp::Curve(curve), "points") => *curve = parse_curve(words)?,

        (op, other) => {
            return Err(format!("a {} node has nothing called `{other}`", op_name(op)));
        }
    }
    Ok(())
}

/// A region table is not a command line, so `regions` is deliberately absent: an existing
/// one is edited through `op.output` and the rest of `op.*`, and a new one comes from a
/// preset or a file.
pub fn parse_op(words: &[String]) -> Result<NodeOp, String> {
    let kind = first(words)?;
    let rest = &words[1..];
    match kind.as_str() {
        "constant" => Ok(NodeOp::Constant(number(first(rest)?)?)),
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
            Ok(NodeOp::Noise(spec))
        }
        "fieldref" => Ok(NodeOp::FieldRef(first(rest)?.as_str().into())),
        "slope" => Ok(NodeOp::Slope {
            sample_tiles: number(first(rest).map_err(|_| "a slope op needs a sample distance")?)?,
            mode: match rest.get(1) {
                Some(mode) => parse_slope_mode(mode)?,
                None => SlopeMode::default(),
            },
        }),
        "paint" => Ok(NodeOp::Paint(Raster::default())),
        "shader" => Ok(NodeOp::Shader(ShaderLayer::new(first(rest)?.as_str()))),
        "binary" => Ok(NodeOp::Binary(parse_binary(first(rest)?)?)),
        "lerp" => Ok(NodeOp::Lerp),
        "scale" => Ok(NodeOp::Scale(number(first(rest)?)?)),
        "remap" => {
            if rest.len() < 4 {
                return Err("a remap op needs two bands, as four numbers".to_owned());
            }
            Ok(NodeOp::Remap(Remap::new(
                (number(&rest[0])?, number(&rest[1])?),
                (number(&rest[2])?, number(&rest[3])?),
            )))
        }
        "curve" => Ok(NodeOp::Curve(parse_curve(rest)?)),
        other => Err(format!("no node op called `{other}`")),
    }
}

fn parse_curve(words: &[String]) -> Result<Curve, String> {
    if !words.len().is_multiple_of(2) {
        return Err("a curve is a list of input and output pairs".to_owned());
    }
    let mut points = Vec::with_capacity(words.len() / 2);
    for pair in words.chunks_exact(2) {
        points.push(CurvePoint {
            input: number(&pair[0])?,
            output: number(&pair[1])?,
        });
    }
    Ok(Curve::new(points))
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

/// A change to one of the brush's settings, named and read the way a node's
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

/// Every way a binary node combines its inputs, in the order the panel offers them.
pub const BINARIES: [Binary; 4] = [Binary::Add, Binary::Mul, Binary::Max, Binary::Min];

/// The word this binary mode is named by, in the panel and on the command line.
pub fn binary_name(binary: Binary) -> &'static str {
    match binary {
        Binary::Add => "add",
        Binary::Mul => "mul",
        Binary::Max => "max",
        Binary::Min => "min",
    }
}

fn parse_binary(word: &str) -> Result<Binary, String> {
    BINARIES
        .into_iter()
        .find(|binary| binary_name(*binary) == word)
        .ok_or_else(|| format!("no binary mode called `{word}`"))
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
pub fn op_name(op: &NodeOp) -> &'static str {
    match op {
        NodeOp::Constant(_) => "constant",
        NodeOp::Noise(_) => "noise",
        NodeOp::Paint(_) => "paint",
        NodeOp::Slope { .. } => "slope",
        NodeOp::FieldRef(_) => "fieldref",
        NodeOp::Regions { .. } => "regions",
        NodeOp::External(_) => "external",
        NodeOp::Shader(_) => "shader",
        NodeOp::Binary(_) => "binary",
        NodeOp::Lerp => "lerp",
        NodeOp::Scale(_) => "scale",
        NodeOp::Remap(_) => "remap",
        NodeOp::Curve(_) => "curve",
    }
}

/// One line describing an op and its parameters, for the inspector's collapsed row
/// and the control client's listing. Not a path, and nothing reads it back.
pub fn op_summary(op: &NodeOp) -> String {
    match op {
        NodeOp::Constant(value) => format!("constant {value}"),
        NodeOp::Noise(spec) => format!(
            "{} scale {} x{}",
            noise_kind_name(spec.kind),
            spec.scale,
            spec.octaves
        ),
        NodeOp::Paint(raster) => format!("paint {}x{}", raster.width(), raster.height()),
        NodeOp::Slope { sample_tiles, mode } => {
            format!("slope over {sample_tiles} by {}", slope_mode_name(*mode))
        }
        NodeOp::FieldRef(id) => format!("fieldref {id}"),
        NodeOp::Regions { output, .. } => format!("regions {}", region_output_name(output)),
        NodeOp::External(raster) => format!("external {}x{}", raster.width(), raster.height()),
        NodeOp::Shader(shader) => format!("shader {}", shader.file),
        NodeOp::Binary(binary) => format!("binary {}", binary_name(*binary)),
        NodeOp::Lerp => "lerp".to_owned(),
        NodeOp::Scale(factor) => format!("scale {factor}"),
        NodeOp::Remap(remap) => format!(
            "remap {}..{} -> {}..{}",
            remap.from.0, remap.from.1, remap.to.0, remap.to.1
        ),
        NodeOp::Curve(curve) => format!("curve of {} points", curve.points.len()),
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
    use crate::terrain::WaterSpec;
    use bevy::math::UVec2;
    use watershed::FieldId;

    fn document() -> TerrainSpec {
        TerrainSpec::new(UVec2::new(64, 64))
            .with_field(Field::new("base").with_op(NodeOp::Constant(0.25)))
            .with_field(
                Field::new("height")
                    .with_role(FieldRole::Height)
                    .with_sum([
                        NodeOp::FieldRef(FieldId::from("base")),
                        NodeOp::Noise(NoiseSpec::new(1, NoiseKind::Fbm, 0.02)),
                    ]),
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

    // The structural verbs against one graph, in sequence, because each is addressed
    // by an id the one before it may have changed — and an id that moved would send
    // the panel's next edit to a different node.
    #[test]
    fn a_node_can_be_added_wired_bypassed_and_removed() {
        let mut terrain = document();
        let before = terrain.field("height").unwrap().graph.nodes.len();

        Edit::AddNode {
            field: "height".to_owned(),
            op: NodeOp::Scale(2.0),
            position: None,
        }
        .apply(&mut terrain)
        .unwrap();
        let graph = &terrain.field("height").unwrap().graph;
        assert_eq!(graph.nodes.len(), before + 1);
        let added = graph.nodes.last().unwrap().id;
        let output = graph.output.unwrap();

        Edit::Connect {
            field: "height".to_owned(),
            from: output.to_string(),
            to: added.to_string(),
            pin: 0,
        }
        .apply(&mut terrain)
        .unwrap();
        assert_eq!(
            terrain.field("height").unwrap().graph.node(added).unwrap().inputs,
            vec![Some(output)]
        );

        Edit::Bypass {
            field: "height".to_owned(),
            node: added.to_string(),
            bypassed: None,
        }
        .apply(&mut terrain)
        .unwrap();
        assert!(terrain.field("height").unwrap().graph.node(added).unwrap().bypassed);

        Edit::RemoveNode {
            field: "height".to_owned(),
            node: added.to_string(),
        }
        .apply(&mut terrain)
        .unwrap();
        assert_eq!(terrain.field("height").unwrap().graph.nodes.len(), before);
    }

    // An edge that would make the graph feed itself has to be refused where it is
    // drawn, not left for the bake to reject — the document must never hold one.
    #[test]
    fn an_edge_that_would_close_a_cycle_is_refused() {
        let mut terrain = document();
        let graph = &terrain.field("height").unwrap().graph;
        let output = graph.output.unwrap();
        let source = graph.nodes[0].id;

        assert!(
            Edit::Connect {
                field: "height".to_owned(),
                from: output.to_string(),
                to: source.to_string(),
                pin: 0,
            }
            .apply(&mut terrain)
            .is_err()
        );
    }

    // The assertion the scenario exists for, made where it can be made numerically: the
    // editor's whole claim is that editing the graph changes the field it bakes.
    #[test]
    fn a_node_wired_into_the_output_moves_the_bake_it_produces() {
        let mut terrain = document();
        terrain.bake_in_place().unwrap();
        let before = terrain.field("height").unwrap().baked().data().to_vec();

        let previous = terrain.field("height").unwrap().graph.output.unwrap();
        let added = {
            let graph = &mut terrain.field_mut("height").unwrap().graph;
            let value = graph.node_with(NodeOp::Constant(0.25), &[]);
            let total = graph.node_with(NodeOp::Binary(Binary::Add), &[previous, value]);
            graph.set_output(Some(total)).unwrap();
            total
        };
        terrain.bake_in_place().unwrap();
        let after = terrain.field("height").unwrap().baked().data().to_vec();

        assert_ne!(before, after);
        assert!(
            before
                .iter()
                .zip(&after)
                .all(|(before, after)| after >= before),
            "a texel fell after a value was added onto the output"
        );

        Edit::SetOutput {
            field: "height".to_owned(),
            node: Some(previous.to_string()),
        }
        .apply(&mut terrain)
        .unwrap();
        terrain.bake_in_place().unwrap();
        assert_eq!(terrain.field("height").unwrap().baked().data(), &before[..]);
        assert!(terrain.field("height").unwrap().graph.node(added).is_some());
    }

    // Bypassing a node has to be exactly as if it passed its input straight through,
    // since that is what the panel's checkbox promises.
    #[test]
    fn bypassing_a_node_reads_what_it_passes_through() {
        let mut terrain = document();
        let output = terrain.field("height").unwrap().graph.output.unwrap();
        let scaled = {
            let graph = &mut terrain.field_mut("height").unwrap().graph;
            let scaled = graph.node_with(NodeOp::Scale(3.0), &[output]);
            graph.set_output(Some(scaled)).unwrap();
            scaled
        };
        terrain.bake_in_place().unwrap();
        let tripled = terrain.field("height").unwrap().baked().data().to_vec();

        Edit::Bypass {
            field: "height".to_owned(),
            node: scaled.to_string(),
            bypassed: Some(true),
        }
        .apply(&mut terrain)
        .unwrap();
        terrain.bake_in_place().unwrap();
        let passed = terrain.field("height").unwrap().baked().data().to_vec();

        assert_ne!(tripled, passed);
    }

    // Every one of these arrives from a caller working against a document that has
    // changed under it, so each has to be a message rather than a panic or a silent
    // no-op that looks like the edit was applied.
    #[test]
    fn an_edit_naming_something_the_document_does_not_have_is_refused() {
        let mut terrain = document();
        assert!(
            Edit::AddNode {
                field: "nowhere".to_owned(),
                op: NodeOp::Constant(0.5),
                position: None,
            }
            .apply(&mut terrain)
            .is_err()
        );
        assert!(
            Edit::RemoveNode {
                field: "height".to_owned(),
                node: "n9".to_owned(),
            }
            .apply(&mut terrain)
            .is_err()
        );
        assert!(set_line(&mut terrain, "height.n0.sideways 1").is_err());
        assert!(set_line(&mut terrain, "height.n0.op.scale 1").is_err());
        assert!(set_line(&mut terrain, "height").is_err());
    }

    // The path grammar is the whole surface the control client edits through, so a
    // property nothing can address is a control the panel has and a script cannot use.
    #[test]
    fn every_node_property_is_reachable_by_its_path() {
        let mut terrain = document();
        let sum = terrain
            .field("height")
            .unwrap()
            .graph
            .output
            .expect("the fixture reads a node");

        set_line(&mut terrain, &format!("height.{sum}.bypassed on")).unwrap();
        set_line(&mut terrain, &format!("height.{sum}.name total")).unwrap();
        set_line(&mut terrain, "height.total.mode mul").unwrap();

        let node = terrain.field("height").unwrap().graph.node(sum).unwrap();
        assert!(node.bypassed);
        assert_eq!(node.name.as_deref(), Some("total"));
        assert_eq!(node.op, NodeOp::Binary(Binary::Mul));
    }

    // A node is addressable by its id or by the name it was given, and the two have to
    // reach the same node — otherwise a script and the panel would edit different ones.
    #[test]
    fn a_node_is_reachable_by_its_id_and_by_its_name() {
        let mut terrain = document();
        let id = terrain.field("base").unwrap().graph.output.unwrap();

        set_line(&mut terrain, &format!("base.{id}.name ground")).unwrap();
        set_line(&mut terrain, "base.ground.value 0.75").unwrap();

        let node = terrain.field("base").unwrap().graph.node(id).unwrap();
        assert_eq!(node.op, NodeOp::Constant(0.75));
    }

    // The reason op parameters are addressable one at a time: the seed here comes from
    // the document rather than from any of the three edits, where rewriting the op
    // wholesale would have to restate every parameter and would silently reset the ones
    // it forgot.
    #[test]
    fn an_op_parameter_can_be_moved_without_rewriting_the_op_around_it() {
        let mut terrain = document();
        let noise = terrain
            .field("height")
            .unwrap()
            .graph
            .nodes
            .iter()
            .find(|node| matches!(node.op, NodeOp::Noise(_)))
            .map(|node| node.id)
            .expect("the fixture carries a noise node");
        set_line(&mut terrain, &format!("height.{noise}.op.scale 0.004")).unwrap();
        set_line(&mut terrain, &format!("height.{noise}.op.octaves 6")).unwrap();
        set_line(&mut terrain, &format!("height.{noise}.op.kind ridged")).unwrap();

        let NodeOp::Noise(spec) = &terrain.field("height").unwrap().graph.node(noise).unwrap().op
        else {
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

        let noise = terrain
            .field("height")
            .unwrap()
            .graph
            .nodes
            .iter()
            .find(|node| matches!(node.op, NodeOp::Noise(_)))
            .map(|node| node.id)
            .expect("the fixture carries a noise node");
        Edit::Set {
            path: format!("height.{noise}.op.scale"),
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
            ("slope 4", "slope"),
            ("binary mul", "binary"),
            ("lerp", "lerp"),
            ("scale 2", "scale"),
            ("remap 0 1 0 2", "remap"),
            ("curve 0 0 1 2", "curve"),
        ] {
            let op = parse_op(&words(line)).unwrap_or_else(|error| panic!("{line}: {error}"));
            assert_eq!(op_name(&op), name, "{line}");
        }
        assert!(parse_op(&words("regions")).is_err());
        assert!(parse_op(&words("noise sideways 0.01")).is_err());
        assert!(parse_op(&words("noise fbm")).is_err());
        assert!(parse_op(&words("binary sideways")).is_err());
        assert!(parse_op(&words("remap 0 1")).is_err());
        assert!(parse_op(&words("curve 0 0 1")).is_err());
    }

    // Naming and parsing are written out separately for each enum, so nothing but this
    // forces them to agree; a name that does not parse back makes a value the panel can
    // display and no script can set.
    #[test]
    fn every_binary_mode_and_noise_kind_parses_back_from_the_name_it_prints() {
        for binary in BINARIES {
            assert_eq!(parse_binary(binary_name(binary)).unwrap(), binary);
        }
        for kind in NOISE_KINDS {
            assert_eq!(parse_noise_kind(noise_kind_name(kind)).unwrap(), kind);
        }
        assert!(parse_binary("sideways").is_err());
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
