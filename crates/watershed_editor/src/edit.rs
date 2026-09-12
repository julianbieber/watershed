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
use watershed::FieldId;
use watershed::FieldRole;
use watershed::raster::Raster;

/// The place in a document a change writes, for deciding whether a later change
/// makes an earlier one pointless.
///
/// It exists only so that a stream of values aimed at one control costs one held
/// change rather than a queue: two changes with equal slots are the same place written
/// twice, and only the last of them has to land. A change that overwrites nothing in
/// particular is [`Slot::Once`] and is never dropped for another.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Slot {
    /// Writes nothing a later change can make pointless — a node added, an edge
    /// wired, a node removed. Never dropped, however many pile up.
    Once,
    /// Writes the property at this dotted path: the one [`Edit::Set`] names, or one
    /// built in the same shape for a change that always writes the same property of a
    /// node.
    Path(String),
    /// Writes a control in the field panel, named by the property it edits rather
    /// than by a path, because a panel binding does not build one.
    Control {
        /// What the control edits, distinct per binding.
        property: &'static str,
        /// The node it belongs to, or `None` for a field-level control.
        node: Option<NodeId>,
        /// Which of a property's several numbers, or `[0, 0]` when it has one.
        index: [usize; 2],
    },
}

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
    /// Adds an empty field to the document and leaves every other field alone.
    AddField {
        /// The name the field is addressed by. Surrounding whitespace is trimmed.
        /// Refused when what is left is blank, or when the document already has a
        /// field of that name.
        name: String,
    },
    /// Renames a field and rewrites everything that named the old name: every
    /// [`NodeOp::FieldRef`] in the document, and the water spec's height or moisture
    /// field.
    RenameField {
        /// The field to rename. Must exist. Surrounding whitespace is trimmed.
        from: String,
        /// What to call it. Trimmed too; refused when what is left is blank, or when
        /// the document already has a field of that name.
        to: String,
    },
    /// Takes a field out of the document, with its graph and its bake.
    RemoveField {
        /// The field to remove. Must exist, must be declared read by no other field,
        /// and must not be named by the water spec.
        name: String,
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
    /// The place this edit writes, for [`Slot`]'s purpose.
    ///
    /// Only the edits that write one place over and over name it: a property, a card's
    /// position, a node's name. Everything else is [`Slot::Once`], so two of them held
    /// together both land.
    pub fn slot(&self) -> Slot {
        match self {
            Self::Set { path, .. } => Slot::Path(path.clone()),
            Self::PlaceNode { field, node, .. } => Slot::Path(format!("{field}.{node}.position")),
            Self::RenameNode { field, node, .. } => Slot::Path(format!("{field}.{node}.name")),
            _ => Slot::Once,
        }
    }

    /// Whether what this edit changes is read by a bake.
    ///
    /// A node's position and its name are authoring data: they are written to the
    /// document and saved with it, but nothing that evaluates a texel reads either. An
    /// edit that touches only those must not make the bake stale, or dragging a card
    /// would throw away the whole field — and the solved water with it.
    ///
    /// A field's display properties are the same kind of thing: they say how the map
    /// draws the field, not what the field holds, so a `Set` on one of them is exempt
    /// too. An overlay added later adds its properties to that list rather than
    /// replacing it.
    ///
    /// A field that has just been added is exempt for a different reason: it holds no
    /// raster and nothing reads it, so no texel of any field already baked changes
    /// value. Were it not exempt, adding a field would discard every bake in the
    /// document.
    ///
    /// A rename changes no value either, and is still not exempt.
    /// [`Snapshot::restore`](crate::history::Snapshot::restore) matches a held field to
    /// the live document by name, so undoing a rename puts the field back under its old
    /// name with no bake to give it; the document has to be re-baked from both sides of
    /// the change for that field to hold values again. A removal has the same hole.
    pub fn reaches_the_bake(&self) -> bool {
        match self {
            Self::PlaceNode { .. } | Self::RenameNode { .. } | Self::AddField { .. } => false,
            Self::Set { path, .. } => !is_display_property(path),
            _ => true,
        }
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
                if let Some(referenced) = op.dependency() {
                    check_field_ref(terrain, field, referenced)?;
                }
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

            Self::AddField { name } => {
                let name = name.trim();
                if name.is_empty() {
                    return Err("a field needs a name".to_owned());
                }
                if terrain.field(name).is_some() {
                    return Err(format!("this document already has a field named `{name}`"));
                }
                terrain.fields.push(Field::new(name));
                Ok(json!({ "added": name, "fields": terrain.fields.len() }))
            }

            Self::RenameField { from, to } => rename_field(terrain, from.trim(), to.trim()),

            Self::RemoveField { name } => remove_field(terrain, name.trim()),

            Self::Set { path, words } => set(terrain, path, words),
        }
    }

    /// The field this edit leaves on screen, or `None` for an edit that leaves the
    /// view where it was.
    ///
    /// [`Document::apply`](crate::document::Document::apply) reads this and puts the
    /// field on screen as part of the same change, so undoing the edit puts the
    /// previously shown field back in the same step.
    ///
    /// `terrain` is the document *after* the edit applied and `active` the field that
    /// was on screen before it. A rename and a removal answer only when it was the
    /// shown field they changed; a removal then answers the first field left in the
    /// document, and `None` when none is left.
    pub fn shows(&self, terrain: &TerrainSpec, active: &str) -> Option<String> {
        match self {
            Self::AddField { name } => Some(name.trim().to_owned()),
            Self::RenameField { from, to } => (active == from.trim()).then(|| to.trim().to_owned()),
            Self::RemoveField { name } => (active == name.trim())
                .then(|| terrain.fields.first().map(|field| field.id.to_string()))
                .flatten(),
            _ => None,
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

/// Refuses a `FieldRef` in `owner`'s graph that names `referenced`, when the document
/// has no such field or when reading it would make the fields read each other in a
/// circle.
///
/// Every path that writes a reference's field name goes through here — the `node` and
/// `set` verbs, the panel's op menu, the panel's add row — so a cycle or a dangling
/// name cannot reach a bake from an edit made in the editor. A cycle is reported as the
/// chain `owner -> ... -> owner`, the spelling
/// [`PlanError::Cycle`](crate::terrain::bake::PlanError::Cycle) uses. A document loaded
/// with a cycle already in it is not this function's business and still fails at the
/// bake.
pub fn check_field_ref(
    terrain: &TerrainSpec,
    owner: &str,
    referenced: &FieldId,
) -> Result<(), String> {
    if terrain.field(referenced.as_str()).is_none() {
        return Err(format!(
            "field `{referenced}`, read by `{owner}`, is not in the document"
        ));
    }
    let Some(chain) = field_cycle(terrain, owner, referenced) else {
        return Ok(());
    };
    let chain = std::iter::once(owner.to_owned())
        .chain(chain)
        .collect::<Vec<_>>()
        .join(" -> ");
    Err(format!(
        "reading `{referenced}` from `{owner}` makes the fields depend on each other in a cycle: {chain}"
    ))
}

fn field_cycle(terrain: &TerrainSpec, owner: &str, referenced: &FieldId) -> Option<Vec<String>> {
    let mut seen = Vec::new();
    let mut path = Vec::new();
    reaches(terrain, owner, referenced.as_str(), &mut seen, &mut path).then_some(path)
}

fn reaches(
    terrain: &TerrainSpec,
    owner: &str,
    current: &str,
    seen: &mut Vec<String>,
    path: &mut Vec<String>,
) -> bool {
    if seen.iter().any(|name| name == current) {
        return false;
    }
    seen.push(current.to_owned());
    path.push(current.to_owned());
    if current == owner {
        return true;
    }
    if let Some(field) = terrain.field(current) {
        for read in field.declared_reads() {
            if reaches(terrain, owner, read.as_str(), seen, path) {
                return true;
            }
        }
    }
    path.pop();
    false
}

/// The fields that declare a read of `name`, in declaration order, each named once.
///
/// The same relation [`check_field_ref`] walks, read from the other end. A reference
/// under a bypassed node does not count, which is the rule
/// [`Field::declared_reads`](crate::terrain::Field::declared_reads) already applies.
/// The answer is derived from the document on every call rather than cached, so it
/// cannot fall out of step with an edit.
pub fn readers_of(terrain: &TerrainSpec, name: &str) -> Vec<String> {
    terrain
        .fields
        .iter()
        .filter(|field| field.id.as_str() != name)
        .filter(|field| field.declared_reads().any(|id| id.as_str() == name))
        .map(|field| field.id.to_string())
        .collect()
}

/// The fields `field` declares a read of, in declaration order, each named once.
///
/// [`readers_of`] read from the other end, and the two answer about one relation: a
/// field's own name is never in the list, and a bypassed node contributes nothing,
/// because [`Field::declared_reads`](crate::terrain::Field::declared_reads) already
/// leaves it out. Duplicates are dropped keeping the first occurrence — a graph may
/// name the same field from several nodes, and a caller listing what a field reads
/// wants the field once.
pub fn reads_of(field: &Field) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for read in field.declared_reads() {
        if read.as_str() == field.id.as_str() {
            continue;
        }
        if !names.iter().any(|seen| seen == read.as_str()) {
            names.push(read.to_string());
        }
    }
    names
}

fn rename_field(terrain: &mut TerrainSpec, from: &str, to: &str) -> Result<Value, String> {
    if to.is_empty() {
        return Err("a field needs a name".to_owned());
    }
    if terrain.field(from).is_none() {
        return Err(format!("no field named `{from}`"));
    }
    if terrain.field(to).is_some() {
        return Err(format!("this document already has a field named `{to}`"));
    }

    let mut references = 0usize;
    for field in &mut terrain.fields {
        if field.id.as_str() == from {
            field.id = FieldId::from(to);
        }
        for node in &mut field.graph.nodes {
            if let NodeOp::FieldRef(id) = &mut node.op
                && id.as_str() == from
            {
                *id = FieldId::from(to);
                references += 1;
            }
        }
    }

    let mut water = false;
    if let Some(spec) = &mut terrain.water_spec {
        if spec.height.as_str() == from {
            spec.height = FieldId::from(to);
            water = true;
        }
        if spec.moisture.as_ref().is_some_and(|id| id.as_str() == from) {
            spec.moisture = Some(FieldId::from(to));
            water = true;
        }
    }

    Ok(json!({ "renamed": from, "to": to, "references": references, "water": water }))
}

fn remove_field(terrain: &mut TerrainSpec, name: &str) -> Result<Value, String> {
    if terrain.field(name).is_none() {
        return Err(format!("no field named `{name}`"));
    }

    let readers = readers_of(terrain, name);
    if !readers.is_empty() {
        let list = readers
            .iter()
            .map(|reader| format!("`{reader}`"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "`{name}` is read by {list} — take those references out first"
        ));
    }

    if terrain.water_spec.as_ref().is_some_and(|spec| {
        spec.height.as_str() == name || spec.moisture.as_ref().is_some_and(|id| id.as_str() == name)
    }) {
        return Err(format!(
            "`{name}` is named by the water spec of this terrain — reset the water first"
        ));
    }

    terrain.fields.retain(|field| field.id.as_str() != name);
    Ok(json!({ "removed": name, "fields": terrain.fields.len() }))
}

fn field_ref_written(
    field: &Field,
    id: NodeId,
    property: &str,
    nested: Option<&str>,
    words: &[String],
) -> Result<Option<FieldId>, String> {
    match (property, nested) {
        ("op", None) => Ok(parse_op(words)?.dependency().cloned()),
        ("op", Some("field")) | ("field", None) => {
            match field.graph.node(id).map(|node| &node.op) {
                Some(NodeOp::FieldRef(_)) => Ok(Some(first(words)?.as_str().into())),
                _ => Ok(None),
            }
        }
        _ => Ok(None),
    }
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

    let field = terrain
        .field(name)
        .ok_or_else(|| format!("no field named `{name}`"))?;
    let id = node_id(&field.graph, parts[1])?;
    let property = parts[2];
    if let Some(referenced) = field_ref_written(field, id, property, parts.get(3).copied(), words)?
    {
        check_field_ref(terrain, name, &referenced)?;
    }
    let field = field_mut(terrain, name)?;
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
        Some("hillshade") => {
            let on = boolean(first(words)?)?;
            field.hillshade = on;
            Ok(json!({ "hillshade": on }))
        }
        Some("light_azimuth") => {
            let degrees: f32 = number(first(words)?)?;
            field.light_azimuth = degrees;
            Ok(json!({ "light_azimuth": degrees }))
        }
        Some("contours") => {
            let on = boolean(first(words)?)?;
            field.contours = on;
            Ok(json!({ "contours": on }))
        }
        Some("contour_interval") => {
            let spacing: f32 = number(first(words)?)?;
            if !(spacing >= MIN_CONTOUR_INTERVAL) {
                return Err(format!(
                    "a contour interval has to be at least {MIN_CONTOUR_INTERVAL}"
                ));
            }
            field.contour_interval = spacing;
            Ok(json!({ "contour_interval": spacing }))
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
            return Err(format!(
                "a {} node has nothing called `{other}`",
                op_name(op)
            ));
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

/// What an op is set to, without the op's own name: the parameter half of
/// [`op_summary`], and what a node's card shows on its own line under the op name.
///
/// Empty for an op that carries no parameters.
pub fn op_params(op: &NodeOp) -> String {
    match op {
        NodeOp::Constant(value) => format!("{value}"),
        NodeOp::Noise(spec) => format!(
            "{} scale {} x{}",
            noise_kind_name(spec.kind),
            spec.scale,
            spec.octaves
        ),
        NodeOp::Paint(raster) => format!("{}x{}", raster.width(), raster.height()),
        NodeOp::Slope { sample_tiles, mode } => {
            format!("over {sample_tiles} by {}", slope_mode_name(*mode))
        }
        NodeOp::FieldRef(id) => format!("{id}"),
        NodeOp::Regions { output, .. } => region_output_name(output),
        NodeOp::External(raster) => format!("{}x{}", raster.width(), raster.height()),
        NodeOp::Shader(shader) => shader.params_line(None),
        NodeOp::Binary(binary) => binary_name(*binary).to_owned(),
        NodeOp::Lerp => String::new(),
        NodeOp::Scale(factor) => format!("{factor}"),
        NodeOp::Remap(remap) => format!(
            "{}..{} -> {}..{}",
            remap.from.0, remap.from.1, remap.to.0, remap.to.1
        ),
        NodeOp::Curve(curve) => format!("of {} points", curve.points.len()),
    }
}

/// One line describing an op and its parameters, for the inspector's collapsed row
/// and the control client's listing. Not a path, and nothing reads it back.
pub fn op_summary(op: &NodeOp) -> String {
    match op {
        NodeOp::Noise(_) => op_params(op),
        NodeOp::Lerp => "lerp".to_owned(),
        NodeOp::Shader(shader) => format!("shader {}", shader.file),
        _ => format!("{} {}", op_name(op), op_params(op)),
    }
}

/// The smallest contour interval the map will draw. Below this an `f32` cannot
/// separate one level from the next on a field of order one, so the lines would be
/// noise rather than a reading.
pub const MIN_CONTOUR_INTERVAL: f32 = 1e-6;

const DISPLAY_PROPERTIES: [&str; 4] =
    ["hillshade", "light_azimuth", "contours", "contour_interval"];

fn is_display_property(path: &str) -> bool {
    let parts: Vec<&str> = path.split('.').collect();
    parts.len() == 2 && DISPLAY_PROPERTIES.contains(&parts[1])
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

    fn document() -> TerrainSpec {
        TerrainSpec::new(UVec2::new(64, 64))
            .with_field(Field::new("base").with_op(NodeOp::Constant(0.25)))
            .with_field(Field::new("height").with_role(FieldRole::Height).with_sum([
                NodeOp::FieldRef(FieldId::from("base")),
                NodeOp::Noise(NoiseSpec::new(1, NoiseKind::Fbm, 0.02)),
            ]))
    }

    // `op_summary` is read by the control client, the inspector's collapsed row and
    // this module's own JSON replies, and none of them parse it back — so a change to
    // its wording is invisible until someone reads a listing and cannot find their
    // node. Pinning one op of every variant is what holds the split into `op_params`
    // to the spelling it replaced.
    #[test]
    fn every_op_summarises_to_the_words_it_always_did() {
        use crate::terrain::regions::{Region, RegionSpec};

        let regions = RegionSpec::new(7, 128, 16, ["base"]).with_region(Region::new(4, [0.5]));
        let cases = [
            (NodeOp::Constant(0.25), "constant 0.25"),
            (
                NodeOp::Noise(NoiseSpec::new(1, NoiseKind::Fbm, 0.02).with_octaves(4)),
                "fbm scale 0.02 x4",
            ),
            (
                NodeOp::Paint(Raster::new(UVec2::new(4, 2), 0u8)),
                "paint 4x2",
            ),
            (
                NodeOp::External(Raster::new(UVec2::new(8, 3), 0.0f32)),
                "external 8x3",
            ),
            (
                NodeOp::Slope {
                    sample_tiles: 2.0,
                    mode: SlopeMode::Gradient,
                },
                "slope over 2 by gradient",
            ),
            (NodeOp::FieldRef(FieldId::from("base")), "fieldref base"),
            (
                NodeOp::Regions {
                    spec: regions,
                    output: RegionOutput::Blended("base".to_owned()),
                },
                "regions base",
            ),
            (
                NodeOp::Shader(ShaderLayer::new("warped.wgsl")),
                "shader warped.wgsl",
            ),
            (NodeOp::Binary(Binary::Add), "binary add"),
            (NodeOp::Lerp, "lerp"),
            (NodeOp::Scale(1.5), "scale 1.5"),
            (
                NodeOp::Remap(Remap {
                    from: (0.0, 1.0),
                    to: (2.0, 3.0),
                }),
                "remap 0..1 -> 2..3",
            ),
            (
                NodeOp::Curve(Curve::new(vec![
                    CurvePoint {
                        input: 0.0,
                        output: 0.0,
                    },
                    CurvePoint {
                        input: 1.0,
                        output: 1.0,
                    },
                ])),
                "curve of 2 points",
            ),
        ];
        for (op, expected) in cases {
            assert_eq!(op_summary(&op), expected);
        }
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
            terrain
                .field("height")
                .unwrap()
                .graph
                .node(added)
                .unwrap()
                .inputs,
            vec![Some(output)]
        );

        Edit::Bypass {
            field: "height".to_owned(),
            node: added.to_string(),
            bypassed: None,
        }
        .apply(&mut terrain)
        .unwrap();
        assert!(
            terrain
                .field("height")
                .unwrap()
                .graph
                .node(added)
                .unwrap()
                .bypassed
        );

        Edit::RemoveNode {
            field: "height".to_owned(),
            node: added.to_string(),
        }
        .apply(&mut terrain)
        .unwrap();
        assert_eq!(terrain.field("height").unwrap().graph.nodes.len(), before);
    }

    // Every default the issue names: nothing in the graph, shift 0, the unit range,
    // and last in declaration order so the field menu grows at the end.
    #[test]
    fn a_field_added_from_the_editor_is_empty_at_shift_zero_over_the_unit_range() {
        let mut terrain = document();
        let reply = Edit::AddField {
            name: " biomes ".to_owned(),
        }
        .apply(&mut terrain)
        .expect("a free name is accepted");

        assert_eq!(reply["added"], "biomes");
        let added = terrain.field("biomes").expect("the field was added");
        assert_eq!(added.role, FieldRole::Custom);
        assert_eq!(added.shift, 0);
        assert_eq!(added.range, (0.0, 1.0));
        assert!(added.graph.nodes.is_empty());
        assert_eq!(added.graph.output, None);
        assert_eq!(
            terrain.fields.last().map(|field| field.id.as_str()),
            Some("biomes")
        );
    }

    // A second field of one name would make every path that addresses a field
    // ambiguous, so the name is refused — and the refusal has to leave the document
    // alone, the same rule a refused node edit is held to.
    #[test]
    fn a_field_whose_name_is_already_taken_is_refused_and_changes_nothing() {
        let mut terrain = document();
        let before = terrain.clone();
        let error = Edit::AddField {
            name: "height".to_owned(),
        }
        .apply(&mut terrain)
        .unwrap_err();
        assert!(error.contains("height"), "{error}");
        assert_eq!(terrain, before);
    }

    // A blank name is what an empty name box sends, and a field nothing can address
    // would be unreachable from either surface.
    #[test]
    fn a_field_with_a_blank_name_is_refused() {
        let mut terrain = document();
        for name in ["", "   "] {
            let error = Edit::AddField {
                name: name.to_owned(),
            }
            .apply(&mut terrain)
            .unwrap_err();
            assert!(error.contains("name"), "{error}");
        }
        assert_eq!(terrain.fields.len(), 2);
    }

    // The two properties the acceptance criteria rest on: adding a field throws away
    // no bake, and the field added is the one left on screen.
    #[test]
    fn adding_a_field_reaches_no_bake_and_shows_the_field_it_added() {
        let edit = Edit::AddField {
            name: " biomes ".to_owned(),
        };
        assert!(!edit.reaches_the_bake());
        assert_eq!(edit.shows(&document(), "height"), Some("biomes".to_owned()));
    }

    // Acceptance 1: the rename that the issue names, from the document's side — the
    // field answers to the new name and the reader's `FieldRef` was rewritten with it,
    // so nothing is left naming a name no field carries.
    #[test]
    fn renaming_a_field_rewrites_the_references_that_read_it() {
        let mut terrain = document();
        let reply = Edit::RenameField {
            from: "base".to_owned(),
            to: " continent ".to_owned(),
        }
        .apply(&mut terrain)
        .expect("a free name is accepted");

        assert_eq!(reply["renamed"], "base");
        assert_eq!(reply["to"], "continent");
        assert_eq!(reply["references"], 1);
        assert!(terrain.field("base").is_none());
        assert!(terrain.field("continent").is_some());
        let reads: Vec<String> = terrain
            .field("height")
            .unwrap()
            .declared_reads()
            .map(|id| id.to_string())
            .collect();
        assert_eq!(reads, vec!["continent".to_owned()]);
    }

    // Acceptance 2: the water spec names fields by name too, so a rename that left it
    // behind would point the solve at a field the document no longer has.
    #[test]
    fn renaming_a_field_rewrites_the_water_spec_that_names_it() {
        let mut terrain = document();
        terrain.water_spec = Some(WaterSpec::new("height").with_moisture("base"));

        Edit::RenameField {
            from: "height".to_owned(),
            to: "elevation".to_owned(),
        }
        .apply(&mut terrain)
        .expect("a free name is accepted");
        let reply = Edit::RenameField {
            from: "base".to_owned(),
            to: "continent".to_owned(),
        }
        .apply(&mut terrain)
        .expect("a free name is accepted");

        assert_eq!(reply["water"], true);
        let spec = terrain.water_spec.clone().unwrap();
        assert_eq!(spec.height.as_str(), "elevation");
        assert_eq!(
            spec.moisture.map(|id| id.to_string()),
            Some("continent".to_owned())
        );
    }

    // Acceptance 3: a name already taken would make every path that addresses a field
    // ambiguous, and the refusal has to leave the document alone — the rule a refused
    // `AddField` is held to.
    #[test]
    fn renaming_a_field_onto_a_taken_name_is_refused_and_changes_nothing() {
        let mut terrain = document();
        let before = terrain.clone();
        let error = Edit::RenameField {
            from: "base".to_owned(),
            to: "height".to_owned(),
        }
        .apply(&mut terrain)
        .unwrap_err();
        assert!(error.contains("height"), "{error}");
        assert_eq!(terrain, before);
    }

    // Acceptance 3, the other half: a blank name is what an empty name box sends, and
    // a field nothing can address would be unreachable from either surface.
    #[test]
    fn renaming_a_field_to_a_blank_name_is_refused_and_changes_nothing() {
        let mut terrain = document();
        let before = terrain.clone();
        for name in ["", "   "] {
            let error = Edit::RenameField {
                from: "base".to_owned(),
                to: name.to_owned(),
            }
            .apply(&mut terrain)
            .unwrap_err();
            assert!(error.contains("name"), "{error}");
        }
        assert_eq!(terrain, before);
    }

    // A rename must not manufacture the dangling reference `check_field_ref` exists to
    // refuse, so the document it leaves behind still plans a bake.
    #[test]
    fn a_renamed_document_still_bakes() {
        let mut terrain = document();
        Edit::RenameField {
            from: "base".to_owned(),
            to: "continent".to_owned(),
        }
        .apply(&mut terrain)
        .expect("a free name is accepted");
        assert!(terrain.bake_order().is_ok());
    }

    // Bypass turns a reference off, it does not unname it: a bypassed `FieldRef` left
    // naming the old name would fail at the bake the moment it was un-bypassed.
    #[test]
    fn renaming_a_field_rewrites_a_reference_under_a_bypassed_node() {
        let mut terrain = document();
        let reference = terrain
            .field("height")
            .unwrap()
            .graph
            .nodes
            .iter()
            .find(|node| matches!(node.op, NodeOp::FieldRef(_)))
            .unwrap()
            .id;
        Edit::Bypass {
            field: "height".to_owned(),
            node: node_path(reference),
            bypassed: Some(true),
        }
        .apply(&mut terrain)
        .unwrap();

        let reply = Edit::RenameField {
            from: "base".to_owned(),
            to: "continent".to_owned(),
        }
        .apply(&mut terrain)
        .expect("a free name is accepted");

        assert_eq!(reply["references"], 1);
        let op = &terrain
            .field("height")
            .unwrap()
            .graph
            .node(reference)
            .unwrap()
            .op;
        assert!(matches!(op, NodeOp::FieldRef(id) if id.as_str() == "continent"));
    }

    // Acceptance 4: removing a field something reads would orphan that reference, so
    // it is refused — and the message names the readers, which is the whole of what
    // tells someone what to take out first.
    #[test]
    fn removing_a_field_a_reference_reads_is_refused_and_names_the_reader() {
        let mut terrain = document();
        let before = terrain.clone();
        let error = Edit::RemoveField {
            name: "base".to_owned(),
        }
        .apply(&mut terrain)
        .unwrap_err();
        assert!(error.contains("height"), "{error}");
        assert_eq!(terrain, before);
    }

    // Acceptance 5: the water spec names a field the same way a reference does, and
    // the refusal ends on the clause changing the height role already ends on.
    #[test]
    fn removing_a_field_the_water_spec_names_is_refused() {
        for spec in [
            WaterSpec::new("height"),
            WaterSpec::new("base").with_moisture("height"),
        ] {
            let mut terrain = document();
            terrain.water_spec = Some(spec);
            let before = terrain.clone();
            let error = Edit::RemoveField {
                name: "height".to_owned(),
            }
            .apply(&mut terrain)
            .unwrap_err();
            assert!(error.contains("reset the water"), "{error}");
            assert_eq!(terrain, before);
        }
    }

    // Acceptance 6, the document half: with nothing reading it and no water spec over
    // it, the field goes and every other field is left exactly as it was.
    #[test]
    fn removing_an_unread_field_takes_it_out_and_leaves_the_rest_alone() {
        let mut terrain = document();
        let base = terrain.field("base").unwrap().clone();
        let reply = Edit::RemoveField {
            name: " height ".to_owned(),
        }
        .apply(&mut terrain)
        .expect("nothing reads `height` and no water spec names it");

        assert_eq!(reply["removed"], "height");
        assert_eq!(reply["fields"], 1);
        assert!(terrain.field("height").is_none());
        assert_eq!(terrain.field("base"), Some(&base));
    }

    // `readers_of` is the relation both the removal refusal and issue #41's panel read,
    // so it has to answer the declared readers and only those: not the field itself,
    // and not a reader whose reference is bypassed.
    #[test]
    fn readers_of_names_the_declared_readers_and_nothing_else() {
        let mut terrain = document();
        assert_eq!(readers_of(&terrain, "base"), vec!["height".to_owned()]);
        assert!(readers_of(&terrain, "height").is_empty());

        let reference = terrain
            .field("height")
            .unwrap()
            .graph
            .nodes
            .iter()
            .find(|node| matches!(node.op, NodeOp::FieldRef(_)))
            .unwrap()
            .id;
        Edit::Bypass {
            field: "height".to_owned(),
            node: node_path(reference),
            bypassed: Some(true),
        }
        .apply(&mut terrain)
        .unwrap();
        assert!(readers_of(&terrain, "base").is_empty());
    }

    // The panel's `reads` row and `observe field`'s `reads` list are this one answer,
    // and `ridges`/`height` names `base` from two nodes — so the dedup is the whole
    // point: a field that reads another twice reads it once.
    #[test]
    fn reads_of_names_each_field_read_once() {
        let mut terrain = document();
        let height = terrain.field("height").unwrap();
        assert_eq!(reads_of(height), vec!["base".to_owned()]);
        assert!(reads_of(terrain.field("base").unwrap()).is_empty());

        Edit::AddNode {
            field: "height".to_owned(),
            op: NodeOp::FieldRef(FieldId::from("base")),
            position: None,
        }
        .apply(&mut terrain)
        .expect("a second reference to `base` does not cycle");
        assert_eq!(
            reads_of(terrain.field("height").unwrap()),
            vec!["base".to_owned()]
        );
    }

    // The view has to follow the field it was on: a rename of the shown field keeps it
    // on screen under its new name, a removal falls back to a field that still exists,
    // and an edit to some other field leaves the view alone.
    #[test]
    fn a_rename_and_a_removal_move_the_view_only_when_it_was_on_that_field() {
        let terrain = document();
        let rename = Edit::RenameField {
            from: "base".to_owned(),
            to: "continent".to_owned(),
        };
        assert_eq!(rename.shows(&terrain, "base"), Some("continent".to_owned()));
        assert_eq!(rename.shows(&terrain, "height"), None);

        let mut removed = document();
        let removal = Edit::RemoveField {
            name: "height".to_owned(),
        };
        removal.apply(&mut removed).expect("nothing reads `height`");
        assert_eq!(removal.shows(&removed, "height"), Some("base".to_owned()));
        assert_eq!(removal.shows(&removed, "base"), None);
    }

    // The reference guard has to hold over a field this edit created just as it does
    // over one that came out of a file: `biomes` may read `height`, and `height` may
    // then not read `biomes` back.
    #[test]
    fn a_reference_may_name_a_field_that_was_added_from_the_editor() {
        let mut terrain = document();
        Edit::AddField {
            name: "biomes".to_owned(),
        }
        .apply(&mut terrain)
        .expect("a free name is accepted");

        Edit::AddNode {
            field: "biomes".to_owned(),
            op: NodeOp::FieldRef(FieldId::from("height")),
            position: None,
        }
        .apply(&mut terrain)
        .expect("reading an existing field is allowed");

        let error = Edit::AddNode {
            field: "height".to_owned(),
            op: NodeOp::FieldRef(FieldId::from("biomes")),
            position: None,
        }
        .apply(&mut terrain)
        .unwrap_err();
        assert!(error.contains("height -> biomes -> height"), "{error}");
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

        let NodeOp::Noise(spec) = &terrain
            .field("height")
            .unwrap()
            .graph
            .node(noise)
            .unwrap()
            .op
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

    // The display properties are the only field properties the panel writes that a
    // bake never reads, so both the write and its reply are pinned here, along with the
    // one value an interval refuses.
    #[test]
    fn the_display_properties_are_written_and_reported_back() {
        let mut terrain = document();

        let reply = set_line(&mut terrain, "height.hillshade on").unwrap();
        assert_eq!(reply, json!({ "hillshade": true }));
        assert!(terrain.field("height").unwrap().hillshade);

        let reply = set_line(&mut terrain, "height.light_azimuth 135").unwrap();
        assert_eq!(reply, json!({ "light_azimuth": 135.0 }));
        assert_eq!(terrain.field("height").unwrap().light_azimuth, 135.0);

        let reply = set_line(&mut terrain, "height.contours on").unwrap();
        assert_eq!(reply, json!({ "contours": true }));
        assert!(terrain.field("height").unwrap().contours);

        let reply = set_line(&mut terrain, "height.contour_interval 0.25").unwrap();
        assert_eq!(reply, json!({ "contour_interval": 0.25 }));
        assert_eq!(terrain.field("height").unwrap().contour_interval, 0.25);

        assert!(set_line(&mut terrain, "height.hillshade sideways").is_err());
        assert!(set_line(&mut terrain, "height.contour_interval 0").is_err());
        assert_eq!(terrain.field("height").unwrap().contour_interval, 0.25);
    }

    // How the map draws a field is not what the field holds, so toggling an overlay must
    // not throw the bake away; the length guard is what keeps a node property spelled the
    // same from claiming the exemption.
    #[test]
    fn a_display_property_is_the_only_set_that_does_not_reach_the_bake() {
        let exempt = |path: &str| {
            Edit::Set {
                path: path.to_owned(),
                words: vec!["on".to_owned()],
            }
            .reaches_the_bake()
        };

        assert!(!exempt("height.hillshade"));
        assert!(!exempt("height.light_azimuth"));
        assert!(!exempt("height.contours"));
        assert!(!exempt("height.contour_interval"));
        assert!(exempt("height.range"));
        assert!(exempt("height.n3.hillshade"));
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

    // `hold` drops an earlier held change only when a later one writes the same place,
    // so the slot has to separate two values for one property from two values for two
    // — and has to keep every structural edit apart from every other.
    #[test]
    fn an_edit_that_overwrites_a_value_names_the_slot_it_overwrites() {
        let set = |path: &str| Edit::Set {
            path: path.to_owned(),
            words: vec!["1".to_owned()],
        };
        assert_eq!(set("base.n0.value").slot(), set("base.n0.value").slot());
        assert_ne!(set("base.n0.value").slot(), set("base.n1.value").slot());

        let add = Edit::AddNode {
            field: "base".to_owned(),
            op: NodeOp::Constant(0.5),
            position: None,
        };
        let connect = Edit::Connect {
            field: "base".to_owned(),
            from: "n0".to_owned(),
            to: "n1".to_owned(),
            pin: 0,
        };
        assert_eq!(add.slot(), Slot::Once);
        assert_eq!(connect.slot(), Slot::Once);
    }
    fn reference_node(terrain: &TerrainSpec, field: &str) -> NodeId {
        terrain
            .field(field)
            .expect("the test document carries it")
            .graph
            .nodes
            .iter()
            .find(|node| matches!(node.op, NodeOp::FieldRef(_)))
            .expect("the test document carries a reference")
            .id
    }

    // A field cycle used to reach the next bake and fail there; this pins that the
    // edit itself is refused, and that the refusal names the chain the way the plan
    // error does. `height` already reads `base`, so `base` reading `height` closes it.
    #[test]
    fn adding_a_reference_that_closes_a_field_cycle_is_refused() {
        let mut terrain = document();
        let before = terrain.field("base").unwrap().graph.nodes.len();
        let error = Edit::AddNode {
            field: "base".to_owned(),
            op: NodeOp::FieldRef(FieldId::from("height")),
            position: None,
        }
        .apply(&mut terrain)
        .unwrap_err();
        assert!(error.contains("base -> height -> base"), "{error}");
        assert_eq!(terrain.field("base").unwrap().graph.nodes.len(), before);
    }

    // The other fault the bake used to catch: a name the document does not carry. The
    // document has to be left alone, because a refused edit that half-landed would put
    // the panel and the graph out of step.
    #[test]
    fn adding_a_reference_to_a_field_that_is_not_there_is_refused() {
        let mut terrain = document();
        let before = terrain.field("base").unwrap().graph.nodes.len();
        let error = Edit::AddNode {
            field: "base".to_owned(),
            op: NodeOp::FieldRef(FieldId::from("nowhere")),
            position: None,
        }
        .apply(&mut terrain)
        .unwrap_err();
        assert!(error.contains("nowhere"), "{error}");
        assert_eq!(terrain.field("base").unwrap().graph.nodes.len(), before);
    }

    // The shortest cycle there is, and it arrives by `set` rather than by `node add` —
    // which is the path the panel's field menu takes, so this covers that too.
    #[test]
    fn pointing_a_reference_at_its_own_field_is_refused() {
        let mut terrain = document();
        let id = reference_node(&terrain, "height");
        let line = format!("height.{}.field height", node_path(id));
        let error = set_line(&mut terrain, &line).unwrap_err();
        assert!(error.contains("height -> height"), "{error}");
        assert!(matches!(
            &terrain.field("height").unwrap().graph.node(id).unwrap().op,
            NodeOp::FieldRef(held) if held.as_str() == "base"
        ));
    }

    // Bypass is how a reference is turned off, and the bake reads a bypassed node as
    // no dependency at all — so a cycle that runs only through one is not a cycle, and
    // the guard has to accept the edge the bake would accept.
    #[test]
    fn a_cycle_that_runs_only_through_a_bypassed_reference_is_accepted() {
        let mut terrain = document();
        let id = reference_node(&terrain, "height");
        Edit::Bypass {
            field: "height".to_owned(),
            node: node_path(id),
            bypassed: Some(true),
        }
        .apply(&mut terrain)
        .expect("bypassing a node is always allowed");
        Edit::AddNode {
            field: "base".to_owned(),
            op: NodeOp::FieldRef(FieldId::from("height")),
            position: None,
        }
        .apply(&mut terrain)
        .expect("the only path back to `base` is bypassed");
    }
}
