//! The editor's whole vocabulary for naming and changing a document: the paths that
//! address a field and a property, the words every enum is spelled with, and the
//! edits themselves.
//!
//! The panel and the control client both go through here rather than each writing
//! their own. Two spellings of one property would be two things to keep in step, and
//! a path that worked from one and not the other would make the two disagree about
//! what a document even contains.

use crate::terrain::{Field, TerrainSpec};
use serde_json::{Value, json};
use watershed::FieldRole;

/// The place in a document a change writes, for deciding whether a later change
/// makes an earlier one pointless.
///
/// It exists only so that a stream of values aimed at one control costs one held
/// change rather than a queue: two changes with equal slots are the same place written
/// twice, and only the last of them has to land. A change that overwrites nothing in
/// particular is [`Slot::Once`] and is never dropped for another.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Slot {
    /// Writes nothing a later change can make pointless — a field added or removed.
    /// Never dropped, however many pile up.
    Once,
    /// Writes the property at this dotted path: the one [`Edit::Set`] names.
    Path(String),
    /// Writes a control in the field panel, named by the property it edits rather
    /// than by a path, because a panel binding does not build one.
    Control {
        /// What the control edits, distinct per binding.
        property: &'static str,
        /// Which of a property's several numbers, or `[0, 0]` when it has one.
        index: [usize; 2],
    },
}

/// A change to a document, as a value rather than a method.
///
/// Being a value is the point: a button builds one and a socket parses one, and both
/// then take the identical path through [`Edit::apply`]. Neither side can acquire a
/// shortcut the other lacks, and neither can change a document in a way the other
/// could not have.
#[derive(Clone)]
pub enum Edit {
    /// Adds a field with no parameter values to the document and leaves every other
    /// field alone. A file operation: see [`Edit::is_file_operation`].
    AddField {
        /// The name the field is addressed by. Surrounding whitespace is trimmed, and
        /// what is left is refused by [`check_add`].
        name: String,
    },
    /// Takes a field out of the document, with its bake. A file operation: see
    /// [`Edit::is_file_operation`].
    RemoveField {
        /// The field to remove. Refused by [`check_remove`].
        name: String,
    },
    /// Writes one property, named by a dotted path. See the module's grammar.
    Set {
        /// `field.property`, where property is a field setting or a parameter the
        /// field's shader declares.
        path: String,
        /// The value, as words. Most properties take one; a range, or a shader
        /// parameter with several components, takes several.
        words: Vec<String>,
    },
}

impl Edit {
    /// The place this edit writes, for [`Slot`]'s purpose: a `Set` names its path, and
    /// everything else is [`Slot::Once`], so two of them held together both land.
    pub fn slot(&self) -> Slot {
        match self {
            Self::Set { path, .. } => Slot::Path(path.clone()),
            _ => Slot::Once,
        }
    }

    /// Whether what this edit changes is read by a bake.
    ///
    /// A field's display properties say how the map draws the field, not what the
    /// field holds, so a `Set` on one of them is exempt. An overlay added later adds
    /// its properties to that list rather than replacing it. Every other edit reaches
    /// the bake.
    pub fn reaches_the_bake(&self) -> bool {
        match self {
            Self::Set { path, .. } => !is_display_property(path),
            _ => true,
        }
    }

    /// Whether this edit adds or removes a field, which is a file in the document's
    /// shader directory as well as an entry in the terrain.
    ///
    /// Applying one here changes only the terrain; the file is
    /// [`Document::apply`](crate::document::Document::apply)'s to write or delete, and
    /// neither is recorded in the history.
    pub fn is_file_operation(&self) -> bool {
        matches!(self, Self::AddField { .. } | Self::RemoveField { .. })
    }

    /// Applies the edit and describes what it did, as the reply the control client
    /// sends back.
    ///
    /// Refused, with a message fit to show, if the edit names a field or a property
    /// the document does not have, or a value it cannot read. A refusal leaves the
    /// document exactly as it was.
    ///
    /// Nothing here notices that the bake is now stale — that is
    /// [`Document::apply`](crate::document::Document::apply)'s job, and why edits go
    /// through the document rather than through the terrain directly.
    pub fn apply(&self, terrain: &mut TerrainSpec) -> Result<Value, String> {
        match self {
            Self::AddField { name } => {
                let name = check_add(terrain, name)?;
                terrain.fields.push(Field::new(name.as_str()));
                Ok(json!({ "added": name, "fields": terrain.fields.len() }))
            }
            Self::RemoveField { name } => {
                let name = check_remove(terrain, name)?;
                terrain.fields.retain(|field| field.id.as_str() != name);
                Ok(json!({ "removed": name, "fields": terrain.fields.len() }))
            }
            Self::Set { path, words } => set(terrain, path, words),
        }
    }

    /// The field this edit leaves on screen, or `None` for an edit that leaves the
    /// view where it was.
    ///
    /// `terrain` is the document *after* the edit applied and `active` the field that
    /// was on screen before it. A removal answers only when it was the shown field it
    /// removed, and then answers the first field left in the document, or `None` when
    /// none is left.
    pub fn shows(&self, terrain: &TerrainSpec, active: &str) -> Option<String> {
        match self {
            Self::AddField { name } => Some(name.trim().to_owned()),
            Self::RemoveField { name } => (active == name.trim())
                .then(|| terrain.fields.first().map(|field| field.id.to_string()))
                .flatten(),
            Self::Set { .. } => None,
        }
    }
}

fn field_mut<'a>(terrain: &'a mut TerrainSpec, name: &str) -> Result<&'a mut Field, String> {
    terrain
        .field_mut(name)
        .ok_or_else(|| format!("no field named `{name}`"))
}

/// The trimmed name a field may be added under, or why it may not.
///
/// Refused when the name is blank, already taken by a field of the document, or
/// could not stand as the stem of a shader file that is read back as the same field:
/// one containing `/` or `\`, or starting with `.` or `_`.
pub fn check_add(terrain: &TerrainSpec, name: &str) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("a field needs a name".to_owned());
    }
    if name.contains(['/', '\\']) || name.starts_with(['.', '_']) {
        return Err(format!(
            "`{name}` cannot name a shader file: no `/` or `\\`, and no leading `.` or `_`"
        ));
    }
    if terrain.field(name).is_some() {
        return Err(format!("this document already has a field named `{name}`"));
    }
    Ok(name.to_owned())
}

/// The trimmed name of a field that may be removed, or why it may not.
///
/// Refused when the document has no such field, when another field's shader reads it
/// — the message names every reader — or when the water spec names it.
pub fn check_remove(terrain: &TerrainSpec, name: &str) -> Result<String, String> {
    let name = name.trim();
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
            "`{name}` is read by {list} — take the `@layer` out of those files first"
        ));
    }

    if terrain.water_spec.as_ref().is_some_and(|spec| {
        spec.height.as_str() == name || spec.moisture.as_ref().is_some_and(|id| id.as_str() == name)
    }) {
        return Err(format!(
            "`{name}` is named by the water spec of this terrain — reset the water first"
        ));
    }
    Ok(name.to_owned())
}

/// The fields whose shader reads `name`, in declaration order, each named once.
///
/// The answer is derived from the document on every call rather than cached, so it
/// cannot fall out of step with an edit or a re-read file.
pub fn readers_of(terrain: &TerrainSpec, name: &str) -> Vec<String> {
    terrain
        .fields
        .iter()
        .filter(|field| field.id.as_str() != name)
        .filter(|field| field.dependencies().any(|id| id.as_str() == name))
        .map(|field| field.id.to_string())
        .collect()
}

/// The fields `field`'s shader reads, in declaration order, each named once.
///
/// [`readers_of`] read from the other end, and the two answer about one relation: a
/// field's own name is never in the list, and a file naming one field on several
/// bindings lists it once.
pub fn reads_of(field: &Field) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for read in field.dependencies() {
        if read.as_str() == field.id.as_str() {
            continue;
        }
        if !names.iter().any(|seen| seen == read.as_str()) {
            names.push(read.to_string());
        }
    }
    names
}

fn set(terrain: &mut TerrainSpec, path: &str, words: &[String]) -> Result<Value, String> {
    let parts: Vec<&str> = path.split('.').collect();
    let name = *parts.first().ok_or("a path needs a field name")?;
    match parts.len() {
        0 | 1 => Err(format!("`{path}` names a field and nothing on it")),
        2 => set_field(terrain, name, parts[1], words),
        _ => Err(format!(
            "`{path}` names more than a field and a property — a path is `field.property`"
        )),
    }
}

fn set_field(
    terrain: &mut TerrainSpec,
    name: &str,
    property: &str,
    words: &[String],
) -> Result<Value, String> {
    match property {
        "shift" => {
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
        "role" => set_field_role(terrain, name, words),
        _ => set_other_field_property(terrain, name, property, words),
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
    property: &str,
    words: &[String],
) -> Result<Value, String> {
    let field = field_mut(terrain, name)?;
    match property {
        "range" => {
            let low: f32 = number(first(words)?)?;
            let high: f32 = number(words.get(1).ok_or("a range needs two numbers")?)?;
            field.range = (low, high);
            Ok(json!({ "range": [low, high] }))
        }
        "hillshade" => {
            let on = boolean(first(words)?)?;
            field.hillshade = on;
            Ok(json!({ "hillshade": on }))
        }
        "light_azimuth" => {
            let degrees: f32 = number(first(words)?)?;
            field.light_azimuth = degrees;
            Ok(json!({ "light_azimuth": degrees }))
        }
        "contours" => {
            let on = boolean(first(words)?)?;
            field.contours = on;
            Ok(json!({ "contours": on }))
        }
        "contour_interval" => {
            let spacing: f32 = number(first(words)?)?;
            if !(spacing >= MIN_CONTOUR_INTERVAL) {
                return Err(format!(
                    "a contour interval has to be at least {MIN_CONTOUR_INTERVAL}"
                ));
            }
            field.contour_interval = spacing;
            Ok(json!({ "contour_interval": spacing }))
        }
        param if field.shader.params.contains_key(param) => {
            let values = words
                .iter()
                .map(|word| number(word))
                .collect::<Result<Vec<f32>, _>>()?;
            let held = field
                .shader
                .params
                .get_mut(param)
                .expect("the guard found the parameter");
            if values.len() != held.len() {
                return Err(format!("`{param}` takes {} numbers", held.len()));
            }
            *held = values.clone();
            Ok(json!({ param: values }))
        }
        other => Err(format!("a field has nothing called `{other}`")),
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
            .with_field(Field::new("base").held(0.25))
            .with_field(
                Field::new("height")
                    .with_role(FieldRole::Height)
                    .held(0.5)
                    .reading(&["base"]),
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
    // Every default a new field takes: no parameter values, shift 0, the unit range,
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
        assert!(added.shader.params.is_empty());
        assert_eq!(
            terrain.fields.last().map(|field| field.id.as_str()),
            Some("biomes")
        );
    }

    // A second field of one name would make every path that addresses a field
    // ambiguous, so the name is refused — and the refusal has to leave the document
    // alone, the rule every refused edit is held to.
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
    // A field is its file `shaders/<name>.wgsl`, so a name that is a path, a hidden
    // file or a template would write somewhere else or never be read back as a field.
    #[test]
    fn a_field_name_that_cannot_be_a_file_stem_is_refused() {
        let terrain = document();
        for name in ["a/b", "a\\b", ".hidden", "_x"] {
            assert!(check_add(&terrain, name).is_err(), "{name} was accepted");
        }
        assert_eq!(check_add(&terrain, " biomes ").unwrap(), "biomes");
    }

    // The field added is the one left on screen, which is what lets the panel open on
    // it straight away.
    #[test]
    fn adding_a_field_shows_the_field_it_added() {
        let edit = Edit::AddField {
            name: " biomes ".to_owned(),
        };
        assert!(edit.is_file_operation());
        assert_eq!(edit.shows(&document(), "height"), Some("biomes".to_owned()));
    }

    // Removing a field something reads would leave that file naming a field that is
    // not there, so it is refused — and the message names the readers, which is the
    // whole of what tells someone which file to change first.
    #[test]
    fn removing_a_field_a_shader_reads_is_refused_and_names_the_reader() {
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
    // `readers_of` is the relation both the removal refusal and the panel read, so it
    // has to answer the readers and only those, and never the field itself.
    #[test]
    fn readers_of_names_the_readers_and_nothing_else() {
        let terrain = document().with_field(Field::new("loop").reading(&["loop"]));
        assert_eq!(readers_of(&terrain, "base"), vec!["height".to_owned()]);
        assert!(readers_of(&terrain, "height").is_empty());
        assert!(readers_of(&terrain, "loop").is_empty());
    }

    // The panel's `reads` row and `observe field`'s `reads` list are this one answer,
    // and a file may name one field on several bindings — so the dedup is the whole
    // point: a field that reads another twice reads it once.
    #[test]
    fn reads_of_names_each_field_read_once() {
        let terrain = document();
        assert_eq!(
            reads_of(terrain.field("height").unwrap()),
            vec!["base".to_owned()]
        );
        assert!(reads_of(terrain.field("base").unwrap()).is_empty());

        let twice = Field::new("height").reading(&["base", "base"]);
        assert_eq!(reads_of(&twice), vec!["base".to_owned()]);
    }

    // The view has to follow the field it was on: a removal of the shown field falls
    // back to a field that still exists, and a removal of another leaves the view alone.
    #[test]
    fn a_removal_moves_the_view_only_when_it_was_on_that_field() {
        let mut removed = document();
        let removal = Edit::RemoveField {
            name: "height".to_owned(),
        };
        removal.apply(&mut removed).expect("nothing reads `height`");
        assert_eq!(removal.shows(&removed, "height"), Some("base".to_owned()));
        assert_eq!(removal.shows(&removed, "base"), None);
    }
    // Every one of these arrives from a caller working against a document that has
    // changed under it, so each has to be a message rather than a panic or a silent
    // no-op that looks like the edit was applied.
    #[test]
    fn an_edit_naming_something_the_document_does_not_have_is_refused() {
        let mut terrain = document();
        assert!(set_line(&mut terrain, "nowhere.value 1").is_err());
        assert!(set_line(&mut terrain, "height.sideways 1").is_err());
        assert!(set_line(&mut terrain, "height").is_err());
        assert!(
            Edit::RemoveField {
                name: "nowhere".to_owned(),
            }
            .apply(&mut terrain)
            .is_err()
        );
    }

    // A field's shader parameters are addressed as properties of the field, and one
    // given a count of numbers it does not have is refused rather than stored.
    #[test]
    fn a_parameter_is_written_by_the_field_and_its_name() {
        let mut terrain = document();
        let reply = set_line(&mut terrain, "height.value 0.75").unwrap();
        assert_eq!(reply, json!({ "value": [0.75] }));
        assert_eq!(
            terrain.field("height").unwrap().shader.params.get("value"),
            Some(&vec![0.75])
        );

        let refused = set_line(&mut terrain, "height.value 1 2").unwrap_err();
        assert!(refused.contains("value"), "{refused}");
    }

    // There is no node between a field and its parameters any more, so a path written
    // for one is refused rather than read as something else.
    #[test]
    fn a_path_through_a_node_is_refused() {
        let mut terrain = document();
        let before = terrain.clone();
        assert!(set_line(&mut terrain, "height.n0.value 1").is_err());
        assert_eq!(terrain, before);
    }

    // The settings of a field are reachable by the same two-segment path the
    // parameters are.
    #[test]
    fn a_field_setting_is_written_by_its_path() {
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
            path: "base.value".to_owned(),
            words: vec!["0.5".to_owned()],
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
    // not throw the bake away; the length guard is what keeps a longer path spelled the
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
    // `hold` drops an earlier held change only when a later one writes the same place,
    // so the slot has to separate two values for one property from two values for two
    // — and has to keep every field added or removed apart from every other.
    #[test]
    fn an_edit_that_overwrites_a_value_names_the_slot_it_overwrites() {
        let set = |path: &str| Edit::Set {
            path: path.to_owned(),
            words: vec!["1".to_owned()],
        };
        assert_eq!(set("base.value").slot(), set("base.value").slot());
        assert_ne!(set("base.value").slot(), set("height.value").slot());

        let add = Edit::AddField {
            name: "biomes".to_owned(),
        };
        assert_eq!(add.slot(), Slot::Once);
    }
}
