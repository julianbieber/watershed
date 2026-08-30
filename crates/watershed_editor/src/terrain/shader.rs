//! A node op written as WGSL: the file a node names, the parameters that file
//! declares, and the values a document carries for them.
//!
//! Nothing here touches a GPU. This is the document's half of a shader layer — what
//! is saved, what the panel is generated from, and what a dispatch is handed — so it
//! parses, packs and holds values without knowing how they are produced.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use watershed::raster::Raster;

/// The directory, inside a terrain, that a document's shaders live in.
pub const SHADER_DIR: &str = "shaders";

/// The struct a shader declares its parameters in, and the only one this grammar
/// reads.
pub const PARAMS_STRUCT: &str = "Params";

/// The largest uniform a shader's parameters may pack into.
pub const MAX_PARAM_BYTES: usize = 1024;

/// What a parameter is, which decides how many components it has and how it packs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParamType {
    /// `f32`.
    F32,
    /// `i32`, carried as an `f32` and rounded on the way into the uniform.
    I32,
    /// `u32`, carried as an `f32` and rounded on the way into the uniform.
    U32,
    /// `vec2<f32>`.
    Vec2,
    /// `vec3<f32>`.
    Vec3,
    /// `vec4<f32>`.
    Vec4,
}

impl ParamType {
    /// The WGSL spelling this type is written as, and the only one [`ParamType::parse`]
    /// accepts.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::Vec2 => "vec2<f32>",
            Self::Vec3 => "vec3<f32>",
            Self::Vec4 => "vec4<f32>",
        }
    }

    /// The type spelled exactly as [`ParamType::as_str`] writes it, with whitespace
    /// inside a vector's angle brackets tolerated, or `None`.
    pub fn parse(word: &str) -> Option<Self> {
        let compact: String = word.chars().filter(|c| !c.is_whitespace()).collect();
        match compact.as_str() {
            "f32" => Some(Self::F32),
            "i32" => Some(Self::I32),
            "u32" => Some(Self::U32),
            "vec2<f32>" => Some(Self::Vec2),
            "vec3<f32>" => Some(Self::Vec3),
            "vec4<f32>" => Some(Self::Vec4),
            _ => None,
        }
    }

    /// How many `f32`s a value of this type carries. Never zero, never over four.
    pub fn components(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 1,
            Self::Vec2 => 2,
            Self::Vec3 => 3,
            Self::Vec4 => 4,
        }
    }

    /// Whether the uniform holds this as an integer, which is what decides that a
    /// value is rounded rather than written through.
    pub fn is_integer(self) -> bool {
        matches!(self, Self::I32 | Self::U32)
    }

    /// The alignment WGSL gives this type in a uniform, in bytes.
    pub fn alignment(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::Vec2 => 8,
            Self::Vec3 | Self::Vec4 => 16,
        }
    }

    /// The bytes this type occupies, which for a `vec3` is three components rather
    /// than its alignment.
    pub fn size(self) -> usize {
        self.components() * 4
    }
}

/// How a parameter is offered in the panel.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Widget {
    /// One slider per component, over `min..=max`.
    Slider {
        /// Low end of the slider.
        min: f32,
        /// High end of the slider.
        max: f32,
        /// The increment a drag snaps to, or `None` for a continuous slider.
        step: Option<f32>,
    },
    /// Three or four sliders over `0.0..=1.0`, written in sRGB and stored linear.
    Color,
    /// A checkbox, written into the uniform as `0.0` or `1.0`.
    Toggle,
    /// No widget at all. The declared default is what the shader is handed.
    Hidden,
}

/// One parameter of a shader: what it is called, what it is, and how it is offered.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamField {
    /// The field name as the WGSL struct spells it, and the key a document stores
    /// its value under.
    pub name: String,
    /// What the panel prints. The field name unless the annotation overrode it.
    pub label: String,
    /// The section this parameter was declared under, empty for one declared before
    /// any `@group`.
    pub group: String,
    /// What the uniform holds.
    pub ty: ParamType,
    /// How it is offered.
    pub widget: Widget,
    /// The value a document takes when it does not carry one, in the shader's own
    /// terms — linear for a colour, since the sRGB an annotation is written in is
    /// converted once, here.
    pub default: Vec<f32>,
    /// Byte offset into the packed uniform.
    pub offset: usize,
}

/// Every parameter one shader declares, in declaration order, and the uniform they
/// pack into.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ParamsLayout {
    /// In declaration order, which is the order the panel draws them.
    pub fields: Vec<ParamField>,
    /// The size of the packed uniform, rounded up to sixteen bytes as WGSL requires.
    pub byte_size: usize,
}

impl ParamsLayout {
    /// The parameter of that exact name, or `None`.
    pub fn field(&self, name: &str) -> Option<&ParamField> {
        self.fields.iter().find(|field| field.name == name)
    }

    /// `values` packed into the uniform this layout describes.
    ///
    /// A parameter `values` does not carry takes its default, and one it carries
    /// with the wrong number of components is padded with zeroes or truncated — a
    /// shader edited under a document must not be able to produce a short uniform.
    pub fn pack(&self, values: &BTreeMap<String, Vec<f32>>) -> Vec<u8> {
        let mut bytes = vec![0u8; self.byte_size];
        for field in &self.fields {
            let value = values.get(&field.name).unwrap_or(&field.default);
            for component in 0..field.ty.components() {
                let raw = value.get(component).copied().unwrap_or(0.0);
                let raw = if field.ty.is_integer() {
                    raw.round()
                } else {
                    raw
                };
                let at = field.offset + component * 4;
                let word = match field.ty {
                    ParamType::I32 => (raw as i32).to_le_bytes(),
                    ParamType::U32 => (raw.max(0.0) as u32).to_le_bytes(),
                    _ => raw.to_le_bytes(),
                };
                bytes[at..at + 4].copy_from_slice(&word);
            }
        }
        bytes
    }
}

/// Why a shader's parameters could not be read.
///
/// A parse failure is not fatal to a document: the layer keeps the values it had and
/// the reason is shown, because a shader is edited in place and is expected to be
/// broken for as long as it takes to type the next line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamError {
    /// Which line of the file, counting from one, or `0` for a fault about the file
    /// as a whole.
    pub line: usize,
    /// What is wrong, in the words shown in the status bar.
    pub reason: String,
}

impl fmt::Display for ParamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line == 0 {
            f.write_str(&self.reason)
        } else {
            write!(f, "line {}: {}", self.line, self.reason)
        }
    }
}

impl std::error::Error for ParamError {}

fn fault(line: usize, reason: impl Into<String>) -> ParamError {
    ParamError {
        line,
        reason: reason.into(),
    }
}

/// One channel of an sRGB colour as a linear value.
///
/// The whole pipeline downstream of a shader is linear, so the transfer function is
/// applied once — here, where the annotation is read — rather than anywhere a value
/// is used.
pub fn srgb_to_linear(channel: f32) -> f32 {
    if channel <= 0.04045 {
        channel / 12.92
    } else {
        ((channel + 0.055) / 1.055).powf(2.4)
    }
}

/// The parameters a shader declares, read out of its source.
///
/// Reads the `Params` struct and nothing else: a field without a `@ui` annotation is
/// a fault rather than a parameter with a guessed widget, because a shader whose
/// panel is guessed at is one whose uniform the editor and the GPU disagree about.
/// A source with no `Params` struct declares no parameters and is not a fault.
///
/// Fails on the first fault and reports the line it is on.
pub fn parse_params(source: &str) -> Result<ParamsLayout, ParamError> {
    let lines: Vec<&str> = source.lines().collect();
    let Some(open) = lines.iter().position(|line| {
        let trimmed = line.trim_start();
        trimmed.starts_with("struct ")
            && trimmed
                .trim_start_matches("struct ")
                .trim_start()
                .starts_with(PARAMS_STRUCT)
    }) else {
        return Ok(ParamsLayout::default());
    };
    let Some(close) = lines
        .iter()
        .enumerate()
        .skip(open)
        .position(|(_, line)| line.trim() == "}")
    else {
        return Err(fault(
            open + 1,
            format!("`{PARAMS_STRUCT}` is never closed"),
        ));
    };

    let mut group = String::new();
    let mut fields: Vec<ParamField> = Vec::new();
    let mut offset = 0usize;
    for (index, line) in lines.iter().enumerate().take(open + close).skip(open + 1) {
        let number = index + 1;
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if let Some(name) = text.strip_prefix("// @group ") {
            group = name.trim().to_owned();
            continue;
        }
        if text.starts_with("//") {
            continue;
        }
        let (declaration, annotation) = text
            .split_once("//")
            .ok_or_else(|| fault(number, "no `@ui` annotation"))?;
        let annotation = annotation.trim();
        let annotation = annotation
            .strip_prefix("@ui")
            .ok_or_else(|| fault(number, "no `@ui` annotation"))?
            .trim();

        let declaration = declaration.trim().trim_end_matches(',').trim();
        let (name, ty) = declaration
            .split_once(':')
            .ok_or_else(|| fault(number, "not a `name: type` declaration"))?;
        let name = name.trim().to_owned();
        let ty = ParamType::parse(ty.trim())
            .ok_or_else(|| fault(number, format!("`{}` is not a parameter type", ty.trim())))?;
        if fields.iter().any(|field| field.name == name) {
            return Err(fault(number, format!("`{name}` is declared twice")));
        }

        let (label, rest) = split_label(annotation);
        let (widget, default) = parse_widget(number, ty, rest)?;

        offset = offset.next_multiple_of(ty.alignment());
        fields.push(ParamField {
            name: name.clone(),
            label: label.unwrap_or(name),
            group: group.clone(),
            ty,
            widget,
            default,
            offset,
        });
        offset += ty.size();
    }

    let byte_size = if fields.is_empty() {
        0
    } else {
        offset.next_multiple_of(16)
    };
    if byte_size > MAX_PARAM_BYTES {
        return Err(fault(
            0,
            format!("the parameters pack into {byte_size} bytes, over the {MAX_PARAM_BYTES} limit"),
        ));
    }
    Ok(ParamsLayout { fields, byte_size })
}

fn split_label(annotation: &str) -> (Option<String>, &str) {
    let Some(rest) = annotation.strip_prefix('"') else {
        return (None, annotation);
    };
    match rest.split_once('"') {
        Some((label, tail)) => (Some(label.to_owned()), tail.trim()),
        None => (None, annotation),
    }
}

fn parse_widget(
    line: usize,
    ty: ParamType,
    annotation: &str,
) -> Result<(Widget, Vec<f32>), ParamError> {
    if annotation == "hidden" {
        return Ok((Widget::Hidden, vec![0.0; ty.components()]));
    }
    if let Some(rest) = annotation.strip_prefix("toggle") {
        let on = match rest.trim() {
            "true" => 1.0,
            "false" => 0.0,
            other => return Err(fault(line, format!("`{other}` is not `true` or `false`"))),
        };
        return Ok((Widget::Toggle, vec![on]));
    }
    if let Some(rest) = annotation.strip_prefix("color") {
        if ty.components() < 3 {
            return Err(fault(line, "a colour needs three components or four"));
        }
        let channels = parse_tuple(line, rest.trim().trim_start_matches("srgb"))?;
        if channels.len() < 3 {
            return Err(fault(line, "a colour is written as `srgb(r, g, b)`"));
        }
        let mut default: Vec<f32> = channels
            .iter()
            .take(3)
            .map(|c| srgb_to_linear(*c))
            .collect();
        while default.len() < ty.components() {
            default.push(1.0);
        }
        return Ok((Widget::Color, default));
    }

    let annotation = annotation.trim_start_matches("vec").trim_start();
    let (default_text, rest) = split_default(line, annotation)?;
    let default = parse_tuple(line, &default_text)?;
    if default.len() != ty.components() {
        return Err(fault(
            line,
            format!(
                "the default has {} components, where `{}` has {}",
                default.len(),
                ty.as_str(),
                ty.components()
            ),
        ));
    }

    let rest = rest.trim();
    let (range, rest) = rest
        .strip_prefix('[')
        .and_then(|rest| rest.split_once(']'))
        .ok_or_else(|| fault(line, "no `[min, max]` range"))?;
    let bounds = parse_numbers(line, range)?;
    let [min, max] = bounds[..] else {
        return Err(fault(line, "a range is written as `[min, max]`"));
    };
    let step = match rest.trim().strip_prefix("step") {
        Some(text) => Some(
            text.trim()
                .parse::<f32>()
                .map_err(|_| fault(line, format!("`{}` is not a step", text.trim())))?,
        ),
        None => None,
    };
    Ok((Widget::Slider { min, max, step }, default))
}

fn split_default(line: usize, annotation: &str) -> Result<(String, &str), ParamError> {
    if let Some(rest) = annotation.strip_prefix('(') {
        let (inside, tail) = rest
            .split_once(')')
            .ok_or_else(|| fault(line, "the default is never closed"))?;
        return Ok((inside.to_owned(), tail));
    }
    let end = annotation
        .find('[')
        .ok_or_else(|| fault(line, "no `[min, max]` range"))?;
    Ok((annotation[..end].to_owned(), &annotation[end..]))
}

fn parse_tuple(line: usize, text: &str) -> Result<Vec<f32>, ParamError> {
    let text = text.trim();
    let inside = text
        .strip_prefix('(')
        .and_then(|rest| rest.strip_suffix(')'))
        .unwrap_or(text);
    parse_numbers(line, inside)
}

fn parse_numbers(line: usize, text: &str) -> Result<Vec<f32>, ParamError> {
    text.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<f32>()
                .map_err(|_| fault(line, format!("`{part}` is not a number")))
        })
        .collect()
}

/// A layer whose values a WGSL shader produces.
///
/// Holds what a document carries — the file and the parameter values — and the
/// raster the last dispatch left. The raster is derived, so it is not serialized and
/// a loaded document reads the layer as zero until it has been resolved.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ShaderLayer {
    /// The file, as a plain name inside the document's `shaders` directory. Never a
    /// path.
    pub file: String,
    /// A value per parameter the shader declares, keyed by the name the WGSL struct
    /// spells. A key the shader no longer declares is dropped when the file is
    /// parsed; one it declares that is missing here takes the shader's default.
    pub params: BTreeMap<String, Vec<f32>>,
    #[serde(skip)]
    values: Raster<f32>,
}

impl ShaderLayer {
    /// A layer naming `file` with no parameter values, so every parameter the shader
    /// declares takes its default until the file has been parsed.
    pub fn new(file: impl Into<String>) -> Self {
        Self {
            file: file.into(),
            params: BTreeMap::new(),
            values: Raster::default(),
        }
    }

    /// The values from the last dispatch. Empty for a node that has never been
    /// resolved or that came from a loaded document, which reads as `0.0`.
    pub fn values(&self) -> &Raster<f32> {
        &self.values
    }

    /// Installs `values` as the dispatch result, dropping whatever was there.
    /// Nothing checks it against the field's resolution.
    pub fn put_values(&mut self, values: Raster<f32>) {
        self.values = values;
    }

    /// Drops the values, leaving the layer reading as `0.0` until it is resolved
    /// again. What a failed parse must *not* do — a broken shader keeps what it last
    /// produced.
    pub fn clear(&mut self) {
        self.values = Raster::default();
    }

    /// Drops every value the layout does not declare and fills in every default it
    /// declares that is missing, which is what a shader edited under a document
    /// leaves behind.
    pub fn reconcile(&mut self, layout: &ParamsLayout) {
        self.params.retain(|name, _| layout.field(name).is_some());
        for field in &layout.fields {
            self.params
                .entry(field.name.clone())
                .or_insert_with(|| field.default.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The common case, and the one every stock shader is written in: a scalar with a
    // default and a range, read into a slider.
    #[test]
    fn a_scalar_declares_a_slider_over_its_range() {
        let layout =
            parse_params("struct Params {\n  scale: f32, // @ui 8.0 [0.1, 64.0]\n}\n").unwrap();
        let field = layout.field("scale").unwrap();
        assert_eq!(field.ty, ParamType::F32);
        assert_eq!(field.default, vec![8.0]);
        assert_eq!(
            field.widget,
            Widget::Slider {
                min: 0.1,
                max: 64.0,
                step: None
            }
        );
    }

    // A step is what makes an integer parameter usable, and it is the one part of the
    // slider form that is optional — so its absence must not be read as a zero step.
    #[test]
    fn a_step_is_optional_and_read_when_it_is_there() {
        let layout = parse_params(
            "struct Params {\n  octaves: u32, // @ui 4 [1, 8] step 1\n  gain: f32, // @ui 0.5 [0.0, 1.0]\n}\n",
        )
        .unwrap();
        assert_eq!(
            layout.field("octaves").unwrap().widget,
            Widget::Slider {
                min: 1.0,
                max: 8.0,
                step: Some(1.0)
            }
        );
        assert!(matches!(
            layout.field("gain").unwrap().widget,
            Widget::Slider { step: None, .. }
        ));
    }

    // A colour is written in sRGB and stored linear, and the conversion happens once —
    // here. A default that came through unconverted would be visibly too bright.
    #[test]
    fn a_colour_default_is_written_in_srgb_and_stored_linear() {
        let layout = parse_params(
            "struct Params {\n  tint: vec3<f32>, // @ui color srgb(0.5, 0.5, 0.5)\n}\n",
        )
        .unwrap();
        let default = &layout.field("tint").unwrap().default;
        assert_eq!(default.len(), 3);
        assert!((default[0] - srgb_to_linear(0.5)).abs() < 1e-6);
        assert!(default[0] < 0.5, "the value was not converted");
    }

    // A field with no annotation is a fault rather than a parameter with a guessed
    // widget: a guessed uniform is one the editor and the GPU disagree about.
    #[test]
    fn a_parameter_without_an_annotation_is_a_fault() {
        let error = parse_params("struct Params {\n  scale: f32,\n}\n").unwrap_err();
        assert_eq!(error.line, 2);
    }

    // The line number is what the status bar prints, so it has to be the line the
    // fault is on rather than the line the struct opened on.
    #[test]
    fn a_fault_reports_the_line_it_is_on() {
        let error = parse_params(
            "struct Params {\n  a: f32, // @ui 1.0 [0.0, 2.0]\n  b: mat4x4<f32>, // @ui 1.0 [0.0, 2.0]\n}\n",
        )
        .unwrap_err();
        assert_eq!(error.line, 3);
    }

    // A shader may take no parameters at all, and that is not the same as one whose
    // parameters could not be read.
    #[test]
    fn a_source_with_no_params_struct_declares_no_parameters() {
        let layout = parse_params("fn value(p: vec2<f32>) -> f32 { return 0.0; }\n").unwrap();
        assert!(layout.fields.is_empty());
        assert_eq!(layout.byte_size, 0);
    }

    // WGSL aligns a vec2 to eight bytes and a vec3 to sixteen, and the uniform is
    // rounded to sixteen. A layout that packed tightly would feed every field after
    // the first vector the wrong bytes.
    #[test]
    fn the_uniform_is_laid_out_the_way_wgsl_aligns_it() {
        let layout = parse_params(
            "struct Params {\n  a: f32, // @ui 1.0 [0.0, 2.0]\n  b: vec2<f32>, // @ui (1.0, 2.0) [0.0, 4.0]\n  c: vec3<f32>, // @ui color srgb(1.0, 1.0, 1.0)\n  d: f32, // @ui 1.0 [0.0, 2.0]\n}\n",
        )
        .unwrap();
        assert_eq!(layout.field("a").unwrap().offset, 0);
        assert_eq!(layout.field("b").unwrap().offset, 8);
        assert_eq!(layout.field("c").unwrap().offset, 16);
        assert_eq!(layout.field("d").unwrap().offset, 28);
        assert_eq!(layout.byte_size, 32);
    }

    // The packed bytes are what the GPU reads, so an integer parameter has to arrive
    // as an integer rather than as the bit pattern of a float.
    #[test]
    fn an_integer_parameter_packs_as_an_integer() {
        let layout =
            parse_params("struct Params {\n  octaves: u32, // @ui 4 [1, 8] step 1\n}\n").unwrap();
        let mut layer = ShaderLayer::new("octaves.wgsl");
        layer.reconcile(&layout);
        let bytes = layout.pack(&layer.params);
        assert_eq!(bytes[..4], 4u32.to_le_bytes());
        assert_eq!(bytes.len(), 16);
    }

    // A document is edited while its shader is, so a value for a parameter that no
    // longer exists must not reach the uniform and a new parameter must not be zero.
    #[test]
    fn packing_falls_back_to_the_default_for_a_value_the_document_lacks() {
        let layout = parse_params(
            "struct Params {\n  a: f32, // @ui 3.0 [0.0, 8.0]\n  b: f32, // @ui 5.0 [0.0, 8.0]\n}\n",
        )
        .unwrap();
        let mut values = BTreeMap::new();
        values.insert("a".to_owned(), vec![1.5]);
        values.insert("gone".to_owned(), vec![9.0]);
        let bytes = layout.pack(&values);
        assert_eq!(f32::from_le_bytes(bytes[0..4].try_into().unwrap()), 1.5);
        assert_eq!(f32::from_le_bytes(bytes[4..8].try_into().unwrap()), 5.0);
    }

    // What the editor does to a document when the file under it changed: the value
    // for a dropped parameter goes, and a new one arrives at its default.
    #[test]
    fn reconciling_drops_what_the_shader_no_longer_declares() {
        let layout = parse_params(
            "struct Params {\n  kept: f32, // @ui 1.0 [0.0, 2.0]\n  added: f32, // @ui 7.0 [0.0, 8.0]\n}\n",
        )
        .unwrap();
        let mut layer = ShaderLayer::new("ridged.wgsl");
        layer.params.insert("kept".to_owned(), vec![0.5]);
        layer.params.insert("gone".to_owned(), vec![9.0]);
        layer.reconcile(&layout);

        assert_eq!(layer.params.get("kept"), Some(&vec![0.5]));
        assert_eq!(layer.params.get("added"), Some(&vec![7.0]));
        assert!(!layer.params.contains_key("gone"));
    }

    // A label is what the panel prints and the name is what the uniform is keyed by,
    // so an annotation that overrides one must not touch the other.
    #[test]
    fn a_label_overrides_what_is_printed_and_not_what_is_addressed() {
        let layout = parse_params(
            "struct Params {\n  ridge_gain: f32, // @ui \"Sharpness\" 0.6 [0.0, 1.0]\n}\n",
        )
        .unwrap();
        let field = layout.field("ridge_gain").unwrap();
        assert_eq!(field.label, "Sharpness");
        assert_eq!(field.name, "ridge_gain");
    }

    // A group is a section heading, and it applies to every parameter after it rather
    // than to the one line it is on.
    #[test]
    fn a_group_applies_to_every_parameter_under_it() {
        let layout = parse_params(
            "struct Params {\n  loose: f32, // @ui 1.0 [0.0, 2.0]\n  // @group Shape\n  a: f32, // @ui 1.0 [0.0, 2.0]\n  b: f32, // @ui 1.0 [0.0, 2.0]\n}\n",
        )
        .unwrap();
        assert_eq!(layout.field("loose").unwrap().group, "");
        assert_eq!(layout.field("a").unwrap().group, "Shape");
        assert_eq!(layout.field("b").unwrap().group, "Shape");
    }

    // Two parameters of one name would give the document one key for two uniform
    // slots, so the second is refused rather than silently shadowing the first.
    #[test]
    fn a_parameter_declared_twice_is_a_fault() {
        let error = parse_params(
            "struct Params {\n  a: f32, // @ui 1.0 [0.0, 2.0]\n  a: f32, // @ui 2.0 [0.0, 2.0]\n}\n",
        )
        .unwrap_err();
        assert_eq!(error.line, 3);
    }

    // A hidden parameter still occupies its slot in the uniform, so leaving it out of
    // the layout would shift every field after it.
    #[test]
    fn a_hidden_parameter_still_takes_its_place_in_the_uniform() {
        let layout = parse_params(
            "struct Params {\n  secret: f32, // @ui hidden\n  shown: f32, // @ui 1.0 [0.0, 2.0]\n}\n",
        )
        .unwrap();
        assert_eq!(layout.field("secret").unwrap().offset, 0);
        assert_eq!(layout.field("shown").unwrap().offset, 4);
    }
}
