//! Running a layer's shader: what WGSL a document carries, what each file declares,
//! which layers the files in the document's directory stand for, and the dispatch
//! that turns one file into the raster its layer is baked from.
//!
//! When a dispatch happens is the bake's business, not this module's: the layers a
//! shader reads exist only inside the bake, which is off the main thread. What is
//! answered for here is everything a dispatch needs before it can run.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::SystemTime;

use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroupEntry, BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType,
    BufferDescriptor, BufferInitDescriptor, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, Extent3d, MapMode, Origin3d, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PollType, RawComputePipelineDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureView, TextureViewDescriptor, TextureViewDimension,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use glam::UVec2;
use watershed::raster::Raster;

use crate::document::{Document, EditorSystems, JobKind};
use crate::preset::Preset;
use crate::terrain::TerrainSpec;
use crate::terrain::shader::{LayerRead, ParamsLayout, parse_layers, parse_params, parse_retired};

const LAYER_LIB: &str = include_str!("../assets/shaders/layer_lib.wgsl");

const TEMPLATE_SOURCE: &str = include_str!("../assets/shaders/stock/_template.wgsl");

const ENTRY_POINT: &str = r#"
@compute @workgroup_size(8, 8, 1)
fn generate(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= globals.texels.x || id.y >= globals.texels.y {
        return;
    }
    layer_out[id.y * globals.texels.x + id.x] = value(cell_position(id.xy));
}
"#;

/// The shaders the editor ships, as `(file name, source)`.
///
/// A name beginning with `_` is a template: it is what a new layer is copied from, and a
/// file of that name in a document's directory is not a layer.
pub const STOCK: [(&str, &str); 7] = [
    ("_template.wgsl", TEMPLATE_SOURCE),
    (
        "ridged.wgsl",
        include_str!("../assets/shaders/stock/ridged.wgsl"),
    ),
    (
        "warped.wgsl",
        include_str!("../assets/shaders/stock/warped.wgsl"),
    ),
    (
        "terrace.wgsl",
        include_str!("../assets/shaders/stock/terrace.wgsl"),
    ),
    ("fbm.wgsl", include_str!("../assets/shaders/stock/fbm.wgsl")),
    (
        "continents.wgsl",
        include_str!("../assets/shaders/stock/continents.wgsl"),
    ),
    (
        "mountains_over_base.wgsl",
        include_str!("../assets/shaders/stock/mountains_over_base.wgsl"),
    ),
];

/// Everything a shader may call and everything a shader file may declare, as prose:
/// the header of the template a new layer's shader is copied from, with its comment
/// markers taken off.
///
/// This is literally the text a copied shader carries, so a panel rendering it and a
/// reader scrolling that file cannot be told two different things.
pub fn shader_reference() -> &'static str {
    static REFERENCE: LazyLock<String> = LazyLock::new(|| strip_header(TEMPLATE_SOURCE));
    &REFERENCE
}

fn strip_header(source: &str) -> String {
    let mut prose = String::new();
    for line in source.lines() {
        let trimmed = line.trim();
        let Some(text) = trimmed.strip_prefix("//") else {
            if trimmed.is_empty() {
                continue;
            }
            break;
        };
        let text = text.strip_prefix(' ').unwrap_or(text);
        prose.push_str(text);
        prose.push('\n');
    }
    prose.trim_matches('\n').to_owned()
}

fn assemble(source: &str) -> String {
    format!("{LAYER_LIB}\n{source}\n{ENTRY_POINT}")
}

fn library_lines() -> usize {
    LAYER_LIB.lines().count() + 1
}

fn validate(source: &str) -> Result<(), String> {
    let assembled = assemble(source);
    let module = match naga::front::wgsl::parse_str(&assembled) {
        Ok(module) => module,
        Err(error) => {
            let line = error
                .location(&assembled)
                .map(|at| at.line_number as usize)
                .unwrap_or(0);
            return Err(fault_at(line, error.message()));
        }
    };
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    match validator.validate(&module) {
        Ok(_) => Ok(()),
        Err(error) => {
            let line = error
                .location(&assembled)
                .map(|at| at.line_number as usize)
                .unwrap_or(0);
            Err(fault_at(line, &error.to_string()))
        }
    }
}

struct Declared {
    layout: ParamsLayout,
    layers: Vec<LayerRead>,
}

fn declare(source: &str) -> Result<Declared, String> {
    let layout = parse_params(source).map_err(|error| error.to_string())?;
    let layers = parse_layers(source).map_err(|error| error.to_string())?;
    parse_retired(source).map_err(|error| error.to_string())?;
    validate(source)?;
    Ok(Declared { layout, layers })
}

fn fault_at(line: usize, message: &str) -> String {
    match line.checked_sub(library_lines()) {
        Some(own) if own > 0 => format!("line {own}: {message}"),
        _ => message.to_owned(),
    }
}

/// One shader file as the editor last read it.
#[derive(Clone, Debug)]
pub struct ShaderEntry {
    /// The file's own source, without the library or the entry point.
    pub source: String,
    /// What the file declares. The last layout that parsed, so a file that is broken
    /// now still draws the panel it drew before.
    pub layout: ParamsLayout,
    /// The layers the file reads by name, in declaration order. As with the layout,
    /// the last list that parsed.
    pub layers: Vec<LayerRead>,
    /// Why the file did not parse or compile: `line N: message` against the file's own
    /// lines, or the bare message when the fault is not on a line the file owns.
    /// `None` when it is good.
    ///
    /// The file name is not part of it — every reader already has the name and says it
    /// in its own words.
    pub error: Option<String>,
    modified: Option<SystemTime>,
}

/// What a shader is told about where it is being evaluated.
///
/// Laid out as the `Globals` struct in `layer_lib.wgsl`, which is the other half of
/// this type. A change here is a change there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchGlobals {
    /// The document's extent, in cells.
    pub document: UVec2,
    /// The extent of this dispatch, in texels of the layer's raster.
    pub texels: UVec2,
    /// Where the dispatch starts, in texels of the layer's raster.
    pub origin: UVec2,
    /// The layer's raster shift.
    pub shift: u32,
    /// The document's seed.
    pub seed: u32,
}

impl DispatchGlobals {
    fn bytes(self) -> [u8; 32] {
        let words: [u32; 8] = [
            self.document.x,
            self.document.y,
            self.texels.x,
            self.texels.y,
            self.origin.x,
            self.origin.y,
            self.shift,
            self.seed,
        ];
        let mut bytes = [0u8; 32];
        for (index, word) in words.iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// Every shader a document carries, and where they live.
///
/// The root is the document's own `shaders` directory once it has been saved, and a
/// scratch directory before that — so a layer can be added to a document that has
/// never been written, and the first save moves the directory in whole.
#[derive(Resource, Debug)]
pub struct ShaderLibrary {
    root: PathBuf,
    entries: BTreeMap<String, ShaderEntry>,
    generation: u64,
    present: bool,
}

impl Default for ShaderLibrary {
    fn default() -> Self {
        Self {
            root: scratch_root(),
            entries: BTreeMap::new(),
            generation: 0,
            present: false,
        }
    }
}

/// The directory a document's shaders live in before it has been saved: one per
/// editor process, under the system temp directory.
pub fn scratch_root() -> PathBuf {
    std::env::temp_dir().join(format!("watershed-shaders-{}", std::process::id()))
}

/// The source of the stock shader of that file name, or `None` for a name this build
/// does not ship.
pub fn stock_source(name: &str) -> Option<&'static str> {
    STOCK
        .iter()
        .find(|(stock, _)| *stock == name)
        .map(|(_, source)| *source)
}

/// The source a layer added to a document starts as.
pub fn template_source() -> &'static str {
    TEMPLATE_SOURCE
}

/// Makes `dir` hold exactly the shader files of `preset`: creates the directory,
/// **deletes every `.wgsl` file already in it**, then writes each layer's
/// `<layer>.wgsl` from its stock file.
///
/// Refused at the first removal or write that fails, leaving what was done before it.
pub fn write_preset(dir: &Path, preset: Preset) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|error| error.to_string())?;
    let entries = std::fs::read_dir(dir).map_err(|error| error.to_string())?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "wgsl")
        {
            std::fs::remove_file(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        }
    }
    for (layer, stock) in preset.files() {
        let source = stock_source(stock)
            .ok_or_else(|| format!("`{stock}` is not a shader this build ships"))?;
        let file = format!("{layer}.wgsl");
        std::fs::write(dir.join(&file), source).map_err(|error| format!("{file}: {error}"))?;
    }
    Ok(())
}

impl ShaderLibrary {
    /// The directory the document's shaders are read from and written to.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// What the editor last read for `file`, or `None` for a name no file answers.
    pub fn entry(&self, file: &str) -> Option<&ShaderEntry> {
        self.entries.get(file)
    }

    /// Every file the directory holds, in name order, templates included.
    pub fn files(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Points the library at a document's own directory, dropping what it read from
    /// wherever it was looking before. Reading the new directory is the next scan's
    /// business.
    pub fn look_at(&mut self, root: PathBuf) {
        if self.root != root {
            self.root = root;
            self.entries.clear();
            self.generation += 1;
        }
    }

    /// Whether the last scan found the directory at all. A directory that does not
    /// exist says nothing about which layers a document has.
    pub fn present(&self) -> bool {
        self.present
    }

    /// The layers the directory stands for: the stem of every `.wgsl` file not
    /// beginning with `_`, in name order.
    pub fn layer_names(&self) -> Vec<String> {
        self.entries
            .keys()
            .filter(|name| !name.starts_with('_'))
            .filter_map(|name| name.strip_suffix(".wgsl"))
            .map(str::to_owned)
            .collect()
    }
}

/// The systems that keep a document's shaders read, parsed and runnable.
pub struct ShaderPlugin;

impl Plugin for ShaderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShaderLibrary>().add_systems(
            Update,
            (follow_document, scan, attend_shaders)
                .chain()
                .before(EditorSystems::Document),
        );
    }
}

/// One shader file, ready to run: everything a dispatch needs and nothing that has to
/// be looked up on the main thread.
#[derive(Clone, Debug)]
pub struct ShaderProgram {
    /// The file's own source, without the library or the entry point.
    pub source: String,
    /// What its `Params` struct declares, and what a layer's parameter values are
    /// packed against.
    pub layout: ParamsLayout,
    /// The layers it reads by name, in declaration order.
    pub layers: Vec<LayerRead>,
}

struct Runtime {
    device: RenderDevice,
    queue: RenderQueue,
    programs: BTreeMap<String, ShaderProgram>,
    generation: u64,
    seed: u32,
}

/// What a bake needs to run a shader: the device, the queue, and every program the
/// document's shader directory currently compiles to.
///
/// Derived state a document carries so that a bake — which is off the main thread and
/// holds the document — can dispatch without reaching back for a resource. Cloning is
/// an `Arc` clone, so a history snapshot costs nothing; two runtimes always compare
/// equal, because what a document *is* does not include the device it was last run
/// against.
///
/// A default one holds no device, and running anything through it is an error rather
/// than a panic — which is what a headless test and a document baked before the first
/// sweep both get.
#[derive(Clone, Default)]
pub struct ShaderRuntime(Option<Arc<Runtime>>);

impl fmt::Debug for ShaderRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some(runtime) => write!(
                f,
                "ShaderRuntime(generation {}, {} programs)",
                runtime.generation,
                runtime.programs.len()
            ),
            None => f.write_str("ShaderRuntime(none)"),
        }
    }
}

impl PartialEq for ShaderRuntime {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl ShaderRuntime {
    /// The library's error-free entries as programs, held with the device that will
    /// run them and the document seed they will be told about.
    pub fn compile(
        device: &RenderDevice,
        queue: &RenderQueue,
        library: &ShaderLibrary,
        seed: u32,
    ) -> Self {
        let programs = library
            .entries
            .iter()
            .filter(|(_, entry)| entry.error.is_none())
            .map(|(name, entry)| {
                (
                    name.clone(),
                    ShaderProgram {
                        source: entry.source.clone(),
                        layout: entry.layout.clone(),
                        layers: entry.layers.clone(),
                    },
                )
            })
            .collect();
        Self(Some(Arc::new(Runtime {
            device: device.clone(),
            queue: queue.clone(),
            programs,
            generation: library.generation,
            seed,
        })))
    }

    /// A runtime holding programs built from `sources`, as `(file name, source)`, on
    /// this runtime's device and queue and at its library generation, telling a
    /// dispatch `seed`. A source that does not parse or compile is left out.
    ///
    /// Answers a runtime holding no device when this one holds none.
    pub fn with_sources(
        &self,
        seed: u32,
        sources: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        let Some(runtime) = &self.0 else {
            return Self::default();
        };
        Self(Some(Arc::new(Runtime {
            device: runtime.device.clone(),
            queue: runtime.queue.clone(),
            programs: programs(sources),
            generation: runtime.generation,
            seed,
        })))
    }

    /// As [`ShaderRuntime::with_sources`], over every `.wgsl` file in `dir`. A
    /// directory that does not exist, or a file that cannot be read, contributes no
    /// source.
    pub fn with_directory(&self, seed: u32, dir: &Path) -> Self {
        if self.0.is_none() {
            return Self::default();
        }
        let sources = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                if !name.ends_with(".wgsl") {
                    return None;
                }
                let source = std::fs::read_to_string(entry.path()).ok()?;
                Some((name, source))
            });
        self.with_sources(seed, sources)
    }

    /// The library generation these programs were taken at, or `None` for a runtime
    /// holding no device. What decides that the programs are stale.
    pub fn generation(&self) -> Option<u64> {
        self.0.as_ref().map(|runtime| runtime.generation)
    }

    /// The document seed a dispatch through this runtime tells the shader, or `None`
    /// for a runtime holding no device. Part of the runtime because a bake has no
    /// other way to reach the document's seed.
    pub fn seed(&self) -> Option<u32> {
        self.0.as_ref().map(|runtime| runtime.seed)
    }

    /// The program for that file, or `None` for a name this runtime does not carry —
    /// a file that does not compile, or one added since the programs were taken.
    pub fn program(&self, file: &str) -> Option<&ShaderProgram> {
        self.0.as_ref()?.programs.get(file)
    }

    /// Runs one program over the whole of `globals.texels` and answers its values,
    /// row-major.
    ///
    /// `layers` is one entry per [`ShaderProgram::layers`], in order; `None` is a layer
    /// that has no raster to hand over and reads `0.0` everywhere. A shorter slice
    /// leaves the layers after it reading `0.0`.
    ///
    /// Blocks until the GPU has finished and the result has been read back. Answers
    /// an error rather than panicking when the runtime holds no device.
    pub fn run(
        &self,
        program: &ShaderProgram,
        params: &[u8],
        globals: DispatchGlobals,
        layers: &[Option<&Raster<f32>>],
    ) -> Result<Vec<f32>, String> {
        let Some(runtime) = &self.0 else {
            return Err("there is no render device to dispatch a shader on".to_owned());
        };
        let bound: Vec<(u32, Option<&Raster<f32>>)> = program
            .layers
            .iter()
            .enumerate()
            .map(|(at, read)| (read.binding, layers.get(at).copied().flatten()))
            .collect();
        dispatch(
            &runtime.device,
            &runtime.queue,
            &program.source,
            params,
            globals,
            &bound,
        )
    }
}

fn programs(
    sources: impl IntoIterator<Item = (String, String)>,
) -> BTreeMap<String, ShaderProgram> {
    sources
        .into_iter()
        .filter_map(|(name, source)| {
            let declared = declare(&source).ok()?;
            Some((
                name,
                ShaderProgram {
                    source,
                    layout: declared.layout,
                    layers: declared.layers,
                },
            ))
        })
        .collect()
}

fn follow_document(document: Res<Document>, mut library: ResMut<ShaderLibrary>) {
    if !document.is_changed() {
        return;
    }
    let root = document.shader_root();
    if library.root() == root {
        return;
    }
    if document.job() == Some(JobKind::Save) {
        let moving = library.root() == scratch_root();
        carry_shaders(library.root(), &root, moving);
    }
    library.look_at(root);
}

fn carry_shaders(from: &Path, to: &Path, moving: bool) {
    let Ok(entries) = std::fs::read_dir(from) else {
        return;
    };
    if std::fs::create_dir_all(to).is_err() {
        return;
    }
    for entry in entries.flatten() {
        let destination = to.join(entry.file_name());
        if destination.exists() {
            continue;
        }
        if !moving || std::fs::rename(entry.path(), &destination).is_err() {
            let _ = std::fs::copy(entry.path(), &destination);
        }
    }
}

fn scan(mut library: ResMut<ShaderLibrary>, mut document: ResMut<Document>) {
    let root = library.root.clone();
    let Ok(dir) = std::fs::read_dir(&root) else {
        library.present = false;
        if !library.entries.is_empty() {
            library.entries.clear();
            library.generation += 1;
        }
        return;
    };
    library.present = true;

    let mut seen: Vec<String> = Vec::new();
    for entry in dir.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".wgsl") {
            continue;
        }
        seen.push(name.clone());
        let modified = entry.metadata().ok().and_then(|meta| meta.modified().ok());
        if library
            .entries
            .get(&name)
            .is_some_and(|held| held.modified == modified)
        {
            continue;
        }
        let Ok(source) = std::fs::read_to_string(root.join(&name)) else {
            continue;
        };
        library.generation += 1;
        let previous = library
            .entries
            .get(&name)
            .map(|held| (held.layout.clone(), held.layers.clone()));
        let entry = match declare(&source) {
            Ok(declared) => ShaderEntry {
                source,
                layout: declared.layout,
                layers: declared.layers,
                error: None,
                modified,
            },
            Err(reason) => {
                let (layout, layers) = previous.unwrap_or_default();
                ShaderEntry {
                    source,
                    layout,
                    layers,
                    error: Some(reason),
                    modified,
                }
            }
        };
        if let Some(reason) = &entry.error {
            warn!("{name}: {reason}");
            document.refuse(format!("{name}: {reason}"));
        }
        library.entries.insert(name, entry);
    }

    let gone: Vec<String> = library
        .entries
        .keys()
        .filter(|name| !seen.contains(name))
        .cloned()
        .collect();
    for name in gone {
        library.entries.remove(&name);
        library.generation += 1;
    }
}

fn shaders_moved(terrain: &TerrainSpec, current: &ShaderRuntime, next: &ShaderRuntime) -> bool {
    let reseeded = current.seed() != next.seed();
    terrain.layers.iter().any(|layer| {
        let file = layer.file();
        reseeded
            || next.program(&file).map(|held| &held.source)
                != current.program(&file).map(|held| &held.source)
    })
}

fn attend_shaders(
    device: Option<Res<RenderDevice>>,
    queue: Option<Res<RenderQueue>>,
    library: Res<ShaderLibrary>,
    mut document: ResMut<Document>,
) {
    if document.is_busy() {
        return;
    }
    let stale = {
        let current = document.shader_runtime();
        current.generation() != Some(library.generation) || current.seed() != Some(document.seed)
    };
    let mut touched = false;
    if stale && let (Some(device), Some(queue)) = (device, queue) {
        let next = ShaderRuntime::compile(&device, &queue, &library, document.seed);
        touched = document
            .terrain()
            .is_some_and(|terrain| shaders_moved(terrain, terrain.shader_runtime(), &next));
        document.set_shader_runtime(next);
    }

    if library.present() {
        document.sync_layers(&library.layer_names());
    }
    let Some(terrain) = document.terrain_mut() else {
        return;
    };
    for layer in &mut terrain.layers {
        let Some(entry) = library.entry(&layer.file()) else {
            continue;
        };
        touched |= layer.shader.reconcile(&entry.layout);
        touched |= layer.shader.reconcile_layers(&entry.layers);
    }
    if touched {
        document.note_edit();
    }
}

/// A key over everything one dispatch's result depends on that a caller can cheaply
/// re-derive: the source, the packed parameters and the extent.
///
/// What a shader *reads* is not in it — a layer whose file reads a layer is
/// dispatched on every bake regardless, because that layer may have moved.
pub fn dispatch_key(source: &str, params: &[u8], texels: UVec2) -> u64 {
    let mut bytes = source.as_bytes().to_vec();
    bytes.extend_from_slice(params);
    bytes.extend_from_slice(&texels.x.to_le_bytes());
    bytes.extend_from_slice(&texels.y.to_le_bytes());
    fingerprint(&bytes)
}

fn fingerprint(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn upload_input(
    device: &RenderDevice,
    queue: &RenderQueue,
    raster: Option<&Raster<f32>>,
) -> TextureView {
    let zero = [0.0f32];
    let (size, data) = match raster.filter(|held| !held.is_empty()) {
        Some(raster) => (raster.size(), raster.data()),
        None => (UVec2::ONE, &zero[..]),
    };
    let extent = Extent3d {
        width: size.x,
        height: size.y,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("watershed shader layer"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let bytes: Vec<u8> = data.iter().flat_map(|value| value.to_le_bytes()).collect();
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        &bytes,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.x * 4),
            rows_per_image: Some(size.y),
        },
        extent,
    );
    texture.create_view(&TextureViewDescriptor::default())
}

/// One shader run over one rectangle, read back as the values its layer holds.
///
/// `inputs` is `(binding, raster)` per declared layer. A `None` raster is bound as a
/// single texel of `0.0`, which is what makes a layer with no raster read `0.0`
/// without a branch in the shader or a hole in the bind group.
///
/// Blocks until the GPU has finished and the result has been mapped: the bake samples
/// the result as soon as this returns, and the bake is synchronous.
///
/// The shader is compiled from the library, the file's own source and the entry
/// point, in that order, because WGSL has no forward declaration.
pub fn dispatch(
    device: &RenderDevice,
    queue: &RenderQueue,
    source: &str,
    params: &[u8],
    globals: DispatchGlobals,
    inputs: &[(u32, Option<&Raster<f32>>)],
) -> Result<Vec<f32>, String> {
    let texels = (globals.texels.x as u64) * (globals.texels.y as u64);
    if texels == 0 {
        return Ok(Vec::new());
    }
    let bytes = texels * 4;
    let assembled = format!("{LAYER_LIB}\n{source}\n{ENTRY_POINT}");

    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("watershed layer shader"),
        source: ShaderSource::Wgsl(assembled.into()),
    });

    let uniform = |binding: u32| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let mut layout_entries = vec![
        uniform(0),
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        uniform(2),
    ];
    for (binding, _) in inputs {
        layout_entries.push(BindGroupLayoutEntry {
            binding: *binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: false },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
    }
    let layout = device.create_bind_group_layout(Some("watershed layer shader"), &layout_entries);

    let views: Vec<(u32, TextureView)> = inputs
        .iter()
        .map(|(binding, raster)| (*binding, upload_input(device, queue, *raster)))
        .collect();

    let globals_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("watershed shader globals"),
        contents: &globals.bytes(),
        usage: BufferUsages::UNIFORM,
    });
    let params_bytes = if params.is_empty() {
        &[0u8; 16][..]
    } else {
        params
    };
    let params_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("watershed shader params"),
        contents: params_bytes,
        usage: BufferUsages::UNIFORM,
    });
    let output = device.create_buffer(&BufferDescriptor {
        label: Some("watershed shader output"),
        size: bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("watershed shader readback"),
        size: bytes,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut group_entries = vec![
        BindGroupEntry {
            binding: 0,
            resource: globals_buffer.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 1,
            resource: output.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 2,
            resource: params_buffer.as_entire_binding(),
        },
    ];
    for (binding, view) in &views {
        group_entries.push(BindGroupEntry {
            binding: *binding,
            resource: BindingResource::TextureView(view),
        });
    }
    let bind_group =
        device.create_bind_group(Some("watershed layer shader"), &layout, &group_entries);

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("watershed layer shader"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("watershed layer shader"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("generate"),
        compilation_options: PipelineCompilationOptions {
            constants: &[],
            zero_initialize_workgroup_memory: false,
        },
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("watershed layer shader"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("watershed layer shader"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            globals.texels.x.div_ceil(8),
            globals.texels.y.div_ceil(8),
            1,
        );
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, bytes);
    queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    device.map_buffer(&slice, MapMode::Read, move |result| {
        let _ = sender.send(result.is_ok());
    });
    device
        .poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|error| format!("the shader dispatch did not finish: {error}"))?;
    match receiver.recv() {
        Ok(true) => {}
        _ => return Err("the shader output could not be read back".to_owned()),
    }

    let view = slice.get_mapped_range();
    let values: Vec<f32> = view
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    drop(view);
    staging.unmap();
    Ok(values)
}

#[cfg(test)]
impl ShaderLibrary {
    /// A library holding `file` as a shader that failed with `fault`, for tests that
    /// need a broken entry without a directory to read one from.
    pub fn with_fault(file: &str, fault: &str) -> Self {
        let mut library = Self::default();
        library.entries.insert(
            file.to_owned(),
            ShaderEntry {
                source: String::new(),
                layout: ParamsLayout::default(),
                layers: Vec::new(),
                error: Some(fault.to_owned()),
                modified: None,
            },
        );
        library.generation = 1;
        library
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The stock shaders are what a preset's layers are copied from, so one that does
    // not parse would hand the user a broken layer on the first `new`.
    #[test]
    fn every_stock_shader_declares_readable_parameters() {
        for (name, source) in STOCK {
            let layout = parse_params(source)
                .unwrap_or_else(|error| panic!("{name} does not parse: {error}"));
            assert!(!layout.fields.is_empty(), "{name} declares no parameters");
        }
    }

    // A shader's parameters and the entry point are compiled as one source, so the
    // library has to declare its bindings before either of them uses one.
    #[test]
    fn the_assembled_source_puts_the_library_first_and_the_entry_point_last() {
        let assembled = assemble(STOCK[1].1);
        assert!(
            assembled.find("var<uniform> globals").unwrap() < assembled.find("fn value").unwrap()
        );
        assert!(assembled.find("fn value").unwrap() < assembled.find("fn generate").unwrap());
    }

    // The library, the entry point and every shipped shader are compiled as one
    // source, and this is the only thing that says they still are — the dispatch
    // itself needs a GPU, and this does not.
    #[test]
    fn every_stock_shader_compiles_against_the_library_and_the_entry_point() {
        for (name, source) in STOCK {
            if let Err(error) = validate(source) {
                panic!("{name} does not compile: {error}");
            }
        }
    }

    // The acceptance is that the header alone is enough to write a shader, so what the
    // panel renders has to be prose rather than a commented file, and it has to reach
    // every grammar a file may use.
    #[test]
    fn the_reference_reads_as_prose_and_covers_every_annotation() {
        let reference = shader_reference();
        assert!(
            !reference
                .lines()
                .any(|line| line.trim_start().starts_with("//")),
            "the reference still carries comment markers"
        );
        for needle in [
            "uv(",
            "document_extent(",
            "@ui",
            "@layer",
            "layer_value(",
            "@group",
        ] {
            assert!(
                reference.contains(needle),
                "the reference never names `{needle}`"
            );
        }
    }

    // A helper added to the library with no line in the header would be invisible to
    // anyone who only reads the header, which is the one thing the acceptance asks for.
    #[test]
    fn every_library_function_is_named_in_the_reference() {
        let reference = shader_reference();
        let internal = ["hash2", "gradient", "fade"];
        for line in LAYER_LIB.lines() {
            let Some(rest) = line.strip_prefix("fn ") else {
                continue;
            };
            let name = rest.split('(').next().expect("a declaration with no name");
            if internal.contains(&name) {
                continue;
            }
            assert!(
                reference.contains(&format!("{name}(")),
                "`{name}` is in the library but not in the reference"
            );
        }
    }

    // The worked example the header gives, compiled: the acceptance observation short
    // of a GPU. The header's own examples must not be read as this file's declaration —
    // a layer example would make every layer copied from the template read `base`.
    #[test]
    fn the_headers_worked_example_compiles_and_its_examples_declare_nothing() {
        let source =
            "fn value(p: vec2<f32>) -> f32 {\n    return fbm_unit(uv(p) * 4.0, 4u, 0.5, 2.0);\n}\n";
        if let Err(error) = validate(source) {
            panic!("the header's worked example does not compile: {error}");
        }
        assert!(parse_layers(TEMPLATE_SOURCE).unwrap().is_empty());
        assert!(
            declare(TEMPLATE_SOURCE).is_ok(),
            "the template does not declare"
        );
    }

    // What the `@layer` annotation is for: a file naming a layer and reading it
    // through the library helper has to compile, or the reference would describe a
    // read no shader can make.
    #[test]
    fn a_shader_reading_a_layer_through_the_helper_compiles() {
        let source = "@group(0) @binding(3) var base: texture_2d<f32>; // @layer base\n\nfn value(p: vec2<f32>) -> f32 {\n    return layer_value(base, p) + f32(layer_shift(base));\n}\n";
        assert_eq!(parse_layers(source).unwrap().len(), 1);
        if let Err(error) = validate(source) {
            panic!("a read of a named layer does not compile: {error}");
        }
    }

    // A file written for a node graph still declares a pin or a reach, and a layer
    // that silently ignored either would not do what its author wrote.
    #[test]
    fn a_file_declaring_a_retired_annotation_is_refused_naming_it() {
        let reach = "// @reach 2\nfn value(p: vec2<f32>) -> f32 {\n    return 0.0;\n}\n";
        let error = declare(reach).err().expect("a `@reach` was accepted");
        assert!(error.contains("@reach"), "{error}");

        let input = "@group(0) @binding(3) var a: texture_2d<f32>; // @in\n\nfn value(p: vec2<f32>) -> f32 {\n    return 0.0;\n}\n";
        let error = declare(input).err().expect("an `@in` was accepted");
        assert!(error.contains("@in"), "{error}");
    }

    // The reason a shader is validated before the device sees it: wgpu reports a
    // compile fault through a handler that panics, so a typo has to be caught here.
    #[test]
    fn a_shader_that_does_not_compile_is_reported_rather_than_handed_over() {
        let error = validate("fn value(p: vec2<f32>) -> f32 { return nonesuch(p); }\n")
            .expect_err("a call to nothing compiled");
        assert!(error.contains("nonesuch"), "{error}");
    }

    // The fault is shown against the file the author is looking at, so the library's
    // own length has to be taken back off the line the compiler reported.
    #[test]
    fn a_compile_fault_is_reported_against_the_shaders_own_lines() {
        let error = validate("fn value(p: vec2<f32>) -> f32 {\n    return oops;\n}\n")
            .expect_err("a reference to nothing compiled");
        assert!(error.starts_with("line 2:"), "{error}");
    }

    // A preset's first bake compiles its programs from sources rather than from a
    // scanned directory, and a source the GPU would refuse has to be left out there
    // too, since handing one over panics.
    #[test]
    fn programs_keep_a_source_that_compiles_and_leave_out_one_that_does_not() {
        let built = programs([
            (
                "fbm.wgsl".to_owned(),
                stock_source("fbm.wgsl").unwrap().to_owned(),
            ),
            (
                "bad.wgsl".to_owned(),
                "fn value(p: vec2<f32>) -> f32 { return nonesuch(p); }\n".to_owned(),
            ),
        ]);
        assert!(built.contains_key("fbm.wgsl"));
        assert!(!built.contains_key("bad.wgsl"));
    }

    // Headless tests and a document built before the editor has seen a device both
    // start from a runtime holding none, and building from sources must not pretend
    // to have one.
    #[test]
    fn a_runtime_built_from_sources_on_one_holding_no_device_holds_none() {
        let built = ShaderRuntime::default().with_sources(
            7,
            [(
                "fbm.wgsl".to_owned(),
                stock_source("fbm.wgsl").unwrap().to_owned(),
            )],
        );
        assert_eq!(built.generation(), None);
        assert!(built.program("fbm.wgsl").is_none());
    }

    // A preset's `new` has to leave exactly its own layers' files on disk, as this
    // build ships them: a file left from an earlier document would become a layer of
    // this one.
    #[test]
    fn writing_a_preset_removes_stale_files_and_writes_one_per_layer() {
        let dir = std::env::temp_dir().join(format!("watershed-preset-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("stale.wgsl"), "edited").unwrap();
        std::fs::write(dir.join("notes.txt"), "kept").unwrap();

        write_preset(&dir, Preset::Ridges).unwrap();
        assert!(!dir.join("stale.wgsl").exists());
        assert!(dir.join("notes.txt").is_file());
        assert_eq!(
            std::fs::read_to_string(dir.join("base.wgsl")).unwrap(),
            stock_source("continents.wgsl").unwrap()
        );
        assert!(dir.join("height.wgsl").is_file() && dir.join("moisture.wgsl").is_file());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // Saving a document that already has a directory to a second one must leave the
    // first directory's shaders where they were, and must not overwrite a shader the
    // destination already carries.
    #[test]
    fn carrying_by_copy_keeps_the_source_and_leaves_an_existing_destination_file() {
        let base = std::env::temp_dir().join(format!("watershed-carry-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let from = base.join("from");
        let to = base.join("to");
        std::fs::create_dir_all(&from).unwrap();
        std::fs::create_dir_all(&to).unwrap();
        std::fs::write(from.join("a.wgsl"), "a").unwrap();
        std::fs::write(from.join("b.wgsl"), "b").unwrap();
        std::fs::write(to.join("b.wgsl"), "kept").unwrap();

        carry_shaders(&from, &to, false);
        assert!(from.join("a.wgsl").is_file() && from.join("b.wgsl").is_file());
        assert_eq!(std::fs::read_to_string(to.join("a.wgsl")).unwrap(), "a");
        assert_eq!(std::fs::read_to_string(to.join("b.wgsl")).unwrap(), "kept");
        std::fs::remove_dir_all(&base).unwrap();
    }

    // The uniform is read by the shader through a struct with a fixed layout, so its
    // size is part of the contract rather than an implementation detail.
    #[test]
    fn the_globals_uniform_is_the_size_the_shader_declares() {
        let globals = DispatchGlobals {
            document: UVec2::new(256, 128),
            texels: UVec2::new(64, 32),
            origin: UVec2::new(4, 8),
            shift: 2,
            seed: 7,
        };
        let bytes = globals.bytes();
        assert_eq!(bytes.len(), 32);
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 256);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 64);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 4);
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(bytes[28..32].try_into().unwrap()), 7);
    }
}
