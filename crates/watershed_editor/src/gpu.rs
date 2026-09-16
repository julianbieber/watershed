//! Running a layer's shader: what WESL a document carries, what each file declares and
//! imports, which layers the files in the document's directory stand for, and the
//! dispatch that turns one file into the raster its layer is baked from.
//!
//! When a dispatch happens is the bake's business, not this module's: the layers a
//! shader reads exist only inside the bake, which is off the main thread. What is
//! answered for here is everything a dispatch needs before it can run.

mod dispatch;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::SystemTime;

use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::shader::Shader;
use glam::UVec2;
use watershed::raster::Raster;
use wesl::syntax::{GlobalDeclaration, ImportContent, ModulePath, PathOrigin, TranslationUnit};

use crate::document::{Document, EditorSystems};
use crate::preset::Preset;
use crate::terrain::TerrainSpec;
use crate::terrain::shader::{
    LayerHeader, LayerRead, ParamsLayout, parse_header, parse_layers, parse_params, parse_retired,
};

use dispatch::{DispatchBridge, DispatchSender, LayerInput};

const LAYER_LIB: &str = include_str!("../assets/shaders/stock/lib.wesl");

const WESL_TOML: &str = include_str!("../assets/shaders/stock/wesl.toml");

const TEMPLATE_SOURCE: &str = include_str!("../assets/shaders/stock/_template.wesl");

const LIBRARY_HEADER: &str = "// The watershed editor overwrites this file whenever a document is opened or created; edits here are lost.";

/// The library's file in a document's shader directory. It is never a layer.
pub const LIBRARY_FILE: &str = "lib.wesl";

const ENTRY_POINT: &str = r#"
@compute @workgroup_size(8, 8, 1)
fn generate(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= package::lib::globals.texels.x || id.y >= package::lib::globals.texels.y {
        return;
    }
    package::lib::layer_out[id.y * package::lib::globals.texels.x + id.x] = value(package::lib::cell_position(id.xy));
}
"#;

/// The shaders the editor ships, as `(file name, source)`.
///
/// A name beginning with `_` is a template: it is what a new layer is copied from, and a
/// file of that name in a document's directory is not a layer.
pub const STOCK: [(&str, &str); 8] = [
    ("_template.wesl", TEMPLATE_SOURCE),
    (
        "ridged.wesl",
        include_str!("../assets/shaders/stock/ridged.wesl"),
    ),
    (
        "warped.wesl",
        include_str!("../assets/shaders/stock/warped.wesl"),
    ),
    (
        "terrace.wesl",
        include_str!("../assets/shaders/stock/terrace.wesl"),
    ),
    ("fbm.wesl", include_str!("../assets/shaders/stock/fbm.wesl")),
    (
        "continents.wesl",
        include_str!("../assets/shaders/stock/continents.wesl"),
    ),
    (
        "base.wesl",
        include_str!("../assets/shaders/stock/base.wesl"),
    ),
    (
        "mountains_over_base.wesl",
        include_str!("../assets/shaders/stock/mountains_over_base.wesl"),
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
            if trimmed.is_empty() || trimmed.starts_with("import ") {
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

/// The library as the editor writes it into a document's shader directory: its first
/// line says the editor overwrites the file.
pub fn library_source() -> &'static str {
    static SOURCE: LazyLock<String> = LazyLock::new(|| format!("{LIBRARY_HEADER}\n{LAYER_LIB}"));
    &SOURCE
}

/// Writes the library into `dir` as [`LIBRARY_FILE`], with the `wesl.toml` that lets a
/// WESL language server resolve `package::lib` to it, **overwriting both** whatever they
/// held. Creates the directory.
///
/// Refused at the first write that fails.
pub fn write_library(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    std::fs::write(dir.join(LIBRARY_FILE), library_source())
        .map_err(|error| format!("{LIBRARY_FILE}: {error}"))?;
    std::fs::write(dir.join("wesl.toml"), WESL_TOML).map_err(|error| format!("wesl.toml: {error}"))
}

fn library_shader() -> Shader {
    Shader::from_wesl(library_source(), LIBRARY_FILE)
}

fn layer_shader(source: &str, asset_path: &str) -> Shader {
    Shader::from_wesl(assemble(source), asset_path.to_owned())
}

fn assemble(source: &str) -> String {
    format!("{source}\n{ENTRY_POINT}")
}

/// One import a layer's file makes, one item or module at a time: a collection is
/// reported as one import per name in it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibraryImport {
    /// The full path of the item or module, as `package::lib::fbm_unit`.
    pub path: String,
    /// Whether the path names the library module itself or an item `lib.wesl`
    /// declares. An unresolved import keeps the file from running.
    pub resolved: bool,
}

fn library_items() -> &'static BTreeSet<String> {
    static ITEMS: LazyLock<BTreeSet<String>> = LazyLock::new(|| {
        let unit = LAYER_LIB
            .parse::<TranslationUnit>()
            .expect("lib.wesl does not parse");
        unit.global_declarations
            .iter()
            .filter_map(|declaration| match &**declaration {
                GlobalDeclaration::Declaration(declaration) => Some(declaration.ident.to_string()),
                GlobalDeclaration::TypeAlias(alias) => Some(alias.ident.to_string()),
                GlobalDeclaration::Struct(item) => Some(item.ident.to_string()),
                GlobalDeclaration::Function(function) => Some(function.ident.to_string()),
                _ => None,
            })
            .collect()
    });
    &ITEMS
}

fn resolves(path: &ModulePath) -> bool {
    path.origin == PathOrigin::Absolute
        && match path.components.as_slice() {
            [module] => module == "lib",
            [module, item] => module == "lib" && library_items().contains(item),
            _ => false,
        }
}

fn import_leaves(content: &ImportContent, path: ModulePath, out: &mut Vec<ModulePath>) {
    match content {
        ImportContent::Item(item) => {
            let mut full = path;
            full.push(&item.ident.to_string());
            out.push(full);
        }
        ImportContent::Collection(imports) => {
            for import in imports {
                let path = path.clone().join(import.path.iter().cloned());
                import_leaves(&import.content, path, out);
            }
        }
    }
}

fn read_imports(source: &str) -> Vec<LibraryImport> {
    let Ok(unit) = source.parse::<TranslationUnit>() else {
        return Vec::new();
    };
    let mut paths = Vec::new();
    for statement in &unit.imports {
        match (&statement.path, &statement.content) {
            (Some(path), content) => import_leaves(content, path.clone(), &mut paths),
            (None, ImportContent::Collection(imports)) => {
                for import in imports {
                    let mut components = import.path.iter().cloned();
                    if let Some(package) = components.next() {
                        let path =
                            ModulePath::new(PathOrigin::Package(package), components.collect());
                        import_leaves(&import.content, path, &mut paths);
                    }
                }
            }
            (None, ImportContent::Item(item)) => {
                paths.push(ModulePath::new(
                    PathOrigin::Package(item.ident.to_string()),
                    Vec::new(),
                ));
            }
        }
    }
    paths
        .into_iter()
        .map(|path| LibraryImport {
            resolved: resolves(&path),
            path: path.to_string(),
        })
        .collect()
}

struct Declared {
    layout: ParamsLayout,
    layers: Vec<LayerRead>,
    header: LayerHeader,
}

fn declare(source: &str) -> Result<Declared, String> {
    let layout = parse_params(source).map_err(|error| error.to_string())?;
    let layers = parse_layers(source).map_err(|error| error.to_string())?;
    let header = parse_header(source).map_err(|error| error.to_string())?;
    parse_retired(source).map_err(|error| error.to_string())?;
    if let Some(import) = read_imports(source)
        .into_iter()
        .find(|import| !import.resolved)
    {
        return Err(format!(
            "`{}` is not in {LIBRARY_FILE}; a layer imports from `package::lib`",
            import.path
        ));
    }
    Ok(Declared {
        layout,
        layers,
        header,
    })
}

fn compile_fault(file: &str, description: &str) -> String {
    let lines = || {
        description
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
    };
    let Some(message) = lines().find(|line| line.contains("error")).map(|line| {
        let line = line.strip_prefix("Shader validation ").unwrap_or(line);
        let line = line
            .strip_prefix("Shader '")
            .and_then(|rest| rest.split_once("' parsing "))
            .map_or(line, |(_, after)| after);
        line.strip_prefix("error: ").unwrap_or(line)
    }) else {
        return lines().next().unwrap_or(description).to_owned();
    };
    let own = lines()
        .find_map(|line| line.strip_prefix("--> "))
        .and_then(|at| {
            let mut parts = at.rsplitn(3, ':');
            let _column = parts.next()?;
            let line = parts.next()?.parse::<usize>().ok()?;
            let path = parts.next()?;
            (path.starts_with("watershed/layers/") && path.ends_with(&format!("/{file}")))
                .then_some(line)
        });
    match own {
        Some(line) => format!("line {line}: {message}"),
        None => message.to_owned(),
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
    /// What the file declares about the layer itself. As with the layout, the last
    /// header that parsed, so a file half-way through an edit does not put a layer
    /// back to shift 0.
    pub header: LayerHeader,
    /// Why the file does not run: its annotations did not parse, or its source did not
    /// compile. `line N: message` against the file's own lines, or the bare message when
    /// the fault is not on a line the file owns. `None` when it is good.
    ///
    /// A compile fault arrives a few frames after the file is read, and a fault from the
    /// previous save stays until the new source has compiled or failed.
    ///
    /// The file name is not part of it — every reader already has the name and says it
    /// in its own words.
    pub error: Option<String>,
    /// What the file imports, in the order written. Empty when the source does not
    /// parse; that fault arrives as `error` once the source has been compiled.
    pub imports: Vec<LibraryImport>,
    modified: Option<SystemTime>,
    read_at: Instant,
    handle: Option<Handle<Shader>>,
    asset_path: String,
    key: u64,
    declared: bool,
    compile_fault: Option<u64>,
    settled: bool,
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
/// The root is the project's `shaders` directory, and empty before a project has
/// been opened.
#[derive(Resource, Debug)]
pub struct ShaderLibrary {
    root: PathBuf,
    entries: BTreeMap<String, ShaderEntry>,
    ignored: BTreeSet<String>,
    generation: u64,
    next_asset: u64,
    present: bool,
}

impl Default for ShaderLibrary {
    fn default() -> Self {
        Self {
            root: PathBuf::new(),
            entries: BTreeMap::new(),
            ignored: BTreeSet::new(),
            generation: 0,
            next_asset: 0,
            present: false,
        }
    }
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
/// **deletes every `.wesl` file already in it**, writes the library as in
/// [`write_library`], then writes each layer's `<layer>.wesl` from its stock file.
///
/// Refused at the first removal or write that fails, leaving what was done before it.
pub fn write_preset(dir: &Path, preset: Preset) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|error| error.to_string())?;
    let entries = std::fs::read_dir(dir).map_err(|error| error.to_string())?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "wesl")
        {
            std::fs::remove_file(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        }
    }
    write_library(dir)?;
    for (layer, stock) in preset.files() {
        let source = stock_source(stock)
            .ok_or_else(|| format!("`{stock}` is not a shader this build ships"))?;
        let file = format!("{layer}.wesl");
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
            self.ignored.clear();
            self.generation += 1;
        }
    }

    /// Whether the last scan found the directory at all. A directory that does not
    /// exist says nothing about which layers a document has.
    pub fn present(&self) -> bool {
        self.present
    }

    /// The layers the directory stands for: the stem of every `.wesl` file not
    /// beginning with `_` and not the library's, in name order.
    pub fn layer_names(&self) -> Vec<String> {
        self.entries
            .keys()
            .filter(|name| !name.starts_with('_') && name.as_str() != LIBRARY_FILE)
            .filter_map(|name| name.strip_suffix(".wesl"))
            .map(str::to_owned)
            .collect()
    }
}

#[derive(Resource)]
struct LibraryShader {
    _handle: Handle<Shader>,
}

fn add_library_shader(mut commands: Commands, shaders: Option<ResMut<Assets<Shader>>>) {
    let Some(mut shaders) = shaders else {
        return;
    };
    commands.insert_resource(LibraryShader {
        _handle: shaders.add(library_shader()),
    });
}

/// The systems that keep a document's shaders read, parsed and runnable.
pub struct ShaderPlugin;

impl Plugin for ShaderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShaderLibrary>()
            .add_plugins(dispatch::DispatchPlugin)
            .add_systems(Startup, add_library_shader)
            .add_systems(
                Update,
                (
                    follow_document,
                    scan,
                    dispatch::land_compile_states,
                    attend_shaders,
                )
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
    /// What its header lines declare about the layer itself.
    pub header: LayerHeader,
    /// The fingerprint of `source`: which compiled pipeline a dispatch of this program
    /// runs on.
    pub key: u64,
    file: String,
}

fn program(
    file: &str,
    source: String,
    layout: ParamsLayout,
    layers: Vec<LayerRead>,
    header: LayerHeader,
) -> ShaderProgram {
    ShaderProgram {
        key: fingerprint(source.as_bytes()),
        source,
        layout,
        layers,
        header,
        file: file.to_owned(),
    }
}

struct Runtime {
    sender: DispatchSender,
    programs: BTreeMap<String, ShaderProgram>,
    generation: u64,
    seed: u32,
}

/// What a bake needs to run a shader: the way to the render world, and every program the
/// document's shader directory currently declares.
///
/// Derived state a document carries so that a bake — which is off the main thread and
/// holds the document — can dispatch without reaching back for a resource. Cloning is
/// an `Arc` clone, so a history snapshot costs nothing; two runtimes always compare
/// equal, because what a document *is* does not include the render world it was last run
/// in.
///
/// A default one holds no render world, and running anything through it is an error
/// rather than a panic — which is what a headless test and a document baked before the
/// first sweep both get.
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
    /// The library's entries whose annotations parse as programs, held with the render
    /// world that will run them and the document seed they will be told about. A file
    /// that parses but does not compile is kept; its dispatch answers not compiled.
    pub(crate) fn compile(sender: &DispatchSender, library: &ShaderLibrary, seed: u32) -> Self {
        let programs = library
            .entries
            .iter()
            .filter(|(_, entry)| entry.declared)
            .map(|(name, entry)| {
                (
                    name.clone(),
                    program(
                        name,
                        entry.source.clone(),
                        entry.layout.clone(),
                        entry.layers.clone(),
                        entry.header,
                    ),
                )
            })
            .collect();
        Self(Some(Arc::new(Runtime {
            sender: sender.clone(),
            programs,
            generation: library.generation,
            seed,
        })))
    }

    /// A runtime holding programs built from `sources`, as `(file name, source)`, on
    /// this runtime's render world and at its library generation, telling a dispatch
    /// `seed`. A source whose annotations do not parse is left out.
    ///
    /// Answers a runtime holding no render world when this one holds none.
    pub fn with_sources(
        &self,
        seed: u32,
        sources: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        let Some(runtime) = &self.0 else {
            return Self::default();
        };
        Self(Some(Arc::new(Runtime {
            sender: runtime.sender.clone(),
            programs: programs(sources),
            generation: runtime.generation,
            seed,
        })))
    }

    /// As [`ShaderRuntime::with_sources`], over every `.wesl` file in `dir` but the
    /// library's. A directory that does not exist, or a file that cannot be read,
    /// contributes no source.
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
                if !name.ends_with(".wesl") || name == LIBRARY_FILE {
                    return None;
                }
                let source = std::fs::read_to_string(entry.path()).ok()?;
                Some((name, source))
            });
        self.with_sources(seed, sources)
    }

    /// The library generation these programs were taken at, or `None` for a runtime
    /// holding no render world. What decides that the programs are stale.
    pub fn generation(&self) -> Option<u64> {
        self.0.as_ref().map(|runtime| runtime.generation)
    }

    /// The document seed a dispatch through this runtime tells the shader, or `None`
    /// for a runtime holding no render world. Part of the runtime because a bake has no
    /// other way to reach the document's seed.
    pub fn seed(&self) -> Option<u32> {
        self.0.as_ref().map(|runtime| runtime.seed)
    }

    /// The program for that file, or `None` for a name this runtime does not carry —
    /// a file whose annotations do not parse, or one added since the programs were taken.
    pub fn program(&self, file: &str) -> Option<&ShaderProgram> {
        self.0.as_ref()?.programs.get(file)
    }

    /// Runs one program over the whole of `globals.texels` and answers its values,
    /// row-major, or `None` when the program's source did not run: it failed to compile,
    /// or its file was read again as another source before the dispatch started.
    ///
    /// `layers` is one entry per [`ShaderProgram::layers`], in order; `None` is a layer
    /// that has no raster to hand over and reads `0.0` everywhere. A shorter slice
    /// leaves the layers after it reading `0.0`.
    ///
    /// Blocks until the render world has dispatched and the result has been read back,
    /// so it must not be called on the main thread. Answers an error rather than
    /// panicking when the runtime holds no render world, and when the dispatch does not
    /// answer within a minute.
    pub fn run(
        &self,
        program: &ShaderProgram,
        params: &[u8],
        globals: DispatchGlobals,
        layers: &[Option<&Raster<f32>>],
    ) -> Result<Option<Vec<f32>>, String> {
        let Some(runtime) = &self.0 else {
            return Err("there is no render world to dispatch a shader in".to_owned());
        };
        let inputs = program
            .layers
            .iter()
            .enumerate()
            .map(|(at, read)| LayerInput::new(read.binding, layers.get(at).copied().flatten()))
            .collect();
        runtime
            .sender
            .run(&program.file, program.key, params, globals, inputs)
    }
}

fn programs(
    sources: impl IntoIterator<Item = (String, String)>,
) -> BTreeMap<String, ShaderProgram> {
    sources
        .into_iter()
        .filter_map(|(name, source)| {
            let declared = declare(&source).ok()?;
            let program = program(
                &name,
                source,
                declared.layout,
                declared.layers,
                declared.header,
            );
            Some((name, program))
        })
        .collect()
}

fn follow_document(document: Res<Document>, mut library: ResMut<ShaderLibrary>) {
    if !document.is_changed() {
        return;
    }
    let Some(root) = document.shader_root() else {
        return;
    };
    if library.root() == root {
        return;
    }
    library.look_at(root);
}

fn scan(
    mut library: ResMut<ShaderLibrary>,
    mut document: ResMut<Document>,
    mut shaders: ResMut<Assets<Shader>>,
    bridge: Option<Res<DispatchBridge>>,
) {
    let root = library.root.clone();
    let Ok(dir) = std::fs::read_dir(&root) else {
        library.present = false;
        note_ignored(&mut library.ignored, BTreeSet::new());
        if !library.entries.is_empty() {
            library.entries.clear();
            library.generation += 1;
        }
        return;
    };
    library.present = true;

    let mut seen: Vec<String> = Vec::new();
    let mut wgsl = BTreeSet::new();
    for entry in dir.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".wgsl") {
            wgsl.insert(name);
            continue;
        }
        if !name.ends_with(".wesl") || name == LIBRARY_FILE {
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
        let previous = library.entries.remove(&name);
        let entry = read_entry(
            &name,
            source,
            modified,
            previous,
            &mut shaders,
            bridge.is_none(),
            &mut library.next_asset,
        );
        if let Some(reason) = entry.error.as_ref().filter(|_| !entry.declared) {
            warn!("{name}: {reason}");
            document.refuse(format!("{name}: {reason}"));
        }
        library.entries.insert(name, entry);
    }
    for name in note_ignored(&mut library.ignored, wgsl) {
        warn!("{name}: a .wgsl file is not a layer; a layer is a .wesl file");
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

fn note_ignored(ignored: &mut BTreeSet<String>, present: BTreeSet<String>) -> Vec<String> {
    let arrived = present.difference(ignored).cloned().collect();
    *ignored = present;
    arrived
}

fn read_entry(
    name: &str,
    source: String,
    modified: Option<SystemTime>,
    previous: Option<ShaderEntry>,
    shaders: &mut Assets<Shader>,
    confirmed: bool,
    next_asset: &mut u64,
) -> ShaderEntry {
    let read_at = Instant::now();
    let imports = read_imports(&source);
    match declare(&source) {
        Ok(declared) => {
            let key = fingerprint(source.as_bytes());
            let settled = confirmed
                || previous
                    .as_ref()
                    .is_some_and(|held| held.declared && held.settled && held.key == key);
            let bindings = |layers: &[LayerRead]| -> Vec<u32> {
                layers.iter().map(|read| read.binding).collect()
            };
            let reusable = previous
                .as_ref()
                .filter(|held| bindings(&held.layers) == bindings(&declared.layers))
                .and_then(|held| Some((held.handle.clone()?, held.asset_path.clone())));
            let (handle, asset_path) = match reusable {
                Some((handle, asset_path)) => {
                    if let Err(error) = shaders.insert(&handle, layer_shader(&source, &asset_path))
                    {
                        warn!("{name}: {error}");
                    }
                    (handle, asset_path)
                }
                None => {
                    let asset_path = format!("watershed/layers/v{next_asset}/{name}");
                    *next_asset += 1;
                    (shaders.add(layer_shader(&source, &asset_path)), asset_path)
                }
            };
            let (error, compile_fault) = match previous {
                Some(ShaderEntry {
                    error: Some(error),
                    compile_fault: Some(at),
                    ..
                }) => (Some(error), Some(at)),
                _ => (None, None),
            };
            ShaderEntry {
                source,
                layout: declared.layout,
                layers: declared.layers,
                header: declared.header,
                error,
                imports,
                modified,
                read_at,
                handle: Some(handle),
                asset_path,
                key,
                declared: true,
                compile_fault,
                settled,
            }
        }
        Err(reason) => {
            let settled = previous.as_ref().is_some_and(|held| held.settled);
            let (layout, layers, header, handle, asset_path, key) = previous
                .map(|held| {
                    (
                        held.layout,
                        held.layers,
                        held.header,
                        held.handle,
                        held.asset_path,
                        held.key,
                    )
                })
                .unwrap_or_default();
            ShaderEntry {
                source,
                layout,
                layers,
                header,
                error: Some(reason),
                imports,
                modified,
                read_at,
                handle,
                asset_path,
                key,
                declared: false,
                compile_fault: None,
                settled,
            }
        }
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
    bridge: Option<Res<DispatchBridge>>,
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
    if stale && let Some(bridge) = bridge {
        let next = ShaderRuntime::compile(bridge.sender(), &library, document.seed);
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
        let Some(entry) = library.entry(&layer.file()).filter(|entry| entry.settled) else {
            continue;
        };
        touched |= layer.shader.reconcile(&entry.layout);
        touched |= layer.shader.reconcile_layers(&entry.layers);
        touched |= layer.reconcile_header(&entry.header);
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
                header: LayerHeader::default(),
                error: Some(fault.to_owned()),
                imports: Vec::new(),
                modified: None,
                read_at: Instant::now(),
                handle: None,
                asset_path: String::new(),
                key: 0,
                declared: true,
                compile_fault: Some(0),
                settled: true,
            },
        );
        library.generation = 1;
        library
    }

    /// A library holding `file` read from `source` as a scan with nothing to compile
    /// it would read it, for tests that need a read entry without a directory.
    pub fn reading(file: &str, source: &str) -> Self {
        let mut library = Self::default();
        let mut shaders = Assets::<Shader>::default();
        let entry = read_entry(
            file,
            source.to_owned(),
            None,
            None,
            &mut shaders,
            true,
            &mut library.next_asset,
        );
        library.entries.insert(file.to_owned(), entry);
        library.generation = 1;
        library
    }
}

#[cfg(test)]
mod tests {
    use bevy::asset::AssetId;
    use bevy::asset::uuid::Uuid;
    use bevy::shader::{ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
    use wgpu::naga;

    use super::*;

    fn described(report: String) -> String {
        format!("Validation Error\n\nCaused by:\n  In Device::create_shader_module\n    {report}\n")
    }

    fn validated(
        _: &(),
        source: ShaderCacheSource,
        _: &ValidateShader,
    ) -> Result<(), ShaderCacheError> {
        let ShaderCacheSource::Wgsl(wgsl) = source else {
            panic!("a layer compiled to something other than WGSL");
        };
        let module = naga::front::wgsl::parse_str(&wgsl).map_err(|error| {
            let report = naga::error::ShaderError {
                source: wgsl.to_string(),
                label: None,
                inner: Box::new(error),
            };
            ShaderCacheError::CreateShaderModule(described(report.to_string()))
        })?;
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&module).map(|_| ()).map_err(|error| {
            let report = naga::error::ShaderError {
                source: wgsl.to_string(),
                label: None,
                inner: Box::new(error),
            };
            ShaderCacheError::CreateShaderModule(described(report.to_string()))
        })
    }

    fn compiles(file: &str, source: &str) -> Result<(), String> {
        let id = |n| AssetId::<Shader>::Uuid {
            uuid: Uuid::from_u128(n),
        };
        let mut cache = ShaderCache::<(), ()>::new((), validated);
        cache.set_shader(id(1), library_shader());
        cache.set_shader(
            id(2),
            layer_shader(source, &format!("watershed/layers/v0/{file}")),
        );
        match cache.get(0, id(2), &[]) {
            Ok(_) => Ok(()),
            Err(
                ShaderCacheError::ProcessShaderError(description)
                | ShaderCacheError::CreateShaderModule(description),
            ) => Err(compile_fault(file, &description)),
            Err(error) => panic!("{file} never reached a compile: {error}"),
        }
    }

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

    // A file's imports have to precede every declaration, so the entry point can only
    // be appended to the file, never put ahead of it.
    #[test]
    fn the_assembled_source_starts_with_the_file_and_ends_with_the_entry_point() {
        let source = STOCK[1].1;
        let assembled = assemble(source);
        assert!(assembled.starts_with(source));
        let generate = assembled.rfind("fn generate").unwrap();
        assert!(generate > source.len());
        assert!(!assembled[generate..].contains("fn value"));
    }

    // The library, the entry point and every shipped shader are compiled the way the
    // pipeline compiles them, and this is the only thing that says they still do — the
    // dispatch itself needs a GPU, and this does not.
    #[test]
    fn every_stock_shader_compiles_against_the_library_and_the_entry_point() {
        for (name, source) in STOCK {
            if let Err(error) = compiles(name, source) {
                panic!("{name} does not compile: {error}");
            }
        }
    }

    // The acceptance is that the header alone is enough to write a shader, so what the
    // panel renders has to be prose rather than a commented file, and it has to reach
    // every grammar a file may use, the import included.
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
            "import package::lib",
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

    // Both import forms the header gives, compiled: the acceptance observation short of
    // a GPU. The header's own examples must not be read as this file's declaration — a
    // layer example would make every layer copied from the template read `base`.
    #[test]
    fn the_headers_worked_examples_compile_and_its_examples_declare_nothing() {
        let items = "import package::lib::{fbm_unit, uv};\n\nfn value(p: vec2<f32>) -> f32 {\n    return fbm_unit(uv(p) * 4.0, 4u, 0.5, 2.0);\n}\n";
        let module = "import package::lib;\n\nfn value(p: vec2<f32>) -> f32 {\n    return lib::fbm_unit(lib::uv(p) * 4.0, 4u, 0.5, 2.0);\n}\n";
        for source in [items, module] {
            if let Err(error) = compiles("height.wesl", source) {
                panic!("the header's worked example does not compile: {error}\n{source}");
            }
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
        let source = "import package::lib::{layer_shift, layer_value};\n\n@group(0) @binding(3) var base: texture_2d<f32>; // @layer base\n\nfn value(p: vec2<f32>) -> f32 {\n    return layer_value(base, p) + f32(layer_shift(base));\n}\n";
        assert_eq!(parse_layers(source).unwrap().len(), 1);
        if let Err(error) = compiles("height.wesl", source) {
            panic!("a read of a named layer does not compile: {error}");
        }
    }

    // A layer says where every library function it calls comes from, so a call with no
    // import has to fail, and the card has to name what was called.
    #[test]
    fn a_library_function_called_without_an_import_does_not_compile() {
        let error = compiles(
            "height.wesl",
            "fn value(p: vec2<f32>) -> f32 {\n    return fbm_unit(p, 4u, 0.5, 2.0);\n}\n",
        )
        .expect_err("a library call with no import compiled");
        assert!(error.contains("fbm_unit"), "{error}");
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

    // The card shows a syntax fault against the file the author is looking at, not the
    // assembled source.
    #[test]
    fn a_parse_fault_is_reported_against_the_shaders_own_lines() {
        let error = compiles(
            "height.wesl",
            "fn value(p: vec2<f32>) -> f32 {\n    return p.x + ;\n}\n",
        )
        .expect_err("a dangling operator compiled");
        assert!(error.starts_with("line 2:"), "{error}");
    }

    // A name that resolves to nothing is found before the source is lowered to WGSL,
    // so it still carries the file's own line.
    #[test]
    fn a_reference_to_nothing_is_reported_against_the_shaders_own_lines() {
        let error = compiles(
            "height.wesl",
            "fn value(p: vec2<f32>) -> f32 {\n    return oops;\n}\n",
        )
        .expect_err("a reference to nothing compiled");
        assert!(error.starts_with("line 2:"), "{error}");
        assert!(error.contains("oops"), "{error}");
    }

    // A fault only the validator finds is found in the compiled WGSL, whose lines are
    // not the file's, so the card gets the message without a line it would get wrong.
    #[test]
    fn a_validation_fault_is_reported_without_a_line() {
        let error = compiles(
            "height.wesl",
            "fn value(p: vec2<f32>) -> f32 {\n    return p;\n}\n",
        )
        .expect_err("returning a vector from a scalar function compiled");
        assert!(!error.starts_with("line "), "{error}");
        assert!(!error.contains("Shader"), "{error}");
    }

    // The check that keeps a bad import off the pipeline is only as good as its reading
    // of the import grammar: collections and the module form reach the library, and
    // nothing else does.
    #[test]
    fn imports_resolve_only_to_the_library_module_and_what_it_declares() {
        let source = "import package::lib::{fbm_unit, uv};\nimport package::lib;\nimport package::lib::nonesuch;\nimport super::lib::uv;\nimport other::x;\n\nfn value(p: vec2<f32>) -> f32 {\n    return 0.0;\n}\n";
        let read: Vec<(String, bool)> = read_imports(source)
            .into_iter()
            .map(|import| (import.path, import.resolved))
            .collect();
        let expected = [
            ("package::lib::fbm_unit", true),
            ("package::lib::uv", true),
            ("package::lib", true),
            ("package::lib::nonesuch", false),
            ("super::lib::uv", false),
            ("other::x", false),
        ]
        .map(|(path, resolved)| (path.to_owned(), resolved));
        assert_eq!(read, expected);
    }

    // An import the library cannot answer would leave the pipeline waiting on a module
    // that never arrives, so the file is refused before it reaches one.
    #[test]
    fn a_file_importing_what_the_library_does_not_declare_is_refused_naming_the_import() {
        let source = "import package::lib::{fbm_unit, nonesuch};\n\nfn value(p: vec2<f32>) -> f32 {\n    return 0.0;\n}\n";
        let error = declare(source)
            .err()
            .expect("an unresolved import was accepted");
        assert!(error.contains("`package::lib::nonesuch`"), "{error}");
    }

    // A `.wgsl` left in the directory is seen on every scan, and the log has to name it
    // once rather than on every frame.
    #[test]
    fn a_wgsl_file_is_noted_once_and_again_only_after_it_left() {
        let mut ignored = BTreeSet::new();
        let old = || BTreeSet::from(["old.wgsl".to_owned()]);
        assert_eq!(note_ignored(&mut ignored, old()), ["old.wgsl"]);
        assert!(note_ignored(&mut ignored, old()).is_empty());
        assert!(note_ignored(&mut ignored, BTreeSet::new()).is_empty());
        assert_eq!(note_ignored(&mut ignored, old()), ["old.wgsl"]);
    }

    // `lib.wesl` and a template sit beside the layers in one directory, and neither may
    // come back as a layer of the document.
    #[test]
    fn neither_the_library_nor_a_template_is_a_layer() {
        let mut library = ShaderLibrary::reading(
            "height.wesl",
            "fn value(p: vec2<f32>) -> f32 {\n    return 0.0;\n}\n",
        );
        let entry = library.entries["height.wesl"].clone();
        for file in [LIBRARY_FILE, "_template.wesl"] {
            library.entries.insert(file.to_owned(), entry.clone());
        }
        assert_eq!(library.layer_names(), ["height"]);
    }

    // Programs are built before the GPU has compiled anything, so only a declaration can keep a file out; a compile fault is the dispatch's to answer.
    #[test]
    fn programs_leave_out_a_retired_annotation_and_keep_a_source_that_only_fails_to_compile() {
        let built = programs([
            (
                "fbm.wesl".to_owned(),
                stock_source("fbm.wesl").unwrap().to_owned(),
            ),
            (
                "bad.wesl".to_owned(),
                "fn value(p: vec2<f32>) -> f32 { return nonesuch(p); }\n".to_owned(),
            ),
            (
                "retired.wesl".to_owned(),
                "// @reach 2\nfn value(p: vec2<f32>) -> f32 {\n    return 0.0;\n}\n".to_owned(),
            ),
        ]);
        assert!(built.contains_key("fbm.wesl"));
        assert!(built.contains_key("bad.wesl"));
        assert!(!built.contains_key("retired.wesl"));
        assert_eq!(
            built["bad.wesl"].key,
            fingerprint(built["bad.wesl"].source.as_bytes())
        );
    }

    // Every pipeline built from a shader asset is recompiled when it changes, so a file whose `@layer`s changed must not keep the asset an older layout was built from — that recompile is refused and quits the editor. A new asset also needs a module path of its own, or two assets would answer for one module.
    #[test]
    fn a_file_keeps_its_shader_asset_only_while_its_layer_bindings_stay_the_same() {
        let mut shaders = Assets::<Shader>::default();
        let mut next = 0;
        let reads = "import package::lib::layer_value;\n\n@group(0) @binding(3) var base: texture_2d<f32>; // @layer base\n\nfn value(p: vec2<f32>) -> f32 {\n    return layer_value(base, p);\n}\n";
        let tuned = "import package::lib::layer_value;\n\n@group(0) @binding(3) var base: texture_2d<f32>; // @layer base\n\nfn value(p: vec2<f32>) -> f32 {\n    return layer_value(base, p) * 0.5;\n}\n";
        let first = read_entry(
            "height.wesl",
            reads.to_owned(),
            None,
            None,
            &mut shaders,
            false,
            &mut next,
        );
        let same = read_entry(
            "height.wesl",
            tuned.to_owned(),
            None,
            Some(first.clone()),
            &mut shaders,
            false,
            &mut next,
        );
        assert_eq!(same.handle, first.handle);
        let emptied = read_entry(
            "height.wesl",
            String::new(),
            None,
            Some(same.clone()),
            &mut shaders,
            false,
            &mut next,
        );
        assert!(emptied.declared);
        assert_ne!(emptied.handle, same.handle);
        let path = |entry: &ShaderEntry, shaders: &Assets<Shader>| {
            shaders
                .get(entry.handle.as_ref().unwrap())
                .unwrap()
                .path
                .clone()
        };
        assert_ne!(path(&emptied, &shaders), path(&same, &shaders));
        let restored = read_entry(
            "height.wesl",
            reads.to_owned(),
            None,
            Some(emptied.clone()),
            &mut shaders,
            false,
            &mut next,
        );
        assert_ne!(restored.handle, emptied.handle);
    }

    // A save that has not compiled yet — a half-written file, say — must not reconcile its layer's parameter values away against what that source declares.
    #[test]
    fn a_read_source_is_settled_only_once_it_has_compiled_unless_nothing_will_compile_it() {
        let mut shaders = Assets::<Shader>::default();
        let mut next = 0;
        let fbm = stock_source("fbm.wesl").unwrap();
        let first = read_entry(
            "fbm.wesl",
            fbm.to_owned(),
            None,
            None,
            &mut shaders,
            false,
            &mut next,
        );
        assert!(!first.settled);
        let compiled = ShaderEntry {
            settled: true,
            ..first
        };
        let touched = read_entry(
            "fbm.wesl",
            fbm.to_owned(),
            None,
            Some(compiled.clone()),
            &mut shaders,
            false,
            &mut next,
        );
        assert!(touched.settled);
        let emptied = read_entry(
            "fbm.wesl",
            String::new(),
            None,
            Some(compiled),
            &mut shaders,
            false,
            &mut next,
        );
        assert!(emptied.declared && !emptied.settled);
        let headless = read_entry(
            "fbm.wesl",
            String::new(),
            None,
            None,
            &mut shaders,
            true,
            &mut next,
        );
        assert!(headless.settled);
    }

    // Headless tests and a document built before the editor has a render world both
    // start from a runtime holding none, and building from sources must not pretend
    // to have one.
    #[test]
    fn a_runtime_built_from_sources_on_one_holding_no_bridge_holds_none() {
        let built = ShaderRuntime::default().with_sources(
            7,
            [(
                "fbm.wesl".to_owned(),
                stock_source("fbm.wesl").unwrap().to_owned(),
            )],
        );
        assert_eq!(built.generation(), None);
        assert!(built.program("fbm.wesl").is_none());
    }

    // A preset's `new` has to leave exactly its own layers' files on disk, as this
    // build ships them: a file left from an earlier document would become a layer of
    // this one, and a library edited by hand would not be the one that compiles.
    #[test]
    fn writing_a_preset_removes_stale_files_and_writes_the_library_and_one_file_per_layer() {
        let dir = std::env::temp_dir().join(format!("watershed-preset-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("stale.wesl"), "edited").unwrap();
        std::fs::write(dir.join(LIBRARY_FILE), "edited").unwrap();
        std::fs::write(dir.join("notes.txt"), "kept").unwrap();

        write_preset(&dir, Preset::Ridges).unwrap();
        assert!(!dir.join("stale.wesl").exists());
        assert!(dir.join("notes.txt").is_file());
        let library = std::fs::read_to_string(dir.join(LIBRARY_FILE)).unwrap();
        assert_eq!(library.lines().next(), Some(LIBRARY_HEADER));
        assert_eq!(library, library_source());
        assert!(dir.join("wesl.toml").is_file());
        assert_eq!(
            std::fs::read_to_string(dir.join("base.wesl")).unwrap(),
            stock_source("base.wesl").unwrap()
        );
        assert!(dir.join("height.wesl").is_file() && dir.join("moisture.wesl").is_file());
        std::fs::remove_dir_all(&dir).unwrap();
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
