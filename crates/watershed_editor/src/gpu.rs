//! Running a shader layer: what WGSL a document carries, what each file declares,
//! and the dispatch that turns one into the raster its layer reads.
//!
//! A shader is resolved to a whole raster *before* a field's stack is walked, because
//! the walk is a per-texel CPU function and a dispatch cannot join it. What lands in
//! the layer is then read exactly as a painted raster is, which is why every other
//! node around it — the scale on it, the lerp it feeds — needs no arm for this.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroupEntry, BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor,
    BufferInitDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, MapMode,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, RawComputePipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use glam::UVec2;
use watershed::raster::{Raster, resolution};

use crate::document::Document;
use crate::terrain::graph::{NodeId, NodeOp};
use crate::terrain::shader::{ParamsLayout, SHADER_DIR, parse_params};

/// The source every shader layer is compiled against: the bindings a dispatch
/// supplies, the noise the CPU layers agree with, and the position helper the entry
/// point uses.
const FIELD_LIB: &str = include_str!("../assets/shaders/field_lib.wgsl");

/// The entry point, appended after the shader's own source because WGSL has no
/// forward declaration and an entry point that came first could not call a `value`
/// declared after it.
const ENTRY_POINT: &str = r#"
@compute @workgroup_size(8, 8, 1)
fn generate(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= globals.texels.x || id.y >= globals.texels.y {
        return;
    }
    field_out[id.y * globals.texels.x + id.x] = value(cell_position(id.xy));
}
"#;

/// The shaders the editor ships, as `(file name, source)`.
///
/// A name beginning with `_` is a template: it is copied like any other but is not
/// offered as something to add.
pub const STOCK: [(&str, &str); 4] = [
    (
        "_template.wgsl",
        include_str!("../assets/shaders/stock/_template.wgsl"),
    ),
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
];

/// The assembled source a shader is compiled from: the library, the file's own
/// source, then the entry point — in that order, because WGSL has no forward
/// declaration and neither the library's bindings nor the entry point's call to
/// `value` would resolve in any other.
fn assemble(source: &str) -> String {
    format!("{FIELD_LIB}\n{source}\n{ENTRY_POINT}")
}

/// How many lines the library adds ahead of a shader's own first line, so a fault the
/// compiler reports against the assembled source can be reported against the file the
/// author is looking at.
fn library_lines() -> usize {
    FIELD_LIB.lines().count() + 1
}

/// Compiles the assembled source far enough to know whether the GPU would take it.
///
/// This runs before the device sees anything, and it is the reason a shader with a
/// typo in it does not take the editor down: wgpu reports a compile fault through a
/// handler that panics, so a shader has to be known good before it is handed over.
///
/// A fault is reported against the shader's own line numbering, not the assembled
/// source's.
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
    /// Why the file did not parse or compile: `line N: message` against the file's own
    /// lines, or the bare message when the fault is not on a line the file owns.
    /// `None` when it is good.
    ///
    /// The file name is not part of it — every reader already has the name and says it
    /// in its own words.
    pub error: Option<String>,
    modified: Option<SystemTime>,
    generation: u64,
}

/// What a shader is told about where it is being evaluated.
///
/// Laid out as the `Globals` struct in `field_lib.wgsl`, which is the other half of
/// this type. A change here is a change there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchGlobals {
    /// The document's extent, in cells.
    pub document: UVec2,
    /// The extent of this dispatch, in texels of the field's raster.
    pub texels: UVec2,
    /// Where the dispatch starts, in texels of the field's raster.
    pub origin: UVec2,
    /// The field's raster shift.
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
/// scratch directory before that — so a shader layer can be added to a document that
/// has never been written, and the first save moves the directory in whole.
#[derive(Resource, Debug)]
pub struct ShaderLibrary {
    root: PathBuf,
    entries: BTreeMap<String, ShaderEntry>,
    generation: u64,
}

impl Default for ShaderLibrary {
    fn default() -> Self {
        Self {
            root: scratch_root(),
            entries: BTreeMap::new(),
            generation: 0,
        }
    }
}

fn scratch_root() -> PathBuf {
    std::env::temp_dir().join(format!("watershed-shaders-{}", std::process::id()))
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

    /// Copies a stock shader into the document's directory under the first name not
    /// already taken, and answers that name.
    ///
    /// The copy is the document's from that moment: editing it changes this document
    /// and no other, which is the whole reason a shader lives beside the terrain it
    /// belongs to.
    pub fn adopt(&mut self, stock: &str) -> Result<String, String> {
        let source = STOCK
            .iter()
            .find(|(name, _)| *name == stock)
            .map(|(_, source)| *source)
            .ok_or_else(|| format!("`{stock}` is not a shader this build ships"))?;
        std::fs::create_dir_all(&self.root).map_err(|error| error.to_string())?;

        let stem = stock.trim_end_matches(".wgsl").trim_start_matches('_');
        let mut file = format!("{stem}.wgsl");
        let mut suffix = 1;
        while self.root.join(&file).exists() {
            suffix += 1;
            file = format!("{stem}_{suffix}.wgsl");
        }
        std::fs::write(self.root.join(&file), source).map_err(|error| error.to_string())?;
        Ok(file)
    }
}

/// The systems that keep a document's shaders read, parsed and resolved.
pub struct ShaderPlugin;

impl Plugin for ShaderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShaderLibrary>()
            .init_resource::<Resolved>()
            .add_systems(Update, (follow_document, scan, resolve).chain());
    }
}

/// What each shader layer was last resolved from, so a dispatch happens when
/// something it depends on moved and not once a frame.
#[derive(Resource, Default)]
struct Resolved(BTreeMap<(String, NodeId), Stamp>);

#[derive(Clone, Copy, PartialEq, Eq)]
struct Stamp {
    generation: u64,
    params: u64,
    texels: UVec2,
}

fn follow_document(document: Res<Document>, mut library: ResMut<ShaderLibrary>) {
    if !document.is_changed() {
        return;
    }
    let root = match &document.path {
        Some(path) => path.join(SHADER_DIR),
        None => scratch_root(),
    };
    if library.root() == root {
        return;
    }
    if library.root() == scratch_root() {
        carry_scratch_in(&scratch_root(), &root);
    }
    library.look_at(root);
}

/// Moves the shaders a document wrote before it had a directory into the one it has
/// now, so a node added to an unsaved document survives the first save.
///
/// A file the destination already holds is left alone: the document's own copy is the
/// one its layers name.
fn carry_scratch_in(scratch: &Path, root: &Path) {
    let Ok(entries) = std::fs::read_dir(scratch) else {
        return;
    };
    if std::fs::create_dir_all(root).is_err() {
        return;
    }
    for entry in entries.flatten() {
        let name = entry.file_name();
        let destination = root.join(&name);
        if destination.exists() {
            continue;
        }
        if std::fs::rename(entry.path(), &destination).is_err() {
            let _ = std::fs::copy(entry.path(), &destination);
        }
    }
}

fn scan(mut library: ResMut<ShaderLibrary>, mut document: ResMut<Document>) {
    let root = library.root.clone();
    let Ok(dir) = std::fs::read_dir(&root) else {
        if !library.entries.is_empty() {
            library.entries.clear();
            library.generation += 1;
        }
        return;
    };

    let mut seen: Vec<String> = Vec::new();
    let mut changed: Vec<String> = Vec::new();
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
        let generation = library.generation;
        let previous = library.entries.get(&name).map(|held| held.layout.clone());
        let outcome = parse_params(&source)
            .map_err(|error| error.to_string())
            .and_then(|layout| validate(&source).map(|()| layout));
        let entry = match outcome {
            Ok(layout) => ShaderEntry {
                source,
                layout,
                error: None,
                modified,
                generation,
            },
            Err(reason) => ShaderEntry {
                source,
                layout: previous.unwrap_or_default(),
                error: Some(reason),
                modified,
                generation,
            },
        };
        if let Some(reason) = &entry.error {
            document.refuse(format!("{name}: {reason}"));
        }
        library.entries.insert(name.clone(), entry);
        changed.push(name);
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

    if changed.is_empty() {
        return;
    }
    let Some(terrain) = document.terrain_mut() else {
        return;
    };
    let mut touched = false;
    for field in &mut terrain.fields {
        for node in &mut field.graph.nodes {
            let NodeOp::Shader(shader) = &mut node.op else {
                continue;
            };
            if !changed.contains(&shader.file) {
                continue;
            }
            if let Some(entry) = library.entries.get(&shader.file) {
                shader.reconcile(&entry.layout);
                touched = true;
            }
        }
    }
    if touched {
        document.note_edit();
    }
}

fn resolve(
    device: Option<Res<RenderDevice>>,
    queue: Option<Res<RenderQueue>>,
    library: Res<ShaderLibrary>,
    mut resolved: ResMut<Resolved>,
    mut document: ResMut<Document>,
) {
    let (Some(device), Some(queue)) = (device, queue) else {
        return;
    };
    if document.is_busy() {
        return;
    }
    let size = document.size;
    let seed = document.seed;
    let Some(terrain) = document.terrain() else {
        return;
    };

    let mut work: Vec<(String, NodeId, Stamp, String, Vec<u8>, DispatchGlobals)> = Vec::new();
    for field in &terrain.fields {
        for node in &field.graph.nodes {
            let NodeOp::Shader(shader) = &node.op else {
                continue;
            };
            let Some(entry) = library.entry(&shader.file) else {
                continue;
            };
            if entry.error.is_some() {
                continue;
            }
            let texels = resolution(size, field.shift);
            let params = entry.layout.pack(&shader.params);
            let stamp = Stamp {
                generation: entry.generation,
                params: fingerprint(&params),
                texels,
            };
            let key = (field.id.as_str().to_owned(), node.id);
            if resolved.0.get(&key) == Some(&stamp) && !shader.values().is_empty() {
                continue;
            }
            work.push((
                key.0,
                node.id,
                stamp,
                entry.source.clone(),
                params,
                DispatchGlobals {
                    document: size,
                    texels,
                    origin: UVec2::ZERO,
                    shift: field.shift as u32,
                    seed,
                },
            ));
        }
    }
    if work.is_empty() {
        return;
    }

    let mut produced: Vec<((String, NodeId), Stamp, Raster<f32>)> = Vec::new();
    let mut failure = None;
    for (name, id, stamp, source, params, globals) in work {
        match dispatch(&device, &queue, &source, &params, globals) {
            Ok(values) => match Raster::from_vec(globals.texels, values) {
                Some(raster) => produced.push(((name, id), stamp, raster)),
                None => failure = Some("a dispatch produced the wrong number of texels".to_owned()),
            },
            Err(error) => failure = Some(error),
        }
    }

    if produced.is_empty() {
        if let Some(error) = failure {
            document.refuse(error);
        }
        return;
    }

    let Some(terrain) = document.terrain_mut() else {
        return;
    };
    for ((name, id), stamp, raster) in produced {
        let Some(field) = terrain.field_mut(&name) else {
            continue;
        };
        let Some(node) = field.graph.node_mut(id) else {
            continue;
        };
        let NodeOp::Shader(shader) = &mut node.op else {
            continue;
        };
        shader.put_values(raster);
        resolved.0.insert((name, id), stamp);
    }
    document.note_edit();
    if let Some(error) = failure {
        document.refuse(error);
    }
}

fn fingerprint(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// One shader run over one rectangle, read back into the values its layer holds.
///
/// Blocks until the GPU has finished and the result has been mapped: a shader layer
/// has to be resolved before the field's stack is walked, and the walk is synchronous.
///
/// The shader is compiled from the library, the file's own source and the entry
/// point, in that order, because WGSL has no forward declaration.
pub fn dispatch(
    device: &RenderDevice,
    queue: &RenderQueue,
    source: &str,
    params: &[u8],
    globals: DispatchGlobals,
) -> Result<Vec<f32>, String> {
    let texels = (globals.texels.x as u64) * (globals.texels.y as u64);
    if texels == 0 {
        return Ok(Vec::new());
    }
    let bytes = texels * 4;
    let assembled = format!("{FIELD_LIB}\n{source}\n{ENTRY_POINT}");

    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("watershed field shader"),
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
    let layout = device.create_bind_group_layout(
        Some("watershed field shader"),
        &[
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
        ],
    );

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

    let bind_group = device.create_bind_group(
        Some("watershed field shader"),
        &layout,
        &[
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
        ],
    );

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("watershed field shader"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("watershed field shader"),
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
        label: Some("watershed field shader"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("watershed field shader"),
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
                error: Some(fault.to_owned()),
                modified: None,
                generation: 1,
            },
        );
        library.generation = 1;
        library
    }
}

/// Every stock shader parses, which is the only way a copied one arrives with a
/// panel rather than an error.
#[cfg(test)]
mod tests {
    use super::*;

    // The stock shaders are what a new shader layer is copied from, so one that does
    // not parse would hand the user a broken layer on the first click.
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
