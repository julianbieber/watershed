//! Carrying a layer dispatch from the bake's thread through the render world and back.

use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use bevy::asset::RenderAssetUsages;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::render::gpu_readback::{ReadbackComplete, ReadbackOnce};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_sized, texture_2d, uniform_buffer_sized,
};
use bevy::render::render_resource::{
    BindGroupEntry, BindGroupLayoutDescriptor, BufferInitDescriptor, BufferUsages,
    CachedComputePipelineId, CachedPipelineState, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, Extent3d, IntoBinding, PipelineCache, ShaderStages,
    TextureDimension, TextureFormat, TextureSampleType,
};
use bevy::render::renderer::{RenderDevice, RenderQueue, render_system};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::texture::GpuImage;
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};
use bevy::shader::{Shader, ShaderCacheError};
use glam::UVec2;
use watershed::raster::Raster;

use super::{DispatchGlobals, ShaderLibrary, compile_fault};

const DISPATCH_TIMEOUT: Duration = Duration::from_secs(60);

type Answer = Result<Option<Vec<f32>>, String>;

/// One layer a shader reads, as the texels its binding is given.
pub(crate) struct LayerInput {
    binding: u32,
    size: UVec2,
    bytes: Vec<u8>,
}

impl LayerInput {
    /// `raster` bound at `binding`. No raster, or an empty one, is bound as a single
    /// texel of `0.0`, so a layer with nothing to hand over reads `0.0` everywhere.
    pub(crate) fn new(binding: u32, raster: Option<&Raster<f32>>) -> Self {
        match raster.filter(|held| !held.is_empty()) {
            Some(raster) => Self {
                binding,
                size: raster.size(),
                bytes: raster
                    .data()
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            },
            None => Self {
                binding,
                size: UVec2::ONE,
                bytes: 0.0f32.to_le_bytes().to_vec(),
            },
        }
    }

    fn image(self) -> Image {
        Image::new(
            Extent3d {
                width: self.size.x,
                height: self.size.y,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            self.bytes,
            TextureFormat::R32Float,
            RenderAssetUsages::RENDER_WORLD,
        )
    }
}

struct DispatchRequest {
    file: String,
    key: u64,
    params: Vec<u8>,
    globals: DispatchGlobals,
    inputs: Vec<LayerInput>,
    sent: Instant,
    reply: Sender<Answer>,
}

/// The bake's end of the bridge: what hands a dispatch to the editor and waits for its
/// values.
#[derive(Clone)]
pub(crate) struct DispatchSender(Sender<DispatchRequest>);

impl DispatchSender {
    /// Runs `file`, as the source whose fingerprint is `key`, over the whole of
    /// `globals.texels`, and answers its values row-major — or `None` when that source
    /// did not run: its pipeline failed to compile, or the file was read again as another
    /// source before the dispatch started.
    ///
    /// Blocks until the render world has dispatched and the values have been read back.
    /// Answers an error when the editor stops first, or when no pipeline for that source
    /// is ready within a minute, which is what a file removed mid-bake gets.
    pub(crate) fn run(
        &self,
        file: &str,
        key: u64,
        params: &[u8],
        globals: DispatchGlobals,
        inputs: Vec<LayerInput>,
    ) -> Answer {
        if globals.texels.x == 0 || globals.texels.y == 0 {
            return Ok(Some(Vec::new()));
        }
        let (reply, answer) = mpsc::channel();
        let params = if params.is_empty() {
            vec![0; 16]
        } else {
            params.to_vec()
        };
        self.0
            .send(DispatchRequest {
                file: file.to_owned(),
                key,
                params,
                globals,
                inputs,
                sent: Instant::now(),
                reply,
            })
            .map_err(|_| format!("{file}: the editor stopped before the shader was dispatched"))?;
        match answer.recv_timeout(DISPATCH_TIMEOUT) {
            Ok(answer) => answer,
            Err(RecvTimeoutError::Timeout) => Err(format!(
                "{file}: the shader dispatch did not answer within {} seconds",
                DISPATCH_TIMEOUT.as_secs()
            )),
            Err(RecvTimeoutError::Disconnected) => Err(format!(
                "{file}: the shader dispatch was dropped before it answered"
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum PipelineStatus {
    Pending,
    Ready,
    Failed(String),
}

struct GpuJob {
    id: u64,
    file: String,
    key: u64,
    globals: [u8; 32],
    params: Vec<u8>,
    inputs: Vec<(u32, Handle<Image>)>,
    output: Handle<ShaderBuffer>,
    texels: UVec2,
}

#[derive(Default)]
struct Shared {
    states: HashMap<String, (u64, PipelineStatus)>,
    ready: Vec<GpuJob>,
    dispatched: Vec<u64>,
    failed: Vec<u64>,
}

#[derive(Resource, Clone, Default)]
struct SharedDispatch(Arc<Mutex<Shared>>);

impl SharedDispatch {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The main world's end of the bridge. Present only when there is a render world to
/// dispatch in.
#[derive(Resource)]
pub(crate) struct DispatchBridge {
    sender: DispatchSender,
    requests: Mutex<Receiver<DispatchRequest>>,
    shared: SharedDispatch,
}

impl DispatchBridge {
    /// What a runtime hands its dispatches to.
    pub(crate) fn sender(&self) -> &DispatchSender {
        &self.sender
    }
}

struct Carried {
    reply: Sender<Answer>,
    output: Handle<ShaderBuffer>,
}

#[derive(Resource, Default)]
struct InFlight {
    next: u64,
    waiting: Vec<DispatchRequest>,
    jobs: HashMap<u64, Carried>,
}

#[derive(Debug, PartialEq, Eq)]
enum Fate {
    Forward,
    Wait,
    Superseded,
}

fn fate(request_key: u64, library_key: Option<u64>, state_key: Option<u64>, reread: bool) -> Fate {
    if library_key == Some(request_key) && state_key == Some(request_key) {
        Fate::Forward
    } else if reread && library_key.is_some_and(|held| held != request_key) {
        Fate::Superseded
    } else {
        Fate::Wait
    }
}

/// Compiles every declared shader file into a pipeline in the render world, and carries
/// the dispatches a [`DispatchSender`] asks for there and back. Adds nothing when the app
/// has no render world, so a runtime is never built and a bake dispatches nothing.
pub(crate) struct DispatchPlugin;

impl Plugin for DispatchPlugin {
    fn build(&self, app: &mut App) {
        let shared = SharedDispatch::default();
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(shared.clone())
            .init_resource::<LayerShaders>()
            .init_resource::<LayerPipelines>()
            .add_systems(ExtractSchedule, extract_layer_shaders)
            .add_systems(
                Render,
                (
                    queue_layer_pipelines.in_set(RenderSystems::Queue),
                    dispatch_layers
                        .in_set(RenderSystems::Render)
                        .before(render_system),
                ),
            );
        let (sender, requests) = mpsc::channel();
        app.insert_resource(DispatchBridge {
            sender: DispatchSender(sender),
            requests: Mutex::new(requests),
            shared,
        })
        .init_resource::<InFlight>()
        .add_systems(
            Update,
            (forward_requests, read_back).chain().after(super::scan),
        );
    }
}

fn forward_requests(
    bridge: Option<Res<DispatchBridge>>,
    library: Res<ShaderLibrary>,
    flight: Option<ResMut<InFlight>>,
    mut images: ResMut<Assets<Image>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let (Some(bridge), Some(mut flight)) = (bridge, flight) else {
        return;
    };
    let arrived: Vec<DispatchRequest> = bridge
        .requests
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .try_iter()
        .collect();
    if arrived.is_empty() && flight.waiting.is_empty() {
        return;
    }
    let flight = &mut *flight;
    flight.waiting.extend(arrived);
    let mut shared = bridge.shared.lock();
    for request in std::mem::take(&mut flight.waiting) {
        let entry = library.entry(&request.file).filter(|entry| entry.declared);
        let reread = entry.is_some_and(|entry| entry.read_at > request.sent);
        let state_key = shared.states.get(&request.file).map(|(key, _)| *key);
        match fate(request.key, entry.map(|entry| entry.key), state_key, reread) {
            Fate::Superseded => {
                let _ = request.reply.send(Ok(None));
            }
            Fate::Wait if request.sent.elapsed() >= DISPATCH_TIMEOUT => {
                let _ = request.reply.send(Err(format!(
                    "{}: no pipeline for this source was ready within {} seconds",
                    request.file,
                    DISPATCH_TIMEOUT.as_secs()
                )));
            }
            Fate::Wait => flight.waiting.push(request),
            Fate::Forward => {
                let id = flight.next;
                flight.next += 1;
                let texels = request.globals.texels;
                let output = buffers.add(ShaderBuffer::with_size(
                    u64::from(texels.x) * u64::from(texels.y) * 4,
                    RenderAssetUsages::RENDER_WORLD,
                ));
                let inputs = request
                    .inputs
                    .into_iter()
                    .map(|input| (input.binding, images.add(input.image())))
                    .collect();
                shared.ready.push(GpuJob {
                    id,
                    file: request.file,
                    key: request.key,
                    globals: request.globals.bytes(),
                    params: request.params,
                    inputs,
                    output: output.clone(),
                    texels,
                });
                flight.jobs.insert(
                    id,
                    Carried {
                        reply: request.reply,
                        output,
                    },
                );
            }
        }
    }
}

fn read_back(
    mut commands: Commands,
    bridge: Option<Res<DispatchBridge>>,
    flight: Option<ResMut<InFlight>>,
) {
    let (Some(bridge), Some(mut flight)) = (bridge, flight) else {
        return;
    };
    let (dispatched, failed) = {
        let mut shared = bridge.shared.lock();
        (
            std::mem::take(&mut shared.dispatched),
            std::mem::take(&mut shared.failed),
        )
    };
    for id in failed {
        if let Some(carried) = flight.jobs.remove(&id) {
            let _ = carried.reply.send(Ok(None));
        }
    }
    for id in dispatched {
        let Some(carried) = flight.jobs.get(&id) else {
            continue;
        };
        commands
            .spawn(ReadbackOnce::buffer(carried.output.clone()))
            .observe(
                move |event: On<ReadbackComplete>,
                      mut flight: ResMut<InFlight>,
                      mut commands: Commands| {
                    if let Some(carried) = flight.jobs.remove(&id) {
                        let _ = carried.reply.send(Ok(Some(decode(&event.data))));
                    }
                    commands.entity(event.entity).despawn();
                },
            );
    }
}

/// Sets or clears each library entry's compile fault, and marks the entry's source as
/// compiled, from what the render world last reported for that source; refuses on the
/// document a fault seen for the first time.
pub(crate) fn land_compile_states(
    bridge: Option<Res<DispatchBridge>>,
    mut library: ResMut<ShaderLibrary>,
    mut document: ResMut<crate::document::Document>,
) {
    let Some(bridge) = bridge else {
        return;
    };
    let landed: Vec<(String, Option<String>)> = {
        let shared = bridge.shared.lock();
        shared
            .states
            .iter()
            .filter_map(|(file, (key, status))| {
                let entry = library
                    .entries
                    .get(file)
                    .filter(|entry| entry.declared && entry.key == *key)?;
                match status {
                    PipelineStatus::Failed(fault) if entry.compile_fault != Some(*key) => {
                        Some((file.clone(), Some(fault.clone())))
                    }
                    PipelineStatus::Ready if !entry.settled || entry.compile_fault.is_some() => {
                        Some((file.clone(), None))
                    }
                    _ => None,
                }
            })
            .collect()
    };
    for (file, fault) in landed {
        let Some(entry) = library.entries.get_mut(&file) else {
            continue;
        };
        match fault {
            Some(fault) => {
                warn!("{file}: {fault}");
                document.refuse(format!("{file}: {fault}"));
                entry.compile_fault = Some(entry.key);
                entry.error = Some(fault);
            }
            None => {
                entry.compile_fault = None;
                entry.error = None;
                entry.settled = true;
            }
        }
    }
}

struct LayerShader {
    file: String,
    shader: Handle<Shader>,
    key: u64,
    bindings: Vec<u32>,
}

#[derive(Resource, Default)]
struct LayerShaders(Vec<LayerShader>);

struct LayerPipeline {
    shader: AssetId<Shader>,
    key: u64,
    bindings: Vec<u32>,
    layout: BindGroupLayoutDescriptor,
    id: CachedComputePipelineId,
}

#[derive(Resource, Default)]
struct LayerPipelines(HashMap<String, LayerPipeline>);

fn extract_layer_shaders(library: Extract<Res<ShaderLibrary>>, mut shaders: ResMut<LayerShaders>) {
    shaders.0 = library
        .entries
        .iter()
        .filter(|(_, entry)| entry.declared)
        .filter_map(|(file, entry)| {
            Some(LayerShader {
                file: file.clone(),
                shader: entry.handle.clone()?,
                key: entry.key,
                bindings: entry.layers.iter().map(|read| read.binding).collect(),
            })
        })
        .collect();
}

fn layout(bindings: &[u32]) -> BindGroupLayoutDescriptor {
    let mut entries = vec![
        uniform_buffer_sized(false, None).build(0, ShaderStages::COMPUTE),
        storage_buffer_sized(false, None).build(1, ShaderStages::COMPUTE),
        uniform_buffer_sized(false, None).build(2, ShaderStages::COMPUTE),
    ];
    entries.extend(bindings.iter().map(|binding| {
        texture_2d(TextureSampleType::Float { filterable: false })
            .build(*binding, ShaderStages::COMPUTE)
    }));
    BindGroupLayoutDescriptor::new("watershed layer shader", &entries)
}

fn queue_layer_pipelines(
    shaders: Res<LayerShaders>,
    cache: Res<PipelineCache>,
    mut pipelines: ResMut<LayerPipelines>,
) {
    for shader in &shaders.0 {
        if let Some(held) = pipelines.0.get_mut(&shader.file)
            && held.shader == shader.shader.id()
            && held.bindings == shader.bindings
        {
            held.key = shader.key;
            continue;
        }
        let layout = layout(&shader.bindings);
        let id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("watershed layer {}", shader.file).into()),
            layout: vec![layout.clone()],
            shader: shader.shader.clone(),
            entry_point: Some("generate".into()),
            ..default()
        });
        pipelines.0.insert(
            shader.file.clone(),
            LayerPipeline {
                shader: shader.shader.id(),
                key: shader.key,
                bindings: shader.bindings.clone(),
                layout,
                id,
            },
        );
    }
}

fn status_of(state: &CachedPipelineState) -> PipelineStatus {
    match state {
        CachedPipelineState::Ok(_) => PipelineStatus::Ready,
        CachedPipelineState::Queued
        | CachedPipelineState::Creating(_)
        | CachedPipelineState::Err(
            ShaderCacheError::ShaderNotLoaded(_) | ShaderCacheError::ShaderImportNotYetAvailable,
        ) => PipelineStatus::Pending,
        CachedPipelineState::Err(
            ShaderCacheError::CreateShaderModule(description)
            | ShaderCacheError::ProcessShaderError(description),
        ) => PipelineStatus::Failed(compile_fault(description)),
    }
}

fn dispatch_layers(
    shared: Res<SharedDispatch>,
    pipelines: Res<LayerPipelines>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    images: Res<RenderAssets<GpuImage>>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
) {
    let mut shared = shared.lock();
    let mut statuses = HashMap::with_capacity(pipelines.0.len());
    for (file, pipeline) in &pipelines.0 {
        let status = status_of(cache.get_compute_pipeline_state(pipeline.id));
        shared
            .states
            .insert(file.clone(), (pipeline.key, status.clone()));
        statuses.insert(file.as_str(), status);
    }
    for job in std::mem::take(&mut shared.ready) {
        let current = pipelines
            .0
            .get(&job.file)
            .filter(|pipeline| pipeline.key == job.key);
        let (Some(pipeline), Some(status)) = (current, statuses.get(job.file.as_str())) else {
            shared.failed.push(job.id);
            continue;
        };
        match status {
            PipelineStatus::Failed(_) => shared.failed.push(job.id),
            PipelineStatus::Pending => shared.ready.push(job),
            PipelineStatus::Ready => {
                match submit(&job, pipeline, &cache, &device, &queue, &images, &buffers) {
                    Some(()) => shared.dispatched.push(job.id),
                    None => shared.ready.push(job),
                }
            }
        }
    }
}

fn submit(
    job: &GpuJob,
    pipeline: &LayerPipeline,
    cache: &PipelineCache,
    device: &RenderDevice,
    queue: &RenderQueue,
    images: &RenderAssets<GpuImage>,
    buffers: &RenderAssets<GpuShaderBuffer>,
) -> Option<()> {
    let compute = cache.get_compute_pipeline(pipeline.id)?;
    let output = buffers.get(&job.output)?;
    let views = job
        .inputs
        .iter()
        .map(|(binding, image)| images.get(image).map(|gpu| (*binding, &gpu.texture_view)))
        .collect::<Option<Vec<_>>>()?;
    let globals = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("watershed shader globals"),
        contents: &job.globals,
        usage: BufferUsages::UNIFORM,
    });
    let params = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("watershed shader params"),
        contents: &job.params,
        usage: BufferUsages::UNIFORM,
    });
    let mut entries = vec![
        BindGroupEntry {
            binding: 0,
            resource: globals.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 1,
            resource: output.buffer.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 2,
            resource: params.as_entire_binding(),
        },
    ];
    entries.extend(views.iter().map(|(binding, view)| BindGroupEntry {
        binding: *binding,
        resource: (*view).into_binding(),
    }));
    let layout = cache.get_bind_group_layout(&pipeline.layout);
    let group = device.create_bind_group("watershed layer shader", &layout, &entries);
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("watershed layer shader"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("watershed layer shader"),
            timestamp_writes: None,
        });
        pass.set_pipeline(compute);
        pass.set_bind_group(0, &*group, &[]);
        pass.dispatch_workgroups(job.texels.x.div_ceil(8), job.texels.y.div_ceil(8), 1);
    }
    queue.submit([encoder.finish()]);
    Some(())
}

fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use bevy::render::render_resource::BindingType;

    use super::*;

    // The layout has to be exactly the bindings `layer_lib.wgsl` and a file's `@layer`s declare, or the pipeline is refused.
    #[test]
    fn the_layout_binds_the_globals_output_params_and_each_layer_as_a_texture() {
        let descriptor = layout(&[3, 5]);
        let bindings: Vec<u32> = descriptor
            .entries
            .iter()
            .map(|entry| entry.binding)
            .collect();
        assert_eq!(bindings, [0, 1, 2, 3, 5]);
        for entry in &descriptor.entries {
            let texture = matches!(entry.ty, BindingType::Texture { .. });
            assert_eq!(texture, entry.binding > 2, "binding {}", entry.binding);
        }
    }

    // The readback hands over raw bytes, and a layer's values are what the bake samples.
    #[test]
    fn read_back_bytes_decode_to_the_values_written() {
        let values = [0.0f32, -1.5, 0.543_1, f32::MAX];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(decode(&bytes), values);
    }

    // `new` and `load` send dispatches for files the library has not read yet; those must wait rather than run the old source.
    #[test]
    fn a_request_waits_until_the_library_and_the_render_world_both_hold_its_source() {
        assert_eq!(fate(7, Some(3), Some(3), false), Fate::Wait);
        assert_eq!(fate(7, Some(7), Some(3), true), Fate::Wait);
        assert_eq!(fate(7, None, None, false), Fate::Wait);
        assert_eq!(fate(7, Some(7), Some(7), false), Fate::Forward);
    }

    // A file saved again while its dispatch waited is answered as not run, not after a minute's timeout.
    #[test]
    fn a_request_whose_file_was_read_again_as_another_source_is_superseded() {
        assert_eq!(fate(7, Some(9), Some(9), true), Fate::Superseded);
        assert_eq!(fate(7, Some(9), Some(9), false), Fate::Wait);
    }
}
