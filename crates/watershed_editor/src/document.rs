//! The one terrain the editor has open, and everything that happens to it.
//!
//! Every expensive operation — baking, solving, saving, loading — is a *job*, and a
//! job takes the terrain: it is moved onto a task pool and moved back when the job
//! lands. So while one is in flight the editor has no terrain at all, and the view is
//! showing the textures the last job left behind.
//!
//! There is one job slot and no queue, but an edit made while it is full is *held*
//! rather than refused: it is applied when the terrain comes back, and the bake that
//! follows starts from it. Holding a second change for the same place drops the
//! first, so a hand faster than the bake costs one held change rather than a queue —
//! the map lags but never falls further behind.
//!
//! What the bake on screen is worth is tracked apart from the terrain, because an
//! edit invalidates a bake without touching it: see [`Baked`] for whether the
//! document currently matches its own shaders.
//!
//! Which fields the document has follows the shader directory: adding or removing a
//! field is a file operation, and a file added or deleted by hand adds or removes its
//! field. Neither is recorded in the history.

use std::path::PathBuf;

use crate::terrain::{SaveOptions, TerrainSpec};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future};
use serde_json::Value;

use crate::edit::{Edit, Slot, check_add, check_remove};
use crate::gpu::{self, ShaderRuntime};
use crate::history::{History, HistoryDepth, Restored, Snapshot};
use crate::preset::Preset;
use crate::terrain::Field;
use crate::terrain::shader::SHADER_DIR;

/// Holds the document and runs the two systems that land finished jobs and open the
/// re-bakes an edit or a pan has asked for.
pub struct DocumentPlugin;

impl Plugin for DocumentPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Document>().add_systems(
            Update,
            (finish_job, start_pending_bake)
                .chain()
                .in_set(EditorSystems::Document),
        );
    }
}

/// The order the editor's frame runs in.
///
/// Explicit rather than left to the scheduler, because both read and write the same
/// document within one frame: the view has to upload what the document decided to
/// bake after it decided. Unordered, the picture would lag the bake by a frame.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EditorSystems {
    /// Lands finished jobs and starts the ones the frame has asked for.
    Document,
    /// Uploads what the document holds and follows the camera.
    View,
}

/// What the job in flight is doing. Only one runs at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobKind {
    /// Builds a preset and bakes it whole. Distinct from `Bake` because there is no
    /// terrain to take — it hands one back that did not exist before.
    New,
    /// Bakes the whole document.
    Bake,
    /// Solves the water. Refused unless the whole document is baked.
    Solve,
    /// Writes the document to a path and hands it back unchanged.
    Save,
    /// Reads a document from a path, replacing whatever was open.
    Load,
}

impl JobKind {
    /// The lowercase word the status line and the control client name this by.
    pub fn name(self) -> &'static str {
        match self {
            Self::New => "new",
            Self::Bake => "bake",
            Self::Solve => "solve",
            Self::Save => "save",
            Self::Load => "load",
        }
    }
}

/// Whether the document's bake still matches the shaders and settings it was cut from.
///
/// An edit drops this to [`Baked::Nothing`]; every bake is of the whole document, so
/// there is nothing in between.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Baked {
    /// Nothing on screen can be trusted.
    Nothing,
    /// The whole document matches its shaders. The only state a solve will run from.
    Whole,
}

impl Baked {
    /// The lowercase word the control client reports this by.
    pub fn name(self) -> &'static str {
        match self {
            Self::Nothing => "nothing",
            Self::Whole => "whole",
        }
    }
}

struct Outcome {
    terrain: Option<TerrainSpec>,
    error: Option<String>,
}

enum Job {
    Idle,
    Running { kind: JobKind, task: Task<Outcome> },
}

enum Held {
    Edit {
        edit: Edit,
        slot: Slot,
    },
    Write {
        field: String,
        slot: Slot,
        write: Box<dyn FnOnce(&mut crate::terrain::Field) + Send + Sync>,
    },
}

impl Held {
    fn slot(&self) -> &Slot {
        match self {
            Self::Edit { slot, .. } | Self::Write { slot, .. } => slot,
        }
    }
}

/// The open document and everything the editor knows about its state.
///
/// While a job is running the terrain is `None` — it has been moved onto the pool —
/// so every reader has to cope with there being no document. An edit made in that
/// window is held and applied when the job lands; an operation that needs the terrain
/// for something other than an edit — an undo, a save, a second job — is refused
/// until it does.
#[derive(Resource)]
pub struct Document {
    terrain: Option<TerrainSpec>,
    job: Job,
    active: String,
    revision: u64,
    water_revision: u64,
    error: Option<String>,
    /// An edit has landed that no bake has answered yet. Separate from [`Baked`] because a
    /// re-bake clears this the moment it *starts* — an edit made while one is in flight
    /// has to set it again, or the job already running would be taken for its answer.
    dirty: bool,
    baked: Baked,
    /// What the job in flight will have baked when it lands, held here rather than in
    /// the job because only the caller that started it knows whether it bakes.
    baking: Baked,
    /// The document as it stands does not bake. Nothing may re-bake it automatically until
    /// something about it changes, or a document holding a cycle would spend every frame
    /// re-discovering the same cycle — and would never go idle for a caller waiting on the
    /// edit that introduced it.
    bake_failed: bool,
    logged_faults: Vec<String>,
    /// A solve is waiting for the whole-document bake that has to precede it. One flag
    /// rather than a queue, because it is the only pairing of jobs there is.
    pending_solve: bool,
    /// Changes made while a job held the terrain, waiting for it to come back, in the
    /// order they were made. At most one per [`Slot`] other than [`Slot::Once`].
    held: Vec<Held>,
    /// What the document can go back to. Every change a person makes goes through
    /// [`Document::apply`] or [`Document::write`], which is what puts it here.
    history: History,
    runtime: ShaderRuntime,
    pub size: UVec2,
    pub seed: u32,
    pub preset: Preset,
    pub path: Option<PathBuf>,
}

impl Default for Document {
    fn default() -> Self {
        Self {
            held: Vec::new(),
            history: History::default(),
            terrain: None,
            job: Job::Idle,
            active: "height".to_owned(),
            revision: 0,
            water_revision: 0,
            error: None,
            dirty: false,
            baked: Baked::Nothing,
            baking: Baked::Nothing,
            bake_failed: false,
            logged_faults: Vec::new(),
            pending_solve: false,
            runtime: ShaderRuntime::default(),
            size: UVec2::splat(1024),
            seed: 1,
            preset: Preset::default(),
            path: None,
        }
    }
}

impl Document {
    /// The open document, or `None` while a job holds it or none has been opened.
    pub fn terrain(&self) -> Option<&TerrainSpec> {
        self.terrain.as_ref()
    }

    /// The field on screen. Always a name, even when there is no document, and always
    /// one the document carries when there is.
    pub fn active(&self) -> &str {
        &self.active
    }

    /// Bumped whenever the field textures might need re-uploading. Compared against a
    /// remembered value rather than the rasters themselves, which at a full document
    /// size would cost more to compare than to upload.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// As [`Document::revision`], for the water overlay.
    pub fn water_revision(&self) -> u64 {
        self.water_revision
    }

    /// The last refusal or job failure, until the next job or edit clears it.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// A synchronous refusal, put where the toolbar shows it. Cleared by the next job or
    /// the next edit, exactly as a job's own error is.
    pub fn refuse(&mut self, error: String) {
        self.error = Some(error);
    }

    /// What is running, if anything.
    pub fn job(&self) -> Option<JobKind> {
        match &self.job {
            Job::Idle => None,
            Job::Running { kind, .. } => Some(*kind),
        }
    }

    /// Whether a job is running. While it is, there is no terrain to read: an edit is
    /// held until the job lands, and everything else that needs the terrain is
    /// refused.
    pub fn is_busy(&self) -> bool {
        matches!(self.job, Job::Running { .. })
    }

    /// Whether the document currently matches its shaders.
    pub fn baked(&self) -> Baked {
        self.baked
    }

    /// Whether an edit has landed that no bake has answered yet.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Nothing is in flight *and* nothing is waiting to start. A bake is opened by a system
    /// rather than by the edit itself, so a caller that watched only [`Document::is_busy`]
    /// would read the frame between the two as finished.
    pub fn is_settled(&self) -> bool {
        !self.is_busy() && !self.dirty
    }

    /// The terrain, to be edited in place, for a change the history does not own: the
    /// parameters and the fields read that a re-read shader file reconciled.
    /// Whatever is changed through this has to be followed by [`Document::note_edit`],
    /// and cannot be undone.
    ///
    /// Anything a person authors goes through [`Document::apply`] or
    /// [`Document::write`], which is what puts it in the history.
    pub fn terrain_mut(&mut self) -> Option<&mut TerrainSpec> {
        self.terrain.as_mut()
    }

    /// Whether the last bake of the document as it stands failed. While it holds,
    /// nothing re-bakes the document automatically; an edit or an explicit bake
    /// clears it.
    pub fn bake_failed(&self) -> bool {
        self.bake_failed
    }

    fn unlogged_faults(&mut self) -> Vec<String> {
        let faults: Vec<String> = self
            .terrain()
            .map(|terrain| {
                terrain
                    .field_faults()
                    .into_iter()
                    .map(|(_, fault)| fault)
                    .collect()
            })
            .unwrap_or_default();
        let fresh = faults
            .iter()
            .filter(|fault| !self.logged_faults.contains(fault))
            .cloned()
            .collect();
        self.logged_faults = faults;
        fresh
    }

    /// Installs what a field's shader is dispatched through: into the open terrain when
    /// there is one, and into the document, so a document built or loaded later bakes
    /// on the same device.
    ///
    /// Not an edit and not a revision: the runtime is derived from the shader
    /// directory and the render device, so noting it would ask for a bake on the
    /// frame the editor first sees a GPU and on every frame a file is saved.
    pub fn set_shader_runtime(&mut self, runtime: ShaderRuntime) {
        if let Some(terrain) = self.terrain.as_mut() {
            terrain.set_shader_runtime(runtime.clone());
        }
        self.runtime = runtime;
    }

    /// The runtime last installed with [`Document::set_shader_runtime`], whether or not
    /// a terrain is open. A default one holds no device.
    pub fn shader_runtime(&self) -> &ShaderRuntime {
        &self.runtime
    }

    /// Records that a field has changed: the whole bake is stale, the last error no
    /// longer applies, a document that would not bake is worth trying again, and any
    /// solved water is invalidated — it was derived from a height that has just moved.
    ///
    /// The water's *spec* is kept, so the document can be solved again. Every caller
    /// that edits through [`Document::terrain_mut`] has to call this.
    pub fn note_edit(&mut self) {
        self.dirty = true;
        self.baked = Baked::Nothing;
        self.error = None;
        self.bake_failed = false;
        if let Some(terrain) = self.terrain.as_mut()
            && terrain.water().is_some()
        {
            terrain.invalidate_water();
            self.water_revision += 1;
        }
    }

    /// Applies a structural edit and notes it, which is why an edit goes through here
    /// rather than through [`Document::terrain_mut`]: the two are one operation, and
    /// an edit applied without being noted leaves a stale bake on screen with nothing
    /// arranged to replace it.
    ///
    /// While a job holds the terrain the edit is held rather than applied, and the
    /// answer is `true` rather than the reply the edit would have made; it lands, and
    /// asks for its re-bake, when the job does. Refused only with no document open, or
    /// by the edit itself — and a refusal leaves the document untouched.
    ///
    /// An edit that names a field to show — see [`Edit::shows`] — also puts that field
    /// on screen, as part of the same change: undoing it puts back both the document
    /// and the field that was being looked at.
    ///
    /// A file operation — see [`Edit::is_file_operation`] — is different on three
    /// counts. It is refused rather than held while a job runs. Adding a field **writes
    /// `<name>.wgsl`** into [`Document::shader_root`] as a copy of the template, and is
    /// refused when that file already exists; removing one **deletes that file** after
    /// the edit's own refusals have passed. And neither is recorded in the history, so
    /// neither can be undone.
    pub fn apply(&mut self, edit: &Edit) -> Result<Value, String> {
        if edit.is_file_operation() {
            return self.apply_file_operation(edit);
        }
        if self.is_busy() {
            self.hold(Held::Edit {
                edit: edit.clone(),
                slot: edit.slot(),
            });
            return Ok(Value::Bool(true));
        }
        let active = self.active.clone();
        let terrain = self
            .terrain
            .as_mut()
            .ok_or("there is no document to edit")?;
        let before = Snapshot::take(terrain, edit.reaches_the_bake(), &active);
        let reply = edit.apply(terrain)?;
        self.history.record(before);
        if let Some(shown) = edit.shows(terrain, &active) {
            self.active = shown;
        }
        if edit.reaches_the_bake() {
            self.note_edit();
        } else {
            self.revision += 1;
        }
        Ok(reply)
    }

    fn apply_file_operation(&mut self, edit: &Edit) -> Result<Value, String> {
        self.busy_check()?;
        let root = self.shader_root();
        let active = self.active.clone();
        let terrain = self
            .terrain
            .as_mut()
            .ok_or("there is no document to edit")?;
        match edit {
            Edit::AddField { name } => {
                let name = check_add(terrain, name)?;
                let file = root.join(format!("{name}.wgsl"));
                if file.exists() {
                    return Err(format!("{} already exists", file.display()));
                }
                std::fs::create_dir_all(&root)
                    .and_then(|()| std::fs::write(&file, gpu::template_source()))
                    .map_err(|error| format!("{}: {error}", file.display()))?;
            }
            Edit::RemoveField { name } => {
                let name = check_remove(terrain, name)?;
                let file = root.join(format!("{name}.wgsl"));
                match std::fs::remove_file(&file) {
                    Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                        return Err(format!("{}: {error}", file.display()));
                    }
                    _ => {}
                }
            }
            Edit::Set { .. } => return Err("not a file operation".to_owned()),
        }
        let reply = edit.apply(terrain)?;
        if let Some(shown) = edit.shows(terrain, &active) {
            self.active = shown;
        }
        self.revision += 1;
        self.note_edit();
        Ok(reply)
    }

    /// Makes the open document's fields the ones `names` lists: a field of each name
    /// the document lacks is added with no parameter values, and a field no name lists
    /// is removed. What a file added to or deleted from the shader directory by hand
    /// does to the document.
    ///
    /// Not recorded in the history. When anything moved it is noted as an edit, and
    /// a field on screen that is gone gives way to `height`, or to the first field.
    /// Does nothing with no document open.
    pub fn sync_fields(&mut self, names: &[String]) {
        let Some(terrain) = self.terrain.as_mut() else {
            return;
        };
        let before = terrain.fields.len();
        terrain
            .fields
            .retain(|field| names.iter().any(|name| name == field.id.as_str()));
        let mut moved = terrain.fields.len() != before;
        for name in names {
            if terrain.field(name).is_none() {
                terrain.fields.push(Field::new(name.as_str()));
                moved = true;
            }
        }
        if !moved {
            return;
        }
        if terrain.field(&self.active).is_none() {
            if terrain.field("height").is_some() {
                self.active = "height".to_owned();
            } else if let Some(first) = terrain.fields.first() {
                self.active = first.id.to_string();
            }
        }
        self.revision += 1;
        self.note_edit();
    }

    /// The directory the open document's shader files live in: `shaders` inside its
    /// path once it has one, and the scratch directory before that.
    pub fn shader_root(&self) -> PathBuf {
        self.path
            .as_ref()
            .map_or_else(gpu::scratch_root, |path| path.join(SHADER_DIR))
    }

    /// Writes one field in place through `write`, as one change in the history: the
    /// path for a panel control whose value the [`Edit`] grammar cannot spell.
    ///
    /// Answers whether the document has changed or will change: the field's authored
    /// state differs afterwards, and only then is the change recorded and noted — the
    /// closure's own opinion is not consulted, so a control committed at the value it
    /// already had leaves no entry and no re-bake.
    ///
    /// While a job holds the terrain the closure is held under `slot` and run when the
    /// job lands, which answers `true` before anything has been written. Two writes
    /// held under one slot leave only the second, so `slot` has to name the control
    /// being written and not merely the field.
    ///
    /// Answers `false` with no document open, with no field of that name, or for a
    /// value the field already had.
    pub fn write(
        &mut self,
        field: &str,
        slot: Slot,
        write: impl FnOnce(&mut crate::terrain::Field) + Send + Sync + 'static,
    ) -> bool {
        if self.is_busy() {
            self.hold(Held::Write {
                field: field.to_owned(),
                slot,
                write: Box::new(write),
            });
            return true;
        }
        let active = self.active.clone();
        let Some(terrain) = self.terrain.as_mut() else {
            return false;
        };
        let before = Snapshot::take(terrain, true, &active);
        let Some(target) = terrain.field_mut(field) else {
            return false;
        };
        let was = target.authored();
        write(target);
        if target.authored() == was {
            return false;
        }
        self.history.record(before);
        self.note_edit();
        true
    }

    /// Puts the document back to before the last change and arranges the re-bake that
    /// follows, if the change reached the bake. Refused while a job runs, with no
    /// document open, or with nothing to undo — in each case the history is untouched.
    pub fn undo(&mut self) -> Result<(), String> {
        self.busy_check()?;
        let active = self.active.clone();
        let terrain = self
            .terrain
            .as_mut()
            .ok_or("there is no document to undo in")?;
        let restored = self
            .history
            .undo(terrain, &active)
            .ok_or("nothing to undo")?;
        self.note_restored(restored);
        Ok(())
    }

    /// Replays the last change undone, on the same terms as [`Document::undo`].
    pub fn redo(&mut self) -> Result<(), String> {
        self.busy_check()?;
        let active = self.active.clone();
        let terrain = self
            .terrain
            .as_mut()
            .ok_or("there is no document to redo in")?;
        let restored = self
            .history
            .redo(terrain, &active)
            .ok_or("nothing to redo")?;
        self.note_restored(restored);
        Ok(())
    }

    /// How many changes can be undone, and how many redone.
    pub fn history(&self) -> HistoryDepth {
        self.history.depth()
    }

    fn note_restored(&mut self, restored: Restored) {
        if self
            .terrain
            .as_ref()
            .is_some_and(|terrain| terrain.field(&restored.active).is_some())
        {
            self.active = restored.active;
        }
        self.revision += 1;
        if restored.reaches_bake {
            self.note_edit();
        }
    }

    fn hold(&mut self, change: Held) {
        if *change.slot() != Slot::Once {
            let slot = change.slot().clone();
            self.held.retain(|held| *held.slot() != slot);
        }
        self.held.push(change);
    }

    fn land_held(&mut self) {
        let held = std::mem::take(&mut self.held);
        let mut active = self.active.clone();
        let Some(terrain) = self.terrain.as_mut() else {
            return;
        };
        let mut landed = false;
        let mut reached_the_bake = false;
        for change in held {
            match change {
                Held::Edit { edit, .. } => {
                    let before = Snapshot::take(terrain, edit.reaches_the_bake(), &active);
                    match edit.apply(terrain) {
                        Ok(_) => {
                            self.history.record(before);
                            if let Some(shown) = edit.shows(terrain, &active) {
                                active = shown;
                            }
                            landed = true;
                            reached_the_bake |= edit.reaches_the_bake();
                        }
                        Err(error) => warn!("a held edit was refused: {error}"),
                    }
                }
                Held::Write { field, write, .. } => {
                    let before = Snapshot::take(terrain, true, &active);
                    let Some(target) = terrain.field_mut(&field) else {
                        warn!("a held write names no field `{field}`");
                        continue;
                    };
                    let was = target.authored();
                    write(target);
                    if target.authored() == was {
                        continue;
                    }
                    self.history.record(before);
                    landed = true;
                    reached_the_bake = true;
                }
            }
        }
        self.active = active;
        if reached_the_bake {
            self.note_edit();
        } else if landed {
            self.revision += 1;
        }
    }

    /// Puts a field on screen.
    ///
    /// Refused if the open document has no such field: the legend prints the active
    /// name over the picture, so a name nothing baked would caption an empty view with
    /// a field that does not exist. Accepted with no document open, since there is
    /// nothing yet to check against.
    pub fn set_active(&mut self, field: &str) -> Result<(), String> {
        match self.terrain.as_ref() {
            Some(terrain) if terrain.field(field).is_none() => {
                Err(format!("no field named `{field}`"))
            }
            _ => {
                if self.active != field {
                    self.active = field.to_owned();
                    self.revision += 1;
                }
                Ok(())
            }
        }
    }

    /// The open document's field names in declaration order, or empty when there is no
    /// document.
    pub fn field_names(&self) -> Vec<String> {
        self.terrain
            .as_ref()
            .map(|terrain| {
                terrain
                    .fields
                    .iter()
                    .map(|field| field.id.to_string())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn start(&mut self, kind: JobKind, task: Task<Outcome>) {
        self.error = None;
        self.job = Job::Running { kind, task };
        self.baking = Baked::Nothing;
        self.pending_solve = false;
    }

    /// What the "Solve water" button and the `solve-water` verb both do: a solve needs the
    /// whole document baked, so bake it first when it is not.
    ///
    /// [`Document::start_solve`] still refuses a part-baked document — that guard is what
    /// makes it impossible to solve a stale height, and this is the *caller* that knows
    /// what to do about it rather than a relaxation of it.
    pub fn solve_with_bake(&mut self) -> Result<(), String> {
        if self.baked == Baked::Whole && !self.dirty {
            return self.start_solve();
        }
        self.start_bake()?;
        self.pending_solve = true;
        Ok(())
    }

    /// Starts baking the whole document. Refused while a job is running or with no
    /// document open.
    pub fn start_bake(&mut self) -> Result<(), String> {
        let mut terrain = self.take_terrain()?;

        let task = AsyncComputeTaskPool::get().spawn(async move {
            let error = terrain.bake_in_place().err().map(|error| error.to_string());
            Outcome {
                terrain: Some(terrain),
                error,
            }
        });
        self.start(JobKind::Bake, task);
        self.dirty = false;
        self.bake_failed = false;
        self.baking = Baked::Whole;
        Ok(())
    }

    /// Starts building a preset and baking it whole, dropping whatever was open.
    ///
    /// The preset's field files are written into the scratch shader directory first,
    /// **deleting every other `.wgsl` file there**, and the bake dispatches them through
    /// programs built from those sources on the device the document last had.
    ///
    /// Refused while a job is running, or when the directory cannot be written. The
    /// document is emptied immediately, so the view goes blank on the frame this is
    /// called rather than showing the old terrain under the new size in the toolbar.
    pub fn start_new(&mut self, size: UVec2, seed: u32, preset: Preset) -> Result<(), String> {
        self.busy_check()?;
        gpu::write_preset(&gpu::scratch_root(), preset)?;
        let runtime = self.runtime.with_sources(
            seed,
            preset.files().iter().filter_map(|(field, stock)| {
                gpu::stock_source(stock).map(|source| (format!("{field}.wgsl"), source.to_owned()))
            }),
        );
        self.size = size;
        self.seed = seed;
        self.preset = preset;
        self.path = None;
        self.terrain = None;
        self.baked = Baked::Nothing;
        self.history.clear();
        self.held.clear();

        let task = AsyncComputeTaskPool::get().spawn(async move {
            let mut terrain = preset.build(size, seed);
            terrain.set_shader_runtime(runtime);
            let error = terrain.bake_in_place().err().map(|error| error.to_string());
            Outcome {
                terrain: Some(terrain),
                error,
            }
        });
        self.start(JobKind::New, task);
        self.dirty = false;
        self.baking = Baked::Whole;
        Ok(())
    }

    /// Starts solving the water the document's own spec describes.
    ///
    /// Refused unless the whole document is baked and clean: water is derived from the
    /// height everywhere at once, and solving a document only part of which matches
    /// its shaders gives a drainage network for a landscape that no longer exists. Use
    /// [`Document::solve_with_bake`] to bake first.
    ///
    /// Also refused when the document carries no water spec, which is what a document
    /// that has had its water reset is being told.
    pub fn start_solve(&mut self) -> Result<(), String> {
        if self.baked != Baked::Whole || self.dirty {
            return Err("the document is only partly baked; bake it before solving".to_owned());
        }
        let mut terrain = self.take_terrain()?;
        let Some(spec) = terrain.water_spec.clone() else {
            self.terrain = Some(terrain);
            return Err("the document carries no water spec".to_owned());
        };

        let task = AsyncComputeTaskPool::get().spawn(async move {
            let error = terrain.solve_water(&spec).err().map(|e| e.to_string());
            Outcome {
                terrain: Some(terrain),
                error,
            }
        });
        self.start(JobKind::Solve, task);
        Ok(())
    }

    /// Removes the water and its spec, so nothing will re-solve it. Refused while a
    /// job is running or with no document open.
    ///
    /// Synchronous, unlike solving: dropping a solved state is a deallocation and
    /// there is nothing to wait for.
    pub fn reset_water(&mut self) -> Result<(), String> {
        self.busy_check()?;
        let terrain = self
            .terrain
            .as_mut()
            .ok_or("there is no document to reset")?;
        terrain.clear_water();
        self.water_revision += 1;
        Ok(())
    }

    /// Starts writing the document to the directory at `path`, creating it if it is
    /// not there. The terrain comes back unchanged when the job lands. Refused while
    /// a job is running or with no document open.
    ///
    /// The save removes images in that directory it no longer names; see
    /// [`TerrainSpec::save_to_dir`] for when that sweep runs.
    pub fn start_save(&mut self, path: PathBuf, options: SaveOptions) -> Result<(), String> {
        let terrain = self.take_terrain()?;
        self.path = Some(path.clone());
        let task = AsyncComputeTaskPool::get().spawn(async move {
            let error = terrain
                .save_to_dir(&path, options)
                .err()
                .map(|error| error.to_string());
            Outcome {
                terrain: Some(terrain),
                error,
            }
        });
        self.start(JobKind::Save, task);
        Ok(())
    }

    /// Starts reading a document from `path`, dropping whatever was open.
    ///
    /// Refused while a job is running. What lands is wholly baked whatever the file
    /// carried, because the reader re-derives what the file left out; on a failure the
    /// editor is left with no document rather than the old one.
    ///
    /// The bake dispatches each field's shader through programs built from the
    /// document's own `shaders` directory, on the device the document last had.
    pub fn start_load(&mut self, path: PathBuf) -> Result<(), String> {
        self.busy_check()?;
        let base = self.runtime.clone();
        self.path = Some(path.clone());
        self.terrain = None;
        self.baked = Baked::Nothing;
        self.history.clear();
        self.held.clear();

        let task = AsyncComputeTaskPool::get().spawn(async move {
            match TerrainSpec::load_from_dir(&path, base) {
                Ok(terrain) => Outcome {
                    terrain: Some(terrain),
                    error: None,
                },
                Err(error) => Outcome {
                    terrain: None,
                    error: Some(error.to_string()),
                },
            }
        });
        self.start(JobKind::Load, task);
        self.dirty = false;
        self.baking = Baked::Whole;
        Ok(())
    }

    fn busy_check(&self) -> Result<(), String> {
        match self.job() {
            Some(kind) => Err(format!("a {} is already running", kind.name())),
            None => Ok(()),
        }
    }

    /// Puts a terrain straight into the document, for tests: `start_new` and
    /// `start_load` both need a task pool, and a test has none.
    #[cfg(test)]
    pub(crate) fn adopt(&mut self, terrain: TerrainSpec) {
        self.size = terrain.size;
        self.terrain = Some(terrain);
    }

    fn take_terrain(&mut self) -> Result<TerrainSpec, String> {
        self.busy_check()?;
        self.terrain
            .take()
            .ok_or_else(|| "there is no document yet".to_owned())
    }
}

fn finish_job(mut document: ResMut<Document>) {
    let Job::Running { kind, task } = &mut document.job else {
        return;
    };
    let kind = *kind;
    let Some(outcome) = block_on(future::poll_once(task)) else {
        return;
    };

    document.job = Job::Idle;
    if let Some(terrain) = outcome.terrain {
        document.size = terrain.size;
        document.seed = terrain.seed;
        document.terrain = Some(terrain);
    }
    document.revision += 1;
    document.water_revision += 1;

    if outcome.error.is_none() {
        if document.baking == Baked::Whole {
            document.baked = Baked::Whole;
        }
        for fault in document.unlogged_faults() {
            warn!("{fault}");
        }
    } else if matches!(kind, JobKind::Bake | JobKind::New) {
        document.bake_failed = true;
    }
    document.baking = Baked::Nothing;

    document.land_held();

    if let Some(error) = outcome.error {
        error!("{} failed: {error}", kind.name());
        document.error = Some(error);
    }

    let names = document.field_names();
    if !names.is_empty() && !names.iter().any(|name| name == document.active()) {
        let fallback = if names.iter().any(|name| name == "height") {
            "height".to_owned()
        } else {
            names[0].clone()
        };
        document.active = fallback;
    }

    if kind == JobKind::Bake && document.pending_solve {
        document.pending_solve = false;
        if document.error.is_none()
            && let Err(error) = document.start_solve()
        {
            error!("solve failed: {error}");
            document.error = Some(error);
        }
    }
}

fn start_pending_bake(mut document: ResMut<Document>) {
    if document.is_busy() {
        return;
    }
    if wants_bake(
        document.bake_failed,
        document.baked,
        document.terrain().is_some(),
    ) {
        if let Err(error) = document.start_bake() {
            warn!("{error}");
        }
    } else {
        document.dirty = false;
    }
}

fn wants_bake(bake_failed: bool, baked: Baked, open: bool) -> bool {
    open && !bake_failed && baked != Baked::Whole
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::{TerrainSpec, WaterSpec};
    use watershed::FieldRole;

    fn scratch(name: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("watershed-document-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    fn one_field_document() -> Document {
        let mut document = Document::default();
        let mut terrain =
            TerrainSpec::new(UVec2::splat(16)).with_field(Field::new("height").held(0.5));
        terrain.bake_in_place().unwrap();
        document.adopt(terrain);
        document.dirty = false;
        document.baked = Baked::Whole;
        document
    }

    fn two_field_document(name: &str) -> Document {
        let mut document = Document::default();
        let mut terrain = TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("base").held(0.25))
            .with_field(
                Field::new("height")
                    .with_role(FieldRole::Height)
                    .held(0.5)
                    .reading(&["base"]),
            );
        terrain.water_spec = Some(WaterSpec::new("height").with_moisture("base"));
        terrain.bake_in_place().unwrap();
        document.adopt(terrain);
        document.dirty = false;
        document.baked = Baked::Whole;
        document.path = Some(scratch(name));
        let root = document.shader_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("base.wgsl"), "base").unwrap();
        std::fs::write(root.join("height.wgsl"), "height").unwrap();
        document
    }

    fn add(document: &mut Document, name: &str) -> Result<Value, String> {
        document.apply(&Edit::AddField {
            name: name.to_owned(),
        })
    }

    fn remove(document: &mut Document, name: &str) -> Result<Value, String> {
        document.apply(&Edit::RemoveField {
            name: name.to_owned(),
        })
    }

    fn set(document: &mut Document, path: &str, value: &str) -> Result<Value, String> {
        document.apply(&Edit::Set {
            path: path.to_owned(),
            words: vec![value.to_owned()],
        })
    }

    // What `field add` has to do: the file arrives as a copy of the template, the field
    // is on screen, and — because it is a file operation — no undo step is spent on it.
    #[test]
    fn adding_a_field_writes_the_template_and_shows_it_without_an_undo_step() {
        let mut document = two_field_document("add");
        add(&mut document, "temperature").unwrap();

        let file = document.shader_root().join("temperature.wgsl");
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            gpu::template_source()
        );
        assert_eq!(document.active(), "temperature");
        assert_eq!(document.field_names(), ["base", "height", "temperature"]);
        assert_eq!(document.history().undo, 0);
        assert!(document.is_dirty());
        std::fs::remove_dir_all(document.path.unwrap()).unwrap();
    }

    // A file operation cannot be held for later, because the directory and the document
    // would disagree about which fields exist until the job landed.
    #[test]
    fn a_file_operation_is_refused_while_a_job_runs() {
        let mut document = two_field_document("busy");
        AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
        document.start_bake().unwrap();

        assert!(add(&mut document, "temperature").is_err());
        assert!(!document.shader_root().join("temperature.wgsl").exists());
        std::fs::remove_dir_all(document.path.unwrap()).unwrap();
    }

    // A removal refused because another field reads the name must leave the file, or the
    // reader would be left naming a field that can never come back.
    #[test]
    fn removing_a_read_field_is_refused_and_its_file_stays() {
        let mut document = two_field_document("read");
        document.reset_water().unwrap();

        let error = remove(&mut document, "base").unwrap_err();
        assert!(error.contains("height"), "{error}");
        assert!(document.shader_root().join("base.wgsl").is_file());
        assert_eq!(document.field_names(), ["base", "height"]);
        std::fs::remove_dir_all(document.path.unwrap()).unwrap();
    }

    // `field rm` of a field nothing reads takes the file with it, and the panel is left
    // on a field that still exists.
    #[test]
    fn removing_an_unread_field_deletes_its_file_and_moves_the_view() {
        let mut document = two_field_document("unread");
        document.reset_water().unwrap();
        document.set_active("height").unwrap();

        remove(&mut document, "height").unwrap();
        assert!(!document.shader_root().join("height.wgsl").exists());
        assert_eq!(document.field_names(), ["base"]);
        assert_eq!(document.active(), "base");
        assert_eq!(document.history().undo, 0);
        std::fs::remove_dir_all(document.path.unwrap()).unwrap();
    }

    // A file dropped into or deleted from the directory by hand is a field added or
    // removed, and like one added from the editor it is no undo step.
    #[test]
    fn syncing_the_field_names_adds_and_removes_fields_without_history() {
        let mut document = one_field_document();
        document.sync_fields(&["dunes".to_owned(), "height".to_owned()]);
        assert_eq!(document.field_names(), ["height", "dunes"]);
        assert!(document.is_dirty());

        document.baked = Baked::Whole;
        document.dirty = false;
        document.sync_fields(&["height".to_owned()]);
        assert_eq!(document.field_names(), ["height"]);
        assert_eq!(document.history().undo, 0);

        document.baked = Baked::Whole;
        document.dirty = false;
        document.sync_fields(&["height".to_owned()]);
        assert!(
            !document.is_dirty(),
            "a sync that moved nothing asked for a bake"
        );
    }

    // Ctrl+Z covers parameter values: the value comes back, and the document asks for
    // the whole re-bake that shows it.
    #[test]
    fn undoing_a_parameter_restores_it_and_asks_for_a_bake() {
        let mut document = one_field_document();
        set(&mut document, "height.value", "0.75").unwrap();
        assert_eq!(constant_of(&document), 0.75);

        document.baked = Baked::Whole;
        document.dirty = false;
        document.undo().unwrap();
        assert_eq!(constant_of(&document), 0.5);
        assert_eq!(document.baked, Baked::Nothing);
        assert!(document.is_dirty());
    }

    // A display property is saved with the document and undone like any other edit, but
    // it says how the map draws the field rather than what the field holds — so, like a
    // card drag, it must cost neither a re-bake nor the solved water.
    #[test]
    fn toggling_hillshade_is_undoable_and_does_not_make_the_bake_stale() {
        let mut document = one_field_document();

        document
            .apply(&Edit::Set {
                path: "height.hillshade".to_owned(),
                words: vec!["on".to_owned()],
            })
            .unwrap();

        assert!(
            document
                .terrain()
                .unwrap()
                .field("height")
                .unwrap()
                .hillshade
        );
        assert_eq!(document.baked, Baked::Whole);
        assert_eq!(document.history().undo, 1);

        document.undo().unwrap();

        assert!(
            !document
                .terrain()
                .unwrap()
                .field("height")
                .unwrap()
                .hillshade
        );
        assert_eq!(document.baked, Baked::Whole);
    }

    // The contour overlay is the second display property pair, added after the
    // hillshade one, and is under the same rule: undoable, and free of the bake.
    #[test]
    fn toggling_contours_is_undoable_and_does_not_make_the_bake_stale() {
        let mut document = one_field_document();

        document
            .apply(&Edit::Set {
                path: "height.contours".to_owned(),
                words: vec!["on".to_owned()],
            })
            .unwrap();

        assert!(
            document
                .terrain()
                .unwrap()
                .field("height")
                .unwrap()
                .contours
        );
        assert_eq!(document.baked, Baked::Whole);
        assert_eq!(document.history().undo, 1);

        document.undo().unwrap();

        assert!(
            !document
                .terrain()
                .unwrap()
                .field("height")
                .unwrap()
                .contours
        );
        assert_eq!(document.baked, Baked::Whole);
    }

    // A parameter is what the field holds, so changing one has to make the bake stale.
    #[test]
    fn changing_a_parameter_does_make_the_bake_stale() {
        let mut document = one_field_document();
        set(&mut document, "height.value", "0.75").unwrap();
        assert_eq!(document.baked, Baked::Nothing);
        assert!(document.dirty);
    }

    // A job takes the terrain with it, so a value made while one runs has nothing to
    // write to — and is held rather than refused, so it is not lost with a refusal in
    // the status bar. The job is real and never polled, because the terrain has to be
    // genuinely gone.
    #[test]
    fn a_value_made_while_a_job_holds_the_terrain_is_kept_rather_than_refused() {
        let (mut document, _) = bake_in_flight();
        set(&mut document, "height.value", "0.75").expect("a value is held, not refused");
        set(&mut document, "height.range", "0").expect("a range is held, not refused");
        assert_eq!(document.held.len(), 2);
    }

    fn set_value(field: &mut Field, value: f32) {
        field.shader.params.insert("value".to_owned(), vec![value]);
    }

    fn authored_of(document: &Document) -> Field {
        document
            .terrain()
            .unwrap()
            .field("height")
            .unwrap()
            .authored()
    }

    // Three undos return the field to what it was, each one leaving the document with a
    // re-bake to run, and three redos replay the edits.
    #[test]
    fn three_undos_return_the_field_and_three_redos_replay_the_edits() {
        let mut document = one_field_document();
        let original = authored_of(&document);

        set(&mut document, "height.value", "0.6").unwrap();
        set(&mut document, "height.value", "0.7").unwrap();
        document
            .apply(&Edit::Set {
                path: "height.range".to_owned(),
                words: vec!["0".to_owned(), "2".to_owned()],
            })
            .unwrap();
        let edited = authored_of(&document);
        assert_ne!(edited, original);
        assert_eq!(document.history().undo, 3);

        for remaining in [2, 1, 0] {
            document.baked = Baked::Whole;
            document.dirty = false;
            document.undo().unwrap();
            assert_eq!(document.baked, Baked::Nothing, "no re-bake was asked for");
            assert!(document.dirty);
            assert_eq!(document.history().undo, remaining);
        }
        assert_eq!(authored_of(&document), original);
        assert_eq!(document.undo(), Err("nothing to undo".to_owned()));

        for _ in 0..3 {
            document.redo().unwrap();
        }
        assert_eq!(authored_of(&document), edited);
        assert_eq!(document.history().redo, 0);
    }

    // A refusal leaves the document as it was, so there is nothing to go back to — an
    // entry for it would undo a change that never happened.
    #[test]
    fn a_refused_edit_leaves_no_history_entry() {
        let mut document = one_field_document();
        assert!(set(&mut document, "nowhere.value", "1").is_err());
        assert_eq!(document.history().undo, 0);
    }

    // Undo needs the terrain exactly as an edit that reaches the bake does, and a job
    // has it — so it is refused, and the entry stays where it was for the next try.
    #[test]
    fn undo_is_refused_while_a_job_holds_the_terrain_and_keeps_its_entry() {
        let mut document = one_field_document();
        set(&mut document, "height.value", "0.75").unwrap();
        AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
        document.start_bake().unwrap();

        assert!(document.undo().is_err());
        assert_eq!(document.history().undo, 1);
    }

    // The panel's path into the history: a control committed at the value the field
    // already has is not a change, so it must neither re-bake nor cost a redo.
    #[test]
    fn a_write_that_changes_nothing_records_nothing() {
        let mut document = one_field_document();
        set(&mut document, "height.value", "0.75").unwrap();
        document.undo().unwrap();
        assert_eq!(document.history().redo, 1);

        let changed = document.write("height", Slot::Once, |field| set_value(field, 0.5));
        assert!(!changed);
        assert_eq!(document.history().redo, 1, "a no-op write forgot the redo");

        let changed = document.write("height", Slot::Once, |field| set_value(field, 0.25));
        assert!(changed);
        assert_eq!(
            document.history(),
            crate::history::HistoryDepth { undo: 1, redo: 0 }
        );
        assert_eq!(document.baked, Baked::Nothing);
    }

    // A value held while a job held the terrain lands when the job does, and it is a
    // change like any other — so it has to be undoable from there too, and a held edit
    // the landed document refuses must not leave an entry.
    #[test]
    fn a_held_value_is_recorded_when_it_lands_and_a_refused_one_is_not() {
        let mut document = one_field_document();
        let held = |edit: Edit| Held::Edit {
            slot: edit.slot(),
            edit,
        };
        document.hold(held(Edit::Set {
            path: "height.value".to_owned(),
            words: vec!["0.75".to_owned()],
        }));
        document.hold(held(Edit::Set {
            path: "nowhere.value".to_owned(),
            words: vec!["1".to_owned()],
        }));

        document.land_held();
        assert_eq!(document.history().undo, 1);
        assert_eq!(constant_of(&document), 0.75);
        document.undo().unwrap();
        assert_eq!(constant_of(&document), 0.5);
    }

    fn landed(document: Document) -> Document {
        use bevy::ecs::system::RunSystemOnce;
        let mut world = World::new();
        world.insert_resource(document);
        for _ in 0..2000 {
            world.run_system_once(finish_job).unwrap();
            if !world.resource::<Document>().is_busy() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let document = world.remove_resource::<Document>().expect("the document");
        assert!(!document.is_busy(), "the job never landed");
        document
    }

    fn constant_of(document: &Document) -> f32 {
        document
            .terrain()
            .unwrap()
            .field("height")
            .unwrap()
            .shader
            .params["value"][0]
    }

    fn bake_in_flight() -> (Document, ()) {
        let mut document = one_field_document();
        AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
        document.start_bake().unwrap();
        assert!(document.terrain().is_none(), "a job holds the terrain");
        (document, ())
    }

    // A value committed while a bake holds the terrain is accepted rather than refused,
    // lands when the bake does, and leaves the document asking for the re-bake that
    // shows it — so the map follows the last value rather than dropping it.
    #[test]
    fn an_edit_made_while_a_job_runs_lands_when_the_job_does_and_asks_for_the_rebake() {
        let (mut document, _) = bake_in_flight();

        set(&mut document, "height.value", "0.75").expect("an edit made during a bake is accepted");

        let document = landed(document);
        assert_eq!(constant_of(&document), 0.75);
        assert!(document.is_dirty(), "the landed edit asks for its re-bake");
        assert_eq!(
            document.baked,
            Baked::Nothing,
            "the bake that landed answered the document as it was before the edit"
        );
        assert_eq!(document.history().undo, 1);
    }

    // The other half of accepting a stream: a hand faster than the bake costs one held
    // change and one undo step, not one of each per value it passed through.
    #[test]
    fn two_values_for_one_control_held_together_leave_only_the_last() {
        let (mut document, _) = bake_in_flight();

        for value in ["0.6", "0.7", "0.8"] {
            set(&mut document, "height.value", value).unwrap();
        }
        assert_eq!(document.held.len(), 1, "one control, one held change");

        let document = landed(document);
        assert_eq!(constant_of(&document), 0.8);
        assert_eq!(
            document.history().undo,
            1,
            "the values passed through are not undo steps"
        );
    }

    // The panel's own path is `write` rather than `apply`, so it has to be held on the
    // same terms — a shader parameter committed during a bake is the case the issue
    // was raised about.
    #[test]
    fn a_write_held_while_a_job_runs_lands_when_the_job_does() {
        let (mut document, _) = bake_in_flight();
        let slot = Slot::Control {
            property: "constant",
            index: [0, 0],
        };

        for value in [0.6, 0.8] {
            assert!(
                document.write("height", slot.clone(), move |field| {
                    set_value(field, value);
                }),
                "a write during a bake is accepted"
            );
        }
        assert_eq!(document.held.len(), 1);

        let document = landed(document);
        assert_eq!(constant_of(&document), 0.8);
        assert!(document.is_dirty());
        assert_eq!(document.history().undo, 1);
    }

    // A held write is not consulted about whether it changed anything until it runs, so
    // the check `write` makes when it is idle has to be made again when it lands — or a
    // control committed at the value it already had would cost a re-bake and a redo.
    #[test]
    fn a_held_write_that_changes_nothing_records_nothing() {
        let (mut document, _) = bake_in_flight();

        document.write("height", Slot::Once, |field| set_value(field, 0.5));

        let document = landed(document);
        assert_eq!(constant_of(&document), 0.5);
        assert_eq!(
            document.history().undo,
            0,
            "nothing changed, nothing to undo"
        );
        assert!(!document.is_dirty(), "and nothing to re-bake");
    }

    // A new document has nothing to go back to: an entry from the one before would
    // restore fields that never belonged to it.
    #[test]
    fn a_new_document_starts_with_an_empty_history() {
        let mut document = one_field_document();
        set(&mut document, "height.value", "0.75").unwrap();
        AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
        document
            .start_new(UVec2::splat(16), 1, Preset::default())
            .unwrap();

        assert_eq!(
            document.history(),
            crate::history::HistoryDepth { undo: 0, redo: 0 }
        );
    }

    // The decision the editor makes every frame: an open document that is not wholly
    // baked asks for a bake, unless the last one failed — a document holding a cycle
    // would otherwise re-discover it every frame and never go idle.
    #[test]
    fn a_bake_is_wanted_only_for_an_open_unbaked_document_that_did_not_just_fail() {
        assert!(wants_bake(false, Baked::Nothing, true));
        assert!(!wants_bake(false, Baked::Whole, true));
        assert!(!wants_bake(true, Baked::Nothing, true));
        assert!(!wants_bake(false, Baked::Nothing, false));
    }

    // A bake lands after every edit, so a fault that has not changed has to reach the
    // log once rather than once per bake that lands.
    #[test]
    fn a_field_fault_that_has_not_changed_is_logged_once() {
        let mut document = Document::default();
        document.adopt(
            TerrainSpec::new(UVec2::splat(16))
                .with_field(Field::new("height").reading(&["nowhere"])),
        );
        let first = document.unlogged_faults();
        assert_eq!(first.len(), 1, "{first:?}");
        assert!(first[0].contains("nowhere"), "{first:?}");
        assert!(document.unlogged_faults().is_empty());
    }
}
