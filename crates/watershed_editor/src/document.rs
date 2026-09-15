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
//! edit invalidates a bake without touching it: see [`Baked`] for how much of the
//! document currently matches its own graphs.

use std::path::PathBuf;

use crate::terrain::{SaveOptions, TerrainSpec};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future};
use serde_json::Value;
use watershed::CellRect;

use crate::edit::{Edit, Slot};
use crate::gpu::{self, ShaderRuntime};
use crate::history::{History, HistoryDepth, Restored, Snapshot};
use crate::preset::Preset;
use crate::terrain::shader::SHADER_DIR;
use crate::view::VisibleCells;

const REBAKE_MARGIN_CELLS: u32 = 64;

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
    /// Bakes the document, or a rectangle of it.
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

/// How much of the document's bake still matches the graphs it was cut from.
///
/// An edit to a graph drops this to [`Baked::Nothing`] rather than to the part it
/// left alone: a node applies to a whole field, and nothing here knows the reach of
/// the one that changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Baked {
    /// Nothing on screen can be trusted.
    Nothing,
    /// Only this rectangle of the document has been baked since the last edit.
    Rect(CellRect),
    /// The whole document matches its graphs. The only state a solve will run from.
    Whole,
}

impl Baked {
    /// Whether that rectangle of the document has been baked since the last edit. An
    /// empty rectangle is covered by anything, including [`Baked::Nothing`].
    pub fn covers(self, rect: CellRect) -> bool {
        match self {
            Self::Whole => true,
            Self::Nothing => rect.is_empty(),
            Self::Rect(have) => have.union(rect) == have,
        }
    }

    fn with(self, added: Self) -> Self {
        match (self, added) {
            (Self::Whole, _) | (_, Self::Whole) => Self::Whole,
            (Self::Nothing, other) | (other, Self::Nothing) => other,
            (Self::Rect(have), Self::Rect(added)) => Self::Rect(have.union(added)),
        }
    }

    /// The lowercase word the control client reports this by. A `Rect` does not carry
    /// its rectangle into the name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Nothing => "nothing",
            Self::Rect(_) => "rect",
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
    /// What the bake in flight will have covered when it lands, held here rather than in
    /// the job because only the caller that started it knows whether it asked for all of
    /// the document or a rectangle of it.
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

    /// How much of the document currently matches its graphs.
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

    /// The terrain, to be edited in place, for a change the history does not own: a
    /// shader's resolved values, the parameters a re-read shader file reconciled.
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

    /// What a re-bake covering `rect` has to actually be asked for: the rectangle, or
    /// `None` — the whole document — when the document holds a shader with something
    /// wired into it, or a field named in an `@layer`, whose file declares no reach.
    ///
    /// Such a shader may read *any* texel of what it is handed, so no rectangle
    /// bounds the ground an edit under it moves. One whose file declares
    /// `// @reach <cells>` is bounded by that, the bake widens the rectangle by it
    /// per hop, and the answer stays the rectangle.
    pub fn bake_ask(&self, rect: CellRect) -> Option<CellRect> {
        let unbounded = self.terrain().is_some_and(TerrainSpec::samples_unbounded);
        (!unbounded).then_some(rect)
    }

    /// Installs what a shader node is dispatched through: into the open terrain when
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

    /// Records that a graph has changed: the whole bake is stale, the last error no
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
    pub fn apply(&mut self, edit: &Edit) -> Result<Value, String> {
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
        self.active = restored.active;
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
        self.start_bake(None)?;
        self.pending_solve = true;
        Ok(())
    }

    /// A rectangle re-bakes only that much of the document and says so afterwards; `None`
    /// is the whole of it, which is the only thing that makes a document solvable again.
    ///
    /// The rectangle is passed in rather than read off the camera, so a caller driving
    /// the editor from outside can ask for ground nobody is looking at.
    pub fn start_bake(&mut self, rect: Option<CellRect>) -> Result<(), String> {
        let mut terrain = self.take_terrain()?;
        let covered = match rect {
            Some(rect) => Baked::Rect(rect.intersect(terrain.rect())),
            None => Baked::Whole,
        };
        let rect = rect.unwrap_or_else(|| terrain.rect());

        let task = AsyncComputeTaskPool::get().spawn(async move {
            let error = terrain.bake_rect(rect).err().map(|error| error.to_string());
            Outcome {
                terrain: Some(terrain),
                error,
            }
        });
        self.start(JobKind::Bake, task);
        self.dirty = false;
        self.bake_failed = false;
        self.baking = covered;
        Ok(())
    }

    /// Starts building a preset and baking it whole, dropping whatever was open.
    ///
    /// The preset's stock shaders are written into the scratch shader directory first,
    /// overwriting files of the same names, and the bake dispatches them through
    /// programs built from those sources on the device the document last had.
    ///
    /// Refused while a job is running, or when a stock shader cannot be written. The
    /// document is emptied immediately, so the view goes blank on the frame this is
    /// called rather than showing the old terrain under the new size in the toolbar.
    pub fn start_new(&mut self, size: UVec2, seed: u32, preset: Preset) -> Result<(), String> {
        self.busy_check()?;
        gpu::write_stock(&gpu::scratch_root(), preset.stock_files())?;
        let runtime = self.runtime.with_sources(
            seed,
            preset.stock_files().iter().filter_map(|file| {
                gpu::stock_source(file).map(|source| ((*file).to_owned(), source.to_owned()))
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
    /// its graphs gives a drainage network for a landscape that no longer exists. Use
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
    /// The bake dispatches the shader nodes through programs built from the document's
    /// own `shaders` directory, on the device the document last had.
    pub fn start_load(&mut self, path: PathBuf) -> Result<(), String> {
        self.busy_check()?;
        let base = self.runtime.clone();
        let seed = self.seed;
        self.path = Some(path.clone());
        self.terrain = None;
        self.baked = Baked::Nothing;
        self.history.clear();
        self.held.clear();

        let task = AsyncComputeTaskPool::get().spawn(async move {
            let runtime = base.with_directory(seed, &path.join(SHADER_DIR));
            match TerrainSpec::load_from_dir(&path, runtime) {
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
        document.terrain = Some(terrain);
    }
    document.revision += 1;
    document.water_revision += 1;

    if outcome.error.is_none() {
        document.baked = document.baked.with(document.baking);
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

fn start_pending_bake(mut document: ResMut<Document>, visible: Res<VisibleCells>) {
    if document.is_busy() {
        return;
    }
    let wanted = if document.terrain().is_some() {
        visible.0
    } else {
        CellRect::EMPTY
    };

    match wanted_rebake(document.bake_failed, document.baked, wanted) {
        None => document.dirty = false,
        Some(rect) => {
            let asked = document.bake_ask(rect);
            if let Err(error) = document.start_bake(asked) {
                warn!("{error}");
            }
        }
    }
}

fn wanted_rebake(bake_failed: bool, baked: Baked, wanted: CellRect) -> Option<CellRect> {
    if bake_failed || wanted.is_empty() || baked.covers(wanted) {
        return None;
    }
    Some(wanted.expand(REBAKE_MARGIN_CELLS))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(min: u32, max: u32) -> CellRect {
        CellRect::new(UVec2::splat(min), UVec2::splat(max))
    }

    fn one_node_document() -> Document {
        use crate::terrain::graph::NodeOp;
        use crate::terrain::{Field, TerrainSpec};
        let mut document = Document::default();
        let mut terrain = TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("height").with_op(NodeOp::held(0.5)));
        terrain.bake_in_place().unwrap();
        document.adopt(terrain);
        document.dirty = false;
        document.baked = Baked::Whole;
        document
    }

    fn two_field_document() -> Document {
        use crate::terrain::graph::NodeOp;
        use crate::terrain::{Field, TerrainSpec, WaterSpec};
        use watershed::{FieldId, FieldRole};
        let mut document = Document::default();
        let mut terrain = TerrainSpec::new(UVec2::splat(16))
            .with_field(Field::new("base").with_op(NodeOp::held(0.25)))
            .with_field(
                Field::new("height")
                    .with_role(FieldRole::Height)
                    .with_op(NodeOp::FieldRef(FieldId::from("base"))),
            );
        terrain.water_spec = Some(WaterSpec::new("height").with_moisture("base"));
        terrain.bake_in_place().unwrap();
        document.adopt(terrain);
        document.dirty = false;
        document.baked = Baked::Whole;
        document
    }

    fn shader_document(wired: bool, reach: Option<u32>) -> Document {
        use crate::terrain::graph::NodeOp;
        use crate::terrain::shader::ShaderLayer;
        use crate::terrain::{Field, TerrainSpec};
        let mut layer = ShaderLayer::new("blur.wgsl");
        layer.inputs = vec!["source".to_owned()];
        layer.reach = reach;
        let mut field = Field::new("height").with_op(NodeOp::held(0.5));
        let upstream = field.graph.nodes[0].id;
        let shaded = field.graph.add_node(NodeOp::Shader(layer), [0.0, 0.0]);
        if wired {
            field.graph.connect(upstream, shaded, 0).unwrap();
        }
        field.graph.set_output(Some(shaded)).unwrap();

        let mut document = Document::default();
        document.adopt(TerrainSpec::new(UVec2::splat(16)).with_field(field));
        document
    }

    // What a wired pin costs turns on the file's declaration, and this draws the
    // line: undeclared, a shader may read any texel of its input and the rectangle a
    // edit moved says nothing about the ground that bakes differently, so the whole
    // document is re-baked; declared, the reach bounds it and the rectangle stands,
    // as it does with the pin unwired.
    #[test]
    fn only_an_undeclared_reach_turns_a_rectangle_re_bake_into_a_whole_one() {
        let asked = rect(0, 8);
        assert_eq!(shader_document(true, None).bake_ask(asked), None);
        assert_eq!(shader_document(true, Some(2)).bake_ask(asked), Some(asked));
        assert_eq!(shader_document(false, None).bake_ask(asked), Some(asked));
    }

    fn only_node(document: &Document) -> String {
        document
            .terrain()
            .unwrap()
            .field("height")
            .unwrap()
            .graph
            .nodes[0]
            .id
            .to_string()
    }

    // Dragging a card writes the document but reaches no bake, so it must not throw the
    // bake away — and `note_edit` also invalidates solved water, which would make moving
    // a node cost a re-solve of the whole document.
    #[test]
    fn moving_a_node_does_not_make_the_bake_stale() {
        let mut document = one_node_document();
        let node = only_node(&document);

        document
            .apply(&Edit::PlaceNode {
                field: "height".to_owned(),
                node: node.clone(),
                position: [40.0, -20.0],
            })
            .unwrap();

        assert_eq!(document.baked, Baked::Whole, "a move invalidated the bake");
        assert!(!document.dirty, "a move made the document dirty");
        assert_eq!(
            document
                .terrain()
                .unwrap()
                .field("height")
                .unwrap()
                .graph
                .nodes[0]
                .position,
            [40.0, -20.0]
        );
    }

    // The whole of what the panel's Add field button and `field add` have to do,
    // including that it is one undo step rather than two: the field arrives, it is on
    // screen, and going back takes both away together.
    #[test]
    fn adding_a_field_puts_it_on_screen_and_undo_puts_the_previous_one_back() {
        let mut document = one_node_document();
        let before = document.history().undo;

        document
            .apply(&Edit::AddField {
                name: "biomes".to_owned(),
            })
            .unwrap();
        assert_eq!(document.active(), "biomes");
        assert_eq!(document.field_names(), ["height", "biomes"]);
        assert_eq!(document.history().undo, before + 1);

        document.undo().unwrap();
        assert_eq!(document.active(), "height");
        assert_eq!(document.field_names(), ["height"]);

        document.redo().unwrap();
        assert_eq!(document.active(), "biomes");
        assert_eq!(document.field_names(), ["height", "biomes"]);
    }

    // Acceptance 7, first half: a rename is one entry on the undo stack, and crossing
    // it has to put the old name back everywhere it was written — in the reader's
    // reference and in the water spec, not only on the field itself.
    #[test]
    fn renaming_a_field_is_one_undo_step_that_restores_the_readers_and_the_water_spec() {
        let mut document = two_field_document();
        let before = document.history().undo;

        document
            .apply(&Edit::RenameField {
                from: "base".to_owned(),
                to: "continent".to_owned(),
            })
            .unwrap();
        assert_eq!(document.history().undo, before + 1);
        assert_eq!(document.field_names(), ["continent", "height"]);
        assert_eq!(declared_reads(&document, "height"), ["continent"]);
        assert_eq!(water_moisture(&document), Some("continent".to_owned()));

        document.undo().unwrap();
        assert_eq!(document.field_names(), ["base", "height"]);
        assert_eq!(declared_reads(&document, "height"), ["base"]);
        assert_eq!(water_moisture(&document), Some("base".to_owned()));
    }

    // Acceptance 7, second half: undoing a removal has to bring the field back with
    // the graph it had, not an empty one, or the undo would lose the work silently.
    #[test]
    fn removing_a_field_is_one_undo_step_that_brings_it_back_with_its_graph() {
        let mut document = two_field_document();
        document.reset_water().unwrap();
        document
            .apply(&Edit::RemoveField {
                name: "height".to_owned(),
            })
            .unwrap();
        let before = document.history().undo;

        document
            .apply(&Edit::RemoveField {
                name: "base".to_owned(),
            })
            .unwrap();
        assert_eq!(document.history().undo, before + 1);
        assert!(document.field_names().is_empty());

        document.undo().unwrap();
        assert_eq!(document.field_names(), ["base"]);
        assert_eq!(
            document
                .terrain()
                .unwrap()
                .field("base")
                .unwrap()
                .graph
                .nodes
                .len(),
            1
        );
    }

    // Acceptance 6, the editor half: the panel is always about the field on screen, so
    // removing that field has to leave some other field showing rather than a name the
    // document no longer has.
    #[test]
    fn removing_the_field_on_screen_puts_another_one_on_screen() {
        let mut document = two_field_document();
        document.reset_water().unwrap();
        document.set_active("height").unwrap();

        document
            .apply(&Edit::RemoveField {
                name: "height".to_owned(),
            })
            .unwrap();
        assert_eq!(document.active(), "base");
    }

    fn declared_reads(document: &Document, field: &str) -> Vec<String> {
        document
            .terrain()
            .unwrap()
            .field(field)
            .unwrap()
            .declared_reads()
            .map(|id| id.to_string())
            .collect()
    }

    fn water_moisture(document: &Document) -> Option<String> {
        document
            .terrain()
            .unwrap()
            .water_spec
            .as_ref()
            .and_then(|spec| spec.moisture.as_ref())
            .map(|id| id.to_string())
    }

    // A field nothing reads yet moves no texel of any field already baked, so adding
    // one must not throw the bake away — the same rule a card drag is held to, and
    // what keeps `observe document` reporting the bake it reported before.
    #[test]
    fn adding_a_field_leaves_the_bake_where_it_was() {
        let mut document = one_node_document();
        let baked = document.baked();

        document
            .apply(&Edit::AddField {
                name: "biomes".to_owned(),
            })
            .unwrap();

        assert_eq!(
            document.baked(),
            baked,
            "an added field invalidated the bake"
        );
        assert!(
            !document.is_dirty(),
            "an added field made the document dirty"
        );
    }

    // Naming a node is the same kind of edit and has to answer the same way.
    #[test]
    fn naming_a_node_does_not_make_the_bake_stale() {
        let mut document = one_node_document();
        let node = only_node(&document);

        document
            .apply(&Edit::RenameNode {
                field: "height".to_owned(),
                node,
                name: Some("ground".to_owned()),
            })
            .unwrap();

        assert_eq!(document.baked, Baked::Whole);
        assert!(!document.dirty);
    }

    // A display property is saved with the document and undone like any other edit, but
    // it says how the map draws the field rather than what the field holds — so, like a
    // card drag, it must cost neither a re-bake nor the solved water.
    #[test]
    fn toggling_hillshade_is_undoable_and_does_not_make_the_bake_stale() {
        let mut document = one_node_document();

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
        let mut document = one_node_document();

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

    // Changing what a node computes is the other half of the rule: that one does reach
    // the bake, so it has to make it stale.
    #[test]
    fn changing_what_a_node_computes_does_make_the_bake_stale() {
        let mut document = one_node_document();
        let node = only_node(&document);

        document
            .apply(&Edit::Set {
                path: format!("height.{node}.value"),
                words: vec!["0.75".to_owned()],
            })
            .unwrap();

        assert_eq!(document.baked, Baked::Nothing);
        assert!(document.dirty);
    }

    // A job takes the terrain with it, so an edit made while one runs has nothing to
    // write to — and is held rather than refused, whether or not a bake reads it. A
    // card dropped mid-bake would otherwise spring back to where it was picked up, and
    // a value committed mid-bake would be lost with a refusal in the status bar. The
    // job is real and never polled, because the terrain has to be genuinely gone.
    #[test]
    fn a_move_made_while_a_job_holds_the_terrain_is_kept_rather_than_refused() {
        let mut document = one_node_document();
        let node = only_node(&document);
        AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
        document.start_bake(None).unwrap();
        assert!(document.is_busy());
        assert!(document.terrain().is_none(), "a job holds the terrain");

        document
            .apply(&Edit::PlaceNode {
                field: "height".to_owned(),
                node: node.clone(),
                position: [12.0, 34.0],
            })
            .expect("a move is held, not refused");

        document
            .apply(&Edit::Set {
                path: format!("height.{node}.value"),
                words: vec!["0.75".to_owned()],
            })
            .expect("a value is held too, not refused");
        assert_eq!(document.held.len(), 2);
    }

    fn scaled() -> crate::terrain::graph::NodeOp {
        use crate::terrain::graph::NodeOp;
        let NodeOp::Shader(mut layer) = NodeOp::piped(1) else {
            unreachable!("a piped node is a shader node");
        };
        layer.params.insert("factor".to_owned(), vec![2.0]);
        NodeOp::Shader(layer)
    }

    fn set_value(field: &mut crate::terrain::Field, value: f32) {
        if let crate::terrain::graph::NodeOp::Shader(shader) = &mut field.graph.nodes[0].op {
            shader.params.insert("value".to_owned(), vec![value]);
        }
    }

    fn graph_of(document: &Document) -> crate::terrain::graph::FieldGraph {
        let mut graph = document
            .terrain()
            .unwrap()
            .field("height")
            .unwrap()
            .graph
            .clone();
        graph.next_id = 0;
        graph
    }

    // The task's own acceptance, made where it can be made without a window: three
    // undos return the graph to what it was, each one leaving the document with a
    // re-bake to run, and three redos replay the edits.
    #[test]
    fn three_undos_return_the_graph_and_three_redos_replay_the_edits() {
        let mut document = one_node_document();
        let node = only_node(&document);
        let original = graph_of(&document);

        document
            .apply(&Edit::AddNode {
                field: "height".to_owned(),
                op: scaled(),
                position: None,
            })
            .unwrap();
        let added = graph_of(&document).nodes.last().unwrap().id.to_string();
        document
            .apply(&Edit::Connect {
                field: "height".to_owned(),
                from: node.clone(),
                to: added.clone(),
                pin: 0,
            })
            .unwrap();
        document
            .apply(&Edit::Set {
                path: format!("height.{added}.factor"),
                words: vec!["3".to_owned()],
            })
            .unwrap();
        let edited = graph_of(&document);
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
        assert_eq!(graph_of(&document), original);
        assert_eq!(document.undo(), Err("nothing to undo".to_owned()));

        for _ in 0..3 {
            document.redo().unwrap();
        }
        assert_eq!(graph_of(&document), edited);
        assert_eq!(document.history().redo, 0);
    }

    // A wire and the output it carried along are one change, so one undo takes both back.
    #[test]
    fn one_undo_takes_back_a_wire_and_the_output_it_moved() {
        use crate::terrain::graph::NodeOp;
        let mut document = one_node_document();
        let node = only_node(&document);
        let original = graph_of(&document);

        document
            .apply(&Edit::AddNode {
                field: "height".to_owned(),
                op: NodeOp::piped(1),
                position: None,
            })
            .unwrap();
        let added = graph_of(&document).nodes.last().unwrap().id;
        document
            .apply(&Edit::Connect {
                field: "height".to_owned(),
                from: node,
                to: added.to_string(),
                pin: 0,
            })
            .unwrap();
        assert_eq!(graph_of(&document).output, Some(added));

        document.undo().unwrap();
        let undone = graph_of(&document);
        assert_eq!(undone.output, original.output);
        assert_eq!(undone.node(added).unwrap().inputs, vec![None]);
    }

    // Undoing a move is the same kind of change as making one: the bake was never
    // reached, so taking the move back must not throw it away either.
    #[test]
    fn undoing_a_move_does_not_make_the_bake_stale() {
        let mut document = one_node_document();
        let node = only_node(&document);
        document
            .apply(&Edit::PlaceNode {
                field: "height".to_owned(),
                node,
                position: [40.0, -20.0],
            })
            .unwrap();

        document.undo().unwrap();
        assert_eq!(document.baked, Baked::Whole);
        assert!(!document.dirty);
        assert_eq!(graph_of(&document).nodes[0].position, [0.0, 0.0]);
    }

    // A refusal leaves the document as it was, so there is nothing to go back to — an
    // entry for it would undo a change that never happened.
    #[test]
    fn a_refused_edit_leaves_no_history_entry() {
        let mut document = one_node_document();
        assert!(
            document
                .apply(&Edit::RemoveNode {
                    field: "height".to_owned(),
                    node: "n9".to_owned(),
                })
                .is_err()
        );
        assert_eq!(document.history().undo, 0);
    }

    // Undo needs the terrain exactly as an edit that reaches the bake does, and a job
    // has it — so it is refused, and the entry stays where it was for the next try.
    #[test]
    fn undo_is_refused_while_a_job_holds_the_terrain_and_keeps_its_entry() {
        let mut document = one_node_document();
        let node = only_node(&document);
        document
            .apply(&Edit::Set {
                path: format!("height.{node}.value"),
                words: vec!["0.75".to_owned()],
            })
            .unwrap();
        AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
        document.start_bake(None).unwrap();

        assert!(document.undo().is_err());
        assert_eq!(document.history().undo, 1);
    }

    // The panel's path into the history: a control committed at the value the field
    // already has is not a change, so it must neither re-bake nor cost a redo.
    #[test]
    fn a_write_that_changes_nothing_records_nothing() {
        let mut document = one_node_document();
        let node = only_node(&document);
        document
            .apply(&Edit::Set {
                path: format!("height.{node}.value"),
                words: vec!["0.75".to_owned()],
            })
            .unwrap();
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

    // A move made while a job held the terrain lands when the job does, and it is a
    // change like any other — so it has to be undoable from there too, and a held edit
    // the landed document refuses must not leave an entry.
    #[test]
    fn a_held_move_is_recorded_when_it_lands_and_a_refused_one_is_not() {
        let mut document = one_node_document();
        let node = only_node(&document);
        let held = |edit: Edit| Held::Edit {
            slot: edit.slot(),
            edit,
        };
        document.hold(held(Edit::PlaceNode {
            field: "height".to_owned(),
            node,
            position: [12.0, 34.0],
        }));
        document.hold(held(Edit::PlaceNode {
            field: "height".to_owned(),
            node: "n9".to_owned(),
            position: [1.0, 1.0],
        }));

        document.land_held();
        assert_eq!(document.history().undo, 1);
        assert_eq!(graph_of(&document).nodes[0].position, [12.0, 34.0]);
        document.undo().unwrap();
        assert_eq!(graph_of(&document).nodes[0].position, [0.0, 0.0]);
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
        match &graph_of(document).nodes[0].op {
            crate::terrain::graph::NodeOp::Shader(shader) => shader.params["value"][0],
            other => panic!("the one node is {other:?}"),
        }
    }

    fn bake_in_flight() -> (Document, String) {
        let mut document = one_node_document();
        let node = only_node(&document);
        AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
        document.start_bake(None).unwrap();
        assert!(document.terrain().is_none(), "a job holds the terrain");
        (document, node)
    }

    // The task's own acceptance, made where it can be made without a window: a value
    // committed while a bake holds the terrain is accepted rather than refused, lands
    // when the bake does, and leaves the document asking for the re-bake that shows
    // it — so the map follows the last value rather than dropping it.
    #[test]
    fn an_edit_made_while_a_job_runs_lands_when_the_job_does_and_asks_for_the_rebake() {
        let (mut document, node) = bake_in_flight();

        document
            .apply(&Edit::Set {
                path: format!("height.{node}.value"),
                words: vec!["0.75".to_owned()],
            })
            .expect("an edit made during a bake is accepted");

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
        let (mut document, node) = bake_in_flight();

        for value in ["0.6", "0.7", "0.8"] {
            document
                .apply(&Edit::Set {
                    path: format!("height.{node}.value"),
                    words: vec![value.to_owned()],
                })
                .unwrap();
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

    // Dropping the earlier change is only safe because it is confined to changes that
    // write one place: two structural edits pile up, and the second needs the first to
    // have landed before it.
    #[test]
    fn a_node_added_and_wired_while_a_job_runs_both_land_in_order() {
        use crate::terrain::graph::NodeOp;
        let mut document = one_node_document();
        let node = only_node(&document);
        let added = format!(
            "n{}",
            document
                .terrain()
                .unwrap()
                .field("height")
                .unwrap()
                .graph
                .next_id
        );
        AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
        document.start_bake(None).unwrap();

        document
            .apply(&Edit::AddNode {
                field: "height".to_owned(),
                op: NodeOp::piped(1),
                position: None,
            })
            .unwrap();
        document
            .apply(&Edit::Connect {
                field: "height".to_owned(),
                from: node,
                to: added,
                pin: 0,
            })
            .unwrap();
        assert_eq!(document.held.len(), 2, "neither drops the other");

        let graph = graph_of(&landed(document));
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.nodes[1].inputs[0], Some(graph.nodes[0].id));
    }

    // The panel's own path is `write` rather than `apply`, so it has to be held on the
    // same terms — a shader parameter committed during a bake is the case the issue
    // was raised about.
    #[test]
    fn a_write_held_while_a_job_runs_lands_when_the_job_does() {
        let (mut document, _) = bake_in_flight();
        let slot = Slot::Control {
            property: "constant",
            node: None,
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
        let mut document = one_node_document();
        let node = only_node(&document);
        document
            .apply(&Edit::PlaceNode {
                field: "height".to_owned(),
                node,
                position: [1.0, 2.0],
            })
            .unwrap();
        AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
        document
            .start_new(UVec2::splat(16), 1, Preset::default())
            .unwrap();

        assert_eq!(
            document.history(),
            crate::history::HistoryDepth { undo: 0, redo: 0 }
        );
    }

    // `covers` is what decides whether a frame starts a job, so it has to be exact at
    // the boundary: a rectangle that only overlaps the baked one is not covered by it,
    // and treating it as covered would leave unbaked ground on screen.
    #[test]
    fn a_whole_bake_covers_every_rectangle_and_nothing_covers_one_a_rebake_has_not_reached() {
        assert!(Baked::Whole.covers(rect(0, 4096)));
        assert!(Baked::Rect(rect(0, 100)).covers(rect(10, 90)));
        assert!(Baked::Rect(rect(0, 100)).covers(rect(0, 100)));
        assert!(!Baked::Rect(rect(0, 100)).covers(rect(50, 150)));
        assert!(!Baked::Nothing.covers(rect(0, 1)));
    }

    // An empty rectangle is covered by anything, including a document with no bake at
    // all — which is what lets a camera that is nowhere near the document leave a
    // pending edit answered rather than waiting for a bake with nothing to show.
    #[test]
    fn an_empty_rectangle_is_covered_by_a_document_with_no_bake() {
        assert!(Baked::Nothing.covers(CellRect::EMPTY));
        assert!(Baked::Rect(rect(0, 10)).covers(CellRect::EMPTY));
    }

    // Rect re-bakes accumulate, so panning across a document gradually makes the whole
    // of it solvable again. The last two cases pin the absorbing and identity ends: a
    // whole bake swallows any extent, and a job that baked nothing — a save, a solve —
    // leaves the extent where it was.
    #[test]
    fn rebaked_rectangles_accumulate_and_a_whole_bake_swallows_them() {
        let grown = Baked::Nothing
            .with(Baked::Rect(rect(0, 10)))
            .with(Baked::Rect(rect(20, 30)));
        assert_eq!(grown, Baked::Rect(rect(0, 30)));
        assert!(grown.covers(rect(5, 25)));

        assert_eq!(grown.with(Baked::Whole), Baked::Whole);
        assert_eq!(grown.with(Baked::Nothing), grown);
    }

    // The decision the editor makes every frame. The last case is the half that is not
    // about editing at all: panning onto ground no bake has reached asks for a job with
    // nothing dirty, which is what keeps the picture whole as the camera moves.
    #[test]
    fn an_edit_asks_for_the_view_and_a_covered_view_asks_for_nothing() {
        let view = rect(100, 200);
        let asked = wanted_rebake(false, Baked::Nothing, view).expect("an edit is answered");
        assert!(
            asked.union(view) == asked,
            "the rebake has to cover what is on screen"
        );

        assert_eq!(wanted_rebake(false, Baked::Whole, view), None);
        assert_eq!(wanted_rebake(false, Baked::Rect(rect(0, 300)), view), None);
        assert!(wanted_rebake(false, Baked::Rect(rect(0, 150)), view).is_some());
    }

    // The defect this guards cost a hang rather than a wrong picture: a stack holding a
    // cycle failed, was retried the next frame, and the document never went idle for the
    // caller waiting on the edit that introduced it.
    #[test]
    fn a_stack_that_will_not_bake_is_not_tried_again_until_something_changes() {
        let view = rect(100, 200);
        assert_eq!(wanted_rebake(true, Baked::Nothing, view), None);
    }

    // A camera pointed away from the document leaves an empty view, and asking for a
    // bake of nothing would start a job every frame that never made the document any
    // less dirty.
    #[test]
    fn a_view_that_holds_no_cells_asks_for_no_rebake() {
        assert_eq!(wanted_rebake(false, Baked::Nothing, CellRect::EMPTY), None);
    }

    // A rectangle re-bake lands every time the view pans, so a fault that has not
    // changed has to reach the log once rather than once per bake that lands.
    #[test]
    fn a_field_fault_that_has_not_changed_is_logged_once() {
        use crate::terrain::graph::NodeOp;
        use crate::terrain::shader::ShaderLayer;
        use crate::terrain::{Field, TerrainSpec};
        let mut shader = ShaderLayer::new("lost.wgsl");
        shader.layers = vec![watershed::FieldId::from("nowhere")];
        let mut document = Document::default();
        document.adopt(
            TerrainSpec::new(UVec2::splat(16))
                .with_field(Field::new("height").with_op(NodeOp::Shader(shader))),
        );
        let first = document.unlogged_faults();
        assert_eq!(first.len(), 1, "{first:?}");
        assert!(first[0].contains("nowhere"), "{first:?}");
        assert!(document.unlogged_faults().is_empty());
    }
}
