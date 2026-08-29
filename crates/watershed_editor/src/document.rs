//! The one terrain the editor has open, and everything that happens to it.
//!
//! Every expensive operation — baking, solving, saving, loading — is a *job*, and a
//! job takes the terrain: it is moved onto a task pool and moved back when the job
//! lands. So while one is in flight the editor has no terrain at all, the view is
//! showing the textures the last job left behind, and every other operation is
//! refused rather than queued. There is one job slot and no queue.
//!
//! What the bake on screen is worth is tracked apart from the terrain, because an
//! edit invalidates a bake without touching it: see [`Baked`] for how much of the
//! document currently matches its own layers.

use std::path::PathBuf;

use crate::terrain::{SaveOptions, TerrainSpec};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future};
use serde_json::Value;
use watershed::CellRect;

use crate::edit::Edit;
use crate::preset::Preset;
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
/// Explicit rather than left to the scheduler, because the three read and write the
/// same document within one frame: a stroke made now has to be noted before the
/// document decides what to bake, and the view has to upload what that decided after
/// it. Unordered, a stroke would be answered a frame late and the picture would lag
/// the bake by another.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EditorSystems {
    /// Takes the pointer and turns it into strokes.
    Brush,
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

/// How much of the document's bake still matches the layers it was cut from.
///
/// An edit to the stack drops this to [`Baked::Nothing`] rather than to the part it
/// left alone: a layer applies to a whole field, and nothing here knows the reach of
/// the one that changed. A stroke is the exception — it names the ground it moved, so
/// it leaves the extent where it was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Baked {
    /// Nothing on screen can be trusted.
    Nothing,
    /// Only this rectangle of the document has been baked since the last edit.
    Rect(CellRect),
    /// The whole document matches its layers. The only state a solve will run from.
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

/// The open document and everything the editor knows about its state.
///
/// While a job is running the terrain is `None` — it has been moved onto the pool —
/// so every reader has to cope with there being no document, and every operation that
/// would need one is refused until the job lands.
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
    /// The stack as it stands does not bake. Nothing may re-bake it automatically until
    /// something about it changes, or a document holding a cycle would spend every frame
    /// re-discovering the same cycle — and would never go idle for a caller waiting on the
    /// edit that introduced it.
    bake_failed: bool,
    /// A solve is waiting for the whole-document bake that has to precede it. One flag
    /// rather than a queue, because it is the only pairing of jobs there is.
    pending_solve: bool,
    /// What a stroke has made stale since the last bake was opened. A rectangle rather than
    /// a flag because that is the whole of what a stroke costs — see [`Document::note_stroke`].
    stroke_rect: CellRect,
    pub size: UVec2,
    pub seed: u32,
    pub preset: Preset,
    pub path: Option<PathBuf>,
}

impl Default for Document {
    fn default() -> Self {
        Self {
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
            pending_solve: false,
            stroke_rect: CellRect::EMPTY,
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

    /// Whether a job is running. While it is, there is no terrain to read and every
    /// operation that needs one is refused.
    pub fn is_busy(&self) -> bool {
        matches!(self.job, Job::Running { .. })
    }

    /// How much of the document currently matches its layers.
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

    /// The terrain, to be edited in place. Whatever is changed through this has to be
    /// followed by [`Document::note_edit`] — which is why the panel and
    /// [`Document::apply`] are the only two callers, and why the second one exists at all.
    pub fn terrain_mut(&mut self) -> Option<&mut TerrainSpec> {
        self.terrain.as_mut()
    }

    /// Records that the stack has changed: the whole bake is stale, the last error no
    /// longer applies, a stack that would not bake is worth trying again, and any
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

    /// What a stroke leaves behind, where [`Document::note_edit`] is what a change to the
    /// *stack* leaves behind. The bake keeps the extent it had and the rectangle is added to
    /// what the next one has to cover — so a stroke costs its own footprint rather than the
    /// whole document, and a document that was wholly baked before one is wholly baked after.
    ///
    /// `reached` is the rectangle the change *reaches* through the fields that read the
    /// painted one, not the rectangle the brush covered — the caller gets it from
    /// [`TerrainSpec::influence_of`], which is the only thing that knows how far a
    /// change travels. Passing the painted rectangle instead leaves a stale fringe in
    /// every field downstream.
    pub fn note_stroke(&mut self, reached: CellRect) {
        self.dirty = true;
        self.error = None;
        self.bake_failed = false;
        self.stroke_rect = self.stroke_rect.union(reached);
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
    /// Refused while a job is running or with no document open. On a refusal from the
    /// edit itself the document is untouched.
    pub fn apply(&mut self, edit: &Edit) -> Result<Value, String> {
        if self.is_busy() {
            return Err(format!(
                "a {} is running",
                self.job().map(JobKind::name).unwrap_or("job")
            ));
        }
        let terrain = self
            .terrain
            .as_mut()
            .ok_or("there is no document to edit")?;
        let reply = edit.apply(terrain)?;
        self.note_edit();
        Ok(reply)
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
        self.stroke_rect = CellRect::EMPTY;
        self.baking = covered;
        Ok(())
    }

    /// Starts building a preset and baking it whole, dropping whatever was open.
    ///
    /// Refused while a job is running. The document is emptied immediately, so the
    /// view goes blank on the frame this is called rather than showing the old terrain
    /// under the new size in the toolbar.
    pub fn start_new(&mut self, size: UVec2, seed: u32, preset: Preset) -> Result<(), String> {
        self.busy_check()?;
        self.size = size;
        self.seed = seed;
        self.preset = preset;
        self.path = None;
        self.terrain = None;
        self.baked = Baked::Nothing;
        self.stroke_rect = CellRect::EMPTY;

        let task = AsyncComputeTaskPool::get().spawn(async move {
            let mut terrain = preset.build(size, seed);
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
    /// its layers gives a drainage network for a landscape that no longer exists. Use
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
    pub fn start_load(&mut self, path: PathBuf) -> Result<(), String> {
        self.busy_check()?;
        self.path = Some(path.clone());
        self.terrain = None;
        self.baked = Baked::Nothing;
        self.stroke_rect = CellRect::EMPTY;

        let task = AsyncComputeTaskPool::get().spawn(async move {
            match TerrainSpec::load_from_dir(&path) {
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
    } else if matches!(kind, JobKind::Bake | JobKind::New) {
        document.bake_failed = true;
    }
    document.baking = Baked::Nothing;

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

    match wanted_rebake(
        document.bake_failed,
        document.baked,
        wanted,
        document.stroke_rect,
    ) {
        None => document.dirty = false,
        Some(rect) => {
            if let Err(error) = document.start_bake(Some(rect)) {
                warn!("{error}");
            }
        }
    }
}

fn wanted_rebake(
    bake_failed: bool,
    baked: Baked,
    wanted: CellRect,
    stroke: CellRect,
) -> Option<CellRect> {
    if bake_failed {
        return None;
    }
    let view = if wanted.is_empty() || baked.covers(wanted) {
        CellRect::EMPTY
    } else {
        wanted.expand(REBAKE_MARGIN_CELLS)
    };
    let ask = view.union(stroke);
    (!ask.is_empty()).then_some(ask)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(min: u32, max: u32) -> CellRect {
        CellRect::new(UVec2::splat(min), UVec2::splat(max))
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
        let asked = wanted_rebake(false, Baked::Nothing, view, CellRect::EMPTY)
            .expect("an edit is answered");
        assert!(
            asked.union(view) == asked,
            "the rebake has to cover what is on screen"
        );

        assert_eq!(
            wanted_rebake(false, Baked::Whole, view, CellRect::EMPTY),
            None
        );
        assert_eq!(
            wanted_rebake(false, Baked::Rect(rect(0, 300)), view, CellRect::EMPTY),
            None
        );
        assert!(wanted_rebake(false, Baked::Rect(rect(0, 150)), view, CellRect::EMPTY).is_some());
    }

    // The whole of what makes a brush usable: a stroke asks for the ground it moved and
    // not for the screen it was drawn on, so a drag costs its own footprint per frame
    // rather than a re-bake of the view sixty times a second.
    #[test]
    fn a_stroke_asks_for_the_ground_it_moved_and_not_for_the_whole_view() {
        let view = rect(0, 1024);
        let stroke = rect(100, 140);
        let asked = wanted_rebake(false, Baked::Whole, view, stroke).expect("a stroke is answered");

        assert_eq!(asked, stroke);
        assert!(
            asked.width() < view.width(),
            "the stroke asked for the whole view"
        );
    }

    // And it is what keeps the document solvable: a solve is refused unless the bake is
    // whole, so a stroke that dropped the extent the way a structural edit does would make
    // every stroke cost a whole-document bake before any water could be solved again.
    #[test]
    fn a_stroke_leaves_the_bake_the_extent_it_had() {
        let mut document = Document {
            baked: Baked::Whole,
            ..Document::default()
        };
        document.note_stroke(rect(10, 20));

        assert_eq!(document.baked(), Baked::Whole);
        assert!(document.is_dirty(), "nothing would have baked the stroke");
        assert!(!document.is_settled());
    }

    // A stroke made while a bake is in flight has to be asked for again, on exactly the
    // terms `dirty` is: the job already running took its rectangle before the paint
    // landed. The two assignments in the body are what `start_bake` does to both, spelled
    // out because a test has no task pool to run a real one on.
    #[test]
    fn a_stroke_made_while_a_bake_runs_is_not_taken_for_answered() {
        let mut document = Document {
            stroke_rect: rect(10, 20),
            dirty: true,
            ..Document::default()
        };

        document.stroke_rect = CellRect::EMPTY;
        document.dirty = false;
        document.note_stroke(rect(30, 40));

        assert_eq!(document.stroke_rect, rect(30, 40));
        assert!(document.is_dirty());
    }

    // The defect this guards cost a hang rather than a wrong picture: a stack holding a
    // cycle failed, was retried the next frame, and the document never went idle for the
    // caller waiting on the edit that introduced it.
    #[test]
    fn a_stack_that_will_not_bake_is_not_tried_again_until_something_changes() {
        let view = rect(100, 200);
        assert_eq!(
            wanted_rebake(true, Baked::Nothing, view, CellRect::EMPTY),
            None
        );
        assert_eq!(
            wanted_rebake(true, Baked::Nothing, view, rect(10, 20)),
            None
        );
    }

    // A camera pointed away from the document leaves an empty view, and asking for a
    // bake of nothing would start a job every frame that never made the document any
    // less dirty.
    #[test]
    fn a_view_that_holds_no_cells_asks_for_no_rebake() {
        assert_eq!(
            wanted_rebake(false, Baked::Nothing, CellRect::EMPTY, CellRect::EMPTY),
            None
        );
    }
}
