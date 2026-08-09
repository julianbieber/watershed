// TODO(jb-doc): module docs — that the editor owns exactly one terrain, that every
// expensive operation on it is a job the terrain is *moved into*, and what the view is
// therefore looking at while one is in flight.

use std::path::PathBuf;

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future};
use serde_json::Value;
use watershed::{CellRect, SaveOptions, Terrain};

use crate::edit::Edit;
use crate::preset::Preset;
use crate::view::VisibleCells;

/// Cells of slack around the visible rectangle when a re-bake is started for it. A pan of
/// less than this costs nothing, where re-baking exactly what is on screen would start a
/// job on the first frame the camera moved.
const REBAKE_MARGIN_CELLS: u32 = 64;

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

/// TODO(jb-doc): why the editor's frame is ordered rather than left to the executor — the
/// same argument `city_panel.rs` makes in wusel, and which two sets would otherwise race.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EditorSystems {
    Brush,
    Document,
    View,
}

/// TODO(jb-doc): what each kind moves and what it hands back, and why `New` is a bake
/// rather than a kind of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobKind {
    New,
    Bake,
    Solve,
    Save,
    Load,
}

impl JobKind {
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

/// How much of the document's bake matches the layers it was cut from.
///
/// TODO(jb-doc): why an edit drops this to [`Baked::Nothing`] rather than to the rectangle
/// it did not touch — that a layer is a whole-field quantity and nothing here knows the
/// reach of the one that changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Baked {
    Nothing,
    Rect(CellRect),
    Whole,
}

impl Baked {
    pub fn covers(self, rect: CellRect) -> bool {
        match self {
            Self::Whole => true,
            Self::Nothing => rect.is_empty(),
            // A union that adds nothing is a rectangle already inside this one, which is
            // the containment test without a second way of spelling it.
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

    pub fn name(self) -> &'static str {
        match self {
            Self::Nothing => "nothing",
            Self::Rect(_) => "rect",
            Self::Whole => "whole",
        }
    }
}

/// TODO(jb-doc): why every job answers with the same shape — that a save hands back the
/// terrain it borrowed and a load invents one, and the caller cannot tell which.
struct Outcome {
    terrain: Option<Terrain>,
    error: Option<String>,
}

enum Job {
    Idle,
    // TODO(jb-comment): why the task lives inside the enum rather than beside it, and what
    // dropping the resource therefore does to a solve that is half done.
    Running { kind: JobKind, task: Task<Outcome> },
}

#[derive(Resource)]
pub struct Document {
    terrain: Option<Terrain>,
    job: Job,
    active: String,
    /// TODO(jb-doc): why the view watches a counter rather than the terrain itself, given
    /// a 4096-square bake is 64 MB and the comparison would cost more than the upload.
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
    pub fn terrain(&self) -> Option<&Terrain> {
        self.terrain.as_ref()
    }

    pub fn active(&self) -> &str {
        &self.active
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn water_revision(&self) -> u64 {
        self.water_revision
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// A synchronous refusal, put where the toolbar shows it. Cleared by the next job or
    /// the next edit, exactly as a job's own error is.
    pub fn refuse(&mut self, error: String) {
        self.error = Some(error);
    }

    pub fn job(&self) -> Option<JobKind> {
        match &self.job {
            Job::Idle => None,
            Job::Running { kind, .. } => Some(*kind),
        }
    }

    pub fn is_busy(&self) -> bool {
        matches!(self.job, Job::Running { .. })
    }

    pub fn baked(&self) -> Baked {
        self.baked
    }

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
    pub fn terrain_mut(&mut self) -> Option<&mut Terrain> {
        self.terrain.as_mut()
    }

    /// TODO(jb-doc): the three things an edit invalidates and why the water is one of them
    /// — that a solved water state is derived from a height that has just moved.
    pub fn note_edit(&mut self) {
        self.dirty = true;
        self.baked = Baked::Nothing;
        self.error = None;
        // The stack has changed, so whatever it failed on last time is worth trying again
        // — which is what lets a cycle be undone by the toggle that made it.
        self.bake_failed = false;
        // `invalidate_water`, never `clear_water`: the latter drops the *spec* too, which
        // is right for "Reset water" and catastrophic here — the first edit after a solve
        // would take away the recipe, and every later solve would refuse a document that
        // looks perfectly ordinary.
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
    /// TODO(jb-doc): why the caller passes the rectangle a change *reaches* rather than the
    /// one it painted, and which of the two `watershed` works out.
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

    /// TODO(jb-doc): why a structural edit goes through the document rather than through
    /// the terrain it holds, given the terrain is reachable either way.
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

    /// TODO(jb-doc): why an unknown name is refused rather than silently kept — that the
    /// legend names the field and a name nothing baked would label an empty view.
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
        // Any job starting supersedes a solve that was waiting on a bake — including the
        // bake it was waiting for, which is why `solve_with_bake` sets the flag *after*
        // asking for it.
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
    /// TODO(jb-doc): why the rectangle is passed in rather than read off the camera here.
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
        // Asked for by name, so it is tried again however the last one went.
        self.bake_failed = false;
        // TODO(jb-comment): why this is cleared for any rectangle rather than only for one
        // that covers it — what every caller of this is obliged to have asked for.
        self.stroke_rect = CellRect::EMPTY;
        self.baking = covered;
        Ok(())
    }

    /// TODO(jb-doc): why building the preset happens on the pool with the bake rather than
    /// here, given only the bake is slow.
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
            let error = terrain.bake().err().map(|error| error.to_string());
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

    /// TODO(jb-doc): why the spec is read off the document rather than passed in, and what
    /// a document with no spec is being told when this refuses.
    pub fn start_solve(&mut self) -> Result<(), String> {
        // Refused rather than quietly solving what is there: water is derived from the
        // height everywhere at once, and a document only part of which matches its layers
        // would produce a drainage network for a landscape that no longer exists.
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

    /// Synchronous, unlike its opposite: dropping a solved state is a deallocation, and
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

    pub fn start_save(&mut self, path: PathBuf, options: SaveOptions) -> Result<(), String> {
        let terrain = self.take_terrain()?;
        self.path = Some(path.clone());
        let task = AsyncComputeTaskPool::get().spawn(async move {
            let error = terrain
                .save_to_path(&path, options)
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

    pub fn start_load(&mut self, path: PathBuf) -> Result<(), String> {
        self.busy_check()?;
        self.path = Some(path.clone());
        self.terrain = None;
        self.baked = Baked::Nothing;
        self.stroke_rect = CellRect::EMPTY;

        let task = AsyncComputeTaskPool::get().spawn(async move {
            match Terrain::load_from_path(&path) {
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
        // A load answers with a document whose bake is complete however the file carried
        // it — the reader re-derives what the file left out, which is the same contract
        // `SaveOptions` documents from the writing end.
        self.baking = Baked::Whole;
        Ok(())
    }

    fn busy_check(&self) -> Result<(), String> {
        match self.job() {
            Some(kind) => Err(format!("a {} is already running", kind.name())),
            None => Ok(()),
        }
    }

    /// A document holding a terrain, without the task pool a job would need to make one.
    #[cfg(test)]
    pub(crate) fn adopt(&mut self, terrain: Terrain) {
        self.size = terrain.size;
        self.terrain = Some(terrain);
    }

    fn take_terrain(&mut self) -> Result<Terrain, String> {
        self.busy_check()?;
        self.terrain
            .take()
            .ok_or_else(|| "there is no document yet".to_owned())
    }
}

/// TODO(jb-comment): why both revisions are bumped for every job rather than only the one
/// the job touched, and what a solve that left the field revision alone would leave on the
/// GPU after a load.
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

    // A bake that failed wrote nothing worth claiming — the document keeps the extent it
    // had, so a cycle introduced by a toggle leaves the last good bake on screen rather
    // than a rectangle of whatever the error interrupted.
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

    // A document whose active field the load did not bring along would draw nothing and
    // say nothing about why, so the name falls back to whatever the terrain does have.
    let names = document.field_names();
    if !names.is_empty() && !names.iter().any(|name| name == document.active()) {
        let fallback = if names.iter().any(|name| name == "height") {
            "height".to_owned()
        } else {
            names[0].clone()
        };
        document.active = fallback;
    }

    // The solve that was standing behind a whole-document bake. Nothing is started if the
    // bake failed — that error is the answer, and a solve over a stack that would not bake
    // has nothing to read.
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

/// Opens the re-bake an edit is waiting for, and the one a pan into unbaked ground needs.
///
/// Both are the same job because they answer the same question — the rectangle on screen
/// does not match the layers — and separating them would mean two callers racing for the
/// one slot a document has.
///
/// TODO(jb-comment): why an edit made while a bake is in flight costs a second whole job
/// rather than being folded into the one already running.
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
        // Nothing will be baked, so nothing is waiting on one: an edit left dirty here
        // would have every caller watching for the document to settle wait forever.
        None => document.dirty = false,
        Some(rect) => {
            if let Err(error) = document.start_bake(Some(rect)) {
                warn!("{error}");
            }
        }
    }
}

/// The decision [`start_pending_bake`] makes, separated from the world it reads it out of.
///
/// TODO(jb-comment): why the answer is a rectangle rather than a yes, and what the margin
/// on it buys a camera that has moved a little.
///
/// TODO(jb-comment): why a structural edit is read off [`Baked`] rather than off the dirty
/// flag, and what asking for the whole view on every frame of a drag would have cost.
fn wanted_rebake(
    bake_failed: bool,
    baked: Baked,
    wanted: CellRect,
    stroke: CellRect,
) -> Option<CellRect> {
    // A stack that failed to bake fails the same way every frame, so retrying it would
    // burn a core and never let the document go idle. The error stands until the next
    // edit, which is the only thing that could change the answer.
    if bake_failed {
        return None;
    }
    let view = if wanted.is_empty() || baked.covers(wanted) {
        CellRect::EMPTY
    } else {
        wanted.expand(REBAKE_MARGIN_CELLS)
    };
    // A union rather than two jobs, and it costs nothing to claim: the bake covers every
    // cell of the rectangle it is given, including the ground between two distant ones.
    let ask = view.union(stroke);
    (!ask.is_empty()).then_some(ask)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(min: u32, max: u32) -> CellRect {
        CellRect::new(UVec2::splat(min), UVec2::splat(max))
    }

    #[test]
    fn a_whole_bake_covers_every_rectangle_and_nothing_covers_one_a_rebake_has_not_reached() {
        assert!(Baked::Whole.covers(rect(0, 4096)));
        assert!(Baked::Rect(rect(0, 100)).covers(rect(10, 90)));
        assert!(Baked::Rect(rect(0, 100)).covers(rect(0, 100)));
        assert!(!Baked::Rect(rect(0, 100)).covers(rect(50, 150)));
        assert!(!Baked::Nothing.covers(rect(0, 1)));
    }

    /// An empty rectangle is covered by anything, including a document with no bake at
    /// all — which is what lets a camera that is nowhere near the document leave a
    /// pending edit answered rather than waiting for a bake with nothing to show.
    #[test]
    fn an_empty_rectangle_is_covered_by_a_document_with_no_bake() {
        assert!(Baked::Nothing.covers(CellRect::EMPTY));
        assert!(Baked::Rect(rect(0, 10)).covers(CellRect::EMPTY));
    }

    #[test]
    fn rebaked_rectangles_accumulate_and_a_whole_bake_swallows_them() {
        let grown = Baked::Nothing
            .with(Baked::Rect(rect(0, 10)))
            .with(Baked::Rect(rect(20, 30)));
        assert_eq!(grown, Baked::Rect(rect(0, 30)));
        assert!(grown.covers(rect(5, 25)));

        assert_eq!(grown.with(Baked::Whole), Baked::Whole);
        // A job that covered nothing — a save, a solve — leaves the extent where it was.
        assert_eq!(grown.with(Baked::Nothing), grown);
    }

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
        // Panning onto ground no bake has reached since the edit asks for it, with nothing
        // dirty — that is the half of this that is not about editing at all.
        assert!(wanted_rebake(false, Baked::Rect(rect(0, 150)), view, CellRect::EMPTY).is_some());
    }

    /// The whole of what makes a brush usable: a stroke asks for the ground it moved and
    /// not for the screen it was drawn on, so a drag costs its own footprint per frame
    /// rather than a re-bake of the view sixty times a second.
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

    /// And it is what keeps the document solvable: a solve is refused unless the bake is
    /// whole, so a stroke that dropped the extent the way a structural edit does would make
    /// every stroke cost a whole-document bake before any water could be solved again.
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

    /// A stroke made while a bake is in flight has to be asked for again, on exactly the
    /// terms `dirty` is: the job already running took its rectangle before the paint landed.
    #[test]
    fn a_stroke_made_while_a_bake_runs_is_not_taken_for_answered() {
        let mut document = Document {
            stroke_rect: rect(10, 20),
            dirty: true,
            ..Document::default()
        };

        // What `start_bake` does to both, without a task pool to run one on.
        document.stroke_rect = CellRect::EMPTY;
        document.dirty = false;
        document.note_stroke(rect(30, 40));

        assert_eq!(document.stroke_rect, rect(30, 40));
        assert!(document.is_dirty());
    }

    /// The defect this guards cost a hang rather than a wrong picture: a stack holding a
    /// cycle failed, was retried the next frame, and the document never went idle for the
    /// caller waiting on the edit that introduced it.
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

    #[test]
    fn a_view_that_holds_no_cells_asks_for_no_rebake() {
        assert_eq!(
            wanted_rebake(false, Baked::Nothing, CellRect::EMPTY, CellRect::EMPTY),
            None
        );
    }
}
