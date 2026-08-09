// TODO(jb-doc): module docs — that the editor owns exactly one terrain, that every
// expensive operation on it is a job the terrain is *moved into*, and what the view is
// therefore looking at while one is in flight.

use std::path::PathBuf;

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future};
use watershed::{SaveOptions, Terrain};

use crate::preset::Preset;

pub struct DocumentPlugin;

impl Plugin for DocumentPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Document>()
            .add_systems(Update, finish_job.in_set(EditorSystems::Document));
    }
}

/// TODO(jb-doc): why the editor's frame is ordered rather than left to the executor — the
/// same argument `city_panel.rs` makes in wusel, and which two sets would otherwise race.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EditorSystems {
    Document,
    View,
}

/// TODO(jb-doc): what each kind moves and what it hands back, and why `New` is a bake
/// rather than a kind of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobKind {
    New,
    Solve,
    Save,
    Load,
}

impl JobKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::New => "new",
            Self::Solve => "solve",
            Self::Save => "save",
            Self::Load => "load",
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

    pub fn job(&self) -> Option<JobKind> {
        match &self.job {
            Job::Idle => None,
            Job::Running { kind, .. } => Some(*kind),
        }
    }

    pub fn is_busy(&self) -> bool {
        matches!(self.job, Job::Running { .. })
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

        let task = AsyncComputeTaskPool::get().spawn(async move {
            let mut terrain = preset.build(size, seed);
            let error = terrain.bake().err().map(|error| error.to_string());
            Outcome {
                terrain: Some(terrain),
                error,
            }
        });
        self.start(JobKind::New, task);
        Ok(())
    }

    /// TODO(jb-doc): why the spec is read off the document rather than passed in, and what
    /// a document with no spec is being told when this refuses.
    pub fn start_solve(&mut self) -> Result<(), String> {
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
        Ok(())
    }

    fn busy_check(&self) -> Result<(), String> {
        match self.job() {
            Some(kind) => Err(format!("a {} is already running", kind.name())),
            None => Ok(()),
        }
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
}
