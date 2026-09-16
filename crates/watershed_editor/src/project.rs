//! Which directory the editor is working in, what makes it hold a terrain, and how
//! the editor comes to be working in that one.

use std::path::{Path, PathBuf};

use bevy::prelude::*;

use crate::document::Document;
use crate::terrain::recipe::RECIPE_FILE;
use crate::terrain::shader::SHADER_DIR;

/// The directory the editor works in. Absolute, always — the type is what keeps it so:
/// the only constructors absolutise.
#[derive(Resource)]
pub struct Project {
    dir: PathBuf,
    terrain: bool,
}

impl Project {
    /// The project directory, absolute.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Whether the project directory holds a terrain, as last read. See
    /// [`holds_terrain`].
    pub fn holds_terrain(&self) -> bool {
        self.terrain
    }

    /// Re-reads whether the directory holds a terrain, for a caller that changed it —
    /// a `new` that wrote or cleared one. `look_again` does not watch the filesystem:
    /// a change made by hand while the editor is running is not seen until this is
    /// called.
    pub fn look_again(&mut self) {
        self.terrain = holds_terrain(&self.dir);
    }
}

/// Whether `dir` holds a terrain: `recipe.ron` is a file there, or `shaders` is a
/// directory there. The issue's own definition of a project with a terrain, so
/// nothing else in the editor is entitled to disagree with it.
pub fn holds_terrain(dir: &Path) -> bool {
    dir.join(RECIPE_FILE).is_file() || dir.join(SHADER_DIR).is_dir()
}

/// Removes exactly what makes `dir` a project holding a terrain: `shaders/` whole,
/// `recipe.ron`, `terrain.ron`, and every `layer_<n>.png`. Nothing else in the
/// directory is touched, and a file that is already missing is not a failure.
pub fn clear_terrain(dir: &Path) -> Result<(), String> {
    let shaders = dir.join(SHADER_DIR);
    if shaders.is_dir() {
        std::fs::remove_dir_all(&shaders)
            .map_err(|error| format!("{}: {error}", shaders.display()))?;
    }
    for name in [RECIPE_FILE, watershed::io::META_FILE] {
        let path = dir.join(name);
        if path.is_file() {
            std::fs::remove_file(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        }
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => return Err(format!("{}: {error}", dir.display())),
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if watershed::io::is_layer_file(name) {
            std::fs::remove_file(entry.path())
                .map_err(|error| format!("{}: {error}", entry.path().display()))?;
        }
    }
    Ok(())
}

fn argument(args: &[String]) -> Result<Option<&str>, String> {
    match args {
        [] => Ok(None),
        [dir] => Ok(Some(dir)),
        _ => Err("watershed_editor takes one directory and nothing else".to_owned()),
    }
}

/// The project named on the command line, or the working directory with none. The
/// directory is created when it is not there, and a path that exists and is not a
/// directory is refused, naming it.
pub fn from_arguments() -> Result<Project, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = match argument(&args)? {
        Some(dir) => std::path::PathBuf::from(dir),
        None => std::env::current_dir().map_err(|error| error.to_string())?,
    };
    absolutise(dir)
}

fn absolutise(dir: std::path::PathBuf) -> Result<Project, String> {
    if dir.is_file() {
        return Err(format!("{} is a file, not a directory", dir.display()));
    }
    std::fs::create_dir_all(&dir)
        .map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
    let dir = std::path::absolute(&dir).map_err(|error| error.to_string())?;
    let terrain = holds_terrain(&dir);
    Ok(Project { dir, terrain })
}

/// The window title for `project`. Plain ASCII: winit's legacy `WM_NAME` X11 property
/// mis-encodes a non-ASCII title, which breaks anything (`xdotool search --name`
/// included) that reads it rather than `_NET_WM_NAME`.
pub fn title(project: &Project) -> String {
    format!("watershed - {}", project.dir().display())
}

/// Absolutises and creates `dir`, puts it in `project`, and either loads the terrain
/// there (`Ok(true)`) or leaves the document empty and warns (`Ok(false)`) — a
/// directory holding no terrain is not a failure, it is a project waiting for New… or
/// Save. The project is switched either way.
pub fn open(
    project: &mut Project,
    document: &mut Document,
    dir: std::path::PathBuf,
) -> Result<bool, String> {
    let opened = absolutise(dir)?;
    let dir = opened.dir().to_path_buf();
    *project = opened;
    if dir.join(watershed::io::META_FILE).is_file() {
        document.start_load(dir)?;
        Ok(true)
    } else {
        warn!("{} holds no terrain; New… makes one", dir.display());
        document.look_at(dir);
        Ok(false)
    }
}

/// Opens the project on start, and keeps the primary window's title in step with it.
pub struct ProjectPlugin;

impl Plugin for ProjectPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, open_on_start)
            .add_systems(Update, follow_project);
    }
}

fn open_on_start(
    mut project: ResMut<Project>,
    mut document: ResMut<Document>,
    mut dialog: ResMut<crate::ui::NewDialog>,
) {
    let dir = project.dir().to_path_buf();
    let result = open(&mut project, &mut document, dir).map(|_| ());
    crate::ui::report(&mut document, result);
    dialog.open = !project.holds_terrain();
}

fn follow_project(
    project: Res<Project>,
    window: Option<Single<&mut Window, With<bevy::window::PrimaryWindow>>>,
) {
    let Some(mut window) = window else {
        return;
    };
    let wanted = title(&project);
    if window.title != wanted {
        window.title = wanted;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The command line takes at most one directory: none means the working directory,
    // more than one is refused rather than silently taking the first.
    #[test]
    fn the_argument_is_none_one_directory_or_a_refusal() {
        let none: Vec<String> = vec![];
        assert_eq!(argument(&none).unwrap(), None);

        let one = vec!["/tmp/lakes".to_owned()];
        assert_eq!(argument(&one).unwrap(), Some("/tmp/lakes"));

        let two = vec!["/tmp/lakes".to_owned(), "extra".to_owned()];
        assert!(argument(&two).is_err());
    }

    // `from_arguments`-shaped path handling: a directory that does not exist is
    // created, and a file is refused rather than silently treated as one.
    #[test]
    fn a_missing_directory_is_created_and_a_file_is_refused() {
        let dir =
            std::env::temp_dir().join(format!("watershed-project-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let project = absolutise(dir.clone()).unwrap();
        assert!(project.dir().is_dir());
        assert!(project.dir().is_absolute());

        let file = dir.join("not-a-directory");
        std::fs::write(&file, b"x").unwrap();
        assert!(absolutise(file).is_err());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    // A directory holding no terrain opens empty rather than failing, which is what
    // lets startup and the socket share one code path.
    #[test]
    fn opening_a_directory_with_no_terrain_leaves_the_document_empty() {
        let dir =
            std::env::temp_dir().join(format!("watershed-project-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut project = absolutise(dir.clone()).unwrap();
        let mut document = Document::default();
        let loaded = open(&mut project, &mut document, dir.clone()).unwrap();
        assert!(!loaded);
        assert!(document.terrain().is_none());
        assert_eq!(project.dir(), std::path::absolute(&dir).unwrap());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("watershed-project-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    // The issue's own definition, over the four shapes a directory can be in: neither
    // file present is no terrain, and either one alone is already enough.
    #[test]
    fn holds_terrain_is_true_when_either_file_is_there() {
        let dir = scratch("shapes");
        assert!(!holds_terrain(&dir));

        std::fs::write(dir.join(RECIPE_FILE), "recipe only").unwrap();
        assert!(holds_terrain(&dir));
        std::fs::remove_file(dir.join(RECIPE_FILE)).unwrap();

        std::fs::create_dir_all(dir.join(SHADER_DIR)).unwrap();
        assert!(holds_terrain(&dir));

        std::fs::write(dir.join(RECIPE_FILE), "both").unwrap();
        assert!(holds_terrain(&dir));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    // The destructive half: every kind of file a project is made of goes, and nothing
    // else in the directory is touched.
    #[test]
    fn clear_terrain_removes_exactly_the_four_kinds_and_leaves_the_rest() {
        let dir = scratch("clear");
        std::fs::create_dir_all(dir.join(SHADER_DIR)).unwrap();
        std::fs::write(dir.join(SHADER_DIR).join("height.wesl"), "height").unwrap();
        std::fs::write(dir.join(RECIPE_FILE), "recipe").unwrap();
        std::fs::write(dir.join(watershed::io::META_FILE), "values").unwrap();
        std::fs::write(dir.join("layer_000.png"), "layer").unwrap();
        std::fs::write(dir.join("notes.txt"), "mine").unwrap();
        std::fs::write(dir.join("layers.png"), "mine").unwrap();

        clear_terrain(&dir).unwrap();

        assert!(!dir.join(SHADER_DIR).exists());
        assert!(!dir.join(RECIPE_FILE).exists());
        assert!(!dir.join(watershed::io::META_FILE).exists());
        assert!(!dir.join("layer_000.png").exists());
        assert_eq!(std::fs::read(dir.join("notes.txt")).unwrap(), b"mine");
        assert_eq!(std::fs::read(dir.join("layers.png")).unwrap(), b"mine");

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
