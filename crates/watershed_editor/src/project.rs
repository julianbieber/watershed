//! Which directory the editor is working in, and how it comes to be that one.

use bevy::prelude::*;

use crate::document::Document;

/// The directory the editor works in. Absolute, always — the type is what keeps it so:
/// the only constructors absolutise.
#[derive(Resource)]
pub struct Project(std::path::PathBuf);

impl Project {
    /// The project directory, absolute.
    pub fn dir(&self) -> &std::path::Path {
        &self.0
    }
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
    Ok(Project(dir))
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

fn open_on_start(mut project: ResMut<Project>, mut document: ResMut<Document>) {
    let dir = project.dir().to_path_buf();
    let result = open(&mut project, &mut document, dir).map(|_| ());
    crate::ui::report(&mut document, result);
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
}
