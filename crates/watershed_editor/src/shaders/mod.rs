//! The WESL package `watershed` that a layer's shader imports: what it is named, the
//! module the editor compiles every layer against, and the copy of it a terrain carries
//! so a language server resolves the same imports.

use std::path::Path;
use std::sync::LazyLock;

use bevy::prelude::*;
use bevy::shader::Shader;

use crate::terrain::shader::SHADER_DIR;

/// The package name a layer imports the library under, in the editor and in a
/// terrain's `wesl.toml` alike.
pub const PACKAGE: &str = "watershed";

/// The library's module within [`PACKAGE`]: a layer imports `watershed::lib`.
pub const LIBRARY_MODULE: &str = "lib";

/// The directory, beside a terrain's `shaders`, that the editor writes its copy of the
/// package into.
pub const PACKAGE_DIR: &str = ".watershed";

/// The manifest that makes a terrain a WESL package, beside its `shaders`.
pub const MANIFEST: &str = "wesl.toml";

/// The library's source as this build embeds it.
pub const LIBRARY: &str = include_str!("lib.wesl");

const LIBRARY_ASSET: &str = "embedded://watershed/lib.wesl";

const LIBRARY_FILE: &str = "lib.wesl";

const PACKAGE_TOML: &str = include_str!("wesl.toml");

const OVERWRITTEN: &str = "whenever a document is opened or created; edits here are lost.";

const LEGACY_MANIFEST: &str = "# Written by the watershed editor beside lib.wesl";

/// The library as the editor compiles it and writes it out: its first line says the
/// editor overwrites the file.
pub fn library_source() -> &'static str {
    static SOURCE: LazyLock<String> =
        LazyLock::new(|| format!("// The watershed editor overwrites this file {OVERWRITTEN}\n{LIBRARY}"));
    &SOURCE
}

/// The library as a shader asset, registered under the module path
/// `watershed::lib`.
pub fn library_shader() -> Shader {
    Shader::from_wesl(library_source(), LIBRARY_ASSET)
}

fn project_manifest() -> String {
    format!(
        "# Written by the watershed editor {OVERWRITTEN}\n\
         edition = \"2026_pre\"\n\
         root = \"{SHADER_DIR}\"\n\
         \n\
         [dependencies]\n\
         {PACKAGE} = {{ path = \"{PACKAGE_DIR}\" }}\n"
    )
}

/// Makes the terrain directory `dir` a WESL package whose layers resolve
/// `watershed::lib` in a language server: writes [`MANIFEST`] into `dir`, and the
/// library with a manifest of its own into `dir/`[`PACKAGE_DIR`], **overwriting all
/// three** whatever they held. Creates the directories.
///
/// First carries a terrain written before the library was a package across, logging
/// each change: the editor's own `shaders/lib.wesl` and `shaders/wesl.toml` are
/// deleted, and every `package::lib` in a `shaders/*.wesl` file is rewritten to
/// `watershed::lib`. A file that cannot be carried across is logged and left as it is.
///
/// Refused at the first write of the package that fails.
pub fn prepare(dir: &Path) -> Result<(), String> {
    migrate(&dir.join(SHADER_DIR));
    let package = dir.join(PACKAGE_DIR);
    std::fs::create_dir_all(&package)
        .map_err(|error| format!("{}: {error}", package.display()))?;
    let writes = [
        (dir.join(MANIFEST), project_manifest()),
        (package.join(MANIFEST), PACKAGE_TOML.to_owned()),
        (package.join(LIBRARY_FILE), library_source().to_owned()),
    ];
    for (path, contents) in writes {
        std::fs::write(&path, contents).map_err(|error| format!("{}: {error}", path.display()))?;
    }
    Ok(())
}

fn migrate(shaders: &Path) {
    let Ok(entries) = std::fs::read_dir(shaders) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(source) = std::fs::read_to_string(&path) else {
            continue;
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        let outcome = if is_legacy_library(&name, &source) || is_legacy_manifest(&name, &source) {
            std::fs::remove_file(&path).map(|()| "deleted: the editor no longer writes it")
        } else if let Some(rewritten) = name
            .ends_with(".wesl")
            .then(|| rewrite_imports(&source))
            .flatten()
        {
            std::fs::write(&path, rewritten).map(|()| "imports `watershed::lib` now, not `package::lib`")
        } else {
            continue;
        };
        match outcome {
            Ok(done) => info!("{}: {done}", path.display()),
            Err(error) => warn!("{}: {error}", path.display()),
        }
    }
}

fn is_legacy_library(name: &str, source: &str) -> bool {
    name == LIBRARY_FILE && source.lines().next().is_some_and(|line| line.ends_with(OVERWRITTEN))
}

fn is_legacy_manifest(name: &str, source: &str) -> bool {
    name == MANIFEST && source.starts_with(LEGACY_MANIFEST)
}

fn rewrite_imports(source: &str) -> Option<String> {
    const OLD: &str = "package::lib";
    let mut rewritten = String::with_capacity(source.len());
    let mut rest = source;
    let mut changed = false;
    while let Some(at) = rest.find(OLD) {
        let after = &rest[at + OLD.len()..];
        let whole = !after.starts_with(|c: char| c.is_alphanumeric() || c == '_');
        let before = rest[..at].chars().next_back();
        let starts = !before.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == ':');
        rewritten.push_str(&rest[..at]);
        if whole && starts {
            rewritten.push_str(PACKAGE);
            rewritten.push_str("::");
            rewritten.push_str(LIBRARY_MODULE);
            changed = true;
        } else {
            rewritten.push_str(OLD);
        }
        rest = after;
    }
    rewritten.push_str(rest);
    changed.then_some(rewritten)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("watershed-package-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join(SHADER_DIR)).unwrap();
        dir
    }

    // The language server reads the package from the path the terrain's manifest names, so the manifest and the copy have to agree on where that is and what it holds.
    #[test]
    fn prepare_writes_a_manifest_naming_a_copy_of_the_library_the_editor_compiles() {
        let dir = scratch("prepare");
        std::fs::write(dir.join(MANIFEST), "edited").unwrap();

        prepare(&dir).unwrap();

        let manifest = std::fs::read_to_string(dir.join(MANIFEST)).unwrap();
        assert!(manifest.contains("root = \"shaders\""), "{manifest}");
        assert!(
            manifest.contains("watershed = { path = \".watershed\" }"),
            "{manifest}"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join(PACKAGE_DIR).join(LIBRARY_FILE)).unwrap(),
            library_source()
        );
        assert_eq!(
            std::fs::read_to_string(dir.join(PACKAGE_DIR).join(MANIFEST)).unwrap(),
            PACKAGE_TOML
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // A terrain from before the package keeps the editor's lib.wesl in shaders/, which would otherwise come back as a layer named `lib`, and layers importing `package::lib`, which would no longer resolve.
    #[test]
    fn prepare_carries_a_terrain_written_before_the_package_across() {
        let dir = scratch("legacy");
        let shaders = dir.join(SHADER_DIR);
        std::fs::write(shaders.join(LIBRARY_FILE), library_source()).unwrap();
        std::fs::write(
            shaders.join(MANIFEST),
            format!("{LEGACY_MANIFEST}, so a WESL language server\nroot = \".\"\n"),
        )
        .unwrap();
        std::fs::write(
            shaders.join("height.wesl"),
            "import package::lib::{fbm_unit, uv};\n",
        )
        .unwrap();
        std::fs::write(shaders.join("notes.txt"), "package::lib").unwrap();

        prepare(&dir).unwrap();

        assert!(!shaders.join(LIBRARY_FILE).exists());
        assert!(!shaders.join(MANIFEST).exists());
        assert_eq!(
            std::fs::read_to_string(shaders.join("height.wesl")).unwrap(),
            "import watershed::lib::{fbm_unit, uv};\n"
        );
        assert_eq!(
            std::fs::read_to_string(shaders.join("notes.txt")).unwrap(),
            "package::lib"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // A lib.wesl the editor did not write is the user's own layer and must survive the migration.
    #[test]
    fn a_lib_wesl_the_editor_did_not_write_is_kept() {
        let dir = scratch("own-lib");
        let own = "fn value(p: vec2<f32>) -> f32 {\n    return 0.0;\n}\n";
        std::fs::write(dir.join(SHADER_DIR).join(LIBRARY_FILE), own).unwrap();

        prepare(&dir).unwrap();

        assert_eq!(
            std::fs::read_to_string(dir.join(SHADER_DIR).join(LIBRARY_FILE)).unwrap(),
            own
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // Only the old library path is rewritten: a longer name that merely starts the same, or one inside another path, is not that import.
    #[test]
    fn only_whole_package_lib_paths_are_rewritten() {
        assert_eq!(
            rewrite_imports("import package::lib;\nlib::uv(p) + package::lib::uv(p)"),
            Some("import watershed::lib;\nlib::uv(p) + watershed::lib::uv(p)".to_owned())
        );
        assert_eq!(rewrite_imports("import package::library::x;"), None);
        assert_eq!(rewrite_imports("import my_package::lib::x;"), None);
        assert_eq!(rewrite_imports("import watershed::lib::x;"), None);
    }
}
