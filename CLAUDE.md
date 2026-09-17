# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

`just --list` is the entry point; `just all` is CI's order and CI runs nothing else,
so a check not reachable from `all` is a check that does not happen.

| | |
|---|---|
| `just all` | `fmt docs clippy bevy-lints test check-web check-goals` |
| `just test` | `cargo test --locked --workspace` |
| `just clippy` / `just fmt` / `just docs` | on the `ci` profile |
| `just bevy-lints` | needs `bevy_lint` on PATH — `just bevy-lint-install` |
| `just check-web` | wasm32 check of `watershed` only; the editor is native |
| `just check-goals` | builds `crates/watershed/examples/load_terrain.rs`, the worked statement of the library's interface |
| `just deps` | apt packages CI needs (alsa, udev, wayland, mesa) |
| `just run <dir>` | the editor on a project directory |
| `just install` | `watershed_editor` and `watershed-ctl` into `~/.cargo/bin` |
| `just check-placeholders` | outstanding `TODO(jb-doc)` / `TODO(jb-comment)`; deliberately not in `all` |

A single test: `cargo test --locked --workspace <substring>`, or
`cargo test -p watershed_editor terrain::water::tests::<name>`. Every test is a
`mod tests` inside the module it covers — there is no `tests/` directory — and none
needs a GPU or a window, so `just test` is headless.

Pinned nightly (`rust-toolchain.toml`), and every recipe sets
`RUSTFLAGS=-Zshare-generics=y -Zthreads=0`. Bevy is pinned to a git rev of main, which
is why `bevy_lint` is pinned to a `bevy_cli` commit rather than a release.

## Driving the editor

The editor can be driven from outside the process over a unix socket, and this is how
a change is verified end to end:

```
just drive-start                                 # editor on /tmp/watershed-project, socket open
just drive observe water                         # one command
just drive run scenarios/water_finds_the_lakes.txt
```

The socket exists only when `WATERSHED_CONTROL` is set. **A command's reply is held
until the effect has actually happened** — `solve-water` answers when the water is
solved, `capture` when the PNG is on disk — so a scenario never sleeps or counts
frames. `scenarios/*.txt` are these commands one per line, replayed by the client
(not the editor), stopping at the first failure. `observe log` *drains*, so it comes
first in a scenario or the startup errors are gone.

**Standing rule: every capability the window has is reachable from the control client
too.** A new action goes in `src/edit.rs` (or another place both reach) and is then
given a button in `src/ui/` and a verb in `src/control/command.rs`. An action only a
person holding the keys can exercise cannot be verified.

## Architecture

Two crates, and **nothing does both halves**:

- `crates/watershed` — the **read** side. `Terrain`: the values and nothing that made
  them. It bakes nothing, solves nothing, evaluates nothing; `Terrain::load_from_dir`
  is the only way to obtain one. No GPU, no thread pool, builds for wasm. Do not add
  authoring types here.
- `crates/watershed_editor` — the **author** side, carrying the whole model a document
  is made of: layers, shaders, the water solve, the bake. `TerrainSpec` (authored) is
  here; baking one produces a `watershed::Terrain`.

A consuming project ships the directory the editor wrote and never builds a terrain at
run time. That split is what makes a reader never parse a shader.

### The document model

A terrain is an extent in cells plus named layers over it. **A layer is one
`shaders/<name>.wesl` file in the document's own directory, and the file declares the
layer as well as its values**: `// @role`, `// @shift`, `// @range`, `// @categorical`
header lines, `@ui` annotations on each `Params` field, and `@layer` after a texture
binding to read another layer. Adding a layer copies the template in; a file added or
deleted by hand adds or removes its layer; `_`-prefixed files are not layers. Those
four header lines are not in the undo history, because the next read of the file would
put them back.

Reading another layer by name is what orders the bake — a layer is baked after
everything it reads — and every edit re-bakes the whole document. A layer lives on a
raster at its own *shift* (one texel per `2^shift` cells) and still answers at every
cell, interpolated (or nearest, when categorical).

### Where things live in the editor

| | |
|---|---|
| `document.rs` | the one open terrain and every job that happens to it |
| `edit.rs` | the whole vocabulary for naming and changing a document — the panel and the control client both go through it |
| `terrain/bake.rs` | plan (layer order, structural faults) and bake |
| `terrain/water.rs` | the whole-grid solve: fill, flow direction, accumulation, lakes |
| `terrain/shader.rs` | the document's half of a shader — parsing headers/`@ui`/`@layer`, packing params. Touches no GPU |
| `gpu.rs` + `gpu/dispatch.rs` | the WESL the document carries, and carrying a dispatch from the bake's thread through the render world and back |
| `terrain/recipe.rs` | `recipe.ron`, the authored half of the directory |
| `control/` | socket server, verbs (`command.rs`), read-only topics (`observe.rs`), drained log |
| `canvas/`, `ui/`, `view.rs`, `material.rs` | cards and ribbons, chrome, the viewport quad and its camera |

**Jobs.** Baking, solving, saving and loading are jobs, and a job *takes* the terrain:
it moves onto the task pool and moves back when it lands, so while one is in flight the
editor has no terrain and the view shows the last job's textures. One slot, no queue —
an edit made while it is full is *held* and applied when the terrain returns, and a
second change to the same `Slot` drops the first. Systems are ordered by
`EditorSystems::{Document, View}`.

**The editor reads nothing from disk beside its binary.** `lib.wesl`, `wesl.toml`, the
template and every stock shader are `include_str!`'d from
`crates/watershed_editor/assets/shaders/stock/`, and `layer.wesl` is an embedded asset.
The editor writes `lib.wesl` and `wesl.toml` into a document's `shaders/` on every new
and every open, overwriting both.

`material.rs` and `src/layer.wesl` are two halves of one thing — the uniform and the
colour ramps are written out in both. A change on either side is a change on both.

### The directory format

`terrain.ron` + `layer_<n>.png` are the values, read by both crates; `recipe.ron` and
`shaders/` are the editor's. The two move at different speeds: **the recipe is
rewritten after every edit that settles**, the values only on Save — so quitting
without saving loses nothing authored, and a directory whose recipe was deleted is
still a terrain. `meta::VERSION` is refused outright if it does not match; there is no
migration path. A terrain read is treated as untrusted input: declared sizes are
checked against the metadata before allocation, and names are checked to be names
rather than paths.

Water is all-or-nothing over the shift-0 height layer — it leaves the document only at
the border, so no rectangle can be re-solved alone, and any height edit invalidates the
whole answer.
