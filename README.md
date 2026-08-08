# watershed

| crate | what it is |
|---|---|
| `watershed` | library: named terrain fields built from layer stacks, a whole-grid water solver, a terrain file format |
| `watershed_editor` | bevy editor: 2D false-colour view, noise layers, painting, water solve, save/load |

TODO(jb-doc): overview of the terrain model and the water solve.

## Commands

`just --list` is the entry point.

| | |
|---|---|
| `just run` | run the editor |
| `just test` | `cargo test --locked --workspace` |
| `just clippy` | Clippy over all targets/features on the `ci` profile |
| `just bevy-lints` | Bevy-specific lints over the editor (needs `bevy_lint`; install via `just bevy-lint-install`) |
| `just fmt` | `cargo fmt --check` |
| `just docs` | `cargo doc` over the workspace |
| `just check-web` | wasm32 compile check of the library |
| `just all` | everything, in CI order |
| `just deps` | apt packages CI needs (alsa, udev, wayland headers) |

## Licence

MIT OR Apache-2.0.
