# watershed

| crate | what it is |
|---|---|
| `watershed` | library: named terrain fields built from layer stacks, a whole-grid water solver, a terrain file format |
| `watershed_editor` | bevy editor: 2D false-colour view, noise layers, painting, water solve, save/load |

## The terrain model

A terrain is an extent in cells and a set of named fields over it. A field is a stack of
layers — noise, painted rasters, the slope of another field, a region tiling — each scaled,
confined by a mask and blended onto what is under it, then clamped to the field's declared
range. Layers may reference other fields, so baking is ordered: every field after the ones
it reads.

A field is evaluated onto a raster of its own resolution, chosen as a *shift*: one texel per
cell at 0, one per `2^shift` cells above that. A coarse field still answers at every cell of
the document, interpolated between its texels — unless its values name a class rather than
measure a quantity, in which case it is read to the nearest texel instead.

Two types carry all of this, and which one you hold says what you are doing. `TerrainSpec`
is the authored document: it holds the layers, it is what is saved and loaded, and it is
what an editor mutates. `Terrain` is what baking one produces: the values and nothing that
made them, for an application that reads a document it could not author.

Nothing derived is stored in a file. A loaded document arrives unbaked and every field
samples as `0.0` until it is baked; the recipe is what the format carries, not the result.

## The water solve

Water is a whole-grid answer over one field — the one holding the `Height` role, at shift 0.
Depressions are flooded to their outlets, every cell is given a downhill neighbour on the
filled surface, and the weight each cell contributes (`1.0`, or a moisture field's value) is
accumulated downstream. What comes back is a depth, a flow direction, an accumulation and a
lake id per cell.

It is all or nothing: water leaves the document only at its border, so a rectangle cannot be
re-solved on its own. An edit to the height therefore invalidates the whole answer — but not
the spec that produced it, which is what a document carries so it can be solved again.

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
