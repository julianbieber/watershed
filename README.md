# watershed

| crate | what it is |
|---|---|
| `watershed` | library: reading a terrain — named fields over an extent, and a solved water state |
| `watershed_editor` | bevy editor: authoring a terrain, baking it, and writing it out |

## The two halves

**A terrain is read from a directory and authored in the editor, and nothing does both.**

`watershed` is the read side. It holds `Terrain` — the values and nothing that made
them — its channels and metadata, and the loader that opens a terrain directory. It
bakes nothing, solves nothing and evaluates nothing: everything a consuming project
reads was settled when the directory was written, so `Terrain::load_from_dir` is the
only way to obtain one. It needs no GPU and no thread pool, and builds for the web.

`watershed_editor` is the author side, and it carries the whole model a document is
made of: the layers and their shader files, the water solve, and the bake. A project
that wants a terrain ships the directory the editor wrote; it does not build one at run
time.

## The terrain model

A terrain is an extent in cells and a set of named layers over it. A layer is one WGSL
file in the document's `shaders` directory, named after the layer, and its values are
what that shader produces, clamped to the layer's declared range. The file declares the
layer as well as its values: its role in the bake, its resolution shift, that range and
whether it holds classes are four header lines in the file itself. A layer reads another
by naming it in its file with `@layer`, so baking is ordered: every layer after the ones
it reads. Every edit re-bakes the whole document.

A layer is evaluated onto a raster of its own resolution, chosen as a *shift*: one
texel per cell at 0, one per `2^shift` cells above that. A coarse layer still answers
at every cell of the document, interpolated between its texels.

Two types carry all of this. `TerrainSpec` is the authored document: it holds each
layer's settings and parameter values, it is what an editor mutates, and it lives in the
editor. `Terrain` is what baking one produces, and it is what a consuming project holds.

## Shaders

A layer's values come from its WESL compute shader. This is how a new way of making a
layer is added without touching Rust: write the file, and the editor reads what it
declares and draws the panel for it.

The shaders live in the document's own `shaders` directory, so a terrain stays portable,
and they are hot-reloaded — save a file and the document re-bakes. Adding a layer copies
the template into that directory as `<layer>.wesl`, where it is then yours to edit, and
removing one deletes its file. A file added to or deleted from the directory by hand
adds or removes its layer the same way; a file whose name begins with `_` is not a
layer. Neither is an undo step: undo covers parameter values, the display settings and
the water spec — not the four the file declares, which the next read of the file would
put back anyway.

The panel names the layer's file by its absolute path and carries an **Open** button
that hands it to `$VISUAL`, or to `xdg-open` when that is not set.

```wesl
import package::lib::ridged_fbm;

struct Params {
    // @group Shape
    scale: f32,     // @ui 0.02 [0.001, 0.2]
    octaves: u32,   // @ui 5 [1, 8] step 1
    // @group Crest
    sharpness: f32, // @ui "Sharpness" 1.0 [0.2, 4.0]
}
@group(0) @binding(2) var<uniform> params: Params;

fn value(p: vec2<f32>) -> f32 {
    return pow(ridged_fbm(p * params.scale, params.octaves, 0.5, 2.0), params.sharpness);
}
```

`p` is a position in **document cells**, not a normalised coordinate, so a scale means
the same thing at every shift. The entry point is appended by the editor; the bindings,
the noise, `cell_position`, and `uv` and `document_extent` — the 0..1 coordinate across
the document, and the cells it spans — come from the library, `lib.wesl`.

The editor writes `lib.wesl` into `shaders/`, with a `wesl.toml` beside it, on every new
document and every open, overwriting both each time. A layer imports what it uses from
`package::lib` — `import package::lib::{fbm_unit, uv};`, or `import package::lib;` and
then `lib::uv(p)` — and nothing in the library is available without the import. Because
the library is a file on disk, a WESL language server resolves the import too. A `.wgsl`
file in `shaders/` is not a layer, and the log names it.

A new layer's file arrives with all of this in its own header: every library function,
the bindings, the coordinate convention and the annotations below, so the file need
not be left to write the first line. The same text is in the editor, under
**Reference** on a layer's panel.

### `@ui` annotations

One per field of the `Params` struct. A field without one is a parse error, reported in
the status bar; the layer keeps the values it had, because a shader is edited in place
and is expected to be broken for as long as it takes to type the next line.

| Form | Field types | Widget |
|---|---|---|
| `@ui <default> [<min>, <max>]` | `f32`, `i32`, `u32` | number |
| `@ui <default> [<min>, <max>] step <s>` | `f32`, `i32`, `u32` | number, stepped |
| `@ui (<x>, <y>) [<min>, <max>]` | `vec2<f32>` | one per component |
| `@ui color srgb(<r>, <g>, <b>)` | `vec3<f32>`, `vec4<f32>` | colour, stored linear |
| `@ui toggle <true\|false>` | `u32`, `i32` | toggle |
| `@ui hidden` | any | none; the default is used |
| `@ui "Label" ...` | any | overrides the displayed name |
| `// @group <Name>` on its own line | — | starts a section |

A parameter the document carries that the file no longer declares is dropped when thefile is re-read; one the file declares that the document lacks takes the file's
default.

### `@layer`

Another layer of the document, read by naming it after a texture binding's `//`:

```wesl
import package::lib::layer_value;

@group(0) @binding(3) var base: texture_2d<f32>; // @layer base

fn value(p: vec2<f32>) -> f32 {
    return layer_value(base, p) + 0.1;
}
```

Bindings start at 3, since 0, 1 and 2 are the globals, the output and the parameters.
The binding holds the named layer's baked raster at that layer's own shift, and
`layer_value` reads it at a document position whatever the shift. The layer named is
baked first; a name that is no layer, or files that read each other in a circle, leave
the reading layer unbaked with the reason on its card.

### Header lines

What the layer itself is, as lines of their own anywhere in the file. Each may be
written once; a line the file does not declare takes the default.

| line | what it says | default |
| --- | --- | --- |
| `// @role height\|moisture\|custom` | what the bake does with the layer | `custom` |
| `// @shift <n>` | one texel per `2^n` cells, 0 to 8 | `0` |
| `// @range <low> <high>` | what baked values are clamped into | `0 1` |
| `// @categorical` | the layer holds whole class indices, not a quantity | a quantity |

A categorical layer is read to the nearest texel rather than interpolated — `layer_class`
is the library's read of one — and is exported as a class channel; a value in it that is
not whole is refused when the document is saved.

A document has at most one height layer and at most one moisture layer, and the height
layer is at shift 0. A water spec needs a height layer. Breaking one of those is a fault
on the layer's card and in the log, and the document is not baked until it is mended.

## The water solve

Water is a whole-grid answer over one layer — the one holding the `Height` role, at
shift 0. Depressions are flooded to their outlets, every cell is given a downhill
neighbour on the filled surface, and the weight each cell contributes (`1.0`, or a
moisture layer's value) is accumulated downstream. What comes back is a depth, a flow
direction, an accumulation and a lake id per cell.

It is all or nothing: water leaves the document only at its border, so a rectangle
cannot be re-solved on its own. An edit to the height therefore invalidates the whole
answer — but not the document that produced it, which is what can be solved again.

## The file format

A terrain is a directory.

| file | what it is | who reads it |
|---|---|---|
| `terrain.ron` | the extent, the fields, the images, the water | both |
| `layer_<n>.png` | the values, eight bits to a channel | both |
| `recipe.ron` | the extent, the seed, the water spec, and per layer its parameter values | the editor |
| `shaders/*.wesl` | one file per layer | the editor |
| `shaders/lib.wesl` | the library, overwritten by the editor | the editor |
| `shaders/wesl.toml` | the package root, for a language server | a language server |

A `layer_<n>.png` is not an authored layer: it is an image the library packs the values
of every layer at one shift into, and `terrain.ron` calls the authored ones `fields`.

Every save writes both: the values and the recipe beside them. A terrain directory is
a project the editor is opened on, so a save that left the recipe out would make the
directory unopenable.

The split is what makes the boundary real: a reader of values never parses a shader,
so it never needs the types a layer's shader is made of. A directory whose recipe has
been deleted is still a terrain.

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