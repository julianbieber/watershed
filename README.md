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
made of: the fields and their layer stacks, the noise, the region tiling, the brush,
the water solve, the bake, and the shaders. A project that wants a terrain ships the
directory the editor wrote; it does not build one at run time.

## The terrain model

A terrain is an extent in cells and a set of named fields over it. A field is a stack
of layers — noise, a compute shader, painted rasters, the slope of another field, a
region tiling — each scaled, confined by a mask and blended onto what is under it,
then clamped to the field's declared range. Layers may reference other fields, so
baking is ordered: every field after the ones it reads.

A field is evaluated onto a raster of its own resolution, chosen as a *shift*: one
texel per cell at 0, one per `2^shift` cells above that. A coarse field still answers
at every cell of the document, interpolated between its texels — unless its values
name a class rather than measure a quantity, in which case it is read to the nearest
texel instead.

Two types carry all of this. `TerrainSpec` is the authored document: it holds the
layers, it is what an editor mutates, and it lives in the editor. `Terrain` is what
baking one produces, and it is what a consuming project holds.

## Shader layers

A layer's values may come from a WGSL compute shader instead of from an op the editor
has a name for. This is how a new way of making a field is added without touching
Rust: write the file, and the editor reads what it declares and draws the panel for
it.

A shader lives in the document's own `shaders` directory, so a terrain stays portable,
and it is hot-reloaded — save the file and the field re-bakes. Adding a shader layer
copies one of the shipped shaders in under a fresh name, which is then yours to edit.

```wgsl
// @shader Ridged

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
the same thing at every shift and a rectangle re-bake produces what a whole bake
would. The entry point is appended by the editor; the bindings, the noise,
`cell_position`, and `uv` and `document_extent` — the 0..1 coordinate across the
document, and the cells it spans — come from `assets/shaders/field_lib.wgsl`, whose
noise is the same algorithm the CPU noise layers use so one name does not mean two
functions inside one stack.

A copied shader arrives with all of this in its own header: every library function,
the bindings, the coordinate convention and the annotations below, so the file need
not be left to write the first line.

A shader may also declare input textures, one pin on its node per declaration, and
read the upstream raster at any texel:

```wgsl
@group(0) @binding(3) var source: texture_2d<f32>; // @in "Source"

fn value(p: vec2<f32>) -> f32 {
    return input_texel(source, field_texel(p) + vec2<i32>(1, 0));
}
```

Bindings start at 3, since 0, 1 and 2 are the globals, the output and the parameters,
and an unwired pin reads `0.0`. What the pin carries is evaluated over the whole field
and dispatched inside the bake, which is also where the raster the shader produces is
read back.

A document holding such a shader wired up re-bakes the **whole field** on every
upstream edit, because nothing bounds how far the shader reads — unless the file says
so with `@reach`, in which case the re-bake stays inside the visible rectangle and
pads by the declared reach.

A shader names no field, so it still adds no bake-order dependency. It composes with
the rest of the stack through the layer's own amplitude, mask and blend, exactly as
every other op does — the shader is dispatched to a raster first, and the stack is
then walked on the CPU as it always was.

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

A parameter the document carries that the file no longer declares is dropped when the
file is re-read; one the file declares that the document lacks takes the file's
default. A file whose name begins with `_` is a template and is not offered as a layer.

### `@reach`

How far the shader reads around the texel it writes, on a line of its own:

```wgsl
// @reach 2
```

The unit is **document cells** — the unit `p` is measured in — so a shader that
offsets in texels of its own field declares `offset << shift` cells. The line may sit
anywhere in the file, and only a whole line counts: a trailing `// @reach 2` after
code declares nothing, and neither does a commented-out `// // @reach 2`. Declaring it
twice, or declaring anything but a non-negative whole number, is a parse error.

Without it, a wired shader re-bakes the whole field. With it, an edit upstream re-bakes
only the visible rectangle, widened by the declared reach at each dependency hop the
way a slope's sample distance already widens one. Nothing checks the number against
what the shader actually samples: a shader that reads further than it declares leaves
stale values inside the rectangle.

## The water solve

Water is a whole-grid answer over one field — the one holding the `Height` role, at
shift 0. Depressions are flooded to their outlets, every cell is given a downhill
neighbour on the filled surface, and the weight each cell contributes (`1.0`, or a
moisture field's value) is accumulated downstream. What comes back is a depth, a flow
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
| `recipe.ron` | the layer stacks and the water spec | the editor |
| `paint_<n>.png` | the painted rasters the stacks name | the editor |
| `shaders/*.wgsl` | the shaders the stacks name | the editor |

The values are always written; the recipe only when the editor saves a *document*. An
export writes the values alone, and removes a recipe already in the directory — one
left behind would claim to describe values it no longer produced.

The split is what makes the boundary real: a reader of values never parses a layer
stack, so it never needs the types a layer stack is made of. A directory whose recipe
has been deleted is still a terrain.

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
