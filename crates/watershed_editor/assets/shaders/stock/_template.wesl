import package::lib::fbm_unit;

// A layer's shader: one compute shader that produces every value of one layer. The
// file is the layer — it lives in the document's `shaders` directory as
// `<layer>.wesl`, and a file whose name begins with `_` is not a layer. The editor
// appends the entry point, which calls `value(p)` once per texel of the layer's
// raster. `value` takes a position and returns one number, and what lands, clamped to
// the layer's range, is the layer.
//
// COORDINATES
//
// `p` is in document cells, not a normalised coordinate, so a scale means the same
// thing at every raster shift. For the 0..1 reading, `uv(p)` divides by
// `document_extent()`, the cells the document spans; on a document wider than it is
// tall that stretches, because it runs 0..1 on both axes.
//
//   import package::lib::{fbm_unit, uv};
//
//   fn value(p: vec2<f32>) -> f32 {
//       return fbm_unit(uv(p) * 4.0, 4u, 0.5, 2.0);
//   }
//
// is four noise cells across the document however large the document is.
//
// BINDINGS
//
// Binding 0 is `globals`, 1 is the output the entry point writes, 2 is the `Params`
// uniform the file declares. An `@layer` texture takes a binding from 3 up, no binding
// is used twice, and at most 8 are declared. `globals` carries:
//
//   document: vec2<u32>  the document's extent, in cells
//   texels:   vec2<u32>  the extent of this dispatch, in texels of this raster
//   origin:   vec2<u32>  where this dispatch starts; always zero, every bake is whole
//   shift:    u32        the raster shift: one texel per cell at 0, per 2^shift above
//   seed:     u32        the document's seed
//
// THE LIBRARY
//
// `lib.wesl`, which the editor writes beside this file and overwrites whenever a
// document is opened or created. Nothing in it is available without an import, at the
// top of the file and before any declaration. Import the items a file calls:
//
//   import package::lib::{fbm_unit, uv};
//
// or the module, and name each item through it:
//
//   import package::lib;
//
//   ... lib::uv(p) ...
//
// What it holds:
//
//   gradient_noise(p: vec2<f32>) -> f32
//       one octave, roughly -1..1, exactly zero at every integer lattice point
//   fbm(p, octaves: u32, persistence: f32, lacunarity: f32) -> f32
//       octaves summed and divided by the total amplitude; zero octaves is NaN
//   ridged_fbm(p, octaves: u32, persistence: f32, lacunarity: f32) -> f32
//       0..1, creases high, never negative and with no midpoint
//   fbm_unit(p, octaves: u32, persistence: f32, lacunarity: f32) -> f32
//       `fbm` stretched onto 0..1, the reading a height layer's range expects
//   seed_offset(seed: u32, salt: u32) -> vec2<f32>
//       a displacement to add to a noise position, distinct per seed and per salt
//   uv(p: vec2<f32>) -> vec2<f32>
//       a position as 0..1 across the document
//   document_extent() -> vec2<f32>
//       the cells the document spans
//   cell_position(id: vec2<u32>) -> vec2<f32>
//       the document position a texel of this raster writes; the entry point's own
//   layer_texel(p: vec2<f32>) -> vec2<i32>
//       the texel of this raster a position falls in — the inverse of `cell_position`
//   input_texel(source: texture_2d<f32>, at: vec2<i32>) -> f32
//       one texel of a layer texture, clamped to its edge
//   layer_value(layer: texture_2d<f32>, p: vec2<f32>) -> f32
//       another layer's value at a document position, interpolated, whatever its shift
//   layer_shift(layer: texture_2d<f32>) -> u32
//       the shift that layer was baked at
//   layer_class(layer: texture_2d<f32>, p: vec2<f32>) -> f32
//       another layer's value read whole, to the nearest texel — how a `@categorical`
//       layer is read, where an interpolated class index is a class nothing holds
//
// HEADER LINES
//
// What the layer itself is, as lines of their own anywhere in the file. Each may be
// written once; a line this file does not declare takes the default shown. Only the
// exact `// @` opening declares anything, so the forms written indented below declare
// nothing and neither does a copy of this file.
//
//   @role height|moisture|custom   what the bake does with the layer; custom
//   @shift <n>                     one texel per 2^n cells, 0 to 8; 0
//   @range <low> <high>            what baked values are clamped into; 0 1
//   @categorical                   the layer holds whole class indices, not a
//                                  quantity: it is read to the nearest texel and
//                                  exported as a class channel, and a value in it
//                                  that is not whole is refused at save
//
// A document has at most one height layer and at most one moisture layer, and the
// height layer is at shift 0. A water spec needs a height layer. Breaking one of
// those is a fault on the layer's card, and the document is not baked until it is
// mended.
//
// PARAMETERS — @ui
//
// One per field of `Params`, after the field's `//`. A field without one is a parse
// error shown on the layer's card, and the layer keeps the values it had.
//
//   @ui <default> [<min>, <max>]            f32, i32, u32 — a number
//   @ui <default> [<min>, <max>] step <s>   the same, stepped
//   @ui (<x>, <y>) [<min>, <max>]           vec2<f32> — one per component
//   @ui color srgb(<r>, <g>, <b>)           vec3<f32>, vec4<f32> — stored linear
//   @ui toggle <true|false>                 u32, i32 — a toggle
//   @ui hidden                              any — no widget; the default is used
//   @ui "Label" ...                         any — overrides the displayed name
//
// A `// @group <Name>` line of its own inside `Params` starts a section.
//
// LAYERS — @layer
//
// Another layer of the document, read by naming it after the texture's `//`:
//
//   @group(0) @binding(3) var base: texture_2d<f32>; // @layer base
//
// The binding holds that layer's baked raster, at the layer's own shift. Import
// `layer_value` and read it with `layer_value(base, p)`, or one texel at a time with
// `input_texel(base, layer_texel(p))` when both layers share a shift. The layer named
// is baked first. A name that is no layer, or one that makes layers read each other in
// a circle, leaves this layer unbaked with the reason on its card.

struct Params {
    // @group Shape
    scale: f32, // @ui 0.02 [0.001, 0.2]
}
@group(0) @binding(2) var<uniform> params: Params;

// Uncomment, and name a layer of the document, to read that layer with
// `layer_value(base, p)` — add `layer_value` to the import. Bindings start at 3.
// @group(0) @binding(3) var base: texture_2d<f32>; // @layer base

fn value(p: vec2<f32>) -> f32 {
    return fbm_unit(p * params.scale, 4u, 0.5, 2.0);
}
