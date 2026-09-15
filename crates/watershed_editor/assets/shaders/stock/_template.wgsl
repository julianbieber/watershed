// A field's shader: one compute shader that produces every value of one field. The
// file is the field — it lives in the document's `shaders` directory as
// `<field>.wgsl`, and a file whose name begins with `_` is not a field. The editor
// appends the entry point, which calls `value(p)` once per texel of the field's
// raster. `value` takes a position and returns one number, and what lands, clamped to
// the field's range, is the field.
//
// COORDINATES
//
// `p` is in document cells, not a normalised coordinate, so a scale means the same
// thing at every raster shift. For the 0..1 reading, `uv(p)` divides by
// `document_extent()`, the cells the document spans; on a document wider than it is
// tall that stretches, because it runs 0..1 on both axes.
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
// From `field_lib.wgsl`, compiled ahead of this file.
//
//   gradient_noise(p: vec2<f32>) -> f32
//       one octave, roughly -1..1, exactly zero at every integer lattice point
//   fbm(p, octaves: u32, persistence: f32, lacunarity: f32) -> f32
//       octaves summed and divided by the total amplitude; zero octaves is NaN
//   ridged_fbm(p, octaves: u32, persistence: f32, lacunarity: f32) -> f32
//       0..1, creases high, never negative and with no midpoint
//   fbm_unit(p, octaves: u32, persistence: f32, lacunarity: f32) -> f32
//       `fbm` stretched onto 0..1, the reading a height field's range expects
//   seed_offset(seed: u32, salt: u32) -> vec2<f32>
//       a displacement to add to a noise position, distinct per seed and per salt
//   uv(p: vec2<f32>) -> vec2<f32>
//       a position as 0..1 across the document
//   document_extent() -> vec2<f32>
//       the cells the document spans
//   cell_position(id: vec2<u32>) -> vec2<f32>
//       the document position a texel of this raster writes; the entry point's own
//   field_texel(p: vec2<f32>) -> vec2<i32>
//       the texel of this raster a position falls in — the inverse of `cell_position`
//   input_texel(source: texture_2d<f32>, at: vec2<i32>) -> f32
//       one texel of a layer texture, clamped to its edge
//   layer_value(layer: texture_2d<f32>, p: vec2<f32>) -> f32
//       another field's value at a document position, interpolated, whatever its shift
//   layer_shift(layer: texture_2d<f32>) -> u32
//       the shift that field was baked at
//
// PARAMETERS — @ui
//
// One per field of `Params`, after the field's `//`. A field without one is a parse
// error shown on the field's card, and the field keeps the values it had.
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
// Another field of the document, read by naming it after the texture's `//`:
//
//   @group(0) @binding(3) var base: texture_2d<f32>; // @layer base
//
// The binding holds that field's baked raster, at the field's own shift. Read it with
// `layer_value(base, p)`, or one texel at a time with
// `input_texel(base, field_texel(p))` when both fields share a shift. The field named
// is baked first. A name that is no field, or one that makes fields read each other in
// a circle, leaves this field unbaked with the reason on its card.

struct Params {
    // @group Shape
    scale: f32, // @ui 0.02 [0.001, 0.2]
}
@group(0) @binding(2) var<uniform> params: Params;

// Uncomment, and name a field of the document, to read that field with
// `layer_value(base, p)`. Bindings start at 3.
// @group(0) @binding(3) var base: texture_2d<f32>; // @layer base

fn value(p: vec2<f32>) -> f32 {
    return fbm_unit(p * params.scale, 4u, 0.5, 2.0);
}
