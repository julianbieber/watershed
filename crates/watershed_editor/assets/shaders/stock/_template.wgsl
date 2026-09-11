// @shader Template

struct Params {
    // @group Shape
    scale: f32, // @ui 0.02 [0.001, 0.2]
}
@group(0) @binding(2) var<uniform> params: Params;

// Uncomment to grow an input pin on the node, and read it with
// `input_texel(source, field_texel(p))`. Bindings start at 3.
// @group(0) @binding(3) var source: texture_2d<f32>; // @in "Source"

fn value(p: vec2<f32>) -> f32 {
    return fbm_unit(p * params.scale, 4u, 0.5, 2.0);
}
