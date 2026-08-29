// @shader Template

struct Params {
    // @group Shape
    scale: f32, // @ui 0.02 [0.001, 0.2]
}
@group(0) @binding(2) var<uniform> params: Params;

fn value(p: vec2<f32>) -> f32 {
    return fbm_unit(p * params.scale, 4u, 0.5, 2.0);
}
