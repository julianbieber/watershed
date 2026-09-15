struct Params {
    // @group Shape
    scale: f32,   // @ui 0.004 [0.0005, 0.2]
    octaves: u32, // @ui 4 [1, 8] step 1
    seed: u32,    // @ui hidden
}
@group(0) @binding(2) var<uniform> params: Params;

fn value(p: vec2<f32>) -> f32 {
    return fbm_unit(p * params.scale + seed_offset(params.seed, 0u), params.octaves, 0.5, 2.0);
}
