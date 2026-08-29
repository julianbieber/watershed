// @shader Ridged

struct Params {
    // @group Shape
    scale: f32,      // @ui 0.02 [0.001, 0.2]
    octaves: u32,    // @ui 5 [1, 8] step 1
    // @group Crest
    sharpness: f32,  // @ui "Sharpness" 1.0 [0.2, 4.0]
    lift: f32,       // @ui 0.0 [-1.0, 1.0]
}
@group(0) @binding(2) var<uniform> params: Params;

fn value(p: vec2<f32>) -> f32 {
    let ridge = ridged_fbm(p * params.scale, params.octaves, 0.5, 2.0);
    return clamp(pow(ridge, params.sharpness) + params.lift, 0.0, 1.0);
}
