import package::lib::fbm_unit;

struct Params {
    // @group Shape
    scale: f32,   // @ui 0.02 [0.001, 0.2]
    octaves: u32, // @ui 4 [1, 8] step 1
    // @group Steps
    steps: f32,   // @ui 8.0 [2.0, 32.0] step 1.0
    hardness: f32,// @ui "Edge hardness" 0.7 [0.0, 1.0]
}
@group(0) @binding(2) var<uniform> params: Params;

fn value(p: vec2<f32>) -> f32 {
    let height = fbm_unit(p * params.scale, params.octaves, 0.5, 2.0);
    let scaled = height * params.steps;
    let stepped = floor(scaled);
    let inside = scaled - stepped;
    let eased = mix(inside, smoothstep(0.0, 1.0, inside), params.hardness);
    return (stepped + eased) / params.steps;
}
