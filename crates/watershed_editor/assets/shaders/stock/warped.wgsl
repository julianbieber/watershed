struct Params {
    // @group Shape
    scale: f32,     // @ui 0.015 [0.001, 0.2]
    octaves: u32,   // @ui 5 [1, 8] step 1
    // @group Warp
    strength: f32,  // @ui 40.0 [0.0, 200.0]
    warp_scale: f32,// @ui "Warp scale" 0.004 [0.0005, 0.05]
}
@group(0) @binding(2) var<uniform> params: Params;

fn value(p: vec2<f32>) -> f32 {
    let offset = vec2<f32>(
        gradient_noise(p * params.warp_scale),
        gradient_noise((p + vec2<f32>(137.0, 311.0)) * params.warp_scale),
    );
    return fbm_unit((p + offset * params.strength) * params.scale, params.octaves, 0.5, 2.0);
}
