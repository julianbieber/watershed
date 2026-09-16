// @role height

import package::lib::{fbm_unit, layer_value, ridged_fbm, seed_offset};

struct Params {
    // @group Ridges
    ridge_scale: f32,  // @ui "Ridge scale" 0.006 [0.001, 0.05]
    ridge: f32,        // @ui 0.55 [0.0, 2.0]
    // @group Where
    low: f32,          // @ui "Starts at" 0.45 [0.0, 1.0]
    high: f32,         // @ui "Full at" 0.75 [0.0, 1.0]
    // @group Relief
    relief_scale: f32, // @ui "Relief scale" 0.04 [0.005, 0.2]
    relief: f32,       // @ui 0.08 [0.0, 1.0]
    seed: u32,         // @ui hidden
}
@group(0) @binding(2) var<uniform> params: Params;

@group(0) @binding(3) var base: texture_2d<f32>; // @layer base

fn value(p: vec2<f32>) -> f32 {
    let under = layer_value(base, p);
    let crest = ridged_fbm(p * params.ridge_scale + seed_offset(params.seed, 3u), 5u, 0.5, 2.0);
    let span = params.high - params.low;
    let weight = select(0.0, clamp((under - params.low) / span, 0.0, 1.0), span != 0.0);
    let lifted = mix(under, under + crest * params.ridge, weight);
    let detail = fbm_unit(p * params.relief_scale + seed_offset(params.seed, 4u), 3u, 0.5, 2.0);
    return lifted + detail * params.relief;
}
