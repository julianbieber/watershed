import package::lib::{fbm_unit, seed_offset};

struct Params {
    // @group Land
    land_scale: f32,   // @ui "Land scale" 0.0015 [0.0002, 0.02]
    // @group Relief
    relief_scale: f32, // @ui "Relief scale" 0.04 [0.005, 0.2]
    relief: f32,       // @ui 0.18 [0.0, 1.0]
    seed: u32,         // @ui hidden
}
@group(0) @binding(2) var<uniform> params: Params;

fn value(p: vec2<f32>) -> f32 {
    let land = fbm_unit(p * params.land_scale + seed_offset(params.seed, 1u), 4u, 0.5, 2.0);
    let detail = fbm_unit(p * params.relief_scale + seed_offset(params.seed, 2u), 4u, 0.5, 2.0);
    return land + params.relief * detail;
}
