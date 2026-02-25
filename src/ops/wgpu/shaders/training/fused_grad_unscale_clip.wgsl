// Fused gradient unscale + clip — F32 only (WebGPU)
//
// Pass 1 (fused_grad_unscale_clip_f32):
//   - Check inf/nan → atomicOr to found_inf
//   - Unscale: out = grad * inv_scale
//   - Accumulate norm² via atomicAdd
//
// Pass 2 (clip_scale_f32):
//   - Scale all elements by inv_scale (used as clip_coef)

struct UnscaleClipParams {
    inv_scale: f32,
    max_norm: f32,
    n: u32,
    _pad: u32,
}

// Result buffer layout: [found_inf: atomic<u32>, norm_sq_bits: atomic<u32>]

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: UnscaleClipParams;

// Atomic float add via CAS loop on u32 bits
fn atomic_add_f32(idx: u32, val: f32) {
    var old_bits = atomicLoad(&result[idx]);
    loop {
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let result_bits = atomicCompareExchangeWeak(&result[idx], old_bits, new_bits);
        if (result_bits.exchanged) {
            break;
        }
        old_bits = result_bits.old_value;
    }
}

@compute @workgroup_size(256)
fn fused_grad_unscale_clip_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }

    let gi = grad[idx];

    // Check inf/nan
    if (gi != gi || gi == bitcast<f32>(0x7f800000u) || gi == bitcast<f32>(0xff800000u)) {
        atomicMax(&result[0], 1u);
    }

    let val = gi * params.inv_scale;
    out[idx] = val;

    // Accumulate norm² into result[1] as float bits
    atomic_add_f32(1u, val * val);
}

@compute @workgroup_size(256)
fn clip_scale_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }

    out[idx] = out[idx] * params.inv_scale;
}
