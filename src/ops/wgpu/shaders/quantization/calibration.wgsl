//! Calibration shaders for quantization (AWQ, Fisher Information)
//! F32 only (WebGPU limitation)

// ============================================================================
// AWQ: Per-channel max-abs activation scale
// activations: [N, K], output: [K]
// ============================================================================

struct AwqActScaleParams {
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read> awq_act: array<f32>;
@group(0) @binding(1) var<storage, read_write> awq_act_scale_out: array<f32>;
@group(0) @binding(2) var<uniform> awq_act_params: AwqActScaleParams;

@compute @workgroup_size(256)
fn awq_act_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= awq_act_params.K {
        return;
    }

    var max_val: f32 = 0.0;
    for (var n: u32 = 0u; n < awq_act_params.N; n = n + 1u) {
        let val = abs(awq_act[n * awq_act_params.K + j]);
        max_val = max(max_val, val);
    }
    awq_act_scale_out[j] = max_val;
}

// ============================================================================
// AWQ: Score reduction
// weights: [M, K], act_scale: [K], output: [K]
// score[j] = mean_i(act_scale[j] * |W[i, j]|)
// ============================================================================

struct AwqScoreParams {
    M: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read> awq_weights: array<f32>;
@group(0) @binding(1) var<storage, read> awq_scale: array<f32>;
@group(0) @binding(2) var<storage, read_write> awq_score_out: array<f32>;
@group(0) @binding(3) var<uniform> awq_score_params: AwqScoreParams;

@compute @workgroup_size(256)
fn awq_score_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= awq_score_params.K {
        return;
    }

    let scale = awq_scale[j];
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < awq_score_params.M; i = i + 1u) {
        acc = acc + scale * abs(awq_weights[i * awq_score_params.K + j]);
    }
    awq_score_out[j] = acc / f32(awq_score_params.M);
}

// ============================================================================
// Fisher: Squared gradient accumulation
// gradients: [N, P], output: [P]
// fisher[i] = mean_n(grad[n, i]^2)
// ============================================================================

struct FisherParams {
    N: u32,
    P: u32,
}

@group(0) @binding(0) var<storage, read> fisher_grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> fisher_out: array<f32>;
@group(0) @binding(2) var<uniform> fisher_params: FisherParams;

@compute @workgroup_size(256)
fn fisher_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= fisher_params.P {
        return;
    }

    var acc: f32 = 0.0;
    for (var n: u32 = 0u; n < fisher_params.N; n = n + 1u) {
        let g = fisher_grad[n * fisher_params.P + p];
        acc = acc + g * g;
    }
    fisher_out[p] = acc / f32(fisher_params.N);
}
