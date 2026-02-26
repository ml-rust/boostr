// Speculative Decoding Shaders
//
// Element-wise acceptance probability and expected token count computation.
// Token verification (verify_speculative_tokens) is handled by impl_generic
// using numr's philox_uniform for reproducible, backend-consistent RNG.
// F32 only (WebGPU limitation).

// -----------------------------------------------------------------
// Element-wise acceptance and residual probabilities
// -----------------------------------------------------------------

struct AcceptParams {
    total_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> accept_draft: array<f32>;
@group(0) @binding(1) var<storage, read> accept_target: array<f32>;
@group(0) @binding(2) var<storage, read_write> acceptance_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> residual_out: array<f32>;
@group(0) @binding(4) var<uniform> accept_params: AcceptParams;

@compute @workgroup_size(256)
fn compute_acceptance_probs_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= accept_params.total_elements) {
        return;
    }

    let dp = accept_draft[idx];
    let tp = accept_target[idx];

    // Acceptance: min(1, target / draft)
    var accept = 1.0;
    if (dp > 1e-10) {
        accept = min(1.0, tp / dp);
    }
    acceptance_out[idx] = accept;

    // Residual: max(0, target - draft)
    residual_out[idx] = max(0.0, tp - dp);
}

// -----------------------------------------------------------------
// Expected tokens computation
// -----------------------------------------------------------------

struct ExpectedParams {
    batch_size: u32,
    max_spec_tokens: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> exp_rates: array<f32>;
@group(0) @binding(1) var<storage, read_write> expected_out: array<f32>;
@group(0) @binding(2) var<uniform> expected_params: ExpectedParams;

@compute @workgroup_size(256)
fn compute_expected_tokens_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.x;
    if (batch_idx >= expected_params.batch_size) {
        return;
    }

    let K = expected_params.max_spec_tokens;
    var cumulative_prob = 1.0;
    var expected = 0.0;

    for (var i = 0u; i < K; i = i + 1u) {
        cumulative_prob = cumulative_prob * exp_rates[batch_idx * K + i];
        expected = expected + cumulative_prob;
    }

    // +1 for bonus token
    expected_out[batch_idx] = expected + 1.0;
}
