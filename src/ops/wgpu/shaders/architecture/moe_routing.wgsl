// MoE Top-K Routing Shader (F32)
// Computes softmax over experts, selects top-k, normalizes weights.
// One workgroup per token.

struct MoERoutingParams {
    num_tokens: u32,
    num_experts: u32,
    k: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> out_weights: array<f32>;
@group(0) @binding(3) var<uniform> params: MoERoutingParams;

var<workgroup> probs: array<f32, 256>;  // max num_experts = 256

@compute @workgroup_size(1)
fn moe_routing_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;
    if (token_idx >= params.num_tokens) {
        return;
    }

    let num_experts = params.num_experts;
    let k = params.k;
    let base = token_idx * num_experts;

    // Find max for numerical stability
    var max_val: f32 = -1e30;
    for (var e = 0u; e < num_experts; e = e + 1u) {
        let val = logits[base + e];
        if (val > max_val) {
            max_val = val;
        }
    }

    // Compute exp and sum
    var exp_sum: f32 = 0.0;
    for (var e = 0u; e < num_experts; e = e + 1u) {
        let val = exp(logits[base + e] - max_val);
        probs[e] = val;
        exp_sum = exp_sum + val;
    }

    // Normalize to softmax
    let inv_sum = 1.0 / exp_sum;
    for (var e = 0u; e < num_experts; e = e + 1u) {
        probs[e] = probs[e] * inv_sum;
    }

    // Top-k selection
    let out_base = token_idx * k;
    var top_sum: f32 = 0.0;
    for (var ki = 0u; ki < k; ki = ki + 1u) {
        var best_val: f32 = -1.0;
        var best_idx: u32 = 0u;
        for (var e = 0u; e < num_experts; e = e + 1u) {
            if (probs[e] > best_val) {
                best_val = probs[e];
                best_idx = e;
            }
        }
        out_indices[out_base + ki] = i32(best_idx);
        out_weights[out_base + ki] = best_val;
        top_sum = top_sum + best_val;
        probs[best_idx] = -1.0;
    }

    // Normalize top-k weights
    if (top_sum > 0.0) {
        let inv_top = 1.0 / top_sum;
        for (var ki = 0u; ki < k; ki = ki + 1u) {
            out_weights[out_base + ki] = out_weights[out_base + ki] * inv_top;
        }
    }
}
