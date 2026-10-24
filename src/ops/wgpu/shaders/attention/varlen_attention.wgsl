//! Variable-length (packed) Flash Attention
//! Q, K, V are [total_tokens, num_heads, head_dim] (packed sequences)
//! cu_seqlens are [batch_size + 1] cumulative indices (I32)
//!
//! Each workgroup handles one token position across all heads.

struct VarlenParams {
    total_tokens_q: u32,
    total_tokens_k: u32,
    num_heads: u32,
    head_dim: u32,
    batch_size: u32,
    causal: u32,
    scale: f32,
    window: u32,
}

// True when key `j` is masked for query `i`. `q_pos = key_offset + i`.
// The window test `j + window <= q_pos` is written without the sum, so it cannot overflow.
fn varlen_masked(j: u32, q_pos: u32, causal: u32, window: u32) -> bool {
    let beyond_causal = causal != 0u && j > q_pos;
    let beyond_window = window > 0u && q_pos >= window && j <= q_pos - window;
    return beyond_causal || beyond_window;
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> cu_seqlens_q: array<i32>;
@group(0) @binding(4) var<storage, read> cu_seqlens_k: array<i32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<storage, read_write> lse: array<f32>;
@group(0) @binding(7) var<uniform> params: VarlenParams;

@compute @workgroup_size(256)
fn varlen_attention_fwd_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;

    if token_idx >= params.total_tokens_q {
        return;
    }

    // Find which batch this token belongs to
    var batch_idx = 0u;
    var batch_start_q = 0u;
    var batch_end_q = 0u;
    var batch_start_k = 0u;

    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        let start_q = u32(cu_seqlens_q[b]);
        let end_q = u32(cu_seqlens_q[b + 1u]);
        if token_idx >= start_q && token_idx < end_q {
            batch_idx = b;
            batch_start_q = start_q;
            batch_end_q = end_q;
            batch_start_k = u32(cu_seqlens_k[b]);
            break;
        }
    }

    let batch_end_k = u32(cu_seqlens_k[batch_idx + 1u]);
    let batch_len_k = batch_end_k - batch_start_k;
    let batch_len_q = batch_end_q - batch_start_q;

    // Position within batch
    let pos_q = token_idx - batch_start_q;

    // Bottom-right (ABSOLUTE) alignment for the causal and window rules, per sequence: this sequence's
    // batch_len_q query rows are the LAST positions of its batch_len_k keys, so
    // row pos_q sits at absolute position key_offset + pos_q. A full prefill
    // (batch_len_q == batch_len_k) gives key_offset == 0, leaving the rule
    // unchanged. Same convention as
    // `ops/impl_generic/attention/flash_standard.rs::build_attention_mask`.
    let key_offset = select(0u, batch_len_k - batch_len_q, batch_len_k >= batch_len_q);
    let q_pos = key_offset + pos_q;

    // Compute attention across heads (simplified: single thread per token, iterate heads)
    var accum: array<f32, 512>;  // Max head_dim

    for (var h = 0u; h < params.num_heads; h = h + 1u) {
        var max_score = -1e30f;
        var sum_exp = 0.0f;

        // Read Q[token_idx, h, :]
        let q_base = (token_idx * params.num_heads + h) * params.head_dim;

        // Compute max score
        let k_start = batch_start_k;
        let k_end = batch_end_k;
        var any_key = false;

        for (var k_idx = k_start; k_idx < k_end; k_idx = k_idx + 1u) {
            if varlen_masked(k_idx - batch_start_k, q_pos, params.causal, params.window) {
                continue;
            }
            any_key = true;
            let k_base = (k_idx * params.num_heads + h) * params.head_dim;
            var score = 0.0f;
            for (var d = 0u; d < params.head_dim; d = d + 1u) {
                score += q[q_base + d] * k[k_base + d];
            }
            score *= params.scale;
            max_score = max(max_score, score);
        }

        // A row with no unmasked key outputs zeros and lse = -inf.
        if !any_key {
            let out_zero = (token_idx * params.num_heads + h) * params.head_dim;
            for (var d = 0u; d < params.head_dim; d = d + 1u) {
                out[out_zero + d] = 0.0f;
            }
            var neg_inf_bits = 0xff800000u;
            lse[token_idx * params.num_heads + h] = bitcast<f32>(neg_inf_bits);
            continue;
        }

        // Second pass: softmax and aggregate
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            accum[d] = 0.0f;
        }

        for (var k_idx = k_start; k_idx < k_end; k_idx = k_idx + 1u) {
            if varlen_masked(k_idx - batch_start_k, q_pos, params.causal, params.window) {
                continue;
            }
            let k_base = (k_idx * params.num_heads + h) * params.head_dim;
            var score = 0.0f;
            for (var d = 0u; d < params.head_dim; d = d + 1u) {
                score += q[q_base + d] * k[k_base + d];
            }
            score *= params.scale;

            let weight = exp(score - max_score);
            sum_exp += weight;

            let v_base = (k_idx * params.num_heads + h) * params.head_dim;
            for (var d = 0u; d < params.head_dim; d = d + 1u) {
                accum[d] += weight * v[v_base + d];
            }
        }

        // Write output
        let out_base = (token_idx * params.num_heads + h) * params.head_dim;
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            out[out_base + d] = accum[d] / max(sum_exp, 1e-10f);
        }

        // Store LSE
        let lse_idx = token_idx * params.num_heads + h;
        lse[lse_idx] = log(max(sum_exp, 1e-10f)) + max_score;
    }
}

@compute @workgroup_size(256)
fn varlen_attention_bwd_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Backward pass placeholder
}
