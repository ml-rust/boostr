//! Paged Attention - vLLM style block-table indirection
//! Q [B, num_heads, S_q, head_dim]
//! K_blocks [num_blocks, block_size, head_dim]
//! V_blocks [num_blocks, block_size, head_dim]
//! block_table [B, max_num_blocks] (i32)
//!
//! Maps logical token position to physical block via block_table.
//! Token t in sequence is in block t/block_size, offset t%block_size.

struct PagedParams {
    batch_size: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_len_q: u32,
    seq_len_k: u32,
    head_dim: u32,
    block_size: u32,
    max_num_blocks: u32,
    scale: f32,
    causal: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_blocks: array<f32>;
@group(0) @binding(2) var<storage, read> v_blocks: array<f32>;
@group(0) @binding(3) var<storage, read> block_table: array<i32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read_write> lse: array<f32>;
@group(0) @binding(6) var<uniform> params: PagedParams;

@compute @workgroup_size(256)
fn paged_attention_fwd_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_idx = gid.x;
    let total_queries = params.batch_size * params.num_heads * params.seq_len_q;

    if query_idx >= total_queries {
        return;
    }

    // Decode query position
    let i = query_idx % params.seq_len_q;
    let remainder = query_idx / params.seq_len_q;
    let h = remainder % params.num_heads;
    let b = remainder / params.num_heads;

    // GQA: map query head to kv head
    let h_kv = (h * params.num_kv_heads) / params.num_heads;

    // Read Q[b, h, i, :]
    let q_base = ((b * params.num_heads + h) * params.seq_len_q + i) * params.head_dim;

    var accum: array<f32, 512>;  // Max head_dim
    var max_score = -1e30f;
    var sum_exp = 0.0f;

    // Compute valid key range
    var end_j = params.seq_len_k;
    if params.causal != 0u {
        end_j = i + 1u;
    }

    // First pass: find max score
    for (var j = 0u; j < end_j; j = j + 1u) {
        // Map logical position j to physical position via block table
        let block_idx_logical = j / params.block_size;
        let offset_in_block = j % params.block_size;

        let bt_idx = b * params.max_num_blocks + block_idx_logical;
        if bt_idx >= arrayLength(&block_table) {
            break;
        }
        let block_idx_physical = u32(block_table[bt_idx]);

        // K[block_idx_physical, offset_in_block, h_kv, :]
        let k_base = ((block_idx_physical * params.block_size + offset_in_block) * params.num_kv_heads + h_kv) * params.head_dim;

        if k_base + params.head_dim > arrayLength(&k_blocks) {
            break;
        }

        var score = 0.0f;
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            score += q[q_base + d] * k_blocks[k_base + d];
        }
        score *= params.scale;
        max_score = max(max_score, score);
    }

    if max_score == -1e30f {
        max_score = 0.0f;
    }

    // Second pass: softmax and aggregate
    for (var j = 0u; j < end_j; j = j + 1u) {
        let block_idx_logical = j / params.block_size;
        let offset_in_block = j % params.block_size;

        let bt_idx = b * params.max_num_blocks + block_idx_logical;
        if bt_idx >= arrayLength(&block_table) {
            break;
        }
        let block_idx_physical = u32(block_table[bt_idx]);

        let k_base = ((block_idx_physical * params.block_size + offset_in_block) * params.num_kv_heads + h_kv) * params.head_dim;
        if k_base + params.head_dim > arrayLength(&k_blocks) {
            break;
        }

        var score = 0.0f;
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            score += q[q_base + d] * k_blocks[k_base + d];
        }
        score *= params.scale;

        let weight = exp(score - max_score);
        sum_exp += weight;

        // V[block_idx_physical, offset_in_block, h_kv, :]
        let v_base = ((block_idx_physical * params.block_size + offset_in_block) * params.num_kv_heads + h_kv) * params.head_dim;
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            accum[d] += weight * v_blocks[v_base + d];
        }
    }

    // Write output
    let out_base = ((b * params.num_heads + h) * params.seq_len_q + i) * params.head_dim;
    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        out[out_base + d] = accum[d] / max(sum_exp, 1e-10f);
    }

    let lse_idx = (b * params.num_heads + h) * params.seq_len_q + i;
    lse[lse_idx] = log(max(sum_exp, 1e-10f)) + max_score;
}

@compute @workgroup_size(256)
fn paged_attention_bwd_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Backward pass placeholder
}
