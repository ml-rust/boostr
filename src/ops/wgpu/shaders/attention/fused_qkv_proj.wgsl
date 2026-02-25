// Fused QKV bias + split + reshape shader (F32)
//
// Takes the output of a matmul (qkv [B*S, total_proj]) and applies:
// 1. Optional bias addition
// 2. Split into Q, K, V regions
// 3. Reshape + transpose to [B, heads, S, D] layout
//
// Each workgroup thread handles one element of the qkv output.

struct Params {
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    total_proj: u32,
    has_bias: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> k_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> v_out: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn fused_qkv_bias_split_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.batch_size * params.seq_len * params.total_proj;
    if (idx >= total) {
        return;
    }

    let hq = params.num_heads * params.head_dim;
    let hkv = params.num_kv_heads * params.head_dim;

    let proj_idx = idx % params.total_proj;
    let batch_seq_idx = idx / params.total_proj;
    let b = batch_seq_idx / params.seq_len;
    let s = batch_seq_idx % params.seq_len;

    var val = qkv[idx];
    if (params.has_bias != 0u) {
        val = val + bias[proj_idx];
    }

    if (proj_idx < hq) {
        // Q region
        let h = proj_idx / params.head_dim;
        let d = proj_idx % params.head_dim;
        let out_idx = ((b * params.num_heads + h) * params.seq_len + s) * params.head_dim + d;
        q_out[out_idx] = val;
    } else if (proj_idx < hq + hkv) {
        // K region
        let local_idx = proj_idx - hq;
        let h = local_idx / params.head_dim;
        let d = local_idx % params.head_dim;
        let out_idx = ((b * params.num_kv_heads + h) * params.seq_len + s) * params.head_dim + d;
        k_out[out_idx] = val;
    } else {
        // V region
        let local_idx = proj_idx - hq - hkv;
        let h = local_idx / params.head_dim;
        let d = local_idx % params.head_dim;
        let out_idx = ((b * params.num_kv_heads + h) * params.seq_len + s) * params.head_dim + d;
        v_out[out_idx] = val;
    }
}
