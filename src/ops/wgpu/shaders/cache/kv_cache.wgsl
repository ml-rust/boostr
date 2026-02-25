//! KV Cache update
//! Writes new K/V into contiguous cache at given position.

struct KvCacheParams {
    batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    new_len: u32,
    max_seq_len: u32,
    position: u32,
    _pad1: u32,
    _pad2: u32,
}

// Read-only first (bindings 0-1), then read-write (bindings 2-3)
@group(0) @binding(0) var<storage, read> new_k: array<f32>;
@group(0) @binding(1) var<storage, read> new_v: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(4) var<uniform> params: KvCacheParams;

@compute @workgroup_size(256)
fn kv_cache_update_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let elem_idx = gid.x;
    let total_elems = params.batch_size * params.num_heads * params.new_len * params.head_dim;

    if elem_idx >= total_elems {
        return;
    }

    let d = elem_idx % params.head_dim;
    let remainder = elem_idx / params.head_dim;
    let t_local = remainder % params.new_len;
    let remainder2 = remainder / params.new_len;
    let h = remainder2 % params.num_heads;
    let b = remainder2 / params.num_heads;

    let t_cache = params.position + t_local;

    let new_k_idx = ((b * params.num_heads + h) * params.new_len + t_local) * params.head_dim + d;
    let k_cache_idx = ((b * params.num_heads + h) * params.max_seq_len + t_cache) * params.head_dim + d;
    k_cache[k_cache_idx] = new_k[new_k_idx];

    let new_v_idx = ((b * params.num_heads + h) * params.new_len + t_local) * params.head_dim + d;
    let v_cache_idx = ((b * params.num_heads + h) * params.max_seq_len + t_cache) * params.head_dim + d;
    v_cache[v_cache_idx] = new_v[new_v_idx];
}
