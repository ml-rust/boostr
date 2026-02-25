//! Reshape and cache for paged KV storage
//! Maps tokens to physical blocks via slot_mapping.

struct PagedCacheParams {
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    block_size: u32,
}

// Read-only first (bindings 0-2), then read-write (bindings 3-4)
@group(0) @binding(0) var<storage, read> key: array<f32>;
@group(0) @binding(1) var<storage, read> value: array<f32>;
@group(0) @binding(2) var<storage, read> slot_mapping: array<i32>;
@group(0) @binding(3) var<storage, read_write> key_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> value_cache: array<f32>;
@group(0) @binding(5) var<uniform> paged_params: PagedCacheParams;

@compute @workgroup_size(256)
fn reshape_and_cache_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let elem_idx = gid.x;
    let total_elems = paged_params.num_tokens * paged_params.num_heads * paged_params.head_dim;

    if elem_idx >= total_elems {
        return;
    }

    let d = elem_idx % paged_params.head_dim;
    let remainder = elem_idx / paged_params.head_dim;
    let h = remainder % paged_params.num_heads;
    let t = remainder / paged_params.num_heads;

    let slot = u32(slot_mapping[t]);
    let block_idx = slot / paged_params.block_size;
    let offset_in_block = slot % paged_params.block_size;

    let key_idx = (t * paged_params.num_heads + h) * paged_params.head_dim + d;
    let value_idx = (t * paged_params.num_heads + h) * paged_params.head_dim + d;

    let cache_idx = ((block_idx * paged_params.block_size + offset_in_block) * paged_params.num_heads + h) * paged_params.head_dim + d;

    key_cache[cache_idx] = key[key_idx];
    value_cache[cache_idx] = value[value_idx];
}
