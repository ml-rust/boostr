// MoE Token Permutation Shader (F32)
// Parallel scatter of tokens into expert-grouped order.
// One thread per output element (position * hidden_dim).

struct MoEPermuteParams {
    total: u32,
    hidden_dim: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> tokens: array<f32>;
@group(0) @binding(1) var<storage, read> src_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> permuted: array<f32>;
@group(0) @binding(3) var<uniform> params: MoEPermuteParams;

@compute @workgroup_size(256)
fn moe_permute_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_elements = params.total * params.hidden_dim;
    if (idx >= total_elements) {
        return;
    }

    let out_pos = idx / params.hidden_dim;
    let dim = idx % params.hidden_dim;

    let src_token = u32(src_indices[out_pos]);
    permuted[out_pos * params.hidden_dim + dim] = tokens[src_token * params.hidden_dim + dim];
}
