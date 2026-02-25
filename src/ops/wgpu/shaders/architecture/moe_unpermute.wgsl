// MoE Token Unpermutation Shader (F32)
// Parallel gather from expert output back to original order.
// One thread per output element (position * hidden_dim).

struct MoEUnpermuteParams {
    total: u32,
    hidden_dim: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> expert_output: array<f32>;
@group(0) @binding(1) var<storage, read> inv_perm: array<i32>;
@group(0) @binding(2) var<storage, read_write> unsorted: array<f32>;
@group(0) @binding(3) var<uniform> params: MoEUnpermuteParams;

@compute @workgroup_size(256)
fn moe_unpermute_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_elements = params.total * params.hidden_dim;
    if (idx >= total_elements) {
        return;
    }

    let out_pos = idx / params.hidden_dim;
    let dim = idx % params.hidden_dim;

    let src_pos = u32(inv_perm[out_pos]);
    unsorted[out_pos * params.hidden_dim + dim] = expert_output[src_pos * params.hidden_dim + dim];
}
