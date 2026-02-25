// Fused SSD state passing shader for Mamba2
//
// Sequential scan across chunks:
//   states_out[b, c, h, d, n] = states[b, c, h, d, n] + scale[b, h, c] * states_out[b, c-1, h, d, n]
//
// Each invocation handles one (batch, head, headdim_idx, dstate_idx) position
// and loops over chunks sequentially.

struct SsdStatePassingParams {
    batch: u32,
    nchunks: u32,
    nheads: u32,
    headdim: u32,
    dstate: u32,
    chunk_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> states: array<f32>;
@group(0) @binding(1) var<storage, read> dA_cumsum: array<f32>;
@group(0) @binding(2) var<storage, read_write> states_out: array<f32>;
@group(0) @binding(3) var<uniform> params: SsdStatePassingParams;

@compute @workgroup_size(256)
fn ssd_state_passing_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.batch * params.nheads * params.headdim * params.dstate;
    if (idx >= total) {
        return;
    }

    // Decode idx to (b, h, d_idx, n_idx)
    let n_idx = idx % params.dstate;
    var rem = idx / params.dstate;
    let d_idx = rem % params.headdim;
    rem = rem / params.headdim;
    let h = rem % params.nheads;
    let b = rem / params.nheads;

    // states layout: [batch, nchunks, nheads, headdim, dstate]
    let s_chunk_stride = params.nheads * params.headdim * params.dstate;
    let s_batch_stride = params.nchunks * s_chunk_stride;
    let s_base = b * s_batch_stride + h * params.headdim * params.dstate + d_idx * params.dstate + n_idx;

    // dA_cumsum layout: [batch, nheads, nchunks, chunk_size]
    let da_chunk_stride = params.chunk_size;
    let da_head_stride = params.nchunks * params.chunk_size;
    let da_batch_stride = params.nheads * da_head_stride;
    let da_base = b * da_batch_stride + h * da_head_stride + (params.chunk_size - 1u);

    // Copy first chunk
    var prev = states[s_base];
    states_out[s_base] = prev;

    // Sequential scan
    for (var c = 1u; c < params.nchunks; c = c + 1u) {
        let dA_val = dA_cumsum[da_base + c * da_chunk_stride];
        let scale = exp(min(dA_val, 0.0));

        let s_offset = s_base + c * s_chunk_stride;
        let curr = states[s_offset];
        prev = curr + scale * prev;
        states_out[s_offset] = prev;
    }
}
