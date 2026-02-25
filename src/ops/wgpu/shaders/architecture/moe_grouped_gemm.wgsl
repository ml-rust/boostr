// MoE Grouped GEMM Shader (F32)
// Tiled matmul per expert with 16x16 tiles.
// Expert index from workgroup_id.z.
// Grid: (ceil(out_dim/16), ceil(max_tokens/16), num_experts)

struct MoEGemmParams {
    in_dim: u32,
    out_dim: u32,
    num_experts: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> tokens: array<f32>;
@group(0) @binding(1) var<storage, read> expert_weights: array<f32>;
@group(0) @binding(2) var<storage, read> offsets: array<i32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: MoEGemmParams;

const TILE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

// All threads must call this — no early returns before workgroupBarrier.
fn gemm_core(expert_idx: u32, local_row: u32, local_col: u32, tile_row: u32, tile_col: u32) -> f32 {
    let start = u32(offsets[expert_idx]);
    let end = u32(offsets[expert_idx + 1u]);
    let count = end - start;

    let row = tile_row + local_row;
    let col = tile_col + local_col;

    // Use start as fallback row to avoid OOB reads (result discarded for invalid threads)
    let global_row = select(start, start + row, row < count);
    let in_dim = params.in_dim;
    let out_dim = params.out_dim;

    var acc: f32 = 0.0;

    for (var k_start = 0u; k_start < in_dim; k_start = k_start + TILE) {
        // Load tile from tokens
        let k_idx = k_start + local_col;
        if (row < count && k_idx < in_dim) {
            tile_a[local_row][local_col] = tokens[global_row * in_dim + k_idx];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }

        // Load tile from weights
        let k_idx2 = k_start + local_row;
        if (k_idx2 < in_dim && col < out_dim) {
            tile_b[local_row][local_col] = expert_weights[expert_idx * in_dim * out_dim + k_idx2 * out_dim + col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        let k_end = min(TILE, in_dim - k_start);
        for (var kk = 0u; kk < k_end; kk = kk + 1u) {
            acc = acc + tile_a[local_row][kk] * tile_b[kk][local_col];
        }

        workgroupBarrier();
    }

    return acc;
}

@compute @workgroup_size(16, 16, 1)
fn moe_grouped_gemm_f32(@builtin(workgroup_id) wid: vec3<u32>,
                         @builtin(local_invocation_id) lid: vec3<u32>) {
    let expert_idx = wid.z;
    if (expert_idx >= params.num_experts) {
        return;
    }

    let tile_row = wid.y * TILE;
    let tile_col = wid.x * TILE;

    let acc = gemm_core(expert_idx, lid.y, lid.x, tile_row, tile_col);

    let row = tile_row + lid.y;
    let col = tile_col + lid.x;
    let start = u32(offsets[expert_idx]);
    let count = u32(offsets[expert_idx + 1u]) - start;

    if (row < count && col < params.out_dim) {
        let global_row = start + row;
        output[global_row * params.out_dim + col] = acc;
    }
}

@compute @workgroup_size(16, 16, 1)
fn moe_grouped_gemm_silu_f32(@builtin(workgroup_id) wid: vec3<u32>,
                              @builtin(local_invocation_id) lid: vec3<u32>) {
    let expert_idx = wid.z;
    if (expert_idx >= params.num_experts) {
        return;
    }

    let tile_row = wid.y * TILE;
    let tile_col = wid.x * TILE;

    let acc = gemm_core(expert_idx, lid.y, lid.x, tile_row, tile_col);

    let row = tile_row + lid.y;
    let col = tile_col + lid.x;
    let start = u32(offsets[expert_idx]);
    let count = u32(offsets[expert_idx + 1u]) - start;

    if (row < count && col < params.out_dim) {
        let sigmoid = 1.0 / (1.0 + exp(-acc));
        let global_row = start + row;
        output[global_row * params.out_dim + col] = acc * sigmoid;
    }
}

@compute @workgroup_size(16, 16, 1)
fn moe_grouped_gemm_gelu_f32(@builtin(workgroup_id) wid: vec3<u32>,
                              @builtin(local_invocation_id) lid: vec3<u32>) {
    let expert_idx = wid.z;
    if (expert_idx >= params.num_experts) {
        return;
    }

    let tile_row = wid.y * TILE;
    let tile_col = wid.x * TILE;

    let acc = gemm_core(expert_idx, lid.y, lid.x, tile_row, tile_col);

    let row = tile_row + lid.y;
    let col = tile_col + lid.x;
    let start = u32(offsets[expert_idx]);
    let count = u32(offsets[expert_idx + 1u]) - start;

    if (row < count && col < params.out_dim) {
        let x3 = acc * acc * acc;
        let inner = 0.7978845608 * (acc + 0.044715 * x3);
        let global_row = start + row;
        output[global_row * params.out_dim + col] = 0.5 * acc * (1.0 + tanh(inner));
    }
}
