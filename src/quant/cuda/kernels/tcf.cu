// CUDA kernels for TCF native quantized weights: dequantization, GEMV, GEMM.
//
// # The access pattern, and why plane-major is not the problem it looks like
//
// TCF packs whole planes over the whole tensor: every tile's codes, then every
// group's scales, then minima, then the per-super-block values. A group's
// parameters therefore live in a different region from its codes, and a naive
// per-element kernel would issue two to five scattered global loads per weight.
//
// Every kernel below avoids that the same way, in two phases per unit of work:
//
//   1. Resolve group parameters ONCE per (tile, group) — at most four per tile
//      — into registers or shared memory. That is one load per 16, 32, or 64
//      weights, not one per weight, and the loads a warp issues in this phase
//      are consecutive entries of one plane.
//   2. Stream the code plane. A tile's codes are one contiguous run, and
//      consecutive tiles are adjacent, so a whole weight row's codes are a
//      single contiguous byte range. Lanes read consecutive addresses and the
//      loads coalesce.
//
// The code plane is in fact FRIENDLIER than a GGUF block layout here: GGUF
// interleaves a 2-byte scale into the code stream every 16 or 32 weights, so a
// warp reading 32 consecutive codes straddles scale bytes and lands on
// unaligned offsets. TCF's code plane is a dense array with natural alignment.
// What plane-major costs is concurrent read STREAMS — two for a flat symmetric
// encoding, up to five for the two-level asymmetric one — not coalescing.
//
// Coalescing is necessary and was not sufficient. The GEMV's first form read
// consecutive bytes and still reached a fifth of this card's bandwidth,
// because coalesced 32-byte warp loads and a scale resolution every 64
// elements both cost per ELEMENT. Both were fixed by widening the unit of
// work, not by reordering the payload; see the GEMV section for the numbers.
//
// CONFORMANCE.md Section 8.1 leaves code-plane ordering unfrozen, to be settled
// by benchmark. These kernels are one input to that: they show plane-major
// imposes no structural penalty on a GPU decode, only extra address streams.
//
// The 6-bit encodings are the one exception worth naming. Section 14.2 splits
// their code plane into a low-nibble sub-plane and a high-two-bit sub-plane, so
// a 6-bit tile costs two code streams instead of one. Interleaving those two
// sub-planes per tile would halve that, and is the concrete change Section 8.1
// would be measuring.

#include "tcf.cuh"

#define TCF_WARP_SIZE 32u
#define TCF_WARPS_PER_BLOCK 8u
#define TCF_GEMV_BLOCK (TCF_WARP_SIZE * TCF_WARPS_PER_BLOCK)

// Tiles one dequantization block owns: one super-block, so a block's group
// parameters come from one bit-packed sub-plane run.
#define TCF_DEQUANT_TILES TCF_SUPER_BLOCK_TILES
#define TCF_DEQUANT_BLOCK (TCF_DEQUANT_TILES * TCF_TILE)

// GEMM output tile: 16 activation rows by 16 weight rows, 256 threads.
#define TCF_GEMM_TM 16u
#define TCF_GEMM_TN 16u
// Row stride of a staged 64-element run, padded by one float so 16 lanes
// reading the same element index of 16 different rows hit 16 distinct banks.
#define TCF_GEMM_STRIDE 65u
// Largest group count a tile can carry (group 16 over a 64-element tile).
#define TCF_MAX_GROUPS 4u

static __device__ __forceinline__ float tcf_warp_reduce_sum(float acc) {
#pragma unroll
    for (unsigned int offset = TCF_WARP_SIZE / 2u; offset > 0u; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }
    return acc;
}

// The whole-run part of one warp's GEMV dot product over weight row `col`:
// `runs` steps of `TCF_RUN_TILES` tiles, sixteen elements per lane per step.
//
// `ActWide` is the caller's alignment finding, not a hint. It is a template
// parameter rather than a branch inside the loop because the branch cost 10%
// of the kernel; the two instantiations differ only in how four activation
// floats are fetched. Both are warp-uniform, so every lane reaches the
// `__shfl_sync` below and the full-mask shuffles are legal.
template <bool ActWide>
static __device__ __forceinline__ float tcf_gemv_runs(
    const float* __restrict__ act_row,
    const unsigned char* __restrict__ weight,
    TcfLayout l,
    unsigned int col,
    unsigned int tiles_per_row,
    unsigned int runs,
    unsigned int lane
) {
    const unsigned int resolved = TCF_RUN_TILES * l.groups_per_tile;
    const unsigned int e0 = TCF_RUN_PER_LANE * lane;
    // Sixteen elements from a multiple of sixteen lie in one group at every
    // group width v1 defines, so one broadcast serves the lane's whole run.
    const unsigned int slot = e0 / l.group;
    float acc = 0.0f;

    for (unsigned int r = 0; r < runs; ++r) {
        const unsigned int tile0 = col * tiles_per_row + r * TCF_RUN_TILES;
        float scale = 0.0f;
        float min_value = 0.0f;
        if (lane < resolved) {
            tcf_group_values(weight, l, tile0 + lane / l.groups_per_tile,
                             lane % l.groups_per_tile, &scale, &min_value);
        }
        const float s = __shfl_sync(0xFFFFFFFFu, scale, slot);
        const float mn = __shfl_sync(0xFFFFFFFFu, min_value, slot);
        const TcfCodeRun run = tcf_code_run(weight, l, tile0, lane);
        const float* act = act_row + (size_t)r * (size_t)TCF_RUN + (size_t)e0;

#pragma unroll
        for (unsigned int c = 0; c < TCF_RUN_PER_LANE / 4u; ++c) {
            float a[4];
            if (ActWide) {
                const float4 v = *(const float4*)(act + c * 4u);
                a[0] = v.x;
                a[1] = v.y;
                a[2] = v.z;
                a[3] = v.w;
            } else {
#pragma unroll
                for (unsigned int k = 0; k < 4u; ++k) {
                    a[k] = act[c * 4u + k];
                }
            }
#pragma unroll
            for (unsigned int k = 0; k < 4u; ++k) {
                const int code = tcf_run_code(run, l, c * 4u + k);
                acc += a[k] * tcf_value(code, s, mn, l.symmetric);
            }
        }
    }
    return acc;
}

extern "C" {

// ============================================================================
// Dequantization: whole payload -> f32, in logical row-major order.
//
// One block per super-block of four tiles, 256 threads, one element each.
// Threads 0..(4 * groups_per_tile) resolve the block's group parameters into
// shared memory first, so the scale planes are read once per group rather than
// once per weight.
// ============================================================================

__global__ __launch_bounds__(TCF_DEQUANT_BLOCK, 1) void tcf_dequant_f32(
    const unsigned char* __restrict__ payload,
    float* __restrict__ output,
    unsigned int tiles,
    unsigned long long code_high_off,
    unsigned long long scale_off,
    unsigned long long min_off,
    unsigned long long super_off,
    unsigned long long super_min_off,
    unsigned int bits,
    unsigned int group,
    unsigned int groups_per_tile,
    unsigned int symmetric,
    unsigned int scale_form,
    unsigned int sub_block_bytes
) {
    const TcfLayout l = tcf_layout(code_high_off, scale_off, min_off, super_off,
                                   super_min_off, bits, group, groups_per_tile,
                                   symmetric, scale_form, sub_block_bytes);

    __shared__ float s_scale[TCF_DEQUANT_TILES * TCF_MAX_GROUPS];
    __shared__ float s_min[TCF_DEQUANT_TILES * TCF_MAX_GROUPS];

    const unsigned int tile_base = blockIdx.x * TCF_DEQUANT_TILES;
    const unsigned int tid = threadIdx.x;
    const unsigned int resolved = TCF_DEQUANT_TILES * l.groups_per_tile;

    if (tid < resolved) {
        const unsigned int local_tile = tid / l.groups_per_tile;
        const unsigned int g = tid % l.groups_per_tile;
        const unsigned int tile = tile_base + local_tile;
        float scale = 0.0f;
        float min_value = 0.0f;
        if (tile < tiles) {
            tcf_group_values(payload, l, tile, g, &scale, &min_value);
        }
        s_scale[tid] = scale;
        s_min[tid] = min_value;
    }
    __syncthreads();

    const unsigned int local_tile = tid / TCF_TILE;
    const unsigned int e = tid % TCF_TILE;
    const unsigned int tile = tile_base + local_tile;
    if (tile >= tiles) {
        return;
    }

    const unsigned int slot = local_tile * l.groups_per_tile + (e / l.group);
    const int code = tcf_code(payload, l, tile, e);
    output[(size_t)tile * (size_t)TCF_TILE + (size_t)e] =
        tcf_value(code, s_scale[slot], s_min[slot], l.symmetric);
}

// ============================================================================
// GEMV: activation [M, K] x weight [N, K]^T -> output [M, N], for a small M.
//
// One warp per output column, eight warps per block. A warp owns weight row
// `col`, whose tiles are `col * tiles_per_row ..` — a contiguous run of the
// code plane — and walks it EIGHT TILES AT A TIME, `TCF_RUN_TILES`.
//
// # Why eight tiles and not one
//
// A tile-at-a-time loop gave 20% of this card's bandwidth and got SLOWER as
// the encoding got narrower, which is the signature of a cost tracking element
// count rather than byte count. Two causes, measured separately on an RTX 3060
// with `q_proj` 2048x2048 at M = 1:
//
//   - Narrow code loads. `tcf_code` issues one byte load per element, so a
//     warp covered 32 bytes per instruction at 8 bits and 16 at 4 bits.
//     `tcf_code_run` reads the same codes as one `uint4` or `uint2` per lane:
//     512 or 256 consecutive bytes per instruction.
//   - Scale resolution once per TILE. Holding scale and minimum constant cut
//     the old kernel's time roughly in half at every width, and the whole of
//     the 4-bit deficit was here: `TwoLevelU6M6` costs four scattered plane
//     reads, and paying them every 64 elements dominated everything else.
//     A run pays them every 512 elements instead, and resolves the run's
//     `8 * groups_per_tile <= 32` groups in ONE warp-wide step.
//
// Together: 62.3 -> 19.5 us at 8 bits, 72.9 -> 21.5 at 6, 86.9 -> 21.5 at 4.
// The 8-bit case then sits within 4 us of a kernel that only streams the code
// plane and discards it, so it is at the memory system's limit.
//
// Sixteen elements per lane is one whole quantization group at every group
// width v1 defines, so a lane broadcasts one `(scale, minimum)` pair for its
// whole run. See `tcf_code_run` for that and for the alignment argument.
//
// A row whose tile count is not a multiple of eight finishes on the
// tile-at-a-time path below, which is also the only path when `K` is under
// eight tiles. Nothing here needs super-block alignment: the kernel addresses
// each tile by its global index, which is what a super-block's sub-plane is
// keyed on.
// ============================================================================

__global__ __launch_bounds__(TCF_GEMV_BLOCK, 1) void tcf_gemv_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M,
    unsigned int K,
    unsigned int N,
    unsigned long long code_high_off,
    unsigned long long scale_off,
    unsigned long long min_off,
    unsigned long long super_off,
    unsigned long long super_min_off,
    unsigned int bits,
    unsigned int group,
    unsigned int groups_per_tile,
    unsigned int symmetric,
    unsigned int scale_form,
    unsigned int sub_block_bytes
) {
    const TcfLayout l = tcf_layout(code_high_off, scale_off, min_off, super_off,
                                   super_min_off, bits, group, groups_per_tile,
                                   symmetric, scale_form, sub_block_bytes);

    const unsigned int warp_id = threadIdx.x / TCF_WARP_SIZE;
    const unsigned int lane = threadIdx.x % TCF_WARP_SIZE;
    const unsigned int col = blockIdx.x * TCF_WARPS_PER_BLOCK + warp_id;
    const unsigned int m = blockIdx.y;
    if (col >= N || m >= M) {
        return;
    }

    const unsigned int tiles_per_row = K / TCF_TILE;
    const float* act_row = activation + (size_t)m * (size_t)K;
    const unsigned int runs = tiles_per_row / TCF_RUN_TILES;
    // A `float4` activation load needs a 16-byte-aligned row. The activation
    // pointer is a tensor VIEW's, so it carries that view's element offset and
    // is only guaranteed 4-byte aligned. Checked, not assumed: a misaligned
    // wide load is undefined behaviour, not a slow one. `K` is a whole number
    // of 64-element tiles, so one row being aligned makes every row aligned,
    // and the choice is uniform across the warp.
    const bool act_wide = ((size_t)act_row & 15u) == 0u;
    float acc = act_wide
        ? tcf_gemv_runs<true>(act_row, weight, l, col, tiles_per_row, runs, lane)
        : tcf_gemv_runs<false>(act_row, weight, l, col, tiles_per_row, runs, lane);

    // The tiles a whole run cannot cover, one at a time.
    for (unsigned int j = runs * TCF_RUN_TILES; j < tiles_per_row; ++j) {
        const unsigned int tile = col * tiles_per_row + j;
        float scale = 0.0f;
        float min_value = 0.0f;
        if (lane < l.groups_per_tile) {
            tcf_group_values(weight, l, tile, lane, &scale, &min_value);
        }
        const unsigned int base = j * TCF_TILE;
#pragma unroll
        for (unsigned int half = 0; half < 2u; ++half) {
            const unsigned int e = lane + half * TCF_WARP_SIZE;
            const unsigned int g = e / l.group;
            const float gs = __shfl_sync(0xFFFFFFFFu, scale, g);
            const float gm = __shfl_sync(0xFFFFFFFFu, min_value, g);
            const int code = tcf_code(weight, l, tile, e);
            acc += act_row[base + e] * tcf_value(code, gs, gm, l.symmetric);
        }
    }

    acc = tcf_warp_reduce_sum(acc);
    if (lane == 0) {
        output[(size_t)m * (size_t)N + (size_t)col] = acc;
    }
}

// ============================================================================
// GEMM: activation [M, K] x weight [N, K]^T -> output [M, N], for M > 64.
//
// A 16x16 output tile per block, 256 threads, one output element each. Per
// K-tile the block decodes its 16 weight rows into shared memory ONCE and
// stages the matching 16 activation rows beside them, so a weight element is
// decoded once per block instead of once per output element. The naive
// thread-per-output form would decode every weight 16 times, which a TCF
// decode — two plane reads plus a two-level product — cannot afford.
// ============================================================================

__global__ __launch_bounds__(TCF_GEMM_TM * TCF_GEMM_TN, 1) void tcf_gemm_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M,
    unsigned int K,
    unsigned int N,
    unsigned long long code_high_off,
    unsigned long long scale_off,
    unsigned long long min_off,
    unsigned long long super_off,
    unsigned long long super_min_off,
    unsigned int bits,
    unsigned int group,
    unsigned int groups_per_tile,
    unsigned int symmetric,
    unsigned int scale_form,
    unsigned int sub_block_bytes
) {
    const TcfLayout l = tcf_layout(code_high_off, scale_off, min_off, super_off,
                                   super_min_off, bits, group, groups_per_tile,
                                   symmetric, scale_form, sub_block_bytes);

    __shared__ float s_act[TCF_GEMM_TM * TCF_GEMM_STRIDE];
    __shared__ float s_weight[TCF_GEMM_TN * TCF_GEMM_STRIDE];
    __shared__ float s_scale[TCF_GEMM_TN * TCF_MAX_GROUPS];
    __shared__ float s_min[TCF_GEMM_TN * TCF_MAX_GROUPS];

    const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int threads = TCF_GEMM_TM * TCF_GEMM_TN;
    const unsigned int row0 = blockIdx.y * TCF_GEMM_TM;
    const unsigned int col0 = blockIdx.x * TCF_GEMM_TN;
    const unsigned int row = row0 + threadIdx.y;
    const unsigned int col = col0 + threadIdx.x;

    const unsigned int tiles_per_row = K / TCF_TILE;
    const unsigned int resolved = TCF_GEMM_TN * l.groups_per_tile;
    float acc = 0.0f;

    for (unsigned int j = 0; j < tiles_per_row; ++j) {
        if (tid < resolved) {
            const unsigned int r = tid / l.groups_per_tile;
            const unsigned int g = tid % l.groups_per_tile;
            float scale = 0.0f;
            float min_value = 0.0f;
            if (col0 + r < N) {
                tcf_group_values(weight, l, (col0 + r) * tiles_per_row + j, g,
                                 &scale, &min_value);
            }
            s_scale[tid] = scale;
            s_min[tid] = min_value;
        }
        __syncthreads();

        // 16 rows x 64 elements = 1024 values per stage, four per thread.
#pragma unroll
        for (unsigned int i = 0; i < 4u; ++i) {
            const unsigned int index = tid + i * threads;
            const unsigned int r = index / TCF_TILE;
            const unsigned int e = index % TCF_TILE;

            float w = 0.0f;
            if (col0 + r < N) {
                const unsigned int tile = (col0 + r) * tiles_per_row + j;
                const unsigned int slot = r * l.groups_per_tile + (e / l.group);
                w = tcf_value(tcf_code(weight, l, tile, e), s_scale[slot],
                              s_min[slot], l.symmetric);
            }
            s_weight[r * TCF_GEMM_STRIDE + e] = w;

            float a = 0.0f;
            if (row0 + r < M) {
                a = activation[(size_t)(row0 + r) * (size_t)K + (size_t)(j * TCF_TILE + e)];
            }
            s_act[r * TCF_GEMM_STRIDE + e] = a;
        }
        __syncthreads();

#pragma unroll 8
        for (unsigned int e = 0; e < TCF_TILE; ++e) {
            acc += s_act[threadIdx.y * TCF_GEMM_STRIDE + e]
                 * s_weight[threadIdx.x * TCF_GEMM_STRIDE + e];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        output[(size_t)row * (size_t)N + (size_t)col] = acc;
    }
}

}  // extern "C"
