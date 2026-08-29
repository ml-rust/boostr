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
// GEMV: activation [M, K] x weight [N, K]^T -> output [M, N], for M <= 64.
//
// One warp per output column, eight warps per block. A warp owns weight row
// `col`, whose tiles are `col * tiles_per_row ..` — a contiguous run of the
// code plane. Each lane takes elements `lane` and `lane + 32` of every tile,
// so the warp's loads cover consecutive bytes.
//
// Group parameters are resolved by the warp's first `groups_per_tile` lanes and
// broadcast with a shuffle, so the scale planes are touched once per tile
// rather than once per lane.
//
// Unlike the CPU fused kernel, no tile range needs super-block alignment: the
// kernel addresses each tile by its global index, which is what a super-block's
// sub-plane is keyed on.
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
    float acc = 0.0f;

    for (unsigned int j = 0; j < tiles_per_row; ++j) {
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
            const float s = __shfl_sync(0xFFFFFFFFu, scale, g);
            const float mn = __shfl_sync(0xFFFFFFFFu, min_value, g);
            const int code = tcf_code(weight, l, tile, e);
            acc += act_row[base + e] * tcf_value(code, s, mn, l.symmetric);
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
