// Shared helpers for quantized GEMV kernels
//
// Included by all per-format GEMV .cu files. Contains:
//   - dp4a intrinsic (with fallback for pre-Pascal)
//   - Unaligned loads
//   - MWR constants
//   - Scale unpacking helpers
//   - SiLU activation
//   - In-register F32→Q8_1 quantization helpers

#pragma once

#include <cuda_fp16.h>

#include "../decode.cuh"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE_256 (WARP_SIZE * WARPS_PER_BLOCK)
#define NWARPS_K 4

// ── dp4a intrinsic ──────────────────────────────────────────────────────
// 4-element int8 dot product in a single instruction (compute >= 6.1)

static __device__ __forceinline__ int load_int_ua(const unsigned char* p) {
    const unsigned short* p16 = (const unsigned short*)p;
    return (int)p16[0] | ((int)p16[1] << 16);
}

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const signed char* a8 = (const signed char*)&a;
    const signed char* b8 = (const signed char*)&b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

// ── SiLU activation ─────────────────────────────────────────────────────

static __device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

// Q4_K/Q5_K/Q3_K scale unpacking (unpack_q4k_q5k_scales, unpack_scales_mwr,
// unpack_q3k_scales) now lives in decode.cuh, included above.

// ── MWR reduction template ──────────────────────────────────────────────
// Used by all MWR kernels: shared memory reduction across NWARPS_K warps

static __device__ __forceinline__ float mwr_reduce(
    float acc, int warp_id, int lane_id,
    float smem[NWARPS_K][WARP_SIZE]
) {
    smem[warp_id][lane_id] = acc;
    __syncthreads();

    if (warp_id != 0) return 0.0f;

    float sum = smem[0][lane_id];
    #pragma unroll
    for (int w = 1; w < NWARPS_K; w++)
        sum += smem[w][lane_id];

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    return sum;
}

// Dual-accumulator version for fused SwiGLU

static __device__ __forceinline__ void mwr_reduce_dual(
    float gate_acc, float up_acc, int warp_id, int lane_id,
    float smem[2][NWARPS_K][WARP_SIZE],
    float* gate_out, float* up_out
) {
    smem[0][warp_id][lane_id] = gate_acc;
    smem[1][warp_id][lane_id] = up_acc;
    __syncthreads();

    *gate_out = 0.0f;
    *up_out = 0.0f;
    if (warp_id != 0) return;

    float gate_sum = smem[0][0][lane_id];
    float up_sum = smem[1][0][lane_id];
    #pragma unroll
    for (int w = 1; w < NWARPS_K; w++) {
        gate_sum += smem[0][w][lane_id];
        up_sum += smem[1][w][lane_id];
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, offset);
        up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, offset);
    }

    *gate_out = gate_sum;
    *up_out = up_sum;
}

// ── Warp reduction helper ───────────────────────────────────────────────

static __device__ __forceinline__ float warp_reduce_sum(float acc) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    return acc;
}

// ── Token-batched MWR block shape ───────────────────────────────────────
// Warps per block for a token-batched MWR kernel, as a function of the
// compile-time tile width. A wider tile holds one accumulator per token in
// registers, so the per-thread register demand grows with NTOK; cutting the
// warp count at the wide end trades K-parallelism inside a block for more
// blocks resident per SM. ggml-cuda's `calc_nwarps` (`ggml-cuda/mmvq.cu`)
// makes the same split, 4 warps up to 4 columns and 2 warps from 5 to 8.
//
// Used for the K-stride loop, the reduction's shared array and
// `__launch_bounds__` alike — all three must agree, so all three read this.

static constexpr __host__ __device__ int mwr_nwarps_ntok(int ntok) {
    return ntok <= 4 ? NWARPS_K : 2;
}

// ── Token-batched MWR output-row tiling ─────────────────────────────────
// Output rows per block for a token-batched MWR kernel, as a function of the
// compile-time tile width. With more than one token column in flight, each
// activation word a thread loads is already paid for; dot-producting it
// against two weight rows instead of one halves the activation traffic per
// unit of output, at the cost of one more weight word and one more
// accumulator per token in registers. At a single token column there is no
// activation word to amortize, so the tile stays one row wide.
//
// ggml-cuda's `calc_rows_per_block` (`ggml-cuda/mmvq.cu`) makes the same split
// off `ncols_dst`: one row at one column, two from two columns up.
//
// A kernel that reads this must launch grid x as `ceil(N / ROWS)` and guard
// its writes against `N` — N need not be a multiple of ROWS. Kernels that keep
// one output row per block ignore this and keep grid x at N.

static constexpr __host__ __device__ int mwr_rows_ntok(int ntok) {
    return ntok >= 2 ? 2 : 1;
}

// ── Token-batched MWR reduction ─────────────────────────────────────────
// NTOK-wide counterpart of `mwr_reduce`, for MWR kernels whose block covers
// several token columns at once, and — when the kernel tiles the output-row
// axis as well — several output rows. Layout mirrors ggml-cuda's
// `tmp_shared[nwarps-1][ncols_dst][rows_per_cuda_block][warp_size]`
// (`ggml-cuda/mmvq.cu`), so the shared array is
// `[NWARPS - 1][NTOK][ROWS][WARP_SIZE]`.
//
// NWARPS is a parameter rather than a constant so every MWR format can pick
// its own block shape per tile width — see `mwr_nwarps_ntok` above. ROWS is a
// parameter for the same reason: a format tiles the output-row axis or does
// not, independently of its tile width. `mwr_rows_ntok` gives the rule for
// the formats that do.
//
// Warp 0 keeps its own partials in registers and never stores, which is why
// the leading extent is NWARPS - 1: the per-block footprint is
// (NWARPS - 1) * NTOK * ROWS * WARP_SIZE * 4 bytes.
//
// Every thread of the block must reach this call — it contains a barrier.
// `out` is written by warp 0 only, and only lane 0 holds the final sums.

template <int NTOK, int ROWS, int NWARPS>
static __device__ __forceinline__ void mwr_reduce_ntok(
    const float acc[NTOK][ROWS], int warp_id, int lane_id,
    float smem[NWARPS - 1][NTOK][ROWS][WARP_SIZE],
    float out[NTOK][ROWS]
) {
    static_assert(NWARPS >= 2, "mwr_reduce_ntok needs at least two warps to reduce across");
    if (warp_id != 0) {
        #pragma unroll
        for (int j = 0; j < NTOK; j++)
            #pragma unroll
            for (int i = 0; i < ROWS; i++)
                smem[warp_id - 1][j][i][lane_id] = acc[j][i];
    }
    __syncthreads();

    #pragma unroll
    for (int j = 0; j < NTOK; j++)
        #pragma unroll
        for (int i = 0; i < ROWS; i++)
            out[j][i] = 0.0f;

    if (warp_id != 0) return;

    #pragma unroll
    for (int j = 0; j < NTOK; j++) {
        #pragma unroll
        for (int i = 0; i < ROWS; i++) {
            float sum = acc[j][i];
            #pragma unroll
            for (int w = 0; w < NWARPS - 1; w++)
                sum += smem[w][j][i][lane_id];
            out[j][i] = warp_reduce_sum(sum);
        }
    }
}

// One-output-row form, for the MWR kernels that keep grid x at N. It is the
// ROWS = 1 case of the template above: a `[...][NTOK][1][WARP_SIZE]` array has
// the same storage and the same element order as `[...][NTOK][WARP_SIZE]`, so
// the flat arrays are re-viewed rather than copied. Explicit template
// arguments pick between the two forms: NWARPS appears only in a non-deduced
// position, so a two-argument call can only mean this overload and a
// three-argument call can only mean the general one.

template <int NTOK, int NWARPS>
static __device__ __forceinline__ void mwr_reduce_ntok(
    const float acc[NTOK], int warp_id, int lane_id,
    float smem[NWARPS - 1][NTOK][WARP_SIZE],
    float out[NTOK]
) {
    mwr_reduce_ntok<NTOK, 1, NWARPS>(
        reinterpret_cast<const float (*)[1]>(acc), warp_id, lane_id,
        reinterpret_cast<float (*)[NTOK][1][WARP_SIZE]>(smem),
        reinterpret_cast<float (*)[1]>(out));
}
