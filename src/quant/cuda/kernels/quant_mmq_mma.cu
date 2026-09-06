// Q8_0 x Q8_1 -> f32 (MMQ), tensor-core variant of `quant_mmq_q8_0_q8_1` in
// `quant_gemv.cu`. The compute loop and the output write use
// `mma_m16n8k32_s8` instead of `dp4a`.
//
// All three kernels here stage one k-block ahead: the global loads for block
// `b+1` are issued before block `b`'s `mma` sequence, so their latency
// overlaps compute. nvcc does not reorder this itself: the loads must cross a
// `__syncthreads()` to move earlier.
//
// This is a separate translation unit, so the tile constants and
// `load_int_ua` are re-declared rather than shared via a header.

#include <cuda_fp16.h>

#include "decode.cuh"
#include "iq_dequant.cuh"
#include "mma_int8.cuh"

#define WARP_SIZE 32
#define MMQ_BM 128
#define MMQ_BN 64
#define MMQ_BK 32
// Row-stride padding, in ints. 8 makes the padded stride 8 mod 32, so the four
// word indices a lane reads land 8 banks apart and the warp's 32 accesses cover
// all 32 banks.
#define MMQ_SMEM_PAD 8
#define MMQ_THREADS 256
// `__launch_bounds__` below passes the block size ONLY. The second argument
// (minimum blocks per SM) is deliberately omitted: setting it to 1 tells ptxas
// that a single resident block suffices, so it spends registers freely on the
// accumulator tile and occupancy drops. Left unconstrained, ptxas picks the
// register count itself, per target architecture, at compile or JIT time —
// which is where that decision belongs, since the register file and the
// latency it has to cover are properties of the device, not of this file.
// Four consecutive k values per int, which is one dp4a/mma operand word.
#define MMQ_K4 (MMQ_BK / 4)

#define MMQ_WARPS (MMQ_THREADS / WARP_SIZE)
// A staging warp covers four columns at once, so the block advances by this
// many columns or rows per staging step.
#define MMQ_STAGE_STRIDE (MMQ_WARPS * 4)
#define MMQ_W_STAGES (MMQ_BN / MMQ_STAGE_STRIDE)
#define MMQ_A_STAGES (MMQ_BM / MMQ_STAGE_STRIDE)

// 2-byte-aligned 4-byte load (quant blocks are not always 4-byte aligned).
static __device__ __forceinline__ int load_int_ua(const unsigned char* p) {
    const unsigned short* p16 = (const unsigned short*)p;
    return (int)p16[0] | ((int)p16[1] << 16);
}

// Reads k-block `b` of both operands into registers. Held separate from the
// shared-memory write so the caller issues it one iteration ahead of use.
// Global load latency dominates this loop; nothing else overlaps it.
static __device__ __forceinline__ void mmq_q8_0_stage_load(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    unsigned int M, unsigned int N, unsigned int bpr, unsigned int b,
    unsigned int row0, unsigned int col0, unsigned int base, unsigned int stage_k4,
    int (&w_packed)[MMQ_W_STAGES], float (&w_d)[MMQ_W_STAGES],
    int (&a_packed)[MMQ_A_STAGES], float (&a_d)[MMQ_A_STAGES]
) {
#pragma unroll
    for (int i = 0; i < MMQ_W_STAGES; ++i) {
        const unsigned int gcol = col0 + base + i * MMQ_STAGE_STRIDE;
        w_packed[i] = 0;
        w_d[i] = 0.0f;
        if (gcol < N) {
            const unsigned char* blk = weight + ((unsigned long long)gcol * bpr + b) * 34;
            w_d[i] = __half2float(*reinterpret_cast<const __half*>(blk));
            // The quants start at byte 2, so only 2-byte alignment holds.
            w_packed[i] = load_int_ua(blk + 2 + stage_k4 * 4);
        }
    }
#pragma unroll
    for (int i = 0; i < MMQ_A_STAGES; ++i) {
        const unsigned int grow = row0 + base + i * MMQ_STAGE_STRIDE;
        a_packed[i] = 0;
        a_d[i] = 0.0f;
        if (grow < M) {
            // Q8_1: d, then the block sum, then 32 quants at byte 4.
            const unsigned char* blk = q8_act + ((unsigned long long)grow * bpr + b) * 36;
            a_d[i] = __half2float(*reinterpret_cast<const __half*>(blk));
            a_packed[i] = *reinterpret_cast<const int*>(blk + 4 + stage_k4 * 4);
        }
    }
}

extern "C" __global__ __launch_bounds__(MMQ_THREADS) void quant_mmq_q8_0_q8_1_mma(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    // Padded by MMQ_SMEM_PAD so the word index reaches the bank index. An
    // unpadded row stride of 128 or 64 ints is a multiple of 32, so every
    // fragment word lands in the same bank and the warp serializes 4 ways.
    __shared__ int s_w[MMQ_K4][MMQ_BN + MMQ_SMEM_PAD];
    __shared__ int s_a[MMQ_K4][MMQ_BM + MMQ_SMEM_PAD];
    __shared__ float s_wd[MMQ_BN];
    __shared__ float s_ad[MMQ_BM];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int warp = tid / WARP_SIZE;

    const unsigned int row0 = blockIdx.y * MMQ_BM;
    const unsigned int col0 = blockIdx.x * MMQ_BN;
    const unsigned int bpr = K / 32;  // blocks per row, both operands

    // A staging warp covers four columns at once: lane -> (column, k4).
    const unsigned int stage_sub = lane / MMQ_K4;  // 0..3
    const unsigned int stage_k4 = lane % MMQ_K4;   // 0..7

    // Warp `warp` owns output rows `16*warp .. 16*warp+16`; eight warps cover
    // all 128 rows. Each warp covers all 64 columns as eight 8-column groups.
    float acc[8][4];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            acc[g][l] = 0.0f;
        }
    }

    const unsigned int stage_base = warp * 4 + stage_sub;
    int w_packed[MMQ_W_STAGES];
    float w_d[MMQ_W_STAGES];
    int a_packed[MMQ_A_STAGES];
    float a_d[MMQ_A_STAGES];

    if (bpr > 0) {
        mmq_q8_0_stage_load(q8_act, weight, M, N, bpr, 0, row0, col0, stage_base,
                            stage_k4, w_packed, w_d, a_packed, a_d);
    }

    for (unsigned int b = 0; b < bpr; ++b) {
        __syncthreads();

#pragma unroll
        for (int i = 0; i < MMQ_W_STAGES; ++i) {
            const unsigned int c = stage_base + i * MMQ_STAGE_STRIDE;
            s_w[stage_k4][c] = w_packed[i];
            if (stage_k4 == 0) s_wd[c] = w_d[i];
        }
#pragma unroll
        for (int i = 0; i < MMQ_A_STAGES; ++i) {
            const unsigned int r = stage_base + i * MMQ_STAGE_STRIDE;
            s_a[stage_k4][r] = a_packed[i];
            if (stage_k4 == 0) s_ad[r] = a_d[i];
        }

        __syncthreads();

        if (b + 1 < bpr) {
            mmq_q8_0_stage_load(q8_act, weight, M, N, bpr, b + 1, row0, col0, stage_base,
                                stage_k4, w_packed, w_d, a_packed, a_d);
        }

        // One `mma_m16n8k32_s8` consumes a whole 32-element Q8_0/Q8_1 block
        // as a single k-step, so the block loop needs no inner k4 loop. The
        // int32 accumulation is exact, so the float scale still applies once
        // per block, same as the dp4a kernel. A, B and D each read a
        // DIFFERENT index map from `mma_int8.cuh` — they are not the same
        // register layout.
        int A[4];
        for (int l = 0; l < 4; ++l) {
            A[l] = s_a[mma_a_j(l)][warp * 16 + mma_a_i(l)];
        }

        for (int g = 0; g < 8; ++g) {
            int B[2];
            for (int l = 0; l < 2; ++l) {
                B[l] = s_w[mma_b_j(l)][g * 8 + mma_b_i(l)];
            }

            int D[4] = {0, 0, 0, 0};
            mma_m16n8k32_s8(D, A, B);

            for (int l = 0; l < 4; ++l) {
                const float da = s_ad[warp * 16 + mma_d_i(l)];
                const float dw = s_wd[g * 8 + mma_d_j(l)];
                acc[g][l] += (float)D[l] * da * dw;
            }
        }
    }

    for (int g = 0; g < 8; ++g) {
        for (int l = 0; l < 4; ++l) {
            const unsigned int r = row0 + warp * 16 + mma_d_i(l);
            const unsigned int c = col0 + g * 8 + mma_d_j(l);
            if (r < M && c < N) output[(unsigned long long)r * N + c] = acc[g][l];
        }
    }
}

// ---------------------------------------------------------------------------
// Quantized weight x Q8_1 -> f32 (MMQ), feature-major decomposition,
// specialized per weight format and batch size.
//
// Computes the same product as `quant_mmq_q8_0_q8_1_mma` above, but with the two
// MMA operand roles swapped: the WEIGHT is operand A (16 output features per
// fragment) and the ACTIVATION is operand B (8 tokens per fragment). The block
// owns a 128-feature x MMQ_X-token output tile, and the grid axes are
// transposed relative to the kernel above.
//
// Four things follow, and they are the point of this variant:
//   - The weight tile is staged 256 k-values deep per outer iteration, so a
//     weight row is read in one wide pass instead of eight narrow ones. For a
//     256-element format that is exactly one super-block per pass.
//   - Shared memory is row-major (feature-major / token-major) with the block
//     scales inline in the same row, so a fragment gather is one strided read
//     and the scale it needs is in the same row.
//   - The activation tile is refilled once per 128 k, giving four barriers per
//     256 k-values rather than one pair per 32.
//   - Activations arrive in the repacked layout `quantize_f32_q8_1_mmq` emits,
//     k-group-major and token-minor, so staging one is a flat copy. `ntok` is
//     that layout's token stride; `y_packed` is its base.
//
// `MMQ_X` is the token tile, one compiled entry point per value. A small batch
// wants a small `MMQ_X`: it cuts both the activation tile and the accumulator
// register count, and it stops the grid from launching blocks whose token
// columns are entirely out of range. Host-side selection lives in Rust.
//
// Each `MMQ_X` compiles three entry points: `_x<N>` is tile-parallel (one block
// per output tile), and `_sk_x<N>` plus `_fixup_x<N>` are the stream-k pair,
// for when the tile count is too small to fill the device. The stream-k pair
// splits the K reduction across blocks, so its sum is reassociated relative to
// the tile-parallel path; both are correct, neither is bit-identical to the
// other or to the pre-sm_80 dp4a kernel.
//
// Shared memory is DYNAMIC. The largest variants exceed the 48 KB static
// limit, so the launcher must opt in with
// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` before the launch and
// request `4 * (MMQF_Y * FMT::X_STRIDE + MMQ_X * MMQF_Y_STRIDE)` bytes.
// ---------------------------------------------------------------------------

#define MMQF_Y 128  // output features per block (MMA "rows", operand A)
#define MMQF_THREADS 256
#define MMQF_WARPS (MMQF_THREADS / WARP_SIZE)

// Warp blocking. `granularity` sets how many MMA feature minitiles one warp
// owns (`ntx`) and therefore how the eight warps split the output tile: with
// ntx=1 they stack 8-deep on features, with ntx=2 they form a 4x2 grid and each
// warp's B fragment feeds two `mma` calls. `MMQ_X` must be a multiple of the
// granularity, which is what makes the variant list 8..40 step 8 and 48..128
// step 16.
#define MMQF_GRAN(X) ((X) >= 48 ? 16 : 8)
#define MMQF_NTX(X) (MMQF_GRAN(X) / 8)
#define MMQF_NJ(X) ((X) / (8 * MMQF_NTX(X)))  // token groups per warp

// Word offset of the second 128-k half within a staged weight row.
#define MMQF_HALF_W 32
// Weight blocks consumed per staging iteration: 256 k-values.
#define MMQF_ITER_B 8
// Independent global loads a staging thread issues before the first shared
// store. Trades registers for memory-level parallelism; the staging loops stall
// on global latency, so this is the knob that moves that stall.
//
// It shrinks as MMQ_X grows because the accumulator is MMQ_X/2 floats and is
// live across the whole kernel: at the widest token tile there is no room left
// to hold a deep batch, and a batch that spills to local memory costs more than
// the latency it hides. Must divide the weight tile's row-iteration count.
#define MMQF_STAGE_BATCH(X) ((X) >= 96 ? 2 : ((X) >= 48 ? 4 : 8))

// Activation row: 4 header words, then 32 quant words (128 k-values). Each
// header word covers one 32-value sub-block, `half` scale in bits 0..15 and
// int16 quant sum in bits 16..31 — see `quantize_f32_q8_1_mmq` in
// `quant_act.cu`. This is exactly one repacked record, which is what lets the
// tile be flat-copied.
#define MMQF_Y_DS 0
#define MMQF_Y_QS 4
#define MMQF_Y_STRIDE 36
// Per-token ints in the optional activation scratch region, which sits
// straight after the activation tile. One int per 32-value sub-block of the
// staged 128-k half. Only a format whose minimum term changes every 16
// elements requests it — see `mmqf_stage_y_sums` and `MmqfQ2K`.
#define MMQF_Y_SCRATCH 4

static_assert(MMQF_Y_QS + 32 == MMQF_Y_STRIDE, "Wrong activation row stride.");

// Copied verbatim from `quant_gemv.cu`, including the `__CUDA_ARCH__ >= 610`
// guard, so the row-sum reductions here and in the token-major kernels below
// match the dp4a kernel exactly. Defined above the feature-major section
// because `mmqf_stage_y_sums` needs it.
static __device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const signed char* a8 = (const signed char*)&a;
    const signed char* b8 = (const signed char*)&b;
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif
}

extern __shared__ int mmqf_smem[];

// Bounds handling for both staging functions below is deliberate and is the
// reason they look the way they do.
//
// A bound must be folded into the ADDRESS, never into a branch around the load.
// The feature-row and token-row bounds are warp-uniform, so a guard on them
// compiles to a real branch, and a branch between unrolled iterations stops
// their loads batching: each iteration then waits on its own load before the
// next is issued. Clamping the index instead keeps every load independent, so
// a whole group is in flight at once.
//
// A clamped row re-reads the last valid row; the masked write-back discards it.
// A clamped k-block is staged but never consumed, because the tail limits how
// many k-steps `FMT::vec_dot` runs. `CLAMP_K` is false for whole 256-k groups,
// which is every iteration but the last of a ragged K, so the common path has
// no clamp at all.
//
// Each loop issues a group of independent loads into registers FIRST, then
// stores the group to shared. Interleaving load and store per iteration would
// serialize on the load.

// Shared `vec_dot` for the formats whose weight term is a scale alone. Q8_0,
// Q4_0, Q5_0, IQ4_NL, IQ4_XS and IQ2_XXS differ ONLY in how `stage` unpacks
// quants; once staged, their weight row is the same shape — signed quants in
// the int8 lanes plus eight f32 scales, one per 32 elements — and their
// arithmetic is identical, so all six formats' `vec_dot` forward here. IQ4_XS
// and IQ2_XXS are the 256-element blocks among them: their scale granularity
// is still 32 elements, so a staged 256-k group takes the same eight scales.
// The three offsets are template parameters rather than hard constants so a
// format that shifts its row layout still reuses this body.
//
// Consumes one staged 128-k half. `k00` is the word offset of that half inside
// a weight row; the activation tile always holds the half at word 0. `i0` is
// the warp's feature base, `jb` its token base within a group.
//
// `FULL` is the whole-half case (four 32-k steps); the tail instantiation stops
// early rather than running steps whose scales are zero.
//
// The k-step loop is outermost so only one A fragment per minitile and its two
// scales are live at a time.
//
// One scalar term per fragment element: the weight scale, the activation
// scale, and the exact int32 dot. There is no minimum term, which is what
// separates this body from `mmqf_vec_dot_dm` below.
template <int MMQ_X, bool FULL, int X_QS, int X_DS, int X_STRIDE>
static __device__ __forceinline__ void mmqf_vec_dot_d(
    const int* __restrict__ s_x, const int* __restrict__ s_y,
    float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
    unsigned int k00, unsigned int nks
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);

    const float* s_xd = (const float*)s_x;
    const half2* s_yds = (const half2*)s_y;

#pragma unroll
    for (unsigned int ks = 0; ks < 4; ++ks) {
        if (!FULL && ks >= nks) {
            break;
        }
        const unsigned int kw = k00 + ks * 8;

        // Weight is operand A: 16 feature rows x 32 k-values per minitile.
        int A[NTX][4];
        float dw[NTX][2];
#pragma unroll
        for (int n = 0; n < NTX; ++n) {
            const unsigned int ir = i0 + n * 16;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                A[n][l] = s_x[(ir + mma_a_i(l)) * X_STRIDE + X_QS + kw + mma_a_j(l)];
            }
            // `mma_d_i(l)` takes one value per `l / 2`, so two weight scales
            // cover the whole accumulator fragment.
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                dw[n][h] = s_xd[(ir + mma_d_i(2 * h)) * X_STRIDE + X_DS + kw / 8];
            }
        }

#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            const unsigned int jt = jj * (NTX * 8) + jb;

            // Activation is operand B: 8 tokens x 32 k-values, shared by all
            // NTX minitiles.
            int B[2];
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                B[l] = s_y[(jt + mma_b_i(l)) * MMQF_Y_STRIDE + MMQF_Y_QS + ks * 8 + mma_b_j(l)];
            }
            // `mma_d_j(l)` takes one value per `l % 2`. `MMQF_Y_STRIDE`/`MMQF_Y_DS`
            // are int counts and a header word is one int, so the index is
            // unchanged. The scale is the LOW half of the header word; the high
            // half is the int16 quant sum, which these formats have no use for
            // and never touch. The producer already rounds `d` through `half`, so
            // reading the low half back as float is an exact round-trip,
            // numerically identical to the old f32-scale layout.
            float da[2];
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                da[l] = __low2float(s_yds[(jt + mma_d_j(l)) * MMQF_Y_STRIDE + MMQF_Y_DS + ks]);
            }

#pragma unroll
            for (int n = 0; n < NTX; ++n) {
                int D[4] = {0, 0, 0, 0};
                mma_m16n8k32_s8(D, A[n], B);

#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    // `da` is the activation scale, `dw` the weight scale. The
                    // int32 result is exact, so both apply once per 32-element
                    // block.
                    acc[jj][n][l] += (float)D[l] * da[l % 2] * dw[n][l / 2];
                }
            }
        }
    }
}

// Weight format policy. The format-generic machinery below reaches the weight
// tile only through one of these: the on-disk block geometry, the staged
// weight-row layout, and the two functions that touch weight data. Q8_0, Q4_0,
// Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, Q3_K, Q2_K, IQ4_NL, IQ4_XS, IQ2_XXS,
// IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ1_S are the instantiations.
struct MmqfQ80 {
    // On-disk block: one f16 scale then 32 int8 quants.
    static constexpr int BLOCK_BYTES = 34;
    static constexpr int BLOCK_ELEMS = 32;
    // K is gated only on `k % 32 == 0`, so a row's last 256-k group can hold
    // fewer than MMQF_ITER_B blocks and the tail path must be compiled.
    static constexpr bool RAGGED_K = true;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: 64 quant words (256 k-values), then 8 f32 block
    // scales, then padding. The padding makes the stride an odd multiple of 4
    // ints, so consecutive rows start in different 128-bit segments and the
    // strided fragment gathers below hit all 32 banks.
    static constexpr int X_QS = 0;
    static constexpr int X_DS = 64;
    static constexpr int X_STRIDE = 76;
    static_assert(X_DS + 8 <= X_STRIDE, "Weight row too short: 8 block scales.");

    // Stages 256 k-values (8 blocks) of the weight tile. One warp owns one
    // feature row per step and steps by the warp count; within the row a lane maps
    // to (block, 4-k word) and issues two loads, for the block at `b0` and the one
    // 128 k-values further along. Independent of MMQ_X.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;
        const unsigned int kbx = lane / 8;   // Q8_0 block within a 128-k half
        const unsigned int kqsx = lane % 8;  // 4-k word within that block

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index; the
        // tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        // `bpr` counts 32-element activation blocks and `b0` indexes them.
        // Q8_0's weight block is also 32 elements, so the counts coincide here;
        // a 256-element format divides by `BLOCK_ELEMS / 32` first.
        const unsigned long long rstride = (unsigned long long)bpr * BLOCK_BYTES;

        // Both block offsets are loop-invariant, so the addressing collapses to one
        // add per row.
        const unsigned int blk_lo = CLAMP_K ? min(b0 + kbx, bpr - 1) : b0 + kbx;
        const unsigned int blk_hi = CLAMP_K ? min(b0 + 4 + kbx, bpr - 1) : b0 + 4 + kbx;
        // The quants start at byte 2 of a 34-byte block, so only 2-byte alignment
        // holds and each word is two 16-bit loads.
        const unsigned long long off_lo = (unsigned long long)blk_lo * BLOCK_BYTES + 2 + kqsx * 4;
        const unsigned long long off_hi = (unsigned long long)blk_hi * BLOCK_BYTES + 2 + kqsx * 4;

        constexpr int ROWS = MMQF_Y / MMQF_WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int v_lo[BATCH];
            int v_hi[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                v_lo[u] = load_int_ua(row + off_lo);
                v_hi[u] = load_int_ua(row + off_hi);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                s_x[i * X_STRIDE + X_QS + lane] = v_lo[u];
                s_x[i * X_STRIDE + X_QS + MMQF_HALF_W + lane] = v_hi[u];
            }
        }

        // Scales are a separate pass: eight per row, so a warp covers four rows.
        float* s_xd = (float*)s_x;
        const unsigned int kbxd = lane % 8;
        const unsigned int rsub = lane / 8;
        const unsigned int blk_d = CLAMP_K ? min(b0 + kbxd, bpr - 1) : b0 + kbxd;
        const unsigned long long off_d = (unsigned long long)blk_d * BLOCK_BYTES;

        constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 4);
        float d[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            const unsigned char* blk = weight + (feat0 + min(i, i_max)) * rstride + off_d;
            d[u] = __half2float(*reinterpret_cast<const __half*>(blk));
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            s_xd[i * X_STRIDE + X_DS + kbxd] = d[u];
        }
    }

    // Forwards to `mmqf_vec_dot_d`, which is shared with Q4_0: the staged row
    // layout and the one-term arithmetic are the same for both.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_d<MMQ_X, FULL, X_QS, X_DS, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};

// Q4_0 weight format policy, same contract as `MmqfQ80`.
//
// Q4_0 is a legacy 32-element format like Q8_0, not a K-quant: `f16 d`@0 then
// 16 bytes holding 32 unsigned 4-bit quants, and the value is `d * (q - 8)`.
// Biasing by 8 during staging turns it into exactly Q8_0's staged row —
// signed quants in the int8 lanes plus one f32 scale per 32-element block —
// so the whole row layout and `vec_dot` are Q8_0's, and `stage` is the only
// thing this format defines for itself.
//
// ALIGNMENT. The block is 18 bytes, so a row base (`bpr * 18`) and every block
// base inside it are only 2-byte aligned. Every 4-byte read of `qs` therefore
// goes through `load_int_ua`; a plain `int` load raises
// CUDA_ERROR_MISALIGNED_ADDRESS, which poisons the context for every later
// launch on it. The `f16 d` at byte 0 is 2-byte aligned, which is
// `alignof(__half)`, and is read directly.
struct MmqfQ40 {
    // On-disk block: one f16 scale then 16 bytes of nibble-packed quants.
    static constexpr int BLOCK_BYTES = 18;
    static constexpr int BLOCK_ELEMS = 32;
    // K is gated only on `k % 32 == 0`, so a row's last 256-k group can hold
    // fewer than MMQF_ITER_B blocks and the tail path must be compiled.
    static constexpr bool RAGGED_K = true;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: Q8_0's, byte for byte. 64 quant words (256 k-values),
    // then 8 f32 block scales, then padding that makes the stride an odd
    // multiple of 4 ints so the strided fragment gathers hit all 32 banks.
    static constexpr int X_QS = 0;
    static constexpr int X_DS = 64;
    static constexpr int X_STRIDE = 76;
    static_assert(X_DS + 8 <= X_STRIDE, "Weight row too short: 8 block scales.");
    static_assert(
        X_QS == MmqfQ80::X_QS && X_DS == MmqfQ80::X_DS && X_STRIDE == MmqfQ80::X_STRIDE,
        "Q4_0 must stage into the Q8_0 row; the two share `mmqf_vec_dot_d`."
    );

    // Stages 256 k-values (8 blocks) of the weight tile. A block holds 4
    // source ints and produces 8 staged words, so the warp's 32 lanes cover a
    // whole 256-k group with ONE global load each: lane maps to
    // (block `lane / 4`, 4-k word `lane % 4`). Independent of MMQ_X.
    //
    // Quant map, transcribed from `load_tiles_q4_0` in llama.cpp's
    // `ggml-cuda/mmq.cuh` and cross-checked against `dequant_q4_0` in
    // `src/quant/cpu/kernels/dequant_simple.rs`: within one 32-element block,
    // element `j` (0..15) is the LOW nibble of `qs[j]` and element `j + 16` is
    // the HIGH nibble of the same byte. So the int at `qs + 4*w` (w = 0..3)
    // carries elements `4w..4w+3` in its low nibbles and `4w+16..4w+19` in its
    // high nibbles, which places the two staged words at `8*blk + w` and 4
    // further along. This is NOT Q4_K's map: Q4_K pairs sub-blocks across a
    // shared 32-byte run, while a Q4_0 block is a single 32-element run.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;
        const unsigned int kbx = lane / 4;   // block within the 256-k group
        const unsigned int kqsx = lane % 4;  // 4-k source word within that block

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index; the
        // tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        // `bpr` counts 32-element activation blocks and `b0` indexes them.
        // Q4_0's weight block is also 32 elements, so the counts coincide here;
        // a 256-element format divides by `BLOCK_ELEMS / 32` first.
        const unsigned long long rstride = (unsigned long long)bpr * BLOCK_BYTES;

        // Loop-invariant, so the addressing collapses to one add per row.
        const unsigned int blk = CLAMP_K ? min(b0 + kbx, bpr - 1) : b0 + kbx;
        // The quants start at byte 2 of an 18-byte block, so only 2-byte
        // alignment holds and each word is two 16-bit loads.
        const unsigned long long off_qs = (unsigned long long)blk * BLOCK_BYTES + 2 + kqsx * 4;
        // Both staged words live in the same 256-k group: words 0..31 are the
        // first 128-k half, 32..63 the second.
        const unsigned int w_lo = kbx * 8 + kqsx;

        constexpr int ROWS = MMQF_Y / MMQF_WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int v[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                v[u] = load_int_ua(row + off_qs);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                // The 8 bias is applied here, per byte, so the staged lanes are
                // signed -8..7 and `vec_dot` is Q8_0's unchanged. Inputs are
                // 0..15, so the subtract never saturates.
                s_x[i * X_STRIDE + X_QS + w_lo] = __vsubss4(v[u] & 0x0F0F0F0F, 0x08080808);
                s_x[i * X_STRIDE + X_QS + w_lo + 4] =
                    __vsubss4((v[u] >> 4) & 0x0F0F0F0F, 0x08080808);
            }
        }

        // Scales are a separate pass: eight per row, so a warp covers four rows.
        float* s_xd = (float*)s_x;
        const unsigned int kbxd = lane % 8;
        const unsigned int rsub = lane / 8;
        const unsigned int blk_d = CLAMP_K ? min(b0 + kbxd, bpr - 1) : b0 + kbxd;
        const unsigned long long off_d = (unsigned long long)blk_d * BLOCK_BYTES;

        constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 4);
        float d[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            const unsigned char* blkp = weight + (feat0 + min(i, i_max)) * rstride + off_d;
            d[u] = __half2float(*reinterpret_cast<const __half*>(blkp));
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            s_xd[i * X_STRIDE + X_DS + kbxd] = d[u];
        }
    }

    // Forwards to `mmqf_vec_dot_d`, shared with Q8_0: once the bias is folded
    // in during staging the two rows are indistinguishable.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_d<MMQ_X, FULL, X_QS, X_DS, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};

// Q5_0 weight format policy, same contract as `MmqfQ80`.
//
// Q5_0 is a legacy 32-element format like Q4_0: `f16 d`@0, a 32-bit `qh`@2
// holding one fifth bit per element, then 16 bytes holding the low 4 bits of
// 32 quants@6. The value is `d * (q - 16)` with `q` the unsigned 5-bit
// assembly. Biasing by 16 during staging turns it into exactly Q8_0's staged
// row — signed quants in the int8 lanes plus one f32 scale per 32-element
// block — so the whole row layout and `vec_dot` are Q8_0's, and `stage` is the
// only thing this format defines for itself.
//
// ALIGNMENT. The block is 22 bytes, so a row base (`bpr * 22`) and every block
// base inside it are only 2-byte aligned, and `qh`@2 and `qs`@6 inherit that.
// EVERY 4-byte read here therefore goes through `load_int_ua`; a plain `int`
// load raises CUDA_ERROR_MISALIGNED_ADDRESS, which poisons the context for
// every later launch on it. The `f16 d` at byte 0 is 2-byte aligned, which is
// `alignof(__half)`, and is read directly.
struct MmqfQ50 {
    // On-disk block: f16 scale, 32-bit fifth-bit field, 16 nibble bytes.
    static constexpr int BLOCK_BYTES = 22;
    static constexpr int BLOCK_ELEMS = 32;
    // K is gated only on `k % 32 == 0`, so a row's last 256-k group can hold
    // fewer than MMQF_ITER_B blocks and the tail path must be compiled.
    static constexpr bool RAGGED_K = true;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: Q8_0's, byte for byte. 64 quant words (256 k-values),
    // then 8 f32 block scales, then padding that makes the stride an odd
    // multiple of 4 ints so the strided fragment gathers hit all 32 banks.
    static constexpr int X_QS = 0;
    static constexpr int X_DS = 64;
    static constexpr int X_STRIDE = 76;
    static_assert(X_DS + 8 <= X_STRIDE, "Weight row too short: 8 block scales.");
    static_assert(
        X_QS == MmqfQ80::X_QS && X_DS == MmqfQ80::X_DS && X_STRIDE == MmqfQ80::X_STRIDE,
        "Q5_0 must stage into the Q8_0 row; the two share `mmqf_vec_dot_d`."
    );

    // Stages 256 k-values (8 blocks) of the weight tile, on Q4_0's lane map: a
    // block holds 4 source `qs` ints and produces 8 staged words, so the
    // warp's 32 lanes cover a whole 256-k group with one `qs` load and one
    // `qh` load each — lane maps to (block `lane / 4`, 4-k word `lane % 4`).
    // Independent of MMQ_X.
    //
    // Quant map, transcribed from `load_tiles_q5_0` in llama.cpp's
    // `ggml-cuda/mmq.cuh` and cross-checked element by element against
    // `dequant_q5_0` in `src/quant/cpu/kernels/dequant_simple.rs`: within one
    // 32-element block, element `j` (0..15) is the LOW nibble of `qs[j]` and
    // element `j + 16` is the HIGH nibble of the same byte, exactly as in
    // Q4_0. So the int at `qs + 4*w` (w = 0..3) carries elements `4w..4w+3` in
    // its low nibbles and `4w+16..4w+19` in its high nibbles, placing the two
    // staged words at `8*blk + w` and 4 further along.
    //
    // The fifth bit of element `j` is bit `j` of the 32-bit `qh`, so the whole
    // block shares one `qh` word and this lane needs bits `4w..4w+3` for its
    // low word and `4w+16..4w+19` for its high one. Pre-shifting by `4*w`
    // leaves those in bits 0..3 and 16..19, and the four masked shifts below
    // move bit `t` of that into bit 4 of staged byte `t` — that is upstream's
    // form, kept verbatim because the source bits are packed contiguously
    // inside one nibble rather than spread one per byte, so Q5_K's single
    // `(qh >> sh) & 0x01010101` trick does not apply here.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;
        const unsigned int kbx = lane / 4;   // block within the 256-k group
        const unsigned int kqsx = lane % 4;  // 4-k source word within that block

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index; the
        // tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        // `bpr` counts 32-element activation blocks and `b0` indexes them.
        // Q5_0's weight block is also 32 elements, so the counts coincide here;
        // a 256-element format divides by `BLOCK_ELEMS / 32` first.
        const unsigned long long rstride = (unsigned long long)bpr * BLOCK_BYTES;

        // Loop-invariant, so the addressing collapses to one add per row.
        const unsigned int blk = CLAMP_K ? min(b0 + kbx, bpr - 1) : b0 + kbx;
        // Byte 6 and byte 2 of a 22-byte block: 2-byte aligned, so both reads
        // are `load_int_ua`.
        const unsigned long long off_qs = (unsigned long long)blk * BLOCK_BYTES + 6 + kqsx * 4;
        const unsigned long long off_qh = (unsigned long long)blk * BLOCK_BYTES + 2;
        // Both staged words live in the same 256-k group: words 0..31 are the
        // first 128-k half, 32..63 the second.
        const unsigned int w_lo = kbx * 8 + kqsx;
        const unsigned int shq = 4 * kqsx;

        constexpr int ROWS = MMQF_Y / MMQF_WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int vl[BATCH];
            int vh[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                vl[u] = load_int_ua(row + off_qs);
                vh[u] = load_int_ua(row + off_qh);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                const int qh = (int)((unsigned int)vh[u] >> shq);
                // 4 low bits from `qs`, the fifth from `qh`, then the 16 bias
                // per byte so the staged lanes are signed -16..15 and
                // `vec_dot` is Q8_0's unchanged. Inputs are 0..31, so the
                // subtract never saturates.
                int lo = vl[u] & 0x0F0F0F0F;
                lo |= (qh << 4) & 0x00000010;   // element 4w+0 -> byte 0 bit 4
                lo |= (qh << 11) & 0x00001000;  // element 4w+1 -> byte 1 bit 4
                lo |= (qh << 18) & 0x00100000;  // element 4w+2 -> byte 2 bit 4
                lo |= (qh << 25) & 0x10000000;  // element 4w+3 -> byte 3 bit 4
                int hi = (vl[u] >> 4) & 0x0F0F0F0F;
                hi |= (qh >> 12) & 0x00000010;  // element 4w+16 -> byte 0 bit 4
                hi |= (qh >> 5) & 0x00001000;   // element 4w+17 -> byte 1 bit 4
                hi |= (qh << 2) & 0x00100000;   // element 4w+18 -> byte 2 bit 4
                hi |= (qh << 9) & 0x10000000;   // element 4w+19 -> byte 3 bit 4
                s_x[i * X_STRIDE + X_QS + w_lo] = __vsubss4(lo, 0x10101010);
                s_x[i * X_STRIDE + X_QS + w_lo + 4] = __vsubss4(hi, 0x10101010);
            }
        }

        // Scales are a separate pass: eight per row, so a warp covers four rows.
        float* s_xd = (float*)s_x;
        const unsigned int kbxd = lane % 8;
        const unsigned int rsub = lane / 8;
        const unsigned int blk_d = CLAMP_K ? min(b0 + kbxd, bpr - 1) : b0 + kbxd;
        const unsigned long long off_d = (unsigned long long)blk_d * BLOCK_BYTES;

        constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 4);
        float d[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            const unsigned char* blkp = weight + (feat0 + min(i, i_max)) * rstride + off_d;
            d[u] = __half2float(*reinterpret_cast<const __half*>(blkp));
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            s_xd[i * X_STRIDE + X_DS + kbxd] = d[u];
        }
    }

    // Forwards to `mmqf_vec_dot_d`, shared with Q8_0 and Q4_0: once the 16
    // bias is folded in during staging the three rows are indistinguishable.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_d<MMQ_X, FULL, X_QS, X_DS, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};

// Shared `vec_dot` for the formats carrying a scale and a minimum per
// 32-element block: Q4_K, Q5_K, Q4_1 and Q5_1. They differ ONLY in how `stage`
// unpacks quants and in the SIGN of the minimum it folds in; once staged,
// their weight row is the same shape — unsigned quants in the int8 lanes plus
// eight `(scale_j, min_j)` f32 pairs — and their arithmetic is identical, so
// all four formats' `vec_dot` forward here. The K-quants store
// `(d * sc_j, -dmin * m_j)` because their value is `d * sc * q - dmin * m`;
// Q4_1 and Q5_1 store `(d, +m)` because theirs is `d * q + m`. This body
// always ADDS `pair.y * (activation scale * block sum)`. The three offsets are template parameters rather than hard
// constants so a format that shifts its row layout still reuses this body.
//
// Consumes one staged 128-k half; the loop shape matches `MmqfQ80::vec_dot`
// exactly, four 32-k steps of one `mma_m16n8k32_s8` per (token group,
// minitile). Two scalar terms per fragment element instead of one:
//
//   acc += pair.x * d_act * int32 dot
//   acc += pair.y * d_act * int32 block sum
//
// The second is the minimum correction. Its block sum is the sum of that
// token's 32 activation quants for this 32-k step, which the producer
// already computed and stored EXACTLY as the int16 in bits 16..31 of the
// activation header word. It is read here, not recomputed: an all-ones
// `mma` also yields it exactly, but costs one extra tensor-core issue per
// (token group, k-step), and a `half` `d * sum(q)` field is too coarse for
// the GEMM/GEMV parity bound. |sum| <= 32 * 128 = 4096 fits int16, so the
// stored value equals the MMA's to the bit.
//
// The sum is per token, so it indexes exactly like the scale `da[l % 2]`:
// both come from the same header word at row `mma_d_j(l)`, and neither
// depends on the feature minitile.
template <int MMQ_X, bool FULL, int X_QS, int X_DM, int X_STRIDE>
static __device__ __forceinline__ void mmqf_vec_dot_dm(
    const int* __restrict__ s_x, const int* __restrict__ s_y,
    float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
    unsigned int k00, unsigned int nks
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);

    // The weight pair is f32. The activation header stays a packed 32-bit
    // word, which is the producer's record format for every weight format,
    // and is read here through `s_y` directly.
    const float2* s_xdm = (const float2*)s_x;

#pragma unroll
    for (unsigned int ks = 0; ks < 4; ++ks) {
        if (!FULL && ks >= nks) {
            break;
        }
        const unsigned int kw = k00 + ks * 8;

        int A[NTX][4];
        float2 dmw[NTX][2];
#pragma unroll
        for (int n = 0; n < NTX; ++n) {
            const unsigned int ir = i0 + n * 16;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                A[n][l] = s_x[(ir + mma_a_i(l)) * X_STRIDE + X_QS + kw + mma_a_j(l)];
            }
            // `mma_d_i(l)` takes one value per `l / 2`, and `kw / 8` is the
            // sub-block index the staged pair belongs to.
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                dmw[n][h] = s_xdm[((ir + mma_d_i(2 * h)) * X_STRIDE + X_DM) / 2 + kw / 8];
            }
        }

#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            const unsigned int jt = jj * (NTX * 8) + jb;

            int B[2];
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                B[l] = s_y[(jt + mma_b_i(l)) * MMQF_Y_STRIDE + MMQF_Y_QS + ks * 8 + mma_b_j(l)];
            }
            // One header word per token, holding both scalars this format
            // needs: `half` scale in bits 0..15, int16 quant sum in bits
            // 16..31. `mma_d_j(l)` takes one value per `l % 2`, so two
            // words cover the whole accumulator fragment. The producer
            // rounds `d` through `half`, so reading the low half back as
            // float is an exact round-trip; the high half is sign-extended
            // by an arithmetic shift of the signed word, then widened,
            // which is exact for every attainable value.
            float da[2];
            float sum_i[2];
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int ds = s_y[(jt + mma_d_j(l)) * MMQF_Y_STRIDE + MMQF_Y_DS + ks];
                da[l] = __half2float(__ushort_as_half((unsigned short)(ds & 0xFFFF)));
                sum_i[l] = (float)(ds >> 16);
            }

            // The activation scale times the block sum is per token, so it
            // is formed once here rather than once per minitile inside the
            // `n` loop below.
            float dsum[2];
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                dsum[l] = da[l] * sum_i[l];
            }

#pragma unroll
            for (int n = 0; n < NTX; ++n) {
                int D[4] = {0, 0, 0, 0};
                mma_m16n8k32_s8(D, A[n], B);

#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    acc[jj][n][l] += dmw[n][l / 2].x * da[l % 2] * (float)D[l];
                    acc[jj][n][l] += dmw[n][l / 2].y * dsum[l % 2];
                }
            }
        }
    }
}

// Q4_K weight format policy, same contract as `MmqfQ80`.
//
// The staged row is WIDER than Q8_0's: 64 quant words then 16 ints, because
// the scale/min pair is kept as two f32 rather than one `half2`. Half has an
// 11-bit significand, and `d * sc` rounded to it perturbs every 32-element
// sub-block's contribution by that much; the GEMM path then drifts from the
// GEMV path past the parity bound the backend tests hold both to. The pair is
// read once per fragment and reused across four `mma` results, so the cost of
// exact staged scales is one extra shared word per pair read and 8 extra ints
// per staged row.
//
// Q4_K dequantizes to `d * sc * q - dmin * m` with UNSIGNED 4-bit `q`. The
// minimum term is rank-1 over the k-block: it depends on the weight row and
// the activation block sum, not on the individual quants, so it folds into the
// existing accumulator as one extra multiply-add per fragment element. The
// block sum itself is read straight out of the activation record's header
// word, where the producer stores it as an EXACT int16 — see `vec_dot`.
struct MmqfQ4K {
    // On-disk super-block: f16 d, f16 dmin, 12 packed 6-bit scale/min bytes,
    // then 128 nibble-packed quants.
    static constexpr int BLOCK_BYTES = 144;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole. The ragged tail path is dead for this format and is not compiled.
    static constexpr bool RAGGED_K = false;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: 64 quant words (256 k-values, one unsigned nibble per
    // int8 lane), then 8 `float2` holding `(d * sc_j, -dmin * m_j)`, one per
    // 32-element sub-block, then padding. The `-1` on the minimum is folded in
    // here so the consumer is a plain multiply-add. Offsets are int counts like
    // Q8_0's, so a `float2` spans two of them: 16 ints for the eight pairs.
    // Both `X_DM` and `X_STRIDE` are even, which keeps every pair 8-byte
    // aligned and the access a single `ld.shared.v2.f32`.
    static constexpr int X_QS = 0;
    static constexpr int X_DM = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DM + 16 <= X_STRIDE, "Weight row too short: 8 scale/min pairs.");
    static_assert(X_DM % 2 == 0 && X_STRIDE % 2 == 0, "Scale/min pairs are misaligned.");

    // Stages 256 k-values, which for this format is exactly ONE super-block.
    // `b0` counts 32-element Q8_1 activation blocks, so the super-block index
    // is `b0 / 8`.
    //
    // Quant map, transcribed from `load_tiles_q4_K` in llama.cpp's
    // `ggml-cuda/mmq.cuh`: lane `l` reads the aligned int at `qs + 4*l`, whose
    // low nibbles are 4 k-values of the EVEN sub-block `2*(l/8)` and whose
    // high nibbles are the same 4 positions of the ODD sub-block `2*(l/8)+1`.
    // A sub-block pair shares one 32-byte run of `qs`; it is not a 16-byte run
    // per sub-block. That places the two staged words at
    // `16*(l/8) + l%8` and 8 further along.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;

        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / (BLOCK_ELEMS / 32);
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;
        const unsigned int sup =
            CLAMP_K ? min(b0 / (BLOCK_ELEMS / 32), supers - 1) : b0 / (BLOCK_ELEMS / 32);
        const unsigned long long off_blk = (unsigned long long)sup * BLOCK_BYTES;

        // 144 is a multiple of 16 and the quants start at byte 16, so every
        // word here is 4-byte aligned and a plain `int` load is legal.
        const unsigned long long off_qs = off_blk + 16 + lane * 4;
        const unsigned int w_lo = 16 * (lane / 8) + lane % 8;

        constexpr int ROWS = MMQF_Y / MMQF_WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int v[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                v[u] = *reinterpret_cast<const int*>(row + off_qs);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                // Unsigned 0..15 sits inside the signed int8 range the `mma`
                // takes, so no bias is applied; the minimum term below is what
                // carries the format's asymmetry.
                s_x[i * X_STRIDE + X_QS + w_lo] = v[u] & 0x0F0F0F0F;
                s_x[i * X_STRIDE + X_QS + w_lo + 8] = (v[u] >> 4) & 0x0F0F0F0F;
            }
        }

        // Scale/min pass: eight pairs per row, so a warp covers four rows.
        // The three packed bytes each lane needs are at row-invariant offsets,
        // so the first loop is pure loads and the 6-bit unpack happens in the
        // second, on registers.
        float2* s_xdm = (float2*)s_x;
        const unsigned int j = lane % 8;  // sub-block within the super-block
        const unsigned int rsub = lane / 8;
        const unsigned int o_j4 = j + 4;
        // For j < 4 the third byte is unused; re-reading sc[j] keeps the index
        // in range without a branch around the load.
        const unsigned int o_jm4 = j < 4 ? j : j - 4;

        constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 4);
        half2 dm[SROWS];
        unsigned int b_j[SROWS];
        unsigned int b_j4[SROWS];
        unsigned int b_jm4[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            const unsigned char* blk = weight + (feat0 + min(i, i_max)) * rstride + off_blk;
            // `d` and `dmin` are adjacent f16 at byte 0 of a 16-aligned block,
            // so the pair is one aligned 4-byte load.
            dm[u] = *reinterpret_cast<const half2*>(blk);
            const unsigned char* sc = blk + 4;  // 12 packed 6-bit scale/min bytes
            b_j[u] = sc[j];
            b_j4[u] = sc[o_j4];
            b_jm4[u] = sc[o_jm4];
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            int scale;
            int minimum;
            q4k_scale_min_bytes(b_j[u], b_j4[u], b_jm4[u], (int)j, &scale, &minimum);
            const float d = __low2float(dm[u]);
            const float dmin = __high2float(dm[u]);
            // Stored as f32: no half round-trip on `d * sc`, which is the
            // term the parity bound is sensitive to.
            s_xdm[(i * X_STRIDE + X_DM) / 2 + j] =
                make_float2(d * (float)scale, -dmin * (float)minimum);
        }
    }

    // Forwards to `mmqf_vec_dot_dm`, which is shared with Q5_K: the staged
    // row layout and the two-term arithmetic are the same for both.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_dm<MMQ_X, FULL, X_QS, X_DM, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};

// Shared `vec_dot` for the formats whose scale changes every 16 elements.
// Q6_K, Q3_K, Q2_K, IQ2_XS and IQ2_S differ ONLY in how `stage` unpacks quants
// and in whether the format carries a minimum term; once staged, their weight
// rows are the same shape — quants in the int8 lanes plus one scale record per
// 16-element group — and the MMA structure is identical, so all five formats'
// `vec_dot` forward here. IQ2_XS and IQ2_S reach that granularity from the
// other side: their 4-bit scale is packed two to a `scales` byte, one per two
// grid entries, and a grid entry is eight elements. The offsets are template
// parameters rather than hard constants so a format that shifts its row layout
// still reuses this body.
//
// Consumes one staged 128-k half, four 32-k steps, each split into two
// 16-k `mma_m16n8k16_s8` calls because the scale changes at 16. Words
// `kw .. kw+3` are the low half and take scale `kw / 4`; words
// `kw+4 .. kw+7` are the high half and take scale `kw / 4 + 1`.
//
// `MIN` selects the scale record and the minimum term, and nothing else:
//
//   MIN == false (Q6_K, Q3_K,  `X_DF` holds 16 f32 `d * scale_j`. The int16
//                 IQ2_XS,      block sum in the HIGH half of the activation
//                 IQ2_S)       header word is never read; only the `half`
//                              activation scale in the low half is.
//
//   MIN == true  (Q2_K)        `X_DF` holds 16 `float2`
//                              `(d * sc_j, -dmin * m_j)`. Each 16-k half takes
//                              one extra multiply-add against the activation's
//                              sum over ITS OWN 16 elements — see the split
//                              derived by `mmqf_stage_y_sums`.
//
// The k-step loop is outermost so only one A fragment pair and its two
// scale pairs are live at a time. ggml hoists a `scA[ntx][ne/2][8]`
// register array across the whole tile instead; that costs more registers
// than this kernel's accumulator leaves free.
template <int MMQ_X, bool FULL, bool MIN, int X_QS, int X_DF, int X_STRIDE>
static __device__ __forceinline__ void mmqf_vec_dot_sc16(
    const int* __restrict__ s_x, const int* __restrict__ s_y,
    float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
    unsigned int k00, unsigned int nks
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);

    const float* s_xdf = (const float*)s_x;

#pragma unroll
    for (unsigned int ks = 0; ks < 4; ++ks) {
        if (!FULL && ks >= nks) {
            break;
        }
        const unsigned int kw = k00 + ks * 8;

        // Weight is operand A at the 16x16 shape: 16 feature rows x 16
        // k-values, two ints per lane. `mma_a16_*` is the k=16 map and is
        // NOT interchangeable with `mma_a_*`.
        int A_lo[NTX][2];
        int A_hi[NTX][2];
        float sc_lo[NTX][2];
        float sc_hi[NTX][2];
        float mn_lo[NTX][2];
        float mn_hi[NTX][2];
#pragma unroll
        for (int n = 0; n < NTX; ++n) {
            const unsigned int ir = i0 + n * 16;
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const unsigned int r = (ir + mma_a16_i(l)) * X_STRIDE + X_QS + kw;
                A_lo[n][l] = s_x[r + mma_a16_j(l)];
                A_hi[n][l] = s_x[r + 4 + mma_a16_j(l)];
            }
            // `mma_d_i(l)` takes one value per `l / 2`, so two rows cover
            // the whole accumulator fragment. D keeps the 16x8 shape at
            // both `mma` widths, so this map is the same one the k=32
            // formats use.
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const unsigned int row = (ir + mma_d_i(2 * h)) * X_STRIDE;
                if constexpr (MIN) {
                    // One `float2` per 16-element group: `(d * sc, -dmin * m)`.
                    // `X_DF` and `X_STRIDE` are both even for this layout, so
                    // each pair is one `ld.shared.v2.f32`.
                    const float2* s_xdm = (const float2*)s_x;
                    const float2 lo = s_xdm[(row + X_DF) / 2 + kw / 4];
                    const float2 hi = s_xdm[(row + X_DF) / 2 + kw / 4 + 1];
                    sc_lo[n][h] = lo.x;
                    mn_lo[n][h] = lo.y;
                    sc_hi[n][h] = hi.x;
                    mn_hi[n][h] = hi.y;
                } else {
                    const unsigned int r = row + X_DF + kw / 4;
                    sc_lo[n][h] = s_xdf[r];
                    sc_hi[n][h] = s_xdf[r + 1];
                }
            }
        }

#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            const unsigned int jt = jj * (NTX * 8) + jb;

            // Activation is operand B at the 8x16 shape: 8 tokens x 16
            // k-values, one int per lane, shared by all NTX minitiles.
            int B_lo[1];
            int B_hi[1];
            {
                const unsigned int r =
                    (jt + mma_b16_i(0)) * MMQF_Y_STRIDE + MMQF_Y_QS + ks * 8;
                B_lo[0] = s_y[r + mma_b16_j(0)];
                B_hi[0] = s_y[r + 4 + mma_b16_j(0)];
            }
            // One header word per token: `half` scale in bits 0..15, int16
            // quant sum in bits 16..31. Only the scale is read unless `MIN`.
            // The producer rounds `d` through `half`, so reading the low half
            // back as float is an exact round-trip. `mma_d_j(l)` takes one
            // value per `l % 2`, so two words cover the fragment. None of
            // these reads depends on the feature minitile, so they all stay
            // outside the `n` loop below.
            float da[2];
            float dsum_lo[2];
            float dsum_hi[2];
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int ds =
                    s_y[(jt + mma_d_j(l)) * MMQF_Y_STRIDE + MMQF_Y_DS + ks];
                da[l] = __half2float(__ushort_as_half((unsigned short)(ds & 0xFFFF)));
                if constexpr (MIN) {
                    // The header's high half is the EXACT int16 sum of this
                    // token's 32 quants for this step; `s_ys` holds the exact
                    // int sum of the FIRST 16 of the same 32. Both are integer
                    // sums over the same values, so `s32 - s_a` is the sum of
                    // the last 16 exactly — an integer identity, not an
                    // approximation. The activation scale is per token and is
                    // folded in here, outside the minitile loop below.
                    //
                    // The sums live in their own region straight after the
                    // activation tile, at a fixed offset from it, so the format
                    // contract does not have to carry a fourth pointer.
                    const int* s_ys = s_y + MMQ_X * MMQF_Y_STRIDE;
                    const int s32 = ds >> 16;
                    const int s_a = s_ys[(jt + mma_d_j(l)) * MMQF_Y_SCRATCH + ks];
                    dsum_lo[l] = da[l] * (float)s_a;
                    dsum_hi[l] = da[l] * (float)(s32 - s_a);
                }
            }

#pragma unroll
            for (int n = 0; n < NTX; ++n) {
                int D_lo[4] = {0, 0, 0, 0};
                int D_hi[4] = {0, 0, 0, 0};
                mma_m16n8k16_s8(D_lo, A_lo[n], B_lo);
                mma_m16n8k16_s8(D_hi, A_hi[n], B_hi);

#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    // Each 16-k half is exact in int32, so its own
                    // `d * scale` applies once; the activation scale is
                    // common to both halves and factors out.
                    acc[jj][n][l] += (sc_lo[n][l / 2] * (float)D_lo[l]
                                      + sc_hi[n][l / 2] * (float)D_hi[l])
                                     * da[l % 2];
                    if constexpr (MIN) {
                        acc[jj][n][l] += mn_lo[n][l / 2] * dsum_lo[l % 2]
                                         + mn_hi[n][l / 2] * dsum_hi[l % 2];
                    }
                }
            }
        }
    }
}

// Q6_K weight format policy, same contract as `MmqfQ80`.
//
// Two properties set this format apart from the two above.
//
// ALIGNMENT. The super-block is 210 bytes, which is only 2-byte aligned, so a
// row base and every super-block base inside it are 2-byte aligned as well.
// Every 4-byte read of `ql`, `qh` or `scales` therefore goes through
// `load_int_ua`; a plain `int` load raises CUDA_ERROR_MISALIGNED_ADDRESS,
// which poisons the context for every later launch on it. The `half` scale at
// byte 208 is 2-byte aligned and is read directly.
//
// SCALE GRANULARITY. The scale changes every 16 elements, so a single
// `mma_m16n8k32_s8` cannot express one 32-k step: its two 16-k halves need
// different scales. Each 32-k step is therefore two `mma_m16n8k16_s8` calls,
// each scaled by its own `d * scale`. This is what the token-major
// `quant_mmq_q6_k_q8_1_mma` kernel below does as well.
struct MmqfQ6K {
    // On-disk super-block: 128 bytes `ql`, 64 bytes `qh`, 16 signed `scales`,
    // then the f16 `d` at byte 208.
    static constexpr int BLOCK_BYTES = 210;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: 64 quant words (256 k-values, one signed 6-bit value
    // biased to -32..31 per int8 lane), then 16 f32 holding `d * scale_j`, one
    // per 16-element group, then padding.
    //
    // The scale is staged as f32 already multiplied by `d`, never as the raw
    // `int8` and never through `half`: half has an 11-bit significand, and
    // rounding `d * scale` to it perturbs every 16-element group's
    // contribution past the bound the GEMM/GEMV parity tests hold this path
    // to. Same stride as Q4_K, so the two share the family's shared-memory
    // request exactly.
    static constexpr int X_QS = 0;
    static constexpr int X_DF = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DF + 16 <= X_STRIDE, "Weight row too short: 16 group scales.");

    // Stages 256 k-values, which for this format is exactly ONE super-block.
    // `b0` counts 32-element Q8_1 activation blocks, so the super-block index
    // is `b0 / 8`.
    //
    // Quant map, transcribed from `load_tiles_q6_K` in llama.cpp's
    // `ggml-cuda/mmq.cuh`. Lane `l` reads the unaligned int at `ql + 4*l` and
    // the unaligned int at `qh + 4*(8*(l/16) + l%8)`, and produces TWO staged
    // words: the low nibbles at word `32*(l/16) + l%16` and the high nibbles
    // 16 words further along. `s = (l & 8) >> 2` picks the `qh` 2-bit field
    // for the low-nibble word; the high-nibble word takes `s + 4`. Under that
    // map staged element `k` belongs to scale group `k / 16` for every `k`,
    // so a staged word `w` takes scale `w / 4`.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;

        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / (BLOCK_ELEMS / 32);
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;
        const unsigned int sup =
            CLAMP_K ? min(b0 / (BLOCK_ELEMS / 32), supers - 1) : b0 / (BLOCK_ELEMS / 32);
        const unsigned long long off_blk = (unsigned long long)sup * BLOCK_BYTES;

        // Row-invariant byte offsets, so the addressing collapses to one add
        // per row. `ql` starts at byte 0 of the super-block, `qh` at byte 128.
        const unsigned long long off_ql = off_blk + lane * 4;
        const unsigned long long off_qh = off_blk + 128 + (8 * (lane / 16) + lane % 8) * 4;
        const unsigned int w_lo = 32 * (lane / 16) + lane % 16;
        const unsigned int sh = (lane & 8) >> 2;  // 0 or 2

        constexpr int ROWS = MMQF_Y / MMQF_WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int vl[BATCH];
            int vh[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                vl[u] = load_int_ua(row + off_ql);
                vh[u] = load_int_ua(row + off_qh);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                // 4 low bits from `ql`, 2 high bits from `qh`, then the -32
                // bias that turns the unsigned 0..63 value into the signed one
                // the `mma` takes.
                const int lo = (vl[u] & 0x0F0F0F0F) | (((vh[u] >> sh) & 0x03030303) << 4);
                const int hi = ((vl[u] >> 4) & 0x0F0F0F0F)
                               | (((vh[u] >> (sh + 4)) & 0x03030303) << 4);
                s_x[i * X_STRIDE + X_QS + w_lo] = __vsubss4(lo, 0x20202020);
                s_x[i * X_STRIDE + X_QS + w_lo + 16] = __vsubss4(hi, 0x20202020);
            }
        }

        // Scale pass: sixteen per row, so a warp covers two rows. The scale
        // byte and `d` are at row-invariant offsets, so the first loop is pure
        // loads and the multiply happens in the second, on registers.
        float* s_xdf = (float*)s_x;
        const unsigned int j = lane % 16;  // 16-element group within the super-block
        const unsigned int rsub = lane / 16;

        constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 2);
        __half d_h[SROWS];
        int sc[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 2) + warp * 2 + rsub;
            const unsigned char* blk = weight + (feat0 + min(i, i_max)) * rstride + off_blk;
            // Byte 208 is even and the super-block base is 2-byte aligned, so
            // the f16 load is aligned. The scale is a single signed byte, so
            // it carries no alignment requirement of its own.
            d_h[u] = *reinterpret_cast<const __half*>(blk + 208);
            sc[u] = (int)*reinterpret_cast<const signed char*>(blk + 192 + j);
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 2) + warp * 2 + rsub;
            s_xdf[i * X_STRIDE + X_DF + j] = __half2float(d_h[u]) * (float)sc[u];
        }
    }

    // Forwards to `mmqf_vec_dot_sc16`, which is shared with Q3_K: the staged
    // row layout and the per-16 scale arithmetic are the same for both.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_sc16<MMQ_X, FULL, false, X_QS, X_DF, X_STRIDE>(
            s_x, s_y, acc, i0, jb, k00, nks
        );
    }
};

// Q3_K weight format policy, same contract as `MmqfQ80`.
//
// Q3_K is Q6_K's shape with a narrower quant: the same 16-element scale
// granularity, sixteen signed scales per super-block, and no minimum term. So
// everything but `stage` is shared with Q6_K — the staged row layout, the 16
// f32 `d * scale_j` at `X_DF`, and `mmqf_vec_dot_sc16`.
//
// ALIGNMENT. The super-block is 110 bytes, which is only 2-byte aligned, so a
// row base and every super-block base inside it are 2-byte aligned as well.
// Every 4-byte read of `hmask` or `qs` therefore goes through `load_int_ua`;
// a plain `int` load raises CUDA_ERROR_MISALIGNED_ADDRESS, which poisons the
// context for every later launch on it. The packed `scales` are read one byte
// at a time, which carries no alignment requirement, and the `half` `d` at
// byte 108 is 2-byte aligned and is read directly.
//
// THE QUANT. Three bits: two low bits from `qs`, one high bit from `hmask`,
// and the high bit is INVERTED — a SET bit means do NOT subtract 4. The value
// is `low2 - (hmask_bit ? 0 : 4)`, an int8 in [-4, 3], which is what
// `dequant_q3k` in `src/quant/cpu/kernels/dequant_k_quants/q2k_q3k.rs`
// computes. Folding the -4 in during staging is what lets the staged lanes be
// signed and `vec_dot` be Q6_K's unchanged.
struct MmqfQ3K {
    // On-disk super-block: 32 bytes `hmask`, 64 bytes `qs`, 12 packed 6-bit
    // `scales`, then the f16 `d` at byte 108.
    static constexpr int BLOCK_BYTES = 110;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: Q6_K's, int for int. 64 quant words (256 k-values,
    // one signed value in [-4, 3] per int8 lane), then 16 f32 holding
    // `d * scale_j`, one per 16-element group, then padding.
    //
    // The scale is staged as f32 already multiplied by `d`, never as the raw
    // `int8` and never through `half`: half has an 11-bit significand, and
    // rounding `d * scale` to it perturbs every 16-element group's
    // contribution past the bound the GEMM/GEMV parity tests hold this path
    // to.
    static constexpr int X_QS = 0;
    static constexpr int X_DF = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DF + 16 <= X_STRIDE, "Weight row too short: 16 group scales.");
    static_assert(
        X_QS == MmqfQ6K::X_QS && X_DF == MmqfQ6K::X_DF && X_STRIDE == MmqfQ6K::X_STRIDE,
        "Q3_K must stage into the Q6_K row; the two share `mmqf_vec_dot_sc16`."
    );

    // Stages 256 k-values, which for this format is exactly ONE super-block.
    // `b0` counts 32-element Q8_1 activation blocks, so the super-block index
    // is `b0 / 8`.
    //
    // Quant map, transcribed from `load_tiles_q3_K` in llama.cpp's
    // `ggml-cuda/mmq.cuh` and cross-checked against `dequant_q3k` in
    // `src/quant/cpu/kernels/dequant_k_quants/q2k_q3k.rs`. ggml splits a row
    // across 16 threads each emitting 4 words; this family gives a row a whole
    // warp, so lane `l` takes `c = l % 16` and shift level `t = l / 16`, and
    // emits the TWO words for levels `t` and `t + 2`.
    //
    // Lane `l` reads the unaligned int at `qs + 4*c` and the unaligned int at
    // `hmask + 4*(c % 8)`, the latter shifted right by `4 * (c / 8)` so the
    // four `hmask` bits this lane needs sit at bit 0..3 of each byte. Level
    // `t` then takes `qs` bits `2t..2t+1` and `hmask` bit `t`, and the two
    // staged words land at `32*(c/8) + 8*t + c%8` and 16 words further along.
    //
    // Under that map staged word `w` covers elements `4w..4w+3` of the
    // super-block in natural order, exactly as the CPU dequantizer emits them,
    // so element `k` belongs to scale group `k / 16` and staged word `w` takes
    // scale `w / 4` — the property `mmqf_vec_dot_sc16` relies on.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;

        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / (BLOCK_ELEMS / 32);
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;
        const unsigned int sup =
            CLAMP_K ? min(b0 / (BLOCK_ELEMS / 32), supers - 1) : b0 / (BLOCK_ELEMS / 32);
        const unsigned long long off_blk = (unsigned long long)sup * BLOCK_BYTES;

        // Row-invariant byte offsets, so the addressing collapses to one add
        // per row. `hmask` starts at byte 0 of the super-block, `qs` at 32.
        const unsigned int c = lane % 16;   // `qs` int within the super-block
        const unsigned int t = lane / 16;   // shift level, 0 or 1
        const unsigned long long off_qs = off_blk + 32 + c * 4;
        const unsigned long long off_hm = off_blk + (c % 8) * 4;
        const unsigned int hsh = 4 * (c / 8);
        const unsigned int w_lo = 32 * (c / 8) + 8 * t + c % 8;

        constexpr int ROWS = MMQF_Y / MMQF_WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int vq[BATCH];
            int vh[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                vq[u] = load_int_ua(row + off_qs);
                vh[u] = load_int_ua(row + off_hm);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                const int hm = vh[u] >> hsh;
                // Two low bits from `qs`, then the `hmask` bit moved to bit 2
                // so a SET bit contributes +4. The -4 bias that follows turns
                // the pair into the signed [-4, 3] value the `mma` takes: a
                // set bit cancels the bias, a clear one leaves the -4. That
                // inversion is the format's, not a sign convention of this
                // kernel.
                const int lo = ((vq[u] >> (2 * t)) & 0x03030303)
                               | (((hm >> t) << 2) & 0x04040404);
                const int hi = ((vq[u] >> (2 * t + 4)) & 0x03030303)
                               | (((hm >> (t + 2)) << 2) & 0x04040404);
                s_x[i * X_STRIDE + X_QS + w_lo] = __vsubss4(lo, 0x04040404);
                s_x[i * X_STRIDE + X_QS + w_lo + 16] = __vsubss4(hi, 0x04040404);
            }
        }

        // Scale pass: sixteen per row, so a warp covers two rows. The two
        // packed bytes each lane needs and `d` are at row-invariant offsets,
        // so the first loop is pure loads and the 6-bit unpack happens in the
        // second, on registers.
        float* s_xdf = (float*)s_x;
        const unsigned int j = lane % 16;  // 16-element group within the super-block
        const unsigned int rsub = lane / 16;
        const unsigned int o_low = 96 + GGUF_Q3K_SC_LOW_BYTE(j);
        const unsigned int o_high = 96 + GGUF_Q3K_SC_HIGH_BYTE(j);

        constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 2);
        __half d_h[SROWS];
        unsigned int b_low[SROWS];
        unsigned int b_high[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 2) + warp * 2 + rsub;
            const unsigned char* blk = weight + (feat0 + min(i, i_max)) * rstride + off_blk;
            // Byte 108 is even and the super-block base is 2-byte aligned, so
            // the f16 load is aligned. The two scale bytes are single-byte
            // reads and carry no alignment requirement of their own.
            d_h[u] = *reinterpret_cast<const __half*>(blk + 108);
            b_low[u] = blk[o_low];
            b_high[u] = blk[o_high];
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 2) + warp * 2 + rsub;
            // Already biased by -32 by the decoder, so the scale is signed.
            const int sc = q3k_scale_bytes(b_low[u], b_high[u], (int)j);
            s_xdf[i * X_STRIDE + X_DF + j] = __half2float(d_h[u]) * (float)sc;
        }
    }

    // Forwards to `mmqf_vec_dot_sc16`, shared with Q6_K: once the -4 bias is
    // folded in during staging the two rows are indistinguishable.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_sc16<MMQ_X, FULL, false, X_QS, X_DF, X_STRIDE>(
            s_x, s_y, acc, i0, jb, k00, nks
        );
    }
};

// Q5_K weight format policy, same contract as `MmqfQ80`.
//
// Q5_K is Q4_K with a fifth bit. Same super-block header (f16 `d`, f16
// `dmin`, the same 12-byte 6-bit scale/min packing), same eight 32-element
// sub-blocks, same dequant form `d * sc_j * q - dmin * m_j`, same sub-block
// PAIRING over `qs` — even sub-block takes the low nibbles, odd the high
// nibbles of the SAME 32-byte run. Only the quant is wider: unsigned 5-bit,
// four low bits from `qs` plus one bit from the 32-byte `qh` field.
//
// So everything but `stage` is shared with Q4_K: the staged row layout, the
// `(d * sc, -dmin * m)` f32 pairs at `X_DM`, and `mmqf_vec_dot_dm`.
//
// ALIGNMENT. 176 is a multiple of 16, so a row base and every super-block
// base inside it are 16-byte aligned, and `qs`@48, `qh`@16 and the `d`/`dmin`
// pair@0 are all 4-byte aligned. Plain `int` loads are legal here; unlike
// Q6_K this format needs no `load_int_ua`.
struct MmqfQ5K {
    // On-disk super-block: f16 d, f16 dmin, 12 packed 6-bit scale/min bytes,
    // 32 bytes of fifth bits, then 128 nibble-packed quants.
    static constexpr int BLOCK_BYTES = 176;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole. The ragged tail path is dead for this format and is not compiled.
    static constexpr bool RAGGED_K = false;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: identical to Q4_K's — 64 quant words (256 k-values,
    // one unsigned 0..31 value per int8 lane), then 8 `float2` holding
    // `(d * sc_j, -dmin * m_j)`, then padding. Equal stride means Q4_K, Q5_K
    // and Q6_K share the family's shared-memory request exactly.
    static constexpr int X_QS = 0;
    static constexpr int X_DM = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DM + 16 <= X_STRIDE, "Weight row too short: 8 scale/min pairs.");
    static_assert(X_DM % 2 == 0 && X_STRIDE % 2 == 0, "Scale/min pairs are misaligned.");

    // Stages 256 k-values, which for this format is exactly ONE super-block.
    // `b0` counts 32-element Q8_1 activation blocks, so the super-block index
    // is `b0 / 8`.
    //
    // Quant map, transcribed from `load_tiles_q5_K` in llama.cpp's
    // `ggml-cuda/mmq.cuh`. Write `lane = 8*a + b` with `a = lane / 8` and
    // `b = lane % 8`. Lane `l` reads the aligned int at `qs + 4*l`, whose low
    // nibbles are 4 k-values of the EVEN sub-block `2*a` and whose high
    // nibbles are the same 4 positions of the ODD sub-block `2*a + 1` — the
    // pair shares one 32-byte run of `qs`, exactly as in Q4_K. The two staged
    // words land at `16*a + b` and 8 further along.
    //
    // The fifth bits come from the aligned int at `qh + 4*b`, which is `qh`
    // bytes `4*b .. 4*b+3`. In `qh` the BYTE index is the element within a
    // sub-block and the BIT index is the sub-block number, so the even word
    // takes bit `2*a` of each byte and the odd word bit `2*a + 1`.
    //
    // Both halves of that map were cross-checked element-by-element against
    // `dequant_q5k` in `src/quant/cpu/kernels/dequant_k_quants/q4k_q5k.rs`:
    // staged word `16*a + b` covers k-values `64*a + 4*b .. +3`, which is
    // sub-block `2*a`, elements `4*b .. 4*b+3` — the CPU kernel reads
    // `qs[(j/2)*32 + l]` and `qh[l] >> j` at exactly those indices. Getting
    // either index wrong does not fail loudly; it yields a tensor with the
    // right shape and RMS and the wrong values.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;

        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / (BLOCK_ELEMS / 32);
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;
        const unsigned int sup =
            CLAMP_K ? min(b0 / (BLOCK_ELEMS / 32), supers - 1) : b0 / (BLOCK_ELEMS / 32);
        const unsigned long long off_blk = (unsigned long long)sup * BLOCK_BYTES;

        // Row-invariant byte offsets, so the addressing collapses to one add
        // per row. `qh` starts at byte 16 of the super-block, `qs` at byte 48.
        const unsigned long long off_qs = off_blk + 48 + lane * 4;
        const unsigned long long off_qh = off_blk + 16 + (lane % 8) * 4;
        const unsigned int w_lo = 16 * (lane / 8) + lane % 8;
        const unsigned int sh = 2 * (lane / 8);  // 0, 2, 4 or 6

        constexpr int ROWS = MMQF_Y / MMQF_WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int vl[BATCH];
            int vh[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                vl[u] = *reinterpret_cast<const int*>(row + off_qs);
                vh[u] = *reinterpret_cast<const int*>(row + off_qh);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                // 4 low bits from `qs`, the fifth from `qh`. Unsigned 0..31
                // sits inside the signed int8 range the `mma` takes, so no
                // bias is applied; the minimum term is what carries the
                // format's asymmetry, exactly as in Q4_K.
                const int lo =
                    (vl[u] & 0x0F0F0F0F) | (((vh[u] >> sh) & 0x01010101) << 4);
                const int hi =
                    ((vl[u] >> 4) & 0x0F0F0F0F) | (((vh[u] >> (sh + 1)) & 0x01010101) << 4);
                s_x[i * X_STRIDE + X_QS + w_lo] = lo;
                s_x[i * X_STRIDE + X_QS + w_lo + 8] = hi;
            }
        }

        // Scale/min pass: byte-for-byte the same as Q4_K's, because the
        // header is the same. Eight pairs per row, so a warp covers four
        // rows. The three packed bytes each lane needs are at row-invariant
        // offsets, so the first loop is pure loads and the 6-bit unpack
        // happens in the second, on registers.
        float2* s_xdm = (float2*)s_x;
        const unsigned int j = lane % 8;  // sub-block within the super-block
        const unsigned int rsub = lane / 8;
        const unsigned int o_j4 = j + 4;
        // For j < 4 the third byte is unused; re-reading sc[j] keeps the index
        // in range without a branch around the load.
        const unsigned int o_jm4 = j < 4 ? j : j - 4;

        constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 4);
        half2 dm[SROWS];
        unsigned int b_j[SROWS];
        unsigned int b_j4[SROWS];
        unsigned int b_jm4[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            const unsigned char* blk = weight + (feat0 + min(i, i_max)) * rstride + off_blk;
            // `d` and `dmin` are adjacent f16 at byte 0 of a 16-aligned block,
            // so the pair is one aligned 4-byte load.
            dm[u] = *reinterpret_cast<const half2*>(blk);
            const unsigned char* sc = blk + 4;  // 12 packed 6-bit scale/min bytes
            b_j[u] = sc[j];
            b_j4[u] = sc[o_j4];
            b_jm4[u] = sc[o_jm4];
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            int scale;
            int minimum;
            q4k_scale_min_bytes(b_j[u], b_j4[u], b_jm4[u], (int)j, &scale, &minimum);
            const float d = __low2float(dm[u]);
            const float dmin = __high2float(dm[u]);
            // Stored as f32: no half round-trip on `d * sc`, which is the
            // term the parity bound is sensitive to.
            s_xdm[(i * X_STRIDE + X_DM) / 2 + j] =
                make_float2(d * (float)scale, -dmin * (float)minimum);
        }
    }

    // Forwards to `mmqf_vec_dot_dm`, shared with Q4_K: once staged, the two
    // formats' rows and arithmetic are identical.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_dm<MMQ_X, FULL, X_QS, X_DM, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};

// Q4_1 weight format policy, same contract as `MmqfQ80`.
//
// Q4_1 is a legacy 32-element format, not a K-quant: `f16 d`@0, `f16 m`@2,
// then 16 bytes holding 32 unsigned 4-bit quants@4. The value is `d * q + m`.
//
// THE MIN SIGN. Q4_1 stages into Q4_K's row and shares `mmqf_vec_dot_dm`,
// which adds `pair.y * (activation scale * exact int16 block sum)`. Q4_K's
// value is `d * sc * q - dmin * m`, so Q4_K stores `-dmin * m_j` in `pair.y`.
// Q4_1 is ADDITIVE, so it stores `+m` there. Same `vec_dot`, opposite sign
// folded at staging — this is the one place in the family where the stored
// minimum is not negated.
//
// One `(d, m)` pair covers a whole 32-element block, so a 256-k staging group
// carries exactly 8 pairs, which is the granularity `mmqf_vec_dot_dm` already
// indexes with `kw / 8`.
//
// ALIGNMENT. The block is 20 bytes, a multiple of 4, so a row base
// (`bpr * 20`) and every block base inside it are 4-byte aligned, and `qs`@4
// and the `d`/`m` pair@0 land on multiples of 4. Plain `int` and `half2` loads
// are legal; unlike Q4_0 and Q5_0 this format needs no `load_int_ua`.
struct MmqfQ41 {
    // On-disk block: f16 scale, f16 minimum, 16 nibble-packed quant bytes.
    static constexpr int BLOCK_BYTES = 20;
    static constexpr int BLOCK_ELEMS = 32;
    // K is gated only on `k % 32 == 0`, so a row's last 256-k group can hold
    // fewer than MMQF_ITER_B blocks and the tail path must be compiled.
    static constexpr bool RAGGED_K = true;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: Q4_K's, int for int. 64 quant words (256 k-values,
    // one unsigned 0..15 value per int8 lane), then 8 `float2` holding
    // `(d, m)`, then padding.
    static constexpr int X_QS = 0;
    static constexpr int X_DM = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DM + 16 <= X_STRIDE, "Weight row too short: 8 scale/min pairs.");
    static_assert(X_DM % 2 == 0 && X_STRIDE % 2 == 0, "Scale/min pairs are misaligned.");
    static_assert(
        X_QS == MmqfQ4K::X_QS && X_DM == MmqfQ4K::X_DM && X_STRIDE == MmqfQ4K::X_STRIDE,
        "Q4_1 must stage into the Q4_K row; the two share `mmqf_vec_dot_dm`."
    );

    // Stages 256 k-values (8 blocks) of the weight tile, on Q4_0's lane map: a
    // block holds 4 source ints and produces 8 staged words, so the warp's 32
    // lanes cover a whole 256-k group with ONE global load each — lane maps to
    // (block `lane / 4`, 4-k word `lane % 4`). Independent of MMQ_X.
    //
    // Quant map, transcribed from `load_tiles_q4_1` in llama.cpp's
    // `ggml-cuda/mmq.cuh` and cross-checked element by element against
    // `dequant_q4_1` in `src/quant/cpu/kernels/dequant_simple.rs`: within one
    // 32-element block, element `j` (0..15) is the LOW nibble of `qs[j]` and
    // element `j + 16` is the HIGH nibble of the same byte. So the int at
    // `qs + 4*w` (w = 0..3) carries elements `4w..4w+3` in its low nibbles and
    // `4w+16..4w+19` in its high nibbles, placing the two staged words at
    // `8*blk + w` and 4 further along. This is Q4_0's map, NOT Q4_K's: a
    // legacy block is a single 32-element run, while Q4_K pairs sub-blocks
    // across a shared 32-byte run.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;
        const unsigned int kbx = lane / 4;   // block within the 256-k group
        const unsigned int kqsx = lane % 4;  // 4-k source word within that block

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index; the
        // tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        // `bpr` counts 32-element activation blocks and `b0` indexes them.
        // Q4_1's weight block is also 32 elements, so the counts coincide here;
        // a 256-element format divides by `BLOCK_ELEMS / 32` first.
        const unsigned long long rstride = (unsigned long long)bpr * BLOCK_BYTES;

        // Loop-invariant, so the addressing collapses to one add per row.
        const unsigned int blk = CLAMP_K ? min(b0 + kbx, bpr - 1) : b0 + kbx;
        const unsigned long long off_qs = (unsigned long long)blk * BLOCK_BYTES + 4 + kqsx * 4;
        // Both staged words live in the same 256-k group: words 0..31 are the
        // first 128-k half, 32..63 the second.
        const unsigned int w_lo = kbx * 8 + kqsx;

        constexpr int ROWS = MMQF_Y / MMQF_WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int v[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                v[u] = *reinterpret_cast<const int*>(row + off_qs);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                // Unsigned 0..15 sits inside the signed int8 range the `mma`
                // takes, so no bias is applied; the minimum term below is what
                // carries the format's asymmetry, exactly as in Q4_K.
                s_x[i * X_STRIDE + X_QS + w_lo] = v[u] & 0x0F0F0F0F;
                s_x[i * X_STRIDE + X_QS + w_lo + 4] = (v[u] >> 4) & 0x0F0F0F0F;
            }
        }

        // Scale/min pass: eight pairs per row, so a warp covers four rows.
        float2* s_xdm = (float2*)s_x;
        const unsigned int kbxd = lane % 8;
        const unsigned int rsub = lane / 8;
        const unsigned int blk_d = CLAMP_K ? min(b0 + kbxd, bpr - 1) : b0 + kbxd;
        const unsigned long long off_d = (unsigned long long)blk_d * BLOCK_BYTES;

        constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 4);
        half2 dm[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            const unsigned char* blkp = weight + (feat0 + min(i, i_max)) * rstride + off_d;
            // `d` and `m` are adjacent f16 at byte 0 of a 4-aligned block, so
            // the pair is one aligned 4-byte load.
            dm[u] = *reinterpret_cast<const half2*>(blkp);
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            // `+m`, not `-m`: see THE MIN SIGN above. Stored as f32 for the
            // same parity reason as Q4_K — no half round-trip on the term the
            // GEMM/GEMV bound is sensitive to.
            s_xdm[(i * X_STRIDE + X_DM) / 2 + kbxd] =
                make_float2(__low2float(dm[u]), __high2float(dm[u]));
        }
    }

    // Forwards to `mmqf_vec_dot_dm`, shared with Q4_K and Q5_K: once staged,
    // the rows and the two-term arithmetic are identical.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_dm<MMQ_X, FULL, X_QS, X_DM, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};

// Q5_1 weight format policy, same contract as `MmqfQ80`.
//
// Q5_1 is Q4_1 with a fifth quant bit: `f16 d`@0, `f16 m`@2, a 32-bit `qh`@4
// holding one bit per element, then 16 low-nibble bytes@8. The value is
// `d * q + m` with `q` the unsigned 5-bit assembly, so the staged row, the
// `+m` sign and `mmqf_vec_dot_dm` are Q4_1's and only the quant assembly
// differs — the same relationship Q5_K has to Q4_K.
//
// ALIGNMENT. The block is 24 bytes, a multiple of 4, so a row base
// (`bpr * 24`) and every block base inside it are 4-byte aligned, and `qs`@8,
// `qh`@4 and the `d`/`m` pair@0 land on multiples of 4. Plain `int` and
// `half2` loads are legal; unlike Q5_0 this format needs no `load_int_ua`.
struct MmqfQ51 {
    // On-disk block: f16 scale, f16 minimum, 32-bit fifth-bit field, 16
    // nibble bytes.
    static constexpr int BLOCK_BYTES = 24;
    static constexpr int BLOCK_ELEMS = 32;
    // K is gated only on `k % 32 == 0`, so a row's last 256-k group can hold
    // fewer than MMQF_ITER_B blocks and the tail path must be compiled.
    static constexpr bool RAGGED_K = true;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: Q4_1's, int for int — 64 quant words (256 k-values,
    // one unsigned 0..31 value per int8 lane), then 8 `float2` holding
    // `(d, m)`, then padding.
    static constexpr int X_QS = 0;
    static constexpr int X_DM = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DM + 16 <= X_STRIDE, "Weight row too short: 8 scale/min pairs.");
    static_assert(X_DM % 2 == 0 && X_STRIDE % 2 == 0, "Scale/min pairs are misaligned.");
    static_assert(
        X_QS == MmqfQ41::X_QS && X_DM == MmqfQ41::X_DM && X_STRIDE == MmqfQ41::X_STRIDE,
        "Q5_1 must stage into the Q4_1 row; the two share `mmqf_vec_dot_dm`."
    );

    // Stages 256 k-values (8 blocks), on Q4_1's lane map: lane maps to
    // (block `lane / 4`, 4-k word `lane % 4`), one `qs` load and one `qh`
    // load each. Independent of MMQ_X.
    //
    // Quant map, transcribed from `load_tiles_q5_1` in llama.cpp's
    // `ggml-cuda/mmq.cuh` and cross-checked element by element against
    // `dequant_q5_1` in `src/quant/cpu/kernels/dequant_simple.rs`. The nibble
    // half is Q4_1's: the int at `qs + 4*w` carries elements `4w..4w+3` in its
    // low nibbles and `4w+16..4w+19` in its high nibbles, so the two staged
    // words land at `8*blk + w` and 4 further along.
    //
    // The fifth bit of element `j` is bit `j` of the block's single 32-bit
    // `qh`, so this lane needs bits `4w..4w+3` and `4w+16..4w+19`.
    // Pre-shifting by `4*w` leaves those in bits 0..3 and 16..19, and the four
    // masked shifts move bit `t` of that into bit 4 of staged byte `t`. That
    // is upstream's form, kept verbatim: the source bits are packed
    // contiguously inside one nibble rather than spread one per byte, so
    // Q5_K's single `(qh >> sh) & 0x01010101` trick does not apply here.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;
        const unsigned int kbx = lane / 4;   // block within the 256-k group
        const unsigned int kqsx = lane % 4;  // 4-k source word within that block

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index; the
        // tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        // `bpr` counts 32-element activation blocks and `b0` indexes them.
        // Q5_1's weight block is also 32 elements, so the counts coincide here;
        // a 256-element format divides by `BLOCK_ELEMS / 32` first.
        const unsigned long long rstride = (unsigned long long)bpr * BLOCK_BYTES;

        // Loop-invariant, so the addressing collapses to one add per row.
        const unsigned int blk = CLAMP_K ? min(b0 + kbx, bpr - 1) : b0 + kbx;
        const unsigned long long off_qs = (unsigned long long)blk * BLOCK_BYTES + 8 + kqsx * 4;
        const unsigned long long off_qh = (unsigned long long)blk * BLOCK_BYTES + 4;
        const unsigned int w_lo = kbx * 8 + kqsx;
        const unsigned int shq = 4 * kqsx;

        constexpr int ROWS = MMQF_Y / MMQF_WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int vl[BATCH];
            int vh[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                vl[u] = *reinterpret_cast<const int*>(row + off_qs);
                vh[u] = *reinterpret_cast<const int*>(row + off_qh);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                const int qh = (int)((unsigned int)vh[u] >> shq);
                // 4 low bits from `qs`, the fifth from `qh`. Unsigned 0..31
                // sits inside the signed int8 range the `mma` takes, so no
                // bias is applied; the minimum term carries the asymmetry.
                int lo = vl[u] & 0x0F0F0F0F;
                lo |= (qh << 4) & 0x00000010;   // element 4w+0 -> byte 0 bit 4
                lo |= (qh << 11) & 0x00001000;  // element 4w+1 -> byte 1 bit 4
                lo |= (qh << 18) & 0x00100000;  // element 4w+2 -> byte 2 bit 4
                lo |= (qh << 25) & 0x10000000;  // element 4w+3 -> byte 3 bit 4
                int hi = (vl[u] >> 4) & 0x0F0F0F0F;
                hi |= (qh >> 12) & 0x00000010;  // element 4w+16 -> byte 0 bit 4
                hi |= (qh >> 5) & 0x00001000;   // element 4w+17 -> byte 1 bit 4
                hi |= (qh << 2) & 0x00100000;   // element 4w+18 -> byte 2 bit 4
                hi |= (qh << 9) & 0x10000000;   // element 4w+19 -> byte 3 bit 4
                s_x[i * X_STRIDE + X_QS + w_lo] = lo;
                s_x[i * X_STRIDE + X_QS + w_lo + 4] = hi;
            }
        }

        // Scale/min pass: byte-for-byte Q4_1's, because the header is the
        // same two f16 at byte 0. Eight pairs per row, so a warp covers four
        // rows.
        float2* s_xdm = (float2*)s_x;
        const unsigned int kbxd = lane % 8;
        const unsigned int rsub = lane / 8;
        const unsigned int blk_d = CLAMP_K ? min(b0 + kbxd, bpr - 1) : b0 + kbxd;
        const unsigned long long off_d = (unsigned long long)blk_d * BLOCK_BYTES;

        constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 4);
        half2 dm[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            const unsigned char* blkp = weight + (feat0 + min(i, i_max)) * rstride + off_d;
            dm[u] = *reinterpret_cast<const half2*>(blkp);
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            // `+m`, not `-m`: Q5_1 is additive, see THE MIN SIGN on `MmqfQ41`.
            s_xdm[(i * X_STRIDE + X_DM) / 2 + kbxd] =
                make_float2(__low2float(dm[u]), __high2float(dm[u]));
        }
    }

    // Forwards to `mmqf_vec_dot_dm`, shared with Q4_1, Q4_K and Q5_K.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_dm<MMQ_X, FULL, X_QS, X_DM, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};

// Q2_K weight format policy, same contract as `MmqfQ80`.
//
// The hardest format in the family: it is the only one with BOTH a per-16
// scale granularity and a minimum term. So it takes Q6_K/Q3_K's two 16-k MMAs
// per 32-k step AND Q4_K/Q5_K's rank-1 minimum correction, at twice Q4_K's
// granularity. `mmqf_vec_dot_sc16` carries both under `MIN = true`.
//
// ALIGNMENT. 84 is a multiple of 4, so a row base (`supers * 84`) and every
// super-block base inside it are 4-byte aligned, and `scales`@0, `qs`@16 and
// the `d`/`dmin` pair@80 all land on multiples of 4. Plain `int` and `half2`
// loads are legal; unlike Q6_K and Q3_K this format needs no `load_int_ua`.
//
// THE QUANT. Two bits, UNSIGNED, in 0..3 — no bias is folded in. The format's
// asymmetry is carried entirely by the minimum term, as in Q4_K and Q5_K.
//
// THE MINIMUM. `dequant_q2k` computes `d * (sc[i] & 0x0F) * q - dmin *
// (sc[i] >> 4)` with `i = k / 16`. The minimum does not depend on `q` at all,
// so over a k-block it is rank-1 in the activation's quant SUM, exactly as in
// Q4_K — but over 16 elements, not 32. The activation record's header sum is
// per 32, so `mmqf_stage_y_sums` derives the split once per staged activation
// tile; see that function for why the derived half is exact and why an
// all-ones `mma` is not used.
struct MmqfQ2K {
    // On-disk super-block: 16 packed scale/min bytes, 64 bytes of 2-bit
    // quants, then f16 `d` and f16 `dmin`.
    static constexpr int BLOCK_BYTES = 84;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // The minimum changes every 16 elements, so `vec_dot` needs a per-16
    // activation sum the shared record does not carry.
    static constexpr int Y_SCRATCH = MMQF_Y_SCRATCH;

    // Staged weight row: 64 quant words (256 k-values, one unsigned 0..3 value
    // per int8 lane), then 16 `float2` holding `(d * sc_j, -dmin * m_j)`, one
    // per 16-element group, then padding. That is 32 ints of scale record —
    // twice Q4_K's, because the granularity is twice as fine — which makes
    // this the widest row in the family.
    //
    // The pair is staged as f32 already multiplied by `d`/`dmin`, never
    // through `half`: half has an 11-bit significand, and rounding `d * sc` to
    // it perturbs every 16-element group's contribution past the bound the
    // GEMM/GEMV parity tests hold this path to.
    static constexpr int X_QS = 0;
    static constexpr int X_DM = 64;
    static constexpr int X_STRIDE = 100;
    static_assert(X_DM + 32 <= X_STRIDE, "Weight row too short: 16 scale/min pairs.");
    static_assert(X_DM % 2 == 0 && X_STRIDE % 2 == 0, "Scale/min pairs are misaligned.");

    // Stages 256 k-values, which for this format is exactly ONE super-block.
    // `b0` counts 32-element Q8_1 activation blocks, so the super-block index
    // is `b0 / 8`.
    //
    // Quant map, derived from `dequant_q2k` in
    // `src/quant/cpu/kernels/dequant_k_quants/q2k_q3k.rs` and structurally the
    // same as Q3_K's. That kernel emits element
    // `y = 128*n + 32*t + 16*h + l` from `qs[32*n + 16*h + l] >> 2*t`, with
    // scale index `y / 16`. Four consecutive elements therefore come from the
    // aligned int at `qs + 4*c` with `c = 8*n + 4*h + m`, and land at staged
    // word `32*(c/8) + 8*t + (c%8)`.
    //
    // This family gives a row a whole warp, so lane `l` takes `c = l % 16` and
    // shift level `t = l / 16`, and emits the TWO words for levels `t` and
    // `t + 2`, the second 16 words further along. Under that map staged word
    // `w` covers elements `4w .. 4w+3` in natural order, so staged word `w`
    // takes scale group `w / 4` — the property `mmqf_vec_dot_sc16` relies on.
    // The map was simulated element-by-element against the CPU dequantizer
    // over a random super-block; a wrong index does not fail loudly, it yields
    // a tensor with the right shape and RMS and the wrong values.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;

        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / (BLOCK_ELEMS / 32);
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;
        const unsigned int sup =
            CLAMP_K ? min(b0 / (BLOCK_ELEMS / 32), supers - 1) : b0 / (BLOCK_ELEMS / 32);
        const unsigned long long off_blk = (unsigned long long)sup * BLOCK_BYTES;

        // Row-invariant byte offsets, so the addressing collapses to one add
        // per row. `scales` starts at byte 0 of the super-block, `qs` at 16.
        const unsigned int c = lane % 16;  // `qs` int within the super-block
        const unsigned int t = lane / 16;  // shift level, 0 or 1
        const unsigned long long off_qs = off_blk + 16 + c * 4;
        const unsigned int w_lo = 32 * (c / 8) + 8 * t + c % 8;

        constexpr int ROWS = MMQF_Y / MMQF_WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int vq[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                vq[u] = *reinterpret_cast<const int*>(row + off_qs);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                // Unsigned 0..3 sits inside the signed int8 range the `mma`
                // takes, so no bias is applied.
                s_x[i * X_STRIDE + X_QS + w_lo] = (vq[u] >> (2 * t)) & 0x03030303;
                s_x[i * X_STRIDE + X_QS + w_lo + 16] = (vq[u] >> (2 * t + 4)) & 0x03030303;
            }
        }

        // Scale/min pass: sixteen pairs per row, so a warp covers two rows.
        // One scale byte carries both fields — low nibble the scale, high
        // nibble the minimum — so this format needs no 6-bit unpack. The byte
        // and the `d`/`dmin` pair are at row-invariant offsets, so the first
        // loop is pure loads and the multiplies happen in the second, on
        // registers.
        float2* s_xdm = (float2*)s_x;
        const unsigned int j = lane % 16;  // 16-element group within the super-block
        const unsigned int rsub = lane / 16;

        constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 2);
        half2 dm[SROWS];
        unsigned int sc[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 2) + warp * 2 + rsub;
            const unsigned char* blk = weight + (feat0 + min(i, i_max)) * rstride + off_blk;
            // Byte 80 is a multiple of 4 and the super-block base is 4-byte
            // aligned, so the adjacent `d`/`dmin` f16 are one aligned 4-byte
            // load. The scale byte carries no alignment requirement.
            dm[u] = *reinterpret_cast<const half2*>(blk + 80);
            sc[u] = blk[j];
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 2) + warp * 2 + rsub;
            const float d = __low2float(dm[u]);
            const float dmin = __high2float(dm[u]);
            // The `-1` on the minimum is folded in here so the consumer is a
            // plain multiply-add.
            s_xdm[(i * X_STRIDE + X_DM) / 2 + j] =
                make_float2(d * (float)(sc[u] & 0x0Fu), -dmin * (float)(sc[u] >> 4));
        }
    }

    // Forwards to `mmqf_vec_dot_sc16` with the minimum term switched on. Q6_K
    // and Q3_K instantiate the same body with `MIN = false`, which drops both
    // the pair load and the correction.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_sc16<MMQ_X, FULL, true, X_QS, X_DM, X_STRIDE>(
            s_x, s_y, acc, i0, jb, k00, nks
        );
    }
};

// IQ4_NL weight format policy, same contract as `MmqfQ80`.
//
// The family's first i-quant. IQ4_NL is a legacy 32-element format like Q4_0 —
// `f16 d`@0 then 16 bytes holding 32 4-bit fields@2 — but the field is an
// INDEX into the 16-entry signed codebook `KVALUES_IQ4NL`, not a magnitude.
// Reading it as a magnitude produces finite, plausibly scaled output and no
// error, which is why the codebook lives in exactly one place (`decode.cuh`)
// for every CUDA kernel.
//
// Because the codebook values are already signed int8, the staged row is
// Q8_0's unchanged — signed quants in the int8 lanes plus one f32 scale per
// 32-element block — so `vec_dot` is Q8_0's and `stage` is the only thing this
// format defines for itself. There is no bias to fold in; the table lookup
// replaces the `__vsubss4` Q4_0 does at the same point.
//
// ALIGNMENT. The block is 18 bytes, so a row base (`bpr * 18`) and every block
// base inside it are only 2-byte aligned. Every 4-byte read of `qs` therefore
// goes through `load_int_ua`; a plain `int` load raises
// CUDA_ERROR_MISALIGNED_ADDRESS, which poisons the context for every later
// launch on it. The `f16 d` at byte 0 is 2-byte aligned, which is
// `alignof(__half)`, and is read directly.
struct MmqfIQ4NL {
    // On-disk block: one f16 scale then 16 bytes of nibble-packed codebook
    // indices.
    static constexpr int BLOCK_BYTES = 18;
    static constexpr int BLOCK_ELEMS = 32;
    // K is gated only on `k % 32 == 0`, so a row's last 256-k group can hold
    // fewer than MMQF_ITER_B blocks and the tail path must be compiled.
    static constexpr bool RAGGED_K = true;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: Q8_0's, byte for byte. 64 quant words (256 k-values),
    // then 8 f32 block scales, then padding that makes the stride an odd
    // multiple of 4 ints so the strided fragment gathers hit all 32 banks.
    static constexpr int X_QS = 0;
    static constexpr int X_DS = 64;
    static constexpr int X_STRIDE = 76;
    static_assert(X_DS + 8 <= X_STRIDE, "Weight row too short: 8 block scales.");
    static_assert(
        X_QS == MmqfQ80::X_QS && X_DS == MmqfQ80::X_DS && X_STRIDE == MmqfQ80::X_STRIDE,
        "IQ4_NL must stage into the Q8_0 row; the two share `mmqf_vec_dot_d`."
    );

    // Stages 256 k-values (8 blocks) of the weight tile. A block holds 4
    // source ints and produces 8 staged words, so the warp's 32 lanes cover a
    // whole 256-k group with ONE global load each: lane maps to
    // (block `lane / 4`, 4-k word `lane % 4`). Independent of MMQ_X.
    //
    // Quant map, transcribed from `load_tiles_iq4_nl` in llama.cpp's
    // `ggml-cuda/mmq.cuh` and cross-checked against `dequant_iq4_nl` in
    // `src/quant/cpu/kernels/dequant_iq4.rs`: it is Q4_0's map, because IQ4_NL
    // uses the same split-half nibble order. Within one 32-element block,
    // element `j` (0..15) is the LOW nibble of `qs[j]` and element `j + 16` is
    // the HIGH nibble of the same byte. So the int at `qs + 4*w` (w = 0..3)
    // carries elements `4w..4w+3` in its low nibbles and `4w+16..4w+19` in its
    // high nibbles, which places the two staged words at `8*blk + w` and 4
    // further along.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;
        const unsigned int kbx = lane / 4;   // block within the 256-k group
        const unsigned int kqsx = lane % 4;  // 4-k source word within that block

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index; the
        // tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        // `bpr` counts 32-element activation blocks and `b0` indexes them.
        // IQ4_NL's weight block is also 32 elements, so the counts coincide
        // here; a 256-element format divides by `BLOCK_ELEMS / 32` first.
        const unsigned long long rstride = (unsigned long long)bpr * BLOCK_BYTES;

        // Loop-invariant, so the addressing collapses to one add per row.
        const unsigned int blk = CLAMP_K ? min(b0 + kbx, bpr - 1) : b0 + kbx;
        // The quants start at byte 2 of an 18-byte block, so only 2-byte
        // alignment holds and each word is two 16-bit loads.
        const unsigned long long off_qs = (unsigned long long)blk * BLOCK_BYTES + 2 + kqsx * 4;
        // Both staged words live in the same 256-k group: words 0..31 are the
        // first 128-k half, 32..63 the second.
        const unsigned int w_lo = kbx * 8 + kqsx;

        constexpr int ROWS = MMQF_Y / MMQF_WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int v[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                v[u] = load_int_ua(row + off_qs);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                // The codebook lookup sits in the SECOND loop, so it never
                // separates a global load from the next load's issue. Table
                // values are signed int8 already, so the staged lanes need no
                // bias and `vec_dot` is Q8_0's unchanged.
                int q_lo;
                int q_hi;
                gguf_iq4_table_lookup(v[u], &q_lo, &q_hi);
                s_x[i * X_STRIDE + X_QS + w_lo] = q_lo;
                s_x[i * X_STRIDE + X_QS + w_lo + 4] = q_hi;
            }
        }

        // Scales are a separate pass: eight per row, so a warp covers four rows.
        float* s_xd = (float*)s_x;
        const unsigned int kbxd = lane % 8;
        const unsigned int rsub = lane / 8;
        const unsigned int blk_d = CLAMP_K ? min(b0 + kbxd, bpr - 1) : b0 + kbxd;
        const unsigned long long off_d = (unsigned long long)blk_d * BLOCK_BYTES;

        constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 4);
        float d[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            const unsigned char* blkp = weight + (feat0 + min(i, i_max)) * rstride + off_d;
            d[u] = __half2float(*reinterpret_cast<const __half*>(blkp));
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            s_xd[i * X_STRIDE + X_DS + kbxd] = d[u];
        }
    }

    // Forwards to `mmqf_vec_dot_d`, shared with Q8_0: once the table lookup
    // has run during staging the two rows are indistinguishable.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_d<MMQ_X, FULL, X_QS, X_DS, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};

// IQ4_XS weight format policy, same contract as `MmqfQ80`.
//
// IQ4_XS is IQ4_NL's codebook over a 256-element super-block: `f16 d`@0,
// `u16 scales_h`@2, `u8 scales_l[4]`@4, `u8 qs[128]`@8. Eight sub-blocks of 32
// elements each take a 6-bit scale assembled from a `scales_l` nibble and two
// `scales_h` bits, biased by 32: `dl = d * (ls - 32)`. `scales_h` carries the
// high bits of ALL EIGHT sub-blocks (2 bits each = 16), so reading it as one
// byte drops half of them and shifts `scales_l` by one.
//
// A staged 256-k group is exactly one super-block and its scale granularity is
// 32 elements, which is the granularity `mmqf_vec_dot_d` already indexes at.
// With the codebook values signed int8, the staged row is Q8_0's unchanged and
// so is `vec_dot`; only `stage` is this format's own.
//
// The scale is staged as f32 already multiplied by `d`, never through `half`:
// `half` has an 11-bit significand and rounding `d * (ls - 32)` to it perturbs
// every sub-block's contribution past the bound the GEMM/GEMV parity tests
// hold this path to.
//
// ALIGNMENT. The super-block is 136 bytes, a multiple of 8, so a row base
// (`supers * 136`) and every super-block base inside it are 8-byte aligned.
// `qs`@8 is therefore 4-byte aligned and read with plain `int` loads;
// `scales_h`@2 is 2-byte aligned, which is `alignof(unsigned short)`, and the
// `f16 d`@0 is 2-byte aligned, which is `alignof(__half)`. The `scales_l`
// bytes carry no alignment requirement of their own.
struct MmqfIQ4XS {
    // On-disk super-block: f16 `d`@0, u16 `scales_h`@2, 4 `scales_l`@4, then
    // 128 bytes of nibble-packed codebook indices@8.
    static constexpr int BLOCK_BYTES = 136;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: Q8_0's, byte for byte. 64 quant words (256 k-values),
    // then 8 f32 sub-block scales, then padding that makes the stride an odd
    // multiple of 4 ints so the strided fragment gathers hit all 32 banks.
    static constexpr int X_QS = 0;
    static constexpr int X_DS = 64;
    static constexpr int X_STRIDE = 76;
    static_assert(X_DS + 8 <= X_STRIDE, "Weight row too short: 8 sub-block scales.");
    static_assert(
        X_QS == MmqfQ80::X_QS && X_DS == MmqfQ80::X_DS && X_STRIDE == MmqfQ80::X_STRIDE,
        "IQ4_XS must stage into the Q8_0 row; the two share `mmqf_vec_dot_d`."
    );

    // Stages 256 k-values, which for this format is exactly ONE super-block.
    // `b0` counts 32-element Q8_1 activation blocks, so the super-block index
    // is `b0 / 8`. `qs` holds 32 ints and the warp has 32 lanes, so each lane
    // issues ONE global load and produces two staged words. Independent of
    // MMQ_X.
    //
    // Quant map, transcribed from `load_tiles_iq4_xs` in llama.cpp's
    // `ggml-cuda/mmq.cuh` and cross-checked against `dequant_iq4_xs` in
    // `src/quant/cpu/kernels/dequant_iq4.rs`. Lane `l` reads the int at
    // `qs + 4*l`, which is bytes `4*(l % 4) .. 4*(l % 4) + 3` of sub-block
    // `l / 4`. Split-half nibble order puts its low nibbles at elements
    // `32*(l/4) + 4*(l%4) ..+3` and its high nibbles 16 elements further on,
    // so the staged words are `8*(l/4) + l%4` and 4 beyond. Under that map a
    // staged word `w` belongs to sub-block `w / 8`, which is exactly the scale
    // index `mmqf_vec_dot_d` reads at `X_DS + kw / 8`.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index;
        // the tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / (BLOCK_ELEMS / 32);
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;
        const unsigned int sup =
            CLAMP_K ? min(b0 / (BLOCK_ELEMS / 32), supers - 1) : b0 / (BLOCK_ELEMS / 32);
        const unsigned long long off_blk = (unsigned long long)sup * BLOCK_BYTES;

        // Row-invariant, so the addressing collapses to one add per row.
        const unsigned long long off_qs = off_blk + 8 + lane * 4;
        const unsigned int w_lo = 8 * (lane / 4) + lane % 4;

        constexpr int ROWS = MMQF_Y / MMQF_WARPS;
        constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
        static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
        for (int g = 0; g < ROWS / BATCH; ++g) {
            int v[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
                v[u] = *reinterpret_cast<const int*>(row + off_qs);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
                // The codebook lookup sits in the SECOND loop, so it never
                // separates a global load from the next load's issue. Table
                // values are signed int8 already, so the staged lanes need no
                // bias and `vec_dot` is Q8_0's unchanged.
                int q_lo;
                int q_hi;
                gguf_iq4_table_lookup(v[u], &q_lo, &q_hi);
                s_x[i * X_STRIDE + X_QS + w_lo] = q_lo;
                s_x[i * X_STRIDE + X_QS + w_lo + 4] = q_hi;
            }
        }

        // Scale pass: eight per row, so a warp covers four rows. `d`,
        // `scales_h` and the `scales_l` byte are at row-invariant offsets, so
        // the first loop is pure loads and the 6-bit assembly happens in the
        // second, on registers.
        float* s_xd = (float*)s_x;
        const unsigned int sb = lane % 8;  // 32-element sub-block
        const unsigned int rsub = lane / 8;
        const unsigned long long off_sl = off_blk + 4 + sb / 2;

        constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 4);
        __half d_h[SROWS];
        unsigned int sh[SROWS];
        unsigned int sl[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
            d_h[u] = *reinterpret_cast<const __half*>(row + off_blk);
            sh[u] = *reinterpret_cast<const unsigned short*>(row + off_blk + 2);
            sl[u] = row[off_sl];
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            // Low 4 bits from the `scales_l` nibble for this sub-block, high 2
            // bits from `scales_h` bits `2*sb`, then the -32 bias.
            const int ls =
                (int)(((sl[u] >> (4 * (sb % 2))) & 0x0Fu) | (((sh[u] >> (2 * sb)) & 0x03u) << 4));
            s_xd[i * X_STRIDE + X_DS + sb] = __half2float(d_h[u]) * (float)(ls - 32);
        }
    }

    // Forwards to `mmqf_vec_dot_d`, shared with Q8_0: the staged row layout
    // and the one-term arithmetic are the same for both.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_d<MMQ_X, FULL, X_QS, X_DS, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};

// ── Shared staging for the grid-indexed IQ formats ─────────────────────────
//
// IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ1_S have the same interior. A
// 256-element block is 32 SUB-GROUPS of 8 elements; every sub-group names 8
// magnitude bytes of a codebook grid plus, for five of the six, one byte of
// eight sign bits, one bit per magnitude. Only the way a sub-group reaches its
// magnitudes and its sign byte differs, so that is the only thing `DEC`
// carries.
//
// SIGN SOURCE. `DEC::SIGNED_GRID` says whether the grid's components are
// already signed. The IQ2 and IQ3 grids store magnitudes and a separate sign
// byte selects the sign, so those decoders set it false and the fold below
// runs. The IQ1 grid stores SIGNED bytes and has no sign table at all, so
// IQ1_S sets it true, leaves `signs` alone, and the staged word is the decoded
// word unchanged.
//
// GRID WIDTH. The IQ2 grids are 8-component: one point supplies the whole
// sub-group, as a `u64` split into two magnitude words. The IQ3 grids are
// 4-component: TWO consecutive points supply the sub-group, one `u32`
// magnitude word each. `DEC::GRID_COMPONENTS` names which, and it is checked
// here rather than assumed. Either way a sub-group is exactly TWO staged
// words, so the map below is one skeleton at both widths and `DEC::decode`
// hands back the sub-group's two magnitude words already in element order.
//
// A lane owns one sub-group: `sub = lane` covers all 32 with a single global
// load group and no inner loop. Sub-group `s` is elements `8s .. 8s+7`, which
// is exactly the staged words `2s` and `2s+1`, in element order, which is the
// order an int8 lane quadruple is read back in. So a staged word `w` holds
// elements `4w .. 4w+3` for all five formats, and both of the family's scale
// maps fall out of that — `w / 8` is the 32-element sub-block
// `mmqf_vec_dot_d` indexes at, `w / 4` the 16-element group
// `mmqf_vec_dot_sc16` indexes at. Sign bits `0..3` belong to the first staged
// word and `4..7` to the second, which is `iq_sign_mask4`'s nibble argument.
//
// `DEC` supplies the per-format delta, split in two so the two-loop staging
// shape survives: `DEC::load(row)` issues the sub-group's global loads and
// NOTHING else, `DEC::decode` does the grid and sign lookups afterwards. A
// table lookup placed between two loads would separate a load from the next
// load's issue and serialize the group.
template <int MMQ_X, int X_QS, int X_STRIDE, class DEC>
static __device__ __forceinline__ void mmqf_stage_iq_grid(
    const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int i_max,
    unsigned long long rstride, unsigned int feat0, const DEC& dec
) {
    static_assert(
        DEC::GRID_COMPONENTS == 8 || DEC::GRID_COMPONENTS == 4,
        "A sign sub-group is 8 elements: one 8-component point or two 4-component points."
    );
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    // Sub-group `lane` occupies staged words `2 * lane` and `2 * lane + 1`.
    const unsigned int w_lo = (threadIdx.x % WARP_SIZE) * 2;

    constexpr int ROWS = MMQF_Y / MMQF_WARPS;
    constexpr int BATCH = MMQF_STAGE_BATCH(MMQ_X);
    static_assert(ROWS % BATCH == 0, "Weight row batches are ragged.");

#pragma unroll
    for (int g = 0; g < ROWS / BATCH; ++g) {
        typename DEC::Regs v[BATCH];
#pragma unroll
        for (int u = 0; u < BATCH; ++u) {
            const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
            v[u] = dec.load(weight + (feat0 + min(i, i_max)) * rstride);
        }
#pragma unroll
        for (int u = 0; u < BATCH; ++u) {
            const unsigned int i = (unsigned int)((g * BATCH + u) * MMQF_WARPS) + warp;
            unsigned int mag[2];
            // A signed-grid decoder never writes this; initialized so the
            // unused value is never read from an uninitialized register.
            unsigned int signs = 0;
            dec.decode(v[u], mag, signs);
            // `__vsub4(mag ^ mask, mask)` is a per-byte negate-if: a byte whose
            // mask is 0xFF becomes `(255 - mag) - 255 = -mag`, and a byte whose
            // mask is 0 is untouched. No borrow crosses a lane. The largest
            // magnitude in any of the unsigned grids is 62, so a negated
            // component still fits an int8 lane; the signed IQ1 grid's
            // components are already in `{-1, 0, 1}`.
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                if constexpr (DEC::SIGNED_GRID) {
                    s_x[i * X_STRIDE + X_QS + w_lo + t] = (int)mag[t];
                } else {
                    const unsigned int m = iq_sign_mask4((unsigned char)signs, t);
                    s_x[i * X_STRIDE + X_QS + w_lo + t] = (int)__vsub4(mag[t] ^ m, m);
                }
            }
        }
    }
}

// Shared scale pass for IQ2_XS and IQ2_S, which pack their scales identically:
// a `scales[8]` field of 4-bit values, one per 16-element group, two groups to
// a byte, each giving `d * (0.5 + s) * 0.25`. `SCALES` is the byte offset of
// that field within the block — 66 for IQ2_XS, 74 for IQ2_S — and is the only
// thing that differs.
//
// Sixteen scales per row, so a warp covers two rows. Group `j` (0..15) covers
// entries `2j` and `2j + 1`, which is the `k = entry / 2` of `packed_scale` in
// `src/quant/cpu/kernels/dequant_iq2.rs`: its byte is `scales[j / 2]` and its
// nibble `j % 2`. Staged word `w` sits in group `w / 4`, which is what
// `mmqf_vec_dot_sc16` indexes at.
//
// The scale is staged as f32 already multiplied by `d`, never through `half`:
// `half` has an 11-bit significand and rounding `d * (0.5 + s) * 0.25` to it
// perturbs every group's contribution past the bound the GEMM/GEMV parity
// tests hold this path to.
//
// `d` and the scale byte are at row-invariant offsets, so the first loop is
// pure loads and the nibble is unpacked in the second, on registers.
template <int MMQ_X, int X_DF, int X_STRIDE, int SCALES>
static __device__ __forceinline__ void mmqf_stage_iq2_packed_scales(
    const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int i_max,
    unsigned long long rstride, unsigned int feat0, unsigned long long off_blk
) {
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    float* s_xdf = (float*)s_x;
    const unsigned int j = lane % 16;  // 16-element group within the block
    const unsigned int rsub = lane / 16;

    constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 2);
    __half d_h[SROWS];
    unsigned int sc[SROWS];
#pragma unroll
    for (int u = 0; u < SROWS; ++u) {
        const unsigned int i = (unsigned int)(u * MMQF_WARPS * 2) + warp * 2 + rsub;
        const unsigned char* blk = weight + (feat0 + min(i, i_max)) * rstride + off_blk;
        // Both block sizes are even, so a block base is 2-byte aligned, which
        // is `alignof(__half)`, and `d`@0 is read directly. The scale is a
        // single byte and carries no alignment requirement of its own.
        d_h[u] = *reinterpret_cast<const __half*>(blk);
        sc[u] = blk[SCALES + j / 2];
    }
#pragma unroll
    for (int u = 0; u < SROWS; ++u) {
        const unsigned int i = (unsigned int)(u * MMQF_WARPS * 2) + warp * 2 + rsub;
        const float s = (float)((sc[u] >> (4 * (j % 2))) & 0x0Fu);
        s_xdf[i * X_STRIDE + X_DF + j] = __half2float(d_h[u]) * (0.5f + s) * 0.25f;
    }
}

// Per-entry decoder for IQ2_XXS. Entry `e` sits in group `e / 4` at sub-index
// `e % 4`: its group's index word is at `qs + 8 * (e / 4)` and the group's
// `aux` word 4 bytes further on. `e % 4` picks the byte of the index word and
// the 7-bit `KSIGNS` field of `aux`.
//
// The block is 66 bytes, so a row base and every block base inside it are only
// 2-byte aligned, and `qs`@2 with them. Both 4-byte reads therefore go through
// `load_int_ua`; a plain `int` load raises CUDA_ERROR_MISALIGNED_ADDRESS, which
// poisons the context for every later launch on it.
struct MmqfIQ2XXSEntry {
    // Magnitudes plus a separate sign byte, so the shared skeleton folds
    // the sign in.
    static constexpr bool SIGNED_GRID = false;
    static constexpr int GRID_COMPONENTS = 8;
    struct Regs {
        int ind;
        int aux;
    };
    unsigned long long off_ind;
    unsigned int sub;

    __device__ __forceinline__ Regs load(const unsigned char* row) const {
        Regs r;
        r.ind = load_int_ua(row + off_ind);
        r.aux = load_int_ua(row + off_ind + 4);
        return r;
    }
    __device__ __forceinline__ void decode(
        const Regs& r, unsigned int (&mag)[2], unsigned int& signs
    ) const {
        const unsigned long long point = IQ2XXS_GRID[((unsigned int)r.ind >> (8 * sub)) & 0xFFu];
        mag[0] = (unsigned int)point;
        mag[1] = (unsigned int)(point >> 32);
        signs = KSIGNS[((unsigned int)r.aux >> (7 * sub)) & 0x7Fu];
    }
};

// Per-entry decoder for IQ2_XS. Entry `e` is one little-endian u16 at
// `qs + 2 * e`: its low 9 bits index `IQ2XS_GRID` (512 points) and its top 7
// bits index `KSIGNS`.
//
// The block is 74 bytes and `qs` starts at byte 2, so the entry's address is
// even and the 2-byte load is aligned. This is a 2-byte read, not a 4-byte one,
// so `load_int_ua` does not apply; the 32 lanes of a warp read 64 contiguous
// bytes.
struct MmqfIQ2XSEntry {
    // Magnitudes plus a separate sign byte, so the shared skeleton folds
    // the sign in.
    static constexpr bool SIGNED_GRID = false;
    static constexpr int GRID_COMPONENTS = 8;
    typedef unsigned short Regs;
    unsigned long long off_q;

    __device__ __forceinline__ Regs load(const unsigned char* row) const {
        return *reinterpret_cast<const unsigned short*>(row + off_q);
    }
    __device__ __forceinline__ void decode(
        const Regs& q, unsigned int (&mag)[2], unsigned int& signs
    ) const {
        const unsigned long long point = IQ2XS_GRID[(unsigned int)q & 511u];
        mag[0] = (unsigned int)point;
        mag[1] = (unsigned int)(point >> 32);
        signs = KSIGNS[(unsigned int)q >> 9];
    }
};

// Per-entry decoder for IQ2_S, the one divergence in the group. Entry `e`
// takes its low 8 index bits from `qs[e]` and two more from the field
// `2 * (e % 4)` of `qh[e / 4]`, selecting among 1024 grid points, and its sign
// byte is `signs[e]` DIRECTLY — explicit bits, with no `KSIGNS` indirection.
//
// All three reads are single bytes, so no alignment requirement arises and
// `load_int_ua` does not apply. Across a warp the `qs` and `signs` reads each
// cover 32 contiguous bytes and the `qh` read 8, so all three coalesce.
struct MmqfIQ2SEntry {
    // Magnitudes plus a separate sign byte, so the shared skeleton folds
    // the sign in.
    static constexpr bool SIGNED_GRID = false;
    static constexpr int GRID_COMPONENTS = 8;
    struct Regs {
        unsigned int qs;
        unsigned int sign;
        unsigned int qh;
    };
    unsigned long long off_qs;
    unsigned long long off_sign;
    unsigned long long off_qh;
    unsigned int qh_shift;

    __device__ __forceinline__ Regs load(const unsigned char* row) const {
        Regs r;
        r.qs = row[off_qs];
        r.sign = row[off_sign];
        r.qh = row[off_qh];
        return r;
    }
    __device__ __forceinline__ void decode(
        const Regs& r, unsigned int (&mag)[2], unsigned int& signs
    ) const {
        const unsigned long long point = IQ2S_GRID[r.qs | (((r.qh >> qh_shift) & 0x03u) << 8)];
        mag[0] = (unsigned int)point;
        mag[1] = (unsigned int)(point >> 32);
        signs = r.sign;
    }
};

// Per-sub-group decoder for IQ3_XXS, the family's first 4-COMPONENT grid. A
// sub-group needs TWO grid points: sub-group `s` is elements `8s .. 8s+7`,
// element `8s+j` takes its magnitude from `IQ3XXS_GRID[qs[2s]]` component `j`
// for `j < 4` and from `IQ3XXS_GRID[qs[2s+1]]` component `j - 4` above that.
// The CPU dequantizer reaches the same two bytes as
// `qs[group * 8 + sub * 2 (+1)]` with `group = s / 4` and `sub = s % 4`, and
// `(s / 4) * 8 + (s % 4) * 2` is `2s`.
//
// Its sign byte is `KSIGNS[(aux >> (7 * sub)) & 0x7F]`, where `aux` is the
// group's u32 at `scales + 4 * (s / 4)`.
//
// The two index bytes are adjacent and `qs`@2 keeps `2 + 2s` even, so they are
// read as ONE aligned u16 — low byte `qs[2s]`, high byte `qs[2s+1]` on a
// little-endian device. The block is 98 bytes, so a row base and every block
// base inside it are only 2-byte aligned; the `aux` read is 4 bytes and
// therefore goes through `load_int_ua`. A plain `int` load raises
// CUDA_ERROR_MISALIGNED_ADDRESS, which poisons the context for every later
// launch on it.
struct MmqfIQ3XXSEntry {
    // Magnitudes plus a separate sign byte, so the shared skeleton folds
    // the sign in.
    static constexpr bool SIGNED_GRID = false;
    static constexpr int GRID_COMPONENTS = 4;
    struct Regs {
        unsigned int qs;
        int aux;
    };
    unsigned long long off_qs;
    unsigned long long off_aux;
    unsigned int sub;

    __device__ __forceinline__ Regs load(const unsigned char* row) const {
        Regs r;
        r.qs = *reinterpret_cast<const unsigned short*>(row + off_qs);
        r.aux = load_int_ua(row + off_aux);
        return r;
    }
    __device__ __forceinline__ void decode(
        const Regs& r, unsigned int (&mag)[2], unsigned int& signs
    ) const {
        mag[0] = IQ3XXS_GRID[r.qs & 0xFFu];
        mag[1] = IQ3XXS_GRID[r.qs >> 8];
        signs = KSIGNS[((unsigned int)r.aux >> (7 * sub)) & 0x7Fu];
    }
};

// Per-sub-group decoder for IQ3_S: a 4-COMPONENT grid like IQ3_XXS, reached by
// a 9-bit index and paired with explicit sign bits.
//
// Sub-group `s` is elements `8s .. 8s+7`. Element `e` sits at grid entry
// `e / 4` component `e % 4`, so the sub-group's two entries are `2s` and
// `2s+1`. Entry `n` takes its low 8 index bits from `qs[n]` and its NINTH from
// bit `n % 8` of `qh[n / 8]`, selecting among the 512 points of `IQ3S_GRID`.
// Both `2s` and `2s+1` land in the same `qh` byte `qh[s / 4]`, at bit
// positions `2 * (s % 4)` and one above it, so ONE byte read serves the pair.
//
// Its sign byte is `signs[s]` DIRECTLY — eight explicit bits, one per element
// of the sub-group, with no `KSIGNS` indirection.
//
// The two index bytes are adjacent and `qs`@2 keeps `2 + 2s` even, so they are
// read as ONE aligned u16. The block is 110 bytes, so a row base and every
// block base inside it are only 2-byte aligned — but this format issues NO
// 4-byte read at all, here or in its scale pass, so `load_int_ua` does not
// apply to it.
struct MmqfIQ3SEntry {
    // Magnitudes plus a separate sign byte, so the shared skeleton folds
    // the sign in.
    static constexpr bool SIGNED_GRID = false;
    static constexpr int GRID_COMPONENTS = 4;
    struct Regs {
        unsigned int qs;
        unsigned int qh;
        unsigned int sign;
    };
    unsigned long long off_qs;
    unsigned long long off_qh;
    unsigned long long off_sign;
    unsigned int qh_shift;

    __device__ __forceinline__ Regs load(const unsigned char* row) const {
        Regs r;
        r.qs = *reinterpret_cast<const unsigned short*>(row + off_qs);
        r.qh = row[off_qh];
        r.sign = row[off_sign];
        return r;
    }
    __device__ __forceinline__ void decode(
        const Regs& r, unsigned int (&mag)[2], unsigned int& signs
    ) const {
        mag[0] = IQ3S_GRID[(r.qs & 0xFFu) | (((r.qh >> qh_shift) & 1u) << 8)];
        mag[1] = IQ3S_GRID[(r.qs >> 8) | (((r.qh >> (qh_shift + 1)) & 1u) << 8)];
        signs = r.sign;
    }
};

// Per-sub-group decoder for IQ1_S, the family's only SIGNED grid. Its grid
// width is the IQ2 formats': one 8-component point of `IQ1_GRID` supplies a
// whole 8-element sub-group, as a `u64` split into two magnitude words.
//
// Sub-group `s` is elements `8s .. 8s+7`, which is the CPU dequantizer's
// `(group, sub)` pair with `group = s / 4` and `sub = s % 4`. Its index byte is
// `qs[group * 4 + sub]`, which is `qs[s]`; its three high index bits are field
// `3 * (s % 4)` of the group's little-endian u16 `qh[s / 4]`. Together they
// form an 11-bit index into the 2048-point grid.
//
// SIGNS. There are none to fold: `IQ1_GRID` stores SIGNED bytes in
// `{-1, 0, 1}` and IQ1_S has no sign table, so `SIGNED_GRID` is true and
// `signs` is left untouched. The delta IQ1_S adds to every component is NOT
// handled here — it is a per-group affine term carried by the staged
// scale/min pair; see `MmqfIQ1S`.
//
// The index byte is a single byte and the `qh` field is a u16 at the even
// offset `34 + 2 * (s / 4)` inside a 50-byte block, so neither read is 4 bytes
// and `load_int_ua` does not apply. Across a warp the `qs` reads cover 32
// contiguous bytes and the `qh` reads 16, so both coalesce.
struct MmqfIQ1SEntry {
    // `IQ1_GRID`'s components are already signed, so the shared skeleton stores
    // the decoded words unchanged.
    static constexpr bool SIGNED_GRID = true;
    static constexpr int GRID_COMPONENTS = 8;
    struct Regs {
        unsigned int qs;
        unsigned int qh;
    };
    unsigned long long off_qs;
    unsigned long long off_qh;
    unsigned int qh_shift;

    __device__ __forceinline__ Regs load(const unsigned char* row) const {
        Regs r;
        r.qs = row[off_qs];
        r.qh = *reinterpret_cast<const unsigned short*>(row + off_qh);
        return r;
    }
    __device__ __forceinline__ void decode(
        const Regs& r, unsigned int (&mag)[2], unsigned int& signs
    ) const {
        const unsigned long long point = IQ1_GRID[r.qs | (((r.qh >> qh_shift) & 0x07u) << 8)];
        mag[0] = (unsigned int)point;
        mag[1] = (unsigned int)(point >> 32);
        (void)signs;
    }
};

// IQ2_XXS weight format policy, same contract as `MmqfQ80`.
//
// The family's first GRID-INDEXED format. IQ2_XXS is a 256-element block of 66
// bytes: `f16 d`@0 then `qs[64]`@2, read as eight pairs of u32. The first u32
// of a pair holds four 8-bit indices into `IQ2XXS_GRID`, whose entry expands to
// EIGHT magnitude bytes; the second holds the group's 4-bit scale in its top
// nibble over four 7-bit indices into `KSIGNS`, one sign bit per expanded
// component. Reading `qs` as packed 2-bit magnitudes still yields finite,
// plausibly scaled numbers, which is why both tables live in exactly one place
// (`iq_grid.cuh`) for every CUDA kernel.
//
// Staging EXPANDS the grid point to signed int8 and folds the sign in, so the
// staged row is Q8_0's unchanged — signed quants in the int8 lanes plus one f32
// scale per 32-element sub-block — and `vec_dot` is Q8_0's. The grid's only
// magnitudes are 8, 25 and 43, so a negated component still fits an int8 lane.
// A group of 32 elements is exactly one sub-block, which is the granularity
// `mmqf_vec_dot_d` already indexes at.
//
// The scale is staged as f32 already multiplied by `d`, never through `half`:
// `half` has an 11-bit significand and rounding `d * (0.5 + s) * 0.25` to it
// perturbs every sub-block's contribution past the bound the GEMM/GEMV parity
// tests hold this path to.
//
// ALIGNMENT. The block is 66 bytes, so a row base (`supers * 66`) and every
// block base inside it are only 2-byte aligned, and `qs`@2 with it. Every
// 4-byte read of `qs` therefore goes through `load_int_ua`; a plain `int` load
// raises CUDA_ERROR_MISALIGNED_ADDRESS, which poisons the context for every
// later launch on it. The `f16 d`@0 is 2-byte aligned, which is
// `alignof(__half)`, and is read directly.
struct MmqfIQ2XXS {
    // On-disk block: one f16 scale then 64 bytes of grid indices, scales and
    // sign-table indices.
    static constexpr int BLOCK_BYTES = 66;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // No per-16 minimum, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Staged weight row: Q8_0's, byte for byte. 64 quant words (256 k-values),
    // then 8 f32 sub-block scales, then padding that makes the stride an odd
    // multiple of 4 ints so the strided fragment gathers hit all 32 banks.
    static constexpr int X_QS = 0;
    static constexpr int X_DS = 64;
    static constexpr int X_STRIDE = 76;
    static_assert(X_DS + 8 <= X_STRIDE, "Weight row too short: 8 sub-block scales.");
    static_assert(
        X_QS == MmqfQ80::X_QS && X_DS == MmqfQ80::X_DS && X_STRIDE == MmqfQ80::X_STRIDE,
        "IQ2_XXS must stage into the Q8_0 row; the two share `mmqf_vec_dot_d`."
    );

    // Stages 256 k-values, which for this format is exactly ONE block. `b0`
    // counts 32-element Q8_1 activation blocks, so the block index is `b0 / 8`.
    //
    // Quant map, derived from `dequant_iq2_xxs` in
    // `src/quant/cpu/kernels/dequant_iq2.rs` and matching the per-block decoder
    // `iq2_xxs_dequant_block` in `iq_dequant.cuh`. Element `e` sits at
    // `group = e / 32`, `sub = (e % 32) / 8`, `j = e % 8`. Its group's index
    // word is at `qs + 8 * group` and its `aux` word 4 bytes further on; `sub`
    // picks the byte of the index word and the 7-bit `KSIGNS` field of `aux`;
    // `j` picks the magnitude byte of the grid entry and the sign bit.
    //
    // A lane therefore owns one (group, sub) pair: `group = lane / 4`,
    // `sub = lane % 4`, which covers all 8 groups with 4 lanes each and needs
    // TWO global loads per lane. Since `entry = 4 * group + sub = lane`, that
    // is the one-entry-per-lane map `mmqf_stage_iq_grid` runs, and this
    // format's whole share of it is `MmqfIQ2XXSEntry`. Under that map a staged
    // word `w` belongs to group `w / 8`, which is exactly the scale index
    // `mmqf_vec_dot_d` reads at `X_DS + kw / 8`.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index;
        // the tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / (BLOCK_ELEMS / 32);
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;
        const unsigned int sup =
            CLAMP_K ? min(b0 / (BLOCK_ELEMS / 32), supers - 1) : b0 / (BLOCK_ELEMS / 32);
        const unsigned long long off_blk = (unsigned long long)sup * BLOCK_BYTES;

        // Row-invariant, so the addressing collapses to one add per row.
        MmqfIQ2XXSEntry dec;
        dec.off_ind = off_blk + 2 + (unsigned long long)(lane / 4) * 8;
        dec.sub = lane % 4;
        mmqf_stage_iq_grid<MMQ_X, X_QS, X_STRIDE>(weight, s_x, i_max, rstride, feat0, dec);

        // Scale pass: eight per row, so a warp covers four rows. `d` and the
        // group's `aux` word are at row-invariant offsets, so the first loop is
        // pure loads and the 4-bit scale is unpacked in the second, on
        // registers.
        float* s_xd = (float*)s_x;
        const unsigned int sb = lane % 8;  // 32-element group
        const unsigned int rsub = lane / 8;
        const unsigned long long off_sc = off_blk + 2 + (unsigned long long)sb * 8 + 4;

        constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 4);
        __half d_h[SROWS];
        int aux[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
            d_h[u] = *reinterpret_cast<const __half*>(row + off_blk);
            aux[u] = load_int_ua(row + off_sc);
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            // Top nibble of `aux` is the group's 4-bit scale `s`, and the
            // sub-block scale is `d * (0.5 + s) * 0.25`.
            const float s = (float)((unsigned int)aux[u] >> 28);
            s_xd[i * X_STRIDE + X_DS + sb] = __half2float(d_h[u]) * (0.5f + s) * 0.25f;
        }
    }

    // Forwards to `mmqf_vec_dot_d`, shared with Q8_0: once staging has expanded
    // the grid point and folded the sign in, the two rows are
    // indistinguishable.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_d<MMQ_X, FULL, X_QS, X_DS, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};

// IQ2_XS weight format policy, same contract as `MmqfQ80`.
//
// IQ2_XS is a 256-element block of 74 bytes: `f16 d`@0, `qs[64]`@2 read as 32
// little-endian u16, then `scales[8]`@66. Each u16 packs a 9-bit index into
// `IQ2XS_GRID` (512 points, eight magnitude bytes each) in its low bits and a
// 7-bit index into `KSIGNS` above them. Reading `qs` as packed 2-bit magnitudes
// still yields finite, plausibly scaled numbers, which is why both tables live
// in exactly one place (`iq_grid.cuh`) for every CUDA kernel.
//
// Staging EXPANDS the grid point to signed int8 and folds the sign in, through
// `mmqf_stage_iq_grid`, so the staged quant lanes are Q8_0's and IQ2_XXS's.
//
// SCALE GRANULARITY. This is where IQ2_XS parts company with IQ2_XXS. Its
// 4-bit scale is packed two per `scales` byte and `packed_scale` indexes it at
// `entry / 2`, so it changes every SIXTEEN elements, not every 32. A single
// `mma_m16n8k32_s8` cannot express one 32-k step under that, so the staged row
// is Q6_K's — 64 quant words then 16 f32 `d * (0.5 + s) * 0.25` — and
// `vec_dot` is `mmqf_vec_dot_sc16`, which runs two `mma_m16n8k16_s8` calls per
// 32-k step.
//
// ALIGNMENT. The block is 74 bytes, so a row base (`supers * 74`) and every
// block base inside it are only 2-byte aligned. There is no 4-byte read here:
// the entry is one u16 at an even offset and the scale is one byte, so
// `load_int_ua` does not apply. `f16 d`@0 is 2-byte aligned, which is
// `alignof(__half)`, and is read directly.
struct MmqfIQ2XS {
    // On-disk block: f16 scale, 64 bytes of packed index/sign u16, 8 bytes of
    // packed 4-bit scales.
    static constexpr int BLOCK_BYTES = 74;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // No minimum term, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Byte offset of `scales[8]` within the block.
    static constexpr int SCALES_OFF = 66;

    // Staged weight row: Q6_K's, int for int. 64 quant words (256 k-values),
    // then 16 f32 group scales, then padding that makes the stride an odd
    // multiple of 4 ints so the strided fragment gathers hit all 32 banks.
    static constexpr int X_QS = 0;
    static constexpr int X_DF = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DF + 16 <= X_STRIDE, "Weight row too short: 16 group scales.");
    static_assert(
        X_QS == MmqfQ6K::X_QS && X_DF == MmqfQ6K::X_DF && X_STRIDE == MmqfQ6K::X_STRIDE,
        "IQ2_XS must stage into the Q6_K row; the two share `mmqf_vec_dot_sc16`."
    );

    // Stages 256 k-values, which for this format is exactly ONE block. `b0`
    // counts 32-element Q8_1 activation blocks, so the block index is `b0 / 8`.
    //
    // Quant map, derived from `dequant_iq2_xs` in
    // `src/quant/cpu/kernels/dequant_iq2.rs` and matching the per-block decoder
    // `iq2_xs_dequant_block` in `iq_dequant.cuh`. Element `e` sits at
    // `entry = e / 8`, `j = e % 8`. Its u16 is at `qs + 2 * entry`, `j` picks
    // the magnitude byte of the grid entry and the sign bit, and its scale is
    // group `entry / 2`. Lane `l` owns entry `l` and fills staged words `2l`
    // and `2l + 1`, so staged word `w` sits in scale group `w / 4` — the
    // property `mmqf_vec_dot_sc16` relies on.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index;
        // the tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / (BLOCK_ELEMS / 32);
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;
        const unsigned int sup =
            CLAMP_K ? min(b0 / (BLOCK_ELEMS / 32), supers - 1) : b0 / (BLOCK_ELEMS / 32);
        const unsigned long long off_blk = (unsigned long long)sup * BLOCK_BYTES;

        // Row-invariant, so the addressing collapses to one add per row.
        MmqfIQ2XSEntry dec;
        dec.off_q = off_blk + 2 + (unsigned long long)lane * 2;
        mmqf_stage_iq_grid<MMQ_X, X_QS, X_STRIDE>(weight, s_x, i_max, rstride, feat0, dec);

        mmqf_stage_iq2_packed_scales<MMQ_X, X_DF, X_STRIDE, SCALES_OFF>(
            weight, s_x, i_max, rstride, feat0, off_blk
        );
    }

    // Forwards to `mmqf_vec_dot_sc16`, shared with Q6_K and Q3_K: once staging
    // has expanded the grid point and folded the sign in, the rows are
    // indistinguishable, and the per-16 scale arithmetic is the same.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_sc16<MMQ_X, FULL, false, X_QS, X_DF, X_STRIDE>(
            s_x, s_y, acc, i0, jb, k00, nks
        );
    }
};

// IQ2_S weight format policy, same contract as `MmqfQ80`.
//
// IQ2_S is a 256-element block of 82 bytes: `f16 d`@0, `qs[32]`@2,
// `signs[32]`@34, `qh[8]`@66, `scales[8]`@74. Entry `e` takes its low 8 index
// bits from `qs[e]` and two more from the field `2 * (e % 4)` of `qh[e / 4]`,
// selecting among the 1024 points of `IQ2S_GRID`.
//
// SIGNS. This is the divergence from IQ2_XXS and IQ2_XS. The sign byte is
// `signs[e]` directly — eight explicit bits per entry — with no `KSIGNS`
// indirection. `MmqfIQ2SEntry` absorbs that difference; everything downstream
// of it, including the branchless fold in `mmqf_stage_iq_grid`, is unchanged.
//
// SCALE GRANULARITY. Identical to IQ2_XS: a 4-bit scale per SIXTEEN elements,
// packed two per `scales` byte. So the staged row is Q6_K's and `vec_dot` is
// `mmqf_vec_dot_sc16`, and the scale pass is the one IQ2_XS uses, at the
// format's own `scales` offset.
//
// ALIGNMENT. The block is 82 bytes, so a row base (`supers * 82`) and every
// block base inside it are only 2-byte aligned. There is no 4-byte read here:
// `qs`, `signs`, `qh` and `scales` are all read one byte at a time, so
// `load_int_ua` does not apply. `f16 d`@0 is 2-byte aligned, which is
// `alignof(__half)`, and is read directly.
struct MmqfIQ2S {
    // On-disk block: f16 scale, 32 index bytes, 32 sign bytes, 8 index-high
    // bytes, 8 bytes of packed 4-bit scales.
    static constexpr int BLOCK_BYTES = 82;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // No minimum term, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Byte offsets of the fields an entry reads, within the block.
    static constexpr int QS_OFF = 2;
    static constexpr int SIGNS_OFF = 34;
    static constexpr int QH_OFF = 66;
    static constexpr int SCALES_OFF = 74;

    // Staged weight row: Q6_K's, int for int, same as IQ2_XS.
    static constexpr int X_QS = 0;
    static constexpr int X_DF = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DF + 16 <= X_STRIDE, "Weight row too short: 16 group scales.");
    static_assert(
        X_QS == MmqfQ6K::X_QS && X_DF == MmqfQ6K::X_DF && X_STRIDE == MmqfQ6K::X_STRIDE,
        "IQ2_S must stage into the Q6_K row; the two share `mmqf_vec_dot_sc16`."
    );

    // Stages 256 k-values, which for this format is exactly ONE block. `b0`
    // counts 32-element Q8_1 activation blocks, so the block index is `b0 / 8`.
    //
    // Quant map, derived from `dequant_iq2_s` in
    // `src/quant/cpu/kernels/dequant_iq2.rs` and matching the per-block decoder
    // `iq2_s_dequant_block` in `iq_dequant.cuh`. Element `e` sits at
    // `entry = e / 8`, `j = e % 8`. Its index byte is at `qs + entry`, its two
    // index-high bits at field `2 * (entry % 4)` of `qh[entry / 4]`, its sign
    // byte at `signs + entry`, and its scale is group `entry / 2`. Lane `l`
    // owns entry `l` and fills staged words `2l` and `2l + 1`, so staged word
    // `w` sits in scale group `w / 4` — the property `mmqf_vec_dot_sc16`
    // relies on.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index;
        // the tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / (BLOCK_ELEMS / 32);
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;
        const unsigned int sup =
            CLAMP_K ? min(b0 / (BLOCK_ELEMS / 32), supers - 1) : b0 / (BLOCK_ELEMS / 32);
        const unsigned long long off_blk = (unsigned long long)sup * BLOCK_BYTES;

        // Row-invariant, so the addressing collapses to one add per row.
        MmqfIQ2SEntry dec;
        dec.off_qs = off_blk + QS_OFF + lane;
        dec.off_sign = off_blk + SIGNS_OFF + lane;
        dec.off_qh = off_blk + QH_OFF + lane / 4;
        dec.qh_shift = 2 * (lane % 4);
        mmqf_stage_iq_grid<MMQ_X, X_QS, X_STRIDE>(weight, s_x, i_max, rstride, feat0, dec);

        mmqf_stage_iq2_packed_scales<MMQ_X, X_DF, X_STRIDE, SCALES_OFF>(
            weight, s_x, i_max, rstride, feat0, off_blk
        );
    }

    // Forwards to `mmqf_vec_dot_sc16`, shared with Q6_K, Q3_K and IQ2_XS: once
    // staging has expanded the grid point and folded the explicit sign in, the
    // rows are indistinguishable, and the per-16 scale arithmetic is the same.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_sc16<MMQ_X, FULL, false, X_QS, X_DF, X_STRIDE>(
            s_x, s_y, acc, i0, jb, k00, nks
        );
    }
};

// IQ3_XXS weight format policy, same contract as `MmqfQ80`.
//
// IQ3_XXS is a 256-element block of 98 bytes: `f16 d`@0, `qs[64]`@2, then
// `scales[32]`@66 read as eight little-endian u32, one per 32-element group.
// Each group's u32 carries its 4-bit scale in the TOP nibble over four 7-bit
// indices into `KSIGNS`, one sign byte per 8-element sub-group.
//
// GRID WIDTH. This is the family's first 4-COMPONENT grid: each `qs` byte
// indexes one of the 256 points of `IQ3XXS_GRID`, and a point expands to FOUR
// magnitude bytes, not eight. So an 8-element sign sub-group spans TWO
// consecutive `qs` bytes — the low one for elements 0..3 of the sub-group, the
// high one for 4..7 — where the IQ2 formats spent one grid point on the whole
// sub-group. `MmqfIQ3XXSEntry` absorbs that; the staging skeleton is the same
// one at both widths, since a sub-group is two staged words either way.
//
// Staging EXPANDS the two grid points to signed int8 and folds the sign in, so
// the staged row is Q8_0's unchanged and `vec_dot` is Q8_0's. The grid's
// magnitudes run 4..62, so a negated component still fits an int8 lane.
//
// SCALE GRANULARITY. The 4-bit scale is one per `aux` word and an `aux` word
// covers one group of 32 elements, which is exactly the granularity
// `mmqf_vec_dot_d` already indexes at. So this format stages 8 f32 rather than
// Q6_K's 16 and needs no wider row. The scale is `d * (0.5 + s) * 0.5` — the
// trailing factor is 0.5, not the IQ2 family's 0.25.
//
// The scale is staged as f32 already multiplied by `d`, never through `half`:
// `half` has an 11-bit significand and rounding `d * (0.5 + s) * 0.5` to it
// perturbs every group's contribution past the bound the GEMM/GEMV parity
// tests hold this path to.
//
// ALIGNMENT. The block is 98 bytes, so a row base (`supers * 98`) and every
// block base inside it are only 2-byte aligned. Every 4-byte read of an `aux`
// word — in the quant pass and in the scale pass — therefore goes through
// `load_int_ua`; a plain `int` load raises CUDA_ERROR_MISALIGNED_ADDRESS,
// which poisons the context for every later launch on it. The `f16 d`@0 is
// 2-byte aligned, which is `alignof(__half)`, and is read directly, as is the
// u16 index pair at the even offset `2 + 2 * lane`.
struct MmqfIQ3XXS {
    // On-disk block: one f16 scale, 64 grid-index bytes, 32 bytes of packed
    // scales and sign-table indices.
    static constexpr int BLOCK_BYTES = 98;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // No minimum term, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Byte offset of `scales[32]` within the block.
    static constexpr int SCALES_OFF = 66;

    // Staged weight row: Q8_0's, byte for byte. 64 quant words (256 k-values),
    // then 8 f32 group scales, then padding that makes the stride an odd
    // multiple of 4 ints so the strided fragment gathers hit all 32 banks.
    static constexpr int X_QS = 0;
    static constexpr int X_DS = 64;
    static constexpr int X_STRIDE = 76;
    static_assert(X_DS + 8 <= X_STRIDE, "Weight row too short: 8 group scales.");
    static_assert(
        X_QS == MmqfQ80::X_QS && X_DS == MmqfQ80::X_DS && X_STRIDE == MmqfQ80::X_STRIDE,
        "IQ3_XXS must stage into the Q8_0 row; the two share `mmqf_vec_dot_d`."
    );

    // Stages 256 k-values, which for this format is exactly ONE block. `b0`
    // counts 32-element Q8_1 activation blocks, so the block index is `b0 / 8`.
    //
    // Quant map, derived from `dequant_iq3_xxs` in
    // `src/quant/cpu/kernels/dequant_iq3.rs` and matching the per-block decoder
    // `iq3_xxs_dequant_block` in `iq_dequant.cuh`. Element `e` sits at
    // `group = e / 32`, `sub = (e % 32) / 8`, `j = e % 8`. Lane `l` owns the
    // sign sub-group `s = 4 * group + sub = l`: its two index bytes are
    // `qs[2l]` and `qs[2l + 1]`, its sign byte is
    // `KSIGNS[(aux >> (7 * (l % 4))) & 0x7F]` for the `aux` word at
    // `scales + 4 * (l / 4)`, and it fills staged words `2l` and `2l + 1`.
    // Under that map a staged word `w` belongs to group `w / 8`, which is
    // exactly the scale index `mmqf_vec_dot_d` reads at `X_DS + kw / 8`.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index;
        // the tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / (BLOCK_ELEMS / 32);
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;
        const unsigned int sup =
            CLAMP_K ? min(b0 / (BLOCK_ELEMS / 32), supers - 1) : b0 / (BLOCK_ELEMS / 32);
        const unsigned long long off_blk = (unsigned long long)sup * BLOCK_BYTES;

        // Row-invariant, so the addressing collapses to one add per row.
        MmqfIQ3XXSEntry dec;
        dec.off_qs = off_blk + 2 + (unsigned long long)lane * 2;
        dec.off_aux = off_blk + SCALES_OFF + (unsigned long long)(lane / 4) * 4;
        dec.sub = lane % 4;
        mmqf_stage_iq_grid<MMQ_X, X_QS, X_STRIDE>(weight, s_x, i_max, rstride, feat0, dec);

        // Scale pass: eight per row, so a warp covers four rows. `d` and the
        // group's `aux` word are at row-invariant offsets, so the first loop is
        // pure loads and the 4-bit scale is unpacked in the second, on
        // registers.
        float* s_xd = (float*)s_x;
        const unsigned int sb = lane % 8;  // 32-element group
        const unsigned int rsub = lane / 8;
        const unsigned long long off_sc = off_blk + SCALES_OFF + (unsigned long long)sb * 4;

        constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 4);
        __half d_h[SROWS];
        int aux[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
            d_h[u] = *reinterpret_cast<const __half*>(row + off_blk);
            aux[u] = load_int_ua(row + off_sc);
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            // Top nibble of `aux` is the group's 4-bit scale `s`, and the
            // group scale is `d * (0.5 + s) * 0.5`.
            const float s = (float)((unsigned int)aux[u] >> 28);
            s_xd[i * X_STRIDE + X_DS + sb] = __half2float(d_h[u]) * (0.5f + s) * 0.5f;
        }
    }

    // Forwards to `mmqf_vec_dot_d`, shared with Q8_0: once staging has expanded
    // the two grid points and folded the sign in, the two rows are
    // indistinguishable.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_d<MMQ_X, FULL, X_QS, X_DS, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};

// IQ3_S weight format policy, same contract as `MmqfQ80`.
//
// IQ3_S is a 256-element block of 110 bytes: `f16 d`@0, `qs[64]`@2, `qh[8]`@66,
// `signs[32]`@74, `scales[4]`@106. Grid entry `n` takes its low 8 index bits
// from `qs[n]` and a NINTH from bit `n % 8` of `qh[n / 8]`, selecting among the
// 512 points of `IQ3S_GRID`, four magnitude bytes each. It shares IQ3_XXS's
// 4-component grid width, so an 8-element sign sub-group again spans two
// consecutive entries and `MmqfIQ3SEntry` reads both.
//
// SIGNS. Explicit bits, as in IQ2_S: element `e` takes bit `e % 8` of
// `signs[e / 8]`, so sub-group `s` takes the whole byte `signs[s]`, with no
// `KSIGNS` indirection.
//
// SCALE GRANULARITY. One 4-bit scale per 32-element group, eight of them
// packed two per byte across `scales[4]` — the granularity `mmqf_vec_dot_d`
// already indexes at, so the staged row is Q8_0's and not Q6_K's. Note that
// the packing is IQ2_XS's while the granularity is NOT: `scales` holds 4 bytes
// here, not 8, and the index is the 32-element group, not the pair of entries.
//
// SCALE FORM. `d * (1 + 2 * s)`, which differs in FORM from every other IQ
// format's `d * (0.5 + s) * c`. It is not an affine rewrite of one: at `s = 0`
// it is `d`, not `0.5 * d * c`.
//
// The scale is staged as f32 already multiplied by `d`, never through `half`,
// for the same parity reason as the rest of the family.
//
// ALIGNMENT. The block is 110 bytes, so a row base (`supers * 110`) and every
// block base inside it are only 2-byte aligned. This format issues NO 4-byte
// read: the index pair is one u16 at the even offset `2 + 2 * lane`, and `qh`,
// `signs` and `scales` are read one byte at a time. So `load_int_ua` does not
// apply to it. The `f16 d`@0 is 2-byte aligned, which is `alignof(__half)`,
// and is read directly.
struct MmqfIQ3S {
    // On-disk block: one f16 scale, 64 index bytes, 8 index-high bytes, 32
    // sign bytes, 4 bytes of packed 4-bit scales.
    static constexpr int BLOCK_BYTES = 110;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // No minimum term, so no activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Byte offsets of the fields a sub-group reads, within the block.
    static constexpr int QS_OFF = 2;
    static constexpr int QH_OFF = 66;
    static constexpr int SIGNS_OFF = 74;
    static constexpr int SCALES_OFF = 106;

    // Staged weight row: Q8_0's, byte for byte, same as IQ3_XXS.
    static constexpr int X_QS = 0;
    static constexpr int X_DS = 64;
    static constexpr int X_STRIDE = 76;
    static_assert(X_DS + 8 <= X_STRIDE, "Weight row too short: 8 group scales.");
    static_assert(
        X_QS == MmqfQ80::X_QS && X_DS == MmqfQ80::X_DS && X_STRIDE == MmqfQ80::X_STRIDE,
        "IQ3_S must stage into the Q8_0 row; the two share `mmqf_vec_dot_d`."
    );

    // Stages 256 k-values, which for this format is exactly ONE block. `b0`
    // counts 32-element Q8_1 activation blocks, so the block index is `b0 / 8`.
    //
    // Quant map, derived from `dequant_iq3_s` in
    // `src/quant/cpu/kernels/dequant_iq3.rs` and matching the per-block decoder
    // `iq3_s_dequant_block` in `iq_dequant.cuh`. Element `e` sits at grid entry
    // `e / 4`, component `e % 4`, sign byte `signs[e / 8]` bit `e % 8`. Lane
    // `l` owns the sign sub-group `s = e / 8 = l`: its two entries are `2l` and
    // `2l + 1`, whose index bytes are `qs[2l]` and `qs[2l + 1]` and whose ninth
    // bits are bits `2 * (l % 4)` and `2 * (l % 4) + 1` of the single byte
    // `qh[l / 4]` — both entries fall in that byte because `(2l) / 8` and
    // `(2l + 1) / 8` are both `l / 4`. Its sign byte is `signs[l]`, and it
    // fills staged words `2l` and `2l + 1`. Under that map a staged word `w`
    // belongs to the 32-element group `w / 8`, which is exactly the scale index
    // `mmqf_vec_dot_d` reads at `X_DS + kw / 8`.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index;
        // the tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / (BLOCK_ELEMS / 32);
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;
        const unsigned int sup =
            CLAMP_K ? min(b0 / (BLOCK_ELEMS / 32), supers - 1) : b0 / (BLOCK_ELEMS / 32);
        const unsigned long long off_blk = (unsigned long long)sup * BLOCK_BYTES;

        // Row-invariant, so the addressing collapses to one add per row.
        MmqfIQ3SEntry dec;
        dec.off_qs = off_blk + QS_OFF + (unsigned long long)lane * 2;
        dec.off_qh = off_blk + QH_OFF + lane / 4;
        dec.off_sign = off_blk + SIGNS_OFF + lane;
        dec.qh_shift = 2 * (lane % 4);
        mmqf_stage_iq_grid<MMQ_X, X_QS, X_STRIDE>(weight, s_x, i_max, rstride, feat0, dec);

        // Scale pass: eight per row, so a warp covers four rows. `d` and the
        // group's scale byte are at row-invariant offsets, so the first loop is
        // pure loads and the nibble is unpacked in the second, on registers.
        float* s_xd = (float*)s_x;
        const unsigned int sb = lane % 8;  // 32-element group
        const unsigned int rsub = lane / 8;
        const unsigned long long off_sc = off_blk + SCALES_OFF + sb / 2;

        constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 4);
        __half d_h[SROWS];
        unsigned int sc[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
            d_h[u] = *reinterpret_cast<const __half*>(row + off_blk);
            sc[u] = row[off_sc];
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            // Two groups share a byte, low nibble first, and the group scale is
            // `d * (1 + 2 * s)` — this format's own form.
            const float s = (float)((sc[u] >> (4 * (sb % 2))) & 0x0Fu);
            s_xd[i * X_STRIDE + X_DS + sb] = __half2float(d_h[u]) * (1.0f + 2.0f * s);
        }
    }

    // Forwards to `mmqf_vec_dot_d`, shared with Q8_0 and IQ3_XXS: once staging
    // has expanded the two grid points and folded the explicit sign in, the
    // rows are indistinguishable.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_d<MMQ_X, FULL, X_QS, X_DS, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};

// IQ1_S weight format policy, same contract as `MmqfQ80`.
//
// IQ1_S is a 256-element block of 50 bytes: `f16 d`@0, `qs[32]`@2, `qh[16]`@34
// read as eight little-endian u16, one per 32-element group. Sub-group `s`
// (8 elements) takes its low 8 index bits from `qs[s]` and three more from
// field `3 * (s % 4)` of `qh[s / 4]`, an 11-bit index into the 2048-point
// `IQ1_GRID`. That group's u16 also carries a 3-bit scale at bits 12..14 and
// the group's delta sign at bit 15.
//
// THE AFFINE VALUE. This is what separates IQ1_S from every other format in
// the family. Its dequantized value is
//
//     dl * (g + delta),   dl = d * (2 * sc + 1),   delta = +/- 0.125
//
// with `g` an already-signed grid component. That is an AFFINE transform of
// the grid point, not a scale times an int8, so one scaled int32 dot cannot
// express it. Expanding a 32-element group's contribution against that group's
// activations `a`:
//
//     sum_j a_j * dl * (g_j + delta) = dl * dot(a, g) + dl * delta * sum(a)
//
// Both terms are per 32-element group, which is exactly the shape
// `mmqf_vec_dot_dm` already computes for Q4_K, Q5_K, Q4_1 and Q5_1: one
// `(scale, min)` f32 pair per 32 elements, the first against the int32 dot and
// the second against the activation block sum. So IQ1_S stages the Q4_K row
// and needs NO new `vec_dot` — it stores `(dl, dl * delta)` where those
// formats store their `(d, m)`.
//
// The rewrite is EXACT because the block sum `mmqf_vec_dot_dm` multiplies by
// is the int16 the activation producer stored in the header word's high half,
// which is the true sum of that sub-block's 32 int8 quants. Folding `delta`
// into a `half` field on the activation side instead would not be.
//
// THE MIN SIGN. `mmqf_vec_dot_dm` always ADDS `pair.y * (activation scale *
// block sum)`. IQ1_S's delta term is additive, so it stores `+dl * delta` —
// Q4_1's and Q5_1's convention, not Q4_K's negated `-dmin * m`. The delta's
// own sign, bit 15 of the group's u16, is folded into the stored value at
// staging.
//
// SCALE GRANULARITY. `dequant_iq1_s` computes `dl` and `delta` once per
// `group` and holds both across all four of that group's sub-blocks, so one
// pair covers `4 * 8 = 32` elements. A 256-k staging group therefore carries
// exactly 8 pairs, which is the granularity `mmqf_vec_dot_dm` indexes with
// `kw / 8`.
//
// SIGNS. `IQ1_GRID` stores signed components and IQ1_S has no sign table, so
// there is nothing to fold; `MmqfIQ1SEntry` sets `SIGNED_GRID`.
//
// ALIGNMENT. The block is 50 bytes, so a row base (`supers * 50`) and every
// block base inside it are only 2-byte aligned. This format issues NO 4-byte
// read: `qs` is read one byte at a time and `qh` as a u16 at the even offset
// `34 + 2 * group`. So `load_int_ua` does not apply to it. The `f16 d`@0 is
// 2-byte aligned, which is `alignof(__half)`, and is read directly.
struct MmqfIQ1S {
    // On-disk block: one f16 scale, 32 index bytes, then 16 bytes holding the
    // eight per-group u16 of index-high fields, scale and delta sign.
    static constexpr int BLOCK_BYTES = 50;
    static constexpr int BLOCK_ELEMS = 256;
    // Dispatch gates on `k % 256 == 0`, so a row's last 256-k group is always
    // whole and the ragged tail path is not compiled.
    static constexpr bool RAGGED_K = false;
    // The delta term is per 32 elements, the granularity the activation
    // record's own block sum already carries, so no per-16 split and no
    // activation scratch.
    static constexpr int Y_SCRATCH = 0;

    // Byte offsets of the fields a sub-group reads, within the block.
    static constexpr int QS_OFF = 2;
    static constexpr int QH_OFF = 34;

    // Staged weight row: Q4_K's, int for int. 64 quant words (256 k-values, one
    // SIGNED grid component per int8 lane), then 8 `float2` holding
    // `(dl, dl * delta)`, then padding.
    static constexpr int X_QS = 0;
    static constexpr int X_DM = 64;
    static constexpr int X_STRIDE = 84;
    static_assert(X_DM + 16 <= X_STRIDE, "Weight row too short: 8 scale/min pairs.");
    static_assert(X_DM % 2 == 0 && X_STRIDE % 2 == 0, "Scale/min pairs are misaligned.");
    static_assert(
        X_QS == MmqfQ4K::X_QS && X_DM == MmqfQ4K::X_DM && X_STRIDE == MmqfQ4K::X_STRIDE,
        "IQ1_S must stage into the Q4_K row; the two share `mmqf_vec_dot_dm`."
    );
    static_assert(
        X_QS == MmqfQ41::X_QS && X_DM == MmqfQ41::X_DM && X_STRIDE == MmqfQ41::X_STRIDE,
        "IQ1_S takes Q4_1's ADDITIVE min convention, so it stages Q4_1's row too."
    );

    // Stages 256 k-values, which for this format is exactly ONE block. `b0`
    // counts 32-element Q8_1 activation blocks, so the block index is `b0 / 8`.
    //
    // Quant map, derived from `dequant_iq1_s` in
    // `src/quant/cpu/kernels/dequant_iq1.rs` and matching the per-block decoder
    // `iq1_s_dequant_block` in `iq_dequant.cuh`. Element `e` is
    // `group = e / 32`, `sub = (e % 32) / 8`, `j = e % 8`, so its sub-group is
    // `s = e / 8 = group * 4 + sub`. Lane `l` owns sub-group `l`: index byte
    // `qs[l]`, `qh` word `l / 4`, index-high shift `3 * (l % 4)`. It fills
    // staged words `2l` and `2l + 1`, so a staged word `w` belongs to the
    // 32-element group `w / 8` — exactly the pair index `mmqf_vec_dot_dm`
    // reads at `(X_DM / 2) + kw / 8`.
    template <int MMQ_X, bool CLAMP_K>
    static __device__ __forceinline__ void stage(
        const unsigned char* __restrict__ weight, int* __restrict__ s_x, unsigned int N,
        unsigned int bpr, unsigned int feat0, unsigned int b0
    ) {
        const unsigned int lane = threadIdx.x % WARP_SIZE;
        const unsigned int warp = threadIdx.x / WARP_SIZE;

        // Tile-local cap: clamping `i` to this and adding `feat0` back gives a
        // feature row inside the matrix. The clamp is on the TILE-LOCAL index;
        // the tile origin is added after it, never folded into it.
        const unsigned int i_max = N - feat0 - 1;
        const unsigned int supers = bpr / (BLOCK_ELEMS / 32);
        const unsigned long long rstride = (unsigned long long)supers * BLOCK_BYTES;
        const unsigned int sup =
            CLAMP_K ? min(b0 / (BLOCK_ELEMS / 32), supers - 1) : b0 / (BLOCK_ELEMS / 32);
        const unsigned long long off_blk = (unsigned long long)sup * BLOCK_BYTES;

        // Row-invariant, so the addressing collapses to one add per row.
        MmqfIQ1SEntry dec;
        dec.off_qs = off_blk + QS_OFF + lane;
        dec.off_qh = off_blk + QH_OFF + (unsigned long long)(lane / 4) * 2;
        dec.qh_shift = 3 * (lane % 4);
        mmqf_stage_iq_grid<MMQ_X, X_QS, X_STRIDE>(weight, s_x, i_max, rstride, feat0, dec);

        // Scale/min pass: eight pairs per row, so a warp covers four rows. `d`
        // and the group's u16 are at row-invariant offsets, so the first loop
        // is pure loads and every bitfield is unpacked in the second, on
        // registers.
        float2* s_xdm = (float2*)s_x;
        const unsigned int sb = lane % 8;  // 32-element group within the block
        const unsigned int rsub = lane / 8;
        const unsigned long long off_h = off_blk + QH_OFF + (unsigned long long)sb * 2;

        constexpr int SROWS = MMQF_Y / (MMQF_WARPS * 4);
        __half d_h[SROWS];
        unsigned int h[SROWS];
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            const unsigned char* row = weight + (feat0 + min(i, i_max)) * rstride;
            d_h[u] = *reinterpret_cast<const __half*>(row + off_blk);
            h[u] = *reinterpret_cast<const unsigned short*>(row + off_h);
        }
#pragma unroll
        for (int u = 0; u < SROWS; ++u) {
            const unsigned int i = (unsigned int)(u * MMQF_WARPS * 4) + warp * 4 + rsub;
            // Bits 12..14 hold the 3-bit scale, bit 15 the delta sign.
            const float dl = __half2float(d_h[u]) * (2.0f * (float)((h[u] >> 12) & 7u) + 1.0f);
            const float delta = (h[u] & 0x8000u) == 0 ? IQ1_DELTA : -IQ1_DELTA;
            // `+dl * delta`, never negated: `mmqf_vec_dot_dm` ADDS `pair.y`
            // times the activation block sum and this format's delta term is
            // additive. Both components are f32 for the same parity reason as
            // Q4_K — no `half` round-trip on a term the GEMM/GEMV bound is
            // sensitive to.
            s_xdm[(i * X_STRIDE + X_DM) / 2 + sb] = make_float2(dl, dl * delta);
        }
    }

    // Forwards to `mmqf_vec_dot_dm`, shared with Q4_K, Q5_K, Q4_1 and Q5_1:
    // once staging has expanded the signed grid point and split the affine
    // value into `(dl, dl * delta)`, the row and the two-term arithmetic are
    // Q4_1's.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
        const int* __restrict__ s_x, const int* __restrict__ s_y,
        float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
        unsigned int k00, unsigned int nks
    ) {
        mmqf_vec_dot_dm<MMQ_X, FULL, X_QS, X_DM, X_STRIDE>(s_x, s_y, acc, i0, jb, k00, nks);
    }
};

// Stages 128 k-values of the activation tile as a FLAT COPY. The repacked
// layout indexes records k-group-major, token-minor, so the `MMQ_X` records a
// token tile needs are contiguous and the shared row IS the record: 4 header
// words (`half` scale, int16 quant sum) then 32 quant words. No per-element index math, no token bound, no
// k-block bound — the repack already zeroed the padded token slots and the
// padded k-blocks.
//
// `b0` is always a multiple of 4, so `b0 / 4` is the k-group index.
template <int MMQ_X>
static __device__ __forceinline__ void mmqf_stage_y(
    const int* __restrict__ y_packed, int* __restrict__ s_y, unsigned int ntok,
    unsigned int tok0, unsigned int b0
) {
    const int* src =
        y_packed + ((unsigned long long)(b0 / 4) * ntok + tok0) * MMQF_Y_STRIDE;

    constexpr int TOTAL = MMQ_X * MMQF_Y_STRIDE;
    constexpr int CITER = (TOTAL + MMQF_THREADS - 1) / MMQF_THREADS;
    constexpr int CBATCH = CITER < MMQF_STAGE_BATCH(MMQ_X) ? CITER : MMQF_STAGE_BATCH(MMQ_X);
    constexpr int CGROUPS = (CITER + CBATCH - 1) / CBATCH;

#pragma unroll
    for (int g = 0; g < CGROUPS; ++g) {
        int v[CBATCH];
        // `g` and `u` are compile-time here, so both range tests fold away when
        // the copy divides evenly.
#pragma unroll
        for (int u = 0; u < CBATCH; ++u) {
            const int e = g * CBATCH + u;
            if (e >= CITER) {
                continue;
            }
            const unsigned int l = (unsigned int)(e * MMQF_THREADS) + threadIdx.x;
            if (TOTAL % MMQF_THREADS == 0 || l < (unsigned int)TOTAL) {
                v[u] = src[l];
            }
        }
#pragma unroll
        for (int u = 0; u < CBATCH; ++u) {
            const int e = g * CBATCH + u;
            if (e >= CITER) {
                continue;
            }
            const unsigned int l = (unsigned int)(e * MMQF_THREADS) + threadIdx.x;
            if (TOTAL % MMQF_THREADS == 0 || l < (unsigned int)TOTAL) {
                s_y[l] = v[u];
            }
        }
    }
}

// Derives the per-16 activation sums the Q2_K minimum term needs, once per
// staged 128-k activation tile, into the scratch region after that tile.
//
// WHY THIS EXISTS. The activation record's header word carries an exact int16
// sum of each 32-value sub-block, which is the right granularity for Q4_K and
// Q5_K. Q2_K's minimum changes every 16 elements, so a per-32 sum is the wrong
// granularity: the two halves of a 32-value window carry different minima.
//
// Only ONE extra number per (token, sub-block) is needed, not two: this pass
// stores `s_a`, the sum of the FIRST 16 quants, and `vec_dot` recovers the
// second as `s32 - s_a`. Both are integer sums over the same 32 int8 values,
// so that identity is exact in integer arithmetic — the derived half is as
// exact as the stored one.
//
// COST. It runs once per activation tile and is independent of the feature
// rows, so all MMQF_Y features amortize it. That is what separates it from
// deriving the sums with an all-ones `mma` inside `vec_dot`, which costs one
// extra tensor-core issue per (token group, k-step, minitile).
//
// llama.cpp instead gives Q2_K its own activation layout
// (`MMQ_Q8_1_DS_LAYOUT_D2S6` in `ggml-cuda/quantize.cu`), which coarsens the
// activation scale from 32 values to 64, stores the per-16 sums as `half`
// rather than exactly, and still needs an all-ones `mma` for the last quarter
// of each tile. This kernel keeps one format-neutral, exact activation record
// for all seven formats instead.
template <int MMQ_X>
static __device__ __forceinline__ void mmqf_stage_y_sums(
    const int* __restrict__ s_y, int* __restrict__ s_ys
) {
    constexpr int TOTAL = MMQ_X * MMQF_Y_SCRATCH;

#pragma unroll
    for (int e = 0; e < (TOTAL + MMQF_THREADS - 1) / MMQF_THREADS; ++e) {
        const unsigned int v = (unsigned int)(e * MMQF_THREADS) + threadIdx.x;
        if (TOTAL % MMQF_THREADS == 0 || v < (unsigned int)TOTAL) {
            // Token-major, matching the `MMQF_Y_SCRATCH` ints-per-token
            // framing the host's shared-memory request uses.
            const unsigned int j = v / MMQF_Y_SCRATCH;   // token
            const unsigned int ks = v % MMQF_Y_SCRATCH;  // 32-value sub-block
            const int* q = s_y + j * MMQF_Y_STRIDE + MMQF_Y_QS + ks * 8;
            // Four quant words are 16 int8 values, which is one Q2_K scale
            // group. The reduction is integer and therefore exact:
            // |sum| <= 16 * 128 = 2048.
            int s_a = 0;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                s_a = dp4a(0x01010101, q[w], s_a);
            }
            s_ys[v] = s_a;
        }
    }
}

// Runs `mmqf_stage_y_sums` for the formats that ask for it, with the barrier
// that publishes it. Compiles to nothing for the other six.
template <class FMT, int MMQ_X>
static __device__ __forceinline__ void mmqf_stage_y_sums_if(int* __restrict__ s_y) {
    if constexpr (FMT::Y_SCRATCH > 0) {
        static_assert(FMT::Y_SCRATCH == MMQF_Y_SCRATCH, "Unexpected scratch width.");
        mmqf_stage_y_sums<MMQ_X>(s_y, s_y + MMQ_X * MMQF_Y_STRIDE);
        __syncthreads();
    }
}

// Runs the k-block range `[kb0_start, kb0_stop)` of one output tile into `acc`,
// which it also zeroes. `kb0_start` is always a multiple of MMQF_ITER_B;
// `kb0_stop` is too, unless it is `bpr`, whose last partial 256-k group takes
// the early-stop path.
//
// The leading barrier makes the function self-contained: a caller may run
// several tiles back to back without draining the previous tile's shared reads
// itself.
template <class FMT, int MMQ_X>
static __device__ __forceinline__ void mmqf_accumulate(
    const int* __restrict__ y_packed,
    const unsigned char* __restrict__ weight, int* __restrict__ s_x,
    int* __restrict__ s_y, float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int ntok,
    unsigned int N, unsigned int bpr, unsigned int tok0, unsigned int feat0,
    unsigned int i0, unsigned int jb, unsigned int kb0_start, unsigned int kb0_stop
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);

    // Holds for every format: any multiple of 8 puts consecutive weight rows in
    // the same 128-bit segment and serializes the strided fragment gathers.
    static_assert(FMT::X_STRIDE % 8 == 4, "Wrong weight row padding.");

#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
#pragma unroll
        for (int n = 0; n < NTX; ++n) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                acc[jj][n][l] = 0.0f;
            }
        }
    }

    __syncthreads();

    // The legacy formats (Q8_0, Q4_0) guarantee K only as a multiple of one
    // 32-element block, so the final group of a tile can hold fewer than
    // MMQF_ITER_B blocks. A format that
    // guarantees whole 256-k groups sets `RAGGED_K` false, which folds this to
    // `kb0_stop` and drops the tail instantiations below.
    const unsigned int full_stop =
        (FMT::RAGGED_K && kb0_stop == bpr) ? (bpr & ~(MMQF_ITER_B - 1u)) : kb0_stop;

    for (unsigned int b0 = kb0_start; b0 < full_stop; b0 += MMQF_ITER_B) {
        FMT::template stage<MMQ_X, false>(weight, s_x, N, bpr, feat0, b0);
        mmqf_stage_y<MMQ_X>(y_packed, s_y, ntok, tok0, b0);
        __syncthreads();
        mmqf_stage_y_sums_if<FMT, MMQ_X>(s_y);

        FMT::template vec_dot<MMQ_X, true>(s_x, s_y, acc, i0, jb, 0, 4);
        __syncthreads();

        mmqf_stage_y<MMQ_X>(y_packed, s_y, ntok, tok0, b0 + 4);
        __syncthreads();
        mmqf_stage_y_sums_if<FMT, MMQ_X>(s_y);

        FMT::template vec_dot<MMQ_X, true>(s_x, s_y, acc, i0, jb, MMQF_HALF_W, 4);
        __syncthreads();
    }

    // `bpr` and the range bounds are uniform across the block, so every barrier
    // below is still reached by all threads.
    if constexpr (FMT::RAGGED_K) {
        if (full_stop < kb0_stop) {
            const unsigned int nb = kb0_stop - full_stop;
            FMT::template stage<MMQ_X, true>(weight, s_x, N, bpr, feat0, full_stop);
            mmqf_stage_y<MMQ_X>(y_packed, s_y, ntok, tok0, full_stop);
            __syncthreads();
            mmqf_stage_y_sums_if<FMT, MMQ_X>(s_y);

            FMT::template vec_dot<MMQ_X, false>(s_x, s_y, acc, i0, jb, 0, nb < 4 ? nb : 4);

            if (nb > 4) {
                __syncthreads();
                mmqf_stage_y<MMQ_X>(y_packed, s_y, ntok, tok0, full_stop + 4);
                __syncthreads();
                mmqf_stage_y_sums_if<FMT, MMQ_X>(s_y);
                FMT::template vec_dot<MMQ_X, false>(s_x, s_y, acc, i0, jb, MMQF_HALF_W, nb - 4);
            }
        }
    }
}

// The accumulator is (feature, token)-indexed; the output stays
// (token, feature)-indexed.
template <int MMQ_X>
static __device__ __forceinline__ void mmqf_write_dst(
    float* __restrict__ output, const float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4],
    unsigned int M, unsigned int N, unsigned int tok0, unsigned int feat0, unsigned int i0,
    unsigned int jb
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);
#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
#pragma unroll
        for (int n = 0; n < NTX; ++n) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const unsigned int f = feat0 + i0 + n * 16 + mma_d_i(l);
                const unsigned int t = tok0 + jj * (NTX * 8) + jb + mma_d_j(l);
                if (t < M && f < N) output[(unsigned long long)t * N + f] = acc[jj][n][l];
            }
        }
    }
}

// Writes a partial tile to this block's workspace slot. The slot is a dense
// MMQ_X x MMQF_Y tile in (token, feature) order, so no bounds check applies;
// the fixup pass clips when it folds the slot into the output.
template <int MMQ_X>
static __device__ __forceinline__ void mmqf_write_fixup(
    float* __restrict__ ws, const float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4],
    unsigned int i0, unsigned int jb
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);
#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
#pragma unroll
        for (int n = 0; n < NTX; ++n) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const unsigned int i = i0 + n * 16 + mma_d_i(l);
                const unsigned int j = jj * (NTX * 8) + jb + mma_d_j(l);
                ws[j * MMQF_Y + i] = acc[jj][n][l];
            }
        }
    }
}

// Tile-parallel path: one block per output tile, whole K.
template <class FMT, int MMQ_X>
static __device__ __forceinline__ void mmqf_body(
    const int* __restrict__ y_packed,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N, unsigned int ntok
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);
    static_assert(MMQ_X % MMQF_GRAN(MMQ_X) == 0, "MMQ_X violates the granularity rule.");
    static_assert(MMQF_WARPS % NTX == 0, "Warps do not split into NTX token columns.");
    static_assert((MMQF_WARPS / NTX) * NTX * 16 == MMQF_Y, "Warps do not cover the features.");

    int* s_x = mmqf_smem;
    int* s_y = mmqf_smem + MMQF_Y * FMT::X_STRIDE;

    const unsigned int warp = threadIdx.x / WARP_SIZE;
    // Q8_1 activation blocks per row. Always 32 elements, whatever the weight
    // format packs, because every MMQ path quantizes activations to Q8_1. The
    // k-step count, the stream-k walk and `MMQF_ITER_B` are all counted in
    // these, so a 256-element weight block converts inside `FMT::stage`.
    const unsigned int bpr = K / 32;

    // The eight warps form a (8/NTX) x NTX grid over the output tile: a warp
    // owns NTX consecutive 16-feature minitiles and every NTX-th token group.
    const unsigned int i0 = (warp / NTX) * (NTX * 16);
    const unsigned int jb = (warp % NTX) * 8;

    const unsigned int tok0 = blockIdx.x * MMQ_X;
    const unsigned int feat0 = blockIdx.y * MMQF_Y;

    float acc[NJ][NTX][4];
    mmqf_accumulate<FMT, MMQ_X>(y_packed, weight, s_x, s_y, acc, ntok, N, bpr, tok0, feat0, i0, jb,
                                0, bpr);
    mmqf_write_dst<MMQ_X>(output, acc, M, N, tok0, feat0, i0, jb);
}

// Stream-k path: a fixed block count, sized by the launcher from the device,
// walks the flattened (feature-tile, token-tile, k-block) space. It exists
// because the tile-parallel grid starves the device when the tile count is
// small relative to it; when the tiles already fill the device, that path is
// still the better choice and stays selectable.
//
// A block that reaches the END of a tile stores the tile (plain store). A block
// left holding a tile it did not finish writes that partial to its workspace
// slot. The fixup pass then folds every predecessor's partial into the tile the
// storing block wrote. This splits the K reduction across blocks, so the sum is
// reassociated relative to the tile-parallel path.
template <class FMT, int MMQ_X>
static __device__ __forceinline__ void mmqf_sk_body(
    const int* __restrict__ y_packed,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    float* __restrict__ workspace,
    unsigned int M, unsigned int K, unsigned int N, unsigned int ntok
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);

    int* s_x = mmqf_smem;
    int* s_y = mmqf_smem + MMQF_Y * FMT::X_STRIDE;

    const unsigned int warp = threadIdx.x / WARP_SIZE;
    // Activation blocks, as in `mmqf_body` — never the weight block count.
    const unsigned int bpr = K / 32;
    const unsigned int i0 = (warp / NTX) * (NTX * 16);
    const unsigned int jb = (warp % NTX) * 8;

    // The walk divides by the k-block count, so an empty K has no work to slice.
    if (bpr == 0) {
        return;
    }

    const long long nbk = bpr;
    const long long ntt = (M + MMQ_X - 1) / MMQ_X;   // token tiles
    const long long ntf = (N + MMQF_Y - 1) / MMQF_Y;  // feature tiles
    const long long total = ntf * ntt * nbk;

    // Slice bounds snapped to a 256-k boundary inside their tile, so a block
    // never starts mid-group and the staging map stays intact.
    long long kbc = (long long)blockIdx.x * total / gridDim.x;
    long long kbc_stop = (long long)(blockIdx.x + 1) * total / gridDim.x;
    kbc -= (kbc % nbk) % MMQF_ITER_B;
    kbc_stop -= (kbc_stop % nbk) % MMQF_ITER_B;

    long long kb0_start = kbc % nbk;
    long long kb0_stop = kb0_start + kbc_stop - kbc;
    if (kb0_stop > nbk) {
        kb0_stop = nbk;
    }

    float acc[NJ][NTX][4];

    // Every tile this block carries to its end goes straight to the output.
    while (kbc < kbc_stop && kb0_stop == nbk) {
        const long long tile = kbc / nbk;
        const unsigned int feat0 = (unsigned int)(tile / ntt) * MMQF_Y;
        const unsigned int tok0 = (unsigned int)(tile % ntt) * MMQ_X;

        mmqf_accumulate<FMT, MMQ_X>(y_packed, weight, s_x, s_y, acc, ntok, N, bpr, tok0, feat0,
                                    i0, jb, (unsigned int)kb0_start, (unsigned int)kb0_stop);
        mmqf_write_dst<MMQ_X>(output, acc, M, N, tok0, feat0, i0, jb);

        kbc += nbk;
        kbc -= kbc % nbk;
        kb0_start = 0;
        kb0_stop = kbc_stop - kbc;
        if (kb0_stop > nbk) {
            kb0_stop = nbk;
        }
    }

    if (kbc >= kbc_stop) {
        return;
    }

    // One tile is left unfinished; it becomes this block's workspace slot.
    const long long tile = kbc / nbk;
    const unsigned int feat0 = (unsigned int)(tile / ntt) * MMQF_Y;
    const unsigned int tok0 = (unsigned int)(tile % ntt) * MMQ_X;

    mmqf_accumulate<FMT, MMQ_X>(y_packed, weight, s_x, s_y, acc, ntok, N, bpr, tok0, feat0, i0, jb,
                                (unsigned int)kb0_start, (unsigned int)kb0_stop);
    mmqf_write_fixup<MMQ_X>(workspace + (long long)blockIdx.x * (MMQ_X * MMQF_Y), acc, i0, jb);
}

// Folds workspace partials into the output. Launched with the same grid as the
// stream-k kernel, so a block can rebuild any other block's slice bounds from
// its index alone.
//
// The workspace is read with a flat, coalesced map rather than the MMA
// accumulator map the writer used: the slot is a dense tile, so the two maps
// only have to agree on the layout, not on which lane holds which element.
template <int MMQ_X>
static __device__ __forceinline__ void mmqf_fixup_body(
    float* __restrict__ output, const float* __restrict__ workspace, unsigned int M,
    unsigned int K, unsigned int N
) {
    constexpr int NEL = MMQ_X * MMQF_Y / MMQF_THREADS;
    static_assert(NEL * MMQF_THREADS == MMQ_X * MMQF_Y, "Fixup tile is ragged.");

    // The k-block count and the slice arithmetic must match `mmqf_sk_body`
    // exactly. Both count Q8_1 activation blocks of 32, which no weight format
    // changes, so this pass needs no format parameter.
    if (K < 32) {
        return;
    }

    const long long nbk = K / 32;
    const long long ntt = (M + MMQ_X - 1) / MMQ_X;
    const long long ntf = (N + MMQF_Y - 1) / MMQF_Y;
    const long long total = ntf * ntt * nbk;

    const long long bidx0 = blockIdx.x;
    long long kbc0 = bidx0 * total / gridDim.x;
    long long kbc0_stop = (bidx0 + 1) * total / gridDim.x;
    kbc0 -= (kbc0 % nbk) % MMQF_ITER_B;
    kbc0_stop -= (kbc0_stop % nbk) % MMQF_ITER_B;

    // Nothing to fold unless this block STARTED mid-tile and CARRIED that tile
    // to its end, which is the only case where an earlier block holds a partial
    // of the tile this block stored.
    const bool no_data = kbc0 == kbc0_stop;
    const bool started_on_tile = kbc0 % nbk == 0;
    const bool never_finished = kbc0 / nbk == kbc0_stop / nbk && kbc0_stop % nbk != 0;
    if (no_data || started_on_tile || never_finished) {
        return;
    }

    float sum[NEL];
#pragma unroll
    for (int e = 0; e < NEL; ++e) {
        sum[e] = 0.0f;
    }

    // Walk backwards over the blocks that hold partials of this tile. Blocks
    // whose slice snapped to empty wrote nothing and are stepped over.
    long long bidx = bidx0 - 1;
    long long kbc_hi = kbc0;
    while (true) {
        long long kbc = bidx * total / gridDim.x;
        kbc -= (kbc % nbk) % MMQF_ITER_B;

        if (kbc != kbc_hi) {
            const float* ws = workspace + bidx * (MMQ_X * MMQF_Y);
#pragma unroll
            for (int e = 0; e < NEL; ++e) {
                sum[e] += ws[e * MMQF_THREADS + threadIdx.x];
            }
            // A predecessor that started on a tile boundary, or in an earlier
            // tile, holds the front of this tile: nothing before it contributes.
            if (kbc % nbk == 0 || kbc / nbk < kbc0 / nbk) {
                break;
            }
        }

        --bidx;
        kbc_hi = kbc;
    }

    const long long tile = kbc0 / nbk;
    const unsigned int feat0 = (unsigned int)(tile / ntt) * MMQF_Y;
    const unsigned int tok0 = (unsigned int)(tile % ntt) * MMQ_X;

#pragma unroll
    for (int e = 0; e < NEL; ++e) {
        const unsigned int idx = e * MMQF_THREADS + threadIdx.x;
        const unsigned int t = tok0 + idx / MMQF_Y;
        const unsigned int f = feat0 + idx % MMQF_Y;
        if (t < M && f < N) output[(unsigned long long)t * N + f] += sum[e];
    }
}

// One entry point per (weight format, token tile), for each of the three roles.
// `NAME` is the format's name in the symbol, which the host looks up by string.
// The host picks the variant; the rule it must follow is fixed by the
// `MMQ_X % granularity` assert in `mmqf_body`.
#define MMQ_FM_KERNEL(FMT, NAME, X)                                                   \
    extern "C" __global__ __launch_bounds__(MMQF_THREADS)                             \
        void quant_mmq_##NAME##_q8_1_mma_x##X(                                        \
            const int* __restrict__ y_packed,                                         \
            const unsigned char* __restrict__ weight, float* __restrict__ output,     \
            unsigned int M, unsigned int K, unsigned int N, unsigned int ntok         \
        ) {                                                                           \
        mmqf_body<FMT, X>(y_packed, weight, output, M, K, N, ntok);                   \
    }                                                                                 \
    extern "C" __global__ __launch_bounds__(MMQF_THREADS)                             \
        void quant_mmq_##NAME##_q8_1_mma_sk_x##X(                                     \
            const int* __restrict__ y_packed,                                         \
            const unsigned char* __restrict__ weight, float* __restrict__ output,     \
            float* __restrict__ workspace, unsigned int M, unsigned int K,            \
            unsigned int N, unsigned int ntok                                         \
        ) {                                                                           \
        mmqf_sk_body<FMT, X>(y_packed, weight, output, workspace, M, K, N, ntok);     \
    }                                                                                 \
    extern "C" __global__ __launch_bounds__(MMQF_THREADS)                             \
        void quant_mmq_##NAME##_q8_1_mma_fixup_x##X(                                  \
            float* __restrict__ output, const float* __restrict__ workspace,          \
            unsigned int M, unsigned int K, unsigned int N                            \
        ) {                                                                           \
        mmqf_fixup_body<X>(output, workspace, M, K, N);                               \
    }

MMQ_FM_KERNEL(MmqfQ80, q8_0, 8)
MMQ_FM_KERNEL(MmqfQ80, q8_0, 16)
MMQ_FM_KERNEL(MmqfQ80, q8_0, 24)
MMQ_FM_KERNEL(MmqfQ80, q8_0, 32)
MMQ_FM_KERNEL(MmqfQ80, q8_0, 40)
MMQ_FM_KERNEL(MmqfQ80, q8_0, 48)
MMQ_FM_KERNEL(MmqfQ80, q8_0, 64)
MMQ_FM_KERNEL(MmqfQ80, q8_0, 80)
MMQ_FM_KERNEL(MmqfQ80, q8_0, 96)
MMQ_FM_KERNEL(MmqfQ80, q8_0, 112)
MMQ_FM_KERNEL(MmqfQ80, q8_0, 128)

MMQ_FM_KERNEL(MmqfQ40, q4_0, 8)
MMQ_FM_KERNEL(MmqfQ40, q4_0, 16)
MMQ_FM_KERNEL(MmqfQ40, q4_0, 24)
MMQ_FM_KERNEL(MmqfQ40, q4_0, 32)
MMQ_FM_KERNEL(MmqfQ40, q4_0, 40)
MMQ_FM_KERNEL(MmqfQ40, q4_0, 48)
MMQ_FM_KERNEL(MmqfQ40, q4_0, 64)
MMQ_FM_KERNEL(MmqfQ40, q4_0, 80)
MMQ_FM_KERNEL(MmqfQ40, q4_0, 96)
MMQ_FM_KERNEL(MmqfQ40, q4_0, 112)
MMQ_FM_KERNEL(MmqfQ40, q4_0, 128)

MMQ_FM_KERNEL(MmqfQ4K, q4_k, 8)
MMQ_FM_KERNEL(MmqfQ4K, q4_k, 16)
MMQ_FM_KERNEL(MmqfQ4K, q4_k, 24)
MMQ_FM_KERNEL(MmqfQ4K, q4_k, 32)
MMQ_FM_KERNEL(MmqfQ4K, q4_k, 40)
MMQ_FM_KERNEL(MmqfQ4K, q4_k, 48)
MMQ_FM_KERNEL(MmqfQ4K, q4_k, 64)
MMQ_FM_KERNEL(MmqfQ4K, q4_k, 80)
MMQ_FM_KERNEL(MmqfQ4K, q4_k, 96)
MMQ_FM_KERNEL(MmqfQ4K, q4_k, 112)
MMQ_FM_KERNEL(MmqfQ4K, q4_k, 128)

MMQ_FM_KERNEL(MmqfQ5K, q5_k, 8)
MMQ_FM_KERNEL(MmqfQ5K, q5_k, 16)
MMQ_FM_KERNEL(MmqfQ5K, q5_k, 24)
MMQ_FM_KERNEL(MmqfQ5K, q5_k, 32)
MMQ_FM_KERNEL(MmqfQ5K, q5_k, 40)
MMQ_FM_KERNEL(MmqfQ5K, q5_k, 48)
MMQ_FM_KERNEL(MmqfQ5K, q5_k, 64)
MMQ_FM_KERNEL(MmqfQ5K, q5_k, 80)
MMQ_FM_KERNEL(MmqfQ5K, q5_k, 96)
MMQ_FM_KERNEL(MmqfQ5K, q5_k, 112)
MMQ_FM_KERNEL(MmqfQ5K, q5_k, 128)

MMQ_FM_KERNEL(MmqfQ6K, q6_k, 8)
MMQ_FM_KERNEL(MmqfQ6K, q6_k, 16)
MMQ_FM_KERNEL(MmqfQ6K, q6_k, 24)
MMQ_FM_KERNEL(MmqfQ6K, q6_k, 32)
MMQ_FM_KERNEL(MmqfQ6K, q6_k, 40)
MMQ_FM_KERNEL(MmqfQ6K, q6_k, 48)
MMQ_FM_KERNEL(MmqfQ6K, q6_k, 64)
MMQ_FM_KERNEL(MmqfQ6K, q6_k, 80)
MMQ_FM_KERNEL(MmqfQ6K, q6_k, 96)
MMQ_FM_KERNEL(MmqfQ6K, q6_k, 112)
MMQ_FM_KERNEL(MmqfQ6K, q6_k, 128)

MMQ_FM_KERNEL(MmqfQ3K, q3_k, 8)
MMQ_FM_KERNEL(MmqfQ3K, q3_k, 16)
MMQ_FM_KERNEL(MmqfQ3K, q3_k, 24)
MMQ_FM_KERNEL(MmqfQ3K, q3_k, 32)
MMQ_FM_KERNEL(MmqfQ3K, q3_k, 40)
MMQ_FM_KERNEL(MmqfQ3K, q3_k, 48)
MMQ_FM_KERNEL(MmqfQ3K, q3_k, 64)
MMQ_FM_KERNEL(MmqfQ3K, q3_k, 80)
MMQ_FM_KERNEL(MmqfQ3K, q3_k, 96)
MMQ_FM_KERNEL(MmqfQ3K, q3_k, 112)
MMQ_FM_KERNEL(MmqfQ3K, q3_k, 128)

MMQ_FM_KERNEL(MmqfQ2K, q2_k, 8)
MMQ_FM_KERNEL(MmqfQ2K, q2_k, 16)
MMQ_FM_KERNEL(MmqfQ2K, q2_k, 24)
MMQ_FM_KERNEL(MmqfQ2K, q2_k, 32)
MMQ_FM_KERNEL(MmqfQ2K, q2_k, 40)
MMQ_FM_KERNEL(MmqfQ2K, q2_k, 48)
MMQ_FM_KERNEL(MmqfQ2K, q2_k, 64)
MMQ_FM_KERNEL(MmqfQ2K, q2_k, 80)
MMQ_FM_KERNEL(MmqfQ2K, q2_k, 96)
MMQ_FM_KERNEL(MmqfQ2K, q2_k, 112)
MMQ_FM_KERNEL(MmqfQ2K, q2_k, 128)

MMQ_FM_KERNEL(MmqfQ41, q4_1, 8)
MMQ_FM_KERNEL(MmqfQ41, q4_1, 16)
MMQ_FM_KERNEL(MmqfQ41, q4_1, 24)
MMQ_FM_KERNEL(MmqfQ41, q4_1, 32)
MMQ_FM_KERNEL(MmqfQ41, q4_1, 40)
MMQ_FM_KERNEL(MmqfQ41, q4_1, 48)
MMQ_FM_KERNEL(MmqfQ41, q4_1, 64)
MMQ_FM_KERNEL(MmqfQ41, q4_1, 80)
MMQ_FM_KERNEL(MmqfQ41, q4_1, 96)
MMQ_FM_KERNEL(MmqfQ41, q4_1, 112)
MMQ_FM_KERNEL(MmqfQ41, q4_1, 128)

MMQ_FM_KERNEL(MmqfQ50, q5_0, 8)
MMQ_FM_KERNEL(MmqfQ50, q5_0, 16)
MMQ_FM_KERNEL(MmqfQ50, q5_0, 24)
MMQ_FM_KERNEL(MmqfQ50, q5_0, 32)
MMQ_FM_KERNEL(MmqfQ50, q5_0, 40)
MMQ_FM_KERNEL(MmqfQ50, q5_0, 48)
MMQ_FM_KERNEL(MmqfQ50, q5_0, 64)
MMQ_FM_KERNEL(MmqfQ50, q5_0, 80)
MMQ_FM_KERNEL(MmqfQ50, q5_0, 96)
MMQ_FM_KERNEL(MmqfQ50, q5_0, 112)
MMQ_FM_KERNEL(MmqfQ50, q5_0, 128)

MMQ_FM_KERNEL(MmqfQ51, q5_1, 8)
MMQ_FM_KERNEL(MmqfQ51, q5_1, 16)
MMQ_FM_KERNEL(MmqfQ51, q5_1, 24)
MMQ_FM_KERNEL(MmqfQ51, q5_1, 32)
MMQ_FM_KERNEL(MmqfQ51, q5_1, 40)
MMQ_FM_KERNEL(MmqfQ51, q5_1, 48)
MMQ_FM_KERNEL(MmqfQ51, q5_1, 64)
MMQ_FM_KERNEL(MmqfQ51, q5_1, 80)
MMQ_FM_KERNEL(MmqfQ51, q5_1, 96)
MMQ_FM_KERNEL(MmqfQ51, q5_1, 112)
MMQ_FM_KERNEL(MmqfQ51, q5_1, 128)

MMQ_FM_KERNEL(MmqfIQ4NL, iq4_nl, 8)
MMQ_FM_KERNEL(MmqfIQ4NL, iq4_nl, 16)
MMQ_FM_KERNEL(MmqfIQ4NL, iq4_nl, 24)
MMQ_FM_KERNEL(MmqfIQ4NL, iq4_nl, 32)
MMQ_FM_KERNEL(MmqfIQ4NL, iq4_nl, 40)
MMQ_FM_KERNEL(MmqfIQ4NL, iq4_nl, 48)
MMQ_FM_KERNEL(MmqfIQ4NL, iq4_nl, 64)
MMQ_FM_KERNEL(MmqfIQ4NL, iq4_nl, 80)
MMQ_FM_KERNEL(MmqfIQ4NL, iq4_nl, 96)
MMQ_FM_KERNEL(MmqfIQ4NL, iq4_nl, 112)
MMQ_FM_KERNEL(MmqfIQ4NL, iq4_nl, 128)

MMQ_FM_KERNEL(MmqfIQ4XS, iq4_xs, 8)
MMQ_FM_KERNEL(MmqfIQ4XS, iq4_xs, 16)
MMQ_FM_KERNEL(MmqfIQ4XS, iq4_xs, 24)
MMQ_FM_KERNEL(MmqfIQ4XS, iq4_xs, 32)
MMQ_FM_KERNEL(MmqfIQ4XS, iq4_xs, 40)
MMQ_FM_KERNEL(MmqfIQ4XS, iq4_xs, 48)
MMQ_FM_KERNEL(MmqfIQ4XS, iq4_xs, 64)
MMQ_FM_KERNEL(MmqfIQ4XS, iq4_xs, 80)
MMQ_FM_KERNEL(MmqfIQ4XS, iq4_xs, 96)
MMQ_FM_KERNEL(MmqfIQ4XS, iq4_xs, 112)
MMQ_FM_KERNEL(MmqfIQ4XS, iq4_xs, 128)

MMQ_FM_KERNEL(MmqfIQ2XXS, iq2_xxs, 8)
MMQ_FM_KERNEL(MmqfIQ2XXS, iq2_xxs, 16)
MMQ_FM_KERNEL(MmqfIQ2XXS, iq2_xxs, 24)
MMQ_FM_KERNEL(MmqfIQ2XXS, iq2_xxs, 32)
MMQ_FM_KERNEL(MmqfIQ2XXS, iq2_xxs, 40)
MMQ_FM_KERNEL(MmqfIQ2XXS, iq2_xxs, 48)
MMQ_FM_KERNEL(MmqfIQ2XXS, iq2_xxs, 64)
MMQ_FM_KERNEL(MmqfIQ2XXS, iq2_xxs, 80)
MMQ_FM_KERNEL(MmqfIQ2XXS, iq2_xxs, 96)
MMQ_FM_KERNEL(MmqfIQ2XXS, iq2_xxs, 112)
MMQ_FM_KERNEL(MmqfIQ2XXS, iq2_xxs, 128)

MMQ_FM_KERNEL(MmqfIQ2XS, iq2_xs, 8)
MMQ_FM_KERNEL(MmqfIQ2XS, iq2_xs, 16)
MMQ_FM_KERNEL(MmqfIQ2XS, iq2_xs, 24)
MMQ_FM_KERNEL(MmqfIQ2XS, iq2_xs, 32)
MMQ_FM_KERNEL(MmqfIQ2XS, iq2_xs, 40)
MMQ_FM_KERNEL(MmqfIQ2XS, iq2_xs, 48)
MMQ_FM_KERNEL(MmqfIQ2XS, iq2_xs, 64)
MMQ_FM_KERNEL(MmqfIQ2XS, iq2_xs, 80)
MMQ_FM_KERNEL(MmqfIQ2XS, iq2_xs, 96)
MMQ_FM_KERNEL(MmqfIQ2XS, iq2_xs, 112)
MMQ_FM_KERNEL(MmqfIQ2XS, iq2_xs, 128)

MMQ_FM_KERNEL(MmqfIQ2S, iq2_s, 8)
MMQ_FM_KERNEL(MmqfIQ2S, iq2_s, 16)
MMQ_FM_KERNEL(MmqfIQ2S, iq2_s, 24)
MMQ_FM_KERNEL(MmqfIQ2S, iq2_s, 32)
MMQ_FM_KERNEL(MmqfIQ2S, iq2_s, 40)
MMQ_FM_KERNEL(MmqfIQ2S, iq2_s, 48)
MMQ_FM_KERNEL(MmqfIQ2S, iq2_s, 64)
MMQ_FM_KERNEL(MmqfIQ2S, iq2_s, 80)
MMQ_FM_KERNEL(MmqfIQ2S, iq2_s, 96)
MMQ_FM_KERNEL(MmqfIQ2S, iq2_s, 112)
MMQ_FM_KERNEL(MmqfIQ2S, iq2_s, 128)

MMQ_FM_KERNEL(MmqfIQ3XXS, iq3_xxs, 8)
MMQ_FM_KERNEL(MmqfIQ3XXS, iq3_xxs, 16)
MMQ_FM_KERNEL(MmqfIQ3XXS, iq3_xxs, 24)
MMQ_FM_KERNEL(MmqfIQ3XXS, iq3_xxs, 32)
MMQ_FM_KERNEL(MmqfIQ3XXS, iq3_xxs, 40)
MMQ_FM_KERNEL(MmqfIQ3XXS, iq3_xxs, 48)
MMQ_FM_KERNEL(MmqfIQ3XXS, iq3_xxs, 64)
MMQ_FM_KERNEL(MmqfIQ3XXS, iq3_xxs, 80)
MMQ_FM_KERNEL(MmqfIQ3XXS, iq3_xxs, 96)
MMQ_FM_KERNEL(MmqfIQ3XXS, iq3_xxs, 112)
MMQ_FM_KERNEL(MmqfIQ3XXS, iq3_xxs, 128)

MMQ_FM_KERNEL(MmqfIQ3S, iq3_s, 8)
MMQ_FM_KERNEL(MmqfIQ3S, iq3_s, 16)
MMQ_FM_KERNEL(MmqfIQ3S, iq3_s, 24)
MMQ_FM_KERNEL(MmqfIQ3S, iq3_s, 32)
MMQ_FM_KERNEL(MmqfIQ3S, iq3_s, 40)
MMQ_FM_KERNEL(MmqfIQ3S, iq3_s, 48)
MMQ_FM_KERNEL(MmqfIQ3S, iq3_s, 64)
MMQ_FM_KERNEL(MmqfIQ3S, iq3_s, 80)
MMQ_FM_KERNEL(MmqfIQ3S, iq3_s, 96)
MMQ_FM_KERNEL(MmqfIQ3S, iq3_s, 112)
MMQ_FM_KERNEL(MmqfIQ3S, iq3_s, 128)

MMQ_FM_KERNEL(MmqfIQ1S, iq1_s, 8)
MMQ_FM_KERNEL(MmqfIQ1S, iq1_s, 16)
MMQ_FM_KERNEL(MmqfIQ1S, iq1_s, 24)
MMQ_FM_KERNEL(MmqfIQ1S, iq1_s, 32)
MMQ_FM_KERNEL(MmqfIQ1S, iq1_s, 40)
MMQ_FM_KERNEL(MmqfIQ1S, iq1_s, 48)
MMQ_FM_KERNEL(MmqfIQ1S, iq1_s, 64)
MMQ_FM_KERNEL(MmqfIQ1S, iq1_s, 80)
MMQ_FM_KERNEL(MmqfIQ1S, iq1_s, 96)
MMQ_FM_KERNEL(MmqfIQ1S, iq1_s, 112)
MMQ_FM_KERNEL(MmqfIQ1S, iq1_s, 128)

// Reads Q8_1 sub-block `b` of both operands into registers, one iteration
// ahead, same reason as `mmq_q8_0_stage_load`.
static __device__ __forceinline__ void mmq_q4_k_stage_load(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    unsigned int M, unsigned int N, unsigned int sub_blocks, unsigned int supers,
    unsigned int b, unsigned int row0, unsigned int col0, unsigned int base,
    unsigned int stage_k4,
    int (&w_packed)[MMQ_W_STAGES], float (&w_sd)[MMQ_W_STAGES],
    float (&w_sm)[MMQ_W_STAGES],
    int (&a_packed)[MMQ_A_STAGES], float (&a_d)[MMQ_A_STAGES]
) {
    const unsigned int sup = b / 8;
    const unsigned int j = b % 8;
#pragma unroll
    for (int i = 0; i < MMQ_W_STAGES; ++i) {
        const unsigned int gcol = col0 + base + i * MMQ_STAGE_STRIDE;
        w_packed[i] = 0;
        w_sd[i] = 0.0f;
        w_sm[i] = 0.0f;
        if (gcol < N) {
            const unsigned char* blk = weight + ((unsigned long long)gcol * supers + sup) * 144;
            const float d = __half2float(*reinterpret_cast<const __half*>(blk));
            const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
            int scale;
            int minimum;
            q4k_scale_min(blk + 4, (int)j, &scale, &minimum);
            w_sd[i] = d * (float)scale;
            w_sm[i] = dmin * (float)minimum;
            // 144 is 16-aligned and every offset here is a multiple of 4, so a
            // plain int load is aligned.
            const int v = *reinterpret_cast<const int*>(blk + 16 + (j / 2) * 32 + stage_k4 * 4);
            w_packed[i] = (j & 1) ? ((v >> 4) & 0x0F0F0F0F) : (v & 0x0F0F0F0F);
        }
    }
#pragma unroll
    for (int i = 0; i < MMQ_A_STAGES; ++i) {
        const unsigned int grow = row0 + base + i * MMQ_STAGE_STRIDE;
        a_packed[i] = 0;
        a_d[i] = 0.0f;
        if (grow < M) {
            const unsigned char* blk =
                q8_act + ((unsigned long long)grow * sub_blocks + b) * 36;
            a_d[i] = __half2float(*reinterpret_cast<const __half*>(blk));
            a_packed[i] = *reinterpret_cast<const int*>(blk + 4 + stage_k4 * 4);
        }
    }
}

// Q4_K x Q8_1 -> f32 (MMQ), tensor-core variant of `quant_mmq_q4_k_q8_1` in
// `quant_gemv.cu`. Only the compute loop and the output write use
// `mma_m16n8k32_s8` instead of `dp4a`.
//
// Q4_K adds an asymmetric minimum term, `-dmin * minimum * rowsum`, that one
// `mma` cannot produce: it depends only on the row, not the column, so it is
// reduced separately with `dp4a` the same way the dp4a kernel reduces it.
extern "C" __global__ __launch_bounds__(MMQ_THREADS) void quant_mmq_q4_k_q8_1_mma(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    __shared__ int s_w[MMQ_K4][MMQ_BN + MMQ_SMEM_PAD];
    __shared__ int s_a[MMQ_K4][MMQ_BM + MMQ_SMEM_PAD];
    __shared__ float s_wd[MMQ_BN];  // d * scale
    __shared__ float s_wm[MMQ_BN];  // dmin * minimum
    __shared__ float s_ad[MMQ_BM];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int warp = tid / WARP_SIZE;

    const unsigned int row0 = blockIdx.y * MMQ_BM;
    const unsigned int col0 = blockIdx.x * MMQ_BN;

    const unsigned int sub_blocks = K / 32;      // Q8_1 blocks, and Q4_K sub-blocks
    const unsigned int supers = sub_blocks / 8;  // Q4_K 256-element super-blocks

    const unsigned int stage_sub = lane / MMQ_K4;
    const unsigned int stage_k4 = lane % MMQ_K4;

    float acc[8][4];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            acc[g][l] = 0.0f;
        }
    }

    const unsigned int stage_base = warp * 4 + stage_sub;
    int w_packed[MMQ_W_STAGES];
    float w_sd[MMQ_W_STAGES];
    float w_sm[MMQ_W_STAGES];
    int a_packed[MMQ_A_STAGES];
    float a_d[MMQ_A_STAGES];

    if (sub_blocks > 0) {
        mmq_q4_k_stage_load(q8_act, weight, M, N, sub_blocks, supers, 0, row0, col0,
                            stage_base, stage_k4, w_packed, w_sd, w_sm, a_packed, a_d);
    }

    for (unsigned int b = 0; b < sub_blocks; ++b) {
        __syncthreads();

#pragma unroll
        for (int i = 0; i < MMQ_W_STAGES; ++i) {
            const unsigned int c = stage_base + i * MMQ_STAGE_STRIDE;
            s_w[stage_k4][c] = w_packed[i];
            if (stage_k4 == 0) {
                s_wd[c] = w_sd[i];
                s_wm[c] = w_sm[i];
            }
        }
#pragma unroll
        for (int i = 0; i < MMQ_A_STAGES; ++i) {
            const unsigned int r = stage_base + i * MMQ_STAGE_STRIDE;
            s_a[stage_k4][r] = a_packed[i];
            if (stage_k4 == 0) s_ad[r] = a_d[i];
        }

        __syncthreads();

        if (b + 1 < sub_blocks) {
            mmq_q4_k_stage_load(q8_act, weight, M, N, sub_blocks, supers, b + 1, row0, col0,
                                stage_base, stage_k4, w_packed, w_sd, w_sm, a_packed, a_d);
        }

        // `mma_d_i(l)` takes only two distinct values per lane, one per half
        // of `l / 2`. `rsum2[h]` covers the two rows this lane's accumulator
        // touches, summing this sub-block's 32 activation quants per row.
        const unsigned int lane_q = threadIdx.x & 31;
        int rsum2[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = warp * 16 + h * 8 + (int)(lane_q / 4);
            int s = 0;
#pragma unroll
            for (int k4 = 0; k4 < MMQ_K4; ++k4) {
                s = dp4a(0x01010101, s_a[k4][row], s);
            }
            rsum2[h] = s;
        }

        int A[4];
        for (int l = 0; l < 4; ++l) {
            A[l] = s_a[mma_a_j(l)][warp * 16 + mma_a_i(l)];
        }

        for (int g = 0; g < 8; ++g) {
            int B[2];
            for (int l = 0; l < 2; ++l) {
                B[l] = s_w[mma_b_j(l)][g * 8 + mma_b_i(l)];
            }

            int D[4] = {0, 0, 0, 0};
            mma_m16n8k32_s8(D, A, B);

            for (int l = 0; l < 4; ++l) {
                const int row = warp * 16 + mma_d_i(l);
                const int col = g * 8 + mma_d_j(l);
                acc[g][l] +=
                    s_ad[row] * (s_wd[col] * (float)D[l] - s_wm[col] * (float)rsum2[l / 2]);
            }
        }
    }

    for (int g = 0; g < 8; ++g) {
        for (int l = 0; l < 4; ++l) {
            const unsigned int r = row0 + warp * 16 + mma_d_i(l);
            const unsigned int c = col0 + g * 8 + mma_d_j(l);
            if (r < M && c < N) output[(unsigned long long)r * N + c] = acc[g][l];
        }
    }
}

// Reads Q8_1 sub-block `b` of both operands into registers, one iteration
// ahead, same reason as `mmq_q8_0_stage_load`.
static __device__ __forceinline__ void mmq_q6_k_stage_load(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    unsigned int M, unsigned int N, unsigned int sub_blocks, unsigned int supers,
    unsigned int b, unsigned int row0, unsigned int col0, unsigned int base,
    unsigned int stage_k4,
    int (&w_packed)[MMQ_W_STAGES], float (&w_sd_lo)[MMQ_W_STAGES],
    float (&w_sd_hi)[MMQ_W_STAGES],
    int (&a_packed)[MMQ_A_STAGES], float (&a_d)[MMQ_A_STAGES]
) {
    const unsigned int sup = b / 8;
    const unsigned int j = b % 8;
    const unsigned int half = j / 4;
    const unsigned int t = j % 4;
#pragma unroll
    for (int i = 0; i < MMQ_W_STAGES; ++i) {
        const unsigned int gcol = col0 + base + i * MMQ_STAGE_STRIDE;
        w_packed[i] = 0;
        w_sd_lo[i] = 0.0f;
        w_sd_hi[i] = 0.0f;
        if (gcol < N) {
            const unsigned char* blk = weight + ((unsigned long long)gcol * supers + sup) * 210;
            const unsigned char* ql = blk + half * 64;
            const unsigned char* qh = blk + 128 + half * 32;
            const signed char* sc = reinterpret_cast<const signed char*>(blk + 192) + half * 8;
            __half d_h;
            memcpy(&d_h, blk + 208, 2);
            const float d = __half2float(d_h);
            w_sd_lo[i] = d * (float)sc[t * 2];
            w_sd_hi[i] = d * (float)sc[t * 2 + 1];

            // 210 is only 2-byte aligned, so the 4-byte reads are unaligned.
            const unsigned int e0 = stage_k4 * 4;
            const int ql4 = load_int_ua(ql + ((t & 1) ? e0 + 32 : e0));
            const int qh4 = load_int_ua(qh + e0);
            const int low = (ql4 >> ((t & 2) ? 4 : 0)) & 0x0F0F0F0F;
            const int high = ((qh4 >> (t * 2)) & 0x03030303) << 4;
            // The 6-bit value is unsigned 0..63 biased by 32.
            w_packed[i] = __vsubss4(low | high, 0x20202020);
        }
    }
#pragma unroll
    for (int i = 0; i < MMQ_A_STAGES; ++i) {
        const unsigned int grow = row0 + base + i * MMQ_STAGE_STRIDE;
        a_packed[i] = 0;
        a_d[i] = 0.0f;
        if (grow < M) {
            const unsigned char* blk =
                q8_act + ((unsigned long long)grow * sub_blocks + b) * 36;
            a_d[i] = __half2float(*reinterpret_cast<const __half*>(blk));
            a_packed[i] = *reinterpret_cast<const int*>(blk + 4 + stage_k4 * 4);
        }
    }
}

// Q6_K x Q8_1 -> f32 (MMQ), tensor-core variant of `quant_mmq_q6_k_q8_1` in
// `quant_gemv.cu`. Only the compute loop and the output write differ.
//
// Q6_K's scale changes every 16 elements, so one 32-wide `mma` cannot express
// it. Two `m16n8k16` calls run instead, one per 16-element half, each scaled
// by its own `s_wd_lo` / `s_wd_hi`.
extern "C" __global__ __launch_bounds__(MMQ_THREADS) void quant_mmq_q6_k_q8_1_mma(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    __shared__ int s_w[MMQ_K4][MMQ_BN + MMQ_SMEM_PAD];
    __shared__ float s_wd_lo[MMQ_BN];  // d * scale for elements 0..15
    __shared__ float s_wd_hi[MMQ_BN];  // d * scale for elements 16..31
    __shared__ int s_a[MMQ_K4][MMQ_BM + MMQ_SMEM_PAD];
    __shared__ float s_ad[MMQ_BM];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int warp = tid / WARP_SIZE;

    const unsigned int row0 = blockIdx.y * MMQ_BM;
    const unsigned int col0 = blockIdx.x * MMQ_BN;

    const unsigned int sub_blocks = K / 32;
    const unsigned int supers = sub_blocks / 8;

    const unsigned int stage_sub = lane / MMQ_K4;
    const unsigned int stage_k4 = lane % MMQ_K4;

    float acc[8][4];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            acc[g][l] = 0.0f;
        }
    }

    const unsigned int stage_base = warp * 4 + stage_sub;
    int w_packed[MMQ_W_STAGES];
    float w_sd_lo[MMQ_W_STAGES];
    float w_sd_hi[MMQ_W_STAGES];
    int a_packed[MMQ_A_STAGES];
    float a_d[MMQ_A_STAGES];

    if (sub_blocks > 0) {
        mmq_q6_k_stage_load(q8_act, weight, M, N, sub_blocks, supers, 0, row0, col0,
                            stage_base, stage_k4, w_packed, w_sd_lo, w_sd_hi, a_packed, a_d);
    }

    for (unsigned int b = 0; b < sub_blocks; ++b) {
        __syncthreads();

#pragma unroll
        for (int i = 0; i < MMQ_W_STAGES; ++i) {
            const unsigned int c = stage_base + i * MMQ_STAGE_STRIDE;
            s_w[stage_k4][c] = w_packed[i];
            if (stage_k4 == 0) {
                s_wd_lo[c] = w_sd_lo[i];
                s_wd_hi[c] = w_sd_hi[i];
            }
        }
#pragma unroll
        for (int i = 0; i < MMQ_A_STAGES; ++i) {
            const unsigned int r = stage_base + i * MMQ_STAGE_STRIDE;
            s_a[stage_k4][r] = a_packed[i];
            if (stage_k4 == 0) s_ad[r] = a_d[i];
        }

        __syncthreads();

        if (b + 1 < sub_blocks) {
            mmq_q6_k_stage_load(q8_act, weight, M, N, sub_blocks, supers, b + 1, row0, col0,
                                stage_base, stage_k4, w_packed, w_sd_lo, w_sd_hi, a_packed, a_d);
        }

        // Words 0..3 of the staged block are the low 16-element half, words
        // 4..7 the high half, the same split the dp4a kernel makes at
        // `k4 < 4`. Each half runs its own `m16n8k16` and takes its own scale.
        int A_lo[2];
        int A_hi[2];
        for (int l = 0; l < 2; ++l) {
            A_lo[l] = s_a[mma_a16_j(l)][warp * 16 + mma_a16_i(l)];
            A_hi[l] = s_a[4 + mma_a16_j(l)][warp * 16 + mma_a16_i(l)];
        }

        for (int g = 0; g < 8; ++g) {
            int B_lo[1];
            int B_hi[1];
            B_lo[0] = s_w[mma_b16_j(0)][g * 8 + mma_b16_i(0)];
            B_hi[0] = s_w[4 + mma_b16_j(0)][g * 8 + mma_b16_i(0)];

            int D_lo[4] = {0, 0, 0, 0};
            int D_hi[4] = {0, 0, 0, 0};
            mma_m16n8k16_s8(D_lo, A_lo, B_lo);
            mma_m16n8k16_s8(D_hi, A_hi, B_hi);

            for (int l = 0; l < 4; ++l) {
                const int row = warp * 16 + mma_d_i(l);
                const int col = g * 8 + mma_d_j(l);
                acc[g][l] += s_ad[row] * (s_wd_lo[col] * (float)D_lo[l]
                                          + s_wd_hi[col] * (float)D_hi[l]);
            }
        }
    }

    for (int g = 0; g < 8; ++g) {
        for (int l = 0; l < 4; ++l) {
            const unsigned int r = row0 + warp * 16 + mma_d_i(l);
            const unsigned int c = col0 + g * 8 + mma_d_j(l);
            if (r < M && c < N) output[(unsigned long long)r * N + c] = acc[g][l];
        }
    }
}
