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

static_assert(MMQF_Y_QS + 32 == MMQF_Y_STRIDE, "Wrong activation row stride.");

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

// Weight format policy. The format-generic machinery below reaches the weight
// tile only through one of these: the on-disk block geometry, the staged
// weight-row layout, and the two functions that touch weight data. Q8_0, Q4_K
// and Q6_K are the instantiations.
struct MmqfQ80 {
    // On-disk block: one f16 scale then 32 int8 quants.
    static constexpr int BLOCK_BYTES = 34;
    static constexpr int BLOCK_ELEMS = 32;
    // K is gated only on `k % 32 == 0`, so a row's last 256-k group can hold
    // fewer than MMQF_ITER_B blocks and the tail path must be compiled.
    static constexpr bool RAGGED_K = true;

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

    // Consumes one staged 128-k half. `k00` is the word offset of that half inside
    // a weight row; the activation tile always holds the half at word 0. `i0` is
    // the warp's feature base, `jb` its token base within a group.
    //
    // `FULL` is the whole-half case (four 32-k steps); the tail instantiation stops
    // early rather than running steps whose scales are zero.
    //
    // The k-step loop is outermost so only one A fragment per minitile and its two
    // scales are live at a time.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
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
                    B[l] = s_y[(jt + mma_b_i(l)) * MMQF_Y_STRIDE + MMQF_Y_QS + ks * 8
                               + mma_b_j(l)];
                }
                // `mma_d_j(l)` takes one value per `l % 2`. `MMQF_Y_STRIDE`/`MMQF_Y_DS`
                // are int counts and a header word is one int, so the index is
                // unchanged. The scale is the LOW half of the header word; the high
                // half is the int16 quant sum, which this format has no use for and
                // never touches. The producer already rounds `d` through `half`, so
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
};

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

    // Consumes one staged 128-k half; the loop shape matches `MmqfQ80::vec_dot`
    // exactly, four 32-k steps of one `mma_m16n8k32_s8` per (token group,
    // minitile). Two scalar terms per fragment element instead of one:
    //
    //   acc += (d * sc)     * d_act * int32 dot
    //   acc += (-dmin * m)  * d_act * int32 block sum
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
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
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
                    dmw[n][h] =
                        s_xdm[((ir + mma_d_i(2 * h)) * X_STRIDE + X_DM) / 2 + kw / 8];
                }
            }

#pragma unroll
            for (int jj = 0; jj < NJ; ++jj) {
                const unsigned int jt = jj * (NTX * 8) + jb;

                int B[2];
#pragma unroll
                for (int l = 0; l < 2; ++l) {
                    B[l] = s_y[(jt + mma_b_i(l)) * MMQF_Y_STRIDE + MMQF_Y_QS + ks * 8
                               + mma_b_j(l)];
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
                    const int ds =
                        s_y[(jt + mma_d_j(l)) * MMQF_Y_STRIDE + MMQF_Y_DS + ks];
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
};

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

    // Consumes one staged 128-k half, four 32-k steps, each split into two
    // 16-k `mma_m16n8k16_s8` calls because the scale changes at 16. Words
    // `kw .. kw+3` are the low half and take scale `kw / 4`; words
    // `kw+4 .. kw+7` are the high half and take scale `kw / 4 + 1`.
    //
    // Q6_K has no minimum term, so the int16 block sum in the HIGH half of the
    // activation header word is never read here; only the `half` activation
    // scale in the low half is.
    //
    // The k-step loop is outermost so only one A fragment pair and its two
    // scale pairs are live at a time. ggml hoists a `scA[ntx][ne/2][8]`
    // register array across the whole tile instead; that costs more registers
    // than this kernel's accumulator leaves free.
    template <int MMQ_X, bool FULL>
    static __device__ __forceinline__ void vec_dot(
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
                    const unsigned int r = (ir + mma_d_i(2 * h)) * X_STRIDE + X_DF + kw / 4;
                    sc_lo[n][h] = s_xdf[r];
                    sc_hi[n][h] = s_xdf[r + 1];
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
                // quant sum in bits 16..31. Only the scale is read. The
                // producer rounds `d` through `half`, so reading the low half
                // back as float is an exact round-trip. `mma_d_j(l)` takes one
                // value per `l % 2`, so two words cover the fragment. Neither
                // read depends on the feature minitile, so both stay outside
                // the `n` loop below.
                float da[2];
#pragma unroll
                for (int l = 0; l < 2; ++l) {
                    const int ds =
                        s_y[(jt + mma_d_j(l)) * MMQF_Y_STRIDE + MMQF_Y_DS + ks];
                    da[l] = __half2float(__ushort_as_half((unsigned short)(ds & 0xFFFF)));
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
                    }
                }
            }
        }
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

    // Q8_0's K is only guaranteed to be a multiple of one block, so the final
    // group of a tile can hold fewer than MMQF_ITER_B blocks. A format that
    // guarantees whole 256-k groups sets `RAGGED_K` false, which folds this to
    // `kb0_stop` and drops the tail instantiations below.
    const unsigned int full_stop =
        (FMT::RAGGED_K && kb0_stop == bpr) ? (bpr & ~(MMQF_ITER_B - 1u)) : kb0_stop;

    for (unsigned int b0 = kb0_start; b0 < full_stop; b0 += MMQF_ITER_B) {
        FMT::template stage<MMQ_X, false>(weight, s_x, N, bpr, feat0, b0);
        mmqf_stage_y<MMQ_X>(y_packed, s_y, ntok, tok0, b0);
        __syncthreads();

        FMT::template vec_dot<MMQ_X, true>(s_x, s_y, acc, i0, jb, 0, 4);
        __syncthreads();

        mmqf_stage_y<MMQ_X>(y_packed, s_y, ntok, tok0, b0 + 4);
        __syncthreads();

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

            FMT::template vec_dot<MMQ_X, false>(s_x, s_y, acc, i0, jb, 0, nb < 4 ? nb : 4);

            if (nb > 4) {
                __syncthreads();
                mmqf_stage_y<MMQ_X>(y_packed, s_y, ntok, tok0, full_stop + 4);
                __syncthreads();
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

// Copied verbatim from `quant_gemv.cu`, including the `__CUDA_ARCH__ >= 610`
// guard, so the row-sum reduction below matches the dp4a kernel exactly.
static __device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const signed char* a8 = (const signed char*)&a;
    const signed char* b8 = (const signed char*)&b;
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif
}

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
