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
// Q8_0 x Q8_1 -> f32 (MMQ), feature-major decomposition, specialized per batch
// size.
//
// Computes the same product as `quant_mmq_q8_0_q8_1_mma` above, but with the two
// MMA operand roles swapped: the WEIGHT is operand A (16 output features per
// fragment) and the ACTIVATION is operand B (8 tokens per fragment). The block
// owns a 128-feature x MMQ_X-token output tile, and the grid axes are
// transposed relative to the kernel above.
//
// Four things follow, and they are the point of this variant:
//   - The weight tile is staged 256 k-values deep per outer iteration, so a
//     Q8_0 weight row is read in one wide pass instead of eight narrow ones.
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
// request `MMQF_SMEM_BYTES(MMQ_X)`.
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

// Weight row: 64 quant words (256 k-values), then 8 f32 block scales, then
// padding. The padding makes the stride an odd multiple of 4 ints, so
// consecutive rows start in different 128-bit segments and the strided
// fragment gathers below hit all 32 banks.
#define MMQF_X_QS 0
#define MMQF_X_DS 64
#define MMQF_X_STRIDE 76
// Word offset of the second 128-k half within a staged weight row.
#define MMQF_HALF_W 32
// Q8_0 blocks consumed per staging iteration: 256 k-values.
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

// Activation row: 4 f32 block scales, then 32 quant words (128 k-values). This
// is exactly one repacked record, which is what lets the tile be flat-copied.
#define MMQF_Y_DS 0
#define MMQF_Y_QS 4
#define MMQF_Y_STRIDE 36

#define MMQF_SMEM_BYTES(X) (4 * (MMQF_Y * MMQF_X_STRIDE + (X) * MMQF_Y_STRIDE))

// The weight row stride must be an odd multiple of 4 ints. Any multiple of 8
// puts consecutive rows in the same 128-bit segment and serializes the strided
// fragment gathers.
static_assert(MMQF_X_STRIDE % 8 == 4, "Wrong weight row padding.");
static_assert(MMQF_X_DS + 8 <= MMQF_X_STRIDE, "Weight row too short: 8 block scales.");
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
// many k-steps `mmqf_vec_dot` runs. `CLAMP_K` is false for whole 256-k groups,
// which is every iteration but the last of a ragged K, so the common path has
// no clamp at all.
//
// Each loop issues a group of independent loads into registers FIRST, then
// stores the group to shared. Interleaving load and store per iteration would
// serialize on the load.

// Stages 256 k-values (8 Q8_0 blocks) of the weight tile. One warp owns one
// feature row per step and steps by the warp count; within the row a lane maps
// to (block, 4-k word) and issues two loads, for the block at `b0` and the one
// 128 k-values further along. Independent of MMQ_X.
template <int MMQ_X, bool CLAMP_K>
static __device__ __forceinline__ void mmqf_stage_x(
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
    const unsigned long long rstride = (unsigned long long)bpr * 34;

    // Both block offsets are loop-invariant, so the addressing collapses to one
    // add per row.
    const unsigned int blk_lo = CLAMP_K ? min(b0 + kbx, bpr - 1) : b0 + kbx;
    const unsigned int blk_hi = CLAMP_K ? min(b0 + 4 + kbx, bpr - 1) : b0 + 4 + kbx;
    // The quants start at byte 2 of a 34-byte block, so only 2-byte alignment
    // holds and each word is two 16-bit loads.
    const unsigned long long off_lo = (unsigned long long)blk_lo * 34 + 2 + kqsx * 4;
    const unsigned long long off_hi = (unsigned long long)blk_hi * 34 + 2 + kqsx * 4;

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
            s_x[i * MMQF_X_STRIDE + MMQF_X_QS + lane] = v_lo[u];
            s_x[i * MMQF_X_STRIDE + MMQF_X_QS + MMQF_HALF_W + lane] = v_hi[u];
        }
    }

    // Scales are a separate pass: eight per row, so a warp covers four rows.
    float* s_xd = (float*)s_x;
    const unsigned int kbxd = lane % 8;
    const unsigned int rsub = lane / 8;
    const unsigned int blk_d = CLAMP_K ? min(b0 + kbxd, bpr - 1) : b0 + kbxd;
    const unsigned long long off_d = (unsigned long long)blk_d * 34;

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
        s_xd[i * MMQF_X_STRIDE + MMQF_X_DS + kbxd] = d[u];
    }
}
// Stages 128 k-values of the activation tile as a FLAT COPY. The repacked
// layout indexes records k-group-major, token-minor, so the `MMQ_X` records a
// token tile needs are contiguous and the shared row IS the record: 4 f32 block
// scales then 32 quant words. No per-element index math, no token bound, no
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
static __device__ __forceinline__ void mmqf_vec_dot(
    const int* __restrict__ s_x, const int* __restrict__ s_y,
    float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int i0, unsigned int jb,
    unsigned int k00, unsigned int nks
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);

    const float* s_xd = (const float*)s_x;
    const float* s_yd = (const float*)s_y;

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
                A[n][l] = s_x[(ir + mma_a_i(l)) * MMQF_X_STRIDE + MMQF_X_QS + kw + mma_a_j(l)];
            }
            // `mma_d_i(l)` takes one value per `l / 2`, so two weight scales
            // cover the whole accumulator fragment.
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                dw[n][h] = s_xd[(ir + mma_d_i(2 * h)) * MMQF_X_STRIDE + MMQF_X_DS + kw / 8];
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
            // `mma_d_j(l)` takes one value per `l % 2`.
            float da[2];
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                da[l] = s_yd[(jt + mma_d_j(l)) * MMQF_Y_STRIDE + MMQF_Y_DS + ks];
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

// Runs the k-block range `[kb0_start, kb0_stop)` of one output tile into `acc`,
// which it also zeroes. `kb0_start` is always a multiple of MMQF_ITER_B;
// `kb0_stop` is too, unless it is `bpr`, whose last partial 256-k group takes
// the early-stop path.
//
// The leading barrier makes the function self-contained: a caller may run
// several tiles back to back without draining the previous tile's shared reads
// itself.
template <int MMQ_X>
static __device__ __forceinline__ void mmqf_accumulate(
    const int* __restrict__ y_packed,
    const unsigned char* __restrict__ weight, int* __restrict__ s_x,
    int* __restrict__ s_y, float (&acc)[MMQF_NJ(MMQ_X)][MMQF_NTX(MMQ_X)][4], unsigned int ntok,
    unsigned int N, unsigned int bpr, unsigned int tok0, unsigned int feat0,
    unsigned int i0, unsigned int jb, unsigned int kb0_start, unsigned int kb0_stop
) {
    constexpr int NTX = MMQF_NTX(MMQ_X);
    constexpr int NJ = MMQF_NJ(MMQ_X);

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

    // K is only guaranteed to be a multiple of 32, so the final group of a tile
    // can hold fewer than MMQF_ITER_B blocks.
    const unsigned int full_stop = (kb0_stop == bpr) ? (bpr & ~(MMQF_ITER_B - 1u)) : kb0_stop;

    for (unsigned int b0 = kb0_start; b0 < full_stop; b0 += MMQF_ITER_B) {
        mmqf_stage_x<MMQ_X, false>(weight, s_x, N, bpr, feat0, b0);
        mmqf_stage_y<MMQ_X>(y_packed, s_y, ntok, tok0, b0);
        __syncthreads();

        mmqf_vec_dot<MMQ_X, true>(s_x, s_y, acc, i0, jb, 0, 4);
        __syncthreads();

        mmqf_stage_y<MMQ_X>(y_packed, s_y, ntok, tok0, b0 + 4);
        __syncthreads();

        mmqf_vec_dot<MMQ_X, true>(s_x, s_y, acc, i0, jb, MMQF_HALF_W, 4);
        __syncthreads();
    }

    // `bpr` and the range bounds are uniform across the block, so every barrier
    // below is still reached by all threads.
    if (full_stop < kb0_stop) {
        const unsigned int nb = kb0_stop - full_stop;
        mmqf_stage_x<MMQ_X, true>(weight, s_x, N, bpr, feat0, full_stop);
        mmqf_stage_y<MMQ_X>(y_packed, s_y, ntok, tok0, full_stop);
        __syncthreads();

        mmqf_vec_dot<MMQ_X, false>(s_x, s_y, acc, i0, jb, 0, nb < 4 ? nb : 4);

        if (nb > 4) {
            __syncthreads();
            mmqf_stage_y<MMQ_X>(y_packed, s_y, ntok, tok0, full_stop + 4);
            __syncthreads();
            mmqf_vec_dot<MMQ_X, false>(s_x, s_y, acc, i0, jb, MMQF_HALF_W, nb - 4);
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
template <int MMQ_X>
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
    int* s_y = mmqf_smem + MMQF_Y * MMQF_X_STRIDE;

    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int bpr = K / 32;  // blocks per row, both operands

    // The eight warps form a (8/NTX) x NTX grid over the output tile: a warp
    // owns NTX consecutive 16-feature minitiles and every NTX-th token group.
    const unsigned int i0 = (warp / NTX) * (NTX * 16);
    const unsigned int jb = (warp % NTX) * 8;

    const unsigned int tok0 = blockIdx.x * MMQ_X;
    const unsigned int feat0 = blockIdx.y * MMQF_Y;

    float acc[NJ][NTX][4];
    mmqf_accumulate<MMQ_X>(y_packed, weight, s_x, s_y, acc, ntok, N, bpr, tok0, feat0, i0, jb, 0,
                           bpr);
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
template <int MMQ_X>
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
    int* s_y = mmqf_smem + MMQF_Y * MMQF_X_STRIDE;

    const unsigned int warp = threadIdx.x / WARP_SIZE;
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

        mmqf_accumulate<MMQ_X>(y_packed, weight, s_x, s_y, acc, ntok, N, bpr, tok0, feat0, i0, jb,
                               (unsigned int)kb0_start, (unsigned int)kb0_stop);
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

    mmqf_accumulate<MMQ_X>(y_packed, weight, s_x, s_y, acc, ntok, N, bpr, tok0, feat0, i0, jb,
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

// One entry point per token tile, for each of the three roles. The host picks
// the variant; the rule it must follow is fixed by the `MMQ_X % granularity`
// assert in `mmqf_body`.
#define MMQ_FM_KERNEL(X)                                                             \
    extern "C" __global__ __launch_bounds__(MMQF_THREADS)                             \
        void quant_mmq_q8_0_q8_1_mma_x##X(                                            \
            const int* __restrict__ y_packed,                                         \
            const unsigned char* __restrict__ weight, float* __restrict__ output,     \
            unsigned int M, unsigned int K, unsigned int N, unsigned int ntok         \
        ) {                                                                           \
        mmqf_body<X>(y_packed, weight, output, M, K, N, ntok);                        \
    }                                                                                 \
    extern "C" __global__ __launch_bounds__(MMQF_THREADS)                             \
        void quant_mmq_q8_0_q8_1_mma_sk_x##X(                                         \
            const int* __restrict__ y_packed,                                         \
            const unsigned char* __restrict__ weight, float* __restrict__ output,     \
            float* __restrict__ workspace, unsigned int M, unsigned int K,            \
            unsigned int N, unsigned int ntok                                         \
        ) {                                                                           \
        mmqf_sk_body<X>(y_packed, weight, output, workspace, M, K, N, ntok);          \
    }                                                                                 \
    extern "C" __global__ __launch_bounds__(MMQF_THREADS)                             \
        void quant_mmq_q8_0_q8_1_mma_fixup_x##X(                                      \
            float* __restrict__ output, const float* __restrict__ workspace,          \
            unsigned int M, unsigned int K, unsigned int N                            \
        ) {                                                                           \
        mmqf_fixup_body<X>(output, workspace, M, K, N);                               \
    }

MMQ_FM_KERNEL(8)
MMQ_FM_KERNEL(16)
MMQ_FM_KERNEL(24)
MMQ_FM_KERNEL(32)
MMQ_FM_KERNEL(40)
MMQ_FM_KERNEL(48)
MMQ_FM_KERNEL(64)
MMQ_FM_KERNEL(80)
MMQ_FM_KERNEL(96)
MMQ_FM_KERNEL(112)
MMQ_FM_KERNEL(128)

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
