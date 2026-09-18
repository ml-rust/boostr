// Flash Attention v2 forward - the general kernel
// Based on "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
// Tri Dao, 2023 (https://arxiv.org/abs/2307.08691)
//
// Serves every forward shape no dedicated kernel takes: head dims 32, 64, 96,
// 128, 192 and 256, GQA (KV heads broadcast over query heads), sliding window
// (`window_size > 0`), causal, and query lengths down to 1. F32 accumulation
// for every T. FP8 lives in the separate `flash_v2_fp8.cu`.
//
// Query positions are ABSOLUTE: query row `i` sits at sequence position
// `key_offset + i`, where `key_offset = seq_len_k - seq_len_q`. A KV-cached
// decode or chunked prefill passes seq_len_q < seq_len_k, and those queries are
// the LAST seq_len_q positions of the key sequence, so causal and sliding-window
// masking must both use that absolute position. Prefill (seq_len_q == seq_len_k)
// gives key_offset == 0 and leaves the masks unchanged. Same convention as
// `ops/impl_generic/attention/flash_standard.rs::build_attention_mask`.
//
// Thread mapping - a query row is owned by a GROUP of G lanes, not one thread:
// - Lane `g` of a group owns head dims {4*(g + G*c) .. +4 : c < HEAD_DIM/(4*G)},
//   so at a fixed `c` the G lanes of a group read G consecutive float4 of one
//   K/V row - one shared-memory wavefront, no bank conflicts, no padding.
// - Each group owns R consecutive query rows. Q and the O accumulator for those
//   rows live in REGISTERS (R * HEAD_DIM / G floats each); Q is read from
//   global memory once per block and never staged.
// - A warp holds 32/G groups, so ROWS_PER_WARP = R * 32 / G, and a block of
//   WARPS warps owns WARPS * ROWS_PER_WARP consecutive query rows.
//
// Per K/V tile of BLOCK_N keys:
// 1. All threads stage the K and V tiles into shared memory as float. That is
//    the ONLY shared memory: 2 * BLOCK_N * HEAD_DIM floats, independent of T.
// 2. Each lane accumulates PARTIAL scores s[R][BLOCK_N] over its own dims.
// 3. The partial scores are reduce-scattered across the G lanes, so lane `g`
//    ends holding the full scores of keys [g*BLOCK_N/G, (g+1)*BLOCK_N/G) for
//    its R rows. Masking (tail, causal, window), scaling, the row max, exp and
//    the row sum happen on those; max/sum are all-reduced across the group.
// 4. The probabilities are all-gathered back to s[R][BLOCK_N] on every lane,
//    and each lane accumulates O for its own dims.
// Every dot product is a sum of G partial sums, each over HEAD_DIM/G dims, so
// the accumulation order differs from a serial sum. The group helpers live in
// `group_tile.cuh`, shared with `mqa_gqa.cu`.
//
// Whole-tile skips are block-uniform, so they may cross the barriers:
// - Sliding window: a tile is skipped when it ends before the window of the
//   block's FIRST query row, whose window reaches furthest back.
// - Causal: the loop stops at the first tile that starts past the block's
//   LAST query position.
// A tile a block keeps can still be fully masked for one of its rows; the
// online softmax treats that as an exact no-op (see the loop body).
//
// Left padding: `kv_start` is a `[B]` I32 device array or null. Keys below
// `kv_start[b]` are invalid for every row of batch `b`. The block reads its
// start once, clamps it to `[0, seq_len_k]`, and anchors the tile grid at
// it: tile `kt` holds keys `start + kt * BLOCK_N ..`, the loop runs to the
// last tile of the `seq_len_k - start` valid keys, and the window skip counts
// tiles on that grid. Every key index in the mask is taken RELATIVE to the
// start (`key_rel = key - start`), so the tail check `key_rel >= seq_len_k -
// start` as an unsigned compare, the causal and window bounds and the
// online-softmax sequence are the ones an unpadded run over the same keys
// forms: a padded row's output is that run's output bit for bit. With a
// null pointer the start is 0 and every expression is the unpadded one. A
// row with no valid key keeps `l == 0` and stores zeros with `LSE = -inf`.
//
// LSE is [B, H, S_q], one value m + log(l) per row, in F32. `flash_v2_bwd.cu`
// consumes it unchanged.
//
// O is [B, H, S_q, D] by default; `out_token_major != 0` stores the same
// values at [B, S_q, H, D] addresses instead.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"
#include "group_tile.cuh"

extern __shared__ __align__(16) float flash_fwd_smem[];

// ============================================================================
// Flash Attention forward - templated implementation
//
// Grid: (batch_size * num_heads, ceil(seq_len_q / ROWS))
// Block: WARPS * 32 threads, ROWS = WARPS * R * 32 / G query rows.
// Dynamic shared memory: 2 * BLOCK_N * HEAD_DIM * sizeof(float).
// ============================================================================

template<typename T, int HEAD_DIM, int G, int R, int WARPS, int BLOCK_N>
__device__ void flash_attention_fwd_impl(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    float* __restrict__ L,
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,  // GQA: can be less than num_heads
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal,
    const int window_size,   // Sliding window: 0 or -1 = full attention, >0 = local window
    const int* __restrict__ kv_start,
    const int out_token_major
) {
    static_assert(HEAD_DIM % (4 * G) == 0, "each lane owns whole float4 chunks");
    static_assert(BLOCK_N % G == 0, "keys split evenly across the group");
    static_assert(32 % G == 0, "groups tile a warp");

    constexpr int THREADS = WARPS * 32;
    constexpr int ROWS_PER_WARP = (32 / G) * R;
    constexpr int ROWS = WARPS * ROWS_PER_WARP;
    constexpr int CL = HEAD_DIM / (4 * G);      // float4 chunks per lane
    constexpr int KPL = BLOCK_N / G;            // keys per lane after reduce-scatter
    constexpr int VEC_PER_ROW = HEAD_DIM / 4;
    constexpr int TILE_VECS = BLOCK_N * VEC_PER_ROW;

    float* K_smem = flash_fwd_smem;
    float* V_smem = flash_fwd_smem + BLOCK_N * HEAD_DIM;

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int g = lane % G;
    const int rg = lane / G;

    const int batch_head_idx = blockIdx.x;
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    // GQA: several query heads share one KV head.
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    const size_t q_offset = ((size_t)batch_idx * num_heads + head_idx) * seq_len_q * HEAD_DIM;
    const size_t kv_offset = ((size_t)batch_idx * num_kv_heads + kv_head_idx) * seq_len_k * HEAD_DIM;
    const size_t lse_offset = ((size_t)batch_idx * num_heads + head_idx) * seq_len_q;

    const T* Q_base = Q + q_offset;
    const T* K_base = K + kv_offset;
    const T* V_base = V + kv_offset;
    float* L_base = L + lse_offset;

    // Output rows: head-major is Q's layout, one head's rows contiguous.
    // Token-major stores row `r` at `o[b, r, h, :]`, so consecutive rows of a
    // head are `num_heads * HEAD_DIM` apart. Only the store address differs.
    T* O_base = O + q_offset;
    size_t o_row_stride = HEAD_DIM;
    if (out_token_major) {
        O_base = O + ((size_t)batch_idx * seq_len_q * num_heads + head_idx) * HEAD_DIM;
        o_row_stride = (size_t)num_heads * HEAD_DIM;
    }

    const int q_start = blockIdx.y * ROWS;
    const int row0 = q_start + warp * ROWS_PER_WARP + rg * R;
    const int key_offset = max(0, seq_len_k - seq_len_q);
    // Left-padding start of this batch row (see the header); 0 when unpadded.
    const int pad_start = kv_start ? min(max(kv_start[batch_idx], 0), seq_len_k) : 0;
    const int seq_len_k_rel = seq_len_k - pad_start;

    // Q rows in registers. Rows past seq_len_q read as zero and are never stored.
    float q[R][CL][4];
    float o[R][CL][4];
    float m[R];
    float l[R];
    #pragma unroll
    for (int t = 0; t < R; ++t) {
        const int row = row0 + t;
        const bool valid = row < seq_len_q;
        m[t] = -INFINITY;
        l[t] = 0.0f;
        #pragma unroll
        for (int c = 0; c < CL; ++c) {
            float4 qv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (valid) {
                qv = load4_dtype(Q_base, row * HEAD_DIM + 4 * (g + G * c));
            }
            q[t][c][0] = qv.x; q[t][c][1] = qv.y; q[t][c][2] = qv.z; q[t][c][3] = qv.w;
            o[t][c][0] = 0.0f; o[t][c][1] = 0.0f; o[t][c][2] = 0.0f; o[t][c][3] = 0.0f;
        }
    }

    // First and last absolute query positions of this block: the window skip
    // is governed by the first row, the causal stop by the last. The tile
    // grid starts at the padding start, so the skip counts tiles from there
    // and no tile holds a padded key.
    const int first_q_pos = key_offset + q_start;
    const int last_q_pos = key_offset + min(q_start + ROWS, seq_len_q) - 1;
    const int min_key_win = window_size > 0 ? max(0, first_q_pos - window_size + 1) : 0;
    const int min_key_rel = max(min_key_win - pad_start, 0);
    const int num_k_tiles = (seq_len_k_rel + BLOCK_N - 1) / BLOCK_N;

    for (int kt = min_key_rel / BLOCK_N; kt < num_k_tiles; ++kt) {
        const int k_start_rel = kt * BLOCK_N;
        const int k_start = pad_start + k_start_rel;
        if (causal && k_start > last_q_pos) break;

        // Stage K and V as float. Keys past seq_len_k stage as zero and are
        // masked below, so they never feed a NaN into the reductions.
        for (int v = tid; v < TILE_VECS; v += THREADS) {
            const int row = v / VEC_PER_ROW;
            const int col = (v % VEC_PER_ROW) * 4;
            const int key = k_start + row;
            float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 vv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (key < seq_len_k) {
                kv = load4_dtype(K_base, key * HEAD_DIM + col);
                vv = load4_dtype(V_base, key * HEAD_DIM + col);
            }
            *reinterpret_cast<float4*>(K_smem + row * HEAD_DIM + col) = kv;
            *reinterpret_cast<float4*>(V_smem + row * HEAD_DIM + col) = vv;
        }
        __syncthreads();

        // Partial scores over this lane's dims.
        float s[R][BLOCK_N];
        #pragma unroll
        for (int t = 0; t < R; ++t) {
            #pragma unroll
            for (int j = 0; j < BLOCK_N; ++j) s[t][j] = 0.0f;
        }
        #pragma unroll
        for (int j = 0; j < BLOCK_N; ++j) {
            group_qk_partial<HEAD_DIM, G, R, CL, BLOCK_N>(s, q, K_smem, j, g);
        }

        reduce_scatter_keys<R, BLOCK_N, BLOCK_N, G / 2>(s, g);

        // Lane `g` now holds keys k_start + g*KPL + i, i < KPL. Mask, scale,
        // and advance the online softmax. A tile fully masked for a row keeps
        // m at -inf, and then alpha = 1 and every p = 0 make it an exact no-op
        // instead of exp(-inf - -inf) = NaN. This must stay a computation, not
        // a `continue`: the barrier below is reached by every thread.
        // Key indices are relative to the padding start (see the header).
        #pragma unroll
        for (int t = 0; t < R; ++t) {
            const int q_pos = key_offset + row0 + t - pad_start;
            const int win_lo = q_pos - window_size + 1;
            float m_tile = -INFINITY;
            #pragma unroll
            for (int i = 0; i < KPL; ++i) {
                const int key = k_start_rel + g * KPL + i;
                const bool masked = (unsigned)key >= (unsigned)seq_len_k_rel
                                 || (causal && key > q_pos)
                                 || (window_size > 0 && key < win_lo);
                s[t][i] = masked ? -INFINITY : s[t][i] * scale;
                m_tile = fmaxf(m_tile, s[t][i]);
            }
            const float m_new = fmaxf(m[t], group_max<G>(m_tile));
            const bool dead = m_new == -INFINITY;
            const float alpha = dead ? 1.0f : __expf(m[t] - m_new);
            float l_tile = 0.0f;
            #pragma unroll
            for (int i = 0; i < KPL; ++i) {
                const float p = dead ? 0.0f : __expf(s[t][i] - m_new);
                s[t][i] = p;
                l_tile += p;
            }
            l[t] = alpha * l[t] + group_sum<G>(l_tile);
            m[t] = m_new;
            #pragma unroll
            for (int c = 0; c < CL; ++c) {
                o[t][c][0] *= alpha; o[t][c][1] *= alpha;
                o[t][c][2] *= alpha; o[t][c][3] *= alpha;
            }
        }

        all_gather_keys<R, BLOCK_N, KPL, 1, G>(s, g);

        #pragma unroll
        for (int j = 0; j < BLOCK_N; ++j) {
            group_pv_accumulate<HEAD_DIM, G, R, CL, BLOCK_N>(o, s, V_smem, j, g);
        }
        __syncthreads();
    }

    // Normalize and store. LSE layout is [B, H, S_q], one value per row.
    #pragma unroll
    for (int t = 0; t < R; ++t) {
        const int row = row0 + t;
        if (row >= seq_len_q) continue;
        const float inv_l = (l[t] == 0.0f) ? 1.0f : 1.0f / l[t];
        T* O_row = O_base + (size_t)row * o_row_stride;
        #pragma unroll
        for (int c = 0; c < CL; ++c) {
            const float4 ov = make_float4(o[t][c][0] * inv_l, o[t][c][1] * inv_l,
                                          o[t][c][2] * inv_l, o[t][c][3] * inv_l);
            store4_dtype(O_row, 4 * (g + G * c), ov);
        }
        if (g == 0) {
            L_base[row] = m[t] + __logf(l[t]);
        }
    }
}

// ============================================================================
// Kernel Entry Points
//
// Two block sizes per (head_dim, dtype), as separate symbols:
//   flash_attention_fwd_{head_dim}_{dtype}       4 warps per block
//   flash_attention_fwd_{head_dim}_sm_{dtype}    2 warps per block
// Rows per block follow from (G, R, WARPS); `flash_fwd_tile` in
// `src/ops/cuda/attention/flash/flash_block_config.rs` mirrors this table
// and picks the symbol at runtime from how many blocks the grid would have.
//
// Lanes per row G and rows per group R:
//   head_dim  32,  64: G = 4, R = 4 (8 and 16 dims per lane)
//   head_dim  96, 128: G = 8, R = 4 (12 and 16 dims per lane)
//   head_dim 192, 256: G = 8, R = 2 (24 and 32 dims per lane)
// R drops to 2 at 192 and 256 because Q and O together take
// 2 * R * HEAD_DIM / G registers per lane, and R = 4 there does not fit the
// register file without spilling.
//
// Nothing here is compile-time gated on the GPU architecture: this translation
// unit builds at sm_75, and the BF16 entries must exist there. All BF16 work
// goes through `load4_dtype` / `store4_dtype`, whose conversions have a valid
// path at every architecture.
// ============================================================================

#define FLASH_FWD_ENTRY(T, HEAD_DIM, G, R, WARPS, BLOCK_N, SUFFIX)             \
    extern "C" __global__ void __launch_bounds__(WARPS * 32)                   \
    flash_attention_fwd_##HEAD_DIM##_##SUFFIX(                                 \
        const T* Q, const T* K, const T* V,                                    \
        T* O, float* L,                                                        \
        const int batch_size, const int num_heads, const int num_kv_heads,     \
        const int seq_len_q, const int seq_len_k,                              \
        const float scale, const int causal, const int window_size,            \
        const int* kv_start, const int out_token_major                         \
    ) {                                                                        \
        flash_attention_fwd_impl<T, HEAD_DIM, G, R, WARPS, BLOCK_N>(           \
            Q, K, V, O, L, batch_size, num_heads, num_kv_heads,                \
            seq_len_q, seq_len_k, scale, causal, window_size, kv_start,        \
            out_token_major                                                    \
        );                                                                     \
    }

// Keys per staged tile, every entry. `FLASH_FWD_BLOCK_N` in
// flash_block_config.rs mirrors it; the launcher sizes shared memory from it.
#define FLASH_FWD_BLOCK_N 16

#define FLASH_FWD_DTYPE(T, SUFFIX)                                             \
    FLASH_FWD_ENTRY(T, 32, 4, 4, 4, FLASH_FWD_BLOCK_N, SUFFIX)                 \
    FLASH_FWD_ENTRY(T, 64, 4, 4, 4, FLASH_FWD_BLOCK_N, SUFFIX)                 \
    FLASH_FWD_ENTRY(T, 96, 8, 4, 4, FLASH_FWD_BLOCK_N, SUFFIX)                 \
    FLASH_FWD_ENTRY(T, 128, 8, 4, 4, FLASH_FWD_BLOCK_N, SUFFIX)                \
    FLASH_FWD_ENTRY(T, 192, 8, 2, 4, FLASH_FWD_BLOCK_N, SUFFIX)                \
    FLASH_FWD_ENTRY(T, 256, 8, 2, 4, FLASH_FWD_BLOCK_N, SUFFIX)                \
    FLASH_FWD_ENTRY(T, 32, 4, 4, 2, FLASH_FWD_BLOCK_N, sm_##SUFFIX)            \
    FLASH_FWD_ENTRY(T, 64, 4, 4, 2, FLASH_FWD_BLOCK_N, sm_##SUFFIX)            \
    FLASH_FWD_ENTRY(T, 96, 8, 4, 2, FLASH_FWD_BLOCK_N, sm_##SUFFIX)            \
    FLASH_FWD_ENTRY(T, 128, 8, 4, 2, FLASH_FWD_BLOCK_N, sm_##SUFFIX)           \
    FLASH_FWD_ENTRY(T, 192, 8, 2, 2, FLASH_FWD_BLOCK_N, sm_##SUFFIX)           \
    FLASH_FWD_ENTRY(T, 256, 8, 2, 2, FLASH_FWD_BLOCK_N, sm_##SUFFIX)

FLASH_FWD_DTYPE(float, fp32)
FLASH_FWD_DTYPE(__half, fp16)
FLASH_FWD_DTYPE(__nv_bfloat16, bf16)

#undef FLASH_FWD_DTYPE
#undef FLASH_FWD_BLOCK_N
#undef FLASH_FWD_ENTRY
