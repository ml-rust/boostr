// Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) forward
//
// MQA: 1 KV head shared across all Q heads (Llama 2, PaLM)
// GQA: Multiple KV heads, each shared across a group of Q heads (Llama 3, Mistral)
//
// Reference: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
// https://arxiv.org/abs/2305.13245
//
// Head mapping: kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads).
// Query positions are ABSOLUTE: query row `i` sits at key position
// `seq_len_k - seq_len_q + i`, same convention as flash_v2.cu.
//
// One implementation and one entry-point signature, all dtypes:
// `mqa_gqa_fwd_impl<T, HEAD_DIM, G, R, WARPS, BLOCK_N>` serves F32, F16, BF16
// and both FP8 formats. Accumulation is F32 for every T.
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
// 1. All threads stage the K and V tiles into shared memory as float
//    (converting / dequantizing at the load). That is the ONLY shared memory:
//    2 * BLOCK_N * HEAD_DIM floats, independent of T.
// 2. Each lane accumulates PARTIAL scores s[R][BLOCK_N] over its own dims.
//    One K float4 read feeds 4*R FMAs.
// 3. The partial scores are reduce-scattered across the G lanes with
//    __shfl_xor butterflies, so lane `g` ends holding the full scores of keys
//    [g*BLOCK_N/G, (g+1)*BLOCK_N/G) for its R rows. Masking, scaling, the row
//    max, exp, and the row sum happen on those; max/sum are all-reduced across
//    the group so every lane sees the same online-softmax state per row.
// 4. The probabilities are all-gathered back to s[R][BLOCK_N] on every lane,
//    and each lane accumulates O for its own dims: one V float4 read feeds
//    4*R FMAs. Every dot product is a sum of G partial sums, each over
//    HEAD_DIM/G dims, so the accumulation order differs from a serial sum.
// The group reductions, gathers and per-key FMA blocks are the helpers in
// `group_tile.cuh`, shared with `flash_v2.cu`.
//
// Causal tiles past the block's last query position are skipped as a whole;
// the skip is block-uniform, so it may `break` across the barriers.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"
#include "group_tile.cuh"

extern __shared__ __align__(16) float mqa_gqa_fwd_smem[];

// ============================================================================
// MQA/GQA Forward - templated implementation
//
// Grid: (batch_size * num_q_heads, ceil(seq_len_q / ROWS))
// Block: WARPS * 32 threads, ROWS = WARPS * R * 32 / G query rows.
// Dynamic shared memory: 2 * BLOCK_N * HEAD_DIM * sizeof(float).
//
// `q_scale` / `k_scale` / `v_scale` / `o_scale` are the FP8 dequant/quant
// scales. `load4_dtype` and `store4_dtype` ignore them for every other dtype.
// ============================================================================

template<typename T, int HEAD_DIM, int G, int R, int WARPS, int BLOCK_N>
__device__ void mqa_gqa_fwd_impl(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    float* __restrict__ L,
    const int batch_size,
    const int num_q_heads,
    const int num_kv_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const int causal,
    const float q_scale,
    const float k_scale,
    const float v_scale,
    const float o_scale
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

    float* K_smem = mqa_gqa_fwd_smem;
    float* V_smem = mqa_gqa_fwd_smem + BLOCK_N * HEAD_DIM;

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int g = lane % G;
    const int rg = lane / G;

    const int batch_head_idx = blockIdx.x;
    const int batch_idx = batch_head_idx / num_q_heads;
    const int q_head_idx = batch_head_idx % num_q_heads;
    const int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);

    const size_t q_offset = ((size_t)batch_idx * num_q_heads + q_head_idx) * seq_len_q * HEAD_DIM;
    const size_t kv_offset = ((size_t)batch_idx * num_kv_heads + kv_head_idx) * seq_len_k * HEAD_DIM;
    const size_t lse_offset = ((size_t)batch_idx * num_q_heads + q_head_idx) * seq_len_q;

    const T* Q_base = Q + q_offset;
    const T* K_base = K + kv_offset;
    const T* V_base = V + kv_offset;
    T* O_base = O + q_offset;
    float* L_base = L + lse_offset;

    const int q_start = blockIdx.y * ROWS;
    const int row0 = q_start + warp * ROWS_PER_WARP + rg * R;
    const int key_offset = max(0, seq_len_k - seq_len_q);

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
                qv = load4_dtype(Q_base, row * HEAD_DIM + 4 * (g + G * c), q_scale);
            }
            q[t][c][0] = qv.x; q[t][c][1] = qv.y; q[t][c][2] = qv.z; q[t][c][3] = qv.w;
            o[t][c][0] = 0.0f; o[t][c][1] = 0.0f; o[t][c][2] = 0.0f; o[t][c][3] = 0.0f;
        }
    }

    // Last absolute query position any row of this block can hold; causal
    // tiles that start past it are masked for the whole block.
    const int last_q_pos = key_offset + min(q_start + ROWS, seq_len_q) - 1;
    const int num_k_tiles = (seq_len_k + BLOCK_N - 1) / BLOCK_N;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k_start = kt * BLOCK_N;
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
                kv = load4_dtype(K_base, key * HEAD_DIM + col, k_scale);
                vv = load4_dtype(V_base, key * HEAD_DIM + col, v_scale);
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
        // instead of exp(-inf - -inf) = NaN.
        #pragma unroll
        for (int t = 0; t < R; ++t) {
            const int q_pos = key_offset + row0 + t;
            float m_tile = -INFINITY;
            #pragma unroll
            for (int i = 0; i < KPL; ++i) {
                const int key = k_start + g * KPL + i;
                const bool masked = key >= seq_len_k || (causal && key > q_pos);
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

        // O += P V over this lane's dims.
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
        #pragma unroll
        for (int c = 0; c < CL; ++c) {
            const float4 ov = make_float4(o[t][c][0] * inv_l, o[t][c][1] * inv_l,
                                          o[t][c][2] * inv_l, o[t][c][3] * inv_l);
            store4_dtype(O_base, row * HEAD_DIM + 4 * (g + G * c), ov, o_scale);
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
//   mqa_gqa_fwd_{head_dim}_{dtype}       4 warps per block
//   mqa_gqa_fwd_{head_dim}_{dtype}_sm    2 warps per block
// Rows per block follow from (G, R, WARPS); `mqa_fwd_tile` in
// `src/ops/cuda/attention/mqa_gqa/block_config.rs` mirrors this table and
// picks the symbol at runtime from how many blocks the grid would have.
//
// Lanes per row G: 8 at head_dim 128 (16 dims per lane), 4 at head_dim 64 and
// 32 (16 and 8 dims per lane). R = 4 rows per group everywhere.
//
// ONE signature for every dtype, including the four trailing quantization
// scales. Only the FP8 entries read them; the launcher passes 1.0f otherwise.
// ============================================================================

#define MQA_GQA_FWD_ENTRY(T, HEAD_DIM, G, R, WARPS, BLOCK_N, SUFFIX)          \
    extern "C" __global__ void __launch_bounds__(WARPS * 32)                   \
    mqa_gqa_fwd_##HEAD_DIM##_##SUFFIX(                                         \
        const T* Q, const T* K, const T* V,                                    \
        T* O, float* L,                                                        \
        const int batch_size, const int num_q_heads, const int num_kv_heads,   \
        const int seq_len_q, const int seq_len_k,                              \
        const float scale, const int causal,                                   \
        const float q_scale, const float k_scale,                              \
        const float v_scale, const float o_scale                               \
    ) {                                                                        \
        mqa_gqa_fwd_impl<T, HEAD_DIM, G, R, WARPS, BLOCK_N>(                   \
            Q, K, V, O, L, batch_size, num_q_heads, num_kv_heads,              \
            seq_len_q, seq_len_k, scale, causal,                               \
            q_scale, k_scale, v_scale, o_scale                                 \
        );                                                                     \
    }

// Keys per staged tile, every entry. `MQA_FWD_BLOCK_N` in block_config.rs
// mirrors it; the launcher sizes shared memory from that constant.
#define MQA_GQA_FWD_BLOCK_N 16

#define MQA_GQA_FWD_DTYPE(T, SUFFIX)                                           \
    MQA_GQA_FWD_ENTRY(T, 32, 4, 4, 4, MQA_GQA_FWD_BLOCK_N, SUFFIX)             \
    MQA_GQA_FWD_ENTRY(T, 64, 4, 4, 4, MQA_GQA_FWD_BLOCK_N, SUFFIX)             \
    MQA_GQA_FWD_ENTRY(T, 128, 8, 4, 4, MQA_GQA_FWD_BLOCK_N, SUFFIX)            \
    MQA_GQA_FWD_ENTRY(T, 32, 4, 4, 2, MQA_GQA_FWD_BLOCK_N, SUFFIX##_sm)        \
    MQA_GQA_FWD_ENTRY(T, 64, 4, 4, 2, MQA_GQA_FWD_BLOCK_N, SUFFIX##_sm)        \
    MQA_GQA_FWD_ENTRY(T, 128, 8, 4, 2, MQA_GQA_FWD_BLOCK_N, SUFFIX##_sm)

MQA_GQA_FWD_DTYPE(float, fp32)
MQA_GQA_FWD_DTYPE(__half, fp16)
MQA_GQA_FWD_DTYPE(__nv_bfloat16, bf16)
MQA_GQA_FWD_DTYPE(boostr_fp8_e4m3, fp8_e4m3)
MQA_GQA_FWD_DTYPE(boostr_fp8_e5m2, fp8_e5m2)

#undef MQA_GQA_FWD_DTYPE
#undef MQA_GQA_FWD_BLOCK_N
#undef MQA_GQA_FWD_ENTRY
