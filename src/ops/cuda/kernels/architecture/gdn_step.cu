// Fused Gated DeltaNet decode step, F32.
//
// One token of the gated delta rule per (batch, head), with the state
// `S: [S_k, S_v]` read once and written once:
//
//   qs = q * (1 / sqrt(S_k))
//   dec = exp(g)
//   S1 = S * dec
//   sk = k @ S1                    [S_v]
//   d = (v - sk) * beta            [S_v]
//   S2 = S1 + k^T (x) d            S2[i][j] = S1[i][j] + k[i] * d[j]
//   o = qs @ S2                    [S_v]
//
// Column independence: every quantity indexed by j depends only on column
// j of S plus the shared vectors k, qs and the scalars dec, beta. sk[j] is
// the dot of k with column j, d[j] uses sk[j] and v[j], column j of S2 uses
// d[j], and o[j] is the dot of qs with column j of S2. No column reads
// another, so one thread owns one column end to end and the block needs
// no reduction and no synchronisation past the initial k/q load.
//
// Two kernel families share the column update (`gdn_step_common.cuh`):
//
// `gdn_step_f32_sk*` takes q, k, v, g, beta already prepared:
//   q, k:      [B, 1, H, S_k]     L2-normalized, repeated to H value heads
//   v:         [B, 1, H, S_v]
//   g, beta:   [B, 1, H]
//
// `gdn_step_v2_f32_sk*` takes the post-SiLU conv output and the raw gate
// projections and runs the whole per-token chain in the prologue:
//   qkv:       [B, 1, 2 * key_dim + value_dim]
//              q at [h_k * S_k], k at [key_dim + h_k * S_k],
//              v at [2 * key_dim + h_v * S_v], with h_k = h_v % H_k
//   alpha_raw: [B, 1, H_v]        g = ssm_a * softplus(alpha_raw + dt_bias)
//   beta_raw:  [B, 1, H_v]        beta = sigmoid(beta_raw)
//   dt_bias:   [H_v]
//   ssm_a:     [H_v]
//   q, k are L2-normalized over S_k with floor `eps`.
//
// Both:
//   state:     [B, H, S_k, S_v]
//   o:         [B, 1, H, S_v]
//   state_out: [B, H, S_k, S_v]
//
// Grid: x = column chunk of GDN_STEP_BLOCK threads, y = b * H + h.
// Block: GDN_STEP_BLOCK threads; thread t owns column
// j = blockIdx.x * GDN_STEP_BLOCK + t. Consecutive threads touch
// consecutive columns, so every state read and write is coalesced.
//
// S_k is a compile-time constant so the column stays in registers across
// the two passes. Instantiated for S_k in {32, 64, 128}.

#include "gdn_step_common.cuh"

template <int SK>
__device__ __forceinline__ void gdn_step_impl(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    const float* __restrict__ state,
    float* __restrict__ o,
    float* __restrict__ state_out,
    int s_v,
    float q_scale
) {
    __shared__ float k_s[SK];
    __shared__ float qs_s[SK];

    const int bh = blockIdx.y;
    const float* q_bh = q + (size_t)bh * SK;
    const float* k_bh = k + (size_t)bh * SK;
    const float* v_bh = v + (size_t)bh * s_v;
    const float* s_bh = state + (size_t)bh * SK * s_v;
    float* o_bh = o + (size_t)bh * s_v;
    float* s_out_bh = state_out + (size_t)bh * SK * s_v;

    // Cooperative load of the per-head vectors. Every thread takes part,
    // including those whose column is past s_v.
    for (int i = threadIdx.x; i < SK; i += blockDim.x) {
        k_s[i] = k_bh[i];
        qs_s[i] = __fmul_rn(q_bh[i], q_scale);
    }
    __syncthreads();

    const int j = blockIdx.x * GDN_STEP_BLOCK + threadIdx.x;
    if (j >= s_v) return;

    const float dec = expf(g[bh]);
    gdn_step_column<SK>(k_s, qs_s, s_bh + j, s_out_bh + j, s_v, dec, beta[bh], v_bh[j], o_bh + j);
}

template <int SK>
__device__ __forceinline__ void gdn_step_v2_impl(
    const float* __restrict__ qkv,
    const float* __restrict__ alpha_raw,
    const float* __restrict__ beta_raw,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    const float* __restrict__ state,
    float* __restrict__ o,
    float* __restrict__ state_out,
    int s_v,
    int h_v,
    int h_k,
    int key_dim,
    int value_dim,
    float eps,
    float q_scale
) {
    __shared__ float k_s[SK];
    __shared__ float qs_s[SK];
    __shared__ float red_q[GDN_STEP_BLOCK];
    __shared__ float red_k[GDN_STEP_BLOCK];

    const int bh = blockIdx.y;
    const int b = bh / h_v;
    const int hv = bh - b * h_v;
    const int hk = hv % h_k;
    const size_t qkv_dim = (size_t)2 * key_dim + value_dim;
    const float* row = qkv + (size_t)b * qkv_dim;
    const float* q_row = row + (size_t)hk * SK;
    const float* k_row = row + (size_t)key_dim + (size_t)hk * SK;
    const float* v_row = row + (size_t)2 * key_dim + (size_t)hv * s_v;
    const float* s_bh = state + (size_t)bh * SK * s_v;
    float* o_bh = o + (size_t)bh * s_v;
    float* s_out_bh = state_out + (size_t)bh * SK * s_v;

    // L2 norms: every thread takes part in the block reduction.
    const float q_floor = gdn_l2_floor(gdn_sum_sq_block<SK>(q_row, red_q), eps);
    const float k_floor = gdn_l2_floor(gdn_sum_sq_block<SK>(k_row, red_k), eps);

    for (int i = threadIdx.x; i < SK; i += blockDim.x) {
        k_s[i] = k_row[i] / k_floor;
        qs_s[i] = __fmul_rn(q_row[i] / q_floor, q_scale);
    }
    __syncthreads();

    const int j = blockIdx.x * GDN_STEP_BLOCK + threadIdx.x;
    if (j >= s_v) return;

    const float beta = gdn_sigmoid(beta_raw[bh]);
    const float sp = gdn_softplus(__fadd_rn(alpha_raw[bh], dt_bias[hv]));
    const float dec = expf(__fmul_rn(sp, ssm_a[hv]));
    gdn_step_column<SK>(k_s, qs_s, s_bh + j, s_out_bh + j, s_v, dec, beta, v_row[j], o_bh + j);
}

extern "C" {

#define GDN_STEP_KERNEL(SK)                                                    \
    __global__ void __launch_bounds__(GDN_STEP_BLOCK) gdn_step_f32_sk##SK(     \
        const float* q, const float* k, const float* v,                        \
        const float* g, const float* beta, const float* state,                 \
        float* o, float* state_out, int s_v, float q_scale                     \
    ) {                                                                        \
        gdn_step_impl<SK>(q, k, v, g, beta, state, o, state_out, s_v, q_scale); \
    }

#define GDN_STEP_V2_KERNEL(SK)                                                 \
    __global__ void __launch_bounds__(GDN_STEP_BLOCK) gdn_step_v2_f32_sk##SK(  \
        const float* qkv, const float* alpha_raw, const float* beta_raw,       \
        const float* dt_bias, const float* ssm_a, const float* state,          \
        float* o, float* state_out, int s_v, int h_v, int h_k,                 \
        int key_dim, int value_dim, float eps, float q_scale                   \
    ) {                                                                        \
        gdn_step_v2_impl<SK>(qkv, alpha_raw, beta_raw, dt_bias, ssm_a, state,  \
                             o, state_out, s_v, h_v, h_k, key_dim, value_dim,  \
                             eps, q_scale);                                    \
    }

GDN_STEP_KERNEL(32)
GDN_STEP_KERNEL(64)
GDN_STEP_KERNEL(128)

GDN_STEP_V2_KERNEL(32)
GDN_STEP_V2_KERNEL(64)
GDN_STEP_V2_KERNEL(128)

} // extern "C"
