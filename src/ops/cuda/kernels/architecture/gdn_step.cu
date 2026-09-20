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
// Layout (all contiguous, row-major):
//   q, k:      [B, 1, H, S_k]
//   v:         [B, 1, H, S_v]
//   g, beta:   [B, 1, H]
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

#define GDN_STEP_BLOCK 256

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
        qs_s[i] = q_bh[i] * q_scale;
    }
    __syncthreads();

    const int j = blockIdx.x * GDN_STEP_BLOCK + threadIdx.x;
    if (j >= s_v) return;

    const float dec = expf(g[bh]);
    const float b = beta[bh];
    const float vj = v_bh[j];

    // Pass 1: decay the column into registers and take k . S1[:, j].
    float col[SK];
    float sk = 0.0f;
#pragma unroll
    for (int i = 0; i < SK; ++i) {
        const float s = s_bh[(size_t)i * s_v + j] * dec;
        col[i] = s;
        sk += k_s[i] * s;
    }

    const float d = (vj - sk) * b;

    // Pass 2: rank-1 update, write the new column, take qs . S2[:, j].
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < SK; ++i) {
        const float s2 = col[i] + k_s[i] * d;
        s_out_bh[(size_t)i * s_v + j] = s2;
        acc += qs_s[i] * s2;
    }

    o_bh[j] = acc;
}

extern "C" {

__global__ void __launch_bounds__(GDN_STEP_BLOCK) gdn_step_f32_sk32(
    const float* q, const float* k, const float* v,
    const float* g, const float* beta, const float* state,
    float* o, float* state_out, int s_v, float q_scale
) {
    gdn_step_impl<32>(q, k, v, g, beta, state, o, state_out, s_v, q_scale);
}

__global__ void __launch_bounds__(GDN_STEP_BLOCK) gdn_step_f32_sk64(
    const float* q, const float* k, const float* v,
    const float* g, const float* beta, const float* state,
    float* o, float* state_out, int s_v, float q_scale
) {
    gdn_step_impl<64>(q, k, v, g, beta, state, o, state_out, s_v, q_scale);
}

__global__ void __launch_bounds__(GDN_STEP_BLOCK) gdn_step_f32_sk128(
    const float* q, const float* k, const float* v,
    const float* g, const float* beta, const float* state,
    float* o, float* state_out, int s_v, float q_scale
) {
    gdn_step_impl<128>(q, k, v, g, beta, state, o, state_out, s_v, q_scale);
}

} // extern "C"
