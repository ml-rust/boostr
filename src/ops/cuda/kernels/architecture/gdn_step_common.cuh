// Shared pieces of the fused Gated DeltaNet decode kernels.
//
// `gdn_step_column` is the per-column delta-rule update both kernel
// families run. `gdn_l2_norm_block` and the gate helpers are the prologue
// the `_v2` kernels run on the raw conv output. Every float operation in
// the prologue names its rounding explicitly (`__fadd_rn`, `__fmul_rn`,
// `fmaf`) so the compiler cannot contract or reassociate it: the goal is
// bit equality with the separate numr launches the prologue replaces.
//
// The reduction reproduces numr's `reduce_sum_dim_f32` for a reduced axis
// of `SK` elements: a serial walk for `SK <= 32` (numr's `_serial` kernel)
// and the 256-thread shared-memory halving tree otherwise. The clamp is
// numr's `min(max(x, eps), +inf)` with `max` written `a > b ? a : b`; the
// `min` against `+inf` is the identity on every value `max` can produce
// and is left out.

#ifndef GDN_STEP_COMMON_CUH
#define GDN_STEP_COMMON_CUH

#define GDN_STEP_BLOCK 256

// Longest axis numr reduces serially; above it the halving tree runs.
#define GDN_SERIAL_REDUCE_MAX 32

// Sum of squares of `x[0..SK)`, in the order numr's `sum` kernel adds them.
// Every thread of the block takes part and every thread receives the sum.
// `red` is a GDN_STEP_BLOCK-wide scratch array. `x` is only read by threads
// `tid < SK` (tree) or by every thread (serial).
template <int SK>
__device__ __forceinline__ float gdn_sum_sq_block(const float* __restrict__ x, float* red) {
    const unsigned int tid = threadIdx.x;
    if (SK <= GDN_SERIAL_REDUCE_MAX) {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < SK; ++i) {
            const float xi = x[i];
            acc = __fadd_rn(acc, __fmul_rn(xi, xi));
        }
        return acc;
    }
    float own = 0.0f;
    if (tid < SK) {
        const float xi = x[tid];
        own = __fmul_rn(xi, xi);
    }
    red[tid] = own;
    __syncthreads();
    for (unsigned int s = GDN_STEP_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] = __fadd_rn(red[tid], red[tid + s]);
        }
        __syncthreads();
    }
    const float total = red[0];
    __syncthreads();
    return total;
}

// `max(sqrt(sum_sq), eps)`: numr's `sqrt` kernel then the lower half of
// its `clamp`.
__device__ __forceinline__ float gdn_l2_floor(float sum_sq, float eps) {
    const float n = sqrtf(sum_sq);
    return n > eps ? n : eps;
}

// numr's `sigmoid_f32`.
__device__ __forceinline__ float gdn_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// numr's `softplus_impl`: `relu(x) + log(1 + exp(-|x|))`, each step its own
// kernel there, each rounding named here.
__device__ __forceinline__ float gdn_softplus(float x) {
    const float relu_x = fmaxf(0.0f, x);
    const float e = expf(-fabsf(x));
    const float log_term = logf(__fadd_rn(e, 1.0f));
    return __fadd_rn(relu_x, log_term);
}

// One state column of the gated delta rule:
//
//   S1[:, j] = S[:, j] * dec
//   sk       = k . S1[:, j]
//   d        = (v[j] - sk) * beta
//   S2[:, j] = S1[:, j] + k * d
//   o[j]     = qs . S2[:, j]
//
// `k_s` and `qs_s` are the block's shared copies of `k` and `q * q_scale`.
// `s_col`, `s_out_col` point at row 0 of column `j`; rows are `s_v` apart.
template <int SK>
__device__ __forceinline__ void gdn_step_column(
    const float* __restrict__ k_s,
    const float* __restrict__ qs_s,
    const float* __restrict__ s_col,
    float* __restrict__ s_out_col,
    int s_v,
    float dec,
    float beta,
    float vj,
    float* __restrict__ oj
) {
    float col[SK];
    float sk = 0.0f;
#pragma unroll
    for (int i = 0; i < SK; ++i) {
        const float s = __fmul_rn(s_col[(size_t)i * s_v], dec);
        col[i] = s;
        sk = fmaf(k_s[i], s, sk);
    }

    const float d = __fmul_rn(__fsub_rn(vj, sk), beta);

    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < SK; ++i) {
        const float s2 = fmaf(k_s[i], d, col[i]);
        s_out_col[(size_t)i * s_v] = s2;
        acc = fmaf(qs_s[i], s2, acc);
    }

    *oj = acc;
}

#endif // GDN_STEP_COMMON_CUH
