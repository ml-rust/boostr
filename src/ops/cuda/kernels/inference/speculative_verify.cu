// Sampled verification for speculative decoding (Leviathan et al., 2023).
//
// One block per batch row. `blockDim.x` must be a power of two, from 32 to 1024.
// The rules match `impl_generic::inference::speculative`:
// - Draft i is accepted iff its id is in [0, V), q_i(d_i) > 0 and u_i * q_i(d_i) < p_i(d_i).
// - m counts the leading accepts.
// - m < K draws from max(0, p_m - q_m). A one-hot q_m zeroes p_m at d_m.
//   A zero residual falls back to p_m. m == K draws from p_K.
// - The draw uses u_K: the first id with mass whose running sum exceeds u_K * total.
//   `total` is the last running sum. No crossing returns the last id with mass.
//
// `draft` is null when `has_draft` is 0: each draft distribution is one-hot.
// `tokens` is [B, K] int32. `out` is [B, 2] int32: (m, next).

#include <cstdint>

#define SV_MAX_THREADS 1024

// Entry j of the distribution `next` is drawn from.
__device__ __forceinline__ float sv_dist(
    const float* p_row,
    const float* q_row,
    int d_m,
    int use_residual,
    int has_draft,
    int j
) {
    const float p = p_row[j];
    if (!use_residual) {
        return p;
    }
    if (has_draft) {
        return fmaxf(__fsub_rn(p, q_row[j]), 0.0f);
    }
    return (j == d_m) ? 0.0f : p;
}

// Tree reduction of one int per thread. `op` 0 is min, 1 is max.
__device__ int sv_reduce_int(int* scratch, int value, int op) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const int other = scratch[threadIdx.x + s];
            const int mine = scratch[threadIdx.x];
            scratch[threadIdx.x] = op == 0 ? min(mine, other) : max(mine, other);
        }
        __syncthreads();
    }
    const int result = scratch[0];
    __syncthreads();
    return result;
}

extern "C" __global__ void speculative_verify_f32(
    const float* __restrict__ target,      // [B, K+1, V]
    const float* __restrict__ draft,       // [B, K, V], or null
    const int* __restrict__ tokens,        // [B, K]
    const float* __restrict__ uniforms,    // [B, K+1]
    int* __restrict__ out,                 // [B, 2]
    int drafts,
    int vocab,
    int has_draft
) {
    __shared__ float s_prefix[SV_MAX_THREADS];
    __shared__ int s_int[SV_MAX_THREADS];
    __shared__ int s_m;
    __shared__ int s_dm;
    __shared__ float s_total;

    const size_t b = blockIdx.x;
    const size_t v = (size_t)vocab;
    const float* p_all = target + b * (size_t)(drafts + 1) * v;
    const float* q_all = has_draft ? draft + b * (size_t)drafts * v : nullptr;
    const int* d = tokens + b * (size_t)drafts;
    const float* u = uniforms + b * (size_t)(drafts + 1);

    // The accept scan is serial by definition: it stops at the first reject.
    if (threadIdx.x == 0) {
        int m = 0;
        for (; m < drafts; ++m) {
            const int t = d[m];
            const bool in_range = t >= 0 && t < vocab;
            const size_t col = in_range ? (size_t)t : 0;
            const float p = p_all[(size_t)m * v + col];
            const float q = has_draft ? q_all[(size_t)m * v + col] : 1.0f;
            if (!(in_range && q > 0.0f && __fmul_rn(u[m], q) < p)) {
                break;
            }
        }
        s_m = m;
        s_dm = m < drafts ? d[m] : -1;
    }
    __syncthreads();

    const int m = s_m;
    const int d_m = s_dm;
    const float* p_row = p_all + (size_t)m * v;
    const int q_pos = m < drafts ? m : 0;
    const float* q_row = has_draft ? q_all + (size_t)q_pos * v : nullptr;

    // A residual with any positive entry is used. `m` is uniform across the block.
    int use_residual = 0;
    if (m < drafts) {
        int any = 0;
        for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
            if (sv_dist(p_row, q_row, d_m, 1, has_draft, j) > 0.0f) {
                any = 1;
            }
        }
        use_residual = __syncthreads_or(any);
    }

    // Pass 1: each thread sums its contiguous chunk and notes its last id with mass.
    const int chunk = (vocab + (int)blockDim.x - 1) / (int)blockDim.x;
    const int lo = min((int)threadIdx.x * chunk, vocab);
    const int hi = min(lo + chunk, vocab);
    float local = 0.0f;
    int last = -1;
    for (int j = lo; j < hi; ++j) {
        const float x = sv_dist(p_row, q_row, d_m, use_residual, has_draft, j);
        local = __fadd_rn(local, x);
        if (x > 0.0f) {
            last = j;
        }
    }
    s_prefix[threadIdx.x] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        float run = 0.0f;
        for (unsigned t = 0; t < blockDim.x; ++t) {
            const float x = s_prefix[t];
            s_prefix[t] = run;
            run = __fadd_rn(run, x);
        }
    }
    __syncthreads();
    const float prefix = s_prefix[threadIdx.x];

    // Pass 2: the owner of id V-1 publishes the last running sum as the total.
    if (lo < hi && hi == vocab) {
        float run = prefix;
        for (int j = lo; j < hi; ++j) {
            run = __fadd_rn(run, sv_dist(p_row, q_row, d_m, use_residual, has_draft, j));
        }
        s_total = run;
    }
    __syncthreads();
    const float threshold = __fmul_rn(u[drafts], s_total);

    // Pass 3: the first crossing in this chunk.
    int first = vocab;
    float run = prefix;
    for (int j = lo; j < hi; ++j) {
        const float x = sv_dist(p_row, q_row, d_m, use_residual, has_draft, j);
        run = __fadd_rn(run, x);
        if (x > 0.0f && run > threshold) {
            first = j;
            break;
        }
    }
    first = sv_reduce_int(s_int, first, 0);
    last = sv_reduce_int(s_int, last, 1);

    if (threadIdx.x == 0) {
        const int next = first < vocab ? first : max(last, 0);
        out[b * 2] = m;
        out[b * 2 + 1] = next;
    }
}
