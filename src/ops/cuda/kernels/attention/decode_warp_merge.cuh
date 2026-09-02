// Warp-partial merge shared by the contiguous and paged decode attention
// kernels.
//
// Both walk the KV sequence with one warp per position, so each warp ends with
// its own `(acc, m, l)` over a disjoint subset of positions. Combining those is
// the same rescaling the split-KV combine kernel does across slices, and it is
// the only cross-warp step either kernel has.

#ifndef BOOSTR_DECODE_WARP_MERGE_CUH
#define BOOSTR_DECODE_WARP_MERGE_CUH

// Lanes per warp. Also the number of head dimensions a warp covers per load,
// which makes both `D / DECODE_LANES` — warps per block, and dimensions per
// lane.
#define DECODE_LANES 32

// Merges the per-warp partials in shared memory into one `(acc, m, l)`.
//
// `smem_acc` is `[NW][D]`, `smem_m` and `smem_l` are `[NW]`. Thread `tid` owns
// output dimension `tid`. A warp that saw no positions has `l == 0` and is
// dropped without contributing to the maximum, so an empty warp cannot turn the
// rescaling into `exp(-inf - -inf)`.
//
// The caller must `__syncthreads()` between publishing the partials and calling
// this.
template<int D>
__device__ __forceinline__ void decode_merge_warps(
    const float* __restrict__ smem_acc,
    const float* __restrict__ smem_m,
    const float* __restrict__ smem_l,
    float& acc, float& m, float& l
) {
    constexpr int NW = D / DECODE_LANES;
    const int tid = threadIdx.x;

    float m_max = -INFINITY;
    #pragma unroll
    for (int w = 0; w < NW; w++)
        if (smem_l[w] > 0.0f) m_max = fmaxf(m_max, smem_m[w]);

    float acc_sum = 0.0f;
    float l_sum = 0.0f;
    #pragma unroll
    for (int w = 0; w < NW; w++) {
        float l_w = smem_l[w];
        if (l_w <= 0.0f) continue;
        float weight = expf(smem_m[w] - m_max);
        acc_sum += smem_acc[w * D + tid] * weight;
        l_sum += l_w * weight;
    }

    acc = acc_sum;
    m = m_max;
    l = l_sum;
}

#endif  // BOOSTR_DECODE_WARP_MERGE_CUH
