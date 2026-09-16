// Group-wide helpers for register-tiled attention forwards.
//
// A GROUP is G consecutive lanes of a warp, aligned to G, that together own a
// set of query rows: lane `g` holds head dims {4*(g + G*c) .. +4}, so a dot
// product over the head dimension is a sum of G partial sums, one per lane.
// `flash_v2.cu` and `mqa_gqa.cu` both map query rows this way and share these
// helpers; the kernels themselves differ only in masking and head mapping.
//
// Every helper here is called by all 32 lanes of the warp (full mask), and the
// key-axis helpers assume BN is a power of two with BN % G == 0.

#pragma once

#include <cuda_runtime.h>

template<int G>
__device__ __forceinline__ float group_max(float v) {
    #pragma unroll
    for (int d = 1; d < G; d <<= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, d));
    }
    return v;
}

template<int G>
__device__ __forceinline__ float group_sum(float v) {
    #pragma unroll
    for (int d = 1; d < G; d <<= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, d);
    }
    return v;
}

// One reduce-scatter step over the key axis: of the N keys per row a lane
// holds, it keeps the half selected by `upper` (compacted to the front) and
// adds the partner's copy of that half. Partner = lane ^ DIST.
template<int R, int BN, int N, int DIST>
__device__ __forceinline__ void reduce_scatter_step(float (&s)[R][BN], bool upper) {
    #pragma unroll
    for (int t = 0; t < R; ++t) {
        #pragma unroll
        for (int i = 0; i < N / 2; ++i) {
            const float lo = s[t][i];
            const float hi = s[t][i + N / 2];
            const float send = upper ? lo : hi;
            const float keep = upper ? hi : lo;
            s[t][i] = keep + __shfl_xor_sync(0xffffffffu, send, DIST);
        }
    }
}

// Steps DIST = G/2, G/4, ..., 1. Afterwards lane `g` holds keys
// [g*BN/G, (g+1)*BN/G) of every row in s[t][0 .. BN/G).
template<int R, int BN, int N, int DIST>
__device__ __forceinline__ void reduce_scatter_keys(float (&s)[R][BN], int g) {
    if constexpr (DIST >= 1) {
        reduce_scatter_step<R, BN, N, DIST>(s, (g & DIST) != 0);
        reduce_scatter_keys<R, BN, N / 2, DIST / 2>(s, g);
    }
}

// One all-gather step: N keys per row become 2N, the lane's own keys landing
// in the upper or lower half according to `upper`.
template<int R, int BN, int N, int DIST>
__device__ __forceinline__ void all_gather_step(float (&s)[R][BN], bool upper) {
    #pragma unroll
    for (int t = 0; t < R; ++t) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            const float mine = s[t][i];
            const float other = __shfl_xor_sync(0xffffffffu, mine, DIST);
            s[t][i] = upper ? other : mine;
            s[t][i + N] = upper ? mine : other;
        }
    }
}

// Steps DIST = 1, 2, ..., G/2: the inverse of `reduce_scatter_keys`.
template<int R, int BN, int N, int DIST, int G>
__device__ __forceinline__ void all_gather_keys(float (&s)[R][BN], int g) {
    if constexpr (DIST < G) {
        all_gather_step<R, BN, N, DIST>(s, (g & DIST) != 0);
        all_gather_keys<R, BN, N * 2, DIST * 2, G>(s, g);
    }
}

// Partial scores of R rows against key `j` of a staged K tile, over this
// lane's CL float4 chunks. One K float4 read feeds 4*R FMAs.
template<int HEAD_DIM, int G, int R, int CL, int BN>
__device__ __forceinline__ void group_qk_partial(
    float (&s)[R][BN], const float (&q)[R][CL][4], const float* K_smem, int j, int g
) {
    #pragma unroll
    for (int c = 0; c < CL; ++c) {
        const float4 kv = *reinterpret_cast<const float4*>(
            K_smem + j * HEAD_DIM + 4 * (g + G * c));
        #pragma unroll
        for (int t = 0; t < R; ++t) {
            s[t][j] = fmaf(q[t][c][0], kv.x, s[t][j]);
            s[t][j] = fmaf(q[t][c][1], kv.y, s[t][j]);
            s[t][j] = fmaf(q[t][c][2], kv.z, s[t][j]);
            s[t][j] = fmaf(q[t][c][3], kv.w, s[t][j]);
        }
    }
}

// O += p[t] * V[j] over this lane's CL float4 chunks. One V float4 read feeds
// 4*R FMAs.
template<int HEAD_DIM, int G, int R, int CL, int BN>
__device__ __forceinline__ void group_pv_accumulate(
    float (&o)[R][CL][4], const float (&s)[R][BN], const float* V_smem, int j, int g
) {
    #pragma unroll
    for (int c = 0; c < CL; ++c) {
        const float4 vv = *reinterpret_cast<const float4*>(
            V_smem + j * HEAD_DIM + 4 * (g + G * c));
        #pragma unroll
        for (int t = 0; t < R; ++t) {
            o[t][c][0] = fmaf(s[t][j], vv.x, o[t][c][0]);
            o[t][c][1] = fmaf(s[t][j], vv.y, o[t][c][1]);
            o[t][c][2] = fmaf(s[t][j], vv.z, o[t][c][2]);
            o[t][c][3] = fmaf(s[t][j], vv.w, o[t][c][3]);
        }
    }
}
