// Interleaved multi-section RoPE (IMROPE), fused.
//
// One launch replaces the composed op in `impl_generic/position/mrope.rs`:
// per token, the four position streams (t, h, w, e) select through the
// one-hot `selector` the table row each rotated pair reads; the rotation is
// split-half over the first `n_rot` dims and dims `n_rot..head_dim` pass
// through.
//
// Bit-for-bit contract with the composed op:
// - The table entry is `sum_s table[pos[s]][j] * selector[s][j]`, seeded at
//   +0.0 and accumulated in stream order with one rounding per multiply and
//   per add — the values the mul-then-sum chain produces, including the sign
//   of a zero.
// - An out-of-range position reads a zero row, as the embedding gather does.
// - Each product is rounded before the sum or difference. The round-to-
//   nearest intrinsics keep fast-math from fusing a product into an FMA the
//   chain, with a kernel boundary between mul and add, never formed.
//
// Layout:
// - x: a [B, S, H, D] view with unit stride along D, read through
//   (x_stride_b, x_stride_s, x_stride_h) in elements
// - out: dense [B, S, H, D]
// - cos_table, sin_table: dense [max_pos, n_rot / 2]
// - positions: dense [4, S] i32
// - selector: dense [4, n_rot / 2] f32
//
// One thread per (b, s, h, i) with i < n_rot / 2 + (D - n_rot): the first
// n_rot / 2 threads of a head rotate pair (i, i + n_rot / 2), the rest copy
// one pass-through element each.

#include <cuda_runtime.h>

extern "C" __global__ void mrope_interleaved_f32(
    const float* __restrict__ x,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int* __restrict__ positions,
    const float* __restrict__ selector,
    float* __restrict__ out,
    const int batch,
    const int seq,
    const int heads,
    const int head_dim,
    const int n_rot,
    const int max_pos,
    const int x_stride_b,
    const int x_stride_s,
    const int x_stride_h
) {
    const int half_rot = n_rot / 2;
    const int work = half_rot + (head_dim - n_rot);
    const int total = batch * seq * heads * work;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int i = idx % work;
    int rem = idx / work;
    const int h = rem % heads;
    rem /= heads;
    const int s = rem % seq;
    const int b = rem / seq;

    const int src = b * x_stride_b + s * x_stride_s + h * x_stride_h;
    const int dst = ((b * seq + s) * heads + h) * head_dim;

    if (i >= half_rot) {
        const int d = n_rot + (i - half_rot);
        out[dst + d] = x[src + d];
        return;
    }

    float c = 0.0f;
    float sn = 0.0f;
    #pragma unroll
    for (int stream = 0; stream < 4; ++stream) {
        const int pos = positions[stream * seq + s];
        const float sel = selector[stream * half_rot + i];
        float ct = 0.0f;
        float snt = 0.0f;
        if (pos >= 0 && pos < max_pos) {
            ct = cos_table[pos * half_rot + i];
            snt = sin_table[pos * half_rot + i];
        }
        c = __fadd_rn(c, __fmul_rn(ct, sel));
        sn = __fadd_rn(sn, __fmul_rn(snt, sel));
    }

    const float x1 = x[src + i];
    const float x2 = x[src + i + half_rot];
    out[dst + i] = __fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, sn));
    out[dst + i + half_rot] = __fadd_rn(__fmul_rn(x1, sn), __fmul_rn(x2, c));
}
