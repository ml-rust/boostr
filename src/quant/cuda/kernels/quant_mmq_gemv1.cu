// Single-token (M = 1) entry points of the feature-major MMQ family:
// `quant_mmq_<fmt>_q8_1_gemv1` for Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL,
// IQ4_XS, PQ2_0, Q2_0, Q1_0, PTQ1_0, Q4_K, Q5_K, Q6_K, Q3_K and Q2_K, and
// the shared split-range fixup
// `quant_mmq_q8_1_gemv1_fixup`. The body and the contract it keeps with
// `quant_mmq_mma.cu` are in `mmq/gemv1_body.cuh`: same int8 lanes, same
// exact int dot per chunk, same scales, same float expression, same chunk
// order, same K ranges, so a decode step here is the same bits as row 0 of
// any batch through the tensor-core kernels.
//
// Launch: grid `(ceil(N / 32), splits)`, block `GEMV1_THREADS`; the
// staging buffers are static shared memory, so the launch requests none.
// `splits` is the count `split_count(k, n, sms)` gives the
// tensor-core launch; `workspace` holds `(splits - 1) * N` floats and is
// unused at one split. The fixup runs after, on the same stream, with grid
// `ceil(N / GEMV1_FIXUP_THREADS)` and block `GEMV1_FIXUP_THREADS`.
//
// Arguments: `y_packed` is the activation in the feature-major record
// layout (`quantize_f32_q8_1_mmq`) with token stride `ntok`; `K` and `N` are
// the weight's; `output` is `[1, N]` f32.

#include "mmq/gemv1_body.cuh"

// `MIN_BLOCKS` caps the register allocation so `MIN_BLOCKS` blocks stay
// resident per SM: the walk is latency bound, so residency is what hides
// the load round trips. Staging bytes per block bound it from above and the
// register cap the count implies bounds it from below; every kernel here
// sits at the highest count that still spills nothing.
//
//   27 KB (Q8_0)                                              3 blocks
//   19 KB (the legacy 32-element formats, IQ4_XS, the 4-, 5-  5 blocks
//          and 6-bit K-quants)
//   11 KB (the lowbit formats)                                8 blocks
//   11 KB (Q3_K, Q2_K — the widest readers at the narrowest   7, 6 blocks
//          span, so registers bind before staging does)
#define MMQ_GEMV1_KERNEL(FMT, NAME, MIN_BLOCKS)                                      \
    extern "C" __global__ __launch_bounds__(GEMV1_THREADS, MIN_BLOCKS)               \
        void quant_mmq_##NAME##_q8_1_gemv1(                                          \
            const int* __restrict__ y_packed,                                        \
            const unsigned char* __restrict__ weight, float* __restrict__ output,    \
            float* __restrict__ workspace, unsigned int K, unsigned int N,           \
            unsigned int ntok                                                        \
        ) {                                                                          \
        gemv1_body<FMT>(y_packed, weight, output, workspace, K, N, ntok);            \
    }

MMQ_GEMV1_KERNEL(Gemv1Q80, q8_0, 3)
MMQ_GEMV1_KERNEL(Gemv1PQ20, pq2_0, 8)
MMQ_GEMV1_KERNEL(Gemv1Q20, q2_0, 8)
MMQ_GEMV1_KERNEL(Gemv1Q10, q1_0, 8)
MMQ_GEMV1_KERNEL(Gemv1PTQ10, ptq1_0, 8)
MMQ_GEMV1_KERNEL(Gemv1Q4K, q4_k, 5)
MMQ_GEMV1_KERNEL(Gemv1Q5K, q5_k, 5)
MMQ_GEMV1_KERNEL(Gemv1Q6K, q6_k, 5)
MMQ_GEMV1_KERNEL(Gemv1Q3K, q3_k, 7)
MMQ_GEMV1_KERNEL(Gemv1Q2K, q2_k, 6)
MMQ_GEMV1_KERNEL(Gemv1Q40, q4_0, 5)
MMQ_GEMV1_KERNEL(Gemv1Q50, q5_0, 5)
MMQ_GEMV1_KERNEL(Gemv1IQ4NL, iq4_nl, 5)
MMQ_GEMV1_KERNEL(Gemv1IQ4XS, iq4_xs, 5)
MMQ_GEMV1_KERNEL(Gemv1Q41, q4_1, 5)
MMQ_GEMV1_KERNEL(Gemv1Q51, q5_1, 5)

// Format-neutral: it reads only the partials.
extern "C" __global__ __launch_bounds__(GEMV1_FIXUP_THREADS) void quant_mmq_q8_1_gemv1_fixup(
    float* __restrict__ output, const float* __restrict__ workspace, unsigned int N,
    unsigned int splits
) {
    gemv1_fixup_body(output, workspace, N, splits);
}
