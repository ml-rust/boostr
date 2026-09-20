// Single-token (M = 1) entry points of the feature-major MMQ family:
// `quant_mmq_<fmt>_q8_1_gemv1` for Q8_0, PQ2_0, Q2_0, Q1_0 and PTQ1_0, and
// the shared split-range fixup `quant_mmq_q8_1_gemv1_fixup`. The body and
// the contract it keeps with `quant_mmq_mma.cu` are in
// `mmq/gemv1_body.cuh`: same int8 lanes, same exact int dot per chunk, same
// scales, same float expression, same chunk order, same K ranges, so a
// decode step here is the same bits as row 0 of any batch through the
// tensor-core kernels.
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
// the load round trips. The lowbit formats fit the 64-register cap eight
// blocks imply without spilling (a 48-register cap spills). Q8_0's 27 KB
// of staging bounds it at three blocks per SM, so its cap is set there.
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

// Format-neutral: it reads only the partials.
extern "C" __global__ __launch_bounds__(GEMV1_FIXUP_THREADS) void quant_mmq_q8_1_gemv1_fixup(
    float* __restrict__ output, const float* __restrict__ workspace, unsigned int N,
    unsigned int splits
) {
    gemv1_fixup_body(output, workspace, N, splits);
}
