// One half step of the single-token (M = 1) MMQ body: the four 32-element
// chunks a warp's four chunk slots hold for one feature, their int dots,
// their scale constants and the owner lane's float epilogue. Included by
// `gemv1_body.cuh` only; the staging and the K walk are there, the weight
// readers are in `gemv1_formats.cuh`, `gemv1_legacy.cuh` and
// `gemv1_kquant.cuh`.
//
// A reader's `EPILOGUE` tag picks the shape:
//
//   GEMV1_EPI_D     One int dot over the chunk's 32 elements and one weight
//                   scale, the form `mmqf_vec_dot_d` applies. The owner
//                   reads the four scales itself.
//   GEMV1_EPI_DM    One int dot plus the rank-1 minimum term, the form
//                   `mmqf_vec_dot_dm` applies. Each lane decodes its own
//                   chunk's `(d * scale, -dmin * minimum)` pair; the owner
//                   gathers three values per chunk.
//   GEMV1_EPI_SC16  Two int dots, one per 16 elements, each with its own
//                   `d * scale`, the form `mmqf_vec_dot_sc16` applies with
//                   `MIN` false. Each lane forms both dots and both scales;
//                   the owner gathers four values per chunk.
//   GEMV1_EPI_SC16_MIN
//                   The same at `MIN` true: each 16-element half also takes
//                   a minimum term against the activation's int sum over ITS
//                   OWN 16 elements. Each lane forms its own `(d * scale,
//                   -dmin * minimum)` pair per half and the int sum of the
//                   chunk's FIRST 16 activation quants; the owner gathers
//                   seven values per chunk and recovers the second half's sum
//                   as `s32 - s_a` from the record header's exact int16 sum
//                   over the 32.
//
// Every gather is a warp shuffle, which moves bits and does not compute, so
// the owner applies exactly what the lane formed. The owner then walks the
// four chunks in ascending k and adds each chunk's term onto `acc` with the
// float expression of the matching `mmqf_vec_dot_*`, written character for
// character; the build compiles both files with the same flags, so the
// compiler contracts them the same way. Nothing here reorders or fuses them
// by hand, and the three non-owner lanes keep no float state.
#pragma once

#include <cuda_fp16.h>

#include "../gemv/common.cuh"
#include "gemv1_formats.cuh"
#include "gemv1_kquant.cuh"
#include "gemv1_legacy.cuh"

// Activation record: 4 header words, then 32 quant words (128 k-values);
// `MMQF_Y_DS`, `MMQF_Y_QS` and `MMQF_Y_STRIDE` in `quant_mmq_mma.cu`. A
// step stages two records: 72 words, 18 windows. Header word `b` of a
// record carries chunk `b`'s activation scale as `half` bits in its low
// half and the exact int16 sum of the chunk's 32 activation quants in its
// high half.
#define GEMV1_Y_QS 4
#define GEMV1_Y_STRIDE 36
#define GEMV1_Y_WORDS (2 * GEMV1_Y_STRIDE)
#define GEMV1_Y_WINDOWS (GEMV1_Y_WORDS / 4)

// `dp4a` with `a` read as unsigned bytes and `b` as signed bytes.
static __device__ __forceinline__ int gemv1_dp4a_us(unsigned int a, int b, int c) {
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// `c + a.lo16 * b.byte0 + a.hi16 * b.byte1`, all signed.
static __device__ __forceinline__ int gemv1_dp2a_lo(int a, int b, int c) {
    int d;
    asm("dp2a.lo.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Copies `v` from the three other lanes of this feature into `out[1..3]`;
// `out[0]` is this lane's own. `base` is the feature's first lane. Every
// lane issues the shuffles, only the owner reads the results.
template <class T>
static __device__ __forceinline__ void gemv1_gather(T v, unsigned int base, T (&out)[4]) {
    out[0] = v;
#pragma unroll
    for (int jj = 1; jj < 4; ++jj) {
        out[jj] = __shfl_sync(0xFFFFFFFFu, v, base + jj);
    }
}

// The int32 dot of the chunk's 32 elements with its 32 activation bytes, in
// element order: word `i` of `w` against activation word `i`. The chunk's
// activation bytes are two 16-byte loads.
static __device__ __forceinline__ int gemv1_dot32(
    const int (&w)[8], const int* __restrict__ rec, unsigned int c
) {
    const int4* q4 = reinterpret_cast<const int4*>(rec + GEMV1_Y_QS + c * 8);
    const int4 q0 = q4[0];
    const int4 q1 = q4[1];
    int dot = 0;
    dot = dp4a(w[0], q0.x, dot);
    dot = dp4a(w[1], q0.y, dot);
    dot = dp4a(w[2], q0.z, dot);
    dot = dp4a(w[3], q0.w, dot);
    dot = dp4a(w[4], q1.x, dot);
    dot = dp4a(w[5], q1.y, dot);
    dot = dp4a(w[6], q1.z, dot);
    dot = dp4a(w[7], q1.w, dot);
    return dot;
}

// The same dot for a reader whose words are not in element order: word pair
// `p` dots with the activation word pair `FMT::act_pair(ln, p)`, and the
// words hold `value + 1` as unsigned bytes.
//
// `sum((code - 1) * a) = sum(code * a) - sum(a)`: the chunk's exact
// activation sum is the int16 high half of its header word, and `dp2a` with
// `(0, -1)` seeds the dot with its negation. Every term is an int, so the
// dot is the same integer as the signed-byte form's.
template <class FMT>
static __device__ __forceinline__ int gemv1_dot32_paired(
    const int (&w)[8], const int* __restrict__ rec, unsigned int c, const typename FMT::Lane& ln
) {
    int dot = gemv1_dp2a_lo(rec[c], 0x0000FF00, 0);
    const int2* q2 = reinterpret_cast<const int2*>(rec + GEMV1_Y_QS + c * 8);
    const int2 q0 = q2[FMT::act_pair(ln, 0)];
    const int2 q1 = q2[FMT::act_pair(ln, 1)];
    const int2 q2v = q2[FMT::act_pair(ln, 2)];
    const int2 q3 = q2[FMT::act_pair(ln, 3)];
    dot = gemv1_dp4a_us((unsigned int)w[0], q0.x, dot);
    dot = gemv1_dp4a_us((unsigned int)w[1], q0.y, dot);
    dot = gemv1_dp4a_us((unsigned int)w[2], q1.x, dot);
    dot = gemv1_dp4a_us((unsigned int)w[3], q1.y, dot);
    dot = gemv1_dp4a_us((unsigned int)w[4], q2v.x, dot);
    dot = gemv1_dp4a_us((unsigned int)w[5], q2v.y, dot);
    dot = gemv1_dp4a_us((unsigned int)w[6], q3.x, dot);
    dot = gemv1_dp4a_us((unsigned int)w[7], q3.y, dot);
    return dot;
}

// The 32-element-dot forms, `GEMV1_EPI_D` and `GEMV1_EPI_DM`: one int dot per
// chunk, then the owner's epilogue. Separate from the 16-element form so each
// scale interface gets its own instantiation.
template <class FMT>
static __device__ __forceinline__ void gemv1_dot_step(
    const int (&w)[8], const int* __restrict__ my_row, unsigned int o,
    const int* __restrict__ rec, const typename FMT::Lane& ln, unsigned int j, unsigned int j0,
    unsigned int c, unsigned int base, unsigned int b0, unsigned int kb1, float& acc
) {
    int dot;
    if constexpr (FMT::PAIRED_WORDS) {
        static_assert(FMT::CODES_PLUS_ONE, "Paired words are the code form.");
        dot = gemv1_dot32_paired<FMT>(w, rec, c, ln);
    } else {
        dot = gemv1_dot32(w, rec, c);
    }
    int dots[4];
    gemv1_gather(dot, base, dots);

    if constexpr (FMT::EPILOGUE == GEMV1_EPI_DM) {
        // The weight pair of this lane's own chunk; the owner gathers the
        // other three rather than decoding four packed scale fields itself.
        float scw;
        float minw;
        FMT::scales(my_row, o, j, ln, &scw, &minw);
        float scws[4];
        float minws[4];
        gemv1_gather(scw, base, scws);
        gemv1_gather(minw, base, minws);
        if (c == 0) {
            const int4 hdr = *reinterpret_cast<const int4*>(rec);
            const int hdrs[4] = {hdr.x, hdr.y, hdr.z, hdr.w};
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                if (b0 + j0 + (unsigned int)jj < kb1) {
                    const float da =
                        gemv1_half_bits_to_float((unsigned short)(hdrs[jj] & 0xFFFF));
                    // The int16 high half of the header word is the exact sum
                    // of the chunk's 32 activation quants, which is what the
                    // minimum term multiplies.
                    const float sum_i = (float)(hdrs[jj] >> 16);
                    const float dsum = da * sum_i;
                    // `mmqf_vec_dot_dm`'s two terms, character for character.
                    acc += scws[jj] * da * (float)dots[jj];
                    acc += minws[jj] * dsum;
                }
            }
        }
    } else if (c == 0) {
        const int4 hdr = *reinterpret_cast<const int4*>(rec);
        const int hdrs[4] = {hdr.x, hdr.y, hdr.z, hdr.w};
        // The four `dw` are one block's scale when a block spans 4 chunks.
        float dws[4];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            dws[jj] = (FMT::CHUNKS == 4 && jj > 0) ? dws[0] : FMT::scale(my_row, o, j0 + jj);
        }
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            if (b0 + j0 + (unsigned int)jj < kb1) {
                const float da = gemv1_half_bits_to_float((unsigned short)(hdrs[jj] & 0xFFFF));
                // `mmqf_vec_dot_d`'s term, character for character.
                acc += (float)dots[jj] * da * dws[jj];
            }
        }
    }
}

// The 16-element-dot forms, `GEMV1_EPI_SC16` and `GEMV1_EPI_SC16_MIN`: two int
// dots per chunk, each with its own weight scale, then the owner's epilogue.
// `MIN` adds the rank-1 minimum term of each 16-element half.
template <class FMT>
static __device__ __forceinline__ void gemv1_sc16_step(
    const int (&w)[8], const int* __restrict__ my_row, unsigned int o,
    const int* __restrict__ rec, const typename FMT::Lane& ln, unsigned int j, unsigned int j0,
    unsigned int c, unsigned int base, unsigned int b0, unsigned int kb1, float& acc
) {
    constexpr bool MIN = FMT::EPILOGUE == GEMV1_EPI_SC16_MIN;

    // The scale changes every 16 elements, so the chunk is two dots:
    // words 0..3 take scale `2j`, words 4..7 take `2j + 1`.
    const int4* q4 = reinterpret_cast<const int4*>(rec + GEMV1_Y_QS + c * 8);
    const int4 q0 = q4[0];
    const int4 q1 = q4[1];
    int d_lo = 0;
    d_lo = dp4a(w[0], q0.x, d_lo);
    d_lo = dp4a(w[1], q0.y, d_lo);
    d_lo = dp4a(w[2], q0.z, d_lo);
    d_lo = dp4a(w[3], q0.w, d_lo);
    int d_hi = 0;
    d_hi = dp4a(w[4], q1.x, d_hi);
    d_hi = dp4a(w[5], q1.y, d_hi);
    d_hi = dp4a(w[6], q1.z, d_hi);
    d_hi = dp4a(w[7], q1.w, d_hi);

    int los[4];
    int his[4];
    float scls[4];
    float schs[4];
    gemv1_gather(d_lo, base, los);
    gemv1_gather(d_hi, base, his);
    float mnls[4];
    float mnhs[4];
    int sas[4];
    if constexpr (MIN) {
        float sc_lo;
        float mn_lo;
        float sc_hi;
        float mn_hi;
        FMT::scales(my_row, o, j, ln, &sc_lo, &mn_lo, &sc_hi, &mn_hi);
        // The int sum of the chunk's FIRST 16 activation quants, which is the
        // integer `mmqf_stage_y_sums` stores for the tensor-core path. Both
        // are exact int reductions over the same four words, so they are the
        // same integer whatever order the adds take.
        int s_a = 0;
        s_a = dp4a(0x01010101, q0.x, s_a);
        s_a = dp4a(0x01010101, q0.y, s_a);
        s_a = dp4a(0x01010101, q0.z, s_a);
        s_a = dp4a(0x01010101, q0.w, s_a);
        gemv1_gather(sc_lo, base, scls);
        gemv1_gather(sc_hi, base, schs);
        gemv1_gather(mn_lo, base, mnls);
        gemv1_gather(mn_hi, base, mnhs);
        gemv1_gather(s_a, base, sas);
    } else {
        float sc_lo;
        float sc_hi;
        FMT::scales(my_row, o, j, ln, &sc_lo, &sc_hi);
        gemv1_gather(sc_lo, base, scls);
        gemv1_gather(sc_hi, base, schs);
    }

    if (c != 0) {
        return;
    }
    // The four `da` of the half: the low halves of the record's four header
    // words, one 16-byte load. They are read before the range guard; a chunk
    // past the range end reads a stale word its guard then drops.
    const int4 hdr = *reinterpret_cast<const int4*>(rec);
    const int hdrs[4] = {hdr.x, hdr.y, hdr.z, hdr.w};
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
        if (b0 + j0 + (unsigned int)jj < kb1) {
            const float da = gemv1_half_bits_to_float((unsigned short)(hdrs[jj] & 0xFFFF));
            // `mmqf_vec_dot_sc16`'s terms, character for character.
            acc += (scls[jj] * (float)los[jj] + schs[jj] * (float)his[jj]) * da;
            if constexpr (MIN) {
                // The header's high half is the exact int16 sum of the
                // chunk's 32 activation quants, so `s32 - s_a` is the sum of
                // the last 16 exactly — an integer identity.
                const int s32 = hdrs[jj] >> 16;
                const float dsum_lo = da * (float)sas[jj];
                const float dsum_hi = da * (float)(s32 - sas[jj]);
                acc += mnls[jj] * dsum_lo + mnhs[jj] * dsum_hi;
            }
        }
    }
}

// One half step: chunks `j0 .. j0 + 3` of the staged 256-k group, this lane
// holding chunk `j0 + c`. `rec` is the half's activation record, `my_row`
// the lane's staged weight row and `o` the span start's offset inside its
// first window. `b0 + j0 + jj` is chunk `jj`'s index in the K walk, which
// the range end `kb1` cuts. Only the owner lane (`c == 0`) touches `acc`.
template <class FMT>
static __device__ __forceinline__ void gemv1_half_step(
    const int* __restrict__ my_row, unsigned int o, const int* __restrict__ rec,
    const typename FMT::Lane& ln, unsigned int j0, unsigned int c, unsigned int base,
    unsigned int b0, unsigned int kb1, float& acc
) {
    const unsigned int j = j0 + c;
    int w[8];
    FMT::read(my_row, o, j, ln, w);

    if constexpr (FMT::EPILOGUE == GEMV1_EPI_SC16 ||
                  FMT::EPILOGUE == GEMV1_EPI_SC16_MIN) {
        gemv1_sc16_step<FMT>(w, my_row, o, rec, ln, j, j0, c, base, b0, kb1, acc);
    } else {
        gemv1_dot_step<FMT>(w, my_row, o, rec, ln, j, j0, c, base, b0, kb1, acc);
    }
}
