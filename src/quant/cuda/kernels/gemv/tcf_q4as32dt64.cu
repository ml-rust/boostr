// TCF `Q4AS32DT64` GEMV — token-batched dp4a (TCF weight x Q8_1 activation)
//
// One block covers NTOK consecutive token columns and decodes each 32-element
// weight group once for all of them, instead of re-reading and re-decoding the
// whole weight matrix per token as `tcf_gemv_f32` in `kernels/tcf.cu` does.
// Body, lane map, nibble map and the minimum-term sign live in
// `tcf_ntok.cuh`.
//
// Three tile widths, 1 / 2 / 4 token columns. `launch_gemv_dp4a` picks the
// narrowest one that covers M in a single block: a wider tile would idle its
// spare columns, a narrower one would need a second pass over the weights.
// Unlike the legacy GGUF 32-element formats, this format HAS a single-token
// tile — its f32 sibling reconstructs a whole f32 weight per element, so even
// at m = 1 the int8 path is the cheaper decode rather than pure tile overhead.
//
// The eleven trailing arguments are the plane layout, in the order
// `push_layout!` in `quant/cuda/tcf/launch.rs` pushes them and every other TCF
// kernel declares them. They are assembled into a `TcfLayout` here rather than
// passed as a struct, so no host/device struct layout has to agree.

#include "tcf_ntok.cuh"

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(1) * WARP_SIZE, 1) void quant_gemv_tcf_q4as32dt64_q8_1_mwr(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N,
    unsigned long long code_high_off, unsigned long long scale_off,
    unsigned long long min_off, unsigned long long super_off,
    unsigned long long super_min_off, unsigned int bits, unsigned int group,
    unsigned int groups_per_tile, unsigned int symmetric, unsigned int scale_form,
    unsigned int sub_block_bytes
) {
    quant_gemv_tcf_q4as32dt64_q8_1_mwr_ntok<1>(
        q8_act, weight, output, M, K, N,
        tcf_layout(code_high_off, scale_off, min_off, super_off, super_min_off, bits,
                   group, groups_per_tile, symmetric, scale_form, sub_block_bytes));
}

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(2) * WARP_SIZE, 1) void quant_gemv_tcf_q4as32dt64_q8_1_mwr_n2(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N,
    unsigned long long code_high_off, unsigned long long scale_off,
    unsigned long long min_off, unsigned long long super_off,
    unsigned long long super_min_off, unsigned int bits, unsigned int group,
    unsigned int groups_per_tile, unsigned int symmetric, unsigned int scale_form,
    unsigned int sub_block_bytes
) {
    quant_gemv_tcf_q4as32dt64_q8_1_mwr_ntok<2>(
        q8_act, weight, output, M, K, N,
        tcf_layout(code_high_off, scale_off, min_off, super_off, super_min_off, bits,
                   group, groups_per_tile, symmetric, scale_form, sub_block_bytes));
}

extern "C" __global__ __launch_bounds__(mwr_nwarps_ntok(4) * WARP_SIZE, 1) void quant_gemv_tcf_q4as32dt64_q8_1_mwr_n4(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N,
    unsigned long long code_high_off, unsigned long long scale_off,
    unsigned long long min_off, unsigned long long super_off,
    unsigned long long super_min_off, unsigned int bits, unsigned int group,
    unsigned int groups_per_tile, unsigned int symmetric, unsigned int scale_form,
    unsigned int sub_block_bytes
) {
    quant_gemv_tcf_q4as32dt64_q8_1_mwr_ntok<4>(
        q8_act, weight, output, M, K, N,
        tcf_layout(code_high_off, scale_off, min_off, super_off, super_min_off, bits,
                   group, groups_per_tile, symmetric, scale_form, sub_block_bytes));
}
