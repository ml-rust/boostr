// KV-cache quant BF16 kernels, split out of kv_cache_quant.cu.
// `__nv_bfloat16` conversion needs sm_80. Splitting keeps the F32/F16/FP8
// kernels in kv_cache_quant.cu on sm_75 so they still load on Turing.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "kv_cache_quant.cuh"

// BF16 quantization
extern "C" __global__ void quantize_kv_fp8_per_token_bf16(
    const __nv_bfloat16* input, boostr_fp8_e4m3* output, float* scales,
    int num_tokens, int head_dim
) {
    quantize_kv_fp8_per_token_impl<__nv_bfloat16>(input, output, scales, num_tokens, head_dim);
}

extern "C" __global__ void quantize_kv_fp8_per_head_bf16(
    const __nv_bfloat16* input, boostr_fp8_e4m3* output, float* scales,
    int num_heads, int seq_len, int head_dim
) {
    quantize_kv_fp8_per_head_impl<__nv_bfloat16>(input, output, scales, num_heads, seq_len, head_dim);
}

// BF16 INT8 quantization
extern "C" __global__ void quantize_kv_int8_per_token_bf16(
    const __nv_bfloat16* input, int8_t* output, float* scales,
    int num_tokens, int head_dim
) {
    quantize_kv_int8_per_token_impl<__nv_bfloat16>(input, output, scales, num_tokens, head_dim);
}
