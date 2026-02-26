// Fused INT4 triple-GEMM QKV projection
// Fused INT4 triple-GEMM QKV: (input@Wq, input@Wk, input@Wv)
// All weights in AWQ INT4 format
// Uses blockIdx.z to select Q/K/V projection

#include <stdint.h>

__constant__ int QKV_AWQ_SHIFTS[8] = {0, 16, 4, 20, 8, 24, 12, 28};

__device__ __forceinline__ int unpack_int4_awq(uint32_t packed, int idx) {
    return (packed >> QKV_AWQ_SHIFTS[idx]) & 0xF;
}

// Grid: (ceil(max(Nq,Nkv)/16), ceil(M/16), 3), Block: (16, 16)
// blockIdx.z: 0=Q, 1=K, 2=V
extern "C" __global__ void fused_int4_qkv_f32(
    const float* __restrict__ input,         // [M, K]
    const uint32_t* __restrict__ qweight_q,  // [K, Nq/8]
    const float* __restrict__ scales_q,      // [num_groups, Nq]
    const float* __restrict__ zeros_q,       // [num_groups, Nq]
    const uint32_t* __restrict__ qweight_k,  // [K, Nkv/8]
    const float* __restrict__ scales_k,      // [num_groups, Nkv]
    const float* __restrict__ zeros_k,       // [num_groups, Nkv]
    const uint32_t* __restrict__ qweight_v,  // [K, Nkv/8]
    const float* __restrict__ scales_v,      // [num_groups, Nkv]
    const float* __restrict__ zeros_v,       // [num_groups, Nkv]
    float* __restrict__ out_q,               // [M, Nq]
    float* __restrict__ out_k,               // [M, Nkv]
    float* __restrict__ out_v,               // [M, Nkv]
    uint32_t M, uint32_t K, uint32_t Nq, uint32_t Nkv, uint32_t group_size
) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t proj = blockIdx.z; // 0=Q, 1=K, 2=V

    // Select the right weight/output based on projection
    const uint32_t* qw;
    const float* sc;
    const float* zr;
    float* out;
    uint32_t N_proj;

    if (proj == 0) {
        qw = qweight_q; sc = scales_q; zr = zeros_q; out = out_q; N_proj = Nq;
    } else if (proj == 1) {
        qw = qweight_k; sc = scales_k; zr = zeros_k; out = out_k; N_proj = Nkv;
    } else {
        qw = qweight_v; sc = scales_v; zr = zeros_v; out = out_v; N_proj = Nkv;
    }

    if (row >= M || col >= N_proj) return;

    uint32_t n_packed = N_proj / 8;
    uint32_t pack_col = col / 8;
    uint32_t sub = col % 8;

    float acc = 0.0f;
    for (uint32_t ki = 0; ki < K; ki++) {
        float a = input[row * K + ki];
        uint32_t packed = qw[ki * n_packed + pack_col];
        int q = unpack_int4_awq(packed, sub);

        uint32_t group = ki / group_size;
        float scale = sc[group * N_proj + col];
        float zero = zr[group * N_proj + col];
        float w = ((float)q - zero) * scale;
        acc += a * w;
    }
    out[row * N_proj + col] = acc;
}
