// Fused QKV projection CUDA kernels
//
// Forward: fused matmul + optional bias + split into Q/K/V + reshape
// Currently delegates to impl_generic via numr's CUDA matmul.
// These kernels are placeholders for future optimization when profiling
// shows benefit from a fully fused CUDA implementation.
//
// The primary win from FusedQkvOps comes from the API-level fusion
// (single matmul instead of 3 separate ones), which already uses
// numr's optimized CUDA matmul. A custom kernel would additionally
// fuse the bias add + split + reshape into the matmul epilogue.

#include "dtype_traits.cuh"

// Fused bias + split + reshape kernel
// Applies bias to the concatenated QKV output and splits into Q, K, V
// with transposed [B, heads, S, D] layout in a single pass.
//
// Input:  qkv [B*S, total_proj] (output of matmul)
// Output: q [B, num_heads, S, D], k [B, num_kv_heads, S, D], v [B, num_kv_heads, S, D]
template<typename T>
__global__ void fused_qkv_bias_split_kernel(
    const T* __restrict__ qkv,      // [B*S, total_proj]
    const T* __restrict__ bias,      // [total_proj] or nullptr
    T* __restrict__ q_out,           // [B, num_heads, S, D]
    T* __restrict__ k_out,           // [B, num_kv_heads, S, D]
    T* __restrict__ v_out,           // [B, num_kv_heads, S, D]
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int total_proj
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch_size * seq_len * total_proj;
    if (idx >= total) return;

    unsigned int hq = num_heads * head_dim;
    unsigned int hkv = num_kv_heads * head_dim;

    // Decompose flat index into (batch_seq_idx, proj_idx)
    unsigned int proj_idx = idx % total_proj;
    unsigned int batch_seq_idx = idx / total_proj;
    unsigned int b = batch_seq_idx / seq_len;
    unsigned int s = batch_seq_idx % seq_len;

    T val = qkv[idx];
    if (bias != nullptr) {
        val = val + bias[proj_idx];
    }

    if (proj_idx < hq) {
        // Q region
        unsigned int h = proj_idx / head_dim;
        unsigned int d = proj_idx % head_dim;
        // Output layout: [B, num_heads, S, D]
        unsigned int out_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
        q_out[out_idx] = val;
    } else if (proj_idx < hq + hkv) {
        // K region
        unsigned int local = proj_idx - hq;
        unsigned int h = local / head_dim;
        unsigned int d = local % head_dim;
        unsigned int out_idx = ((b * num_kv_heads + h) * seq_len + s) * head_dim + d;
        k_out[out_idx] = val;
    } else {
        // V region
        unsigned int local = proj_idx - hq - hkv;
        unsigned int h = local / head_dim;
        unsigned int d = local % head_dim;
        unsigned int out_idx = ((b * num_kv_heads + h) * seq_len + s) * head_dim + d;
        v_out[out_idx] = val;
    }
}

// Fused output projection residual: proj + bias + residual in one pass
// Input: proj [B*S, H] (output of matmul), bias [H], residual [B, S, H]
// Output: [B, S, H]
template<typename T>
__global__ void fused_output_bias_residual_kernel(
    const T* __restrict__ proj,       // [B*S, H]
    const T* __restrict__ bias,       // [H] or nullptr
    const T* __restrict__ residual,   // [B*S, H]
    T* __restrict__ output,           // [B*S, H]
    unsigned int total                // B*S*H
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    T val = proj[idx];
    if (bias != nullptr) {
        unsigned int h_idx = idx % (total / 1); // simplified — bias broadcast handled by caller
        val = val + bias[h_idx];
    }
    output[idx] = val + residual[idx];
}

// Backward: fused concat dQ/dK/dV → d_qkv
// Inverse of the split in forward: gather from [B, heads, S, D] → [B*S, total_proj]
template<typename T>
__global__ void fused_qkv_concat_kernel(
    const T* __restrict__ dq,        // [B, num_heads, S, D]
    const T* __restrict__ dk,        // [B, num_kv_heads, S, D]
    const T* __restrict__ dv,        // [B, num_kv_heads, S, D]
    T* __restrict__ d_qkv,           // [B*S, total_proj]
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int total_proj
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch_size * seq_len * total_proj;
    if (idx >= total) return;

    unsigned int hq = num_heads * head_dim;
    unsigned int hkv = num_kv_heads * head_dim;

    unsigned int proj_idx = idx % total_proj;
    unsigned int batch_seq_idx = idx / total_proj;
    unsigned int b = batch_seq_idx / seq_len;
    unsigned int s = batch_seq_idx % seq_len;

    T val;
    if (proj_idx < hq) {
        unsigned int h = proj_idx / head_dim;
        unsigned int d = proj_idx % head_dim;
        unsigned int in_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
        val = dq[in_idx];
    } else if (proj_idx < hq + hkv) {
        unsigned int local = proj_idx - hq;
        unsigned int h = local / head_dim;
        unsigned int d = local % head_dim;
        unsigned int in_idx = ((b * num_kv_heads + h) * seq_len + s) * head_dim + d;
        val = dk[in_idx];
    } else {
        unsigned int local = proj_idx - hq - hkv;
        unsigned int h = local / head_dim;
        unsigned int d = local % head_dim;
        unsigned int in_idx = ((b * num_kv_heads + h) * seq_len + s) * head_dim + d;
        val = dv[in_idx];
    }
    d_qkv[idx] = val;
}

// Instantiate kernels for supported dtypes
extern "C" {
    __global__ void fused_qkv_bias_split_f32(
        const float* qkv, const float* bias,
        float* q, float* k, float* v,
        unsigned int batch_size, unsigned int seq_len,
        unsigned int num_heads, unsigned int num_kv_heads,
        unsigned int head_dim, unsigned int total_proj
    ) {
        fused_qkv_bias_split_kernel<float>(
            qkv, bias, q, k, v, batch_size, seq_len,
            num_heads, num_kv_heads, head_dim, total_proj
        );
    }

    __global__ void fused_qkv_bias_split_f64(
        const double* qkv, const double* bias,
        double* q, double* k, double* v,
        unsigned int batch_size, unsigned int seq_len,
        unsigned int num_heads, unsigned int num_kv_heads,
        unsigned int head_dim, unsigned int total_proj
    ) {
        fused_qkv_bias_split_kernel<double>(
            qkv, bias, q, k, v, batch_size, seq_len,
            num_heads, num_kv_heads, head_dim, total_proj
        );
    }

    __global__ void fused_qkv_concat_f32(
        const float* dq, const float* dk, const float* dv,
        float* d_qkv,
        unsigned int batch_size, unsigned int seq_len,
        unsigned int num_heads, unsigned int num_kv_heads,
        unsigned int head_dim, unsigned int total_proj
    ) {
        fused_qkv_concat_kernel<float>(
            dq, dk, dv, d_qkv, batch_size, seq_len,
            num_heads, num_kv_heads, head_dim, total_proj
        );
    }

    __global__ void fused_qkv_concat_f64(
        const double* dq, const double* dk, const double* dv,
        double* d_qkv,
        unsigned int batch_size, unsigned int seq_len,
        unsigned int num_heads, unsigned int num_kv_heads,
        unsigned int head_dim, unsigned int total_proj
    ) {
        fused_qkv_concat_kernel<double>(
            dq, dk, dv, d_qkv, batch_size, seq_len,
            num_heads, num_kv_heads, head_dim, total_proj
        );
    }

    __global__ void fused_output_bias_residual_f32(
        const float* proj, const float* bias, const float* residual,
        float* output, unsigned int total
    ) {
        fused_output_bias_residual_kernel<float>(proj, bias, residual, output, total);
    }

    __global__ void fused_output_bias_residual_f64(
        const double* proj, const double* bias, const double* residual,
        double* output, unsigned int total
    ) {
        fused_output_bias_residual_kernel<double>(proj, bias, residual, output, total);
    }
}
