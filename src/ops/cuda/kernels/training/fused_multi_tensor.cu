// Fused multi-tensor AdamW kernel
//
// Processes multiple parameter groups in a single kernel launch.
// Metadata (pointers + sizes) is passed via global memory buffers:
//   - ptrs: [param0, grad0, m0, v0, param1, grad1, m1, v1, ...] (4 pointers per group)
//   - sizes: [size0, size1, ...] cumulative prefix sums for group boundaries
//   - cum_sizes: [0, size0, size0+size1, ...] (num_groups+1 entries)
//
// Each thread maps to a global element index and binary-searches cum_sizes to
// find which param group it belongs to.

extern "C" __global__ void fused_multi_tensor_adamw_f32(
    const unsigned long long* __restrict__ ptrs,  // 4*num_groups device pointers
    const int* __restrict__ cum_sizes,             // num_groups+1 cumulative sizes
    int num_groups,
    float lr, float beta1, float beta2, float eps, float wd, float step_size,
    int total_n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_n) return;

    // Binary search for group index in cum_sizes
    int lo = 0, hi = num_groups - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cum_sizes[mid + 1] <= idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    int g = lo;
    int local_idx = idx - cum_sizes[g];

    // Each group has 4 pointers: param, grad, m, v
    float* param = (float*)ptrs[g * 4 + 0];
    const float* grad = (const float*)ptrs[g * 4 + 1];
    float* m = (float*)ptrs[g * 4 + 2];
    float* v = (float*)ptrs[g * 4 + 3];

    float gi = grad[local_idx];
    float mi = beta1 * m[local_idx] + (1.0f - beta1) * gi;
    float vi = beta2 * v[local_idx] + (1.0f - beta2) * gi * gi;

    float update = step_size * mi / (sqrtf(vi) + eps);
    float decayed = param[local_idx] * (1.0f - lr * wd);

    param[local_idx] = decayed - update;
    m[local_idx] = mi;
    v[local_idx] = vi;
}
