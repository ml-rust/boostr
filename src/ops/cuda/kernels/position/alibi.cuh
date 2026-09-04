#pragma once

// Compute head-specific ALiBi slope
// Formula: m_h = 2^(-8h/H) where h = head index, H = total heads
//
// Example (H=8):
//   Head 0: m = 2^0 = 1.0
//   Head 1: m = 2^(-1) = 0.5
//   Head 2: m = 2^(-2) = 0.25
//   ...
//   Head 7: m = 2^(-7) = 0.0078125

__device__ __forceinline__ float get_alibi_slope(int head_idx, int num_heads) {
    return powf(2.0f, -8.0f * head_idx / (float)num_heads);
}
