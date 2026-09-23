// KV cache insert kernel for CUDA graph decode loop.
//
// Replaces `slice_assign` (which uses a CPU-side offset frozen at graph capture
// time) with a kernel that reads the insert position from device memory.  This
// makes the write position a runtime value that can be updated between graph
// replays without re-capturing the graph.
//
// Layout:
//   K_new / V_new : [B, H_kv, 1, D]   — one token per batch, per head
//   K_cache / V_cache : [B, H_kv, capacity, D]  — full-capacity cache (stable address)
//   write_pos : device i32 [B] — row `b` inserts at `write_pos[b]`, so lanes at
//               different lengths share one launch
//
// Each thread writes one element (b, h, write_pos[b], d) for both K and V.
// Grid: (B * H_kv * D, 1, 1), Block: (1, 1, 1) would work but is wasteful.
// Better: Grid: (ceil(B * H_kv * D / BLOCK), 1, 1), Block: (BLOCK, 1, 1).

#define BLOCK 256

extern "C" __global__ void kv_insert_f32(
    const float* __restrict__ K_new,   // [B, H_kv, 1, D]
    const float* __restrict__ V_new,   // [B, H_kv, 1, D]
    float* __restrict__ K_cache,       // [B, H_kv, capacity, D]  (stable address)
    float* __restrict__ V_cache,
    int B, int H_kv, int D, int capacity,
    const int* write_pos               // device ptr: [B] sequence index to write
) {
    // Total elements to copy: B * H_kv * D (one element per thread)
    const int total = B * H_kv * D;
    const int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= total) return;

    // Decompose idx into (b, h, d)
    const int d = idx % D;
    const int bh = idx / D;  // b * H_kv + h
    const int h = bh % H_kv;
    const int b = bh / H_kv;

    // Row `b`'s own next slot, so each lane writes where its sequence ended.
    const int wpos = write_pos[b];

    // Source: K_new[b, h, 0, d] = K_new + (b * H_kv + h) * D + d
    const size_t src_off = (size_t)(b * H_kv + h) * D + d;

    // Destination: K_cache[b, h, wpos, d] = K_cache + ((b * H_kv + h) * capacity + wpos) * D + d
    const size_t dst_off = ((size_t)(b * H_kv + h) * capacity + wpos) * D + d;

    K_cache[dst_off] = K_new[src_off];
    V_cache[dst_off] = V_new[src_off];
}

// F16 variant — identical logic, different dtype
#include <cuda_fp16.h>

extern "C" __global__ void kv_insert_f16(
    const __half* __restrict__ K_new,
    const __half* __restrict__ V_new,
    __half* __restrict__ K_cache,
    __half* __restrict__ V_cache,
    int B, int H_kv, int D, int capacity,
    const int* write_pos
) {
    const int total = B * H_kv * D;
    const int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= total) return;

    const int d = idx % D;
    const int bh = idx / D;
    const int h = bh % H_kv;
    const int b = bh / H_kv;

    const int wpos = write_pos[b];

    const size_t src_off = (size_t)(b * H_kv + h) * D + d;
    const size_t dst_off = ((size_t)(b * H_kv + h) * capacity + wpos) * D + d;

    K_cache[dst_off] = K_new[src_off];
    V_cache[dst_off] = V_new[src_off];
}
