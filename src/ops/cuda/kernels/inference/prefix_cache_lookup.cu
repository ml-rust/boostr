// GPU prefix-cache lookup kernel.
//
// Uses open addressing with linear probing to match the CPU-side
// GpuRadixTree layout so that the same flat arrays can be uploaded
// once and queried many times per batch.
//
// Each thread handles exactly one hash probe.

#include <cstdint>

extern "C" __global__ void prefix_cache_lookup(
    const uint64_t* __restrict__ query_hashes,  // [num_queries]
    const uint64_t* __restrict__ table_keys,    // [capacity]
    const int32_t*  __restrict__ table_values,  // [capacity] (-1 = empty slot)
    int32_t*        __restrict__ out_block_ids, // [num_queries] (-1 = miss)
    const int32_t capacity,
    const int32_t num_queries
) {
    int32_t qidx = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x)
                   + static_cast<int32_t>(threadIdx.x);
    if (qidx >= num_queries) return;

    uint64_t key  = query_hashes[qidx];
    int32_t  mask = capacity - 1; // capacity is always a power of two

    // Linear probing: probe at most `capacity` slots.
    int32_t slot = static_cast<int32_t>(key) & mask;
    int32_t result = -1;

    for (int32_t probe = 0; probe < capacity; ++probe) {
        int32_t v = table_values[slot];
        if (v == -1) {
            // Empty slot: definite miss.
            break;
        }
        if (table_keys[slot] == key) {
            result = v;
            break;
        }
        slot = (slot + 1) & mask;
    }

    out_block_ids[qidx] = result;
}
