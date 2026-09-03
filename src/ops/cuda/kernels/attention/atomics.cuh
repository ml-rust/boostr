#pragma once
// FP16 atomic add helper taking a __half addend
//
// Included by kernels that scatter gradients into half-precision buffers where
// several producers hit the same element (e.g. GQA backward, where multiple Q
// heads write to one KV head's dK/dV).
//
// This is NOT a duplicate of `atomic_add_dtype(__half*, float)` in
// dtype_traits.cuh: that one takes a float addend and rounds once, after adding
// in float. This one takes a __half addend and adds with `__hadd`, so the
// caller's value is rounded to half before the accumulate. The two round
// differently, so they are not interchangeable without changing kernel
// numerics. `varlen_attention_bwd_fp16.cu` is the only includer.
//
// The BF16 counterpart was removed: nothing called it, and 16-bit BF16 adds go
// through `atomic_add_dtype` in dtype_traits.cuh.

#include <cuda_fp16.h>

// ============================================================================
// atomicAddHalf — FP16 atomic add (sm_70+)
// ============================================================================

#if __CUDA_ARCH__ >= 700
__device__ __forceinline__ void atomicAddHalf(__half* address, __half val) {
    // CAS-based implementation.  We operate on the 32-bit aligned word that
    // contains the target __half element.
    //
    // Two adjacent __half values share one 32-bit word.  We determine which
    // of the two halves our address occupies via the low bit of the byte offset,
    // update only that half, then CAS the whole 32-bit word.
    unsigned int* address_as_uint = (unsigned int*)((size_t)address & ~(size_t)2);
    unsigned int old = *address_as_uint;
    unsigned int assumed;

    do {
        assumed = old;
        // Reconstruct the two __half values from the packed word.
        __half lo = *(__half*)&assumed;
        __half hi = *((__half*)&assumed + 1);

        unsigned int updated;
        if ((size_t)address & 2) {
            // address points to the high half
            hi = __hadd(hi, val);
        } else {
            // address points to the low half
            lo = __hadd(lo, val);
        }
        // Pack back into a 32-bit word.
        updated = (unsigned int)(*(unsigned short*)&lo) |
                  ((unsigned int)(*(unsigned short*)&hi) << 16);

        old = atomicCAS(address_as_uint, assumed, updated);
    } while (assumed != old);
}
#endif  // __CUDA_ARCH__ >= 700
