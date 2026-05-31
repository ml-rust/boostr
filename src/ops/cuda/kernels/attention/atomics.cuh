#pragma once
// Atomic add helpers for FP16 and BF16
//
// Included by kernels that need to scatter gradients into half-precision
// buffers using atomicAdd (e.g. GQA backward where multiple Q heads write
// to the same KV head's dK/dV).
//
// FP16 uses a CAS-based implementation on sm_70+ (Volta and later).
// BF16 uses the native atomicAdd available on sm_80+ (Ampere and later).

#include <cuda_fp16.h>
#include <cuda_bf16.h>

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

// ============================================================================
// atomicAddBF16 — BF16 atomic add (sm_80+)
// ============================================================================

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void atomicAddBF16(__nv_bfloat16* address, __nv_bfloat16 val) {
    // Ampere and later support native BF16 atomicAdd.
    atomicAdd(address, val);
}
#endif  // __CUDA_ARCH__ >= 800
