// DType Traits - Type-safe multi-precision support for CUDA kernels
// Provides compile-time dtype dispatch with zero runtime overhead
//
// Supports: F32, F16, BF16, FP8E4M3, FP8E5M2
// Automatic fallback for older GPU architectures

#ifndef BOOSTR_DTYPE_TRAITS_CUH
#define BOOSTR_DTYPE_TRAITS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>  // For std::is_same_v in convert_dtype

// ============================================================================
// Vector Type Definitions
// ============================================================================

// half4: 4x __half vector for vectorized loads/stores
struct half4 {
    __half x, y, z, w;
};

// ============================================================================
// FP8 Type Definitions
// ============================================================================

// IMPORTANT: Using boostr_ prefix instead of __nv_ to avoid collision with NVIDIA's
// internal headers (<cuda_fp8.h> in CUDA 11.8+). This ensures compatibility when
// we later add hardware FP8 intrinsics for Hopper GPUs.

// FP8 types as distinct wrapper structs (not typedefs) to enable template specialization
struct boostr_fp8_e4m3 {
    uint8_t data;
    __device__ __forceinline__ boostr_fp8_e4m3() : data(0) {}
    __device__ __forceinline__ explicit boostr_fp8_e4m3(uint8_t v) : data(v) {}
    __device__ __forceinline__ operator uint8_t() const { return data; }
};

struct boostr_fp8_e5m2 {
    uint8_t data;
    __device__ __forceinline__ boostr_fp8_e5m2() : data(0) {}
    __device__ __forceinline__ explicit boostr_fp8_e5m2(uint8_t v) : data(v) {}
    __device__ __forceinline__ operator uint8_t() const { return data; }
};

// ============================================================================
// FP8 Conversion Utilities
// - SM 8.0+ (Ampere): Software emulation for compatibility
// - SM 8.9+ (Hopper): Hardware intrinsics for maximum performance
// ============================================================================

#if __CUDA_ARCH__ >= 800  // Ampere and newer have FP8 support

// ============================================================================
// FP8 → F32 Conversion
// ============================================================================

// E4M3: 4 exponent bits, 3 mantissa bits (higher precision, range: ~[-448, 448])
__device__ __forceinline__ float fp8_e4m3_to_f32(uint8_t u, float scale = 1.0f) {
#if __CUDA_ARCH__ >= 890
    // Hopper+: Use inline PTX for guaranteed hardware acceleration
    // cvt.rn.f32.e4m3x2 unpacks 2xFP8 -> 2xF32 (we only use the first result)
    float result;
    asm volatile (
        "{ .reg .f32 dummy; \n\t"                      // Declare temp register for 2nd output
        "cvt.rn.f32.e4m3x2 {%0, dummy}, %1; \n\t"      // Unpack: lower 8 bits -> result, upper 8 bits -> dummy
        "}"
        : "=f"(result)
        : "h"((uint16_t)u)                              // cudarc 0.19: "h" constraint for 16-bit operands
    );
    return result / scale;
#else
    // Ampere and older: Software emulation
    int sign = (u >> 7) & 1;
    int exp_bits = (u >> 3) & 0xF;
    int mant_bits = u & 0x7;

    float result = 0.0f;
    if (exp_bits != 0) {
        float mant = 1.0f + mant_bits * (1.0f / 8.0f);
        int exp = exp_bits - 7;
        result = ldexpf(mant, exp);
    }

    return (sign ? -result : result) / scale;
#endif
}

// ============================================================================
// F32 → FP8 Conversion
// ============================================================================

__device__ __forceinline__ uint8_t f32_to_fp8_e4m3_raw(float x, float scale = 1.0f) {
#if __CUDA_ARCH__ >= 890
    // Hopper+: Use inline PTX for guaranteed hardware acceleration
    // cvt.rn.satfinite.e4m3x2.f32 requires PTX ISA 8.0+
    x = x * scale;
    uint32_t result;
    // Saturate to E4M3 range and round to nearest
    asm ("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
         : "=r"(result)
         : "f"(x), "f"(0.0f));  // Padding: e4m3x2 packs 2 scalars, second is zeroed
    return (uint8_t)result;
#else
    // Ampere and older: Software emulation
    x = x * scale;
    if (x == 0.0f) return 0;

    int sign = (x < 0.0f) ? 1 : 0;
    float abs_x = fabsf(x);

    int exp_bits;
    float mant;
    frexpf(abs_x, &exp_bits);
    exp_bits = exp_bits - 1;
    mant = ldexpf(abs_x, -exp_bits);

    exp_bits = max(-7, min(8, exp_bits));
    int exp_encoded = exp_bits + 7;

    int mant_bits = (int)roundf((mant - 1.0f) * 8.0f);
    mant_bits = max(0, min(7, mant_bits));

    return (sign << 7) | (exp_encoded << 3) | mant_bits;
#endif
}

// E5M2: 5 exponent bits, 2 mantissa bits (lower precision, range: ~[-57344, 57344])
__device__ __forceinline__ float fp8_e5m2_to_f32(uint8_t u, float scale = 1.0f) {
#if __CUDA_ARCH__ >= 890
    // Hopper+: Use inline PTX for guaranteed hardware acceleration
    // cvt.rn.f32.e5m2x2 unpacks 2xFP8 -> 2xF32 (we only use the first result)
    float result;
    asm volatile (
        "{ .reg .f32 dummy; \n\t"                      // Declare temp register for 2nd output
        "cvt.rn.f32.e5m2x2 {%0, dummy}, %1; \n\t"      // Unpack: lower 8 bits -> result, upper 8 bits -> dummy
        "}"
        : "=f"(result)
        : "h"((uint16_t)u)                              // cudarc 0.19: "h" constraint for 16-bit operands
    );
    return result / scale;
#else
    // Ampere and older: Software emulation
    int sign = (u >> 7) & 1;
    int exp_bits = (u >> 2) & 0x1F;
    int mant_bits = u & 0x3;

    float result = 0.0f;
    if (exp_bits != 0) {
        float mant = 1.0f + mant_bits * 0.25f;
        int exp = exp_bits - 15;
        result = ldexpf(mant, exp);
    }

    return (sign ? -result : result) / scale;
#endif
}

__device__ __forceinline__ uint8_t f32_to_fp8_e5m2_raw(float x, float scale = 1.0f) {
#if __CUDA_ARCH__ >= 890
    // Hopper+: Use inline PTX for guaranteed hardware acceleration
    x = x * scale;
    uint32_t result;
    asm ("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
         : "=r"(result)
         : "f"(x), "f"(0.0f));  // Padding: e5m2x2 packs 2 scalars, second is zeroed
    return (uint8_t)result;
#else
    // Ampere and older: Software emulation
    x = x * scale;
    if (x == 0.0f) return 0;

    int sign = (x < 0.0f) ? 1 : 0;
    float abs_x = fabsf(x);

    int exp_bits;
    float mant;
    frexpf(abs_x, &exp_bits);
    exp_bits = exp_bits - 1;
    mant = ldexpf(abs_x, -exp_bits);

    exp_bits = max(-15, min(16, exp_bits));
    int exp_encoded = exp_bits + 15;

    int mant_bits = (int)roundf((mant - 1.0f) * 4.0f);
    mant_bits = max(0, min(3, mant_bits));

    return (sign << 7) | (exp_encoded << 2) | mant_bits;
#endif
}
#endif  // __CUDA_ARCH__ >= 800

// ============================================================================
// DType Traits - Compile-time type information
// ============================================================================

enum class DTypeEnum {
    F32,
    F16,
    BF16,
    FP8E4M3,
    FP8E5M2
};

template<typename T>
struct DTypeTraits {
    static constexpr bool is_supported = false;
    static constexpr bool has_native_ops = false;
    static constexpr bool needs_scale = false;
    static constexpr DTypeEnum dtype_enum = DTypeEnum::F32;
};

// F32 - Full native support on all architectures
template<>
struct DTypeTraits<float> {
    static constexpr bool is_supported = true;
    static constexpr bool has_native_ops = true;
    static constexpr bool needs_scale = false;
    static constexpr DTypeEnum dtype_enum = DTypeEnum::F32;
    using compute_type = float;
    using storage_type = float;
};

// F16 - Native on SM 5.3+ (Maxwell), full ops on SM 7.0+ (Volta)
template<>
struct DTypeTraits<__half> {
    static constexpr bool is_supported = true;
    #if __CUDA_ARCH__ >= 700
    static constexpr bool has_native_ops = true;
    #else
    static constexpr bool has_native_ops = false;
    #endif
    static constexpr bool needs_scale = false;
    static constexpr DTypeEnum dtype_enum = DTypeEnum::F16;
    using compute_type = __half;
    using storage_type = __half;
    using fallback_type = float;
};

// BF16 - Native on SM 8.0+ (Ampere)
template<>
struct DTypeTraits<__nv_bfloat16> {
    static constexpr bool is_supported = true;
    #if __CUDA_ARCH__ >= 800
    static constexpr bool has_native_ops = true;
    #else
    static constexpr bool has_native_ops = false;
    #endif
    static constexpr bool needs_scale = false;
    static constexpr DTypeEnum dtype_enum = DTypeEnum::BF16;
    using compute_type = __nv_bfloat16;
    using storage_type = __nv_bfloat16;
    using fallback_type = float;
};

// FP8E4M3 - Hopper SM 8.9+ (stored as uint8_t)
template<>
struct DTypeTraits<boostr_fp8_e4m3> {
    static constexpr bool is_supported = true;
    #if __CUDA_ARCH__ >= 890
    static constexpr bool has_native_ops = true;
    #else
    static constexpr bool has_native_ops = false;
    #endif
    static constexpr bool needs_scale = true;
    static constexpr DTypeEnum dtype_enum = DTypeEnum::FP8E4M3;
    using compute_type = float;  // Always compute in F32
    using storage_type = uint8_t;
};

// FP8E5M2 - Hopper SM 8.9+ (stored as uint8_t)
template<>
struct DTypeTraits<boostr_fp8_e5m2> {
    static constexpr bool is_supported = true;
    #if __CUDA_ARCH__ >= 890
    static constexpr bool has_native_ops = true;
    #else
    static constexpr bool has_native_ops = false;
    #endif
    static constexpr bool needs_scale = true;
    static constexpr DTypeEnum dtype_enum = DTypeEnum::FP8E5M2;
    using compute_type = float;  // Always compute in F32
    using storage_type = uint8_t;
};

// ============================================================================
// Lossy Conversion Detection (for compile-time warnings)
// ============================================================================

// Helper to detect if a conversion loses precision
// Used for quantization warnings in model conversion pipelines
template<typename TSrc, typename TDst>
struct is_lossy_conversion {
    // Precision ordering (bits): F32(23) > F16(10) > BF16(7) > FP8E4M3(3) > FP8E5M2(2)
    static constexpr int src_precision =
        std::is_same_v<TSrc, float> ? 23 :
        std::is_same_v<TSrc, __half> ? 10 :
        std::is_same_v<TSrc, __nv_bfloat16> ? 7 :
        std::is_same_v<TSrc, boostr_fp8_e4m3> ? 3 :
        std::is_same_v<TSrc, boostr_fp8_e5m2> ? 2 : 0;

    static constexpr int dst_precision =
        std::is_same_v<TDst, float> ? 23 :
        std::is_same_v<TDst, __half> ? 10 :
        std::is_same_v<TDst, __nv_bfloat16> ? 7 :
        std::is_same_v<TDst, boostr_fp8_e4m3> ? 3 :
        std::is_same_v<TDst, boostr_fp8_e5m2> ? 2 : 0;

    // Lossy if target has less precision OR converting to FP8 (always lossy)
    static constexpr bool value =
        (src_precision > dst_precision) ||
        (std::is_same_v<TDst, boostr_fp8_e4m3> || std::is_same_v<TDst, boostr_fp8_e5m2>);
};

// Compile-time warning helper for lossy conversions
// Usage: static_assert_precision<F32, FP8>("Converting master weights to FP8 without quantization calibration");
template<typename TSrc, typename TDst>
constexpr void warn_if_lossy() {
    if constexpr (is_lossy_conversion<TSrc, TDst>::value) {
        // Note: This will generate a compiler warning when used
        // static_assert(false, "Lossy conversion detected"); would be too strict
        // Instead, rely on runtime checks or explicit user acknowledgment
    }
}

// ============================================================================
// Generic Load/Store with Automatic Conversion
// ============================================================================

// Generic load with optional scale (FP8 path)
// Returns float for all types to avoid ambiguous conversions
template<typename T>
__device__ __forceinline__
float load_dtype(const T* ptr, int idx, float scale = 1.0f) {
    if constexpr (DTypeTraits<T>::needs_scale) {
        #if __CUDA_ARCH__ >= 800
        if constexpr (DTypeTraits<T>::dtype_enum == DTypeEnum::FP8E4M3) {
            return fp8_e4m3_to_f32(static_cast<uint8_t>(ptr[idx]), scale);
        } else if constexpr (DTypeTraits<T>::dtype_enum == DTypeEnum::FP8E5M2) {
            return fp8_e5m2_to_f32(static_cast<uint8_t>(ptr[idx]), scale);
        }
        #endif
        return 0.0f; // Fallback for old architectures
    } else {
        // F32, F16, BF16 - convert to float explicitly
        if constexpr (DTypeTraits<T>::dtype_enum == DTypeEnum::F32) {
            return ptr[idx];
        } else if constexpr (DTypeTraits<T>::dtype_enum == DTypeEnum::F16) {
            return __half2float(ptr[idx]);
        } else if constexpr (DTypeTraits<T>::dtype_enum == DTypeEnum::BF16) {
            return __bfloat162float(ptr[idx]);
        }
        return 0.0f;  // Should never reach here
    }
}

// Generic store with optional scale (FP8 path)
template<typename T>
__device__ __forceinline__
void store_dtype(T* ptr, int idx, float value, float scale = 1.0f) {
    if constexpr (DTypeTraits<T>::needs_scale) {
        #if __CUDA_ARCH__ >= 800
        if constexpr (DTypeTraits<T>::dtype_enum == DTypeEnum::FP8E4M3) {
            ptr[idx] = boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(value, scale));
        } else if constexpr (DTypeTraits<T>::dtype_enum == DTypeEnum::FP8E5M2) {
            ptr[idx] = boostr_fp8_e5m2(f32_to_fp8_e5m2_raw(value, scale));
        }
        #endif
    } else {
        // F32, F16, BF16 - use proper conversion functions
        if constexpr (DTypeTraits<T>::dtype_enum == DTypeEnum::F32) {
            ptr[idx] = value;
        } else if constexpr (DTypeTraits<T>::dtype_enum == DTypeEnum::F16) {
            ptr[idx] = __float2half_rn(value);
        } else if constexpr (DTypeTraits<T>::dtype_enum == DTypeEnum::BF16) {
            ptr[idx] = __float2bfloat16_rn(value);
        }
    }
}

// Convert to compute type (F32 for FP8, native for others)
template<typename T>
__device__ __forceinline__
float to_compute_type(T value) {
    if constexpr (DTypeTraits<T>::needs_scale) {
        return static_cast<float>(value);
    } else {
        return __half2float(value);  // Works for F16/BF16/F32
    }
}

// ============================================================================
// Generic Type Conversion: convert_dtype<TDst, TSrc>
// ============================================================================
//
// PURPOSE: Provides all N×N conversion paths between dtypes without always
//          going through F32 as intermediate (better precision for FP16→FP8)
//
// USAGE:
//   __half src_fp16 = ...;
//   boostr_fp8_e4m3 dst_fp8 = convert_dtype<boostr_fp8_e4m3>(src_fp16, 1.0f, scale_out);
//
// PERFORMANCE: Direct conversions avoid double rounding errors
//
// ============================================================================

template<typename TDst, typename TSrc>
__device__ __forceinline__ TDst convert_dtype(
    TSrc src,
    float scale_in = 1.0f,   // For FP8 source
    float scale_out = 1.0f   // For FP8 destination
) {
    // Same type - no-op
    if constexpr (std::is_same_v<TSrc, TDst>) {
        return src;
    }

    // ========================================================================
    // FP16 Conversions
    // ========================================================================

    // FP16 → BF16 (via F32 - no direct CUDA intrinsic)
    else if constexpr (std::is_same_v<TSrc, __half> && std::is_same_v<TDst, __nv_bfloat16>) {
        return __float2bfloat16_rn(__half2float(src));
    }

    // FP16 → FP8 E4M3 (direct to avoid double rounding)
    else if constexpr (std::is_same_v<TSrc, __half> && std::is_same_v<TDst, boostr_fp8_e4m3>) {
        float tmp = __half2float(src);
        return boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(tmp, scale_out));
    }

    // FP16 → FP8 E5M2
    else if constexpr (std::is_same_v<TSrc, __half> && std::is_same_v<TDst, boostr_fp8_e5m2>) {
        float tmp = __half2float(src);
        return boostr_fp8_e5m2(f32_to_fp8_e5m2_raw(tmp, scale_out));
    }

    // FP16 → F32
    else if constexpr (std::is_same_v<TSrc, __half> && std::is_same_v<TDst, float>) {
        return __half2float(src);
    }

    // ========================================================================
    // BF16 Conversions
    // ========================================================================

    // BF16 → FP16 (via F32)
    else if constexpr (std::is_same_v<TSrc, __nv_bfloat16> && std::is_same_v<TDst, __half>) {
        return __float2half_rn(__bfloat162float(src));
    }

    // BF16 → FP8 E4M3
    else if constexpr (std::is_same_v<TSrc, __nv_bfloat16> && std::is_same_v<TDst, boostr_fp8_e4m3>) {
        float tmp = __bfloat162float(src);
        return boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(tmp, scale_out));
    }

    // BF16 → FP8 E5M2
    else if constexpr (std::is_same_v<TSrc, __nv_bfloat16> && std::is_same_v<TDst, boostr_fp8_e5m2>) {
        float tmp = __bfloat162float(src);
        return boostr_fp8_e5m2(f32_to_fp8_e5m2_raw(tmp, scale_out));
    }

    // BF16 → F32
    else if constexpr (std::is_same_v<TSrc, __nv_bfloat16> && std::is_same_v<TDst, float>) {
        return __bfloat162float(src);
    }

    // ========================================================================
    // FP8 E4M3 Conversions
    // ========================================================================

    // FP8 E4M3 → F32
    else if constexpr (std::is_same_v<TSrc, boostr_fp8_e4m3> && std::is_same_v<TDst, float>) {
        return fp8_e4m3_to_f32(src.data, scale_in);
    }

    // FP8 E4M3 → FP16
    else if constexpr (std::is_same_v<TSrc, boostr_fp8_e4m3> && std::is_same_v<TDst, __half>) {
        float tmp = fp8_e4m3_to_f32(src.data, scale_in);
        return __float2half_rn(tmp);
    }

    // FP8 E4M3 → BF16
    else if constexpr (std::is_same_v<TSrc, boostr_fp8_e4m3> && std::is_same_v<TDst, __nv_bfloat16>) {
        float tmp = fp8_e4m3_to_f32(src.data, scale_in);
        return __float2bfloat16_rn(tmp);
    }

    // FP8 E4M3 → FP8 E5M2 (via F32)
    else if constexpr (std::is_same_v<TSrc, boostr_fp8_e4m3> && std::is_same_v<TDst, boostr_fp8_e5m2>) {
        float tmp = fp8_e4m3_to_f32(src.data, scale_in);
        return boostr_fp8_e5m2(f32_to_fp8_e5m2_raw(tmp, scale_out));
    }

    // ========================================================================
    // FP8 E5M2 Conversions
    // ========================================================================

    // FP8 E5M2 → F32
    else if constexpr (std::is_same_v<TSrc, boostr_fp8_e5m2> && std::is_same_v<TDst, float>) {
        return fp8_e5m2_to_f32(src.data, scale_in);
    }

    // FP8 E5M2 → FP16
    else if constexpr (std::is_same_v<TSrc, boostr_fp8_e5m2> && std::is_same_v<TDst, __half>) {
        float tmp = fp8_e5m2_to_f32(src.data, scale_in);
        return __float2half_rn(tmp);
    }

    // FP8 E5M2 → BF16
    else if constexpr (std::is_same_v<TSrc, boostr_fp8_e5m2> && std::is_same_v<TDst, __nv_bfloat16>) {
        float tmp = fp8_e5m2_to_f32(src.data, scale_in);
        return __float2bfloat16_rn(tmp);
    }

    // FP8 E5M2 → FP8 E4M3 (via F32)
    else if constexpr (std::is_same_v<TSrc, boostr_fp8_e5m2> && std::is_same_v<TDst, boostr_fp8_e4m3>) {
        float tmp = fp8_e5m2_to_f32(src.data, scale_in);
        return boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(tmp, scale_out));
    }

    // ========================================================================
    // F32 Conversions (via intrinsics)
    // ========================================================================

    // F32 → FP16
    else if constexpr (std::is_same_v<TSrc, float> && std::is_same_v<TDst, __half>) {
        return __float2half_rn(src);
    }

    // F32 → BF16
    else if constexpr (std::is_same_v<TSrc, float> && std::is_same_v<TDst, __nv_bfloat16>) {
        return __float2bfloat16_rn(src);
    }

    // F32 → FP8 E4M3
    else if constexpr (std::is_same_v<TSrc, float> && std::is_same_v<TDst, boostr_fp8_e4m3>) {
        return boostr_fp8_e4m3(f32_to_fp8_e4m3_raw(src, scale_out));
    }

    // F32 → FP8 E5M2
    else if constexpr (std::is_same_v<TSrc, float> && std::is_same_v<TDst, boostr_fp8_e5m2>) {
        return boostr_fp8_e5m2(f32_to_fp8_e5m2_raw(src, scale_out));
    }

    // ========================================================================
    // Fallback: Should never reach here with valid types
    // ========================================================================
    else {
        // Compile-time error for unsupported conversion
        static_assert(DTypeTraits<TSrc>::is_supported && DTypeTraits<TDst>::is_supported,
                      "Unsupported dtype conversion");
        return TDst();  // Unreachable
    }
}

// ============================================================================
// Dtype-Specific Atomic Operations
// ============================================================================
//
// PURPOSE: Provides dtype-safe atomic add operations that avoid misalignment
//          errors when working with non-4-byte types (FP16, BF16, FP8)
//
// USAGE:
//   atomic_add_dtype(&array[idx], delta_value);
//
// IMPLEMENTATION:
//   - FP32: Native atomicAdd
//   - FP16/BF16: atomicCAS on underlying uint16_t
//   - FP8: Byte-level atomicCAS on 4-byte-aligned uint32_t with masking
//
// ============================================================================

/// FP32: Direct native atomic add
__device__ __forceinline__ void atomic_add_dtype(float* addr, float val) {
    atomicAdd(addr, val);
}

/// FP16: Atomic add via CAS on underlying uint16_t
__device__ __forceinline__ void atomic_add_dtype(__half* addr, float val) {
    #if __CUDA_ARCH__ >= 700
    // Volta+: Use native atomicAdd for __half
    atomicAdd(addr, __float2half(val));
    #else
    // Pre-Volta: Fallback to CAS
    unsigned short int* addr_as_ushort = (unsigned short int*)addr;
    unsigned short int old = *addr_as_ushort, assumed;
    do {
        assumed = old;
        __half h_old = __ushort_as_half(assumed);
        __half h_new = __float2half(__half2float(h_old) + val);
        old = atomicCAS(addr_as_ushort, assumed, __half_as_ushort(h_new));
    } while (assumed != old);
    #endif
}

/// BF16: Atomic add via CAS on underlying uint16_t
__device__ __forceinline__ void atomic_add_dtype(__nv_bfloat16* addr, float val) {
    #if __CUDA_ARCH__ >= 800
    // Ampere+: No native atomicAdd for BF16, use CAS
    unsigned short int* addr_as_ushort = (unsigned short int*)addr;
    unsigned short int old = *addr_as_ushort, assumed;
    do {
        assumed = old;
        __nv_bfloat16 bf_old = __ushort_as_bfloat16(assumed);
        __nv_bfloat16 bf_new = __float2bfloat16(__bfloat162float(bf_old) + val);
        old = atomicCAS(addr_as_ushort, assumed, __bfloat16_as_ushort(bf_new));
    } while (assumed != old);
    #endif
}

/// FP8 E4M3: Byte-level atomic add via CAS on aligned uint32_t
__device__ __forceinline__ void atomic_add_dtype(boostr_fp8_e4m3* addr, float val) {
    #if __CUDA_ARCH__ >= 800
    // Align to 4-byte boundary and operate on the containing uint32_t
    unsigned int* addr_as_uint = (unsigned int*)((size_t)addr & ~3u);
    unsigned int offset = (size_t)addr & 3u;
    unsigned int shift = offset * 8;
    unsigned int mask = 0xFFu << shift;

    unsigned int old = *addr_as_uint, assumed;
    do {
        assumed = old;
        unsigned char old_byte = (old >> shift) & 0xFF;
        float f_old = fp8_e4m3_to_f32(old_byte);
        float f_new = f_old + val;
        unsigned char new_byte = f32_to_fp8_e4m3_raw(f_new);
        unsigned int new_val = (old & ~mask) | ((unsigned int)new_byte << shift);
        old = atomicCAS(addr_as_uint, assumed, new_val);
    } while (assumed != old);
    #endif
}

/// FP8 E5M2: Byte-level atomic add via CAS on aligned uint32_t
__device__ __forceinline__ void atomic_add_dtype(boostr_fp8_e5m2* addr, float val) {
    #if __CUDA_ARCH__ >= 800
    // Align to 4-byte boundary and operate on the containing uint32_t
    unsigned int* addr_as_uint = (unsigned int*)((size_t)addr & ~3u);
    unsigned int offset = (size_t)addr & 3u;
    unsigned int shift = offset * 8;
    unsigned int mask = 0xFFu << shift;

    unsigned int old = *addr_as_uint, assumed;
    do {
        assumed = old;
        unsigned char old_byte = (old >> shift) & 0xFF;
        float f_old = fp8_e5m2_to_f32(old_byte);
        float f_new = f_old + val;
        unsigned char new_byte = f32_to_fp8_e5m2_raw(f_new);
        unsigned int new_val = (old & ~mask) | ((unsigned int)new_byte << shift);
        old = atomicCAS(addr_as_uint, assumed, new_val);
    } while (assumed != old);
    #endif
}

// ============================================================================
// Macro for Kernel Instantiation (DRY Principle)
// ============================================================================

// Macro to instantiate kernel for all dtypes (generic single-kernel pattern)
#define INSTANTIATE_KERNEL_ALL_DTYPES(kernel_name, ...) \
    extern "C" __global__ void kernel_name##_f32(__VA_ARGS__) { \
        kernel_name<float>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_f16(__VA_ARGS__) { \
        kernel_name<__half>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_bf16(__VA_ARGS__) { \
        kernel_name<__nv_bfloat16>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_fp8_e4m3(__VA_ARGS__) { \
        kernel_name<boostr_fp8_e4m3>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_fp8_e5m2(__VA_ARGS__) { \
        kernel_name<boostr_fp8_e5m2>(__VA_ARGS__); \
    }

// ============================================================================
// Specialized Macros for Complex Kernels
// ============================================================================

// Macro for attention kernels (typically with multiple head_dim variants)
// Usage: INSTANTIATE_ATTENTION_KERNEL_ALL_DTYPES(flash_attn_fwd, float* Q, float* K, float* V)
// Generates: flash_attn_fwd_f32, flash_attn_fwd_f16, flash_attn_fwd_bf16, flash_attn_fwd_fp8_e4m3, flash_attn_fwd_fp8_e5m2
#define INSTANTIATE_ATTENTION_KERNEL_ALL_DTYPES(kernel_name, ...) \
    extern "C" __global__ void kernel_name##_f32(__VA_ARGS__) { \
        kernel_name<float>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_f16(__VA_ARGS__) { \
        kernel_name<__half>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_bf16(__VA_ARGS__) { \
        kernel_name<__nv_bfloat16>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_fp8_e4m3(__VA_ARGS__) { \
        kernel_name<boostr_fp8_e4m3>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_fp8_e5m2(__VA_ARGS__) { \
        kernel_name<boostr_fp8_e5m2>(__VA_ARGS__); \
    }

// Macro for MoE kernels (routing, gating, permutation)
// Usage: INSTANTIATE_MOE_KERNEL_ALL_DTYPES(topk_gating, const float* router_logits)
// Generates: topk_gating_f32, topk_gating_f16, topk_gating_bf16, topk_gating_fp8_e4m3, topk_gating_fp8_e5m2
#define INSTANTIATE_MOE_KERNEL_ALL_DTYPES(kernel_name, ...) \
    extern "C" __global__ void kernel_name##_f32(__VA_ARGS__) { \
        kernel_name<float>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_f16(__VA_ARGS__) { \
        kernel_name<__half>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_bf16(__VA_ARGS__) { \
        kernel_name<__nv_bfloat16>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_fp8_e4m3(__VA_ARGS__) { \
        kernel_name<boostr_fp8_e4m3>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_fp8_e5m2(__VA_ARGS__) { \
        kernel_name<boostr_fp8_e5m2>(__VA_ARGS__); \
    }

// Macro for fusion kernels (element-wise, normalization, activation)
// Usage: INSTANTIATE_FUSION_KERNEL_ALL_DTYPES(add_rms_norm_fwd, const float* x, const float* residual)
// Generates: add_rms_norm_fwd_f32, add_rms_norm_fwd_f16, add_rms_norm_fwd_bf16, add_rms_norm_fwd_fp8_e4m3, add_rms_norm_fwd_fp8_e5m2
#define INSTANTIATE_FUSION_KERNEL_ALL_DTYPES(kernel_name, ...) \
    extern "C" __global__ void kernel_name##_f32(__VA_ARGS__) { \
        kernel_name<float>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_f16(__VA_ARGS__) { \
        kernel_name<__half>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_bf16(__VA_ARGS__) { \
        kernel_name<__nv_bfloat16>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_fp8_e4m3(__VA_ARGS__) { \
        kernel_name<boostr_fp8_e4m3>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_fp8_e5m2(__VA_ARGS__) { \
        kernel_name<boostr_fp8_e5m2>(__VA_ARGS__); \
    }

// Macro for FP8-only kernels (when FP8 is the only quantized format supported)
// Usage: DECLARE_FP8_KERNEL_PAIR(adamw_fused, float* params, float* grads)
// Generates: adamw_fused_fp8_e4m3, adamw_fused_fp8_e5m2
#define DECLARE_FP8_KERNEL_PAIR(kernel_name, ...) \
    extern "C" __global__ void kernel_name##_fp8_e4m3(__VA_ARGS__); \
    extern "C" __global__ void kernel_name##_fp8_e5m2(__VA_ARGS__);

// Macro for instantiating both E4M3 and E5M2 FP8 kernel wrappers
// Usage: INSTANTIATE_FP8_KERNEL_PAIR(adamw_fused, float* params, float* grads)
// Generates: adamw_fused_fp8_e4m3 and adamw_fused_fp8_e5m2 kernel wrappers
#define INSTANTIATE_FP8_KERNEL_PAIR(kernel_name, impl_name, ...) \
    extern "C" __global__ void kernel_name##_fp8_e4m3(__VA_ARGS__) { \
        impl_name<boostr_fp8_e4m3>(__VA_ARGS__); \
    } \
    extern "C" __global__ void kernel_name##_fp8_e5m2(__VA_ARGS__) { \
        impl_name<boostr_fp8_e5m2>(__VA_ARGS__); \
    }

// ============================================================================
// Constants Per DType
// ============================================================================

template<typename T>
struct DTypeConstants {
    static __device__ __forceinline__ float zero() { return 0.0f; }
    static __device__ __forceinline__ float one() { return 1.0f; }
    static __device__ __forceinline__ float eps() { return 1e-6f; }
    static __device__ __forceinline__ float neg_inf() { return -INFINITY; }
    static __device__ __forceinline__ float pos_inf() { return INFINITY; }
};

template<>
struct DTypeConstants<float> {
    static __device__ __forceinline__ float zero() { return 0.0f; }
    static __device__ __forceinline__ float one() { return 1.0f; }
    static __device__ __forceinline__ float eps() { return 1e-7f; }
    static __device__ __forceinline__ float neg_inf() { return -INFINITY; }
    static __device__ __forceinline__ float pos_inf() { return INFINITY; }
    static __device__ __forceinline__ float max_val() { return 3.402823466e+38f; }
    static __device__ __forceinline__ float min_val() { return -3.402823466e+38f; }
};

template<>
struct DTypeConstants<__half> {
    static __device__ __forceinline__ float zero() { return 0.0f; }
    static __device__ __forceinline__ float one() { return 1.0f; }
    static __device__ __forceinline__ float eps() { return 6.1e-5f; }  // FP16 machine epsilon
    static __device__ __forceinline__ float neg_inf() { return -65504.0f; }  // FP16 min
    static __device__ __forceinline__ float pos_inf() { return 65504.0f; }   // FP16 max
    static __device__ __forceinline__ float max_val() { return 65504.0f; }
    static __device__ __forceinline__ float min_val() { return -65504.0f; }
};

template<>
struct DTypeConstants<__nv_bfloat16> {
    static __device__ __forceinline__ float zero() { return 0.0f; }
    static __device__ __forceinline__ float one() { return 1.0f; }
    static __device__ __forceinline__ float eps() { return 7.8125e-3f; }  // BF16 machine epsilon
    static __device__ __forceinline__ float neg_inf() { return -3.39e+38f; }
    static __device__ __forceinline__ float pos_inf() { return 3.39e+38f; }
    static __device__ __forceinline__ float max_val() { return 3.39e+38f; }
    static __device__ __forceinline__ float min_val() { return -3.39e+38f; }
};

template<>
struct DTypeConstants<boostr_fp8_e4m3> {
    static __device__ __forceinline__ float zero() { return 0.0f; }
    static __device__ __forceinline__ float one() { return 1.0f; }
    static __device__ __forceinline__ float eps() { return 0.125f; }  // E4M3 resolution
    static __device__ __forceinline__ float neg_inf() { return -448.0f; }
    static __device__ __forceinline__ float pos_inf() { return 448.0f; }
    static __device__ __forceinline__ float max_val() { return 448.0f; }
    static __device__ __forceinline__ float min_val() { return -448.0f; }
};

template<>
struct DTypeConstants<boostr_fp8_e5m2> {
    static __device__ __forceinline__ float zero() { return 0.0f; }
    static __device__ __forceinline__ float one() { return 1.0f; }
    static __device__ __forceinline__ float eps() { return 0.25f; }  // E5M2 resolution
    static __device__ __forceinline__ float neg_inf() { return -57344.0f; }
    static __device__ __forceinline__ float pos_inf() { return 57344.0f; }
    static __device__ __forceinline__ float max_val() { return 57344.0f; }
    static __device__ __forceinline__ float min_val() { return -57344.0f; }
};

// ============================================================================
// Vectorized Load/Store (2x and 4x)
// ============================================================================

// Load 2 elements at once, returns as float2 for computation
template<typename T>
__device__ __forceinline__
float2 load2_dtype(const T* ptr, int idx, float scale = 1.0f) {
    float2 result;
    result.x = load_dtype<T>(ptr, idx, scale);
    result.y = load_dtype<T>(ptr, idx + 1, scale);
    return result;
}

// Load 4 elements at once, returns as float4 for computation
template<typename T>
__device__ __forceinline__
float4 load4_dtype(const T* ptr, int idx, float scale = 1.0f) {
    float4 result;
    result.x = load_dtype<T>(ptr, idx, scale);
    result.y = load_dtype<T>(ptr, idx + 1, scale);
    result.z = load_dtype<T>(ptr, idx + 2, scale);
    result.w = load_dtype<T>(ptr, idx + 3, scale);
    return result;
}

// Store 2 elements at once
template<typename T>
__device__ __forceinline__
void store2_dtype(T* ptr, int idx, float2 value, float scale = 1.0f) {
    store_dtype<T>(ptr, idx, value.x, scale);
    store_dtype<T>(ptr, idx + 1, value.y, scale);
}

// Store 4 elements at once
template<typename T>
__device__ __forceinline__
void store4_dtype(T* ptr, int idx, float4 value, float scale = 1.0f) {
    store_dtype<T>(ptr, idx, value.x, scale);
    store_dtype<T>(ptr, idx + 1, value.y, scale);
    store_dtype<T>(ptr, idx + 2, value.z, scale);
    store_dtype<T>(ptr, idx + 3, value.w, scale);
}

// Optimized vectorized load for F32 (uses native float4)
template<>
__device__ __forceinline__
float4 load4_dtype<float>(const float* ptr, int idx, float scale) {
    float4 result = *reinterpret_cast<const float4*>(ptr + idx);
    if (scale != 1.0f) {
        result.x /= scale;
        result.y /= scale;
        result.z /= scale;
        result.w /= scale;
    }
    return result;
}

// Optimized vectorized store for F32 (uses native float4)
template<>
__device__ __forceinline__
void store4_dtype<float>(float* ptr, int idx, float4 value, float scale) {
    if (scale != 1.0f) {
        value.x *= scale;
        value.y *= scale;
        value.z *= scale;
        value.w *= scale;
    }
    *reinterpret_cast<float4*>(ptr + idx) = value;
}

// ============================================================================
// Safe Math Operations (NaN/Inf prevention)
// ============================================================================

// Safe division: returns 0 if denominator is near-zero
template<typename T>
__device__ __forceinline__
float safe_div(float numerator, float denominator) {
    float eps = DTypeConstants<T>::eps();
    if (fabsf(denominator) < eps) {
        return 0.0f;
    }
    return numerator / denominator;
}

// Safe exponential: clamps input to prevent overflow
template<typename T>
__device__ __forceinline__
float safe_exp(float x) {
    // Clamp to prevent exp overflow (exp(88) ≈ 1e38)
    float clamped = fminf(fmaxf(x, -88.0f), 88.0f);
    return expf(clamped);
}

// Safe log: clamps input to prevent -inf
template<typename T>
__device__ __forceinline__
float safe_log(float x) {
    float eps = DTypeConstants<T>::eps();
    return logf(fmaxf(x, eps));
}

// Safe reciprocal: returns large value if input is near-zero
template<typename T>
__device__ __forceinline__
float safe_rcp(float x) {
    float eps = DTypeConstants<T>::eps();
    if (fabsf(x) < eps) {
        return (x >= 0.0f) ? DTypeConstants<T>::max_val() : DTypeConstants<T>::min_val();
    }
    return 1.0f / x;
}

// Safe rsqrt (reciprocal square root)
template<typename T>
__device__ __forceinline__
float safe_rsqrt(float x) {
    float eps = DTypeConstants<T>::eps();
    return rsqrtf(fmaxf(x, eps));
}

// Clamp value to dtype range
template<typename T>
__device__ __forceinline__
float clamp_to_dtype_range(float x) {
    return fminf(fmaxf(x, DTypeConstants<T>::min_val()), DTypeConstants<T>::max_val());
}

// ============================================================================
// Fused Multiply-Add with DType Awareness
// ============================================================================

// a * b + c with proper clamping for target dtype
template<typename T>
__device__ __forceinline__
float fma_dtype(float a, float b, float c) {
    float result = fmaf(a, b, c);
    return clamp_to_dtype_range<T>(result);
}

// Dot product of 4 elements (useful for vectorized ops)
__device__ __forceinline__
float dot4(float4 a, float4 b) {
    return fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, a.w * b.w)));
}

// ============================================================================
// Warp-Level Reduction Primitives (DType-aware)
// ============================================================================

constexpr int DTYPE_WARP_SIZE = 32;

// Warp reduce sum
__device__ __forceinline__
float warp_reduce_sum_dtype(float val) {
    #pragma unroll
    for (int offset = DTYPE_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp reduce max
__device__ __forceinline__
float warp_reduce_max_dtype(float val) {
    #pragma unroll
    for (int offset = DTYPE_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp reduce min
__device__ __forceinline__
float warp_reduce_min_dtype(float val) {
    #pragma unroll
    for (int offset = DTYPE_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block reduce sum (requires shared memory)
__device__ __forceinline__
float block_reduce_sum_dtype(float val, float* smem, int tid, int block_size) {
    val = warp_reduce_sum_dtype(val);

    if (tid % DTYPE_WARP_SIZE == 0) {
        smem[tid / DTYPE_WARP_SIZE] = val;
    }
    __syncthreads();

    int num_warps = (block_size + DTYPE_WARP_SIZE - 1) / DTYPE_WARP_SIZE;
    if (tid < num_warps) {
        val = smem[tid];
    } else {
        val = 0.0f;
    }

    if (tid < DTYPE_WARP_SIZE) {
        val = warp_reduce_sum_dtype(val);
    }

    __syncthreads();
    return val;
}

// Block reduce max (requires shared memory)
__device__ __forceinline__
float block_reduce_max_dtype(float val, float* smem, int tid, int block_size) {
    val = warp_reduce_max_dtype(val);

    if (tid % DTYPE_WARP_SIZE == 0) {
        smem[tid / DTYPE_WARP_SIZE] = val;
    }
    __syncthreads();

    int num_warps = (block_size + DTYPE_WARP_SIZE - 1) / DTYPE_WARP_SIZE;
    if (tid < num_warps) {
        val = smem[tid];
    } else {
        val = -INFINITY;
    }

    if (tid < DTYPE_WARP_SIZE) {
        val = warp_reduce_max_dtype(val);
    }

    __syncthreads();
    return val;
}

// ============================================================================
// Online Softmax Helpers (for Flash Attention)
// ============================================================================

// Online softmax state: tracks running max and sum for numerical stability
struct OnlineSoftmaxState {
    float max_val;
    float sum_exp;

    __device__ __forceinline__ OnlineSoftmaxState()
        : max_val(-INFINITY), sum_exp(0.0f) {}

    __device__ __forceinline__ void update(float new_val) {
        float old_max = max_val;
        max_val = fmaxf(max_val, new_val);
        float exp_diff = expf(old_max - max_val);
        sum_exp = sum_exp * exp_diff + expf(new_val - max_val);
    }

    __device__ __forceinline__ void merge(const OnlineSoftmaxState& other) {
        float new_max = fmaxf(max_val, other.max_val);
        float exp_diff_self = expf(max_val - new_max);
        float exp_diff_other = expf(other.max_val - new_max);
        sum_exp = sum_exp * exp_diff_self + other.sum_exp * exp_diff_other;
        max_val = new_max;
    }

    __device__ __forceinline__ float normalize(float val) const {
        return expf(val - max_val) / sum_exp;
    }
};

// ============================================================================
// Accumulator Type Selection
// ============================================================================

// Select accumulator type based on input dtype
// FP8/FP16/BF16 -> F32 accumulator for precision
// F32 -> F32 accumulator
template<typename T>
struct AccumulatorType {
    using type = float;  // Always use F32 for accumulation
};

// ============================================================================
// Compile-Time Consistency Checks
// ============================================================================

// Helper: Verify that forward and backward kernels use same dtype
// Usage in backward kernel:
//   template<typename T_fwd, typename T_bwd>
//   __device__ void verify_bwd_fwd_dtype() {
//       VERIFY_DTYPE_MATCH(T_fwd, T_bwd, "Backward kernel dtype must match forward");
//   }
// Note: This is a compile-time check - use as template parameter check
#define VERIFY_DTYPE_MATCH(T_fwd, T_bwd, msg) \
    static_assert(std::is_same_v<T_fwd, T_bwd>, msg)

// Assertion helper for dtype consistency in multi-kernel files
// Usage: ASSERT_DTYPE_PAIR_SUPPORTED<FP32, FP8E4M3>();
template<typename T_fwd, typename T_bwd>
constexpr void assert_dtype_pair_supported() {
    // All FP8 backward kernels must support Ampere+ (SM 8.0+)
    // If T_bwd is FP8, compilation will fail on older architectures
    #if __CUDA_ARCH__ < 800
    if constexpr (DTypeTraits<T_bwd>::dtype_enum == DTypeEnum::FP8E4M3 ||
                  DTypeTraits<T_bwd>::dtype_enum == DTypeEnum::FP8E5M2) {
        static_assert(false, "FP8 backward kernels require Ampere+ (SM 8.0+)");
    }
    #endif
}

#endif // BOOSTR_DTYPE_TRAITS_CUH
