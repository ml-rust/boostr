// Device-side decoder for TCF native quantized payloads.
//
// # Why this file holds a second copy of the format
//
// `tcf-core` IS the reference codec, and boostr's CPU kernels call it rather
// than restating it. A CUDA kernel cannot: device code has no path back into
// the Rust crate. So the read-direction rules below are written once, here,
// and every CUDA TCF kernel goes through them.
//
// What is duplicated is deliberately minimal, and is exactly the part that
// cannot be hoisted to the host:
//   - the nibble/bit position of one element inside one tile (Section 14.1,
//     Section 14.2, Section 14.3),
//   - the 6-bit sub-field position inside one super-block (Section 14.6),
//   - the reconstruction math (Section 13.0, Section 13.0.1, Section 13.3,
//     Section 13.4).
//
// What is NOT duplicated: plane order, plane sizes, and every byte offset
// between planes. The host computes those from `tcf-core`'s own `QuantLayout`
// and passes them in, so a layout change reaches this file as new numbers
// rather than as a stale constant. See `quant/tcf/planes.rs`.
//
// The gate is `tests/backend_parity/quant_tcf.rs`: every encoding, decoded by
// this file and by `tcf_core::unpack` + `tcf_core::dequantize`, must agree.

#pragma once

#include <cuda_fp16.h>

// Scale forms, mirroring `tcf_core::encoding::ScaleForm`. The host passes the
// discriminant; see `quant/tcf/planes.rs` for the one place it is set.
#define TCF_SCALE_FLAT 0u
#define TCF_SCALE_TWO_LEVEL_U8 1u
#define TCF_SCALE_TWO_LEVEL_U6M6 2u

// Section 13.3: one super-scale covers four 64-element tiles.
#define TCF_SUPER_BLOCK_TILES 4u

// v1 fixes the execution tile at 64 logical elements (Section 12.1), which is
// what `Code64` can address. The host refuses any other width before launch.
#define TCF_TILE 64u

// Section 13.3 / Section 13.4 decode divisors for the two-level forms.
#define TCF_SUB_SCALE_LEVELS_U8 255.0f
#define TCF_SUB_SCALE_LEVELS_U6 63.0f
#define TCF_SUB_MIN_LEVELS_U6 31.0f

// Byte offsets of every plane of one payload, plus the geometry needed to
// address a group. Assembled inside a kernel from its scalar arguments, so
// nothing here depends on a host/device struct layout agreeing.
struct TcfLayout {
    unsigned long long code_high_off;
    unsigned long long scale_off;
    unsigned long long min_off;
    unsigned long long super_off;
    unsigned long long super_min_off;
    unsigned int bits;
    unsigned int group;
    unsigned int groups_per_tile;
    unsigned int symmetric;
    unsigned int scale_form;
    unsigned int sub_block_bytes;
};

static __device__ __forceinline__ TcfLayout tcf_layout(
    unsigned long long code_high_off,
    unsigned long long scale_off,
    unsigned long long min_off,
    unsigned long long super_off,
    unsigned long long super_min_off,
    unsigned int bits,
    unsigned int group,
    unsigned int groups_per_tile,
    unsigned int symmetric,
    unsigned int scale_form,
    unsigned int sub_block_bytes
) {
    TcfLayout l;
    l.code_high_off = code_high_off;
    l.scale_off = scale_off;
    l.min_off = min_off;
    l.super_off = super_off;
    l.super_min_off = super_min_off;
    l.bits = bits;
    l.group = group;
    l.groups_per_tile = groups_per_tile;
    l.symmetric = symmetric;
    l.scale_form = scale_form;
    l.sub_block_bytes = sub_block_bytes;
    return l;
}

// A little-endian binary16 read as f32. Every binary16 is exactly
// representable in f32, so this agrees with `tcf_core::binary16::bits_to_f32`
// bit for bit.
static __device__ __forceinline__ float tcf_read_binary16(const unsigned char* p) {
    __half h;
    memcpy(&h, p, sizeof(__half));
    return __half2float(h);
}

// Section 14.6 read direction. Field `slot` of super-block `block` occupies
// bits [6*slot, 6*slot+6) of that block's byte run, LSB-first, and a field
// starting past bit 2 continues into the next byte.
static __device__ __forceinline__ unsigned int tcf_read_packed6(
    const unsigned char* plane,
    unsigned int block,
    unsigned int slot,
    unsigned int sub_block_bytes
) {
    size_t base = (size_t)block * (size_t)sub_block_bytes;
    unsigned int bit = slot * 6u;
    size_t byte_index = base + (size_t)(bit >> 3);
    unsigned int offset = bit & 7u;
    unsigned int value = (unsigned int)plane[byte_index] >> offset;
    if (offset > 2u) {
        value |= ((unsigned int)plane[byte_index + 1]) << (8u - offset);
    }
    return value & 0x3fu;
}

// The effective (scale, minimum) of group `g` of tile `tile`.
//
// Section 13.0 / Section 13.0.1: a flat layout stores both outright as
// binary16. Section 13.3: a two-level u8 layout stores a u8 sub-scale under a
// per-super-block binary16 super-scale. Section 13.4: a two-level 6-bit layout
// stores a 6-bit unsigned sub-scale and a 6-bit signed sub-minimum, both
// bit-packed per super-block, under a super-scale and a super-minimum.
//
// Multiply then divide once, in that fixed order, matching
// `QuantLayout::group_scale` and `QuantLayout::group_min`. The rounding
// intrinsics are deliberate: this translation unit is compiled with
// `--use_fast_math`, whose approximate division would otherwise put the last
// bit of every two-level scale out of step with the CPU path.
//
// A symmetric group has no minimum and yields 0.0f, which its caller must not
// add — see `tcf_value`.
static __device__ __forceinline__ void tcf_group_values(
    const unsigned char* __restrict__ payload,
    TcfLayout l,
    unsigned int tile,
    unsigned int g,
    float* out_scale,
    float* out_min
) {
    unsigned int block = tile / TCF_SUPER_BLOCK_TILES;
    unsigned int slot = (tile % TCF_SUPER_BLOCK_TILES) * l.groups_per_tile + g;
    size_t global = (size_t)tile * (size_t)l.groups_per_tile + (size_t)g;

    if (l.scale_form == TCF_SCALE_TWO_LEVEL_U8) {
        float super = tcf_read_binary16(payload + l.super_off + (size_t)block * 2u);
        unsigned int sub = (unsigned int)payload[(size_t)l.scale_off + global];
        *out_scale = __fdiv_rn(__fmul_rn(super, (float)sub), TCF_SUB_SCALE_LEVELS_U8);
        *out_min = 0.0f;
        return;
    }
    if (l.scale_form == TCF_SCALE_TWO_LEVEL_U6M6) {
        float super = tcf_read_binary16(payload + l.super_off + (size_t)block * 2u);
        unsigned int sub = tcf_read_packed6(
            payload + l.scale_off, block, slot, l.sub_block_bytes);
        *out_scale = __fdiv_rn(__fmul_rn(super, (float)sub), TCF_SUB_SCALE_LEVELS_U6);

        float super_min = tcf_read_binary16(payload + l.super_min_off + (size_t)block * 2u);
        unsigned int field = tcf_read_packed6(
            payload + l.min_off, block, slot, l.sub_block_bytes);
        // A 6-bit two's-complement field: 0..=31 is itself, 32..=63 is
        // `field - 64`. The reserved -32 never reaches here in a payload
        // `tcf-core` accepted.
        int level = (int)field;
        if (level > 31) {
            level -= 64;
        }
        *out_min = __fdiv_rn(__fmul_rn(super_min, (float)level), TCF_SUB_MIN_LEVELS_U6);
        return;
    }

    *out_scale = tcf_read_binary16(payload + l.scale_off + global * 2u);
    *out_min = l.symmetric
        ? 0.0f
        : tcf_read_binary16(payload + l.min_off + global * 2u);
}

// The code of element `e` of tile `tile`, already sign-resolved.
//
// Section 14.1 / Section 14.1.1: a 4-bit tile is a 32-byte run, element `e` in
// byte `e / 2`, low nibble when `e` is even. Section 14.3: an 8-bit tile is one
// byte per element. Section 14.2: a 6-bit code plane is a whole low-nibble
// sub-plane followed by a whole high-two-bit sub-plane, the second starting at
// `code_high_off`.
//
// Symmetric codes sign-extend from `bits`; Section 13.2's reserved
// most-negative pattern is rejected by `tcf-core` when the payload is read, so
// device code sign-extends it rather than re-checking per element.
static __device__ __forceinline__ int tcf_code(
    const unsigned char* __restrict__ payload,
    TcfLayout l,
    unsigned int tile,
    unsigned int e
) {
    unsigned int field;
    if (l.bits == 4u) {
        unsigned char byte = payload[(size_t)tile * (size_t)(TCF_TILE / 2u) + (size_t)(e >> 1)];
        field = (e & 1u) ? ((unsigned int)byte >> 4) & 0x0fu : (unsigned int)byte & 0x0fu;
    } else if (l.bits == 8u) {
        field = (unsigned int)payload[(size_t)tile * (size_t)TCF_TILE + (size_t)e];
    } else {
        unsigned char low_byte =
            payload[(size_t)tile * (size_t)(TCF_TILE / 2u) + (size_t)(e >> 1)];
        unsigned int low =
            (e & 1u) ? ((unsigned int)low_byte >> 4) & 0x0fu : (unsigned int)low_byte & 0x0fu;
        unsigned char high_byte = payload[(size_t)l.code_high_off
                                          + (size_t)tile * (size_t)(TCF_TILE / 4u)
                                          + (size_t)(e >> 2)];
        unsigned int top = ((unsigned int)high_byte >> ((e & 3u) * 2u)) & 0x03u;
        field = low | (top << 4);
    }

    if (l.symmetric == 0u) {
        return (int)field;
    }
    int reserved = (int)(1u << (l.bits - 1u));
    int value = (int)field;
    return (value > reserved) ? value - 2 * reserved : value;
}

// Section 13.0 / Section 13.0.1 applied to one element, against its group's
// already-resolved parameters.
//
// Symmetric and asymmetric are separate expressions rather than one with a
// zero minimum: `-0.0f + 0.0f` is `+0.0f`, so folding them would change the
// sign of a zero the CPU path emits.
static __device__ __forceinline__ float tcf_value(
    int code,
    float scale,
    float min_value,
    unsigned int symmetric
) {
    if (symmetric != 0u) {
        return __fmul_rn(scale, (float)code);
    }
    return __fadd_rn(__fmul_rn(scale, (float)code), min_value);
}
