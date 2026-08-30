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

// Section 13.2 sign resolution of one raw code field.
//
// An asymmetric code is an unsigned level. A symmetric code sign-extends from
// `bits`; Section 13.2's reserved most-negative pattern is rejected by
// `tcf-core` when the payload is read, so device code sign-extends it rather
// than re-checking per element.
//
// Factored out because two readers need it — `tcf_code` below, one element at
// a time, and `tcf_run_code`, a whole word of codes at a time. A second copy
// of this expression is exactly the kind of drift the parity test would only
// catch after it had already shipped a wrong sign.
static __device__ __forceinline__ int tcf_sign_resolve(
    unsigned int field,
    unsigned int bits,
    unsigned int symmetric
) {
    if (symmetric == 0u) {
        return (int)field;
    }
    int reserved = (int)(1u << (bits - 1u));
    int value = (int)field;
    return (value > reserved) ? value - 2 * reserved : value;
}

// The code of element `e` of tile `tile`, already sign-resolved.
//
// Section 14.1 / Section 14.1.1: a 4-bit tile is a 32-byte run, element `e` in
// byte `e / 2`, low nibble when `e` is even. Section 14.3: an 8-bit tile is one
// byte per element. Section 14.2: a 6-bit code plane is a whole low-nibble
// sub-plane followed by a whole high-two-bit sub-plane, the second starting at
// `code_high_off`.
//
// The sign is resolved by `tcf_sign_resolve`.
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

    return tcf_sign_resolve(field, l.bits, l.symmetric);
}

// Execution tiles one warp decodes per step of the GEMV inner loop, and the
// elements one lane owns of that step. 8 * 64 / 32 == 16.
#define TCF_RUN_TILES 8u
#define TCF_RUN (TCF_RUN_TILES * TCF_TILE)
#define TCF_RUN_PER_LANE 16u

// The 16 codes lane `lane` owns of the eight-tile run starting at `tile0`,
// read as whole machine words.
//
// # Why a run, and why sixteen elements
//
// `tcf_code` recomputes a byte address per element and issues one narrow load,
// so a warp reading 32 codes covers 32 bytes for an 8-bit encoding and 16 for
// a 4-bit one — against the 128 bytes a memory transaction carries. Reading
// the same codes as one `uint4` (8-bit) or one `uint2` plus one `uint` (4- and
// 6-bit) per lane puts 512 or 256 consecutive bytes under one instruction.
//
// Sixteen elements per lane is the width that makes the SCALE side work out
// too. `tcf-core`'s `MAX_GROUPS_PER_TILE` fixes the group at 16, 32, or 64
// elements, so 16 consecutive elements starting at a multiple of 16 always lie
// in ONE group: a lane needs one `(scale, minimum)` pair for its whole run,
// and the eight tiles' `8 * groups_per_tile <= 32` groups are resolved by one
// warp-wide step instead of eight serialized ones.
//
// # Alignment, proven from the layout rather than assumed
//
// `payload` is a device allocation base, so 256-byte aligned. On top of that,
// with offsets as `quant/tcf/planes.rs` computes them:
//   - Section 14.3, 8-bit: the run starts at `tile0 * 64`, 64-byte aligned for
//     any `tile0`, and lane `l` reads at `+ 16 * l`. A `uint4` needs 16.
//   - Section 14.1, 4-bit, and Section 14.2's low sub-plane: the run starts at
//     `tile0 * 32`, and lane `l` reads at `+ 8 * l`. A `uint2` needs 8.
//   - Section 14.2's high sub-plane: `code_high_off` is `tiles * 32`, the run
//     adds `tile0 * 16`, and lane `l` reads at `+ 4 * l`. A `uint` needs 4.
// No case depends on `tile0` being run-aligned, which it is not when a row's
// tile count is not a multiple of eight.
struct TcfCodeRun {
    unsigned int w[4];
};

static __device__ __forceinline__ TcfCodeRun tcf_code_run(
    const unsigned char* __restrict__ payload,
    TcfLayout l,
    unsigned int tile0,
    unsigned int lane
) {
    TcfCodeRun run;
    if (l.bits == 8u) {
        const uint4 v =
            ((const uint4*)(payload + (size_t)tile0 * (size_t)TCF_TILE))[lane];
        run.w[0] = v.x;
        run.w[1] = v.y;
        run.w[2] = v.z;
        run.w[3] = v.w;
        return run;
    }
    const uint2 v =
        ((const uint2*)(payload + (size_t)tile0 * (size_t)(TCF_TILE / 2u)))[lane];
    run.w[0] = v.x;
    run.w[1] = v.y;
    run.w[2] = 0u;
    run.w[3] = 0u;
    if (l.bits == 6u) {
        run.w[2] = ((const unsigned int*)(payload + l.code_high_off
                                         + (size_t)tile0 * (size_t)(TCF_TILE / 4u)))[lane];
    }
    return run;
}

// Element `i` of `tcf_code_run`'s 16, sign-resolved. `i` indexes the lane's own
// run elements, so run-local element `TCF_RUN_PER_LANE * lane + i`.
//
// The bit positions restate Section 14 in word terms and must agree with
// `tcf_code` element for element; `tests/backend_parity/quant_tcf.rs` is what
// holds them together. Little-endian: an 8-bit code sits at bit `8 * i` of
// word `i / 4`, a 4-bit code at bit `4 * i` of word `i / 8`, and a 6-bit
// code's high two bits at bit `2 * i` of the single high word.
static __device__ __forceinline__ int tcf_run_code(
    TcfCodeRun run,
    TcfLayout l,
    unsigned int i
) {
    unsigned int field;
    if (l.bits == 8u) {
        field = (run.w[i >> 2] >> ((i & 3u) * 8u)) & 0xffu;
    } else {
        field = (run.w[i >> 3] >> ((i & 7u) * 4u)) & 0x0fu;
        if (l.bits == 6u) {
            field |= ((run.w[2] >> (i * 2u)) & 0x03u) << 4u;
        }
    }
    return tcf_sign_resolve(field, l.bits, l.symmetric);
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
