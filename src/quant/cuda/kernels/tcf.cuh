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

// One payload byte through the read-only data cache.
//
// Every plane this file reads is global memory the kernel never writes, which
// is exactly `__ldg`'s contract. The scale planes are the reason it is worth
// naming: a group parameter is one or two bytes out of a whole-tensor plane
// that sits megabytes from the code plane, so each such read costs a full
// memory burst that delivers almost nothing else the warp wants. Routing them
// through the read-only cache lets the burst stay resident for the neighbouring
// groups the warp resolves next, instead of evicting code-plane lines.
//
// A byte load has no alignment requirement, which is why the 16-bit readers
// below assemble two of these rather than issuing one 16-bit load: plane
// offsets are computed by `quant/tcf/planes.rs` and are not guaranteed even.
// `__ldg` is a cache hint and nothing else — the bytes returned are the bytes
// in memory, so every value built from them is bit-identical to a plain load.
static __device__ __forceinline__ unsigned int tcf_ldg_u8(const unsigned char* p) {
    return (unsigned int)__ldg(p);
}

// A little-endian binary16 read as f32. Every binary16 is exactly
// representable in f32, so this agrees with `tcf_core::binary16::bits_to_f32`
// bit for bit.
//
// Section 13.0 / Section 13.0.1 scales and minima only. A two-level SUPER
// value is a bfloat16 and belongs to `tcf_read_bfloat16`; the two formats
// never share a reader.
static __device__ __forceinline__ float tcf_read_binary16(const unsigned char* p) {
    unsigned short bits =
        (unsigned short)(tcf_ldg_u8(p) | (tcf_ldg_u8(p + 1) << 8u));
    __half h;
    memcpy(&h, &bits, sizeof(__half));
    return __half2float(h);
}

// A little-endian bfloat16 read as f32, exactly. Section 13.3, Section 13.4.
//
// bfloat16's sign bit and 8-bit exponent field ARE f32's, and its 7 fraction
// bits are f32's leading 7, so widening is a 16-bit left shift into the f32 bit
// pattern and nothing else. It is exact for every one of the 65536 patterns —
// subnormals, infinities and NaN payloads included — so it agrees with
// `tcf_core::bfloat16::bits_to_f32` bit for bit with no normalize step and no
// float operation that `--use_fast_math` could reach.
static __device__ __forceinline__ float tcf_read_bfloat16(const unsigned char* p) {
    unsigned int bits = tcf_ldg_u8(p) | (tcf_ldg_u8(p + 1) << 8u);
    return __uint_as_float(bits << 16u);
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
    unsigned int value = tcf_ldg_u8(plane + byte_index) >> offset;
    if (offset > 2u) {
        value |= tcf_ldg_u8(plane + byte_index + 1) << (8u - offset);
    }
    return value & 0x3fu;
}

// The effective (scale, minimum) of group `g` of tile `tile`.
//
// Section 13.0 / Section 13.0.1: a flat layout stores both outright as
// binary16. Section 13.3: a two-level u8 layout stores a u8 sub-scale under a
// per-super-block super-scale. Section 13.4: a two-level 6-bit layout stores a
// 6-bit unsigned sub-scale and a 6-bit signed sub-minimum, both bit-packed per
// super-block, under a super-scale and a super-minimum.
//
// A two-level resolution is ONE multiply, matching `QuantLayout::group_scale`
// and `QuantLayout::group_min` exactly. Section 13.3 and Section 13.4 store
// every super value PRE-DIVIDED by its form's sub-level count, as a bfloat16,
// so nothing is divided at decode. The product is exact as well: a bfloat16
// carries an 8-bit significand and the widest sub-level field is 8 bits, so it
// needs at most 16 of f32's 24 significand bits. `__fmul_rn` still names the
// rounding rather than leaving the multiply to `--use_fast_math` contraction.
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
        float super = tcf_read_bfloat16(payload + l.super_off + (size_t)block * 2u);
        unsigned int sub = tcf_ldg_u8(payload + (size_t)l.scale_off + global);
        *out_scale = __fmul_rn(super, (float)sub);
        *out_min = 0.0f;
        return;
    }
    if (l.scale_form == TCF_SCALE_TWO_LEVEL_U6M6) {
        float super = tcf_read_bfloat16(payload + l.super_off + (size_t)block * 2u);
        unsigned int sub = tcf_read_packed6(
            payload + l.scale_off, block, slot, l.sub_block_bytes);
        *out_scale = __fmul_rn(super, (float)sub);

        float super_min = tcf_read_bfloat16(payload + l.super_min_off + (size_t)block * 2u);
        unsigned int field = tcf_read_packed6(
            payload + l.min_off, block, slot, l.sub_block_bytes);
        // A 6-bit two's-complement field: 0..=31 is itself, 32..=63 is
        // `field - 64`. The reserved -32 never reaches here in a payload
        // `tcf-core` accepted.
        int level = (int)field;
        if (level > 31) {
            level -= 64;
        }
        *out_min = __fmul_rn(super_min, (float)level);
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
// `bits` over the full two's-complement range; no code plane reserves a
// value (Section 13.2), so device code sign-extends unconditionally.
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
    // `>=`, not `>`: two's complement's most-negative pattern (`field ==
    // reserved`) is a legal code, not a gap. `>` under-counted it, wrapping
    // it back to a positive value one past qmax instead of `-reserved`.
    return (value >= reserved) ? value - 2 * reserved : value;
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

// Expands four ADJACENT 4-bit TCF codes into one int8 lane word.
//
// Section 14.1 packs a 4-bit tile as `byte = u[2e] | (u[2e+1] << 4)`, so four
// consecutive elements fill bits 0..15 of `h` in element order and the four
// int8 lanes come out in that same order. This is NOT ggml's 4-bit map, where
// one byte's low and high nibbles belong to sub-blocks 32 elements apart —
// compare `MmqfQ4K::stage` in `quant_mmq_mma.cu`, which splits one word into
// two staged words 8 apart. `tcf_code`'s `l.bits == 4u` branch above is the
// layout this mirrors.
//
// The codes are UNSIGNED levels for an asymmetric encoding, so no sign
// resolution applies and 0..15 already sits inside the signed int8 range dp4a
// and `mma` read. The minimum term carries the asymmetry, as it does for Q4_K.
//
// Two callers unpack this plane — `MmqfTcfQ4AS32DT64::stage` and the dp4a GEMV
// body in `gemv/tcf_ntok.cuh` — so the map is written once, here, with the
// rest of Section 14's read direction.
static __device__ __forceinline__ int tcf_expand_nibble_quad(unsigned int h) {
    return (int)((h & 0x000Fu) | ((h & 0x00F0u) << 4) | ((h & 0x0F00u) << 8)
                 | ((h & 0xF000u) << 12));
}

// Execution tiles one warp decodes per CODE read of the GEMV inner loop, and
// the elements one lane owns of that read. 8 * 64 / 32 == 16.
//
// This is the CODE-plane unit and is fixed by the wide-load geometry, not a
// tuning knob: `tcf_code_run` puts a lane's sixteen elements in one `uint4`
// (8-bit) or one `uint2` plus one `uint` (4- and 6-bit), and 32 lanes times
// sixteen elements is exactly eight 64-element tiles. The GEMM's staging chunk
// (`TCF_GEMM_CHUNK`) is the same sixteen. Changing either would change both.
// The SCALE-plane unit is separate and is `TCF_GEMV_RUN_TILES` below.
#define TCF_RUN_TILES 8u
#define TCF_RUN (TCF_RUN_TILES * TCF_TILE)
#define TCF_RUN_PER_LANE 16u

// Lanes in a warp. The GEMV run geometry below is warp-wide, so the count
// belongs beside it rather than in one kernel file.
#define TCF_GEMV_LANES 32u

// Largest group count one execution tile can carry: v1's narrowest group is 16
// elements over a 64-element tile (`tcf_core::MAX_GROUPS_PER_TILE`). Sizes the
// dequant kernel's shared parameter arrays and the GEMV's per-lane ones.
#define TCF_MAX_GROUPS 4u

// ── The GEMV's scale-resolve run width ──────────────────────────────────
//
// Execution tiles the f32 GEMV resolves group parameters for in ONE warp-wide
// step. It is a whole number of `TCF_RUN_TILES` code reads: the code plane is
// still streamed eight tiles at a time, and this only says how many of those
// reads share one resolution.
//
// # Why it is separate from the code width
//
// Resolving one group touches up to FOUR whole-tensor planes — sub-scale,
// sub-minimum, super-scale, super-minimum for Section 13.4 — that sit megabytes
// apart. Each touch delivers one or two useful bytes and costs a full memory
// burst, so the cost tracks the number of RESOLUTIONS, not the number of bytes
// the row actually needs. A block format that interleaves its scale into the
// code stream pays none of it. Widening the resolve is what amortizes that
// fixed four-plane cost over more elements; the code-plane traffic is identical
// either way.
//
// # A measured constant
//
// This is the ONE value to edit to re-tune. It must be a multiple of
// `TCF_RUN_TILES`; 8 (one code read, the un-widened behaviour), 16 and 32 are
// the widths to compare. Re-measure with:
//
//   cargo bench --features cuda --bench quant_throughput -- --backend cuda --filter Q4
//
// On a CUDA row, read the `ns*` column, NOT the `tcf/gguf` ratio column: that
// ratio is built from retired HOST instructions, which on a CUDA row count
// kernel launch work rather than kernel work.
//
// Wider is not automatically better: the resolved parameters live in per-lane
// registers whose count is `TCF_GEMV_GROUP_SLOTS` below, and register pressure
// costs occupancy. Measure occupancy alongside runtime.
#define TCF_GEMV_RUN_TILES 32u

// Code reads inside one scale-resolve run, and its element span.
#define TCF_GEMV_CODE_RUNS (TCF_GEMV_RUN_TILES / TCF_RUN_TILES)
#define TCF_GEMV_RUN (TCF_GEMV_RUN_TILES * TCF_TILE)

// Per-lane resolved `(scale, minimum)` slots one scale-resolve run needs.
//
// A run holds `TCF_GEMV_RUN_TILES * groups_per_tile` groups, and the warp
// resolves them 32 at a time — lane `i` taking group `32 * t + i` of slot `t`.
// The array is sized for the widest case, `groups_per_tile == TCF_MAX_GROUPS`;
// a wider group leaves the upper slots unused, and the loops that fill them are
// guarded on the run's real group count.
#define TCF_GEMV_GROUP_SLOTS \
    ((TCF_GEMV_RUN_TILES * TCF_MAX_GROUPS + TCF_GEMV_LANES - 1u) / TCF_GEMV_LANES)

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

// ── The GEMV's widened scale resolution ─────────────────────────────────
//
// One warp's resolved `(scale, minimum)` pairs for a whole scale-resolve run,
// held in registers: slot `t` of lane `i` is the run's group `32 * t + i`.
struct TcfGroupRun {
    float scale[TCF_GEMV_GROUP_SLOTS];
    float min_value[TCF_GEMV_GROUP_SLOTS];
};

// Resolves `groups` consecutive groups starting at GLOBAL group index `group0`,
// 32 per slot, one `tcf_group_values` call per lane per slot.
//
// The group index is global — flattened over the whole tensor, the same index
// the planes are keyed on — so a caller that knows a row's group range needs no
// super-block alignment and no per-row special case. `groups` is the run's real
// group count, which is below `32 * TCF_GEMV_GROUP_SLOTS` whenever the encoding
// uses a group wider than 16; the slots past it stay zero and are never read.
//
// The values are `tcf_group_values`' own, unmodified: this only changes HOW
// OFTEN that function runs, never what it returns.
static __device__ __forceinline__ TcfGroupRun tcf_resolve_group_run(
    const unsigned char* __restrict__ payload,
    TcfLayout l,
    unsigned int group0,
    unsigned int groups,
    unsigned int lane
) {
    TcfGroupRun r;
#pragma unroll
    for (unsigned int t = 0; t < TCF_GEMV_GROUP_SLOTS; ++t) {
        const unsigned int index = t * TCF_GEMV_LANES + lane;
        float scale = 0.0f;
        float min_value = 0.0f;
        if (index < groups) {
            const unsigned int gg = group0 + index;
            tcf_group_values(payload, l, gg / l.groups_per_tile,
                             gg % l.groups_per_tile, &scale, &min_value);
        }
        r.scale[t] = scale;
        r.min_value[t] = min_value;
    }
    return r;
}

// The run's group `32 * slot + src`, broadcast to every lane.
//
// # `slot` MUST be warp-uniform
//
// `__shfl_sync` reads the named variable ON THE SOURCE LANE, so the slot
// selection has to have already picked the same array entry there. Both callers
// satisfy this the same way, and it is a property of the geometry rather than a
// convention: the 32 lanes of one code read span exactly `TCF_RUN / group`
// consecutive groups, that count divides 32, and a run's group ranges are
// aligned to it — so the whole warp's group indices for one code read fall
// inside ONE aligned block of 32, and `index >> 5` is the same on every lane.
// `src`, `index & 31`, is the part that differs per lane.
//
// The selection is a compare chain over a compile-time slot count rather than
// an array subscript: a runtime subscript on a per-thread array spills it to
// local memory, which is the one thing a register-held run must not do.
static __device__ __forceinline__ void tcf_run_broadcast(
    TcfGroupRun r,
    unsigned int slot,
    unsigned int src,
    float* out_scale,
    float* out_min
) {
    float scale = 0.0f;
    float min_value = 0.0f;
#pragma unroll
    for (unsigned int t = 0; t < TCF_GEMV_GROUP_SLOTS; ++t) {
        if (t == slot) {
            scale = r.scale[t];
            min_value = r.min_value[t];
        }
    }
    *out_scale = __shfl_sync(0xFFFFFFFFu, scale, src);
    *out_min = __shfl_sync(0xFFFFFFFFu, min_value, src);
}
