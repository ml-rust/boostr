// GGUF lowbit block decoders — the ONE place these layouts are written
// for CUDA.
//
// Every CUDA kernel that decodes a Q1_0, Q2_0, PQ2_0 or PTQ1_0 block includes
// this instead of restating the layout. `src/quant/cpu/kernels/dequant_lowbit.rs`
// is the CPU mirror; keep the two in step.
//
// Q1_0, Q2_0 and PQ2_0 keep `d` at the START and order elements byte-major,
// low bits first. PTQ1_0 is TQ1_0's base-3 trit packing at group 128 with `d`
// at the END; `gguf_base3_trit` in decode.cuh is shared with TQ1_0.
//
//   Q1_0   (18B): d[0..2],  qs[2..18]              1 bit  each, 128 elements
//   Q2_0   (18B): d[0..2],  qs[2..18]              2 bits each,  64 elements
//   PQ2_0  (34B): d[0..2],  qs[2..34]              2 bits each, 128 elements
//   PTQ1_0 (28B): qs[0..24], qh[24..26], d[26..28]
//
// PTQ1_0 runs, in order:
//   [  0,  80)  qs[0..16]   x 5 levels, 16 per level
//   [ 80, 120)  qs[16..24]  x 5 levels,  8 per level
//   [120, 128)  qh[0..2]    x 4 levels,  2 per level

#pragma once

#include <cuda_fp16.h>

#include "decode.cuh"

#define GGUF_LOWBIT_QS_OFFSET   2
#define GGUF_PTQ1_0_D_OFFSET   26

static __device__ __forceinline__ float lowbit_load_d(const unsigned char* p) {
    __half tmp;
    memcpy(&tmp, p, sizeof(__half));
    return __half2float(tmp);
}

// Returns {-1, 1} of element `elem` of a Q1_0 `qs` run: bit set is +1.
static __device__ __forceinline__ int gguf_sign_bit(
    const unsigned char* qs, int elem
) {
    return ((qs[elem >> 3] >> (elem & 7)) & 1) ? 1 : -1;
}

// Returns {-1, 0, 1, 2} of element `elem` of a Q2_0 or PQ2_0 `qs` run:
// code `q` maps to `q - 1` (00=-1, 01=0, 10=+1, 11=+2).
static __device__ __forceinline__ int gguf_code2_minus_1(
    const unsigned char* qs, int elem
) {
    return (int)((qs[elem >> 2] >> ((elem & 3) * 2)) & 0x03) - 1;
}

// Returns the ternary value {-1, 0, 1} of element `elem` (0..128) of a PTQ1_0
// block. `block` points at the start of the 28-byte block.
static __device__ __forceinline__ int gguf_ptq1_0_trit(
    const unsigned char* block, int elem
) {
    unsigned char byte;
    int level;
    if (elem < 80) {
        level = elem >> 4;
        byte = block[elem & 15];
    } else if (elem < 120) {
        const int r = elem - 80;
        level = r >> 3;
        byte = block[16 + (r & 7)];
    } else {
        const int r = elem - 120;
        level = r >> 1;
        byte = block[24 + (r & 1)];
    }
    return gguf_base3_trit(byte, level);
}

// ── int8x4 expansions ───────────────────────────────────────────────────
//
// Ports of `vec_dot_q1_0_q8_1` and `vec_dot_q2_0_q8_1` from `ggml-cuda`'s
// `vecdotq.cuh` — same LUT words, same `__byte_perm` selectors.
// Shared by the token-batched dp4a GEMV (`gemv/lowbit_ntok.cuh`) and the
// feature-major MMQ staging (`mmq/lowbit_tiles.cuh`), which both want a
// code run as signed int8 lanes.
//
// `__byte_perm` reads three bits per selector nibble; bit 3 of each nibble
// is ignored, which is what lets a raw code word act as the selector.

// Eight 2-bit codes -> two int8x4 words of {-1, 0, 1, 2}. `lo` and `hi` are
// one code byte each: the four codes of `lo` land in `*v_lo`, those of `hi`
// in `*v_hi`, low code first.
//
// `qe` takes the even codes of the 16-bit word (bits 4i..4i+1 as selector
// nibble i), `qo` the odd ones (`>> 2`). LUT `0x020100FF` is the byte table
// {-1, 0, 1, 2} indexed by the code. The two final permutes re-interleave
// even and odd: `0x5140` gathers codes 0..3 (the `lo` byte), `0x7362`
// codes 4..7 (the `hi` byte).
static __device__ __forceinline__ void lowbit_expand_code2x8(
    int lo, int hi, int* v_lo, int* v_hi
) {
    const int q = lo | (hi << 8);
    const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
    const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
    *v_lo = __byte_perm(qe, qo, 0x5140);
    *v_hi = __byte_perm(qe, qo, 0x7362);
}

// Eight sign bits -> two int8x4 words of {-1, +1}. Bits 0..3 of `bits8`
// land in `*v_lo`, bits 4..7 in `*v_hi`, low bit first.
//
// First permute pair spreads bit pairs into nibble indices: LUT
// `0x11100100` maps the 2-bit selector `(b1 b0)` to the byte `b1:b0` as two
// nibbles, so `n0` holds bits (0,1) in byte 0 and (4,5) in byte 1, `n1`
// bits (2,3) and (6,7). Second pair turns each nibble into a signed byte via
// LUT `0x01FF` ({-1, +1}). Final pair unshuffles: `0x5410` = bits 0..3,
// `0x7632` = bits 4..7.
static __device__ __forceinline__ void lowbit_expand_sign8(
    int bits8, int* v_lo, int* v_hi
) {
    const int n0 = __byte_perm(0x11100100, 0x11100100, bits8 >> 0);
    const int n1 = __byte_perm(0x11100100, 0x11100100, bits8 >> 2);
    const int s0 = __byte_perm(0x01FF, 0x01FF, n0);
    const int s1 = __byte_perm(0x01FF, 0x01FF, n1);
    *v_lo = __byte_perm(s0, s1, 0x5410);
    *v_hi = __byte_perm(s0, s1, 0x7632);
}

// ── PTQ1_0 int8x4 expansions ────────────────────────────────────────────
//
// Used by the feature-major MMQ staging (`mmq/lowbit_tiles.cuh`), which gives
// one lane 8 consecutive elements of a block. In the two `qs` runs those 8
// elements read 8 CONSECUTIVE bytes at ONE trit level (the run's per-level
// width, 16 or 8, is a multiple of 8 and `e0 % 8 == 0`); in the `qh` tail
// they read the two `qh` bytes at four levels. `gguf_base3_trit` is the
// per-element decode; the staging cannot afford it 8x per row, so the same
// arithmetic — wrapping 8-bit multiply by `pow3[level]`, then
// `(q * 3) >> 8`, minus 1 — runs here on four bytes at once.

// `pow3[level]` for `level` in 0..5 as a select chain. `gguf_base3_trit`
// keeps `pow3` in a local array, which a runtime `level` turns into local
// memory; a lane whose level is fixed evaluates this once.
static __device__ __forceinline__ unsigned int ptq1_0_pow3(int level) {
    return level == 0 ? 1u : level == 1 ? 3u : level == 2 ? 9u : level == 3 ? 27u : 81u;
}

// SWAR core: two product words -> one int8x4 word of trits {-1, 0, 1}.
//
// `prod_even` holds the products of source bytes 0 and 2 in its two 16-bit
// lanes (byte 0 in the low lane), `prod_odd` those of bytes 1 and 3; each
// product is `byte * pow3` for that byte's level. Per lane:
//   q = prod & 0xFF                  the wrapping 8-bit product
//   t = (q * 3) >> 8                 the trit code, {0, 1, 2}
// Lane invariants, which keep the lanes from carrying into each other:
//   byte * pow3 <= 255 * 81 = 20655 < 2^16, so `& 0x00FF00FF` after the
//   multiply is the 8-bit wrap of each lane's own product;
//   q * 3 <= 765 < 2^16, so the shift moves only that lane's bits 8..9
//   down, and `& 0x00FF00FF` drops the other lane's low bits that the shift
//   pulled into bits 8..15.
// The even codes sit in bytes 0 and 2, the odd ones move up to bytes 1 and
// 3, so element `r` lands in byte `r`; `__vsub4` then subtracts 1 per byte
// (0 -> 0xFF = -1). Element 0 in the low byte, the order
// `lowbit_expand_code2x8` produces and Q8_0's staged row stores.
static __device__ __forceinline__ int ptq1_0_codes_to_int8(
    unsigned int prod_even, unsigned int prod_odd
) {
    const unsigned int q_e = prod_even & 0x00FF00FFu;
    const unsigned int q_o = prod_odd & 0x00FF00FFu;
    const unsigned int t_e = ((q_e * 3u) >> 8) & 0x00FF00FFu;
    const unsigned int t_o = ((q_o * 3u) >> 8) & 0x00FF00FFu;
    return (int)__vsub4(t_e | (t_o << 8), 0x01010101u);
}

// Four source bytes `w` (byte 0 low) -> one int8x4 word of trits, through
// [`ptq1_0_codes_to_int8`]. `mask` and `mul` select the two lane shapes the
// staging needs, and are loop-invariant per lane there:
//
//   Run lane (8 consecutive bytes at one level `L`, two calls, `w` = bytes
//   0..3 then 4..7): `mask = 0x00FF00FF`, `mul = pow3[L]`. Each 16-bit lane
//   holds one byte and the scalar multiply scales both lanes by the same
//   power, so byte `r` of the result is `gguf_base3_trit(byte r, L)`.
//
//   Tail lane (`qh[0]` and `qh[1]` at levels `L0`, `L0 + 1`; `w` = the word
//   at block offset 24, bytes `qh0, qh1, d_lo, d_hi`): `mask = 0x000000FF`,
//   `mul = pow3[L0] | (pow3[L0 + 1] << 16)`. `w & mask` is the scalar `qh0`,
//   and `qh0 * mul = qh0 * pow3[L0] + (qh0 * pow3[L0 + 1]) << 16` puts the
//   two levels of `qh0` in the even lanes with no cross term because the
//   multiplicand is a scalar; `(w >> 8) & mask` does the same for `qh1` in
//   the odd lanes. The result is then bytes `(qh0, L0), (qh1, L0),
//   (qh0, L0 + 1), (qh1, L0 + 1)`, which is elements `120 + 2 * L0 ..` of
//   the block as `gguf_ptq1_0_trit` orders them: `mul = 0x00030001` gives
//   elements 120..124 and `mul = 0x001B0009` (9, 27) elements 124..128.
//   Both lane products stay under 2^16 (the `ptq1_0_codes_to_int8`
//   invariant) since each is one byte times one power of three.
static __device__ __forceinline__ int ptq1_0_expand4(
    unsigned int w, unsigned int mask, unsigned int mul
) {
    return ptq1_0_codes_to_int8((w & mask) * mul, ((w >> 8) & mask) * mul);
}

// ── Whole-block decoders ────────────────────────────────────────────────

static __device__ __forceinline__ void q1_0_dequant_block(
    const unsigned char* block, float* out
) {
    const float d = lowbit_load_d(block);
    const unsigned char* qs = block + GGUF_LOWBIT_QS_OFFSET;
    for (int i = 0; i < 128; i++) out[i] = d * (float)gguf_sign_bit(qs, i);
}

static __device__ __forceinline__ void q2_0_dequant_block(
    const unsigned char* block, float* out
) {
    const float d = lowbit_load_d(block);
    const unsigned char* qs = block + GGUF_LOWBIT_QS_OFFSET;
    for (int i = 0; i < 64; i++) out[i] = d * (float)gguf_code2_minus_1(qs, i);
}

static __device__ __forceinline__ void pq2_0_dequant_block(
    const unsigned char* block, float* out
) {
    const float d = lowbit_load_d(block);
    const unsigned char* qs = block + GGUF_LOWBIT_QS_OFFSET;
    for (int i = 0; i < 128; i++) out[i] = d * (float)gguf_code2_minus_1(qs, i);
}

static __device__ __forceinline__ void ptq1_0_dequant_block(
    const unsigned char* block, float* out
) {
    const float d = lowbit_load_d(block + GGUF_PTQ1_0_D_OFFSET);
    for (int i = 0; i < 128; i++) out[i] = d * (float)gguf_ptq1_0_trit(block, i);
}
