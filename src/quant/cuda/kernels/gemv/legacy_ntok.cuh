// Token-batched dp4a GEMV for the legacy 32-element quant formats
//
// Q4_0, Q5_0, Q4_1 and Q5_1 share one block geometry — 32 elements, an f16
// scale at byte 0, and 16 nibble-packed quant bytes at the end — so they also
// share one batched MWR body. Only the per-block decode differs, and that is
// what the four policy structs below carry. The alternative is four copies of
// the same ~90-line kernel differing in a dozen lines of bit twiddling.
//
// These four have no single-token dp4a GEMV: their `_n2` tile exists because
// batching pays at m = 2, while m = 1 is served by the F32-activation kernel
// beside them. `dispatch_gemv` reflects that by entering the dp4a branch for
// these formats only from m = 2 up.
//
// Weight decode is lifted from the MMQ staging structs `MmqfQ40`, `MmqfQ50`,
// `MmqfQ41` and `MmqfQ51` in `../quant_mmq_mma.cu`, which are in turn
// cross-checked element by element against `dequant_q4_0`, `dequant_q5_0`,
// `dequant_q4_1` and `dequant_q5_1` in
// `src/quant/cpu/kernels/dequant_simple.rs`. Nibble map, shared by all four:
// within one 32-element block, element `j` (0..15) is the LOW nibble of
// `qs[j]` and element `j + 16` is the HIGH nibble of the same byte. So the
// int at `qs + 4*w` carries elements `4w..4w+3` in its low nibbles and
// `4w+16..4w+19` in its high nibbles — two dp4a operands from one load.
//
// ALIGNMENT. Q4_0's block is 18 bytes and Q5_0's is 22, so a row base
// (`bpr * BLOCK_BYTES`) and every block base inside it are only 2-byte
// aligned. Every 4-byte read in those two decoders therefore goes through
// `load_int_ua`; a plain `int` load raises CUDA_ERROR_MISALIGNED_ADDRESS,
// which poisons the context for every later launch on it. Q4_1's 20 and
// Q5_1's 24 are multiples of 4, so those two read `int` directly, exactly as
// their MMQ counterparts do.

#pragma once

#include "common.cuh"

// ── Per-format decode policies ──────────────────────────────────────────
//
// Contract. `decode` reads block `blk` and 4-element source word `w` (0..3)
// and produces:
//   *d     — the block scale
//   *m     — the block minimum, 0 for a format without one
//   *v_lo  — elements 4w..4w+3 as int8x4, ready for dp4a
//   *v_hi  — elements 4w+16..4w+19 as int8x4, ready for dp4a
// `HAS_MIN` says whether the minimum term is part of the format's value, and
// gates the block-sum correction in the kernel body at compile time.

// Q4_0: 18 bytes / 32 elements. `d` f16@0, 16 nibble bytes@2.
// Value `d * (q - 8)`, so the 8 bias is folded per byte here and the dot
// product below needs no correction term.
struct LegacyQ40 {
    static constexpr int BLOCK_BYTES = 18;
    static constexpr bool HAS_MIN = false;

    static __device__ __forceinline__ void decode(
        const unsigned char* __restrict__ blk, int w,
        float* d, float* m, int* v_lo, int* v_hi
    ) {
        *d = __half2float(*(const __half*)blk);
        *m = 0.0f;
        // 2-byte aligned block base: unaligned load, see ALIGNMENT above.
        const int q = load_int_ua(blk + 2 + w * 4);
        // Inputs are 0..15, so the subtract never saturates. The arithmetic
        // shift's sign bits are masked off before the subtract.
        *v_lo = __vsubss4(q & 0x0F0F0F0F, 0x08080808);
        *v_hi = __vsubss4((q >> 4) & 0x0F0F0F0F, 0x08080808);
    }
};

// Q5_0: 22 bytes / 32 elements. `d` f16@0, `qh` u32@2, 16 nibble bytes@6.
// Value `d * (q - 16)` with `q` the unsigned 5-bit assembly, so the 16 bias
// is folded per byte here.
//
// The fifth bit of element `j` is bit `j` of the block's single 32-bit `qh`,
// so this lane needs bits `4w..4w+3` for its low word and `4w+16..4w+19` for
// its high one. Pre-shifting by `4*w` leaves those in bits 0..3 and 16..19,
// and the four masked shifts move bit `t` of that into bit 4 of byte `t`.
// That is llama.cpp's form, kept verbatim: the source bits are packed
// contiguously inside one nibble rather than spread one per byte, so Q5_K's
// single `(qh >> sh) & 0x01010101` trick does not apply here.
struct LegacyQ50 {
    static constexpr int BLOCK_BYTES = 22;
    static constexpr bool HAS_MIN = false;

    static __device__ __forceinline__ void decode(
        const unsigned char* __restrict__ blk, int w,
        float* d, float* m, int* v_lo, int* v_hi
    ) {
        *d = __half2float(*(const __half*)blk);
        *m = 0.0f;
        // Both 2-byte aligned: unaligned loads, see ALIGNMENT above.
        const int qh = (int)((unsigned int)load_int_ua(blk + 2) >> (4 * w));
        const int q = load_int_ua(blk + 6 + w * 4);

        int lo = q & 0x0F0F0F0F;
        lo |= (qh << 4) & 0x00000010;   // element 4w+0 -> byte 0 bit 4
        lo |= (qh << 11) & 0x00001000;  // element 4w+1 -> byte 1 bit 4
        lo |= (qh << 18) & 0x00100000;  // element 4w+2 -> byte 2 bit 4
        lo |= (qh << 25) & 0x10000000;  // element 4w+3 -> byte 3 bit 4
        int hi = (q >> 4) & 0x0F0F0F0F;
        hi |= (qh >> 12) & 0x00000010;  // element 4w+16 -> byte 0 bit 4
        hi |= (qh >> 5) & 0x00001000;   // element 4w+17 -> byte 1 bit 4
        hi |= (qh << 2) & 0x00100000;   // element 4w+18 -> byte 2 bit 4
        hi |= (qh << 9) & 0x10000000;   // element 4w+19 -> byte 3 bit 4
        // Inputs are 0..31, so the subtract never saturates.
        *v_lo = __vsubss4(lo, 0x10101010);
        *v_hi = __vsubss4(hi, 0x10101010);
    }
};

// Q4_1: 20 bytes / 32 elements. `d` f16@0, `m` f16@2, 16 nibble bytes@4.
// Value `d * q + m` with UNSIGNED `q` — the minimum is ADDITIVE, unlike
// Q4_K's `-dmin * m`. No bias is folded: unsigned 0..15 already sits inside
// the int8 range dp4a takes, and the minimum rides the block-sum term in the
// kernel body.
struct LegacyQ41 {
    static constexpr int BLOCK_BYTES = 20;
    static constexpr bool HAS_MIN = true;

    static __device__ __forceinline__ void decode(
        const unsigned char* __restrict__ blk, int w,
        float* d, float* m, int* v_lo, int* v_hi
    ) {
        // `d` and `m` are adjacent f16 at byte 0 of a 4-aligned block.
        const half2 dm = *(const half2*)blk;
        *d = __low2float(dm);
        *m = __high2float(dm);
        const int q = *(const int*)(blk + 4 + w * 4);
        *v_lo = q & 0x0F0F0F0F;
        *v_hi = (q >> 4) & 0x0F0F0F0F;
    }
};

// Q5_1: 24 bytes / 32 elements. `d` f16@0, `m` f16@2, `qh` u32@4, 16 nibble
// bytes@8. Value `d * q + m` — Q4_1's additive minimum with Q5_0's fifth-bit
// assembly, and no bias for the same reason as Q4_1.
struct LegacyQ51 {
    static constexpr int BLOCK_BYTES = 24;
    static constexpr bool HAS_MIN = true;

    static __device__ __forceinline__ void decode(
        const unsigned char* __restrict__ blk, int w,
        float* d, float* m, int* v_lo, int* v_hi
    ) {
        const half2 dm = *(const half2*)blk;
        *d = __low2float(dm);
        *m = __high2float(dm);
        const int qh = (int)(*(const unsigned int*)(blk + 4) >> (4 * w));
        const int q = *(const int*)(blk + 8 + w * 4);

        int lo = q & 0x0F0F0F0F;
        lo |= (qh << 4) & 0x00000010;   // element 4w+0 -> byte 0 bit 4
        lo |= (qh << 11) & 0x00001000;  // element 4w+1 -> byte 1 bit 4
        lo |= (qh << 18) & 0x00100000;  // element 4w+2 -> byte 2 bit 4
        lo |= (qh << 25) & 0x10000000;  // element 4w+3 -> byte 3 bit 4
        int hi = (q >> 4) & 0x0F0F0F0F;
        hi |= (qh >> 12) & 0x00000010;  // element 4w+16 -> byte 0 bit 4
        hi |= (qh >> 5) & 0x00001000;   // element 4w+17 -> byte 1 bit 4
        hi |= (qh << 2) & 0x00100000;   // element 4w+18 -> byte 2 bit 4
        hi |= (qh << 9) & 0x10000000;   // element 4w+19 -> byte 3 bit 4
        *v_lo = lo;
        *v_hi = hi;
    }
};

// ── Shared token-batched MWR body ───────────────────────────────────────
//
// Grid: (N, ceil(M / NTOK), 1) — one output column per block, NTOK token
// columns per block. Block: `mwr_nwarps_ntok(NTOK) * WARP_SIZE` threads; the
// launch side must size the block from the same function, because the
// reduction's shared array and `__launch_bounds__` both read it.
//
// Lane map. A 32-element block holds 4 source words, so a warp's 32 lanes
// cover 8 whole blocks per step: lane maps to (block `lane / 4` inside an
// 8-block group, source word `lane % 4`). Four consecutive lanes read 16
// contiguous bytes of the same block, and the 8-block group is contiguous in
// the row, so the group's loads coalesce.
//
// The weight load and its decode sit OUTSIDE the token loop — that is the
// whole point of the tile. Only the activation load and the dp4a repeat per
// token, so a weight block is read and unpacked once for all NTOK columns
// instead of once per column.
//
// Minimum term (Q4_1, Q5_1). The value is `d * q + m` with unsigned `q`, so
// the block's contribution is `d * sum(q_e * a_e) + m * sum(a_e)`. The second
// sum is rank-1 over the block: it depends on the activation alone. It is
// formed here as an EXACT integer with `dp4a(0x01010101, a, ...)` and scaled
// by the activation's own `d`, which is what `mmqf_vec_dot_dm` does with the
// int16 sum the MMQ activation record carries — the two paths therefore agree
// bit-pattern for bit-pattern on that term. The per-token Q8_1 record's `s`
// field is NOT used: its producer stores `d * sum(x)` over the ORIGINAL
// floats, not `d * sum(q)` over the quants, so it would not match MMQ.
//
// Ragged tail. M need not be a multiple of NTOK. Each token slot clamps its
// activation row index to M - 1, so every load stays inside the activation
// buffer, and the write is skipped for slots past M - 1. A clamped slot
// recomputes the last token's dot product and discards it, costing at most
// NTOK - 1 wasted columns in one block of the grid. Both early exits are
// block-uniform, so every thread reaches the barrier inside the reduction.
//
// Ragged K. K is gated only on `k % 32 == 0`, so the last 8-block group can
// be partial; the block index is bounds-checked rather than read past the row.

template <typename FMT, int NTOK>
static __device__ __forceinline__ void quant_gemv_legacy_q8_1_mwr_ntok(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    constexpr int NWARPS = mwr_nwarps_ntok(NTOK);

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int col = blockIdx.x;
    const unsigned int m0 = blockIdx.y * NTOK;
    if (col >= (int)N || m0 >= M) return;

    // The weight block and the Q8_1 activation block are both 32 elements, so
    // one count serves both and the block indices coincide.
    const int bpr = K / 32;
    const int gpr = (bpr + 7) / 8; // 8-block groups, rounded up

    const unsigned char* w_row =
        weight + (unsigned long long)col * bpr * FMT::BLOCK_BYTES;

    const unsigned char* q8_rows[NTOK];
    #pragma unroll
    for (int j = 0; j < NTOK; j++) {
        const unsigned int mj = (m0 + j < M) ? (m0 + j) : (M - 1);
        q8_rows[j] = q8_act + (unsigned long long)mj * bpr * 36;
    }

    const int kbx = lane_id / 4;      // block within the 8-block group
    const int w4 = lane_id % 4;       // 4-element source word within that block
    const int pos_lo = 4 + w4 * 4;    // activation byte offset of elements 4w..4w+3
    const int pos_hi = 20 + w4 * 4;   // and of elements 4w+16..4w+19

    float acc[NTOK];
    #pragma unroll
    for (int j = 0; j < NTOK; j++) acc[j] = 0.0f;

    for (int g = warp_id; g < gpr; g += NWARPS) {
        const int b = g * 8 + kbx;
        if (b >= bpr) continue;

        float dw, mw;
        int v_lo, v_hi;
        FMT::decode(w_row + (unsigned long long)b * FMT::BLOCK_BYTES, w4,
                    &dw, &mw, &v_lo, &v_hi);

        #pragma unroll
        for (int j = 0; j < NTOK; j++) {
            const unsigned char* ablk = q8_rows[j] + (unsigned long long)b * 36;
            const float da = __half2float(*(const __half*)ablk);
            const int a_lo = *(const int*)(ablk + pos_lo);
            const int a_hi = *(const int*)(ablk + pos_hi);

            acc[j] += dw * da * (float)dp4a(v_lo, a_lo, dp4a(v_hi, a_hi, 0));
            if constexpr (FMT::HAS_MIN) {
                const int sumi = dp4a(0x01010101, a_lo, dp4a(0x01010101, a_hi, 0));
                acc[j] += mw * da * (float)sumi;
            }
        }
    }

    __shared__ float smem[NWARPS - 1][NTOK][WARP_SIZE];
    float sums[NTOK];
    mwr_reduce_ntok<NTOK, NWARPS>(acc, warp_id, lane_id, smem, sums);

    if (warp_id != 0 || lane_id != 0) return;
    #pragma unroll
    for (int j = 0; j < NTOK; j++) {
        const unsigned int mj = m0 + j;
        if (mj < M) output[(unsigned long long)mj * N + col] = sums[j];
    }
}
