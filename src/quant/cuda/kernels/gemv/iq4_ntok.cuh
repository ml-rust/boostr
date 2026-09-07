// Token-batched dp4a GEMV for the IQ4 codebook formats
//
// IQ4_NL and IQ4_XS share one 4-bit field map — split-half nibble order over
// 32-element runs — and one 16-entry signed codebook, `KVALUES_IQ4NL` in
// `../decode.cuh`. What they do NOT share with the legacy 32-element formats
// in `legacy_ntok.cuh` is block addressing: IQ4_XS carries eight 32-element
// runs inside one 136-byte super-block and changes scale at every run, so a
// run's byte offset is not `run_index * BLOCK_BYTES`. `legacy_ntok.cuh`'s
// policy contract hands the decoder a single block pointer computed from that
// product, which cannot express a sub-block. Hence a second header rather than
// a fifth policy there: the contract below hands the decoder the ROW base and
// the 32-element run index and lets each format do its own addressing.
//
// Neither format has an additive minimum — the codebook value is the whole
// magnitude — so the dot is `scale * dp4a_sum` with no block-sum term, and the
// per-token Q8_1 record's `s` field is unused. (Its producer stores
// `d * sum(x)` over the ORIGINAL floats, not `d * sum(q)` over the quants, so
// it is not a usable block sum in any case.)
//
// Like the legacy four, these two have no single-token dp4a GEMV: the `_n2`
// tile exists because batching pays at m = 2, while m = 1 is served by the
// F32-activation kernel beside it. `dispatch_gemv` reflects that by entering
// the dp4a branch for these formats only from m = 2 up.
//
// Weight decode is lifted from the MMQ staging structs `MmqfIQ4NL` and
// `MmqfIQ4XS` in `../quant_mmq_mma.cu`, which are in turn cross-checked
// element by element against `dequant_iq4_nl` and `dequant_iq4_xs` in
// `src/quant/cpu/kernels/dequant_iq4.rs`. Nibble map, shared by both: within
// one 32-element run, element `j` (0..15) is the LOW nibble of `qs[j]` and
// element `j + 16` is the HIGH nibble of the same byte. So the int at
// `qs + 4*w` carries elements `4w..4w+3` in its low nibbles and `4w+16..4w+19`
// in its high nibbles — two dp4a operands from one load, which is exactly what
// `gguf_iq4_table_lookup` returns.
//
// ALIGNMENT. IQ4_NL's block is 18 bytes, so a row base (`bpr * 18`) and every
// block base inside it are only 2-byte aligned; its 4-byte quant read goes
// through `load_int_ua`, because a plain `int` load raises
// CUDA_ERROR_MISALIGNED_ADDRESS, which poisons the context for every later
// launch on it. IQ4_XS's super-block is 136 bytes, a multiple of 8, so `qs`@8
// is 4-byte aligned and read directly, exactly as `MmqfIQ4XS` does.

#pragma once

#include "common.cuh"

// ── Per-format decode policies ──────────────────────────────────────────
//
// Contract. `decode` reads the weight ROW base `w_row`, 32-element run index
// `b` and 4-element source word `w` (0..3) and produces:
//   *d     — the scale that applies to that run
//   *v_lo  — elements 4w..4w+3 as codebook values, int8x4, ready for dp4a
//   *v_hi  — elements 4w+16..4w+19, likewise
// `row_bytes(bpr)` gives the byte stride between output columns, with `bpr`
// counting 32-element runs. The row base is the caller's job because it is
// loop-invariant; the run addressing is the policy's because it is what
// differs between the two formats.

// IQ4_NL: 18 bytes / 32 elements. `d` f16@0, 16 nibble bytes@2.
// Value `d * KVALUES_IQ4NL[q]`, one scale per 32-element block.
struct Iq4Nl {
    static constexpr int BLOCK_BYTES = 18;

    static __host__ __device__ __forceinline__ unsigned long long row_bytes(unsigned int bpr) {
        return (unsigned long long)bpr * BLOCK_BYTES;
    }

    static __device__ __forceinline__ void decode(
        const unsigned char* __restrict__ w_row, int b, int w,
        float* d, int* v_lo, int* v_hi
    ) {
        const unsigned char* blk = w_row + (unsigned long long)b * BLOCK_BYTES;
        *d = __half2float(*(const __half*)blk);
        // 2-byte aligned block base: unaligned load, see ALIGNMENT above.
        const int q = load_int_ua(blk + 2 + w * 4);
        // Table values are signed int8 already, so nothing is folded in here;
        // the lookup replaces the `__vsubss4` a magnitude format does.
        gguf_iq4_table_lookup(q, v_lo, v_hi);
    }
};

// IQ4_XS: 136 bytes / 256 elements, matching llama.cpp's `block_iq4_xs`:
//   { f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128]; }
// so `scales_h` is a TWO-byte field at offset 2 and `scales_l` occupies 4..8
// with no pad byte. Eight 32-element sub-blocks each take a 6-bit scale from
// one `scales_l` nibble plus two `scales_h` bits, biased by 32:
// `dl = d * (ls - 32)`. `scales_h` carries the high bits of ALL EIGHT
// sub-blocks (2 bits each = 16), so reading it as one byte would drop half of
// them and shift `scales_l` by one.
//
// The scale changes every 32 elements, which is the granularity the kernel's
// run loop already works at, so the per-run decode below resolves it directly
// instead of hoisting it — a lane's runs are 8 apart and rarely share a
// super-block.
struct Iq4Xs {
    static constexpr int BLOCK_BYTES = 136;
    static constexpr int RUNS_PER_BLOCK = 8;

    static __host__ __device__ __forceinline__ unsigned long long row_bytes(unsigned int bpr) {
        return (unsigned long long)(bpr / RUNS_PER_BLOCK) * BLOCK_BYTES;
    }

    static __device__ __forceinline__ void decode(
        const unsigned char* __restrict__ w_row, int b, int w,
        float* d, int* v_lo, int* v_hi
    ) {
        const int sb = b % RUNS_PER_BLOCK;  // 32-element sub-block
        const unsigned char* blk =
            w_row + (unsigned long long)(b / RUNS_PER_BLOCK) * BLOCK_BYTES;

        const float dsuper = __half2float(*(const __half*)blk);
        const unsigned int sh = *(const unsigned short*)(blk + 2);
        const unsigned int sl = blk[4 + sb / 2];
        // Low 4 bits from the `scales_l` nibble for this sub-block, high 2
        // bits from `scales_h` bits `2*sb`, then the -32 bias.
        const int ls = (int)(((sl >> (4 * (sb % 2))) & 0x0Fu) | (((sh >> (2 * sb)) & 0x03u) << 4));
        *d = dsuper * (float)(ls - 32);

        // `qs`@8 of an 8-byte-aligned super-block is 4-byte aligned, so this
        // is a plain `int` load. Sub-block `sb` owns bytes `16*sb .. 16*sb+15`.
        const int q = *(const int*)(blk + 8 + sb * 16 + w * 4);
        gguf_iq4_table_lookup(q, v_lo, v_hi);
    }
};

// ── Shared token-batched MWR body ───────────────────────────────────────
//
// Grid: (N, ceil(M / NTOK), 1) — one output column per block, NTOK token
// columns per block. Block: `mwr_nwarps_ntok(NTOK) * WARP_SIZE` threads; the
// launch side must size the block from the same function, because the
// reduction's shared array and `__launch_bounds__` both read it.
//
// Lane map. A 32-element run holds 4 source words, so a warp's 32 lanes cover
// 8 whole runs per step: lane maps to (run `lane / 4` inside an 8-run group,
// source word `lane % 4`). Four consecutive lanes read 16 contiguous bytes of
// the same run, and the 8-run group is contiguous in the row, so the group's
// loads coalesce. For IQ4_XS an 8-run group is exactly one super-block, so a
// group's 32 lanes read its whole 128-byte `qs` field.
//
// The weight load and its decode sit OUTSIDE the token loop — that is the
// whole point of the tile. Only the activation load and the dp4a repeat per
// token, so a weight run is read and unpacked once for all NTOK columns
// instead of once per column.
//
// Ragged tail. M need not be a multiple of NTOK. Each token slot clamps its
// activation row index to M - 1, so every load stays inside the activation
// buffer, and the write is skipped for slots past M - 1. A clamped slot
// recomputes the last token's dot product and discards it, costing at most
// NTOK - 1 wasted columns in one block of the grid. Both early exits are
// block-uniform, so every thread reaches the barrier inside the reduction.
//
// Ragged K. IQ4_NL is gated only on `k % 32 == 0`, so its last 8-run group can
// be partial; the run index is bounds-checked rather than read past the row.
// IQ4_XS is gated on `k % 256 == 0`, which makes its groups always whole, and
// the same check simply never fires.

template <typename FMT, int NTOK>
static __device__ __forceinline__ void quant_gemv_iq4_q8_1_mwr_ntok(
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

    // 32-element runs per row. The Q8_1 activation block is also 32 elements,
    // so one count serves both and the run indices coincide.
    const int bpr = K / 32;
    const int gpr = (bpr + 7) / 8; // 8-run groups, rounded up

    const unsigned char* w_row = weight + (unsigned long long)col * FMT::row_bytes(bpr);

    const unsigned char* q8_rows[NTOK];
    #pragma unroll
    for (int j = 0; j < NTOK; j++) {
        const unsigned int mj = (m0 + j < M) ? (m0 + j) : (M - 1);
        q8_rows[j] = q8_act + (unsigned long long)mj * bpr * 36;
    }

    const int kbx = lane_id / 4;      // run within the 8-run group
    const int w4 = lane_id % 4;       // 4-element source word within that run
    const int pos_lo = 4 + w4 * 4;    // activation byte offset of elements 4w..4w+3
    const int pos_hi = 20 + w4 * 4;   // and of elements 4w+16..4w+19

    float acc[NTOK];
    #pragma unroll
    for (int j = 0; j < NTOK; j++) acc[j] = 0.0f;

    for (int g = warp_id; g < gpr; g += NWARPS) {
        const int b = g * 8 + kbx;
        if (b >= bpr) continue;

        float dw;
        int v_lo, v_hi;
        FMT::decode(w_row, b, w4, &dw, &v_lo, &v_hi);

        #pragma unroll
        for (int j = 0; j < NTOK; j++) {
            const unsigned char* ablk = q8_rows[j] + (unsigned long long)b * 36;
            const float da = __half2float(*(const __half*)ablk);
            const int a_lo = *(const int*)(ablk + pos_lo);
            const int a_hi = *(const int*)(ablk + pos_hi);

            acc[j] += dw * da * (float)dp4a(v_lo, a_lo, dp4a(v_hi, a_hi, 0));
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
