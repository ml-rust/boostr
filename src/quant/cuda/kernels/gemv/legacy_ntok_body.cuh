// Shared token-batched MWR body for the 32-element-chunk dp4a formats
//
// `legacy_ntok.cuh` includes this header and holds the four legacy decode
// policies. `prism_ntok.cuh` supplies the PrismML-fork policies. The policy
// contract is stated in `legacy_ntok.cuh`.
//
// Grid: (ceil(N / ROWS), ceil(M / NTOK), 1) — ROWS output columns, NTOK token
// columns per block. Block: `mwr_nwarps_ntok(NTOK) * WARP_SIZE` threads; the
// launch side must size the block from the same function, because the
// reduction's shared array and `__launch_bounds__` both read it. NTOK = 1,
// ROWS = 1 is a valid instance: one accumulator, one activation row, one
// weight row, a `[NWARPS - 1][1][1][WARP_SIZE]` reduction array.
//
// Lane map. A 32-element chunk holds 4 source words, so a warp's 32 lanes
// cover 8 whole chunks per step: lane maps to (chunk `lane / 4` inside an
// 8-chunk group, source word `lane % 4`). Four consecutive lanes read 16
// contiguous bytes of the same chunk, and the 8-chunk group is contiguous in
// the row, so the group's loads coalesce. For a legacy format a chunk is a
// block; for a prism format `CHUNKS_PER_BLOCK` chunks share one block base
// and scale, and the policy receives the source word index widened over the
// whole block.
//
// The weight load and its decode sit OUTSIDE the token loop — that is the
// whole point of the token axis. Only the activation load and the dp4a
// repeat per token, so a weight block is read and unpacked once for all NTOK
// columns instead of once per column.
//
// Output-row axis. Each activation word a lane loads is dot-producted
// against the same word of ROWS weight rows, so the activation row is read
// once per ROWS outputs instead of once per output. At NTOK = 1 that is the
// only traffic there is to save: one block per output column re-reads the
// whole Q8_1 activation row, N times per launch. ROWS = 1 keeps the old
// one-column geometry, and every existing instantiation uses it.
//
// The k-accumulation order of one output does not depend on ROWS: each
// `acc[j][i]` sees the same chunks, in the same order, from the same warp
// and lane as at ROWS = 1, and `mwr_reduce_ntok` reduces every (j, i) slot
// by the same warp-then-shuffle tree. So a row's result is the same bits at
// every ROWS. Only the loop nesting around the FMA changes.
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
// Ragged tails. M need not be a multiple of NTOK, nor N of ROWS. Each token
// slot clamps its activation row index to M - 1 and each row slot clamps its
// weight row index to N - 1, so every load stays inside its buffer, and the
// write is skipped for slots past the end. A clamped slot recomputes the last
// row's dot product and discards it. Both early exits are block-uniform, so
// every thread reaches the barrier inside the reduction.
//
// Ragged K. K is gated on `k % (32 * CHUNKS_PER_BLOCK) == 0`, so the last
// 8-chunk group can be partial; the chunk index is bounds-checked rather than
// read past the row.

#pragma once

#include "common.cuh"

template <typename FMT, int NTOK, int ROWS = 1>
static __device__ __forceinline__ void quant_gemv_legacy_q8_1_mwr_ntok(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    constexpr int NWARPS = mwr_nwarps_ntok(NTOK);

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int n0 = blockIdx.x * ROWS;
    const unsigned int m0 = blockIdx.y * NTOK;
    if (n0 >= N || m0 >= M) return;

    // A weight chunk and a Q8_1 activation block are both 32 elements, so
    // one count serves both and their indices coincide. The weight row holds
    // `bpr / CHUNKS_PER_BLOCK` blocks.
    const int bpr = K / 32;
    const int gpr = (bpr + 7) / 8; // 8-chunk groups, rounded up
    const unsigned long long row_bytes =
        (unsigned long long)(bpr / FMT::CHUNKS_PER_BLOCK) * FMT::BLOCK_BYTES;

    const unsigned char* w_rows[ROWS];
    #pragma unroll
    for (int i = 0; i < ROWS; i++) {
        const unsigned int ni = (n0 + i < N) ? (n0 + i) : (N - 1);
        w_rows[i] = weight + ni * row_bytes;
    }

    const unsigned char* q8_rows[NTOK];
    #pragma unroll
    for (int j = 0; j < NTOK; j++) {
        const unsigned int mj = (m0 + j < M) ? (m0 + j) : (M - 1);
        q8_rows[j] = q8_act + (unsigned long long)mj * bpr * 36;
    }

    const int kbx = lane_id / 4;      // chunk within the 8-chunk group
    const int w4 = lane_id % 4;       // 4-element source word within that chunk
    const int pos_lo = 4 + w4 * 4;    // activation byte offset of elements 4w..4w+3
    const int pos_hi = 20 + w4 * 4;   // and of elements 4w+16..4w+19

    float acc[NTOK][ROWS];
    #pragma unroll
    for (int j = 0; j < NTOK; j++)
        #pragma unroll
        for (int i = 0; i < ROWS; i++) acc[j][i] = 0.0f;

    for (int g = warp_id; g < gpr; g += NWARPS) {
        const int b = g * 8 + kbx;
        if (b >= bpr) continue;

        const unsigned long long blk_off =
            (unsigned long long)(b / FMT::CHUNKS_PER_BLOCK) * FMT::BLOCK_BYTES;
        const int w = (b % FMT::CHUNKS_PER_BLOCK) * 4 + w4;

        float dw[ROWS], mw[ROWS];
        int v_lo[ROWS], v_hi[ROWS];
        #pragma unroll
        for (int i = 0; i < ROWS; i++)
            FMT::decode(w_rows[i] + blk_off, w, &dw[i], &mw[i], &v_lo[i], &v_hi[i]);

        #pragma unroll
        for (int j = 0; j < NTOK; j++) {
            const unsigned char* ablk = q8_rows[j] + (unsigned long long)b * 36;
            const float da = __half2float(*(const __half*)ablk);
            const int a_lo = *(const int*)(ablk + pos_lo);
            const int a_hi = *(const int*)(ablk + pos_hi);
            [[maybe_unused]] int sumi = 0;
            if constexpr (FMT::HAS_MIN)
                sumi = dp4a(0x01010101, a_lo, dp4a(0x01010101, a_hi, 0));

            #pragma unroll
            for (int i = 0; i < ROWS; i++) {
                acc[j][i] += dw[i] * da * (float)dp4a(v_lo[i], a_lo, dp4a(v_hi[i], a_hi, 0));
                if constexpr (FMT::HAS_MIN) acc[j][i] += mw[i] * da * (float)sumi;
            }
        }
    }

    __shared__ float smem[NWARPS - 1][NTOK][ROWS][WARP_SIZE];
    float sums[NTOK][ROWS];
    mwr_reduce_ntok<NTOK, ROWS, NWARPS>(acc, warp_id, lane_id, smem, sums);

    if (warp_id != 0 || lane_id != 0) return;
    #pragma unroll
    for (int j = 0; j < NTOK; j++) {
        const unsigned int mj = m0 + j;
        if (mj >= M) continue;
        #pragma unroll
        for (int i = 0; i < ROWS; i++) {
            const unsigned int ni = n0 + i;
            if (ni < N) output[(unsigned long long)mj * N + ni] = sums[j][i];
        }
    }
}
