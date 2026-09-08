// Token-batched dp4a GEMV for TCF `Q4AS32DT64`
//
// TCF's f32 GEMV (`kernels/tcf.cu`) reconstructs every weight as an f32 and
// multiplies it by an f32 activation, so it re-reads and re-decodes the whole
// weight matrix once per token. This is the int8 counterpart the GGUF formats
// already have: the activation is quantized to Q8_1 once on the host side of
// the launch, a weight group is decoded ONCE per block, and the dot product
// runs on dp4a. `NTOK` token columns share that one decode.
//
// # Why this format shares the legacy lane map and not the legacy decode
//
// A `Q4AS32DT64` quantization group is 32 elements, which is exactly the
// width of a Q8_1 activation block. So the group indices of the two operands
// coincide and `gemv/legacy_ntok.cuh`'s lane map carries over unchanged: four
// consecutive lanes cover one 32-element group, so a warp covers eight whole
// groups per step and their loads coalesce.
//
// The NIBBLE MAP does not carry over. Section 14.1 packs a 4-bit tile as
// `byte = u[2e] | (u[2e+1] << 4)`, so one byte holds two ADJACENT elements —
// while ggml's legacy map puts elements 16 apart in one byte's two nibbles.
// One `int` of the TCF code plane is therefore 8 CONSECUTIVE elements, and its
// two halves expand into the two dp4a operand words through
// `tcf_expand_nibble_quad`, the same routine `MmqfTcfQ4AS32DT64::stage` uses.
//
// # Value convention — THE MIN SIGN
//
// A TCF value is `d_eff * u + m_eff` with an UNSIGNED code `u` and a fully
// signed `m_eff`, which is Q4_1's `d * q + m` convention, NOT the K-quants'
// `d * sc * q - dmin * m`. `m_eff` is therefore used UNNEGATED here, exactly
// as `MmqfTcfQ4AS32DT64::stage` stores it. Negating it would flip the sign of
// every minimum term and yield plausible wrong numbers, not an error.
//
// The block's contribution is `d_eff * sum(u_e * a_e) + m_eff * sum(a_e)`.
// The second sum is rank-1 over the group — it depends on the activation
// alone — and is formed as an EXACT integer with `dp4a(0x01010101, a, ...)`,
// which is what `mmqf_vec_dot_dm` does with the int16 sum the MMQ activation
// record carries, so the two paths agree on that term bit pattern for bit
// pattern. The per-token Q8_1 record's `s` field is NOT used: its producer
// stores `d * sum(x)` over the ORIGINAL floats rather than `d * sum(q)` over
// the quants, so it would not match MMQ.
//
// # Scale resolution, once per group
//
// `d_eff` and `m_eff` are resolved ONCE per 32-element group, outside the
// token loop, by `tcf_group_values` in `../tcf.cuh` — which is also the one
// place the Section 13.4 two-level arithmetic, the Section 14.6 6-bit field
// position (`tcf_read_packed6`) and the bfloat16 widening
// (`tcf_read_bfloat16`) are written. Nothing here restates any of them.
//
// That helper addresses a super-block by the GLOBAL flattened tile number, so
// this kernel does too, and it is correct for any K a TCF payload can have —
// it does NOT need the `K % 256 == 0` gate the feature-major MMQ staging map
// takes, only the `K % 32 == 0` every dp4a path needs for the activation.
//
// # Alignment
//
// Every 4-byte read this kernel issues is 4-byte aligned, so none of them
// needs `load_int_ua`:
//   - Codes. A group's 4-bit run starts at `tile * 32 + g * 16` bytes into a
//     256-byte-aligned device allocation, and lane `w` reads at `+ 4 * w`.
//   - Activations. A Q8_1 record is 36 bytes, its quants start at byte 4, and
//     a lane reads at `+ 8 * w`.
// The scale side reads bytes (`tcf_read_packed6`) and `memcpy`s a `short`
// (`tcf_read_bfloat16`), neither of which has an alignment requirement.

#pragma once

#include "common.cuh"

#include "../tcf.cuh"

// ── Token-batched MWR body ──────────────────────────────────────────────
//
// Grid: (N, ceil(M / NTOK), 1) — one output column per block, NTOK token
// columns per block. Block: `mwr_nwarps_ntok(NTOK) * WARP_SIZE` threads; the
// launch side must size the block from the same function, because the
// reduction's shared array and `__launch_bounds__` both read it.
//
// The weight load and its decode sit OUTSIDE the token loop — that is the
// whole point of the tile. Only the activation load and the dp4a repeat per
// token, so a weight group is read, its codes unpacked and its two-level
// scale resolved once for all NTOK columns instead of once per column.
//
// Ragged tail. M need not be a multiple of NTOK. Each token slot clamps its
// activation row index to M - 1, so every load stays inside the activation
// buffer, and the write is skipped for slots past M - 1. A clamped slot
// recomputes the last token's dot product and discards it. Both early exits
// are block-uniform, so every thread reaches the barrier inside the reduction.
//
// Ragged K. K is gated only on `k % 32 == 0` by the launcher — and on the
// tile width by `matmul_setup` — so the last 8-group step can be partial; the
// group index is bounds-checked rather than read past the row.

template <int NTOK>
static __device__ __forceinline__ void quant_gemv_tcf_q4as32dt64_q8_1_mwr_ntok(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N,
    TcfLayout l
) {
    constexpr int NWARPS = mwr_nwarps_ntok(NTOK);

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col = blockIdx.x;
    const unsigned int m0 = blockIdx.y * NTOK;
    if (col >= N || m0 >= M) return;

    // The quantization group and the Q8_1 activation block are both 32
    // elements, so one count serves both and the group indices coincide.
    const unsigned int bpr = K / 32;
    const unsigned int steps = (bpr + 7) / 8;  // 8-group steps, rounded up

    const unsigned char* q8_rows[NTOK];
    #pragma unroll
    for (int j = 0; j < NTOK; j++) {
        const unsigned int mj = (m0 + j < M) ? (m0 + j) : (M - 1);
        q8_rows[j] = q8_act + (unsigned long long)mj * bpr * 36;
    }

    const unsigned int kbx = (unsigned int)lane_id / 4u;  // group within the step
    const unsigned int w4 = (unsigned int)lane_id % 4u;   // 8-element code word in it
    // Activation byte offsets of elements 8w..8w+3 and 8w+4..8w+7.
    const unsigned int pos_lo = 4 + w4 * 8;
    const unsigned int pos_hi = pos_lo + 4;
    // Code bytes one group spends, and one lane's offset inside them. A 4-bit
    // code is half a byte, so a 32-element group is 16 bytes and a lane's 8
    // consecutive elements are one `int`.
    const unsigned int group_code_bytes = l.group / 2u;
    const unsigned int lane_code_off = w4 * 4;

    float acc[NTOK];
    #pragma unroll
    for (int j = 0; j < NTOK; j++) acc[j] = 0.0f;

    for (unsigned int step = (unsigned int)warp_id; step < steps; step += NWARPS) {
        const unsigned int b = step * 8 + kbx;
        if (b >= bpr) continue;

        // Planes are indexed by the GLOBAL group number, so a row's groups are
        // located by counting from the start of the tensor, not from the start
        // of the row. That is what makes this kernel independent of whether a
        // row starts on a super-block boundary.
        const unsigned long long gg = (unsigned long long)col * bpr + b;
        const unsigned int tile = (unsigned int)(gg / l.groups_per_tile);
        const unsigned int g = (unsigned int)(gg % l.groups_per_tile);

        float dw, mw;
        tcf_group_values(weight, l, tile, g, &dw, &mw);

        const unsigned char* codes = weight
            + (size_t)tile * (size_t)(TCF_TILE / 2u)
            + (size_t)g * (size_t)group_code_bytes
            + lane_code_off;
        const unsigned int cw = *(const unsigned int*)codes;
        // Adjacent-pair nibbles: the low half of the word is elements
        // 8w..8w+3, the high half is elements 8w+4..8w+7. The codes are
        // unsigned levels 0..15, already inside the int8 range dp4a reads, and
        // the asymmetry rides the minimum term below.
        const int v_lo = tcf_expand_nibble_quad(cw & 0xFFFFu);
        const int v_hi = tcf_expand_nibble_quad(cw >> 16);

        #pragma unroll
        for (int j = 0; j < NTOK; j++) {
            const unsigned char* ablk = q8_rows[j] + (unsigned long long)b * 36;
            const float da = __half2float(*(const __half*)ablk);
            const int a_lo = *(const int*)(ablk + pos_lo);
            const int a_hi = *(const int*)(ablk + pos_hi);

            acc[j] += dw * da * (float)dp4a(v_lo, a_lo, dp4a(v_hi, a_hi, 0));
            // THE MIN SIGN: added, never subtracted. See the header.
            const int sumi = dp4a(0x01010101, a_lo, dp4a(0x01010101, a_hi, 0));
            acc[j] += mw * da * (float)sumi;
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
