// Token-batched dp4a GEMV for the grid-indexed IQ formats
//
// IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ1_S are CODEBOOK
// quantizations: `qs` holds INDICES into a grid of precomputed points, never
// magnitudes. Decoding one weight element means a table read, so the weight
// traffic a per-token GEMV repeats is table traffic as much as memory traffic
// — which is exactly what the `_n2` tile below removes, by decoding a
// sub-group once and dot-producting it against every token column in the
// block.
//
// WHY A THIRD HEADER. `legacy_ntok.cuh` hands its policy a single block
// pointer, which cannot express a sub-block, so none of these six fit it —
// all are 256-element super-blocks. `iq4_ntok.cuh` does hand the policy a row
// base plus a run index, but its lane map is the IQ4 nibble map: a lane owns
// one 4-element source WORD of a 32-element run, and its two dp4a operands
// are the low and high nibble halves of that word, elements `4w..4w+3` and
// `4w+16..4w+19`. A grid format has no nibble halves. Its natural unit is the
// 8-element SUB-GROUP that one grid read covers, and its two dp4a operands
// are the two CONSECUTIVE halves of that sub-group. Lane map, weight
// addressing and activation byte offsets all differ, so generalizing
// `iq4_ntok.cuh` would parameterize every line of its body rather than share
// one.
//
// LANE MAP. A super-block is 256 elements = 32 sub-groups of 8, and a warp
// has 32 lanes, so lane `l` owns sub-group `l` of ONE super-block and a warp
// step covers a whole super-block. That is the same lane-to-sub-group map the
// MMQ staging structs in `../quant_mmq_mma.cu` use, which is what lets the
// decode below be lifted from them unchanged. The 8 elements a lane owns are
// contiguous, and consecutive lanes read consecutive index bytes, so every
// index/sign/scale read across a warp covers a contiguous field of the block.
//
// ACTIVATION. Per-token Q8_1, 36-byte blocks of 32 elements: `f16 d`, `f16 s`,
// then 32 int8 quants. Sub-group `l` lives in the super-block's Q8_1 block
// `l / 4` at quant bytes `8 * (l % 4) .. +7`, so its two dp4a operands are two
// 4-byte-aligned int loads at `4 + 8 * (l % 4)` and four bytes on.
//
// The weight load and its decode sit OUTSIDE the token loop — that is the
// whole point of the tile. Only the activation load and the dp4a repeat per
// token.
//
// K MULTIPLE. All six resolve a sub-group's byte offset through its
// 256-element super-block, so `dispatch_gemv` gates them on `k % 256 == 0`
// rather than the dp4a branch's usual 32. A row whose last super-block were
// partial has no on-disk representation.
//
// SINGLE TOKEN. None of the six has a single-token dp4a sibling: at m = 1 the
// tile's spare column is pure overhead and the F32 kernel beside each policy's
// format serves that shape. `dispatch_gemv` enters the dp4a branch for these
// formats only from m = 2 up.
//
// GROUND TRUTH. Every layout below is checked against
// `src/quant/cpu/kernels/dequant_iq{1,2,3}.rs`, which
// `tests/gguf_conformance_llama_cpp.rs` gates against llama.cpp. The grid
// tables and `KSIGNS` live once in `../iq_grid.cuh`; nothing here adds,
// duplicates or regenerates any of them.

#pragma once

#include "common.cuh"

#include "../iq_dequant.cuh"

// ── Per-format decode policies ──────────────────────────────────────────
//
// Contract. `decode` reads the weight ROW base `w_row`, super-block index
// `sup` and sub-group index `sg` (0..31, eight elements each) and produces:
//   *d     — the scale that applies to that sub-group
//   *dm    — the AFFINE term's coefficient, written only when `AFFINE`
//   *v_lo  — elements `8*sg .. 8*sg+3` as signed int8x4, ready for dp4a
//   *v_hi  — elements `8*sg+4 .. 8*sg+7`, likewise
// `BLOCK_BYTES` gives the super-block's byte size; the row stride is
// `supers * BLOCK_BYTES` for all six, so it is not a policy function.
//
// The row base is the caller's job because it is loop-invariant; the
// super-block addressing is the policy's because it is what differs.
//
// SIGN FOLD. `iq_sign_mask4(sign_byte, t)` builds a per-byte negation mask for
// four consecutive components, and `__vsub4(mag ^ mask, mask)` negates exactly
// the marked ones with no branch — the packed form the MMQ staging pass uses.
// The unsigned grids' largest magnitude is 62, so a negated component still
// fits an int8 lane. IQ1_S's grid is already signed and skips the fold.
//
// SCALE GRANULARITY DIFFERS ACROSS THE SIX, and the index expression is the
// authority, not the packing. IQ2_XS and IQ2_S carry one 4-bit scale per
// SIXTEEN elements (`packed_scale` indexes `entry / 2`, and an entry is eight
// elements). IQ3_XXS and IQ3_S carry one per THIRTY-TWO. IQ2_XXS's is the top
// nibble of its group's `aux` word, also per 32. IQ3_S packs its scales
// exactly as IQ2_XS does and indexes them per 32 anyway.

// IQ2_XXS: 66 bytes / 256 elements. `d` f16@0, `qs[64]`@2 as eight pairs of
// u32. The first u32 of a pair holds four 8-bit indices into `IQ2XXS_GRID`
// (8 magnitude bytes each); the second holds the group's 4-bit scale in its
// top nibble over four 7-bit `KSIGNS` indices.
//
// ALIGNMENT. 66 is not a multiple of 4, so a row base (`supers * 66`) and
// every super-block base inside it are only 2-byte aligned, `qs`@2 with them.
// Both 4-byte reads therefore go through `load_int_ua`; a plain `int` load
// raises CUDA_ERROR_MISALIGNED_ADDRESS, which poisons the context for every
// later launch on it.
struct IqGridIq2Xxs {
    static constexpr int BLOCK_BYTES = 66;
    static constexpr bool AFFINE = false;

    static __device__ __forceinline__ void decode(
        const unsigned char* __restrict__ w_row, int sup, int sg,
        float* d, float* dm, int* v_lo, int* v_hi
    ) {
        (void)dm;
        const unsigned char* blk = w_row + (unsigned long long)sup * BLOCK_BYTES;
        const int group = sg / 4;  // 32-element group
        const int sub = sg % 4;

        const unsigned int ind = (unsigned int)load_int_ua(blk + 2 + group * 8);
        const unsigned int aux = (unsigned int)load_int_ua(blk + 2 + group * 8 + 4);
        *d = __half2float(*(const __half*)blk) * (0.5f + (float)(aux >> 28)) * 0.25f;

        const unsigned long long point = IQ2XXS_GRID[(ind >> (8 * sub)) & 0xFFu];
        const unsigned char signs = KSIGNS[(aux >> (7 * sub)) & 0x7Fu];
        const unsigned int m_lo = iq_sign_mask4(signs, 0);
        const unsigned int m_hi = iq_sign_mask4(signs, 1);
        *v_lo = (int)__vsub4((unsigned int)point ^ m_lo, m_lo);
        *v_hi = (int)__vsub4((unsigned int)(point >> 32) ^ m_hi, m_hi);
    }
};

// IQ2_XS: 74 bytes / 256 elements. `d` f16@0, `qs[64]`@2 as 32 u16,
// `scales[8]`@66. Sub-group `sg` is one u16: its low 9 bits index
// `IQ2XS_GRID` (512 points), its top 7 bits index `KSIGNS`. Two sub-groups
// share one 4-bit scale, so the scale is per SIXTEEN elements.
//
// ALIGNMENT. The u16 sits at the even offset `2 + 2 * sg` inside a 2-byte
// aligned block, so it is aligned; there is no 4-byte read and `load_int_ua`
// does not apply.
struct IqGridIq2Xs {
    static constexpr int BLOCK_BYTES = 74;
    static constexpr bool AFFINE = false;
    static constexpr int SCALES_OFF = 66;

    static __device__ __forceinline__ void decode(
        const unsigned char* __restrict__ w_row, int sup, int sg,
        float* d, float* dm, int* v_lo, int* v_hi
    ) {
        (void)dm;
        const unsigned char* blk = w_row + (unsigned long long)sup * BLOCK_BYTES;

        const unsigned int q = *(const unsigned short*)(blk + 2 + sg * 2);
        const float sc = (float)iq_packed_scale(blk + SCALES_OFF, sg);
        *d = __half2float(*(const __half*)blk) * (0.5f + sc) * 0.25f;

        const unsigned long long point = IQ2XS_GRID[q & 511u];
        const unsigned char signs = KSIGNS[q >> 9];
        const unsigned int m_lo = iq_sign_mask4(signs, 0);
        const unsigned int m_hi = iq_sign_mask4(signs, 1);
        *v_lo = (int)__vsub4((unsigned int)point ^ m_lo, m_lo);
        *v_hi = (int)__vsub4((unsigned int)(point >> 32) ^ m_hi, m_hi);
    }
};

// IQ2_S: 82 bytes / 256 elements. `d` f16@0, `qs[32]`@2, `signs[32]`@34,
// `qh[8]`@66, `scales[8]`@74. Sub-group `sg` takes its low 8 index bits from
// `qs[sg]` and two more from field `2 * (sg % 4)` of `qh[sg / 4]`, selecting
// among the 1024 points of `IQ2S_GRID`.
//
// SIGNS. The divergence from IQ2_XXS and IQ2_XS: `signs[sg]` is eight explicit
// bits, one per element, with no `KSIGNS` indirection.
//
// SCALE. Identical packing AND identical indexing to IQ2_XS — one 4-bit scale
// per sixteen elements — only the field offset differs.
//
// ALIGNMENT. Every read here is a single byte apart from `d`, so
// `load_int_ua` does not apply.
struct IqGridIq2S {
    static constexpr int BLOCK_BYTES = 82;
    static constexpr bool AFFINE = false;
    static constexpr int SIGNS_OFF = 34;
    static constexpr int QH_OFF = 66;
    static constexpr int SCALES_OFF = 74;

    static __device__ __forceinline__ void decode(
        const unsigned char* __restrict__ w_row, int sup, int sg,
        float* d, float* dm, int* v_lo, int* v_hi
    ) {
        (void)dm;
        const unsigned char* blk = w_row + (unsigned long long)sup * BLOCK_BYTES;

        const unsigned int high = (blk[QH_OFF + sg / 4] >> (2 * (sg % 4))) & 0x03u;
        const float sc = (float)iq_packed_scale(blk + SCALES_OFF, sg);
        *d = __half2float(*(const __half*)blk) * (0.5f + sc) * 0.25f;

        const unsigned long long point = IQ2S_GRID[blk[2 + sg] | (high << 8)];
        const unsigned char signs = blk[SIGNS_OFF + sg];
        const unsigned int m_lo = iq_sign_mask4(signs, 0);
        const unsigned int m_hi = iq_sign_mask4(signs, 1);
        *v_lo = (int)__vsub4((unsigned int)point ^ m_lo, m_lo);
        *v_hi = (int)__vsub4((unsigned int)(point >> 32) ^ m_hi, m_hi);
    }
};

// IQ3_XXS: 98 bytes / 256 elements. `d` f16@0, `qs[64]`@2, `scales[32]`@66 as
// eight u32, one per 32-element group, each holding the group's 4-bit scale in
// its top nibble over four 7-bit `KSIGNS` indices.
//
// TWO GRID POINTS PER SUB-GROUP. `IQ3XXS_GRID` entries are FOUR components, so
// sub-group `sg` needs both `qs[2 * sg]` (elements `8sg .. 8sg+3`) and
// `qs[2 * sg + 1]` (elements `8sg+4 .. 8sg+7`). The CPU dequantizer reaches
// the same two bytes as `qs[group * 8 + sub * 2 (+1)]` with `group = sg / 4`
// and `sub = sg % 4`, and `(sg / 4) * 8 + (sg % 4) * 2` is `2 * sg`. The two
// bytes are adjacent at the even offset `2 + 2 * sg`, so ONE aligned u16 read
// serves the pair — low byte first on a little-endian device.
//
// ALIGNMENT. 98 is not a multiple of 4, so the `aux` read is 4 bytes at a
// 2-byte aligned address and goes through `load_int_ua`.
struct IqGridIq3Xxs {
    static constexpr int BLOCK_BYTES = 98;
    static constexpr bool AFFINE = false;
    static constexpr int SCALES_OFF = 66;

    static __device__ __forceinline__ void decode(
        const unsigned char* __restrict__ w_row, int sup, int sg,
        float* d, float* dm, int* v_lo, int* v_hi
    ) {
        (void)dm;
        const unsigned char* blk = w_row + (unsigned long long)sup * BLOCK_BYTES;
        const int group = sg / 4;  // 32-element group
        const int sub = sg % 4;

        const unsigned int aux = (unsigned int)load_int_ua(blk + SCALES_OFF + group * 4);
        // The 0.5 multiplier here, not IQ2's 0.25.
        *d = __half2float(*(const __half*)blk) * (0.5f + (float)(aux >> 28)) * 0.5f;

        const unsigned int qs = *(const unsigned short*)(blk + 2 + sg * 2);
        const unsigned char signs = KSIGNS[(aux >> (7 * sub)) & 0x7Fu];
        const unsigned int m_lo = iq_sign_mask4(signs, 0);
        const unsigned int m_hi = iq_sign_mask4(signs, 1);
        *v_lo = (int)__vsub4(IQ3XXS_GRID[qs & 0xFFu] ^ m_lo, m_lo);
        *v_hi = (int)__vsub4(IQ3XXS_GRID[qs >> 8] ^ m_hi, m_hi);
    }
};

// IQ3_S: 110 bytes / 256 elements. `d` f16@0, `qs[64]`@2, `qh[8]`@66,
// `signs[32]`@74, `scales[4]`@106. A four-component grid like IQ3_XXS, so
// sub-group `sg` again reads grid entries `2 * sg` and `2 * sg + 1` as one
// aligned u16. Entry `n` takes a NINTH index bit from bit `n % 8` of
// `qh[n / 8]`, selecting among the 512 points of `IQ3S_GRID`; both entries of
// a sub-group land in `qh[sg / 4]` at bits `2 * (sg % 4)` and one above, so one
// byte read serves the pair.
//
// SIGNS. Explicit bits, `signs[sg]`, eight per sub-group — no `KSIGNS`.
//
// SCALE GRANULARITY. This is the trap. `scales` packs 4-bit values two per
// byte exactly as IQ2_XS does, but `dequant_iq3_s` indexes it at `e / 32`, not
// at `entry / 2`: one scale per THIRTY-TWO elements, so a sub-group's scale
// group is `sg / 4`. `iq_packed_scale` would give the per-16 indexing and is
// deliberately not used here. The value is `d * (1 + 2 * s)`, not IQ2's
// `d * (0.5 + s) * 0.25`.
//
// ALIGNMENT. No 4-byte read at all here, so `load_int_ua` does not apply.
struct IqGridIq3S {
    static constexpr int BLOCK_BYTES = 110;
    static constexpr bool AFFINE = false;
    static constexpr int QH_OFF = 66;
    static constexpr int SIGNS_OFF = 74;
    static constexpr int SCALES_OFF = 106;

    static __device__ __forceinline__ void decode(
        const unsigned char* __restrict__ w_row, int sup, int sg,
        float* d, float* dm, int* v_lo, int* v_hi
    ) {
        (void)dm;
        const unsigned char* blk = w_row + (unsigned long long)sup * BLOCK_BYTES;
        const int group = sg / 4;  // 32-element group, the scale's granularity

        const unsigned int sc = (blk[SCALES_OFF + group / 2] >> (4 * (group % 2))) & 0x0Fu;
        *d = __half2float(*(const __half*)blk) * (1.0f + 2.0f * (float)sc);

        const unsigned int qs = *(const unsigned short*)(blk + 2 + sg * 2);
        const unsigned int qh = blk[QH_OFF + group];
        const unsigned int qh_shift = 2 * (unsigned int)(sg % 4);
        const unsigned int g_lo = IQ3S_GRID[(qs & 0xFFu) | (((qh >> qh_shift) & 1u) << 8)];
        const unsigned int g_hi = IQ3S_GRID[(qs >> 8) | (((qh >> (qh_shift + 1)) & 1u) << 8)];

        const unsigned char signs = blk[SIGNS_OFF + sg];
        const unsigned int m_lo = iq_sign_mask4(signs, 0);
        const unsigned int m_hi = iq_sign_mask4(signs, 1);
        *v_lo = (int)__vsub4(g_lo ^ m_lo, m_lo);
        *v_hi = (int)__vsub4(g_hi ^ m_hi, m_hi);
    }
};

// IQ1_S: 50 bytes / 256 elements. `d` f16@0, `qs[32]`@2, `qh[16]`@34 as eight
// u16, one per 32-element group. Sub-group `sg` takes its low 8 index bits
// from `qs[sg]` and three more from field `3 * (sg % 4)` of `qh[sg / 4]`, an
// 11-bit index into the 2048-point `IQ1_GRID`. That group's u16 also carries a
// 3-bit scale at bits 12..14 and the group's delta sign at bit 15.
//
// SIGNS. None to fold: `IQ1_GRID` stores SIGNED components in {-1, 0, 1} and
// the format has no sign table, so the grid words go straight into the dp4a
// operands.
//
// THE AFFINE VALUE. This is what separates IQ1_S from the other five. Its
// dequantized value is `dl * (g + delta)` with `dl = d * (2 * s + 1)` and
// `delta = +/- 0.125`, an AFFINE transform of the grid point rather than a
// scale times an int8. Over a sub-group's activations `a`:
//
//     sum_j a_j * dl * (g_j + delta) = dl * dot(a, g) + dl * delta * sum(a)
//
// so the policy emits `dl` in `*d` and `dl * delta` in `*dm`, and the body
// adds the second term against the activation's own sum. Both `dl` and
// `delta` are per 32-element group, and the body's sub-group is one quarter of
// such a group, so a sub-group's slice of the sum is the right unit.
//
// The delta's own sign, bit 15 of the group's u16, is folded into `*dm`, and
// the term is ADDED — the same additive convention `MmqfIQ1S` stores for
// `mmqf_vec_dot_dm`.
//
// ALIGNMENT. No 4-byte read: `qs` is one byte and `qh` a u16 at the even
// offset `34 + 2 * (sg / 4)`. `load_int_ua` does not apply.
struct IqGridIq1S {
    static constexpr int BLOCK_BYTES = 50;
    static constexpr bool AFFINE = true;
    static constexpr int QH_OFF = 34;

    static __device__ __forceinline__ void decode(
        const unsigned char* __restrict__ w_row, int sup, int sg,
        float* d, float* dm, int* v_lo, int* v_hi
    ) {
        const unsigned char* blk = w_row + (unsigned long long)sup * BLOCK_BYTES;
        const int group = sg / 4;  // 32-element group
        const int sub = sg % 4;

        const unsigned int h = *(const unsigned short*)(blk + QH_OFF + group * 2);
        const float dl = __half2float(*(const __half*)blk) * (2.0f * (float)((h >> 12) & 7u) + 1.0f);
        const float delta = (h & 0x8000u) == 0 ? IQ1_DELTA : -IQ1_DELTA;
        *d = dl;
        *dm = dl * delta;

        const unsigned long long point = IQ1_GRID[blk[2 + sg] | (((h >> (3 * sub)) & 7u) << 8)];
        *v_lo = (int)(unsigned int)point;
        *v_hi = (int)(unsigned int)(point >> 32);
    }
};

// ── Shared token-batched MWR body ───────────────────────────────────────
//
// Grid: (N, ceil(M / NTOK), 1) — one output column per block, NTOK token
// columns per block. Block: `mwr_nwarps_ntok(NTOK) * WARP_SIZE` threads; the
// launch side must size the block from the same function, because the
// reduction's shared array and `__launch_bounds__` both read it.
//
// Ragged tail. M need not be a multiple of NTOK. Each token slot clamps its
// activation row index to M - 1, so every load stays inside the activation
// buffer, and the write is skipped for slots past M - 1. A clamped slot
// recomputes the last token's dot product and discards it, costing at most
// NTOK - 1 wasted columns in one block of the grid. Both early exits are
// block-uniform, so every thread reaches the barrier inside the reduction.
//
// THE ACTIVATION BLOCK SUM. The affine formats need `sum(a)` over the eight
// activation quants a sub-group covers. It is formed here as an exact integer,
// `dp4a(0x01010101, ...)` against the int8 bytes, then scaled by the
// activation block's own `d` — the same construction the Q4_K/Q5_K and
// Q4_1/Q5_1 kernels use. The Q8_1 record's `s` field is NOT used: its producer
// stores `d * sum(x)` over the ORIGINAL f32 inputs, not `d * sum(q)` over the
// quants, so an additive term built from it would disagree with MMQ by the
// quantization residual. The integer sum here matches the int16 sum MMQ reads
// from its own record, so the two paths agree.

template <typename FMT, int NTOK>
static __device__ __forceinline__ void quant_gemv_iq_grid_q8_1_mwr_ntok(
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

    // 32-element Q8_1 activation blocks per row, and the 256-element weight
    // super-blocks they group into eight at a time. Dispatch gates on
    // `k % 256 == 0`, so both divisions are exact.
    const int bpr = K / 32;
    const int supers = bpr / 8;

    const unsigned long long row_bytes = (unsigned long long)supers * FMT::BLOCK_BYTES;
    const unsigned char* w_row = weight + (unsigned long long)col * row_bytes;

    const unsigned char* q8_rows[NTOK];
    #pragma unroll
    for (int j = 0; j < NTOK; j++) {
        const unsigned int mj = (m0 + j < M) ? (m0 + j) : (M - 1);
        q8_rows[j] = q8_act + (unsigned long long)mj * bpr * 36;
    }

    // Lane `l` owns sub-group `l`: elements `8l .. 8l+7` of the super-block,
    // which sit in Q8_1 block `l / 4` at quant bytes `8 * (l % 4) .. +7`. The
    // 4-byte header puts both operand loads on a 4-byte boundary.
    const int ablk_in_super = lane_id / 4;
    const int pos_lo = 4 + (lane_id % 4) * 8;
    const int pos_hi = pos_lo + 4;

    float acc[NTOK];
    #pragma unroll
    for (int j = 0; j < NTOK; j++) acc[j] = 0.0f;

    for (int s = warp_id; s < supers; s += NWARPS) {
        float dw;
        float dmw = 0.0f;
        int v_lo, v_hi;
        FMT::decode(w_row, s, lane_id, &dw, &dmw, &v_lo, &v_hi);

        #pragma unroll
        for (int j = 0; j < NTOK; j++) {
            const unsigned char* ablk =
                q8_rows[j] + (unsigned long long)(s * 8 + ablk_in_super) * 36;
            const float da = __half2float(*(const __half*)ablk);
            const int a_lo = *(const int*)(ablk + pos_lo);
            const int a_hi = *(const int*)(ablk + pos_hi);

            float term = dw * da * (float)dp4a(v_lo, a_lo, dp4a(v_hi, a_hi, 0));
            if constexpr (FMT::AFFINE) {
                const int asum = dp4a(0x01010101, a_lo, dp4a(0x01010101, a_hi, 0));
                term += dmw * da * (float)asum;
            }
            acc[j] += term;
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
