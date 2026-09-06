// GGUF K-quant scale/min decoders — the ONE place these bit layouts are
// written. Every kernel file that needs Q3_K/Q4_K/Q5_K scales includes this
// instead of restating the packing, so a layout fix lands once.
//
// Format decode only: no GEMV/GEMM tuning constants, no block-size defines,
// no activation helpers. Self-contained — no dependency on WARP_SIZE,
// load_int_ua, or dp4a from the per-kernel-family common.cuh files, since
// quant_gemv.cu and quant_matmul.cu define their own copies of those and a
// second definition here conflicts.

#pragma once

// ── Q4_K / Q5_K scale unpacking ────────────────────────────────────────
// Shared by Q4_K and Q5_K: 12-byte packed 6-bit scales and mins → 8 each

static __device__ __forceinline__ void unpack_q4k_q5k_scales(
    const unsigned char* sc,
    unsigned char* scales,
    unsigned char* mins
) {
    for (int i = 0; i < 4; i++) {
        scales[i] = sc[i] & 0x3F;
        mins[i] = sc[i + 4] & 0x3F;
    }
    for (int i = 4; i < 8; i++) {
        scales[i] = (sc[i + 4] & 0x0F) | ((sc[i - 4] >> 6) << 4);
        mins[i] = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4);
    }
}

// ── Selective scale unpack via 16-bit loads (MWR optimized) ─────────────
// Used by Q4_K and Q5_K MWR kernels. Given chunk index (j_lo/2),
// returns the scale/min pair for j_lo and j_hi sub-blocks.

static __device__ __forceinline__ void unpack_scales_mwr(
    const unsigned char* sc,
    int j_lo,
    unsigned char* scale_lo, unsigned char* scale_hi,
    unsigned char* min_lo, unsigned char* min_hi
) {
    const unsigned short* sc16 = (const unsigned short*)sc;
    const int j = j_lo / 2;
    if (j < 2) {
        unsigned short s0 = sc16[j] & 0x3F3F;
        unsigned short s1 = sc16[j + 2] & 0x3F3F;
        *scale_lo = (unsigned char)(s0);
        *scale_hi = (unsigned char)(s0 >> 8);
        *min_lo = (unsigned char)(s1);
        *min_hi = (unsigned char)(s1 >> 8);
    } else {
        unsigned short s0 = ((sc16[j + 2]) & 0x0F0F) | ((sc16[j - 2] & 0xC0C0) >> 2);
        unsigned short s1 = ((sc16[j + 2] >> 4) & 0x0F0F) | ((sc16[j] & 0xC0C0) >> 2);
        *scale_lo = (unsigned char)(s0);
        *scale_hi = (unsigned char)(s0 >> 8);
        *min_lo = (unsigned char)(s1);
        *min_hi = (unsigned char)(s1 >> 8);
    }
}

// ── Q3_K scale unpacking ────────────────────────────────────────────────
// 12 bytes → 16 signed 6-bit scales, each stored biased by 32.
//
// Register form: returns scale `j` (0..15) from the two `scales` bytes that
// carry it, so a staging loop that must issue every global load before any ALU
// work loads those bytes itself and decodes here. Same split as
// `q4k_scale_min_bytes` below, and the one place the Q3_K 6-bit layout is
// written — `unpack_q3k_scales` is this plus the loads.
//
// With `g = j / 4`:
//   low 4 bits  — byte `j % 4 + 4 * (g & 1)`, nibble `g / 2`
//   high 2 bits — byte `8 + j % 4`, bit pair `g`
//
// Matches `unpack_q3k_scales` in
// `src/quant/cpu/kernels/dequant_k_quants/q2k_q3k.rs`, which reaches the same
// 16 values as four u32 lanes, and `load_tiles_q3_K` in llama.cpp's
// `ggml-cuda/mmq.cuh`, which reaches them as four vectorized `ksc` groups.
#define GGUF_Q3K_SC_LOW_BYTE(j) ((j) % 4 + 4 * (((j) / 4) & 1))
#define GGUF_Q3K_SC_HIGH_BYTE(j) (8 + (j) % 4)

static __device__ __forceinline__ int q3k_scale_bytes(
    unsigned int b_low, unsigned int b_high, int j
) {
    const int g = j / 4;
    const int lo = (int)((b_low >> (4 * (g / 2))) & 0x0F);
    const int hi = (int)((b_high >> (2 * g)) & 0x03);
    return (lo | (hi << 4)) - 32;
}

static __device__ __forceinline__ void unpack_q3k_scales(
    const unsigned char* sc_raw,
    signed char* scales
) {
    for (int j = 0; j < 16; j++) {
        scales[j] = (signed char)q3k_scale_bytes(
            sc_raw[GGUF_Q3K_SC_LOW_BYTE(j)], sc_raw[GGUF_Q3K_SC_HIGH_BYTE(j)], j
        );
    }
}

// ── Q4_K single sub-block scale/min ──────────────────────────────────────
// Same 12-byte layout as unpack_q4k_q5k_scales, decoded for one sub-block
// `j` (0..7) instead of all eight — used by the monolithic MMQ/tiled-GEMM
// kernels that dequantize one sub-block at a time rather than a whole
// super-block up front.

// Register form of the same unpack, taking the three `sc` bytes it needs
// instead of the array: `b_j` is sc[j], `b_j4` is sc[j + 4], `b_jm4` is
// sc[j - 4] (read only for j >= 4). A staging loop that must issue every
// global load before any ALU work loads those bytes itself and decodes here,
// which is why the bit layout lives in this function and `q4k_scale_min`
// below is just this plus the three loads.
static __device__ __forceinline__ void q4k_scale_min_bytes(
    unsigned int b_j, unsigned int b_j4, unsigned int b_jm4, int j, int* scale, int* minimum
) {
    if (j < 4) {
        *scale = b_j & 63;
        *minimum = b_j4 & 63;
    } else {
        *scale = (b_j4 & 0x0F) | ((b_jm4 >> 6) << 4);
        *minimum = (b_j4 >> 4) | ((b_j >> 6) << 4);
    }
}

static __device__ __forceinline__ void q4k_scale_min(
    const unsigned char* sc, int j, int* scale, int* minimum
) {
    // For j < 4 the third byte is unused; reading sc[j] again keeps the index
    // inside the 12-byte array without a branch around the load.
    q4k_scale_min_bytes(sc[j], sc[j + 4], sc[j < 4 ? j : j - 4], j, scale, minimum);
}

// ── Split-half nibble order (4-bit formats over 32-element blocks) ───────
// llama.cpp `dequantize_row_q4_0` packs element `j` and element `j + 16` of a
// block into ONE byte:
//
//     y[i*qk + j + 0    ] = (qs[j] & 0x0F) * d;   // first half
//     y[i*qk + j + qk/2 ] = (qs[j] >>   4) * d;   // second half
//
// The two nibbles of a byte are 16 elements apart in the output, NOT adjacent.
// Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL and IQ4_XS all use it. Emitting them to
// consecutive positions permutes every weight inside the block and nothing
// errors: shape, block count and tensor RMS all stay correct. Q5_0/Q5_1 index
// their fifth-bit word `qh` with the SAME element index — bit `j` for the
// first half, bit `j + 16` for the second.
//
// `elem` is the element index within the 32-element block.
static __device__ __forceinline__ int gguf_split_half_nibble(
    const unsigned char* qs, int elem
) {
    const unsigned char byte = qs[elem & 15];
    return (elem & 16) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
}

// ── IQ4 non-linear codebook ────────────────────────────────────────────
// IQ4_NL and IQ4_XS are CODEBOOK quantizations: a nibble is an INDEX into
// this 16-entry signed table, never a magnitude. Reading a nibble as a
// magnitude yields finite, plausibly scaled numbers, so the mistake shows up
// as wrong output and never as an error.
//
// This is the ONE copy every kernel file that includes `decode.cuh` uses.
// `KVALUES_IQ4NL` in `src/quant/tables` is the CPU mirror; keep the two in
// step. Each translation unit here compiles to its own PTX module, so the
// `__constant__` definition in a header is per-module and never a duplicate
// symbol.
//
// `__align__(16)` is load-bearing: `gguf_iq4_table_lookup` reads the table as
// four `unsigned int`, and a bare `signed char[16]` carries alignment 1.
__constant__ __align__(16) signed char KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// Codebook lookup for the eight nibble indices packed in `q4`, which holds
// four consecutive `qs` bytes of a 32-element run.
//
// `*lo` receives the four LOW nibbles' table values and `*hi` the four HIGH
// nibbles', each as four signed bytes in one int. Under the split-half nibble
// order those are elements `j..j+3` and `j+16..j+19` of the run.
//
// Transcribed from `get_int_from_table_16` in llama.cpp's
// `ggml-cuda/vecdotq.cuh`. `__byte_perm` selects a byte with a THREE-bit
// index, so the fourth index bit is handled by permuting the low and high
// halves of the table separately and then selecting between the two results
// on that bit. `q4 & 0x77777777` masks each index to those three bits: a
// selector nibble with its msb set means "replicate the sign of the selected
// byte" to `prmt.b32`, not "select byte 8..15", so an unmasked index would
// return 0x00 or 0xFF for every entry in the table's upper half.
static __device__ __forceinline__ void gguf_iq4_table_lookup(
    int q4, int* lo, int* hi
) {
    const unsigned int* table32 = (const unsigned int*)KVALUES_IQ4NL;
    const unsigned int idx3 = (unsigned int)q4 & 0x77777777u;
    // Bit 3 of index `i` becomes bit 2 of selector nibble `i`, which picks the
    // upper-half result (`high`, pool bytes 4..7) over the lower (`low`).
    const unsigned int sel = 0x32103210u | (((unsigned int)q4 & 0x88888888u) >> 1);

    // Two rounds, because one `__byte_perm` consumes only four index nibbles.
    unsigned int tmp[2];
#pragma unroll
    for (unsigned int i = 0; i < 2; ++i) {
        const unsigned int shift = 16u * i;
        const unsigned int low = __byte_perm(table32[0], table32[1], idx3 >> shift);
        const unsigned int high = __byte_perm(table32[2], table32[3], idx3 >> shift);
        tmp[i] = __byte_perm(low, high, sel >> shift);
    }

    // `tmp` holds the eight values in nibble order (low, high, low, high...);
    // these two permutes split them into the all-low and all-high words.
    *lo = (int)__byte_perm(tmp[0], tmp[1], 0x6420u);
    *hi = (int)__byte_perm(tmp[0], tmp[1], 0x7531u);
}

// ── TQ1_0 / TQ2_0 ternary block layout ─────────────────────────────────
//
// Both ternary formats store the f16 scale `d` at the END of the block, not
// the start, and both order elements level-major rather than byte-major.
// Reading `d` from offset 0 yields a scale built from packed trits — a small
// denormal-ish number that keeps the tensor finite while every weight is
// wrong, which is exactly the error class CLAUDE.md warns about.
//
//   TQ1_0 (54B): qs[0..48], qh[48..52], d[52..54]
//   TQ2_0 (66B): qs[0..64],             d[64..66]
//
// TQ1_0 packs FIVE trits per byte in base 3, and does not decode them by
// repeated division. llama.cpp stores each byte pre-scaled so a trit is
// recovered by a wrapping 8-bit multiply against a power of three followed by
// a multiply-shift: `q = (uint8)(byte * pow3[l])`, `trit = ((uint16)q*3 >> 8)`.
// The multiply MUST wrap at 8 bits; widening it changes the result.
//
// The 256 elements come from three differently shaped runs, in this order:
//   [  0, 160)  qs[0..32]   x 5 levels, 32 per level
//   [160, 240)  qs[32..48]  x 5 levels, 16 per level
//   [240, 256)  qh[0..4]    x 4 levels,  4 per level

#define GGUF_TQ1_0_D_OFFSET 52
#define GGUF_TQ2_0_D_OFFSET 64

// Returns the ternary value {-1, 0, 1} of element `elem` (0..256) of a TQ1_0
// block. `block` points at the start of the 54-byte block.
static __device__ __forceinline__ int gguf_tq1_0_trit(
    const unsigned char* block, int elem
) {
    const unsigned char pow3[5] = { 1, 3, 9, 27, 81 };
    unsigned char byte;
    int level;
    if (elem < 160) {
        level = elem >> 5;
        byte = block[elem & 31];
    } else if (elem < 240) {
        const int r = elem - 160;
        level = r >> 4;
        byte = block[32 + (r & 15)];
    } else {
        const int r = elem - 240;
        level = r >> 2;
        byte = block[48 + (r & 3)];
    }
    const unsigned char q = (unsigned char)(byte * pow3[level]);
    return (int)(((unsigned short)q * 3) >> 8) - 1;
}

// Returns the ternary value {-1, 0, 1} of element `elem` (0..256) of a TQ2_0
// block. `block` points at the start of the 66-byte block.
static __device__ __forceinline__ int gguf_tq2_0_trit(
    const unsigned char* block, int elem
) {
    const int group = elem >> 7;         // 128 elements per 32-byte group
    const int r     = elem & 127;
    const int level = r >> 5;            // which 2-bit field
    const int m     = r & 31;            // which byte in the group
    return (int)((block[group * 32 + m] >> (2 * level)) & 0x03) - 1;
}
