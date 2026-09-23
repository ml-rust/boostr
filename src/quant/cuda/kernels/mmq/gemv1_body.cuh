// Single-token (M = 1) body of the feature-major MMQ family: Q8_0, Q4_0,
// Q4_1, Q5_0, Q5_1, IQ4_NL, IQ4_XS, PQ2_0, Q2_0, Q1_0, PTQ1_0, Q4_K, Q5_K,
// Q6_K, Q3_K and Q2_K. Included by `quant_mmq_gemv1.cu`; it is not a
// compilation unit of its own. The per-format readers are in
// `gemv1_formats.cuh`, `gemv1_legacy.cuh` and `gemv1_kquant.cuh`, and the
// per-chunk int dots and float epilogues in `gemv1_chunks.cuh`.
//
// CONTRACT: every output element receives the SAME float sequence the
// tensor-core kernels `quant_mmq_<fmt>_q8_1_mma*` form for it, so a decode
// step through this kernel is the same bits as row 0 of any batch through
// them. The tensor-core path, per 32-element chunk `b` of the K walk,
// computes the exact int32 dot `D` of the chunk's int8 weight lanes with the
// chunk's int8 activation lanes and applies the float form of the format's
// `mmqf_vec_dot_*` — for the Q8_0 row formats
//
//     acc += (float)D * da * dw;
//
// `da` the activation record's `half` scale for the chunk, `dw` the weight
// block's scale as f32. The formats with a minimum term or a per-16 scale
// take the three other forms `gemv1_chunks.cuh` lists. In every form: chunks in ascending k, one
// accumulator per output element starting at `0.0f`, and the K walk cut into
// the ranges `mmqf_split_range` gives, range partials summed in range order.
// Each expression is written in `gemv1_chunks.cuh` character for character
// as in its `mmqf_vec_dot_*`; the build compiles both files with the same
// flags, so the compiler contracts them the same way in both. Nothing
// reorders, refactors or fuses them by hand. Every scale reaches f32 through
// `__half2float` of the stored bits, as the staging and the `vec_dot`
// convert them.
//
// What `D` is: the exact int32 dot of the chunk's 32 elements with its 32
// int8 activation values, formed from the same words the format's `stage`
// writes into the tensor-core kernel's shared row
// (Q8_0 the raw quant bytes, the others through `lowbit_expand_code2x8`,
// `lowbit_expand_sign8` and `ptq1_0_expand4` in `../lowbit_dequant.cuh`),
// so word `i` of a chunk holds elements `4i .. 4i + 3` as signed bytes.
// Every step is an exact int, so the order of the `dp4a` steps and the
// pairing of the lanes do not matter, only that each product is formed
// once. A reader with `PAIRED_WORDS` returns its words out of element
// order and names, per word pair, the activation pair that goes with it; a
// reader with `CODES_PLUS_ONE` (PTQ1_0, through the level chain in
// `../lowbit_dequant.cuh`) returns `value + 1` as unsigned bytes, and the
// dot subtracts the chunk's exact activation sum from the record header,
// which is the same integer `D`.
//
// Lane map. One warp owns 8 output features; lane `l` is (feature slot
// `l / 4`, chunk slot `c = l % 4`). A step is one 256-k group, 8 chunks per
// feature, walked as two 4-chunk halves: in half `h` lane `c` forms the
// int32 dot or dots of chunk `4h + c` from the chunk's 32 activation bytes,
// read as two `int4` loads. Slot 0 of a feature is the OWNER: it gathers
// what slots 1..3 formed by warp shuffle (bit-exact moves), reads the four
// `da` of the half from the record header itself (one 16-byte load) and
// applies the four chunks' terms in ascending chunk order, half 0 then half
// 1. The other three lanes hold no float state. `gemv1_half_step` holds that
// step and what each epilogue gathers.
//
// STAGING. The weight stream per lane is 4 to 32 bytes per chunk at a
// 2-byte alignment inside rows 18 to 210 bytes apart, so lane-wise loads
// touch many sectors per warp instruction. Instead the block stages one
// 256-k group of its 32 rows through shared memory each step: the span of
// each row that holds the group (`SPAN` bytes, only 2-byte aligned whenever
// the block stride is odd in words) is covered by the aligned 16-byte windows that
// contain it, `nwin = ceil((o + span) / 16)` of them where `o` is the span
// start's offset inside its window, loaded as `uint4` with consecutive
// lanes on consecutive windows of one row, so a warp instruction touches
// `ceil(32 / W)` rows and few sectors. The group's activation records (two
// 144-byte records, 16-byte aligned) are staged the same way once per
// block. Double buffered: step `s + 1`'s windows are loaded into registers
// before step `s` computes, stored to the other buffer after, one
// `__syncthreads` per step. Plain loads, no `cp.async`, as the rest of the
// family. The last window of a row can extend up to 15 bytes past the span
// and, on the last row, past the weight's last byte. A window is a 16-byte
// aligned unit that holds at least one valid byte, so it never crosses a
// page boundary (pages are 16-byte multiples): the bytes read past the end
// sit in the page that holds the last valid byte and are mapped. They are
// never consumed, since every chunk past the row ends is excluded by the
// range guard.
//
// Shared memory per block: `2 * 32 * ROW_WORDS * 4` bytes of weight windows
// plus `2 * 288` bytes of activation records. `ROW_WORDS` follows the span,
// so three sizes cover the family: 10,816 bytes at ROW_WORDS 40 (the lowbit
// formats, Q3_K and Q2_K), 19,008 at ROW_WORDS 72 (the legacy 32-element
// formats, IQ4_XS and the 4-, 5- and 6-bit K-quants) and 27,200 at
// ROW_WORDS 104 (Q8_0).
//
// BANKS. A lane reads its chunk as 32-bit words from the row's staged span,
// so at one instruction the 4 lanes of a feature hit at most four of eight
// consecutive banks — words `{0, 2, 4, 6} + k` for the 8-byte chunks, and
// two addresses broadcast over four lanes for the readers whose chunk pairs
// share one packed run. Eight rows per warp must then land on the
// eight offsets `{0, 8, 16, 24}` and `{1, 9, 17, 25}`: row `r` sits at word
// `r * ROW_WORDS + ((r / 4) & 1)` with `ROW_WORDS = 8 (mod 32)`, which is
// conflict free when every row's span starts at the same offset inside
// its window (the case whenever the row stride is a multiple of 16, as at
// K = 5120 for every format here) and at most 2-way when it does not, the
// per-row window offset shifting a row's set by up to four banks. Q8_0's
// 34-byte chunks are 8.5 words apart, which no base pattern separates
// fully: 2-way at worst, on the format that is bandwidth bound anyway. The
// odd word base rules out 128-bit shared stores, so a window is stored as
// four 32-bit words. Activation reads are two 16-byte loads per chunk slot,
// 32 bytes apart, the same addresses for all 8 features: broadcast,
// conflict free.
//
// Ragged K. K is a whole number of the format's blocks, which for the
// 32- and 64-element formats is finer than a 256-k group, so the last range
// can end mid-group. A 256-element format's block IS the group, so its walk
// is never ragged. The chunk count is
// `K / 32`, the same `bpr` the tensor-core walk stops at. A ragged last
// group loads only the windows inside the row's `bpr * BLOCK_BYTES /
// CHUNKS` bytes and only the records that exist; chunks past the range end
// read stale shared words and are excluded from the sum by the `b0 + j <
// kb1` guard.
//
// Grid: `x` over 32-feature blocks (4 warps), `y` over the K ranges; the
// host passes the split count the tensor-core launch would use, so the
// ranges are the same. Range 0 stores its partial to the output; every later
// range writes to `workspace[s - 1][N]`, and `gemv1_fixup_body` adds the
// partials in range order onto the range-0 store, the order `mmqf_ms_body`
// and `mmqf_fixup_body` use. With one range there is no workspace.
#pragma once

#include <cuda_fp16.h>

#include "../gemv/common.cuh"
#include "gemv1_chunks.cuh"

// Threads per block: four warps, 8 features each.
#define GEMV1_WARPS 4
#define GEMV1_THREADS (GEMV1_WARPS * WARP_SIZE)
#define GEMV1_FEATS_PER_WARP 8
#define GEMV1_FEATS_PER_BLOCK (GEMV1_WARPS * GEMV1_FEATS_PER_WARP)
// Threads per block of the fixup pass.
#define GEMV1_FIXUP_THREADS 256

// Windows that cover a `span`-byte run starting anywhere inside a window.
#define GEMV1_WINDOWS_OF(span) (((span) + 15 + 15) / 16)
// Row stride in the staged tile: the smallest `8 (mod 32)` word count
// holding `W` windows plus the one-word bank offset.
#define GEMV1_ROW_WORDS_OF(W) ((((4 * (W) + 1) - 8 + 31) / 32) * 32 + 8)

// Word offset of staged row `r` (0..31): see BANKS in the header.
template <int ROW_WORDS>
static __device__ __forceinline__ unsigned int gemv1_row_base(unsigned int r) {
    return r * ROW_WORDS + ((r >> 2) & 1u);
}

// The windows one thread carries between the load and the store of a step.
template <int UW>
struct Gemv1Windows {
    uint4 w[UW];
    uint4 y;
    // Bit `u` set when `w[u]` was loaded; bit 31 for `y`.
    unsigned int mask;
};

// Where one thread loads from, fixed for the K walk: thread `t` takes
// windows `t, t + 128, ...` of the row-major (row, window) index of the
// block's 32 rows. Per window `u`: the row's global pointer (rows past N
// read row `N - 1`) and the window's byte offset inside the span's first
// window; a window past the 32 rows gets an offset no span reaches, so its
// load never issues. Built once by `gemv1_load_map`.
template <int UW>
struct Gemv1LoadMap {
    const unsigned char* row[UW];
    unsigned int win[UW];
};

template <int W, int UW>
static __device__ __forceinline__ Gemv1LoadMap<UW> gemv1_load_map(
    const unsigned char* __restrict__ weight, unsigned long long rstride, unsigned int N,
    unsigned int feat0
) {
    Gemv1LoadMap<UW> m;
    const unsigned int t = threadIdx.x;
#pragma unroll
    for (int u = 0; u < UW; ++u) {
        const unsigned int i = t + u * GEMV1_THREADS;
        const unsigned int r = i / W;
        const unsigned int wi = i % W;
        const unsigned int f = min(feat0 + min(r, GEMV1_FEATS_PER_BLOCK - 1u), N - 1);
        m.row[u] = weight + (unsigned long long)f * rstride;
        m.win[u] = r < GEMV1_FEATS_PER_BLOCK ? 16 * wi : 0x80000000u;
    }
    return m;
}

// Issues the loads of the group starting at chunk `b0`: the block's row
// windows per `Gemv1LoadMap` and the group's records. Windows past a row's
// end and records past the last are skipped and left stale, which the
// guards below never read.
template <class FMT, int W, int UW>
static __device__ __forceinline__ void gemv1_load(
    const Gemv1LoadMap<UW>& m, const int* __restrict__ y_packed,
    unsigned long long row_len, unsigned long long group_stride,
    unsigned int kgroups, unsigned int b0, Gemv1Windows<UW>& win
) {
    const unsigned int t = threadIdx.x;
    const unsigned long long start = (unsigned long long)(b0 / FMT::CHUNKS) * FMT::BLOCK_BYTES;
    const unsigned long long stop = min(start + FMT::SPAN, row_len);
    win.mask = 0;
#pragma unroll
    for (int u = 0; u < UW; ++u) {
        const unsigned char* span = m.row[u] + start;
        // Pointer arithmetic throughout, so the load stays a global
        // (`LDG`) load rather than a generic one.
        const unsigned int o = (unsigned int)((unsigned long long)span & 15ull);
        const unsigned char* at = span - o + m.win[u];
        if (at < m.row[u] + stop) {
            win.w[u] = __ldg(reinterpret_cast<const uint4*>(at));
            win.mask |= 1u << u;
        }
    }
    if (t < GEMV1_Y_WINDOWS) {
        const unsigned int g = b0 / 4 + t / (GEMV1_Y_WINDOWS / 2);
        if (g < kgroups) {
            const int* rec = y_packed + (unsigned long long)g * group_stride +
                             (t % (GEMV1_Y_WINDOWS / 2)) * 4;
            win.y = __ldg(reinterpret_cast<const uint4*>(rec));
            win.mask |= 1u << 31;
        }
    }
}

// Stores the carried windows into buffer `s_w` / `s_y`. A row's base can be
// an odd word, so a window is four 32-bit stores.
template <int W, int UW, int ROW_WORDS>
static __device__ __forceinline__ void gemv1_store(
    const Gemv1Windows<UW>& win, int* __restrict__ s_w, int* __restrict__ s_y
) {
    const unsigned int t = threadIdx.x;
#pragma unroll
    for (int u = 0; u < UW; ++u) {
        if (win.mask & (1u << u)) {
            const unsigned int i = t + u * GEMV1_THREADS;
            int* dst = s_w + gemv1_row_base<ROW_WORDS>(i / W) + 4 * (i % W);
            dst[0] = (int)win.w[u].x;
            dst[1] = (int)win.w[u].y;
            dst[2] = (int)win.w[u].z;
            dst[3] = (int)win.w[u].w;
        }
    }
    if (win.mask & (1u << 31)) {
        *reinterpret_cast<uint4*>(s_y + 4 * t) = win.y;
    }
}

// The K walk of one range for 8 features per warp. `y_packed` is the
// activation in the feature-major record layout with token stride `ntok`
// records per 128-k group; token 0 is the one token. `workspace` is read
// only by ranges past the first.
template <class FMT>
static __device__ __forceinline__ void gemv1_body(
    const int* __restrict__ y_packed,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    float* __restrict__ workspace,
    unsigned int K, unsigned int N, unsigned int ntok
) {
    constexpr int W = GEMV1_WINDOWS_OF(FMT::SPAN);
    constexpr int UW = (GEMV1_FEATS_PER_BLOCK * W + GEMV1_THREADS - 1) / GEMV1_THREADS;
    constexpr int ROW_WORDS = GEMV1_ROW_WORDS_OF(W);
    static_assert(ROW_WORDS % 32 == 8 && ROW_WORDS >= 4 * W + 1, "Row stride breaks the bank map.");
    constexpr int TILE_WORDS = GEMV1_FEATS_PER_BLOCK * ROW_WORDS;

    __shared__ __align__(16) int s_w[2][TILE_WORDS];
    __shared__ __align__(16) int s_y[2][GEMV1_Y_WORDS];

    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int slot = lane / 4;  // feature slot within the warp
    const unsigned int c = lane % 4;     // chunk slot within a half step
    const unsigned int r = warp * GEMV1_FEATS_PER_WARP + slot;  // staged row
    const unsigned int feat0 = blockIdx.x * GEMV1_FEATS_PER_BLOCK;
    const unsigned int f = feat0 + r;
    const bool live = f < N;

    // Activation blocks per row: 32 elements each, the unit the split
    // ranges and the tensor-core k-steps are counted in.
    const unsigned int bpr = K / 32;
    const unsigned int kgroups = (bpr + 3) / 4;
    const unsigned int splits = gridDim.y;
    const unsigned int s = blockIdx.y;
    unsigned int kb0, kb1;
    mmqf_split_range(bpr, splits, s, kb0, kb1);

    const unsigned long long rstride = (unsigned long long)(bpr / FMT::CHUNKS) * FMT::BLOCK_BYTES;
    const unsigned long long row_len = rstride;
    // Ints from one 128-k group's token-0 record to the next.
    const unsigned long long group_stride = (unsigned long long)ntok * GEMV1_Y_STRIDE;
    // This lane's row in global memory, for the span offset per step.
    const unsigned char* row = weight + (unsigned long long)(live ? f : N - 1) * rstride;
    const int* my_row0 = s_w[0] + gemv1_row_base<ROW_WORDS>(r);
    const int* my_row1 = s_w[1] + gemv1_row_base<ROW_WORDS>(r);
    const unsigned int base = lane & ~3u;
    const typename FMT::Lane ln = FMT::setup(c);

    const Gemv1LoadMap<UW> lmap = gemv1_load_map<W, UW>(weight, rstride, N, feat0);
    Gemv1Windows<UW> win;
    gemv1_load<FMT, W, UW>(lmap, y_packed, row_len, group_stride, kgroups, kb0, win);
    gemv1_store<W, UW, ROW_WORDS>(win, s_w[0], s_y[0]);
    __syncthreads();

    float acc = 0.0f;
    unsigned int buf = 0;
    // `kb0` is a multiple of MMQF_ITER_B, so every step is one whole group
    // and its first chunk opens a record.
    for (unsigned int b0 = kb0; b0 < kb1; b0 += GEMV1_STEP_CHUNKS, buf ^= 1) {
        const bool has_next = b0 + GEMV1_STEP_CHUNKS < kb1;
        if (has_next) {
            gemv1_load<FMT, W, UW>(lmap, y_packed, row_len, group_stride, kgroups,
                                   b0 + GEMV1_STEP_CHUNKS, win);
        }

        const int* my_row = buf ? my_row1 : my_row0;
        const int* y = s_y[buf];
        // The span start's offset inside its window, for this lane's row.
        const unsigned long long start =
            (unsigned long long)(b0 / FMT::CHUNKS) * FMT::BLOCK_BYTES;
        const unsigned int o = (unsigned int)((unsigned long long)(row + start) & 15ull);

#pragma unroll
        for (unsigned int h = 0; h < 2; ++h) {
            gemv1_half_step<FMT>(my_row, o, y + h * GEMV1_Y_STRIDE, ln, 4 * h, c, base, b0,
                                 kb1, acc);
        }

        if (has_next) {
            gemv1_store<W, UW, ROW_WORDS>(win, s_w[buf ^ 1], s_y[buf ^ 1]);
        }
        __syncthreads();
    }

    if (c != 0 || !live) {
        return;
    }
    if (s == 0) {
        output[f] = acc;
    } else {
        workspace[(unsigned long long)(s - 1) * N + f] = acc;
    }
}

// Adds the range partials onto the range-0 store, in range order:
// `((p0 + p1) + p2) + ...`, the sequence `mmqf_ms_body` and
// `mmqf_fixup_body` form. One thread per output feature.
static __device__ __forceinline__ void gemv1_fixup_body(
    float* __restrict__ output, const float* __restrict__ workspace, unsigned int N,
    unsigned int splits
) {
    const unsigned int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= N || splits < 2) {
        return;
    }
    float o = output[f];
    for (unsigned int s = 1; s < splits; ++s) {
        o += workspace[(unsigned long long)(s - 1) * N + f];
    }
    output[f] = o;
}
