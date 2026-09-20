// Single-token (M = 1) body of the feature-major MMQ family, for the
// formats `mmqf_vec_dot_d` serves as a Q8_0 row: Q8_0, PQ2_0, Q2_0, Q1_0 and
// PTQ1_0. Included by `quant_mmq_gemv1.cu`; it is not a compilation unit of
// its own. The per-format readers are in `gemv1_formats.cuh`.
//
// CONTRACT: every output element receives the SAME float sequence the
// tensor-core kernels `quant_mmq_<fmt>_q8_1_mma*` form for it, so a decode
// step through this kernel is the same bits as row 0 of any batch through
// them. The tensor-core path, per 32-element chunk `b` of the K walk,
// computes the exact int32 dot `D` of the chunk's int8 weight lanes with the
// chunk's int8 activation lanes and applies
//
//     acc += (float)D * da * dw;
//
// `da` the activation record's `half` scale for the chunk, `dw` the weight
// block's `half` scale as f32, chunks in ascending k, one accumulator per
// output element starting at `0.0f`, and the K walk cut into the ranges
// `mmqf_split_range` gives, range partials summed in range order. The
// expression is written here character for character as in
// `mmqf_vec_dot_d`; the build compiles both files with the same flags, so
// the compiler contracts it the same way in both (a multiply, then one fused
// multiply-add onto `acc`). Nothing below reorders, refactors or fuses it by
// hand. Both scales reach f32 through `__half2float` of the stored bits, as
// the staging and `mmqf_vec_dot_d` convert them.
//
// What `D` is: the exact int32 dot of the chunk's 32 signed elements with
// its 32 int8 activation values, formed from the same words `MmqfQ80::stage`,
// `MmqfLowbit::stage` and `MmqfPTQ10::stage` write into the shared row
// (Q8_0 the raw quant bytes, the others through `lowbit_expand_code2x8`,
// `lowbit_expand_sign8` and `ptq1_0_expand4` in `../lowbit_dequant.cuh`),
// so word `i` of a chunk holds elements `4i .. 4i + 3` as signed bytes.
// Every step is an exact int, so the order of the `dp4a` steps and the
// pairing of the lanes do not matter, only that each product is formed
// once.
//
// Lane map. One warp owns 8 output features; lane `l` is (feature slot
// `l / 4`, chunk slot `c = l % 4`). A step is one 256-k group, 8 chunks per
// feature, walked as two 4-chunk halves: in half `h` lane `c` forms the
// int32 dot of chunk `4h + c` from the chunk's 32 activation bytes, read as
// two `int4` loads. Slot 0 of a feature is the OWNER: it gathers the int
// dots of slots 1..3 by warp shuffle (bit-exact moves), reads the four `da`
// of the half from the record header itself (one 16-byte load) and the four
// `dw` from the staged row, and applies the four chunks' terms in ascending
// chunk order, half 0 then half 1. When a block spans 4 chunks the half's
// four `dw` are one scale, read once. The other three lanes hold no float
// state.
//
// STAGING. The weight stream per lane is 4 to 32 bytes per chunk at a
// 2-byte alignment inside rows 18 to 34 bytes apart, so lane-wise loads
// touch many sectors per warp instruction. Instead the block stages one
// 256-k group of its 32 rows through shared memory each step: the span of
// each row that holds the group (`SPAN` bytes, 2-byte aligned for the 34-
// and 18-byte formats) is covered by the aligned 16-byte windows that
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
// Shared memory per block: `2 * 32 * ROW_WORDS * 4` bytes of weight
// windows plus `2 * 288` bytes of activation records — 10,816 bytes for the
// lowbit formats (ROW_WORDS 40) and 27,200 for Q8_0 (ROW_WORDS 104).
//
// BANKS. A lane reads its chunk as 32-bit words at stride `CHUNK_BYTES`
// from the row's staged span, so at one instruction the 4 lanes of a
// feature hit words `{0, 2, 4, 6} + k` (8-byte chunks) of their row: four
// of eight consecutive banks. Eight rows per warp must then land on the
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
// Ragged K. K is a whole number of the format's blocks, which is finer than
// a 256-k group, so the last range can end mid-group. The chunk count is
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
#include "gemv1_formats.cuh"

// Threads per block: four warps, 8 features each.
#define GEMV1_WARPS 4
#define GEMV1_THREADS (GEMV1_WARPS * WARP_SIZE)
#define GEMV1_FEATS_PER_WARP 8
#define GEMV1_FEATS_PER_BLOCK (GEMV1_WARPS * GEMV1_FEATS_PER_WARP)
// Threads per block of the fixup pass.
#define GEMV1_FIXUP_THREADS 256

// Activation record: 4 header words, then 32 quant words (128 k-values);
// `MMQF_Y_DS`, `MMQF_Y_QS` and `MMQF_Y_STRIDE` in `quant_mmq_mma.cu`. A
// step stages two records: 72 words, 18 windows.
#define GEMV1_Y_QS 4
#define GEMV1_Y_STRIDE 36
#define GEMV1_Y_WORDS (2 * GEMV1_Y_STRIDE)
#define GEMV1_Y_WINDOWS (GEMV1_Y_WORDS / 4)

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

// Issues the loads of the group starting at chunk `b0`: the windows of the
// block's 32 rows (thread `t` takes windows `t, t + 128, ...` of the
// row-major (row, window) index) and the group's records. Rows past N read
// row `N - 1`; windows past a row's end and records past the last are
// skipped and left stale, which the guards below never read.
template <class FMT, int W, int UW>
static __device__ __forceinline__ void gemv1_load(
    const unsigned char* __restrict__ weight, const int* __restrict__ y_packed,
    unsigned long long rstride, unsigned long long row_len, unsigned long long group_stride,
    unsigned int kgroups, unsigned int N, unsigned int feat0, unsigned int b0,
    Gemv1Windows<UW>& win
) {
    const unsigned int t = threadIdx.x;
    const unsigned long long start = (unsigned long long)(b0 / FMT::CHUNKS) * FMT::BLOCK_BYTES;
    const unsigned long long stop = min(start + FMT::SPAN, row_len);
    win.mask = 0;
#pragma unroll
    for (int u = 0; u < UW; ++u) {
        const unsigned int i = t + u * GEMV1_THREADS;
        const unsigned int r = i / W;
        const unsigned int wi = i % W;
        if (r < GEMV1_FEATS_PER_BLOCK) {
            const unsigned char* row = weight + (unsigned long long)min(feat0 + r, N - 1) * rstride;
            const unsigned char* span = row + start;
            // Pointer arithmetic throughout, so the load stays a global
            // (`LDG`) load rather than a generic one.
            const unsigned int o = (unsigned int)((unsigned long long)span & 15ull);
            const unsigned char* at = span - o + 16 * wi;
            if (at < row + stop) {
                win.w[u] = __ldg(reinterpret_cast<const uint4*>(at));
                win.mask |= 1u << u;
            }
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

    Gemv1Windows<UW> win;
    gemv1_load<FMT, W, UW>(weight, y_packed, rstride, row_len, group_stride, kgroups, N, feat0,
                           kb0, win);
    gemv1_store<W, UW, ROW_WORDS>(win, s_w[0], s_y[0]);
    __syncthreads();

    float acc = 0.0f;
    unsigned int buf = 0;
    // `kb0` is a multiple of MMQF_ITER_B, so every step is one whole group
    // and its first chunk opens a record.
    for (unsigned int b0 = kb0; b0 < kb1; b0 += GEMV1_STEP_CHUNKS, buf ^= 1) {
        const bool has_next = b0 + GEMV1_STEP_CHUNKS < kb1;
        if (has_next) {
            gemv1_load<FMT, W, UW>(weight, y_packed, rstride, row_len, group_stride, kgroups, N,
                                   feat0, b0 + GEMV1_STEP_CHUNKS, win);
        }

        const int* my_row = buf ? my_row1 : my_row0;
        const int* y = s_y[buf];
        // The span start's offset inside its window, for this lane's row.
        const unsigned long long start =
            (unsigned long long)(b0 / FMT::CHUNKS) * FMT::BLOCK_BYTES;
        const unsigned int o = (unsigned int)((unsigned long long)(row + start) & 15ull);

#pragma unroll
        for (unsigned int h = 0; h < 2; ++h) {
            const unsigned int j = 4 * h + c;
            int w[8];
            FMT::read(my_row, o, j, w);
            const int* rec = y + h * GEMV1_Y_STRIDE;
            int dot = 0;
            // The chunk's 32 activation bytes: two 16-byte loads.
            const int4* q4 = reinterpret_cast<const int4*>(rec + GEMV1_Y_QS + c * 8);
            const int4 q0 = q4[0];
            const int4 q1 = q4[1];
            dot = dp4a(w[0], q0.x, dot);
            dot = dp4a(w[1], q0.y, dot);
            dot = dp4a(w[2], q0.z, dot);
            dot = dp4a(w[3], q0.w, dot);
            dot = dp4a(w[4], q1.x, dot);
            dot = dp4a(w[5], q1.y, dot);
            dot = dp4a(w[6], q1.z, dot);
            dot = dp4a(w[7], q1.w, dot);

            // The owner gathers the dots of slots 1..3. Every lane issues
            // the shuffles; only the owner reads the results.
            int dots[4];
            dots[0] = dot;
#pragma unroll
            for (int jj = 1; jj < 4; ++jj) {
                dots[jj] = __shfl_sync(0xFFFFFFFFu, dot, base + jj);
            }
            if (c == 0) {
                // The four `da` of the half: the low halves of the record's
                // four header words, one 16-byte load. The four `dw` are one
                // block's scale when a block spans 4 chunks. Both are read
                // before the range guard; a chunk past the range end reads
                // a stale word its guard then drops.
                const int4 hdr = *reinterpret_cast<const int4*>(rec);
                const int hdrs[4] = {hdr.x, hdr.y, hdr.z, hdr.w};
                float dws[4];
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    dws[jj] = (FMT::CHUNKS == 4 && jj > 0) ? dws[0]
                                                           : FMT::scale(my_row, o, 4 * h + jj);
                }
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    if (b0 + 4 * h + jj < kb1) {
                        const float da =
                            gemv1_half_bits_to_float((unsigned short)(hdrs[jj] & 0xFFFF));
                        // `mmqf_vec_dot_d`'s term, character for character:
                        // see the header.
                        acc += (float)dots[jj] * da * dws[jj];
                    }
                }
            }
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
