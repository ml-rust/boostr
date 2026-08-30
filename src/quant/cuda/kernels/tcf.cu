// CUDA kernels for TCF native quantized weights: dequantization, GEMV, GEMM.
//
// # The access pattern, and why plane-major is not the problem it looks like
//
// TCF packs whole planes over the whole tensor: every tile's codes, then every
// group's scales, then minima, then the per-super-block values. A group's
// parameters therefore live in a different region from its codes, and a naive
// per-element kernel would issue two to five scattered global loads per weight.
//
// Every kernel below avoids that the same way, in two phases per unit of work:
//
//   1. Resolve group parameters ONCE per (tile, group) — at most four per tile
//      — into registers or shared memory. That is one load per 16, 32, or 64
//      weights, not one per weight, and the loads a warp issues in this phase
//      are consecutive entries of one plane.
//   2. Stream the code plane. A tile's codes are one contiguous run, and
//      consecutive tiles are adjacent, so a whole weight row's codes are a
//      single contiguous byte range. Lanes read consecutive addresses and the
//      loads coalesce.
//
// The code plane is in fact FRIENDLIER than a GGUF block layout here: GGUF
// interleaves a 2-byte scale into the code stream every 16 or 32 weights, so a
// warp reading 32 consecutive codes straddles scale bytes and lands on
// unaligned offsets. TCF's code plane is a dense array with natural alignment.
// What plane-major costs is concurrent read STREAMS — two for a flat symmetric
// encoding, up to five for the two-level asymmetric one — not coalescing.
//
// Coalescing is necessary and was not sufficient. The GEMV's first form read
// consecutive bytes and still reached a fifth of this card's bandwidth,
// because coalesced 32-byte warp loads and a scale resolution every 64
// elements both cost per ELEMENT. Both were fixed by widening the unit of
// work, not by reordering the payload; see the GEMV section for the numbers.
//
// CONFORMANCE.md Section 8.1 leaves code-plane ordering unfrozen, to be settled
// by benchmark. These kernels are one input to that: they show plane-major
// imposes no structural penalty on a GPU decode, only extra address streams.
//
// The 6-bit encodings are the one exception worth naming. Section 14.2 splits
// their code plane into a low-nibble sub-plane and a high-two-bit sub-plane, so
// a 6-bit tile costs two code streams instead of one. Interleaving those two
// sub-planes per tile would halve that, and is the concrete change Section 8.1
// would be measuring.

#include "tcf.cuh"

#define TCF_WARP_SIZE 32u
#define TCF_WARPS_PER_BLOCK 8u
#define TCF_GEMV_BLOCK (TCF_WARP_SIZE * TCF_WARPS_PER_BLOCK)

// Tiles one dequantization block owns: one super-block, so a block's group
// parameters come from one bit-packed sub-plane run.
#define TCF_DEQUANT_TILES TCF_SUPER_BLOCK_TILES
#define TCF_DEQUANT_BLOCK (TCF_DEQUANT_TILES * TCF_TILE)

// GEMM block tile: 32 activation rows by 32 weight rows, staged 32 elements of
// K at a time. 64 threads per K range, each owning a 4x4 register patch.
#define TCF_GEMM_BM 32u
#define TCF_GEMM_BN 32u
#define TCF_GEMM_BK 32u
#define TCF_GEMM_TM 4u
#define TCF_GEMM_TN 4u
#define TCF_GEMM_THREADS \
    ((TCF_GEMM_BM / TCF_GEMM_TM) * (TCF_GEMM_BN / TCF_GEMM_TN))
// Independent K ranges one block may split its work into, a thread group of
// `TCF_GEMM_THREADS` each. The host picks the count per shape and sizes the
// dynamic shared memory to match; this is only the ceiling.
#define TCF_GEMM_MAX_SPLIT 4u
#define TCF_GEMM_MAX_BLOCK (TCF_GEMM_THREADS * TCF_GEMM_MAX_SPLIT)
// Floats one K slice stages: its activation rows then its weight rows. The
// host multiplies this by the split it chose to size the dynamic shared
// memory, so `GEMM_SLICE_BYTES` in `quant/cuda/tcf/launch.rs` restates it.
#define TCF_GEMM_SLICE_FLOATS \
    (TCF_GEMM_BK * (TCF_GEMM_ASTRIDE + TCF_GEMM_WSTRIDE))
// Threads spanning the N edge of the block tile; the rest span the M edge.
#define TCF_GEMM_TX (TCF_GEMM_BN / TCF_GEMM_TN)
// Row stride of one staged K row. Padded by FOUR, not one: a `float4` read
// needs a 16-byte-aligned start, so the pad has to keep the stride a multiple
// of four floats. 36 is odd in units of float4, so consecutive K rows start on
// different bank quartets.
#define TCF_GEMM_ASTRIDE (TCF_GEMM_BM + 4u)
#define TCF_GEMM_WSTRIDE (TCF_GEMM_BN + 4u)
// Codes one thread decodes per staging step: one `tcf_code_run` word group.
#define TCF_GEMM_CHUNK TCF_RUN_PER_LANE
// Activation values one thread stages per step.
#define TCF_GEMM_ALOADS ((TCF_GEMM_BM * TCF_GEMM_BK) / TCF_GEMM_THREADS)
// Largest group count a tile can carry (group 16 over a 64-element tile).
#define TCF_MAX_GROUPS 4u

static __device__ __forceinline__ float tcf_warp_reduce_sum(float acc) {
#pragma unroll
    for (unsigned int offset = TCF_WARP_SIZE / 2u; offset > 0u; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }
    return acc;
}

// The whole-run part of one warp's GEMV dot product over weight row `col`:
// `runs` steps of `TCF_RUN_TILES` tiles, sixteen elements per lane per step.
//
// `ActWide` is the caller's alignment finding, not a hint. It is a template
// parameter rather than a branch inside the loop because the branch cost 10%
// of the kernel; the two instantiations differ only in how four activation
// floats are fetched. Both are warp-uniform, so every lane reaches the
// `__shfl_sync` below and the full-mask shuffles are legal.
template <bool ActWide>
static __device__ __forceinline__ float tcf_gemv_runs(
    const float* __restrict__ act_row,
    const unsigned char* __restrict__ weight,
    TcfLayout l,
    unsigned int col,
    unsigned int tiles_per_row,
    unsigned int runs,
    unsigned int lane
) {
    const unsigned int resolved = TCF_RUN_TILES * l.groups_per_tile;
    const unsigned int e0 = TCF_RUN_PER_LANE * lane;
    // Sixteen elements from a multiple of sixteen lie in one group at every
    // group width v1 defines, so one broadcast serves the lane's whole run.
    const unsigned int slot = e0 / l.group;
    float acc = 0.0f;

    for (unsigned int r = 0; r < runs; ++r) {
        const unsigned int tile0 = col * tiles_per_row + r * TCF_RUN_TILES;
        float scale = 0.0f;
        float min_value = 0.0f;
        if (lane < resolved) {
            tcf_group_values(weight, l, tile0 + lane / l.groups_per_tile,
                             lane % l.groups_per_tile, &scale, &min_value);
        }
        const float s = __shfl_sync(0xFFFFFFFFu, scale, slot);
        const float mn = __shfl_sync(0xFFFFFFFFu, min_value, slot);
        const TcfCodeRun run = tcf_code_run(weight, l, tile0, lane);
        const float* act = act_row + (size_t)r * (size_t)TCF_RUN + (size_t)e0;

#pragma unroll
        for (unsigned int c = 0; c < TCF_RUN_PER_LANE / 4u; ++c) {
            float a[4];
            if (ActWide) {
                const float4 v = *(const float4*)(act + c * 4u);
                a[0] = v.x;
                a[1] = v.y;
                a[2] = v.z;
                a[3] = v.w;
            } else {
#pragma unroll
                for (unsigned int k = 0; k < 4u; ++k) {
                    a[k] = act[c * 4u + k];
                }
            }
#pragma unroll
            for (unsigned int k = 0; k < 4u; ++k) {
                const int code = tcf_run_code(run, l, c * 4u + k);
                acc += a[k] * tcf_value(code, s, mn, l.symmetric);
            }
        }
    }
    return acc;
}

extern "C" {

// ============================================================================
// Dequantization: whole payload -> f32, in logical row-major order.
//
// One block per super-block of four tiles, 256 threads, one element each.
// Threads 0..(4 * groups_per_tile) resolve the block's group parameters into
// shared memory first, so the scale planes are read once per group rather than
// once per weight.
// ============================================================================

__global__ __launch_bounds__(TCF_DEQUANT_BLOCK, 1) void tcf_dequant_f32(
    const unsigned char* __restrict__ payload,
    float* __restrict__ output,
    unsigned int tiles,
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
    const TcfLayout l = tcf_layout(code_high_off, scale_off, min_off, super_off,
                                   super_min_off, bits, group, groups_per_tile,
                                   symmetric, scale_form, sub_block_bytes);

    __shared__ float s_scale[TCF_DEQUANT_TILES * TCF_MAX_GROUPS];
    __shared__ float s_min[TCF_DEQUANT_TILES * TCF_MAX_GROUPS];

    const unsigned int tile_base = blockIdx.x * TCF_DEQUANT_TILES;
    const unsigned int tid = threadIdx.x;
    const unsigned int resolved = TCF_DEQUANT_TILES * l.groups_per_tile;

    if (tid < resolved) {
        const unsigned int local_tile = tid / l.groups_per_tile;
        const unsigned int g = tid % l.groups_per_tile;
        const unsigned int tile = tile_base + local_tile;
        float scale = 0.0f;
        float min_value = 0.0f;
        if (tile < tiles) {
            tcf_group_values(payload, l, tile, g, &scale, &min_value);
        }
        s_scale[tid] = scale;
        s_min[tid] = min_value;
    }
    __syncthreads();

    const unsigned int local_tile = tid / TCF_TILE;
    const unsigned int e = tid % TCF_TILE;
    const unsigned int tile = tile_base + local_tile;
    if (tile >= tiles) {
        return;
    }

    const unsigned int slot = local_tile * l.groups_per_tile + (e / l.group);
    const int code = tcf_code(payload, l, tile, e);
    output[(size_t)tile * (size_t)TCF_TILE + (size_t)e] =
        tcf_value(code, s_scale[slot], s_min[slot], l.symmetric);
}

// ============================================================================
// GEMV: activation [M, K] x weight [N, K]^T -> output [M, N], for a small M.
//
// One warp per output column, eight warps per block. A warp owns weight row
// `col`, whose tiles are `col * tiles_per_row ..` — a contiguous run of the
// code plane — and walks it EIGHT TILES AT A TIME, `TCF_RUN_TILES`.
//
// # Why eight tiles and not one
//
// A tile-at-a-time loop gave 20% of this card's bandwidth and got SLOWER as
// the encoding got narrower, which is the signature of a cost tracking element
// count rather than byte count. Two causes, measured separately on an RTX 3060
// with `q_proj` 2048x2048 at M = 1:
//
//   - Narrow code loads. `tcf_code` issues one byte load per element, so a
//     warp covered 32 bytes per instruction at 8 bits and 16 at 4 bits.
//     `tcf_code_run` reads the same codes as one `uint4` or `uint2` per lane:
//     512 or 256 consecutive bytes per instruction.
//   - Scale resolution once per TILE. Holding scale and minimum constant cut
//     the old kernel's time roughly in half at every width, and the whole of
//     the 4-bit deficit was here: `TwoLevelU6M6` costs four scattered plane
//     reads, and paying them every 64 elements dominated everything else.
//     A run pays them every 512 elements instead, and resolves the run's
//     `8 * groups_per_tile <= 32` groups in ONE warp-wide step.
//
// Together: 62.3 -> 19.5 us at 8 bits, 72.9 -> 21.5 at 6, 86.9 -> 21.5 at 4.
// The 8-bit case then sits within 4 us of a kernel that only streams the code
// plane and discards it, so it is at the memory system's limit.
//
// Sixteen elements per lane is one whole quantization group at every group
// width v1 defines, so a lane broadcasts one `(scale, minimum)` pair for its
// whole run. See `tcf_code_run` for that and for the alignment argument.
//
// A row whose tile count is not a multiple of eight finishes on the
// tile-at-a-time path below, which is also the only path when `K` is under
// eight tiles. Nothing here needs super-block alignment: the kernel addresses
// each tile by its global index, which is what a super-block's sub-plane is
// keyed on.
// ============================================================================

__global__ __launch_bounds__(TCF_GEMV_BLOCK, 1) void tcf_gemv_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M,
    unsigned int K,
    unsigned int N,
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
    const TcfLayout l = tcf_layout(code_high_off, scale_off, min_off, super_off,
                                   super_min_off, bits, group, groups_per_tile,
                                   symmetric, scale_form, sub_block_bytes);

    const unsigned int warp_id = threadIdx.x / TCF_WARP_SIZE;
    const unsigned int lane = threadIdx.x % TCF_WARP_SIZE;
    const unsigned int col = blockIdx.x * TCF_WARPS_PER_BLOCK + warp_id;
    const unsigned int m = blockIdx.y;
    if (col >= N || m >= M) {
        return;
    }

    const unsigned int tiles_per_row = K / TCF_TILE;
    const float* act_row = activation + (size_t)m * (size_t)K;
    const unsigned int runs = tiles_per_row / TCF_RUN_TILES;
    // A `float4` activation load needs a 16-byte-aligned row. The activation
    // pointer is a tensor VIEW's, so it carries that view's element offset and
    // is only guaranteed 4-byte aligned. Checked, not assumed: a misaligned
    // wide load is undefined behaviour, not a slow one. `K` is a whole number
    // of 64-element tiles, so one row being aligned makes every row aligned,
    // and the choice is uniform across the warp.
    const bool act_wide = ((size_t)act_row & 15u) == 0u;
    float acc = act_wide
        ? tcf_gemv_runs<true>(act_row, weight, l, col, tiles_per_row, runs, lane)
        : tcf_gemv_runs<false>(act_row, weight, l, col, tiles_per_row, runs, lane);

    // The tiles a whole run cannot cover, one at a time.
    for (unsigned int j = runs * TCF_RUN_TILES; j < tiles_per_row; ++j) {
        const unsigned int tile = col * tiles_per_row + j;
        float scale = 0.0f;
        float min_value = 0.0f;
        if (lane < l.groups_per_tile) {
            tcf_group_values(weight, l, tile, lane, &scale, &min_value);
        }
        const unsigned int base = j * TCF_TILE;
#pragma unroll
        for (unsigned int half = 0; half < 2u; ++half) {
            const unsigned int e = lane + half * TCF_WARP_SIZE;
            const unsigned int g = e / l.group;
            const float gs = __shfl_sync(0xFFFFFFFFu, scale, g);
            const float gm = __shfl_sync(0xFFFFFFFFu, min_value, g);
            const int code = tcf_code(weight, l, tile, e);
            acc += act_row[base + e] * tcf_value(code, gs, gm, l.symmetric);
        }
    }

    acc = tcf_warp_reduce_sum(acc);
    if (lane == 0) {
        output[(size_t)m * (size_t)N + (size_t)col] = acc;
    }
}

// ============================================================================
// GEMM: activation [M, K] x weight [N, K]^T -> output [M, N], for a batch the
// GEMV no longer covers.
//
// A 32x32 output tile per block and a 4x4 REGISTER PATCH per thread, so 64
// threads cover the tile. Every 32 elements of K they stage 32 decoded weight
// rows and the matching 32 activation rows into shared memory, transposed to
// K-major, and each thread then reads one `float4` from each and issues
// sixteen FMAs. A block may run several such thread groups over independent K
// ranges; see below.
//
// # Why a register patch, measured
//
// The first form gave each thread ONE output element and read both operands
// from shared memory, so every fused multiply-add cost two shared loads. That
// ratio, not the decode, is what the kernel was spending its time on. An
// Ampere SM retires four warp-wide FMAs per clock and services one warp-wide
// shared load per clock, so two loads per FMA pins the kernel at an EIGHTH of
// the card's f32 rate whatever the encoding does. On an RTX 3060 at M = 32,
// N = 4096, K = 1024 that predicted 337 us of shared-load issue against a
// measured 454 us, and the kernel ran at 1.2 TFLOP/s of a 12.7 peak.
//
// The decode is not the problem it looks like. At M = 32 the tile grid decodes
// the weight twice, 8.4M decodes; even at ten instructions each that is 13 us
// of this card, three per cent of the 454. Cost tracking M is not evidence of
// a decode that fails to amortize — the FLOP count tracks M too.
//
// A 4x4 patch reads four activations and four weights for sixteen FMAs: eight
// loads become two `float4` loads, and the ratio goes from 2.0 to 0.125 shared
// instructions per FMA. Within a warp the four M positions are shared by eight
// lanes and the eight N positions by four, so the two loads broadcast down to
// 64 and 128 bytes — one clock each against the four clocks of FMA they feed.
// The inner loop is FMA-bound rather than shared-load-bound.
//
// # Why 32x32 and 64 threads, and not a larger tile
//
// M is 32 for 97.6% of the launches a TTS render issues, so BM = 32 covers the
// whole batch in one block row and decodes the weight ONCE. What is then
// scarce is blocks: at M = 32 the whole output is M * N elements, and a thread
// holding sixteen of them leaves M * N / 1024 blocks. N = 1024 gives 32 and
// N = 256 gives 8, against 28 SMs. Every doubling of the tile halves that
// count, so a 64x64 tile — the usual choice when M is large — would leave
// N = 1024 with sixteen blocks on 28 SMs and idle 40% of the card. 32x32 is
// the largest tile that still fills this GPU at the shape that dominates.
//
// # Why the block also splits K
//
// 32x32 is not small enough on its own. At M = 32 the whole output is M * N
// elements and every tiling of it that keeps sixteen outputs per thread yields
// the same M * N / 1024 thread groups: 32 for N = 1024, 8 for N = 256. Shrinking
// the tile moves warps between blocks without creating any.
//
// So a block takes `split` INDEPENDENT K ranges, one 64-thread group each, and
// folds their partial sums at the end. That multiplies the warps in flight
// without touching the output tiling, and costs two barriers and one shared
// fold per block. It is not free: each slice stages its own operands, so
// `split` also divides the blocks an SM can hold — 9216 bytes per slice against
// a 48 KB budget, five blocks at split 1 and one at split 4. The host picks the
// count from the grid size and sizes the dynamic shared memory to match; see
// `gemm_split` in `quant/cuda/tcf/launch.rs` for the measured table. A grid that
// already fills the card is launched at split 1 and pays none of it.
//
// # Why the staging reads whole code words
//
// The old form called `tcf_code` once per element, which recomputes a byte
// address and issues one narrow load, and divided by the runtime group width
// per element. Here each thread stages one 16-code chunk through
// `tcf_code_run`, the same reader the GEMV uses: one `uint4` (8-bit) or one
// `uint2` plus one `uint` (4- and 6-bit) covering all sixteen. Sixteen codes
// from a multiple of sixteen lie in ONE group at every group width v1 defines,
// so the chunk also resolves its `(scale, minimum)` once instead of sixteen
// times, and no scale ever reaches shared memory.
//
// # Accumulation order
//
// Within a slice: K ascending, element by element within a tile and tile by
// tile, which is the order the first form used and the order the CPU path
// uses. At split 1 this kernel is therefore BIT-IDENTICAL to the 16x16 form it
// replaces, checked over every bit width and scale form. A split folds the
// slices' partials in slice order afterwards, which is a blocked summation of
// the same products — measured at 2e-6 of the largest output against split 1,
// the same order as the GEMV/GEMM difference the matmul gate already accepts.
//
// The DECODED VALUE is `tcf_value`'s either way and is unchanged; that is what
// `tests/backend_parity/quant_tcf.rs` holds.
// ============================================================================

__global__ __launch_bounds__(TCF_GEMM_MAX_BLOCK, 1) void tcf_gemm_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M,
    unsigned int K,
    unsigned int N,
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
    unsigned int sub_block_bytes,
    unsigned int split
) {
    const TcfLayout l = tcf_layout(code_high_off, scale_off, min_off, super_off,
                                   super_min_off, bits, group, groups_per_tile,
                                   symmetric, scale_form, sub_block_bytes);

    // `split * TCF_GEMM_SLICE_FLOATS` floats, the activation staging of every
    // slice first so the reduction below can reuse it as one flat run.
    extern __shared__ __align__(16) float s_dyn[];
    float* const s_act = s_dyn;
    float* const s_weight = s_dyn + split * (TCF_GEMM_BK * TCF_GEMM_ASTRIDE);

    const unsigned int slice = threadIdx.x / TCF_GEMM_THREADS;
    const unsigned int tid = threadIdx.x % TCF_GEMM_THREADS;
    float* const act_slice = s_act + slice * (TCF_GEMM_BK * TCF_GEMM_ASTRIDE);
    float* const w_slice = s_weight + slice * (TCF_GEMM_BK * TCF_GEMM_WSTRIDE);

    const unsigned int tx = tid % TCF_GEMM_TX;
    const unsigned int ty = tid / TCF_GEMM_TX;
    const unsigned int row0 = blockIdx.y * TCF_GEMM_BM;
    const unsigned int col0 = blockIdx.x * TCF_GEMM_BN;

    // Two threads stage one weight row per K slice, sixteen codes each.
    const unsigned int stage_row = tid / 2u;
    const unsigned int stage_e0 = (tid & 1u) * TCF_GEMM_CHUNK;
    const unsigned int stage_col = col0 + stage_row;
    const bool stage_live = stage_col < N;

    const unsigned int tiles_per_row = K / TCF_TILE;
    // Every slice runs the same number of steps whether or not its range has
    // that many tiles left: `__syncthreads` is block-wide, so a slice that
    // stopped early would leave the others waiting on a barrier it never
    // reaches. A step past the end stages zeros and accumulates them.
    const unsigned int steps = (tiles_per_row + split - 1u) / split;
    const unsigned int j0 = slice * steps;

    float acc[TCF_GEMM_TM][TCF_GEMM_TN];
#pragma unroll
    for (unsigned int u = 0; u < TCF_GEMM_TM; ++u) {
#pragma unroll
        for (unsigned int v = 0; v < TCF_GEMM_TN; ++v) {
            acc[u][v] = 0.0f;
        }
    }

    for (unsigned int step = 0; step < steps; ++step) {
        const unsigned int j = j0 + step;
        const bool live = stage_live && j < tiles_per_row;
        const unsigned int tile = live ? stage_col * tiles_per_row + j : 0u;
#pragma unroll
        for (unsigned int h = 0; h < TCF_TILE / TCF_GEMM_BK; ++h) {
            // The chunk this thread owns inside the 64-element execution tile.
            const unsigned int sub = h * (TCF_GEMM_BK / TCF_GEMM_CHUNK) + (tid & 1u);

            // Guards the staging the previous step is still reading.
            __syncthreads();

            float scale = 0.0f;
            float min_value = 0.0f;
            TcfCodeRun run;
            run.w[0] = 0u;
            run.w[1] = 0u;
            run.w[2] = 0u;
            run.w[3] = 0u;
            if (live) {
                tcf_group_values(weight, l, tile, (TCF_GEMM_CHUNK * sub) / l.group,
                                 &scale, &min_value);
                run = tcf_code_run(weight, l, tile, sub);
            }
#pragma unroll
            for (unsigned int i = 0; i < TCF_GEMM_CHUNK; ++i) {
                const float w = live
                    ? tcf_value(tcf_run_code(run, l, i), scale, min_value, l.symmetric)
                    : 0.0f;
                w_slice[(stage_e0 + i) * TCF_GEMM_WSTRIDE + stage_row] = w;
            }

            const unsigned int kbase = j * TCF_TILE + h * TCF_GEMM_BK;
#pragma unroll
            for (unsigned int p = 0; p < TCF_GEMM_ALOADS; ++p) {
                const unsigned int index = tid + p * TCF_GEMM_THREADS;
                const unsigned int e = index % TCF_GEMM_BK;
                const unsigned int m = index / TCF_GEMM_BK;
                float a = 0.0f;
                if (j < tiles_per_row && row0 + m < M) {
                    a = activation[(size_t)(row0 + m) * (size_t)K
                                   + (size_t)(kbase + e)];
                }
                act_slice[e * TCF_GEMM_ASTRIDE + m] = a;
            }
            __syncthreads();

#pragma unroll
            for (unsigned int kk = 0; kk < TCF_GEMM_BK; ++kk) {
                const float4 av =
                    *(const float4*)&act_slice[kk * TCF_GEMM_ASTRIDE + ty * TCF_GEMM_TM];
                const float4 bv =
                    *(const float4*)&w_slice[kk * TCF_GEMM_WSTRIDE + tx * TCF_GEMM_TN];
                const float a[TCF_GEMM_TM] = {av.x, av.y, av.z, av.w};
                const float b[TCF_GEMM_TN] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
                for (unsigned int u = 0; u < TCF_GEMM_TM; ++u) {
#pragma unroll
                    for (unsigned int v = 0; v < TCF_GEMM_TN; ++v) {
                        acc[u][v] += a[u] * b[v];
                    }
                }
            }
        }
    }

    // Fold the slices' partial sums into slice 0, in slice order — a blocked
    // summation of the same products, which the matmul gate's 1e-3 relative
    // tolerance covers and every BLAS does. The staging buffer is dead by now,
    // and its `split * TCF_GEMM_BK * TCF_GEMM_ASTRIDE` activation floats
    // exceed the `(split - 1) * TCF_GEMM_THREADS * patch` this needs.
    const unsigned int patch = TCF_GEMM_TM * TCF_GEMM_TN;
    if (split > 1u) {
        __syncthreads();
        if (slice > 0u) {
            float* dst = s_act + (slice - 1u) * (TCF_GEMM_THREADS * patch) + tid * patch;
#pragma unroll
            for (unsigned int u = 0; u < TCF_GEMM_TM; ++u) {
#pragma unroll
                for (unsigned int v = 0; v < TCF_GEMM_TN; ++v) {
                    dst[u * TCF_GEMM_TN + v] = acc[u][v];
                }
            }
        }
        __syncthreads();
        if (slice != 0u) {
            return;
        }
        for (unsigned int s = 1; s < split; ++s) {
            const float* src =
                s_act + (s - 1u) * (TCF_GEMM_THREADS * patch) + tid * patch;
#pragma unroll
            for (unsigned int u = 0; u < TCF_GEMM_TM; ++u) {
#pragma unroll
                for (unsigned int v = 0; v < TCF_GEMM_TN; ++v) {
                    acc[u][v] += src[u * TCF_GEMM_TN + v];
                }
            }
        }
    }

#pragma unroll
    for (unsigned int u = 0; u < TCF_GEMM_TM; ++u) {
        const unsigned int row = row0 + ty * TCF_GEMM_TM + u;
        if (row >= M) {
            continue;
        }
#pragma unroll
        for (unsigned int v = 0; v < TCF_GEMM_TN; ++v) {
            const unsigned int col = col0 + tx * TCF_GEMM_TN + v;
            if (col < N) {
                output[(size_t)row * (size_t)N + (size_t)col] = acc[u][v];
            }
        }
    }
}

}  // extern "C"
