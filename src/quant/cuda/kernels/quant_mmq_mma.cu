// Q8_0 x Q8_1 -> f32 (MMQ), tensor-core variant of `quant_mmq_q8_0_q8_1` in
// `quant_gemv.cu`. The compute loop and the output write use
// `mma_m16n8k32_s8` instead of `dp4a`.
//
// All three kernels here stage one k-block ahead: the global loads for block
// `b+1` are issued before block `b`'s `mma` sequence, so their latency
// overlaps compute. nvcc does not reorder this itself: the loads must cross a
// `__syncthreads()` to move earlier.
//
// This is a separate translation unit, so the tile constants and
// `load_int_ua` are re-declared rather than shared via a header.

#include <cuda_fp16.h>

#include "decode.cuh"
#include "mma_int8.cuh"

#define WARP_SIZE 32
#define MMQ_BM 128
#define MMQ_BN 64
#define MMQ_BK 32
// Row-stride padding, in ints. 8 makes the padded stride 8 mod 32, so the four
// word indices a lane reads land 8 banks apart and the warp's 32 accesses cover
// all 32 banks.
#define MMQ_SMEM_PAD 8
#define MMQ_THREADS 256
// `__launch_bounds__` below passes the block size ONLY. The second argument
// (minimum blocks per SM) is deliberately omitted: setting it to 1 tells ptxas
// that a single resident block suffices, so it spends registers freely on the
// accumulator tile and occupancy drops. Left unconstrained, ptxas picks the
// register count itself, per target architecture, at compile or JIT time —
// which is where that decision belongs, since the register file and the
// latency it has to cover are properties of the device, not of this file.
// Four consecutive k values per int, which is one dp4a/mma operand word.
#define MMQ_K4 (MMQ_BK / 4)

#define MMQ_WARPS (MMQ_THREADS / WARP_SIZE)
// A staging warp covers four columns at once, so the block advances by this
// many columns or rows per staging step.
#define MMQ_STAGE_STRIDE (MMQ_WARPS * 4)
#define MMQ_W_STAGES (MMQ_BN / MMQ_STAGE_STRIDE)
#define MMQ_A_STAGES (MMQ_BM / MMQ_STAGE_STRIDE)

// 2-byte-aligned 4-byte load (quant blocks are not always 4-byte aligned).
static __device__ __forceinline__ int load_int_ua(const unsigned char* p) {
    const unsigned short* p16 = (const unsigned short*)p;
    return (int)p16[0] | ((int)p16[1] << 16);
}

// Reads k-block `b` of both operands into registers. Held separate from the
// shared-memory write so the caller issues it one iteration ahead of use.
// Global load latency dominates this loop; nothing else overlaps it.
static __device__ __forceinline__ void mmq_q8_0_stage_load(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    unsigned int M, unsigned int N, unsigned int bpr, unsigned int b,
    unsigned int row0, unsigned int col0, unsigned int base, unsigned int stage_k4,
    int (&w_packed)[MMQ_W_STAGES], float (&w_d)[MMQ_W_STAGES],
    int (&a_packed)[MMQ_A_STAGES], float (&a_d)[MMQ_A_STAGES]
) {
#pragma unroll
    for (int i = 0; i < MMQ_W_STAGES; ++i) {
        const unsigned int gcol = col0 + base + i * MMQ_STAGE_STRIDE;
        w_packed[i] = 0;
        w_d[i] = 0.0f;
        if (gcol < N) {
            const unsigned char* blk = weight + ((unsigned long long)gcol * bpr + b) * 34;
            w_d[i] = __half2float(*reinterpret_cast<const __half*>(blk));
            // The quants start at byte 2, so only 2-byte alignment holds.
            w_packed[i] = load_int_ua(blk + 2 + stage_k4 * 4);
        }
    }
#pragma unroll
    for (int i = 0; i < MMQ_A_STAGES; ++i) {
        const unsigned int grow = row0 + base + i * MMQ_STAGE_STRIDE;
        a_packed[i] = 0;
        a_d[i] = 0.0f;
        if (grow < M) {
            // Q8_1: d, then the block sum, then 32 quants at byte 4.
            const unsigned char* blk = q8_act + ((unsigned long long)grow * bpr + b) * 36;
            a_d[i] = __half2float(*reinterpret_cast<const __half*>(blk));
            a_packed[i] = *reinterpret_cast<const int*>(blk + 4 + stage_k4 * 4);
        }
    }
}

extern "C" __global__ __launch_bounds__(MMQ_THREADS) void quant_mmq_q8_0_q8_1_mma(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    // Padded by MMQ_SMEM_PAD so the word index reaches the bank index. An
    // unpadded row stride of 128 or 64 ints is a multiple of 32, so every
    // fragment word lands in the same bank and the warp serializes 4 ways.
    __shared__ int s_w[MMQ_K4][MMQ_BN + MMQ_SMEM_PAD];
    __shared__ int s_a[MMQ_K4][MMQ_BM + MMQ_SMEM_PAD];
    __shared__ float s_wd[MMQ_BN];
    __shared__ float s_ad[MMQ_BM];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int warp = tid / WARP_SIZE;

    const unsigned int row0 = blockIdx.y * MMQ_BM;
    const unsigned int col0 = blockIdx.x * MMQ_BN;
    const unsigned int bpr = K / 32;  // blocks per row, both operands

    // A staging warp covers four columns at once: lane -> (column, k4).
    const unsigned int stage_sub = lane / MMQ_K4;  // 0..3
    const unsigned int stage_k4 = lane % MMQ_K4;   // 0..7

    // Warp `warp` owns output rows `16*warp .. 16*warp+16`; eight warps cover
    // all 128 rows. Each warp covers all 64 columns as eight 8-column groups.
    float acc[8][4];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            acc[g][l] = 0.0f;
        }
    }

    const unsigned int stage_base = warp * 4 + stage_sub;
    int w_packed[MMQ_W_STAGES];
    float w_d[MMQ_W_STAGES];
    int a_packed[MMQ_A_STAGES];
    float a_d[MMQ_A_STAGES];

    if (bpr > 0) {
        mmq_q8_0_stage_load(q8_act, weight, M, N, bpr, 0, row0, col0, stage_base,
                            stage_k4, w_packed, w_d, a_packed, a_d);
    }

    for (unsigned int b = 0; b < bpr; ++b) {
        __syncthreads();

#pragma unroll
        for (int i = 0; i < MMQ_W_STAGES; ++i) {
            const unsigned int c = stage_base + i * MMQ_STAGE_STRIDE;
            s_w[stage_k4][c] = w_packed[i];
            if (stage_k4 == 0) s_wd[c] = w_d[i];
        }
#pragma unroll
        for (int i = 0; i < MMQ_A_STAGES; ++i) {
            const unsigned int r = stage_base + i * MMQ_STAGE_STRIDE;
            s_a[stage_k4][r] = a_packed[i];
            if (stage_k4 == 0) s_ad[r] = a_d[i];
        }

        __syncthreads();

        if (b + 1 < bpr) {
            mmq_q8_0_stage_load(q8_act, weight, M, N, bpr, b + 1, row0, col0, stage_base,
                                stage_k4, w_packed, w_d, a_packed, a_d);
        }

        // One `mma_m16n8k32_s8` consumes a whole 32-element Q8_0/Q8_1 block
        // as a single k-step, so the block loop needs no inner k4 loop. The
        // int32 accumulation is exact, so the float scale still applies once
        // per block, same as the dp4a kernel. A, B and D each read a
        // DIFFERENT index map from `mma_int8.cuh` — they are not the same
        // register layout.
        int A[4];
        for (int l = 0; l < 4; ++l) {
            A[l] = s_a[mma_a_j(l)][warp * 16 + mma_a_i(l)];
        }

        for (int g = 0; g < 8; ++g) {
            int B[2];
            for (int l = 0; l < 2; ++l) {
                B[l] = s_w[mma_b_j(l)][g * 8 + mma_b_i(l)];
            }

            int D[4] = {0, 0, 0, 0};
            mma_m16n8k32_s8(D, A, B);

            for (int l = 0; l < 4; ++l) {
                const float da = s_ad[warp * 16 + mma_d_i(l)];
                const float dw = s_wd[g * 8 + mma_d_j(l)];
                acc[g][l] += (float)D[l] * da * dw;
            }
        }
    }

    for (int g = 0; g < 8; ++g) {
        for (int l = 0; l < 4; ++l) {
            const unsigned int r = row0 + warp * 16 + mma_d_i(l);
            const unsigned int c = col0 + g * 8 + mma_d_j(l);
            if (r < M && c < N) output[(unsigned long long)r * N + c] = acc[g][l];
        }
    }
}

// Copied verbatim from `quant_gemv.cu`, including the `__CUDA_ARCH__ >= 610`
// guard, so the row-sum reduction below matches the dp4a kernel exactly.
static __device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const signed char* a8 = (const signed char*)&a;
    const signed char* b8 = (const signed char*)&b;
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif
}

// Reads Q8_1 sub-block `b` of both operands into registers, one iteration
// ahead, same reason as `mmq_q8_0_stage_load`.
static __device__ __forceinline__ void mmq_q4_k_stage_load(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    unsigned int M, unsigned int N, unsigned int sub_blocks, unsigned int supers,
    unsigned int b, unsigned int row0, unsigned int col0, unsigned int base,
    unsigned int stage_k4,
    int (&w_packed)[MMQ_W_STAGES], float (&w_sd)[MMQ_W_STAGES],
    float (&w_sm)[MMQ_W_STAGES],
    int (&a_packed)[MMQ_A_STAGES], float (&a_d)[MMQ_A_STAGES]
) {
    const unsigned int sup = b / 8;
    const unsigned int j = b % 8;
#pragma unroll
    for (int i = 0; i < MMQ_W_STAGES; ++i) {
        const unsigned int gcol = col0 + base + i * MMQ_STAGE_STRIDE;
        w_packed[i] = 0;
        w_sd[i] = 0.0f;
        w_sm[i] = 0.0f;
        if (gcol < N) {
            const unsigned char* blk = weight + ((unsigned long long)gcol * supers + sup) * 144;
            const float d = __half2float(*reinterpret_cast<const __half*>(blk));
            const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
            int scale;
            int minimum;
            q4k_scale_min(blk + 4, (int)j, &scale, &minimum);
            w_sd[i] = d * (float)scale;
            w_sm[i] = dmin * (float)minimum;
            // 144 is 16-aligned and every offset here is a multiple of 4, so a
            // plain int load is aligned.
            const int v = *reinterpret_cast<const int*>(blk + 16 + (j / 2) * 32 + stage_k4 * 4);
            w_packed[i] = (j & 1) ? ((v >> 4) & 0x0F0F0F0F) : (v & 0x0F0F0F0F);
        }
    }
#pragma unroll
    for (int i = 0; i < MMQ_A_STAGES; ++i) {
        const unsigned int grow = row0 + base + i * MMQ_STAGE_STRIDE;
        a_packed[i] = 0;
        a_d[i] = 0.0f;
        if (grow < M) {
            const unsigned char* blk =
                q8_act + ((unsigned long long)grow * sub_blocks + b) * 36;
            a_d[i] = __half2float(*reinterpret_cast<const __half*>(blk));
            a_packed[i] = *reinterpret_cast<const int*>(blk + 4 + stage_k4 * 4);
        }
    }
}

// Q4_K x Q8_1 -> f32 (MMQ), tensor-core variant of `quant_mmq_q4_k_q8_1` in
// `quant_gemv.cu`. Only the compute loop and the output write use
// `mma_m16n8k32_s8` instead of `dp4a`.
//
// Q4_K adds an asymmetric minimum term, `-dmin * minimum * rowsum`, that one
// `mma` cannot produce: it depends only on the row, not the column, so it is
// reduced separately with `dp4a` the same way the dp4a kernel reduces it.
extern "C" __global__ __launch_bounds__(MMQ_THREADS) void quant_mmq_q4_k_q8_1_mma(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    __shared__ int s_w[MMQ_K4][MMQ_BN + MMQ_SMEM_PAD];
    __shared__ int s_a[MMQ_K4][MMQ_BM + MMQ_SMEM_PAD];
    __shared__ float s_wd[MMQ_BN];  // d * scale
    __shared__ float s_wm[MMQ_BN];  // dmin * minimum
    __shared__ float s_ad[MMQ_BM];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int warp = tid / WARP_SIZE;

    const unsigned int row0 = blockIdx.y * MMQ_BM;
    const unsigned int col0 = blockIdx.x * MMQ_BN;

    const unsigned int sub_blocks = K / 32;      // Q8_1 blocks, and Q4_K sub-blocks
    const unsigned int supers = sub_blocks / 8;  // Q4_K 256-element super-blocks

    const unsigned int stage_sub = lane / MMQ_K4;
    const unsigned int stage_k4 = lane % MMQ_K4;

    float acc[8][4];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            acc[g][l] = 0.0f;
        }
    }

    const unsigned int stage_base = warp * 4 + stage_sub;
    int w_packed[MMQ_W_STAGES];
    float w_sd[MMQ_W_STAGES];
    float w_sm[MMQ_W_STAGES];
    int a_packed[MMQ_A_STAGES];
    float a_d[MMQ_A_STAGES];

    if (sub_blocks > 0) {
        mmq_q4_k_stage_load(q8_act, weight, M, N, sub_blocks, supers, 0, row0, col0,
                            stage_base, stage_k4, w_packed, w_sd, w_sm, a_packed, a_d);
    }

    for (unsigned int b = 0; b < sub_blocks; ++b) {
        __syncthreads();

#pragma unroll
        for (int i = 0; i < MMQ_W_STAGES; ++i) {
            const unsigned int c = stage_base + i * MMQ_STAGE_STRIDE;
            s_w[stage_k4][c] = w_packed[i];
            if (stage_k4 == 0) {
                s_wd[c] = w_sd[i];
                s_wm[c] = w_sm[i];
            }
        }
#pragma unroll
        for (int i = 0; i < MMQ_A_STAGES; ++i) {
            const unsigned int r = stage_base + i * MMQ_STAGE_STRIDE;
            s_a[stage_k4][r] = a_packed[i];
            if (stage_k4 == 0) s_ad[r] = a_d[i];
        }

        __syncthreads();

        if (b + 1 < sub_blocks) {
            mmq_q4_k_stage_load(q8_act, weight, M, N, sub_blocks, supers, b + 1, row0, col0,
                                stage_base, stage_k4, w_packed, w_sd, w_sm, a_packed, a_d);
        }

        // `mma_d_i(l)` takes only two distinct values per lane, one per half
        // of `l / 2`. `rsum2[h]` covers the two rows this lane's accumulator
        // touches, summing this sub-block's 32 activation quants per row.
        const unsigned int lane_q = threadIdx.x & 31;
        int rsum2[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = warp * 16 + h * 8 + (int)(lane_q / 4);
            int s = 0;
#pragma unroll
            for (int k4 = 0; k4 < MMQ_K4; ++k4) {
                s = dp4a(0x01010101, s_a[k4][row], s);
            }
            rsum2[h] = s;
        }

        int A[4];
        for (int l = 0; l < 4; ++l) {
            A[l] = s_a[mma_a_j(l)][warp * 16 + mma_a_i(l)];
        }

        for (int g = 0; g < 8; ++g) {
            int B[2];
            for (int l = 0; l < 2; ++l) {
                B[l] = s_w[mma_b_j(l)][g * 8 + mma_b_i(l)];
            }

            int D[4] = {0, 0, 0, 0};
            mma_m16n8k32_s8(D, A, B);

            for (int l = 0; l < 4; ++l) {
                const int row = warp * 16 + mma_d_i(l);
                const int col = g * 8 + mma_d_j(l);
                acc[g][l] +=
                    s_ad[row] * (s_wd[col] * (float)D[l] - s_wm[col] * (float)rsum2[l / 2]);
            }
        }
    }

    for (int g = 0; g < 8; ++g) {
        for (int l = 0; l < 4; ++l) {
            const unsigned int r = row0 + warp * 16 + mma_d_i(l);
            const unsigned int c = col0 + g * 8 + mma_d_j(l);
            if (r < M && c < N) output[(unsigned long long)r * N + c] = acc[g][l];
        }
    }
}

// Reads Q8_1 sub-block `b` of both operands into registers, one iteration
// ahead, same reason as `mmq_q8_0_stage_load`.
static __device__ __forceinline__ void mmq_q6_k_stage_load(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    unsigned int M, unsigned int N, unsigned int sub_blocks, unsigned int supers,
    unsigned int b, unsigned int row0, unsigned int col0, unsigned int base,
    unsigned int stage_k4,
    int (&w_packed)[MMQ_W_STAGES], float (&w_sd_lo)[MMQ_W_STAGES],
    float (&w_sd_hi)[MMQ_W_STAGES],
    int (&a_packed)[MMQ_A_STAGES], float (&a_d)[MMQ_A_STAGES]
) {
    const unsigned int sup = b / 8;
    const unsigned int j = b % 8;
    const unsigned int half = j / 4;
    const unsigned int t = j % 4;
#pragma unroll
    for (int i = 0; i < MMQ_W_STAGES; ++i) {
        const unsigned int gcol = col0 + base + i * MMQ_STAGE_STRIDE;
        w_packed[i] = 0;
        w_sd_lo[i] = 0.0f;
        w_sd_hi[i] = 0.0f;
        if (gcol < N) {
            const unsigned char* blk = weight + ((unsigned long long)gcol * supers + sup) * 210;
            const unsigned char* ql = blk + half * 64;
            const unsigned char* qh = blk + 128 + half * 32;
            const signed char* sc = reinterpret_cast<const signed char*>(blk + 192) + half * 8;
            __half d_h;
            memcpy(&d_h, blk + 208, 2);
            const float d = __half2float(d_h);
            w_sd_lo[i] = d * (float)sc[t * 2];
            w_sd_hi[i] = d * (float)sc[t * 2 + 1];

            // 210 is only 2-byte aligned, so the 4-byte reads are unaligned.
            const unsigned int e0 = stage_k4 * 4;
            const int ql4 = load_int_ua(ql + ((t & 1) ? e0 + 32 : e0));
            const int qh4 = load_int_ua(qh + e0);
            const int low = (ql4 >> ((t & 2) ? 4 : 0)) & 0x0F0F0F0F;
            const int high = ((qh4 >> (t * 2)) & 0x03030303) << 4;
            // The 6-bit value is unsigned 0..63 biased by 32.
            w_packed[i] = __vsubss4(low | high, 0x20202020);
        }
    }
#pragma unroll
    for (int i = 0; i < MMQ_A_STAGES; ++i) {
        const unsigned int grow = row0 + base + i * MMQ_STAGE_STRIDE;
        a_packed[i] = 0;
        a_d[i] = 0.0f;
        if (grow < M) {
            const unsigned char* blk =
                q8_act + ((unsigned long long)grow * sub_blocks + b) * 36;
            a_d[i] = __half2float(*reinterpret_cast<const __half*>(blk));
            a_packed[i] = *reinterpret_cast<const int*>(blk + 4 + stage_k4 * 4);
        }
    }
}

// Q6_K x Q8_1 -> f32 (MMQ), tensor-core variant of `quant_mmq_q6_k_q8_1` in
// `quant_gemv.cu`. Only the compute loop and the output write differ.
//
// Q6_K's scale changes every 16 elements, so one 32-wide `mma` cannot express
// it. Two `m16n8k16` calls run instead, one per 16-element half, each scaled
// by its own `s_wd_lo` / `s_wd_hi`.
extern "C" __global__ __launch_bounds__(MMQ_THREADS) void quant_mmq_q6_k_q8_1_mma(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    __shared__ int s_w[MMQ_K4][MMQ_BN + MMQ_SMEM_PAD];
    __shared__ float s_wd_lo[MMQ_BN];  // d * scale for elements 0..15
    __shared__ float s_wd_hi[MMQ_BN];  // d * scale for elements 16..31
    __shared__ int s_a[MMQ_K4][MMQ_BM + MMQ_SMEM_PAD];
    __shared__ float s_ad[MMQ_BM];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int warp = tid / WARP_SIZE;

    const unsigned int row0 = blockIdx.y * MMQ_BM;
    const unsigned int col0 = blockIdx.x * MMQ_BN;

    const unsigned int sub_blocks = K / 32;
    const unsigned int supers = sub_blocks / 8;

    const unsigned int stage_sub = lane / MMQ_K4;
    const unsigned int stage_k4 = lane % MMQ_K4;

    float acc[8][4];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            acc[g][l] = 0.0f;
        }
    }

    const unsigned int stage_base = warp * 4 + stage_sub;
    int w_packed[MMQ_W_STAGES];
    float w_sd_lo[MMQ_W_STAGES];
    float w_sd_hi[MMQ_W_STAGES];
    int a_packed[MMQ_A_STAGES];
    float a_d[MMQ_A_STAGES];

    if (sub_blocks > 0) {
        mmq_q6_k_stage_load(q8_act, weight, M, N, sub_blocks, supers, 0, row0, col0,
                            stage_base, stage_k4, w_packed, w_sd_lo, w_sd_hi, a_packed, a_d);
    }

    for (unsigned int b = 0; b < sub_blocks; ++b) {
        __syncthreads();

#pragma unroll
        for (int i = 0; i < MMQ_W_STAGES; ++i) {
            const unsigned int c = stage_base + i * MMQ_STAGE_STRIDE;
            s_w[stage_k4][c] = w_packed[i];
            if (stage_k4 == 0) {
                s_wd_lo[c] = w_sd_lo[i];
                s_wd_hi[c] = w_sd_hi[i];
            }
        }
#pragma unroll
        for (int i = 0; i < MMQ_A_STAGES; ++i) {
            const unsigned int r = stage_base + i * MMQ_STAGE_STRIDE;
            s_a[stage_k4][r] = a_packed[i];
            if (stage_k4 == 0) s_ad[r] = a_d[i];
        }

        __syncthreads();

        if (b + 1 < sub_blocks) {
            mmq_q6_k_stage_load(q8_act, weight, M, N, sub_blocks, supers, b + 1, row0, col0,
                                stage_base, stage_k4, w_packed, w_sd_lo, w_sd_hi, a_packed, a_d);
        }

        // Words 0..3 of the staged block are the low 16-element half, words
        // 4..7 the high half, the same split the dp4a kernel makes at
        // `k4 < 4`. Each half runs its own `m16n8k16` and takes its own scale.
        int A_lo[2];
        int A_hi[2];
        for (int l = 0; l < 2; ++l) {
            A_lo[l] = s_a[mma_a16_j(l)][warp * 16 + mma_a16_i(l)];
            A_hi[l] = s_a[4 + mma_a16_j(l)][warp * 16 + mma_a16_i(l)];
        }

        for (int g = 0; g < 8; ++g) {
            int B_lo[1];
            int B_hi[1];
            B_lo[0] = s_w[mma_b16_j(0)][g * 8 + mma_b16_i(0)];
            B_hi[0] = s_w[4 + mma_b16_j(0)][g * 8 + mma_b16_i(0)];

            int D_lo[4] = {0, 0, 0, 0};
            int D_hi[4] = {0, 0, 0, 0};
            mma_m16n8k16_s8(D_lo, A_lo, B_lo);
            mma_m16n8k16_s8(D_hi, A_hi, B_hi);

            for (int l = 0; l < 4; ++l) {
                const int row = warp * 16 + mma_d_i(l);
                const int col = g * 8 + mma_d_j(l);
                acc[g][l] += s_ad[row] * (s_wd_lo[col] * (float)D_lo[l]
                                          + s_wd_hi[col] * (float)D_hi[l]);
            }
        }
    }

    for (int g = 0; g < 8; ++g) {
        for (int l = 0; l < 4; ++l) {
            const unsigned int r = row0 + warp * 16 + mma_d_i(l);
            const unsigned int c = col0 + g * 8 + mma_d_j(l);
            if (r < M && c < N) output[(unsigned long long)r * N + c] = acc[g][l];
        }
    }
}
