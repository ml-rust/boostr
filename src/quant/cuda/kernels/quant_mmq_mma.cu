// Q8_0 x Q8_1 -> f32 (MMQ), tensor-core variant of `quant_mmq_q8_0_q8_1` in
// `quant_gemv.cu`. Staging is copied verbatim; only the compute loop and the
// output write use `mma_m16n8k32_s8` instead of `dp4a`.
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
// Four consecutive k values per int, which is one dp4a/mma operand word.
#define MMQ_K4 (MMQ_BK / 4)

// 2-byte-aligned 4-byte load (quant blocks are not always 4-byte aligned).
static __device__ __forceinline__ int load_int_ua(const unsigned char* p) {
    const unsigned short* p16 = (const unsigned short*)p;
    return (int)p16[0] | ((int)p16[1] << 16);
}

extern "C" __global__ __launch_bounds__(MMQ_THREADS, 1) void quant_mmq_q8_0_q8_1_mma(
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
    const unsigned int warps = MMQ_THREADS / WARP_SIZE;

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

    for (unsigned int b = 0; b < bpr; ++b) {
        __syncthreads();

        for (unsigned int c = warp * 4 + stage_sub; c < MMQ_BN; c += warps * 4) {
            const unsigned int gcol = col0 + c;
            int packed = 0;
            float d = 0.0f;
            if (gcol < N) {
                const unsigned char* blk =
                    weight + ((unsigned long long)gcol * bpr + b) * 34;
                d = __half2float(*reinterpret_cast<const __half*>(blk));
                // The quants start at byte 2, so only 2-byte alignment holds.
                packed = load_int_ua(blk + 2 + stage_k4 * 4);
            }
            s_w[stage_k4][c] = packed;
            if (stage_k4 == 0) s_wd[c] = d;
        }

        for (unsigned int r = warp * 4 + stage_sub; r < MMQ_BM; r += warps * 4) {
            const unsigned int grow = row0 + r;
            int packed = 0;
            float d = 0.0f;
            if (grow < M) {
                // Q8_1: d, then the block sum, then 32 quants at byte 4.
                const unsigned char* blk =
                    q8_act + ((unsigned long long)grow * bpr + b) * 36;
                d = __half2float(*reinterpret_cast<const __half*>(blk));
                packed = *reinterpret_cast<const int*>(blk + 4 + stage_k4 * 4);
            }
            s_a[stage_k4][r] = packed;
            if (stage_k4 == 0) s_ad[r] = d;
        }

        __syncthreads();

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

// Q4_K x Q8_1 -> f32 (MMQ), tensor-core variant of `quant_mmq_q4_k_q8_1` in
// `quant_gemv.cu`. Staging is copied verbatim; only the compute loop and the
// output write use `mma_m16n8k32_s8` instead of `dp4a`.
//
// Q4_K adds an asymmetric minimum term, `-dmin * minimum * rowsum`, that one
// `mma` cannot produce: it depends only on the row, not the column, so it is
// reduced separately with `dp4a` the same way the dp4a kernel reduces it.
extern "C" __global__ __launch_bounds__(MMQ_THREADS, 1) void quant_mmq_q4_k_q8_1_mma(
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
    const unsigned int warps = MMQ_THREADS / WARP_SIZE;

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

    for (unsigned int b = 0; b < sub_blocks; ++b) {
        const unsigned int sup = b / 8;
        const unsigned int j = b % 8;

        __syncthreads();

        for (unsigned int c = warp * 4 + stage_sub; c < MMQ_BN; c += warps * 4) {
            const unsigned int gcol = col0 + c;
            int packed = 0;
            float sd = 0.0f;
            float sm = 0.0f;
            if (gcol < N) {
                const unsigned char* blk =
                    weight + ((unsigned long long)gcol * supers + sup) * 144;
                const float d = __half2float(*reinterpret_cast<const __half*>(blk));
                const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
                int scale;
                int minimum;
                q4k_scale_min(blk + 4, (int)j, &scale, &minimum);
                sd = d * (float)scale;
                sm = dmin * (float)minimum;
                // 144 is 16-aligned and every offset here is a multiple of 4,
                // so a plain int load is aligned.
                const int v = *reinterpret_cast<const int*>(
                    blk + 16 + (j / 2) * 32 + stage_k4 * 4);
                packed = (j & 1) ? ((v >> 4) & 0x0F0F0F0F) : (v & 0x0F0F0F0F);
            }
            s_w[stage_k4][c] = packed;
            if (stage_k4 == 0) {
                s_wd[c] = sd;
                s_wm[c] = sm;
            }
        }

        for (unsigned int r = warp * 4 + stage_sub; r < MMQ_BM; r += warps * 4) {
            const unsigned int grow = row0 + r;
            int packed = 0;
            float d = 0.0f;
            if (grow < M) {
                const unsigned char* blk =
                    q8_act + ((unsigned long long)grow * sub_blocks + b) * 36;
                d = __half2float(*reinterpret_cast<const __half*>(blk));
                packed = *reinterpret_cast<const int*>(blk + 4 + stage_k4 * 4);
            }
            s_a[stage_k4][r] = packed;
            if (stage_k4 == 0) s_ad[r] = d;
        }

        __syncthreads();

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
