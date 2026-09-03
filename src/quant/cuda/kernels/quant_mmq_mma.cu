// Q8_0 x Q8_1 -> f32 (MMQ), tensor-core variant of `quant_mmq_q8_0_q8_1` in
// `quant_gemv.cu`. Staging is copied verbatim; only the compute loop and the
// output write use `mma_m16n8k32_s8` instead of `dp4a`.
//
// This is a separate translation unit, so the tile constants and
// `load_int_ua` are re-declared rather than shared via a header.

#include <cuda_fp16.h>

#include "mma_int8.cuh"

#define WARP_SIZE 32
#define MMQ_BM 128
#define MMQ_BN 64
#define MMQ_BK 32
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
    __shared__ int s_w[MMQ_K4][MMQ_BN];
    __shared__ int s_a[MMQ_K4][MMQ_BM];
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
