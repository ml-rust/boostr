// Q8_0 x Q8_1 -> f32, tensor-core MMQ (sm_80+): the feature-major entry points
// `quant_mmq_q8_0_q8_1_mma*` of one weight format, module `quant_mmq_mma_q8_0`.
// The format policy is `formats/q8_0.cuh`; the family's shared machinery is in
// the headers beside this file.
//
// Also the token-major tensor-core kernel `quant_mmq_q8_0_q8_1_mma`.

#include "formats/q8_0.cuh"
#include "entry_points.cuh"
#include "token_major.cuh"

MMQ_FM_KERNELS(MmqfQ80, q8_0)

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

// Q8_0 x Q8_1 -> f32 (MMQ), tensor-core variant of `quant_mmq_q8_0_q8_1` in
// `quant_gemv.cu`. The compute loop and the output write use
// `mma_m16n8k32_s8` instead of `dp4a`.
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
