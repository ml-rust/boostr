// Q6_K x Q8_1 -> f32, tensor-core MMQ (sm_80+): the feature-major entry points
// `quant_mmq_q6_k_q8_1_mma*` of one weight format, module `quant_mmq_mma_q6_k`.
// The format policy is `formats/q6_k.cuh`; the family's shared machinery is in
// the headers beside this file.
//
// Also the token-major tensor-core kernel `quant_mmq_q6_k_q8_1_mma`.

#include "formats/q6_k.cuh"
#include "entry_points.cuh"
#include "token_major.cuh"

MMQ_FM_KERNELS(MmqfQ6K, q6_k)
MMQ_FM_KERNELS_NARROW(MmqfQ6K, q6_k)

// Reads Q8_1 sub-block `b` of both operands into registers, one iteration
// ahead, same reason as `mmq_q8_0_stage_load` in `q8_0.cu`.
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
