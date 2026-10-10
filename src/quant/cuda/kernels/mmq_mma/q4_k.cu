// Q4_K x Q8_1 -> f32, tensor-core MMQ (sm_80+): the feature-major entry points
// `quant_mmq_q4_k_q8_1_mma*` of one weight format, module `quant_mmq_mma_q4_k`.
// The format policy is `formats/q4_k.cuh`; the family's shared machinery is in
// the headers beside this file.
//
// Also the token-major tensor-core kernel `quant_mmq_q4_k_q8_1_mma`.

#include "formats/q4_k.cuh"
#include "entry_points.cuh"
#include "token_major.cuh"

MMQ_FM_KERNELS(MmqfQ4K, q4_k)
MMQ_FM_KERNELS_NARROW(MmqfQ4K, q4_k)

// Reads Q8_1 sub-block `b` of both operands into registers, one iteration
// ahead, same reason as `mmq_q8_0_stage_load` in `q8_0.cu`.
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
