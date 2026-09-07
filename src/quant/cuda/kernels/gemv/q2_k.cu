// Q2_K GEMV kernels — F32 activation + dp4a MWR + fused SwiGLU
//
// Q2_K block: 256 elements, 84 bytes
// Layout: [sc:16B, qs:64B, d:f16(2), dmin:f16(2)]
// 16 sub-blocks of 16, 2-bit values
// sc[i]: low 4 bits = sub-scale, high 4 bits = sub-min

#include "common.cuh"

#define Q2K_BLOCK_SIZE 256
#define Q2K_BLOCK_BYTES 84

// ============================================================================
// Q2_K GEMV (F32 activation) — warp-per-column
// ============================================================================

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_q2_k_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const unsigned int m = blockIdx.y;
    if (col >= N) return;

    const unsigned int blocks_per_row = K / Q2K_BLOCK_SIZE;
    const unsigned int row_bytes = blocks_per_row * Q2K_BLOCK_BYTES;
    const float* act_row = activation + m * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * Q2K_BLOCK_BYTES;
        const unsigned char* sc = block;
        const unsigned char* qs = block + 16;
        float d = __half2float(*reinterpret_cast<const __half*>(block + 80));
        float dmin = __half2float(*reinterpret_cast<const __half*>(block + 82));
        unsigned int base = b * Q2K_BLOCK_SIZE;

        int y = 0, is = 0;
        for (int n_half = 0; n_half < 2; n_half++) {
            const unsigned char* q = qs + n_half * 32;
            for (int shift = 0; shift < 8; shift += 2) {
                float dl = d * (float)(sc[is] & 0x0F);
                float ml = dmin * (float)(sc[is] >> 4);
                is++;
                if (lane_id < 16)
                    acc += act_row[base + y + lane_id] * (dl * (float)((q[lane_id] >> shift) & 3) - ml);
                y += 16;
                dl = d * (float)(sc[is] & 0x0F);
                ml = dmin * (float)(sc[is] >> 4);
                is++;
                if (lane_id < 16)
                    acc += act_row[base + y + lane_id] * (dl * (float)((q[16 + lane_id] >> shift) & 3) - ml);
                y += 16;
            }
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane_id == 0) output[m * N + col] = acc;
}

// ============================================================================
// Q2_K GEMV with dp4a MWR (Q8_1 activation)
//
// Q2_K has 2-bit values packed 4 per byte. Each Q8_1 block covers 32 elements,
// which is 2 Q2_K sub-blocks of 16.
// block_in_pass (0..3) maps to Q8_1 blocks within each half.
// ============================================================================

extern "C" __global__ __launch_bounds__(128, 1) void quant_gemv_q2_k_q8_1_mwr(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int col = blockIdx.x;
    const int m = blockIdx.y;
    if (col >= (int)N) return;

    const int q2k_bpr = K / Q2K_BLOCK_SIZE;
    const int q8_bpr = K / 32;

    const unsigned char* w_row = weight + (unsigned long long)col * q2k_bpr * Q2K_BLOCK_BYTES;
    const unsigned char* q8_row = q8_act + (unsigned long long)m * q8_bpr * 36;

    const int block_in_pass = lane_id / 8;   // 0..3
    const int pos = (lane_id % 8) * 4;       // 0,4,...,28

    float acc = 0.0f;

    for (int b = warp_id; b < q2k_bpr; b += NWARPS_K) {
        const unsigned char* blk = w_row + b * Q2K_BLOCK_BYTES;
        const unsigned char* sc = blk;
        const unsigned char* qs = blk + 16;
        __half d_h, dmin_h;
        memcpy(&d_h, blk + 80, 2);
        memcpy(&dmin_h, blk + 82, 2);
        float d2 = __half2float(d_h);
        float dmin2 = __half2float(dmin_h);

        #pragma unroll
        for (int half = 0; half < 2; half++) {
            int q8_idx = b * 8 + half * 4 + block_in_pass;
            float d8 = __half2float(*(const __half*)(q8_row + q8_idx * 36));
            int u = *(const int*)(q8_row + q8_idx * 36 + 4 + pos);

            // Each Q8_1 block covers 32 elements = 2 Q2_K sub-blocks of 16.
            // block_in_pass (0..3) selects the shift (0, 2, 4, 6) within a half.
            // Each shift extracts 2 bits from 32 bytes → 32 elements (16 per sub-group).
            int shift = block_in_pass * 2;   // 0, 2, 4, 6
            int qs_off = half * 32;

            // Scale indices: 2 per Q8_1 block (one per 16-element sub-group)
            int sc_idx0 = half * 8 + block_in_pass * 2;
            int sc_idx1 = sc_idx0 + 1;

            float dl0 = d2 * (float)(sc[sc_idx0] & 0x0F);
            float ml0 = dmin2 * (float)(sc[sc_idx0] >> 4);
            float dl1 = d2 * (float)(sc[sc_idx1] & 0x0F);
            float ml1 = dmin2 * (float)(sc[sc_idx1] >> 4);

            // Pack 4 Q2_K values for dp4a
            // pos covers 4 consecutive elements within 32-element Q8_1 block
            // First 16 elements: qs[qs_off + elem], second 16: qs[qs_off + elem]
            int q2_packed = 0;
            int sumi = dp4a(0x01010101, u, 0);
            for (int i = 0; i < 4; i++) {
                int elem = pos + i;
                int val = (qs[qs_off + (elem % 16) + (elem >= 16 ? 16 : 0)] >> shift) & 3;
                q2_packed |= (val & 0xFF) << (i * 8);
            }

            int dot = dp4a(q2_packed, u, 0);

            // Use first sub-group's scale if pos < 16, second if pos >= 16
            float dl = (pos < 16) ? dl0 : dl1;
            float ml = (pos < 16) ? ml0 : ml1;
            acc += d8 * ((float)dot * dl - ml * (float)sumi);
        }
    }

    __shared__ float smem[NWARPS_K][WARP_SIZE];
    float sum = mwr_reduce(acc, warp_id, lane_id, smem);
    if (warp_id == 0 && lane_id == 0) output[m * N + col] = sum;
}

// ============================================================================
// Fused SwiGLU GEMV for Q2_K (dp4a MWR)
// ============================================================================

extern "C" __global__ __launch_bounds__(128, 1) void fused_swiglu_q2k_q8_1_mwr(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ gate_weight,
    const unsigned char* __restrict__ up_weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int col = blockIdx.x;
    const int m = blockIdx.y;
    if (col >= (int)N) return;

    const int q2k_bpr = K / Q2K_BLOCK_SIZE;
    const int q8_bpr = K / 32;

    const unsigned char* g_row = gate_weight + (unsigned long long)col * q2k_bpr * Q2K_BLOCK_BYTES;
    const unsigned char* u_row = up_weight   + (unsigned long long)col * q2k_bpr * Q2K_BLOCK_BYTES;
    const unsigned char* q8_row = q8_act + (unsigned long long)m * q8_bpr * 36;

    const int block_in_pass = lane_id / 8;
    const int pos = (lane_id % 8) * 4;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int b = warp_id; b < q2k_bpr; b += NWARPS_K) {
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            int q8_idx = b * 8 + half * 4 + block_in_pass;
            float d8 = __half2float(*(const __half*)(q8_row + q8_idx * 36));
            int u = *(const int*)(q8_row + q8_idx * 36 + 4 + pos);
            int sumi = dp4a(0x01010101, u, 0);

            int shift = block_in_pass * 2;   // 0, 2, 4, 6
            int qs_off = half * 32;
            int sc_idx0 = half * 8 + block_in_pass * 2;
            int sc_idx1 = sc_idx0 + 1;

            #pragma unroll
            for (int proj = 0; proj < 2; proj++) {
                const unsigned char* w_r = (proj == 0) ? g_row : u_row;
                const unsigned char* blk = w_r + b * Q2K_BLOCK_BYTES;
                const unsigned char* sc = blk;
                const unsigned char* qs = blk + 16;
                __half d_h, dmin_h;
                memcpy(&d_h, blk + 80, 2);
                memcpy(&dmin_h, blk + 82, 2);
                float d2 = __half2float(d_h);
                float dmin2 = __half2float(dmin_h);

                float dl0 = d2 * (float)(sc[sc_idx0] & 0x0F);
                float ml0 = dmin2 * (float)(sc[sc_idx0] >> 4);
                float dl1 = d2 * (float)(sc[sc_idx1] & 0x0F);
                float ml1 = dmin2 * (float)(sc[sc_idx1] >> 4);

                int q2_packed = 0;
                for (int i = 0; i < 4; i++) {
                    int elem = pos + i;
                    int val = (qs[qs_off + (elem % 16) + (elem >= 16 ? 16 : 0)] >> shift) & 3;
                    q2_packed |= (val & 0xFF) << (i * 8);
                }
                int dot = dp4a(q2_packed, u, 0);
                float dl = (pos < 16) ? dl0 : dl1;
                float ml = (pos < 16) ? ml0 : ml1;
                float result = d8 * ((float)dot * dl - ml * (float)sumi);

                if (proj == 0) gate_acc += result;
                else           up_acc += result;
            }
        }
    }

    __shared__ float smem[2][NWARPS_K][WARP_SIZE];
    float gate_sum, up_sum;
    mwr_reduce_dual(gate_acc, up_acc, warp_id, lane_id, smem, &gate_sum, &up_sum);
    if (warp_id == 0 && lane_id == 0)
        output[m * N + col] = silu_f(gate_sum) * up_sum;
}
