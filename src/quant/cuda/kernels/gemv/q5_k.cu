// Q5_K GEMV kernels — F32 activation + dp4a MWR + fused SwiGLU
//
// Q5_K block: 256 elements, 176 bytes
// Layout: [d:f16(2), dmin:f16(2), sc:12B, qh:32B, qs:128B]
// 8 sub-blocks of 32, 5-bit values (4-bit low nibble + 1-bit from qh)
// Scales/mins: same 6-bit packing as Q4_K (shared unpack_q4k_q5k_scales)

#include "common.cuh"

#define Q5K_BLOCK_SIZE 256
#define Q5K_BLOCK_BYTES 176

// ============================================================================
// Q5_K GEMV (F32 activation) — warp-per-column
// ============================================================================

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_q5_k_f32(
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

    const unsigned int blocks_per_row = K / Q5K_BLOCK_SIZE;
    const unsigned int row_bytes = blocks_per_row * Q5K_BLOCK_BYTES;
    const float* act_row = activation + m * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * Q5K_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const __half*>(block));
        float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
        const unsigned char* sc = block + 4;
        const unsigned char* qh = block + 16;
        const unsigned char* qs = block + 48;
        unsigned int base = b * Q5K_BLOCK_SIZE;

        unsigned char scales[8], mins[8];
        unpack_q4k_q5k_scales(sc, scales, mins);

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float dl = d * (float)scales[j];
            float ml = dmin * (float)mins[j];
            int idx = j * 32 + lane_id;
            int qs_idx = j * 16 + lane_id / 2;
            int low4 = (lane_id % 2 == 0)
                ? (qs[qs_idx] & 0x0F)
                : ((qs[qs_idx] >> 4) & 0x0F);
            int high1 = (qh[idx / 8] >> (idx % 8)) & 0x01;
            float q = (float)(low4 | (high1 << 4));
            acc += act_row[base + idx] * (dl * q - ml);
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane_id == 0) output[m * N + col] = acc;
}

// ============================================================================
// Q5_K GEMV with dp4a MWR (Q8_1 activation)
//
// Q5_K has the same scale structure as Q4_K but adds 32 bytes of high bits (qh).
// Lane mapping: chunk = lane/8, pos = (lane%8)*4
// Each lane reads 4 bytes of qs (8 nibbles for 2 sub-blocks) + high bits from qh.
//
// The 5th bit is packed in qh: bit (j*32+l) is the high bit for element j*32+l.
// For dp4a we need to construct packed int8 values with the 5th bit included.
// ============================================================================

extern "C" __global__ __launch_bounds__(128, 1) void quant_gemv_q5_k_q8_1_mwr(
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

    const int q5k_bpr = K / Q5K_BLOCK_SIZE;
    const int q8_bpr = K / 32;

    const unsigned char* w_row = weight + (unsigned long long)col * q5k_bpr * Q5K_BLOCK_BYTES;
    const unsigned char* q8_row = q8_act + (unsigned long long)m * q8_bpr * 36;

    const int chunk = lane_id / 8;
    const int pos = (lane_id % 8) * 4;
    const int j_lo = chunk * 2;
    const int j_hi = chunk * 2 + 1;

    float acc = 0.0f;

    for (int b = warp_id; b < q5k_bpr; b += NWARPS_K) {
        const unsigned char* q5k = w_row + b * Q5K_BLOCK_BYTES;
        float d5 = __half2float(*(const __half*)q5k);
        float dmin5 = __half2float(*(const __half*)(q5k + 2));
        const unsigned char* sc = q5k + 4;
        const unsigned char* qh = q5k + 16;
        const unsigned char* qs = q5k + 48;

        unsigned char scale_lo, scale_hi, min_lo, min_hi;
        unpack_scales_mwr(sc, j_lo, &scale_lo, &scale_hi, &min_lo, &min_hi);

        // Load 4 bytes of qs (8 nibbles covering j_lo and j_hi sub-blocks)
        int v = *(const int*)(qs + lane_id * 4);
        int v_lo = v & 0x0F0F0F0F;
        int v_hi = (v >> 4) & 0x0F0F0F0F;

        // Extract high bits from qh for the 4 elements this lane covers
        // j_lo sub-block: elements at indices j_lo*32 + pos..pos+3
        // j_hi sub-block: elements at indices j_hi*32 + pos..pos+3
        int qh_lo = 0, qh_hi = 0;
        for (int i = 0; i < 4; i++) {
            int idx_lo = j_lo * 32 + pos + i;
            int idx_hi = j_hi * 32 + pos + i;
            int h_lo = (qh[idx_lo / 8] >> (idx_lo % 8)) & 1;
            int h_hi = (qh[idx_hi / 8] >> (idx_hi % 8)) & 1;
            qh_lo |= (h_lo << 4) << (i * 8);
            qh_hi |= (h_hi << 4) << (i * 8);
        }
        v_lo |= qh_lo;  // 5-bit values packed as int8x4
        v_hi |= qh_hi;

        // Q8_1 activation
        int q8_idx_lo = b * 8 + j_lo;
        int q8_idx_hi = b * 8 + j_hi;
        float d8_lo = __half2float(*(const __half*)(q8_row + q8_idx_lo * 36));
        float d8_hi = __half2float(*(const __half*)(q8_row + q8_idx_hi * 36));
        int u_lo = *(const int*)(q8_row + q8_idx_lo * 36 + 4 + pos);
        int u_hi = *(const int*)(q8_row + q8_idx_hi * 36 + 4 + pos);

        int dot_lo = dp4a(v_lo, u_lo, 0);
        int dot_hi = dp4a(v_hi, u_hi, 0);
        int sumi_lo = dp4a(0x01010101, u_lo, 0);
        int sumi_hi = dp4a(0x01010101, u_hi, 0);

        acc += d5 * d8_lo * (float)(dot_lo * (int)scale_lo)
             + d5 * d8_hi * (float)(dot_hi * (int)scale_hi)
             - dmin5 * d8_lo * (float)(sumi_lo * (int)min_lo)
             - dmin5 * d8_hi * (float)(sumi_hi * (int)min_hi);
    }

    __shared__ float smem[NWARPS_K][WARP_SIZE];
    float sum = mwr_reduce(acc, warp_id, lane_id, smem);
    if (warp_id == 0 && lane_id == 0) output[m * N + col] = sum;
}

// ============================================================================
// Fused SwiGLU GEMV for Q5_K (dp4a MWR)
// ============================================================================

extern "C" __global__ __launch_bounds__(128, 1) void fused_swiglu_q5k_q8_1_mwr(
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

    const int q5k_bpr = K / Q5K_BLOCK_SIZE;
    const int q8_bpr = K / 32;

    const unsigned char* g_row = gate_weight + (unsigned long long)col * q5k_bpr * Q5K_BLOCK_BYTES;
    const unsigned char* u_row = up_weight   + (unsigned long long)col * q5k_bpr * Q5K_BLOCK_BYTES;
    const unsigned char* q8_row = q8_act + (unsigned long long)m * q8_bpr * 36;

    const int chunk = lane_id / 8;
    const int pos = (lane_id % 8) * 4;
    const int j_lo = chunk * 2;
    const int j_hi = chunk * 2 + 1;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int b = warp_id; b < q5k_bpr; b += NWARPS_K) {
        // Shared Q8_1 activation
        int q8_idx_lo = b * 8 + j_lo;
        int q8_idx_hi = b * 8 + j_hi;
        float d8_lo = __half2float(*(const __half*)(q8_row + q8_idx_lo * 36));
        float d8_hi = __half2float(*(const __half*)(q8_row + q8_idx_hi * 36));
        int u_lo = *(const int*)(q8_row + q8_idx_lo * 36 + 4 + pos);
        int u_hi = *(const int*)(q8_row + q8_idx_hi * 36 + 4 + pos);
        int sumi_lo = dp4a(0x01010101, u_lo, 0);
        int sumi_hi = dp4a(0x01010101, u_hi, 0);

        // Process gate and up weights with same pattern
        #pragma unroll
        for (int proj = 0; proj < 2; proj++) {
            const unsigned char* w_row_p = (proj == 0) ? g_row : u_row;
            const unsigned char* q5k = w_row_p + b * Q5K_BLOCK_BYTES;
            float d5 = __half2float(*(const __half*)q5k);
            float dmin5 = __half2float(*(const __half*)(q5k + 2));
            const unsigned char* sc = q5k + 4;
            const unsigned char* qh = q5k + 16;
            const unsigned char* qs = q5k + 48;

            unsigned char scale_lo, scale_hi, min_lo, min_hi;
            unpack_scales_mwr(sc, j_lo, &scale_lo, &scale_hi, &min_lo, &min_hi);

            int v = *(const int*)(qs + lane_id * 4);
            int v_lo = v & 0x0F0F0F0F;
            int v_hi = (v >> 4) & 0x0F0F0F0F;

            int qh_lo = 0, qh_hi = 0;
            for (int i = 0; i < 4; i++) {
                int idx_lo = j_lo * 32 + pos + i;
                int idx_hi = j_hi * 32 + pos + i;
                int h_lo = (qh[idx_lo / 8] >> (idx_lo % 8)) & 1;
                int h_hi = (qh[idx_hi / 8] >> (idx_hi % 8)) & 1;
                qh_lo |= (h_lo << 4) << (i * 8);
                qh_hi |= (h_hi << 4) << (i * 8);
            }
            v_lo |= qh_lo;
            v_hi |= qh_hi;

            int dot_lo = dp4a(v_lo, u_lo, 0);
            int dot_hi = dp4a(v_hi, u_hi, 0);

            float result = d5 * d8_lo * (float)(dot_lo * (int)scale_lo)
                         + d5 * d8_hi * (float)(dot_hi * (int)scale_hi)
                         - dmin5 * d8_lo * (float)(sumi_lo * (int)min_lo)
                         - dmin5 * d8_hi * (float)(sumi_hi * (int)min_hi);

            if (proj == 0) gate_acc += result;
            else           up_acc += result;
        }
    }

    __shared__ float smem[2][NWARPS_K][WARP_SIZE];
    float gate_sum, up_sum;
    mwr_reduce_dual(gate_acc, up_acc, warp_id, lane_id, smem, &gate_sum, &up_sum);
    if (warp_id == 0 && lane_id == 0)
        output[m * N + col] = silu_f(gate_sum) * up_sum;
}
