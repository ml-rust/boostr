// Q3_K GEMV kernels — F32 activation + dp4a MWR + fused SwiGLU
//
// Q3_K block: 256 elements, 110 bytes
// Layout: [hmask:32B, qs:64B, scales:12B, d:f16(2)]
// 2-bit values with 1-bit high mask, 16 signed 6-bit scales

#include "common.cuh"

#define Q3K_BLOCK_SIZE 256
#define Q3K_BLOCK_BYTES 110

// ============================================================================
// Q3_K GEMV (F32 activation) — warp-per-column
// ============================================================================

extern "C" __global__ __launch_bounds__(256, 1) void quant_gemv_q3_k_f32(
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

    const unsigned int blocks_per_row = K / Q3K_BLOCK_SIZE;
    const unsigned int row_bytes = blocks_per_row * Q3K_BLOCK_BYTES;
    const float* act_row = activation + m * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * Q3K_BLOCK_BYTES;
        const unsigned char* hmask = block;
        const unsigned char* qs = block + 32;
        const unsigned char* sc_raw = block + 96;
        float d = __half2float(*reinterpret_cast<const __half*>(block + 108));
        unsigned int base = b * Q3K_BLOCK_SIZE;

        signed char scales[16];
        unpack_q3k_scales(sc_raw, scales);

        int y = 0, is = 0;
        unsigned char mask = 1;
        for (int n_half = 0; n_half < 2; n_half++) {
            const unsigned char* q = qs + n_half * 32;
            for (int shift = 0; shift < 8; shift += 2) {
                float dl = d * (float)scales[is++];
                // Process 32 elements, lane handles one from each 16-group
                if (lane_id < 16) {
                    int low2 = (q[lane_id] >> shift) & 3;
                    int hsub = (hmask[lane_id] & mask) ? 0 : 4;
                    acc += act_row[base + y + lane_id] * dl * (float)(low2 - hsub);
                }
                y += 16;
                dl = d * (float)scales[is++];
                if (lane_id < 16) {
                    int low2 = (q[16 + lane_id] >> shift) & 3;
                    int hsub = (hmask[16 + lane_id] & mask) ? 0 : 4;
                    acc += act_row[base + y + lane_id] * dl * (float)(low2 - hsub);
                }
                y += 16;
                mask <<= 1;
            }
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane_id == 0) output[m * N + col] = acc;
}

// ============================================================================
// Q3_K GEMV with dp4a MWR (Q8_1 activation)
//
// Q3_K values are 3-bit: 2 low bits from qs + 1 high bit from hmask.
// The value is: low2 - (hmask_bit ? 0 : 4), i.e., range [-4, 3].
//
// For dp4a we pack 4 signed 3-bit values into int8x4.
// Each lane in a chunk processes 4 elements from one 16-element sub-block.
//
// Layout per Q3_K block: 16 sub-blocks of 16 elements.
// The block is organized as 2 halves × 4 shifts × 2 groups of 16.
// block_in_pass (0..3) selects which Q8_1 block within a half.
// ============================================================================

extern "C" __global__ __launch_bounds__(128, 1) void quant_gemv_q3_k_q8_1_mwr(
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

    const int q3k_bpr = K / Q3K_BLOCK_SIZE;
    const int q8_bpr = K / 32;

    const unsigned char* w_row = weight + (unsigned long long)col * q3k_bpr * Q3K_BLOCK_BYTES;
    const unsigned char* q8_row = q8_act + (unsigned long long)m * q8_bpr * 36;

    // Map lanes to Q8_1 sub-blocks within a Q3_K block.
    // Each Q3_K block has 8 Q8_1 blocks (256/32). Two halves of 4.
    const int block_in_pass = lane_id / 8;   // 0..3
    const int pos = (lane_id % 8) * 4;       // 0,4,...,28

    float acc = 0.0f;

    for (int b = warp_id; b < q3k_bpr; b += NWARPS_K) {
        const unsigned char* blk = w_row + b * Q3K_BLOCK_BYTES;
        const unsigned char* hmask = blk;
        const unsigned char* qs = blk + 32;
        const unsigned char* sc_raw = blk + 96;
        __half d_h;
        memcpy(&d_h, blk + 108, 2);
        float d3 = __half2float(d_h);

        signed char scales[16];
        unpack_q3k_scales(sc_raw, scales);

        // Process 2 halves, each with 4 Q8_1 blocks
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            int q8_idx = b * 8 + half * 4 + block_in_pass;
            float d8 = __half2float(*(const __half*)(q8_row + q8_idx * 36));
            int u = *(const int*)(q8_row + q8_idx * 36 + 4 + pos);

            // block_in_pass = shift_idx: each Q8_1 block covers one shift step (32 elems)
            // block_in_pass 0 → shift=0, elements 0-31 (both sub_grps)
            // block_in_pass 1 → shift=2, elements 32-63
            // block_in_pass 2 → shift=4, elements 64-95
            // block_in_pass 3 → shift=6, elements 96-127
            int shift = block_in_pass * 2;   // 0, 2, 4, 6
            int qs_off = half * 32;
            unsigned char hmask_bit = 1 << (half * 4 + block_in_pass);

            // pos determines sub_grp: 0-15 = sub_grp 0, 16-31 = sub_grp 1
            int sub_grp = (pos >= 16) ? 1 : 0;
            int scale_idx = half * 8 + block_in_pass * 2 + sub_grp;
            float dl = d3 * (float)scales[scale_idx];

            // Pack 4 Q3_K signed values for dp4a
            int q3_packed = 0;
            for (int i = 0; i < 4; i++) {
                int elem = pos + i;
                // Map to qs/hmask: first 16 elems → q[0..15], next 16 → q[16..31]
                int qs_elem = (elem < 16) ? elem : (elem - 16 + 16);
                int low2 = (qs[qs_off + qs_elem] >> shift) & 3;
                int hsub = (hmask[qs_elem] & hmask_bit) ? 0 : 4;
                int val = low2 - hsub;  // range [-4, 3]
                q3_packed |= (val & 0xFF) << (i * 8);
            }

            int dot = dp4a(q3_packed, u, 0);
            acc += dl * d8 * (float)dot;
        }
    }

    __shared__ float smem[NWARPS_K][WARP_SIZE];
    float sum = mwr_reduce(acc, warp_id, lane_id, smem);
    if (warp_id == 0 && lane_id == 0) output[m * N + col] = sum;
}

// ============================================================================
// Fused SwiGLU GEMV for Q3_K (dp4a MWR)
// ============================================================================

extern "C" __global__ __launch_bounds__(128, 1) void fused_swiglu_q3k_q8_1_mwr(
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

    const int q3k_bpr = K / Q3K_BLOCK_SIZE;
    const int q8_bpr = K / 32;

    const unsigned char* g_row = gate_weight + (unsigned long long)col * q3k_bpr * Q3K_BLOCK_BYTES;
    const unsigned char* u_row = up_weight   + (unsigned long long)col * q3k_bpr * Q3K_BLOCK_BYTES;
    const unsigned char* q8_row = q8_act + (unsigned long long)m * q8_bpr * 36;

    const int block_in_pass = lane_id / 8;
    const int pos = (lane_id % 8) * 4;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int b = warp_id; b < q3k_bpr; b += NWARPS_K) {
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            int q8_idx = b * 8 + half * 4 + block_in_pass;
            float d8 = __half2float(*(const __half*)(q8_row + q8_idx * 36));
            int u = *(const int*)(q8_row + q8_idx * 36 + 4 + pos);

            int shift = block_in_pass * 2;   // 0, 2, 4, 6
            int qs_off = half * 32;
            unsigned char hmask_bit = 1 << (half * 4 + block_in_pass);
            int sub_grp = (pos >= 16) ? 1 : 0;
            int scale_idx = half * 8 + block_in_pass * 2 + sub_grp;

            #pragma unroll
            for (int proj = 0; proj < 2; proj++) {
                const unsigned char* w_r = (proj == 0) ? g_row : u_row;
                const unsigned char* blk = w_r + b * Q3K_BLOCK_BYTES;
                const unsigned char* hmask = blk;
                const unsigned char* qs = blk + 32;
                const unsigned char* sc_raw = blk + 96;
                __half d_h; memcpy(&d_h, blk + 108, 2);
                float d3 = __half2float(d_h);

                signed char scales[16];
                unpack_q3k_scales(sc_raw, scales);

                float dl = d3 * (float)scales[scale_idx];

                int q3_packed = 0;
                for (int i = 0; i < 4; i++) {
                    int elem = pos + i;
                    int qs_elem = (elem < 16) ? elem : (elem - 16 + 16);
                    int low2 = (qs[qs_off + qs_elem] >> shift) & 3;
                    int hsub = (hmask[qs_elem] & hmask_bit) ? 0 : 4;
                    int val = low2 - hsub;
                    q3_packed |= (val & 0xFF) << (i * 8);
                }

                float result = dl * d8 * (float)dp4a(q3_packed, u, 0);
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
