// Quantized GEMV CUDA kernels for boostr — optimized for LLM decode (M=1)
//
// Two variants:
//   1. F32 activation: simple warp-per-column, float multiply-add
//   2. Q8_1 activation + dp4a: 4x compute throughput via integer SIMD
//
// Weight layout: [N, K] packed as quantized blocks along K axis
// Output: [M, N] f32

#include <cuda_fp16.h>

#define WARP_SIZE 32

// ============================================================================
// dp4a intrinsic — 4-element int8 dot product in a single instruction
// Available on compute capability >= 6.1 (Pascal GP102+, all Turing/Ampere/etc)
// ============================================================================

// 2-byte-aligned 4-byte load (Q6_K blocks are 210 bytes — always 2-byte aligned,
// and all internal offsets are even, so uint16 loads are safe)
static __device__ __forceinline__ int load_int_ua(const unsigned char* p) {
    const unsigned short* p16 = (const unsigned short*)p;
    return (int)p16[0] | ((int)p16[1] << 16);
}

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const signed char* a8 = (const signed char*)&a;
    const signed char* b8 = (const signed char*)&b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

// ============================================================================
// Q4_0 GEMV (F32 activation)
// ============================================================================

#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)

extern "C" __global__ void quant_gemv_q4_0_f32(
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

    const unsigned int blocks_per_row = K / 32;
    const unsigned int row_bytes = blocks_per_row * 18;
    const float* act_row = activation + m * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 18;
        float d = __half2float(*reinterpret_cast<const __half*>(block));
        const unsigned char* qs = block + 2;
        unsigned int byte_idx = lane_id >> 1;
        unsigned int is_high = lane_id & 1;
        unsigned char bv = qs[byte_idx];
        float q = is_high ? (float)((int)((bv >> 4) & 0x0F) - 8) * d
                          : (float)((int)(bv & 0x0F) - 8) * d;
        acc += act_row[b * 32 + lane_id] * q;
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane_id == 0) output[m * N + col] = acc;
}

// ============================================================================
// Q8_0 GEMV (F32 activation)
// ============================================================================

extern "C" __global__ void quant_gemv_q8_0_f32(
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

    const unsigned int blocks_per_row = K / 32;
    const unsigned int row_bytes = blocks_per_row * 34;
    const float* act_row = activation + m * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 34;
        float d = __half2float(*reinterpret_cast<const __half*>(block));
        const signed char* qs = reinterpret_cast<const signed char*>(block + 2);
        acc += act_row[b * 32 + lane_id] * ((float)qs[lane_id] * d);
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane_id == 0) output[m * N + col] = acc;
}

// ============================================================================
// Q4_K GEMV (F32 activation) — fallback for non-dp4a GPUs
// ============================================================================

extern "C" __global__ void quant_gemv_q4_k_f32(
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

    const unsigned int blocks_per_row = K / 256;
    const unsigned int row_bytes = blocks_per_row * 144;
    const float* act_row = activation + m * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 144;
        float d = __half2float(*reinterpret_cast<const __half*>(block));
        float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
        const unsigned char* sc = block + 4;
        const unsigned char* qs = block + 16;
        unsigned int base = b * 256;

        unsigned char scales[8], mins[8];
        for (int i = 0; i < 4; i++) {
            scales[i] = sc[i] & 0x3F;
            mins[i] = sc[i + 4] & 0x3F;
        }
        for (int i = 4; i < 8; i++) {
            scales[i] = (sc[i + 4] & 0x0F) | ((sc[i - 4] >> 6) << 4);
            mins[i] = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4);
        }

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float dl = d * (float)scales[j];
            float ml = dmin * (float)mins[j];
            int chunk = j >> 1;
            int is_high = j & 1;
            float q = is_high ? (float)((qs[chunk * 32 + lane_id] >> 4) & 0x0F)
                              : (float)(qs[chunk * 32 + lane_id] & 0x0F);
            acc += act_row[base + j * 32 + lane_id] * (dl * q - ml);
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane_id == 0) output[m * N + col] = acc;
}

// ============================================================================
// Q4_K GEMV with dp4a (Q8_1 activation)
//
// Each warp handles 1 output column. 8 warps per block.
// Lane i reads 4 bytes of Q4_K qs data (= 8 nibbles = elements from 2 sub-groups)
// and 4 bytes of Q8_1 int8 data, computes dp4a dot products.
//
// 32 lanes × 4 bytes = 128 bytes = all qs data for one Q4_K block.
// Lane i is in chunk (i/8), at position (i%8)*4 within that chunk.
//
// Q8_1 block layout: [d (half, 2B), s (half, 2B), qs[32] (32B)] = 36 bytes
// ============================================================================

extern "C" __global__ void quant_gemv_q4_k_q8_1(
    const unsigned char* __restrict__ q8_act,  // Q8_1 quantized activation
    const unsigned char* __restrict__ weight,   // Q4_K weights [N, ...]
    float* __restrict__ output,                 // [M, N]
    unsigned int M, unsigned int K, unsigned int N
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const unsigned int m = blockIdx.y;
    if (col >= N) return;

    const unsigned int q4k_blocks_per_row = K / 256;
    const unsigned int q4k_row_bytes = q4k_blocks_per_row * 144;
    const unsigned int q8_blocks_per_row = K / 32;

    const unsigned char* w_row = weight + col * q4k_row_bytes;
    const unsigned char* q8_row = q8_act + m * q8_blocks_per_row * 36;

    // Which chunk (0-3) and position within chunk this lane handles
    const int chunk = lane_id / 8;        // 0..3
    const int pos = (lane_id % 8) * 4;    // 0,4,8,...,28

    float acc = 0.0f;

    for (unsigned int b = 0; b < q4k_blocks_per_row; b++) {
        const unsigned char* q4k_block = w_row + b * 144;
        float d4 = __half2float(*reinterpret_cast<const __half*>(q4k_block));
        float dmin4 = __half2float(*reinterpret_cast<const __half*>(q4k_block + 2));
        const unsigned char* sc = q4k_block + 4;
        const unsigned char* qs = q4k_block + 16;

        // Unpack 6-bit scales and mins
        unsigned char scales[8], mvals[8];
        for (int i = 0; i < 4; i++) {
            scales[i] = sc[i] & 0x3F;
            mvals[i] = sc[i + 4] & 0x3F;
        }
        for (int i = 4; i < 8; i++) {
            scales[i] = (sc[i + 4] & 0x0F) | ((sc[i - 4] >> 6) << 4);
            mvals[i] = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4);
        }

        // Load 4 bytes of Q4_K qs (one int32 = 8 nibbles)
        int v = *reinterpret_cast<const int*>(qs + lane_id * 4);
        int v_lo = v & 0x0F0F0F0F;
        int v_hi = (v >> 4) & 0x0F0F0F0F;

        int j_lo = chunk * 2;
        int j_hi = chunk * 2 + 1;

        // Q8_1 block indices (8 Q8_1 blocks per Q4_K block)
        int q8_idx_lo = b * 8 + j_lo;
        int q8_idx_hi = b * 8 + j_hi;

        // Load Q8_1 scales (d field at offset 0 in each 36-byte block)
        float d8_lo = __half2float(*reinterpret_cast<const __half*>(q8_row + q8_idx_lo * 36));
        float d8_hi = __half2float(*reinterpret_cast<const __half*>(q8_row + q8_idx_hi * 36));

        // Load 4 Q8_1 int8 values (at offset 4 in each 36-byte block, then +pos)
        int u_lo = *reinterpret_cast<const int*>(q8_row + q8_idx_lo * 36 + 4 + pos);
        int u_hi = *reinterpret_cast<const int*>(q8_row + q8_idx_hi * 36 + 4 + pos);

        // dp4a: dot product of 4 Q4_K nibbles × 4 Q8_1 int8 values
        int dot_lo = dp4a(v_lo, u_lo, 0);
        int dot_hi = dp4a(v_hi, u_hi, 0);

        // Sum of Q8_1 values (for min compensation)
        int sumi_lo = dp4a(0x01010101, u_lo, 0);
        int sumi_hi = dp4a(0x01010101, u_hi, 0);

        // Accumulate
        acc += d4 * d8_lo * (float)(dot_lo * (int)scales[j_lo])
             + d4 * d8_hi * (float)(dot_hi * (int)scales[j_hi])
             - dmin4 * d8_lo * (float)(sumi_lo * (int)mvals[j_lo])
             - dmin4 * d8_hi * (float)(sumi_hi * (int)mvals[j_hi]);
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);

    if (lane_id == 0) output[m * N + col] = acc;
}

// ============================================================================
// Q6_K GEMV with dp4a (Q8_1 activation)
//
// Each warp handles 1 output column. 8 warps per block.
// Q6_K block: 210 bytes, 256 elements in 2 halves of 128.
// Each half has 4 groups of 32 elements (q1..q4), each with its own scale.
//
// Lane mapping per half:
//   block_in_pass = lane/8 (0..3) → which group (q1/q2/q3/q4)
//   pos = (lane%8)*4 (0,4,...,28) → position within the 32-element group
//
// Q6_K element = 6-bit unsigned (0..63), stored as:
//   low 4 bits in ql[], high 2 bits in qh[]
// We compute dot(q6, q8) - 32*sum(q8) to handle the -32 offset.
// ============================================================================

extern "C" __global__ void quant_gemv_q6_k_q8_1(
    const unsigned char* __restrict__ q8_act,
    const unsigned char* __restrict__ weight,
    float* __restrict__ output,
    unsigned int M, unsigned int K, unsigned int N
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const unsigned int m = blockIdx.y;
    if (col >= N) return;

    const unsigned int q6k_blocks_per_row = K / 256;
    const unsigned int q6k_row_bytes = q6k_blocks_per_row * 210;
    const unsigned int q8_blocks_per_row = K / 32;

    const unsigned char* w_row = weight + col * q6k_row_bytes;
    const unsigned char* q8_row = q8_act + m * q8_blocks_per_row * 36;

    const int block_in_pass = lane_id / 8;   // 0..3 (q1/q2/q3/q4 group)
    const int pos = (lane_id % 8) * 4;       // 0,4,8,...,28
    const int is = pos >= 16 ? 1 : 0;        // scale sub-index

    float acc = 0.0f;

    for (unsigned int b = 0; b < q6k_blocks_per_row; b++) {
        const unsigned char* blk = w_row + b * 210;
        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;
        const signed char* sc = (const signed char*)(blk + 192);
        // d is f16 at offset 208; 210-byte blocks may be 2-byte aligned — safe for __half
        __half d6_h;
        memcpy(&d6_h, blk + 208, 2);
        float d6 = __half2float(d6_h);

        #pragma unroll
        for (int half = 0; half < 2; half++) {
            int ql_base = half * 64;
            int qh_base = half * 32;
            int sc_base = half * 8;

            int q8_idx = b * 8 + half * 4 + block_in_pass;

            // Load Q8_1 scale and 4 int8 activation values
            float d8 = __half2float(*(const __half*)(q8_row + q8_idx * 36));
            int u = *(const int*)(q8_row + q8_idx * 36 + 4 + pos);

            // Load ql and qh bytes (unaligned — Q6_K blocks are 210 bytes)
            int ql4_lo = load_int_ua(ql + ql_base + pos);
            int ql4_hi = load_int_ua(ql + ql_base + 32 + pos);
            int qh4    = load_int_ua(qh + qh_base + pos);

            // Unpack 4 unsigned Q6_K values (0..63) based on group
            int q6;
            int scale;
            switch (block_in_pass) {
                case 0:  // q1: ql low nibble + qh bits 0-1
                    q6 = (ql4_lo & 0x0F0F0F0F) | ((qh4 & 0x03030303) << 4);
                    scale = (int)sc[sc_base + is];
                    break;
                case 1:  // q2: ql[+32] low nibble + qh bits 2-3
                    q6 = (ql4_hi & 0x0F0F0F0F) | ((qh4 & 0x0C0C0C0C) << 2);
                    scale = (int)sc[sc_base + is + 2];
                    break;
                case 2:  // q3: ql high nibble + qh bits 4-5
                    q6 = ((ql4_lo >> 4) & 0x0F0F0F0F) | (qh4 & 0x30303030);
                    scale = (int)sc[sc_base + is + 4];
                    break;
                default: // q4: ql[+32] high nibble + qh bits 6-7
                    q6 = ((ql4_hi >> 4) & 0x0F0F0F0F) | ((qh4 & 0xC0C0C0C0) >> 2);
                    scale = (int)sc[sc_base + is + 6];
                    break;
            }

            // dp4a: dot(q6_unsigned, q8) and sum(q8)
            int dot = dp4a(q6, u, 0);
            int sumi = dp4a(0x01010101, u, 0);

            // dot(q6-32, q8) = dot(q6, q8) - 32*sum(q8)
            acc += d6 * d8 * (float)scale * ((float)dot - 32.0f * (float)sumi);
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);

    if (lane_id == 0) output[m * N + col] = acc;
}

// ============================================================================
// Q6_K GEMV (F32 activation) — fallback for non-dp4a GPUs
// ============================================================================

extern "C" __global__ void quant_gemv_q6_k_f32(
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

    const unsigned int blocks_per_row = K / 256;
    const unsigned int row_bytes = blocks_per_row * 210;
    const float* act_row = activation + m * K;
    const unsigned char* w_row = weight + col * row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; b++) {
        const unsigned char* block = w_row + b * 210;
        const unsigned char* ql = block;
        const unsigned char* qh = block + 128;
        const signed char* sc = reinterpret_cast<const signed char*>(block + 192);
        float d = __half2float(*reinterpret_cast<const __half*>(block + 208));
        unsigned int base = b * 256;

        #pragma unroll
        for (int n_half = 0; n_half < 2; n_half++) {
            int y_base = n_half * 128;
            int ql_base = n_half * 64;
            int qh_base = n_half * 32;
            int sc_base = n_half * 8;
            int is = lane_id / 16;

            unsigned char ql_lo = ql[ql_base + lane_id];
            unsigned char ql_hi = ql[ql_base + lane_id + 32];
            unsigned char qh_val = qh[qh_base + lane_id];

            int q1 = (int)((ql_lo & 0x0F) | ((qh_val & 0x03) << 4)) - 32;
            int q2 = (int)((ql_hi & 0x0F) | (((qh_val >> 2) & 0x03) << 4)) - 32;
            int q3 = (int)((ql_lo >> 4) | (((qh_val >> 4) & 0x03) << 4)) - 32;
            int q4 = (int)((ql_hi >> 4) | (((qh_val >> 6) & 0x03) << 4)) - 32;

            acc += act_row[base + y_base + lane_id]      * (d * (float)sc[sc_base + is]     * (float)q1);
            acc += act_row[base + y_base + lane_id + 32]  * (d * (float)sc[sc_base + is + 2] * (float)q2);
            acc += act_row[base + y_base + lane_id + 64]  * (d * (float)sc[sc_base + is + 4] * (float)q3);
            acc += act_row[base + y_base + lane_id + 96]  * (d * (float)sc[sc_base + is + 6] * (float)q4);
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane_id == 0) output[m * N + col] = acc;
}
