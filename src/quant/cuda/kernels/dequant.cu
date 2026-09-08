// Dequantization CUDA kernels for boostr
// Supports: Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, IQ3_S, IQ2_XS → f32
//
// The K-quant kernels map one thread to a four-element run and store a float4;
// the remaining kernels map one thread to one whole quantized block.
// Block formats match llama.cpp bit-for-bit.

#include <cuda_fp16.h>

#include "iq_dequant.cuh"
#include "decode.cuh"  // KVALUES_IQ4NL, shared with the other IQ4 readers

extern "C" {

// ============================================================================
// Q4_0 Dequantization
// Block: 32 elements, 18 bytes (2-byte f16 scale + 16 bytes nibbles)
// Formula: x = (nibble - 8) * scale
// One thread per block of 32 elements
// ============================================================================

__global__ void dequant_q4_0_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 18;
    float* out = output + bid * 32;

    // Read f16 scale
    __half d_half = *reinterpret_cast<const __half*>(block);
    float d = __half2float(d_half);

    const unsigned char* qs = block + 2;

    // Split-half nibble order (llama.cpp `dequantize_row_q4_0`): element j takes
    // the LOW nibble of qs[j], element j+16 the HIGH nibble of the SAME byte.
    // They are 16 apart, not adjacent — see decode.cuh.
    for (int j = 0; j < 16; j++) {
        unsigned char byte = qs[j];
        int low = (int)(byte & 0x0F) - 8;
        int high = (int)((byte >> 4) & 0x0F) - 8;
        out[j] = (float)low * d;
        out[j + 16] = (float)high * d;
    }
}

// ============================================================================
// Q5_0 Dequantization
// Block: 32 elements, 22 bytes (2-byte f16 scale + 4-byte qh + 16 bytes nibbles)
// Formula: x = ((low4 | (high1 << 4)) - 16) * scale
// ============================================================================

__global__ void dequant_q5_0_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 22;
    float* out = output + bid * 32;

    __half d_half = *reinterpret_cast<const __half*>(block);
    float d = __half2float(d_half);
    // The block stride is 22 bytes, so `block + 2` is only 2-byte aligned for
    // odd blocks: a 4-byte load through a `unsigned int*` traps with
    // CUDA_ERROR_MISALIGNED_ADDRESS, which poisons the context for every later
    // launch. memcpy is the portable unaligned load.
    unsigned int qh;
    memcpy(&qh, block + 2, sizeof(unsigned int));
    const unsigned char* qs = block + 6;

    // Split-half nibble order (llama.cpp `dequantize_row_q5_0`): element j takes
    // the LOW nibble of qs[j] and fifth bit `qh` bit j; element j+16 takes the
    // HIGH nibble of the same byte and `qh` bit j+16. See decode.cuh.
    for (int j = 0; j < 16; j++) {
        unsigned char byte = qs[j];
        int low  = (byte & 0x0F) | (((qh >> j) & 1) << 4);
        int high = ((byte >> 4) & 0x0F) | (((qh >> (j + 16)) & 1) << 4);
        out[j]      = (float)(low - 16) * d;
        out[j + 16] = (float)(high - 16) * d;
    }
}

// ============================================================================
// Q8_0 Dequantization
// Block: 32 elements, 34 bytes (2-byte f16 scale + 32 bytes i8 values)
// Formula: x = qs[i] * scale
// ============================================================================

__global__ void dequant_q8_0_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 34;
    float* out = output + bid * 32;

    __half d_half = *reinterpret_cast<const __half*>(block);
    float d = __half2float(d_half);

    const signed char* qs = reinterpret_cast<const signed char*>(block + 2);

    for (int i = 0; i < 32; i++) {
        out[i] = (float)qs[i] * d;
    }
}

// ============================================================================
// Q4_K Dequantization
// Block: 256 elements, 144 bytes
// Layout: 2-byte d, 2-byte dmin, 12-byte scales, 128-byte qs
// 8 sub-blocks of 32 elements with 6-bit scales/mins
//
// Thread-to-ELEMENT mapping, four consecutive elements per thread, so
// warp-consecutive threads write warp-consecutive floats and one store
// instruction covers one contiguous burst instead of 32 scattered ones.
//
// One CUDA block owns Q4K_SUPER_PER_BLOCK super-blocks. Its threads first
// resolve the packed 6-bit scale/min header of every sub-block it covers into
// shared memory — one thread per (super-block, sub-block) pair — so the header
// is decoded once per sub-block rather than eight times per thread. After the
// barrier each thread reads only the pair its own elements need.
//
// The four elements a thread owns are indices 4k..4k+3 of one 32-element
// sub-block, so they share one (dl, ml) pair and one nibble half, and the four
// results go out as a single float4.
// ============================================================================

// Super-blocks one CUDA block covers, with Q4K_DEQUANT_BLOCK threads of four
// elements each: 4 * 256 elements / 4 = 256 threads.
#define Q4K_SUPER_PER_BLOCK 4
#define Q4K_ELEMS_PER_THREAD 4
#define Q4K_DEQUANT_BLOCK 256
// Threads covering one super-block's 256 elements.
#define Q4K_THREADS_PER_SUPER (256 / Q4K_ELEMS_PER_THREAD)

__global__ __launch_bounds__(Q4K_DEQUANT_BLOCK) void dequant_q4_k_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    // Per sub-block scale and min, already multiplied by the super-block's `d`
    // and `dmin`. Resolved once, read by all 8 threads that share a sub-block.
    __shared__ float s_dl[Q4K_SUPER_PER_BLOCK * 8];
    __shared__ float s_ml[Q4K_SUPER_PER_BLOCK * 8];

    const unsigned int super_base = blockIdx.x * Q4K_SUPER_PER_BLOCK;
    const unsigned int tid = threadIdx.x;

    // Header decode: threads 0..31 take one (super-block, sub-block) pair each.
    if (tid < Q4K_SUPER_PER_BLOCK * 8) {
        const unsigned int local_super = tid / 8;
        const int j = (int)(tid % 8);
        const unsigned int bid = super_base + local_super;
        float dl = 0.0f;
        float ml = 0.0f;
        if (bid < num_blocks) {
            const unsigned char* block = input + (unsigned long long)bid * 144;
            __half d_half = *reinterpret_cast<const __half*>(block);
            __half dmin_half = *reinterpret_cast<const __half*>(block + 2);
            float d = __half2float(d_half);
            float dmin = __half2float(dmin_half);
            const unsigned char* sc = block + 4;

            // 6-bit scale/min unpack, one sub-block's pair (matches llama.cpp
            // get_scale_min_k4 and the loop this kernel replaced).
            unsigned char scale;
            unsigned char min_value;
            if (j < 4) {
                scale = sc[j] & 0x3F;
                min_value = sc[j + 4] & 0x3F;
            } else {
                scale = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
                min_value = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
            }
            dl = d * (float)scale;
            ml = dmin * (float)min_value;
        }
        s_dl[tid] = dl;
        s_ml[tid] = ml;
    }
    __syncthreads();

    const unsigned int local_super = tid / Q4K_THREADS_PER_SUPER;
    const unsigned int quad = tid % Q4K_THREADS_PER_SUPER;
    const unsigned int bid = super_base + local_super;
    if (bid >= num_blocks) return;

    // Element run this thread owns inside its super-block.
    const unsigned int e0 = quad * Q4K_ELEMS_PER_THREAD;
    const unsigned int j = e0 / 32;   // sub-block
    const unsigned int l = e0 % 32;   // offset inside the sub-block

    const float dl = s_dl[local_super * 8 + j];
    const float ml = s_ml[local_super * 8 + j];

    // Sub-block PAIRS share one 32-byte run of `qs`: the even sub-block takes
    // the low nibbles, the odd one the high nibbles of the SAME bytes. The
    // nibble half is fixed for the thread, so the select is one shift, not a
    // per-element branch.
    const unsigned char* qs = input + (unsigned long long)bid * 144 + 16;
    const unsigned int shift = (j % 2) * 4;
    const unsigned char* q_bytes = qs + (j / 2) * 32 + l;

    float4 v;
    v.x = dl * (float)((q_bytes[0] >> shift) & 0x0F) - ml;
    v.y = dl * (float)((q_bytes[1] >> shift) & 0x0F) - ml;
    v.z = dl * (float)((q_bytes[2] >> shift) & 0x0F) - ml;
    v.w = dl * (float)((q_bytes[3] >> shift) & 0x0F) - ml;

    // The output base is a device allocation, so it is at least 256-byte
    // aligned, and this offset is a multiple of four floats — the 16-byte
    // alignment a float4 store needs.
    float4* out4 = reinterpret_cast<float4*>(
        output + (unsigned long long)bid * 256 + e0);
    *out4 = v;
}

// ============================================================================
// Q6_K Dequantization
// Block: 256 elements, 210 bytes
// Layout: 128-byte ql, 64-byte qh, 16-byte scales (i8), 2-byte d
//
// Thread-to-ELEMENT mapping, four consecutive elements per thread, so
// warp-consecutive threads write warp-consecutive floats and one store
// instruction covers one contiguous burst instead of 32 scattered ones.
//
// The 16 signed scales of a super-block are premultiplied by `d` once into
// shared memory — one thread per (super-block, scale) pair — instead of once
// per output element. `d * sc * q` associates left to right, so folding the
// `d * sc` half out of the element loop leaves each product bit-identical.
//
// The scan order of the loop this replaced is: 128-element half `n`, then the
// four 32-element runs `g` that share one `ql`/`qh` byte column, then the
// offset `l` inside the run. The scale index changes at `l == 16`, and a
// thread's element run starts at a multiple of four, so all four of its
// elements land on the same side of that boundary and share one scale.
// ============================================================================

// Super-blocks one CUDA block covers, with Q6K_DEQUANT_BLOCK threads of four
// elements each: 4 * 256 elements / 4 = 256 threads.
#define Q6K_SUPER_PER_BLOCK 4
#define Q6K_ELEMS_PER_THREAD 4
#define Q6K_DEQUANT_BLOCK 256
// Threads covering one super-block's 256 elements.
#define Q6K_THREADS_PER_SUPER (256 / Q6K_ELEMS_PER_THREAD)

__global__ __launch_bounds__(Q6K_DEQUANT_BLOCK) void dequant_q6_k_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    // Per-scale factor, already multiplied by the super-block's `d`. Resolved
    // once, read by all 16 threads whose elements share that scale.
    __shared__ float s_dsc[Q6K_SUPER_PER_BLOCK * 16];

    const unsigned int super_base = blockIdx.x * Q6K_SUPER_PER_BLOCK;
    const unsigned int tid = threadIdx.x;

    // Header decode: threads 0..63 take one (super-block, scale) pair each.
    if (tid < Q6K_SUPER_PER_BLOCK * 16) {
        const unsigned int local_super = tid / 16;
        const unsigned int i = tid % 16;
        const unsigned int bid = super_base + local_super;
        float dsc = 0.0f;
        if (bid < num_blocks) {
            const unsigned char* block = input + (unsigned long long)bid * 210;
            const signed char* sc = reinterpret_cast<const signed char*>(block + 192);
            __half d_half = *reinterpret_cast<const __half*>(block + 208);
            float d = __half2float(d_half);
            dsc = d * (float)sc[i];
        }
        s_dsc[tid] = dsc;
    }
    __syncthreads();

    const unsigned int local_super = tid / Q6K_THREADS_PER_SUPER;
    const unsigned int quad = tid % Q6K_THREADS_PER_SUPER;
    const unsigned int bid = super_base + local_super;
    if (bid >= num_blocks) return;

    // Element run this thread owns inside its super-block.
    const unsigned int e0 = quad * Q6K_ELEMS_PER_THREAD;
    const unsigned int n = e0 / 128;       // 128-element half
    const unsigned int r = e0 % 128;
    const unsigned int g = r / 32;         // which of the half's four runs
    const unsigned int l = r % 32;         // offset inside the run

    const float dsc = s_dsc[local_super * 16 + n * 8 + (l / 16) + 2 * g];

    const unsigned char* block = input + (unsigned long long)bid * 210;
    // Runs 0 and 1 read the low nibble of `ql`, runs 2 and 3 the high nibble;
    // runs 0 and 2 read the first 32-byte `ql` column, runs 1 and 3 the second.
    // `>> 4` on an unsigned char is already 4-bit, so the mask is a no-op there
    // and the two cases collapse to one shift without changing any value.
    const unsigned char* ql = block + n * 64 + (g % 2) * 32 + l;
    const unsigned char* qh = block + 128 + n * 32 + l;
    const unsigned int ql_shift = (g / 2) * 4;
    const unsigned int qh_shift = g * 2;

    float4 v;
    v.x = dsc * (float)((int)(((ql[0] >> ql_shift) & 0x0F) | (((qh[0] >> qh_shift) & 0x03) << 4)) - 32);
    v.y = dsc * (float)((int)(((ql[1] >> ql_shift) & 0x0F) | (((qh[1] >> qh_shift) & 0x03) << 4)) - 32);
    v.z = dsc * (float)((int)(((ql[2] >> ql_shift) & 0x0F) | (((qh[2] >> qh_shift) & 0x03) << 4)) - 32);
    v.w = dsc * (float)((int)(((ql[3] >> ql_shift) & 0x0F) | (((qh[3] >> qh_shift) & 0x03) << 4)) - 32);

    // The output base is a device allocation, so it is at least 256-byte
    // aligned, and this offset is a multiple of four floats — the 16-byte
    // alignment a float4 store needs.
    float4* out4 = reinterpret_cast<float4*>(
        output + (unsigned long long)bid * 256 + e0);
    *out4 = v;
}

// ============================================================================
// Q2_K Dequantization
// Block: 256 elements, 84 bytes
// Layout: 16-byte sc, 64-byte qs, 2-byte d, 2-byte dmin
// 16 sub-blocks of 16 elements, 2-bit values
//
// Thread-to-ELEMENT mapping, four consecutive elements per thread, so
// warp-consecutive threads write warp-consecutive floats and one store
// instruction covers one contiguous burst instead of 32 scattered ones.
//
// The 16 packed (scale, min) nibble pairs of a super-block are resolved once
// into shared memory — one thread per (super-block, sub-block) pair — instead
// of once per 16-element run inside every thread.
//
// The scan order of the loop this replaced is: 32-byte half `n`, then the
// 16-element run `g`, then the offset `l` inside the run. `g` advances the
// sub-block index by one each time and flips between the low and high 16-byte
// column of `q`, changing the 2-bit shift every second run. The four elements
// a thread owns sit inside one 16-element run, so they share one (dl, ml) pair
// and one shift, and the four results go out as a single float4.
// ============================================================================

// Super-blocks one CUDA block covers, with Q2K_DEQUANT_BLOCK threads of four
// elements each: 4 * 256 elements / 4 = 256 threads.
#define Q2K_SUPER_PER_BLOCK 4
#define Q2K_ELEMS_PER_THREAD 4
#define Q2K_DEQUANT_BLOCK 256
// Threads covering one super-block's 256 elements.
#define Q2K_THREADS_PER_SUPER (256 / Q2K_ELEMS_PER_THREAD)

__global__ __launch_bounds__(Q2K_DEQUANT_BLOCK) void dequant_q2_k_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    // Per sub-block scale and min, already multiplied by the super-block's `d`
    // and `dmin`. Resolved once, read by all 4 threads that share a sub-block.
    __shared__ float s_dl[Q2K_SUPER_PER_BLOCK * 16];
    __shared__ float s_ml[Q2K_SUPER_PER_BLOCK * 16];

    const unsigned int super_base = blockIdx.x * Q2K_SUPER_PER_BLOCK;
    const unsigned int tid = threadIdx.x;

    // Header decode: threads 0..63 take one (super-block, sub-block) pair each.
    if (tid < Q2K_SUPER_PER_BLOCK * 16) {
        const unsigned int local_super = tid / 16;
        const unsigned int is = tid % 16;
        const unsigned int bid = super_base + local_super;
        float dl = 0.0f;
        float ml = 0.0f;
        if (bid < num_blocks) {
            const unsigned char* block = input + (unsigned long long)bid * 84;
            const unsigned char* sc = block;
            __half d_half = *reinterpret_cast<const __half*>(block + 80);
            __half dmin_half = *reinterpret_cast<const __half*>(block + 82);
            float d = __half2float(d_half);
            float dmin = __half2float(dmin_half);
            dl = d * (float)(sc[is] & 0x0F);
            ml = dmin * (float)(sc[is] >> 4);
        }
        s_dl[tid] = dl;
        s_ml[tid] = ml;
    }
    __syncthreads();

    const unsigned int local_super = tid / Q2K_THREADS_PER_SUPER;
    const unsigned int quad = tid % Q2K_THREADS_PER_SUPER;
    const unsigned int bid = super_base + local_super;
    if (bid >= num_blocks) return;

    // Element run this thread owns inside its super-block.
    const unsigned int e0 = quad * Q2K_ELEMS_PER_THREAD;
    const unsigned int n = e0 / 128;   // 32-byte half of `qs`
    const unsigned int r = e0 % 128;
    const unsigned int g = r / 16;     // 16-element run inside that half
    const unsigned int l = r % 16;     // offset inside the run

    const float dl = s_dl[local_super * 16 + n * 8 + g];
    const float ml = s_ml[local_super * 16 + n * 8 + g];

    const unsigned char* q = input + (unsigned long long)bid * 84 + 16
                             + n * 32 + (g % 2) * 16 + l;
    const unsigned int shift = (g / 2) * 2;

    float4 v;
    v.x = dl * (float)((q[0] >> shift) & 3) - ml;
    v.y = dl * (float)((q[1] >> shift) & 3) - ml;
    v.z = dl * (float)((q[2] >> shift) & 3) - ml;
    v.w = dl * (float)((q[3] >> shift) & 3) - ml;

    // The output base is a device allocation, so it is at least 256-byte
    // aligned, and this offset is a multiple of four floats — the 16-byte
    // alignment a float4 store needs.
    float4* out4 = reinterpret_cast<float4*>(
        output + (unsigned long long)bid * 256 + e0);
    *out4 = v;
}

// ============================================================================
// Q3_K Dequantization
// Block: 256 elements, 110 bytes
// Layout: 32-byte hmask, 64-byte qs, 12-byte scales, 2-byte d
//
// Thread-to-ELEMENT mapping, four consecutive elements per thread, so
// warp-consecutive threads write warp-consecutive floats and one store
// instruction covers one contiguous burst instead of 32 scattered ones.
//
// The 16 6-bit scales are packed across all 12 header bytes at once, so the
// unpack stays a single whole-header routine: one thread per super-block runs
// it verbatim and premultiplies each scale by `d` into shared memory, once
// instead of once per thread.
//
// The scan order of the loop this replaced is: 32-byte half `n`, then the
// 16-element run `g`, then the offset `l` inside the run. `g` advances the
// scale index by one each time and flips between the low and high 16-byte
// column of `qs` and of `hmask`, changing the 2-bit shift every second run.
// The high-bit-plane mask walks one bit per shift step across both halves, so
// it is `1 << (n * 4 + g / 2)`. The four elements a thread owns sit inside one
// 16-element run, so they share one scale and one shift.
// ============================================================================

// Super-blocks one CUDA block covers, with Q3K_DEQUANT_BLOCK threads of four
// elements each: 4 * 256 elements / 4 = 256 threads.
#define Q3K_SUPER_PER_BLOCK 4
#define Q3K_ELEMS_PER_THREAD 4
#define Q3K_DEQUANT_BLOCK 256
// Threads covering one super-block's 256 elements.
#define Q3K_THREADS_PER_SUPER (256 / Q3K_ELEMS_PER_THREAD)

__global__ __launch_bounds__(Q3K_DEQUANT_BLOCK) void dequant_q3_k_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    // Per sub-block scale, already multiplied by the super-block's `d`.
    // Resolved once, read by all 4 threads that share a sub-block.
    __shared__ float s_dl[Q3K_SUPER_PER_BLOCK * 16];

    const unsigned int super_base = blockIdx.x * Q3K_SUPER_PER_BLOCK;
    const unsigned int tid = threadIdx.x;

    // Header decode: threads 0..3 take one super-block each.
    if (tid < Q3K_SUPER_PER_BLOCK) {
        const unsigned int bid = super_base + tid;
        if (bid < num_blocks) {
            const unsigned char* block = input + (unsigned long long)bid * 110;
            const unsigned char* sc_raw = block + 96;
            __half d_half = *reinterpret_cast<const __half*>(block + 108);
            float d = __half2float(d_half);

            // Unpack 16 6-bit scales from 12 bytes
            unsigned int aux[4];
            unsigned char aux_bytes[12];
            for (int i = 0; i < 12; i++) aux_bytes[i] = sc_raw[i];
            memcpy(&aux[0], aux_bytes, 4);
            memcpy(&aux[1], aux_bytes + 4, 4);
            memcpy(&aux[2], aux_bytes + 8, 4);

            unsigned int tmp = aux[2];
            const unsigned int KMASK1 = 0x03030303u;
            const unsigned int KMASK2 = 0x0f0f0f0fu;
            unsigned int a0 = aux[0], a1 = aux[1];
            aux[0] = (a0 & KMASK2) | ((tmp & KMASK1) << 4);
            aux[1] = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
            aux[2] = ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);

            signed char scales[16];
            memcpy(&scales[0],  &aux[0], 4);
            memcpy(&scales[4],  &aux[1], 4);
            memcpy(&scales[8],  &aux[2], 4);
            memcpy(&scales[12], &aux[3], 4);
            for (int i = 0; i < 16; i++)
                scales[i] = (signed char)((unsigned char)scales[i] - 32);

            for (int i = 0; i < 16; i++)
                s_dl[tid * 16 + i] = d * (float)scales[i];
        } else {
            for (int i = 0; i < 16; i++)
                s_dl[tid * 16 + i] = 0.0f;
        }
    }
    __syncthreads();

    const unsigned int local_super = tid / Q3K_THREADS_PER_SUPER;
    const unsigned int quad = tid % Q3K_THREADS_PER_SUPER;
    const unsigned int bid = super_base + local_super;
    if (bid >= num_blocks) return;

    // Element run this thread owns inside its super-block.
    const unsigned int e0 = quad * Q3K_ELEMS_PER_THREAD;
    const unsigned int n = e0 / 128;   // 32-byte half of `qs`
    const unsigned int r = e0 % 128;
    const unsigned int g = r / 16;     // 16-element run inside that half
    const unsigned int l = r % 16;     // offset inside the run

    const float dl = s_dl[local_super * 16 + n * 8 + g];

    const unsigned char* block = input + (unsigned long long)bid * 110;
    const unsigned char* hmask = block + (g % 2) * 16 + l;
    const unsigned char* q = block + 32 + n * 32 + (g % 2) * 16 + l;
    const unsigned int shift = (g / 2) * 2;
    const unsigned int m = 1u << (n * 4 + g / 2);

    float4 v;
    v.x = dl * (float)(((q[0] >> shift) & 3) - ((hmask[0] & m) ? 0 : 4));
    v.y = dl * (float)(((q[1] >> shift) & 3) - ((hmask[1] & m) ? 0 : 4));
    v.z = dl * (float)(((q[2] >> shift) & 3) - ((hmask[2] & m) ? 0 : 4));
    v.w = dl * (float)(((q[3] >> shift) & 3) - ((hmask[3] & m) ? 0 : 4));

    // The output base is a device allocation, so it is at least 256-byte
    // aligned, and this offset is a multiple of four floats — the 16-byte
    // alignment a float4 store needs.
    float4* out4 = reinterpret_cast<float4*>(
        output + (unsigned long long)bid * 256 + e0);
    *out4 = v;
}

// ============================================================================
// Q5_K Dequantization
// Block: 256 elements, 176 bytes
// Layout: 2-byte d, 2-byte dmin, 12-byte sc, 32-byte qh, 128-byte qs
// 8 sub-blocks of 32 elements, 5-bit values (4-bit low + 1-bit high)
//
// Thread-to-ELEMENT mapping, four consecutive elements per thread, so
// warp-consecutive threads write warp-consecutive floats and one store
// instruction covers one contiguous burst instead of 32 scattered ones.
//
// One CUDA block owns Q5K_SUPER_PER_BLOCK super-blocks. Its threads first
// resolve the packed 6-bit scale/min header of every sub-block it covers into
// shared memory — one thread per (super-block, sub-block) pair — so the header
// is decoded once per sub-block rather than eight times per thread. After the
// barrier each thread reads only the pair its own elements need.
//
// The four elements a thread owns are indices 4k..4k+3 of one 32-element
// sub-block, so they share one (dl, ml) pair and one nibble half, and the four
// results go out as a single float4.
// ============================================================================

// Super-blocks one CUDA block covers, with Q5K_DEQUANT_BLOCK threads of four
// elements each: 4 * 256 elements / 4 = 256 threads.
#define Q5K_SUPER_PER_BLOCK 4
#define Q5K_ELEMS_PER_THREAD 4
#define Q5K_DEQUANT_BLOCK 256
// Threads covering one super-block's 256 elements.
#define Q5K_THREADS_PER_SUPER (256 / Q5K_ELEMS_PER_THREAD)

__global__ __launch_bounds__(Q5K_DEQUANT_BLOCK) void dequant_q5_k_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    // Per sub-block scale and min, already multiplied by the super-block's `d`
    // and `dmin`. Resolved once, read by all 8 threads that share a sub-block.
    __shared__ float s_dl[Q5K_SUPER_PER_BLOCK * 8];
    __shared__ float s_ml[Q5K_SUPER_PER_BLOCK * 8];

    const unsigned int super_base = blockIdx.x * Q5K_SUPER_PER_BLOCK;
    const unsigned int tid = threadIdx.x;

    // Header decode: threads 0..31 take one (super-block, sub-block) pair each.
    if (tid < Q5K_SUPER_PER_BLOCK * 8) {
        const unsigned int local_super = tid / 8;
        const int i = (int)(tid % 8);
        const unsigned int bid = super_base + local_super;
        float dl = 0.0f;
        float ml = 0.0f;
        if (bid < num_blocks) {
            const unsigned char* block = input + (unsigned long long)bid * 176;
            __half d_half = *reinterpret_cast<const __half*>(block);
            __half dmin_half = *reinterpret_cast<const __half*>(block + 2);
            float d = __half2float(d_half);
            float dmin = __half2float(dmin_half);
            const unsigned char* sc = block + 4;

            // Unpack 6-bit scales and mins (same as Q4_K), one sub-block's pair
            // out of the loop this kernel replaced.
            unsigned char scale;
            unsigned char min_value;
            if (i < 4) {
                scale = sc[i] & 0x3F;
                min_value = sc[i + 4] & 0x3F;
            } else {
                scale = (sc[i + 4] & 0x0F) | ((sc[i - 4] >> 6) << 4);
                min_value = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4);
            }
            dl = d * (float)scale;
            ml = dmin * (float)min_value;
        }
        s_dl[tid] = dl;
        s_ml[tid] = ml;
    }
    __syncthreads();

    const unsigned int local_super = tid / Q5K_THREADS_PER_SUPER;
    const unsigned int quad = tid % Q5K_THREADS_PER_SUPER;
    const unsigned int bid = super_base + local_super;
    if (bid >= num_blocks) return;

    // Element run this thread owns inside its super-block.
    const unsigned int e0 = quad * Q5K_ELEMS_PER_THREAD;
    const unsigned int j = e0 / 32;   // sub-block
    const unsigned int l = e0 % 32;   // offset inside the sub-block

    const float dl = s_dl[local_super * 8 + j];
    const float ml = s_ml[local_super * 8 + j];

    const unsigned char* block = input + (unsigned long long)bid * 176;
    // llama.cpp `dequantize_row_q5_K`: sub-block PAIRS share one 32-byte run of
    // `qs` — the even sub-block takes the low nibbles, the odd one the high
    // nibbles of the SAME bytes (identical to Q4_K above). It is NOT a
    // per-sub-block 16-byte run with interleaved nibbles. The nibble half is
    // fixed for the thread, so the select is one shift, not a per-element
    // branch: `>> 4` on an unsigned char is already 4-bit, so masking after the
    // shift changes no value.
    const unsigned char* q_bytes = block + 48 + (j / 2) * 32 + l;
    const unsigned int shift = (j % 2) * 4;
    // 5th bit: `qh` is indexed by ELEMENT within the sub-block and the BIT is
    // the sub-block index — one qh byte per element carries that element's high
    // bit for all 8 sub-blocks. Not a flat bitstream.
    const unsigned char* qh = block + 16 + l;

    float4 v;
    v.x = dl * (float)(((q_bytes[0] >> shift) & 0x0F) | (((qh[0] >> j) & 0x01) << 4)) - ml;
    v.y = dl * (float)(((q_bytes[1] >> shift) & 0x0F) | (((qh[1] >> j) & 0x01) << 4)) - ml;
    v.z = dl * (float)(((q_bytes[2] >> shift) & 0x0F) | (((qh[2] >> j) & 0x01) << 4)) - ml;
    v.w = dl * (float)(((q_bytes[3] >> shift) & 0x0F) | (((qh[3] >> j) & 0x01) << 4)) - ml;

    // The output base is a device allocation, so it is at least 256-byte
    // aligned, and this offset is a multiple of four floats — the 16-byte
    // alignment a float4 store needs.
    float4* out4 = reinterpret_cast<float4*>(
        output + (unsigned long long)bid * 256 + e0);
    *out4 = v;
}

// ============================================================================
// IQ4_NL Dequantization
// Block: 32 elements, 18 bytes (f16 scale + 16 bytes nibbles)
// Non-linear codebook: x = scale * KVALUES_IQ4NL[nibble]
// ============================================================================

__global__ void dequant_iq4_nl_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 18;
    float* out = output + bid * 32;

    __half d_half;
    memcpy(&d_half, block, sizeof(__half));
    float d = __half2float(d_half);
    const unsigned char* qs = block + 2;

    // Split-half nibble order (llama.cpp `dequantize_row_iq4_nl`): `y[j]` takes
    // the low nibble, `y[j + QK4_NL/2]` the high nibble of the SAME byte.
    for (int j = 0; j < 16; j++) {
        unsigned char byte = qs[j];
        out[j]      = d * (float)KVALUES_IQ4NL[byte & 0x0F];
        out[j + 16] = d * (float)KVALUES_IQ4NL[(byte >> 4) & 0x0F];
    }
}

// ============================================================================
// IQ4_XS Dequantization
// Block: 256 elements, 136 bytes
// Layout matches llama.cpp `block_iq4_xs` exactly:
//   { ggml_half d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; }
// so scales_h is a TWO-byte field at 2..4 and scales_l occupies 4..8. There is
// no pad byte, and scales_h carries high scale bits for all EIGHT sub-blocks
// (16 bits = 8 x 2).
// 8 sub-blocks of 32 elements, 6-bit scales, KVALUES_IQ4NL codebook
// ============================================================================

__global__ void dequant_iq4_xs_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const unsigned char* block = input + bid * 136;
    float* out = output + bid * 256;

    __half d_half;
    memcpy(&d_half, block, sizeof(__half));
    float d = __half2float(d_half);
    unsigned short scales_h;
    memcpy(&scales_h, block + 2, sizeof(unsigned short));
    const unsigned char* scales_l = block + 4;
    const unsigned char* qs = block + 8;

    for (int sb = 0; sb < 8; sb++) {
        // 4 low bits from scales_l (one nibble per sub-block), 2 high bits from
        // scales_h (2 bits per sub-block across all 8).
        int sl = (scales_l[sb / 2] >> (4 * (sb % 2))) & 0x0F;
        int sh = ((unsigned int)scales_h >> (2 * sb)) & 0x03;
        int scale_6bit = sl | (sh << 4);
        float sub_scale = d * (float)(scale_6bit - 32);

        const unsigned char* sub_qs = qs + sb * 16;
        float* sub_out = out + sb * 32;
        // Split-half nibble order within each sub-block.
        for (int j = 0; j < 16; j++) {
            unsigned char byte = sub_qs[j];
            sub_out[j]      = sub_scale * (float)KVALUES_IQ4NL[byte & 0x0F];
            sub_out[j + 16] = sub_scale * (float)KVALUES_IQ4NL[(byte >> 4) & 0x0F];
        }
    }
}

// ============================================================================
// IQ3_S and IQ2_XS Dequantization
//
// Both are codebook quantizations. The block layouts and the grid tables live
// once in iq_dequant.cuh, shared with dequant_generic.cu, the quant-matmul
// path and the GEMV/GEMM kernels, and gated against llama.cpp by
// tests/gguf_conformance_llama_cpp.rs.
// ============================================================================

__global__ void dequant_iq3_s_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;
    iq3_s_dequant_block(input + (unsigned long long)bid * 110, output + (unsigned long long)bid * 256);
}

__global__ void dequant_iq2_xs_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int num_blocks
) {
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;
    iq2_xs_dequant_block(input + (unsigned long long)bid * 74, output + (unsigned long long)bid * 256);
}

} // extern "C"
