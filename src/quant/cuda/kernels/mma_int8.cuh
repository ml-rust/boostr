// Int8 tensor-core MMA wrapper: `m16n8k32` fragment layout and indices.
//
// Every formula below is transcribed verbatim from llama.cpp's
// `ggml/src/ggml-cuda/mma.cuh`. Do not re-derive or "simplify" an index — a
// wrong index here produces silently wrong numbers, not a compile error or a
// crash.
//
// `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` needs sm_80. On sm_75
// llama.cpp emulates it with four `m8n8k16` instructions; this header does
// not implement that emulation.

#pragma once

// Computes D[i][j] = sum over k of A[i][k] * B[j][k]. A is 16x32 int8, B is
// 8x32 int8, D is 16x8 int32. `.row.col` means B is indexed by output
// column, so it is effectively transposed.
//
// Viewed as 32-bit words: A is 16x8 ints, B is 8x8 ints, each int packing
// four int8 with byte 0 holding k*4+0.
static __device__ __forceinline__ void mma_m16n8k32_s8(
    int (&D)[4], const int (&A)[4], const int (&B)[2]
) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}

// A and D share the tile SHAPE, 16x8, but not the register map. llama.cpp's
// `tile<16,8,int>::get_i`/`get_j` describe D, and it fills A with `ldmatrix`,
// which arranges registers to the A operand's own layout. Reusing the D map to
// gather A yields wrong products with no error, so the two are separate here.

// `mma.sync` is warp-collective, so every index below is a position within the
// WARP. llama.cpp reads `threadIdx.x` directly because its kernels launch with
// `blockDim.x == 32`. A 256-thread block makes `threadIdx.x` span eight warps,
// so the lane must be masked out here or warps above the first index past
// their fragment and read another warp's data.
static __device__ __forceinline__ int mma_lane() {
    return threadIdx.x & 31;
}

// Row of the 16x8 A operand held by this lane at element `l`.
// A has 4 elements per lane. `l % 2` picks the row half, `l / 2` the k half.
static __device__ __forceinline__ int mma_a_i(const int l) {
    return ((l % 2) * 8) + (mma_lane() / 4);
}

// Column of the 16x8 A operand, in 32-bit words, held by this lane at element
// `l`. Word `w` covers k in `4*w .. 4*w+3`.
static __device__ __forceinline__ int mma_a_j(const int l) {
    return ((l / 2) * 4) + (mma_lane() % 4);
}

// Row of the 16x8 D accumulator held by this lane at element `l`.
static __device__ __forceinline__ int mma_d_i(const int l) {
    return ((l / 2) * 8) + (mma_lane() / 4);
}

// Column of the 16x8 D accumulator held by this lane at element `l`.
static __device__ __forceinline__ int mma_d_j(const int l) {
    return ((mma_lane() % 4) * 2) + (l % 2);
}

// Row of the 8x8 B operand held by this lane at element `l`.
// B has 2 elements per lane.
static __device__ __forceinline__ int mma_b_i(const int l) {
    return mma_lane() / 4;
}

// Column of the 8x8 B operand, in 32-bit words, held by this lane at element
// `l`.
static __device__ __forceinline__ int mma_b_j(const int l) {
    return (l * 4) + (mma_lane() % 4);
}

// Computes D[i][j] = sum over k in 0..16 of A[i][k] * B[j][k]. A is 16x16
// int8, B is 8x16 int8, D is 16x8 int32. `.row.col` means B is indexed by
// output column, so it is effectively transposed.
//
// Viewed as 32-bit words: A is 16x4 ints, B is 8x4 ints, each int packing
// four int8 with byte 0 holding k*4+0.
static __device__ __forceinline__ void mma_m16n8k16_s8(
    int (&D)[4], const int (&A)[2], const int (&B)[1]
) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]));
}

// D keeps the same 16x8 accumulator shape as `m16n8k32`, so `mma_d_i` and
// `mma_d_j` above are reused unchanged for `m16n8k16`.
//
// llama.cpp's `tile<16,4,int>` has no `get_i`/`get_j` — its `supported()`
// list omits 16x4, so it is loaded only by `ldmatrix`. These maps therefore
// come from the PTX operand spec, not from llama.cpp, and the probe kernel
// is what proves them.

// Row of the 16x4 A operand held by this lane at element `l`.
// A has 2 elements per lane. `l % 2` picks the row half.
static __device__ __forceinline__ int mma_a16_i(const int l) {
    return ((l % 2) * 8) + (mma_lane() / 4);
}

// Column of the 16x4 A operand, in 32-bit words, held by this lane at
// element `l`. `l` is unused: each lane holds one word per row at this
// shape. It stays in the signature so every map here shares one calling
// shape.
static __device__ __forceinline__ int mma_a16_j(const int l) {
    (void)l;
    return mma_lane() % 4;
}

// Row of the 8x4 B operand held by this lane at element `l`.
// B has 1 element per lane.
static __device__ __forceinline__ int mma_b16_i(const int l) {
    return mma_lane() / 4;
}

// Column of the 8x4 B operand, in 32-bit words, held by this lane at
// element `l`. `l` is unused: each lane holds one word per row at this
// shape. It stays in the signature so every map here shares one calling
// shape.
static __device__ __forceinline__ int mma_b16_j(const int l) {
    (void)l;
    return mma_lane() % 4;
}
