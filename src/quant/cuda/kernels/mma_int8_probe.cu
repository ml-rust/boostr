// Probe kernel: exercises one `mma_m16n8k32_s8` instruction with one warp,
// so a Rust test can compare its output against a scalar reference and prove
// the fragment index map in `mma_int8.cuh` is correct.

#include "mma_int8.cuh"

extern "C" __global__ void mma_int8_probe(
    const int* __restrict__ a,   // 16 rows x 8 ints, row-major
    const int* __restrict__ b,   // 8 rows x 8 ints, row-major
    int* __restrict__ d          // 16 rows x 8 ints, row-major
) {
    int A[4];
    int B[2];
    int D[4] = {0, 0, 0, 0};

    for (int l = 0; l < 4; ++l) {
        A[l] = a[mma_a_i(l) * 8 + mma_a_j(l)];
    }
    for (int l = 0; l < 2; ++l) {
        B[l] = b[mma_b_i(l) * 8 + mma_b_j(l)];
    }

    mma_m16n8k32_s8(D, A, B);

    for (int l = 0; l < 4; ++l) {
        d[mma_d_i(l) * 8 + mma_d_j(l)] = D[l];
    }
}

// Probe kernel for `mma_m16n8k16_s8`. A and B row strides are 4 ints here,
// not 8, because k spans 16 rather than 32. d's stride stays 8.
extern "C" __global__ void mma_int8_k16_probe(
    const int* __restrict__ a,   // 16 rows x 4 ints, row-major
    const int* __restrict__ b,   // 8 rows x 4 ints, row-major
    int* __restrict__ d          // 16 rows x 8 ints, row-major
) {
    int A[2];
    int B[1];
    int D[4] = {0, 0, 0, 0};

    for (int l = 0; l < 2; ++l) {
        A[l] = a[mma_a16_i(l) * 4 + mma_a16_j(l)];
    }
    B[0] = b[mma_b16_i(0) * 4 + mma_b16_j(0)];

    mma_m16n8k16_s8(D, A, B);

    for (int l = 0; l < 4; ++l) {
        d[mma_d_i(l) * 8 + mma_d_j(l)] = D[l];
    }
}
