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
