// The K-range rule of the feature-major MMQ family: how many 32-element
// activation blocks one 256-k staging group holds, and which block range
// split `s` of `splits` walks. `quant_mmq_mma.cu` and `quant_mmq_gemv1.cu`
// both include it, so a tensor-core launch and the single-token launch cut K
// at the same boundaries and their range partials are the same floats.
#pragma once

// Weight blocks consumed per staging iteration: 256 k-values. Split
// boundaries land on multiples of it.
#define MMQF_ITER_B 8

// K range of split `s` of `splits`, in activation blocks. Both the fused
// tile-parallel walk and the split-K launch cut K on 256-k group boundaries
// with this one rule, so the partial sums they form are the same floats:
// `kb_s = MMQF_ITER_B * floor(s * groups / splits)`, and the last split runs
// to `bpr`, which takes the ragged tail with it. The host sizes `splits` so
// no range is empty (`splits <= groups`); an empty range would still be
// safe, contributing a zero partial.
static __device__ __forceinline__ void mmqf_split_range(
    unsigned int bpr, unsigned int splits, unsigned int s, unsigned int& kb0,
    unsigned int& kb1
) {
    const unsigned long long groups = (bpr + MMQF_ITER_B - 1) / MMQF_ITER_B;
    kb0 = (unsigned int)(MMQF_ITER_B * ((unsigned long long)s * groups / splits));
    kb1 = (s + 1 == splits)
              ? bpr
              : (unsigned int)(MMQF_ITER_B * ((unsigned long long)(s + 1) * groups / splits));
}
