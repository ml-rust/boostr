// GGUF format IDs and per-format block geometry, shared by the generic
// dequant and quant-matmul fallback kernels. IDs must match
// `QuantFormat::format_id()` in Rust — `test_format_ids_match_cuda_kernels`
// in `src/quant/format.rs` checks these `#define`s against this file.

#pragma once

#define FMT_Q4_0    0
#define FMT_Q4_1    1
#define FMT_Q5_0    2
#define FMT_Q5_1    3
#define FMT_Q8_0    4
#define FMT_Q8_1    5
#define FMT_Q2K     6
#define FMT_Q3K     7
#define FMT_Q4K     8
#define FMT_Q5K     9
#define FMT_Q6K     10
#define FMT_Q8K     11
#define FMT_IQ1S    12
#define FMT_IQ1M    13
#define FMT_IQ2XXS  14
#define FMT_IQ2XS   15
#define FMT_IQ2S    16
#define FMT_IQ3XXS  17
#define FMT_IQ3S    18
#define FMT_IQ4NL   19
#define FMT_IQ4XS   20
#define FMT_TQ1_0   21
#define FMT_TQ2_0   22
#define FMT_PQ2_0   23
#define FMT_PTQ1_0  24
#define FMT_Q1_0    25
#define FMT_Q2_0    26

// Number of logical elements per block, by format_id.
static __device__ __forceinline__ int get_block_size(unsigned int fmt) {
    switch (fmt) {
        case FMT_Q4_0: case FMT_Q4_1: case FMT_Q5_0: case FMT_Q5_1:
        case FMT_Q8_0: case FMT_Q8_1: case FMT_IQ4NL:
            return 32;
        case FMT_Q2_0:
            return 64;
        case FMT_Q1_0: case FMT_PQ2_0: case FMT_PTQ1_0:
            return 128;
        default: // All k-quants, IQ (except IQ4NL), TQ
            return 256;
    }
}

// Exact byte count per block, by format_id. 0 means unknown format.
static __device__ __forceinline__ int get_block_bytes(unsigned int fmt) {
    switch (fmt) {
        case FMT_Q4_0:   return 18;
        case FMT_Q4_1:   return 20;
        case FMT_Q5_0:   return 22;
        case FMT_Q5_1:   return 24;
        case FMT_Q8_0:   return 34;
        case FMT_Q8_1:   return 36;
        case FMT_Q2K:    return 84;
        case FMT_Q3K:    return 110;
        case FMT_Q4K:    return 144;
        case FMT_Q5K:    return 176;
        case FMT_Q6K:    return 210;
        case FMT_Q8K:    return 292;
        case FMT_IQ1S:   return 50;
        case FMT_IQ1M:   return 56;
        case FMT_IQ2XXS: return 66;
        case FMT_IQ2XS:  return 74;
        case FMT_IQ2S:   return 82;
        case FMT_IQ3XXS: return 98;
        case FMT_IQ3S:   return 110;
        case FMT_IQ4NL:  return 18;
        case FMT_IQ4XS:  return 136;
        case FMT_TQ1_0:  return 54;
        case FMT_TQ2_0:  return 66;
        case FMT_PQ2_0:  return 34;
        case FMT_PTQ1_0: return 28;
        case FMT_Q1_0:   return 18;
        case FMT_Q2_0:   return 18;
        default:         return 0;
    }
}
