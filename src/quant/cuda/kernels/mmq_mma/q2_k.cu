// Q2_K x Q8_1 -> f32, tensor-core MMQ (sm_80+): the feature-major entry points
// `quant_mmq_q2_k_q8_1_mma*` of one weight format, module `quant_mmq_mma_q2_k`.
// The format policy is `formats/q2_k.cuh`; the family's shared machinery is in
// the headers beside this file.

#include "formats/q2_k.cuh"
#include "entry_points.cuh"

MMQ_FM_KERNELS(MmqfQ2K, q2_k)
