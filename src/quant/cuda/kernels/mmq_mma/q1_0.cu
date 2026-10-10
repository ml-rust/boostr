// Q1_0 x Q8_1 -> f32, tensor-core MMQ (sm_80+): the feature-major entry points
// `quant_mmq_q1_0_q8_1_mma*` of one weight format, module `quant_mmq_mma_q1_0`.
// The format policy is `formats/lowbit.cuh`; the family's shared machinery is in
// the headers beside this file.

#include "formats/lowbit.cuh"
#include "entry_points.cuh"

MMQ_FM_KERNELS(MmqfQ10, q1_0)
