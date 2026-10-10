// Q4_1 x Q8_1 -> f32, tensor-core MMQ (sm_80+): the feature-major entry points
// `quant_mmq_q4_1_q8_1_mma*` of one weight format, module `quant_mmq_mma_q4_1`.
// The format policy is `formats/q4_1.cuh`; the family's shared machinery is in
// the headers beside this file.

#include "formats/q4_1.cuh"
#include "entry_points.cuh"

MMQ_FM_KERNELS(MmqfQ41, q4_1)
