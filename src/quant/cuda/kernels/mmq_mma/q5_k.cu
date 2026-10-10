// Q5_K x Q8_1 -> f32, tensor-core MMQ (sm_80+): the feature-major entry points
// `quant_mmq_q5_k_q8_1_mma*` of one weight format, module `quant_mmq_mma_q5_k`.
// The format policy is `formats/q5_k.cuh`; the family's shared machinery is in
// the headers beside this file.

#include "formats/q5_k.cuh"
#include "entry_points.cuh"

MMQ_FM_KERNELS(MmqfQ5K, q5_k)
MMQ_FM_KERNELS_NARROW(MmqfQ5K, q5_k)
