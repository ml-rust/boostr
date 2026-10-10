// IQ3_S x Q8_1 -> f32, tensor-core MMQ (sm_80+): the feature-major entry points
// `quant_mmq_iq3_s_q8_1_mma*` of one weight format, module `quant_mmq_mma_iq3_s`.
// The format policy is `formats/iq3_s.cuh`; the family's shared machinery is in
// the headers beside this file.

#include "formats/iq3_s.cuh"
#include "entry_points.cuh"

MMQ_FM_KERNELS(MmqfIQ3S, iq3_s)
