// IQ2_XS x Q8_1 -> f32, tensor-core MMQ (sm_80+): the feature-major entry points
// `quant_mmq_iq2_xs_q8_1_mma*` of one weight format, module `quant_mmq_mma_iq2_xs`.
// The format policy is `formats/iq2_xs.cuh`; the family's shared machinery is in
// the headers beside this file.

#include "formats/iq2_xs.cuh"
#include "entry_points.cuh"

MMQ_FM_KERNELS(MmqfIQ2XS, iq2_xs)
