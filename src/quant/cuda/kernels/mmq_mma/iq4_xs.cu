// IQ4_XS x Q8_1 -> f32, tensor-core MMQ (sm_80+): the feature-major entry points
// `quant_mmq_iq4_xs_q8_1_mma*` of one weight format, module `quant_mmq_mma_iq4_xs`.
// The format policy is `formats/iq4_xs.cuh`; the family's shared machinery is in
// the headers beside this file.

#include "formats/iq4_xs.cuh"
#include "entry_points.cuh"

MMQ_FM_KERNELS(MmqfIQ4XS, iq4_xs)
