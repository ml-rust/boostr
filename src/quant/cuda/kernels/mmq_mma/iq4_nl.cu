// IQ4_NL x Q8_1 -> f32, tensor-core MMQ (sm_80+): the feature-major entry points
// `quant_mmq_iq4_nl_q8_1_mma*` of one weight format, module `quant_mmq_mma_iq4_nl`.
// The format policy is `formats/iq4_nl.cuh`; the family's shared machinery is in
// the headers beside this file.

#include "formats/iq4_nl.cuh"
#include "entry_points.cuh"

MMQ_FM_KERNELS(MmqfIQ4NL, iq4_nl)
