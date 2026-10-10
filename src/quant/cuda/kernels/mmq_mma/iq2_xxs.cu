// IQ2_XXS x Q8_1 -> f32, tensor-core MMQ (sm_80+): the feature-major entry points
// `quant_mmq_iq2_xxs_q8_1_mma*` of one weight format, module `quant_mmq_mma_iq2_xxs`.
// The format policy is `formats/iq2_xxs.cuh`; the family's shared machinery is in
// the headers beside this file.

#include "formats/iq2_xxs.cuh"
#include "entry_points.cuh"

MMQ_FM_KERNELS(MmqfIQ2XXS, iq2_xxs)
