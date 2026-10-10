// Tile constants of the three token-major tensor-core kernels,
// `quant_mmq_q8_0_q8_1_mma` (`q8_0.cu`), `quant_mmq_q4_k_q8_1_mma`
// (`q4_k.cu`) and `quant_mmq_q6_k_q8_1_mma` (`q6_k.cu`). Not a compilation
// unit of its own.
//
// All three kernels stage one k-block ahead: the global loads for block
// `b+1` are issued before block `b`'s `mma` sequence, so their latency
// overlaps compute. nvcc does not reorder this itself: the loads must cross a
// `__syncthreads()` to move earlier.
#pragma once

#include "common.cuh"

#define MMQ_BM 128
#define MMQ_BN 64
#define MMQ_BK 32
// Row-stride padding, in ints. 8 makes the padded stride 8 mod 32, so the four
// word indices a lane reads land 8 banks apart and the warp's 32 accesses cover
// all 32 banks.
#define MMQ_SMEM_PAD 8
#define MMQ_THREADS 256
// `__launch_bounds__` on these kernels passes the block size ONLY. The second argument
// (minimum blocks per SM) is deliberately omitted: setting it to 1 tells ptxas
// that a single resident block suffices, so it spends registers freely on the
// accumulator tile and occupancy drops. Left unconstrained, ptxas picks the
// register count itself, per target architecture, at compile or JIT time —
// which is where that decision belongs, since the register file and the
// latency it has to cover are properties of the device, not of the source.
// Four consecutive k values per int, which is one dp4a/mma operand word.
#define MMQ_K4 (MMQ_BK / 4)

#define MMQ_WARPS (MMQ_THREADS / WARP_SIZE)
// A staging warp covers four columns at once, so the block advances by this
// many columns or rows per staging step.
#define MMQ_STAGE_STRIDE (MMQ_WARPS * 4)
#define MMQ_W_STAGES (MMQ_BN / MMQ_STAGE_STRIDE)
#define MMQ_A_STAGES (MMQ_BM / MMQ_STAGE_STRIDE)
