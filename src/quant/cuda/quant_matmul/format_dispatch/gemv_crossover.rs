//! Per-format GEMV/MMQ crossover for CUDA quantized matmul: the largest `m`
//! at which `dispatch_gemv` is taken over `dispatch_matmul`. Split out of
//! `gemv.rs` to stay under the `cuda/*.rs` 400-line limit.

use crate::quant::QuantFormat;
use numr::runtime::Device;

/// Largest `m` for which the GEMV path beats the feature-major MMQ path.
///
/// Consulted only for the formats without a feature-major kernel and on
/// devices without int8 MMA: a weight with that kernel takes it at every
/// `m`, so a row's bits never depend on the batch (`impl_ops.rs`). The
/// measurements below are kept for the day the crossover is needed again.
///
/// A per-token GEMV re-reads the whole weight matrix once per token, so its
/// cost scales linearly with `m`. MMQ stages a weight tile once per token
/// tile, so its cost stays flat from `m = 1` up to the tile width. The
/// token-batched dp4a kernels close most of that gap: one block decodes a
/// weight block once and dot-products it against up to a format-specific
/// number of token columns, so the weight traffic no longer grows with `m`
/// inside a tile. GEMV wins while that batched read is still cheaper than
/// staging the MMQ tile.
///
/// The crossover is per-format because decode cost per weight block differs:
/// a format with a heavier per-block unpack crosses at a lower `m`, since the
/// MMQ tile's flat cost overtakes the batched read sooner. The values below
/// are measured (see the kernel-comparison example's `--gemv` flag to compare
/// both paths at a given shape) on one GPU architecture, not derived, and
/// will need re-measuring if either kernel changes or a materially different
/// architecture is targeted.
///
/// The MMQ path needs tensor-core int8 MMA (`caps.int8_mma_m16n8k32`). On a
/// device without it, MMQ isn't available at all, so every format falls back
/// to the old, higher GEMV threshold regardless of format.
pub(in crate::quant::cuda::quant_matmul) fn gemv_max_m(
    format: QuantFormat,
    device_index: usize,
) -> usize {
    let mma_crossover = match format {
        // Q8_0 and Q6_K have the lightest per-block decode, and their batched
        // GEMV is MEASURED to stay ahead of the MMQ tile out to a 4-token
        // batch.
        //
        // The four legacy 32-element formats, the two IQ4 codebook formats and
        // the six grid-indexed IQ formats now have the `_n4` tile as well as
        // `_n2`, so their batched read no longer grows with `m` up to 4
        // either. Their 4 is PROVISIONAL: it is the widest tile they can serve,
        // not a measured crossover. Re-measure each with the example named in
        // this function's doc comment and lower the ones that lose — the
        // weight-class ordering says the grid-indexed six are the most likely
        // to come back at 3 or 2, since each of their 8-element sub-groups
        // costs a codebook read into a multi-kilobyte grid table on top of the
        // sign and scale unpack, and Q2_K showed a heavy per-block decode can
        // make batching a LOSS. At m = 1 all twelve fall through to the F32
        // GEMV, which is the only path they have there.
        QuantFormat::Q8_0
        | QuantFormat::Q6K
        | QuantFormat::Q4_0
        | QuantFormat::Q5_0
        | QuantFormat::Q4_1
        | QuantFormat::Q5_1
        | QuantFormat::IQ4NL
        | QuantFormat::IQ4XS
        | QuantFormat::IQ2XXS
        | QuantFormat::IQ2XS
        | QuantFormat::IQ2S
        | QuantFormat::IQ3XXS
        | QuantFormat::IQ3S
        | QuantFormat::IQ1S => 4,
        // Heavier decode crosses earlier: MMQ already wins at m = 3, and these
        // two have only the `_n2` tile.
        QuantFormat::Q4K | QuantFormat::Q5K => 2,
        // Heaviest decode, and neither gains from batching. Q3_K only ties
        // the MMQ tile at m = 2 and loses above it. Q2_K's small gain at a
        // deep reduction reverses into a much larger loss at a shallow one,
        // so a single crossover cannot hold for it across shapes.
        QuantFormat::Q3K | QuantFormat::Q2K => 1,
        // The three PrismML-fork formats have no feature-major kernel; the
        // alternative past the GEMV is the dequantize-then-f32 tiled GEMM.
        // Measured with `mmq_kernel_compare --format pq2_0 --n 5120 --k 5120
        // --gemv` on an RTX 3060: `_n4` (one grid-y pass per 4 tokens) beats
        // that GEMM at every m tried — m=4: 82 vs 457 us, m=8: 159 vs 482 us,
        // m=64: 1280 vs 3172 us — and scales linearly, so the GEMV is the
        // right path at every m until an int8-MMA tile exists for them.
        QuantFormat::PQ2_0 | QuantFormat::Q2_0 | QuantFormat::Q1_0 => usize::MAX,
        // PTQ1_0 has no dedicated kernel yet and stays on the generic path,
        // as does every format without a dp4a GEMV.
        QuantFormat::PTQ1_0 => 0,
        _ => 0,
    };
    let caps = numr::runtime::cuda::CudaDevice::new(device_index)
        .profile()
        .caps;
    if caps.int8_mma_m16n8k32 {
        mma_crossover
    } else {
        16
    }
}
