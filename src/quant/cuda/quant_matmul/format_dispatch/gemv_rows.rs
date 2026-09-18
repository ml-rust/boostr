//! Output-row tile for the single-token prism GEMV. Split out of `gemv.rs`
//! to stay under the `cuda/*.rs` 400-line limit; `PRISM_GEMV_ROWS` is the
//! one const to flip.

/// Output columns per block for the three PrismML-fork formats at `m = 1`.
///
/// Their single-token dp4a kernel exists at 1 (`_mwr`), 4 (`_r4`) and 8
/// (`_r8`) columns per block; a block that owns ROWS columns loads each
/// activation word once for all of them, so the activation re-read that
/// dominates the one-column geometry drops by ROWS. 4 vs 8 is TO BE
/// MEASURED (blazr decode under nsys at K = N = 5120 PQ2_0); flip this
/// const and rebuild. A row's bits are the same at every value, see
/// `kernels/gemv/legacy_ntok_body.cuh`.
pub(in crate::quant::cuda::quant_matmul) const PRISM_GEMV_ROWS: u32 = 4;

// Compile-time only: a value with no compiled kernel fails the build here,
// never at a launch.
const _: () = {
    if !matches!(PRISM_GEMV_ROWS, 1 | 4 | 8) {
        panic!("PRISM_GEMV_ROWS must name a compiled kernel: 1, 4 or 8");
    }
};

/// The single-token prism kernel name for `PRISM_GEMV_ROWS`.
pub(in crate::quant::cuda::quant_matmul) const fn prism_mwr_kernel(
    r1: &'static str,
    r4: &'static str,
    r8: &'static str,
) -> &'static str {
    match PRISM_GEMV_ROWS {
        8 => r8,
        4 => r4,
        _ => r1,
    }
}
