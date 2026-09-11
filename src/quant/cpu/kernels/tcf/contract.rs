//! The activation contract the CPU TCF matmul kernel satisfies.
//!
//! Declared beside the kernel rather than at the dispatch site: what a kernel
//! computes is a property of the kernel, and a router that restated it could
//! drift from it.

use crate::quant::KernelContract;

/// The activation contract [`super::tcf_matmul_f32`] satisfies.
///
/// The kernel reads the caller's activation as f32 and accumulates the dot
/// product in f32. Nothing on the activation side is quantized, so the values
/// the dot product sees are the values the caller handed in.
///
/// Reassociates: 8-lane AVX2 FMA accumulators plus a horizontal reduction
/// (`matmul.rs:44-49`), not a fixed left-to-right sum.
pub const MATMUL_CONTRACT: KernelContract = KernelContract::f32_activation("cpu tcf_matmul_f32");

#[cfg(test)]
mod tests {
    use super::*;

    /// A future edit to this constant that drops reassociation must flip
    /// this back to `true` only if the kernel truly stops reordering.
    #[test]
    fn declares_that_it_reassociates() {
        const { assert!(MATMUL_CONTRACT.reassociates) };
    }
}
