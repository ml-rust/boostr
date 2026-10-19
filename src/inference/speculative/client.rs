//! The client bounds of speculative verification.

use crate::ops::traits::inference::{SamplingOps, SamplingProbsOps, SpeculativeOps};
use numr::ops::{AdvancedRandomOps, BinaryOps, IndexingOps, ShapeOps, TypeConversionOps};
use numr::runtime::Runtime;

/// Every op [`verify_logits`](super::verify_logits) runs on the device.
///
/// A blanket impl covers each client with all the bounds. A model adapter can
/// take `C: ModelClient<R> + SpeculativeClient<R>` and pass its client on.
pub trait SpeculativeClient<R: Runtime>:
    SamplingProbsOps<R>
    + SpeculativeOps<R>
    + SamplingOps<R>
    + AdvancedRandomOps<R>
    + BinaryOps<R>
    + IndexingOps<R>
    + ShapeOps<R>
    + TypeConversionOps<R>
{
}

impl<R, C> SpeculativeClient<R> for C
where
    R: Runtime,
    C: SamplingProbsOps<R>
        + SpeculativeOps<R>
        + SamplingOps<R>
        + AdvancedRandomOps<R>
        + BinaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TypeConversionOps<R>,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    fn takes_client<C: SpeculativeClient<CpuRuntime>>(_: &C) {}

    #[test]
    fn the_cpu_client_is_a_speculative_client() {
        let (client, _) = crate::test_utils::cpu_setup();
        takes_client(&client);
    }
}
