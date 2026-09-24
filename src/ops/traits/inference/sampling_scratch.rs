//! Caller-owned scratch buffers a sampling call writes into, instead of
//! allocating a fresh device tensor every call.
//!
//! [`SamplingScratch::new`] allocates once, up front. [`SamplingOps`]'s
//! `_into` methods fully overwrite every field they use before reading any
//! of them back, so one scratch can be reused call after call — by the
//! same caller, never shared between two callers that could race — without
//! a later call seeing a value an earlier call left behind.
//!
//! [`SamplingOps`]: super::sampling::SamplingOps

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::error::Result;

/// The buffers one [`SamplingOps`](super::sampling::SamplingOps) call needs.
///
/// - `probs`: `[vocab_size]` F32 — softmax/top-k/top-p/min-p working buffer.
///   Used by the CUDA backend; the generic (CPU) backend filters on the
///   host and leaves this field untouched.
/// - `output`: `[1]` I64 — the sampled token, overwritten by every call.
pub struct SamplingScratch<R: Runtime> {
    /// `[vocab_size]` F32 probability working buffer.
    pub probs: Tensor<R>,
    /// `[1]` I64 sampled-token output.
    pub output: Tensor<R>,
}

impl<R: Runtime<DType = DType>> SamplingScratch<R> {
    /// Allocate a scratch sized for `vocab_size`, on `device`.
    ///
    /// # Errors
    ///
    /// An allocation error propagates.
    pub fn new(vocab_size: usize, device: &R::Device) -> Result<Self> {
        Ok(Self {
            probs: Tensor::zeros(&[vocab_size], DType::F32, device)?,
            output: Tensor::zeros(&[1], DType::I64, device)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn new_allocates_the_shapes_each_field_needs() {
        let (_client, device) = cpu_setup();
        let scratch = SamplingScratch::<CpuRuntime>::new(32, &device).expect("scratch");
        assert_eq!(scratch.probs.shape(), &[32]);
        assert_eq!(scratch.probs.dtype(), DType::F32);
        assert_eq!(scratch.output.shape(), &[1]);
        assert_eq!(scratch.output.dtype(), DType::I64);
    }
}
