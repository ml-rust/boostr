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
use numr::ops::RandomOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::error::{Error, Result};

/// The buffers one [`SamplingOps`](super::sampling::SamplingOps) call needs.
///
/// - `probs`: `[vocab_size]` F32 — softmax/top-k/top-p/min-p working buffer.
///   Used by the CUDA backend; the generic (CPU) backend filters on the
///   host and leaves this field untouched.
/// - `output`: `[1]` I64 — the sampled token, overwritten by every call.
/// - `random`: `[1]` F32 — destination for the sampling draw
///   (`RandomOps::rand_into`/`rand_seeded_into`), fully overwritten by every call.
pub struct SamplingScratch<R: Runtime> {
    /// `[vocab_size]` F32 probability working buffer.
    pub probs: Tensor<R>,
    /// `[1]` I64 sampled-token output.
    pub output: Tensor<R>,
    /// `[1]` F32 destination for the sampling draw.
    pub random: Tensor<R>,
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
            random: Tensor::zeros(&[1], DType::F32, device)?,
        })
    }

    /// Draws one sampling value into `self.random`, seeded if `seed` is
    /// `Some`, fully overwriting it. Shared by every `SamplingOps` backend
    /// so the seeded/unseeded branch lives in one place.
    ///
    /// # Errors
    ///
    /// A random-generation error propagates.
    pub fn draw_random<C: RandomOps<R>>(&self, client: &C, seed: Option<u64>) -> Result<()> {
        match seed {
            Some(seed) => client.rand_seeded_into(&self.random, seed),
            None => client.rand_into(&self.random),
        }
        .map_err(Error::Numr)
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
        assert_eq!(scratch.random.shape(), &[1]);
        assert_eq!(scratch.random.dtype(), DType::F32);
    }

    #[test]
    fn draw_random_seeded_is_reproducible_and_unseeded_runs() {
        let (client, device) = cpu_setup();
        let scratch = SamplingScratch::<CpuRuntime>::new(4, &device).expect("scratch");

        scratch.draw_random(&client, Some(7)).expect("seeded draw");
        let first = scratch.random.item::<f32>().expect("read back");
        scratch.draw_random(&client, Some(7)).expect("seeded draw");
        let second = scratch.random.item::<f32>().expect("read back");
        assert_eq!(first, second);

        scratch.draw_random(&client, None).expect("unseeded draw");
    }
}
