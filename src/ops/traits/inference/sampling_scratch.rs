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
use numr::ops::{AdvancedRandomOps, BinaryOps, RandomOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::sample_seed::SampleSeed;
use crate::error::{Error, Result};

/// The buffers one [`SamplingOps`](super::sampling::SamplingOps) call needs.
///
/// - `probs`: `[vocab_size]` F32 — softmax/top-k/top-p/min-p working buffer.
///   Used by the CUDA backend; the generic (CPU) backend filters on the
///   host and leaves this field untouched.
/// - `output`: `[1]` I64 — the sampled token, overwritten by every call.
/// - `random`: `[1]` F32 — destination for the sampling draw
///   (`AdvancedRandomOps::philox_uniform` when seeded, `RandomOps::rand_into`
///   otherwise), fully overwritten by every call.
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

    /// Draws one sampling value into `self.random`, fully overwriting it.
    /// Shared by every `SamplingOps` backend so the seeded/unseeded branch
    /// lives in one place.
    ///
    /// With `Some(seed)`, the value is the Philox4x32-10 uniform at
    /// `(seed.key, seed.counter)`. It is a pure function of that pair, so
    /// each decode step must pass its own counter. With `None`, the value
    /// comes from the backend's unseeded generator.
    ///
    /// # Errors
    ///
    /// A random-generation or device-copy error propagates.
    pub fn draw_random<C>(&self, client: &C, seed: Option<SampleSeed>) -> Result<()>
    where
        C: RandomOps<R> + AdvancedRandomOps<R> + BinaryOps<R>,
    {
        match seed {
            Some(seed) => client
                .philox_uniform(&[1], seed.key, seed.counter, DType::F32)
                .and_then(|draw| client.copy_into(&self.random, &draw)),
            None => client.rand_into(&self.random),
        }
        .map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuRuntime};

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

    /// The seeded draw at `(key, counter)`, read back from the scratch.
    fn seeded_draw(
        client: &CpuClient,
        scratch: &SamplingScratch<CpuRuntime>,
        key: u64,
        counter: u64,
    ) -> f32 {
        scratch
            .draw_random(client, Some(SampleSeed::new(key, counter)))
            .expect("seeded draw");
        scratch.random.item::<f32>().expect("read back")
    }

    #[test]
    fn draw_random_seeded_is_reproducible_and_unseeded_runs() {
        let (client, device) = cpu_setup();
        let scratch = SamplingScratch::<CpuRuntime>::new(4, &device).expect("scratch");

        let first = seeded_draw(&client, &scratch, 7, 0);
        let second = seeded_draw(&client, &scratch, 7, 0);
        assert_eq!(first.to_bits(), second.to_bits());

        scratch.draw_random(&client, None).expect("unseeded draw");
    }

    /// A seeded draw is a pure function of `(key, counter)`: a draw in
    /// between does not change the value a pair gives.
    #[test]
    fn same_key_and_counter_give_the_same_draw() {
        let (client, device) = cpu_setup();
        let scratch = SamplingScratch::<CpuRuntime>::new(4, &device).expect("scratch");
        let before = seeded_draw(&client, &scratch, 42, 5);
        let _other = seeded_draw(&client, &scratch, 42, 6);
        let after = seeded_draw(&client, &scratch, 42, 5);
        assert_eq!(before.to_bits(), after.to_bits());
    }

    /// Consecutive counters of one key draw different uniforms, each in
    /// `[0, 1)`. A reseed-per-call draw gives one value for every counter.
    #[test]
    fn different_counters_give_different_draws() {
        let (client, device) = cpu_setup();
        let scratch = SamplingScratch::<CpuRuntime>::new(4, &device).expect("scratch");
        let draws: Vec<f32> = (0..16)
            .map(|c| seeded_draw(&client, &scratch, 42, c))
            .collect();
        for (c, d) in draws.iter().enumerate() {
            assert!(
                (0.0..1.0).contains(d),
                "counter {c} drew {d}, outside [0, 1)"
            );
        }
        for (c, pair) in draws.windows(2).enumerate() {
            assert_ne!(
                pair[0].to_bits(),
                pair[1].to_bits(),
                "counters {c} and {} match",
                c + 1
            );
        }
    }

    /// Key 0 is a full stream: counters 0 and 1 differ, and its first
    /// draws are not one repeated value.
    #[test]
    fn key_zero_is_not_degenerate() {
        let (client, device) = cpu_setup();
        let scratch = SamplingScratch::<CpuRuntime>::new(4, &device).expect("scratch");
        let c0 = seeded_draw(&client, &scratch, 0, 0);
        let c1 = seeded_draw(&client, &scratch, 0, 1);
        assert_ne!(c0.to_bits(), c1.to_bits());

        let draws: Vec<u32> = (0..8)
            .map(|c| seeded_draw(&client, &scratch, 0, c).to_bits())
            .collect();
        assert!(
            draws.iter().any(|&d| d != draws[0]),
            "key 0 drew one value for every counter: {draws:?}"
        );
        let key_one = seeded_draw(&client, &scratch, 1, 0);
        assert_ne!(
            c0.to_bits(),
            key_one.to_bits(),
            "keys 0 and 1 match at counter 0"
        );
    }
}
