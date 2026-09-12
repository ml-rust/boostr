//! The [`RoPE`] cache type: construction from caches, forward, dtype cast,
//! narrowing, and aliasing.

use crate::error::{Error, Result};
use crate::ops::RoPEOps;
use numr::autograd::Var;
use numr::ops::TypeConversionOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Rotary Position Embedding module
///
/// Stores precomputed cos/sin caches for each position.
/// cos_cache, sin_cache: `[max_seq_len, head_dim/2]`
pub struct RoPE<R: Runtime> {
    pub(super) cos_cache: Var<R>,
    pub(super) sin_cache: Var<R>,
}

impl<R: Runtime> RoPE<R> {
    /// Create from precomputed cos/sin caches.
    ///
    /// cos_cache, sin_cache: `[max_seq_len, head_dim/2]`
    pub fn new(cos_cache: Tensor<R>, sin_cache: Tensor<R>) -> Self {
        Self {
            cos_cache: Var::new(cos_cache, false),
            sin_cache: Var::new(sin_cache, false),
        }
    }

    /// Apply RoPE to input tensor `x: [B, H, S, D]`
    ///
    /// Uses the cached cos/sin values for positions 0..S.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + RoPEOps<R>,
    {
        client.apply_rope(x, &self.cos_cache, &self.sin_cache)
    }

    /// Cast cos/sin caches to the given dtype (e.g. BF16) so that
    /// per-token casts are avoided during inference.
    ///
    /// Errors propagate: a cast that fails must not leave the caches at a
    /// dtype the attention path will later reject with a bare
    /// `DTypeMismatch` far from its cause.
    pub fn cast_caches(&mut self, dtype: numr::dtype::DType) -> Result<()>
    where
        R: Runtime<DType = numr::dtype::DType>,
        R::Client: numr::ops::TypeConversionOps<R>,
    {
        if self.cos_cache.tensor().dtype() == dtype {
            return Ok(());
        }
        let device = self.cos_cache.tensor().device().clone();
        let client = R::default_client(&device);
        self.cos_cache = Var::new(client.cast(self.cos_cache.tensor(), dtype)?, false);
        self.sin_cache = Var::new(client.cast(self.sin_cache.tensor(), dtype)?, false);
        Ok(())
    }

    pub fn cos_cache(&self) -> &Var<R> {
        &self.cos_cache
    }

    pub fn sin_cache(&self) -> &Var<R> {
        &self.sin_cache
    }

    /// Cheap duplicate that preserves `cos_cache`'s and `sin_cache`'s
    /// `TensorId`s, for capturing this table by owned value in a `'static`
    /// activation-checkpointing closure. Both caches are non-trainable
    /// (`requires_grad = false`), so no gradient is at stake here, but
    /// [`Var::alias`] is used anyway — not [`Clone`] — to avoid needless id
    /// churn and stay consistent with every other aliased field a
    /// checkpointed layer captures.
    pub fn alias(&self) -> Self {
        Self {
            cos_cache: self.cos_cache.alias(),
            sin_cache: self.sin_cache.alias(),
        }
    }

    /// Keep only the first `num_positions` rows of the cos/sin caches.
    ///
    /// The caches are built at the model's configured `max_position_embeddings`
    /// because that length selects the scaling regime, but a model that only
    /// ever rotates a few positions does not need the rest resident.
    pub fn narrow_positions(&self, num_positions: usize) -> Result<Self> {
        let len = self.cos_cache.tensor().shape()[0];
        if num_positions == 0 || num_positions > len {
            return Err(Error::InvalidArgument {
                arg: "num_positions",
                reason: format!(
                    "num_positions must be nonzero and at most the cache length {len}, got {num_positions}"
                ),
            });
        }
        let cos = self
            .cos_cache
            .tensor()
            .narrow(0, 0, num_positions)?
            .contiguous()?;
        let sin = self
            .sin_cache
            .tensor()
            .narrow(0, 0, num_positions)?
            .contiguous()?;
        Ok(Self::new(cos, sin))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_rope_forward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let rope = RoPE::<CpuRuntime>::precompute_freqs(8, 16, 10000.0, None, &device).unwrap();

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 2 * 4 * 16], &[1, 2, 4, 16], &device)
                .unwrap(),
            false,
        );
        let out = rope.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, 2, 4, 16]);
    }

    #[test]
    fn test_rope_narrow_positions_keeps_matching_rows() {
        let device = CpuDevice::new();
        let rope = RoPE::<CpuRuntime>::precompute_freqs(128, 64, 10000.0, None, &device).unwrap();
        let narrowed = rope.narrow_positions(5).unwrap();
        assert_eq!(narrowed.cos_cache().shape(), &[5, 32]);
        assert_eq!(narrowed.sin_cache().shape(), &[5, 32]);

        let cos_full: Vec<f32> = rope.cos_cache().tensor().to_vec();
        let sin_full: Vec<f32> = rope.sin_cache().tensor().to_vec();
        let cos_narrow: Vec<f32> = narrowed.cos_cache().tensor().to_vec();
        let sin_narrow: Vec<f32> = narrowed.sin_cache().tensor().to_vec();
        assert_eq!(cos_narrow, cos_full[..5 * 32]);
        assert_eq!(sin_narrow, sin_full[..5 * 32]);
    }

    #[test]
    fn test_rope_narrow_positions_rejects_zero() {
        let device = CpuDevice::new();
        let rope = RoPE::<CpuRuntime>::precompute_freqs(128, 64, 10000.0, None, &device).unwrap();
        let err = rope
            .narrow_positions(0)
            .err()
            .expect("num_positions=0 must error");
        assert!(
            err.to_string().contains("num_positions"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_rope_narrow_positions_rejects_over_length() {
        let device = CpuDevice::new();
        let rope = RoPE::<CpuRuntime>::precompute_freqs(128, 64, 10000.0, None, &device).unwrap();
        let err = rope
            .narrow_positions(129)
            .err()
            .expect("num_positions > cache length must error");
        let msg = err.to_string();
        assert!(msg.contains("128"), "unexpected error: {msg}");
        assert!(msg.contains("129"), "unexpected error: {msg}");
    }
}
