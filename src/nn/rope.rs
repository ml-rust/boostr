//! RoPE (Rotary Position Embedding) module
//!
//! Wraps the RoPEOps trait as a reusable module with precomputed frequency caches.

use crate::error::Result;
use crate::ops::RoPEOps;
use numr::autograd::Var;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Rotary Position Embedding module
///
/// Stores precomputed cos/sin caches for each position.
/// cos_cache, sin_cache: `[max_seq_len, head_dim/2]`
pub struct RoPE<R: Runtime> {
    cos_cache: Var<R>,
    sin_cache: Var<R>,
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

    /// Compute frequency caches on CPU: cos(pos * freq), sin(pos * freq)
    /// where freq_i = 1 / (base^(2i/dim))
    ///
    /// Returns (cos_cache, sin_cache) each `[max_seq_len, dim/2]`
    pub fn precompute_freqs(
        max_seq_len: usize,
        head_dim: usize,
        base: f32,
        device: &<R as Runtime>::Device,
    ) -> Self
    where
        R: Runtime<DType = numr::dtype::DType>,
    {
        let half_dim = head_dim / 2;
        let mut cos_data = vec![0.0f32; max_seq_len * half_dim];
        let mut sin_data = vec![0.0f32; max_seq_len * half_dim];

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos_data[pos * half_dim + i] = angle.cos();
                sin_data[pos * half_dim + i] = angle.sin();
            }
        }

        let cos_cache = Tensor::<R>::from_slice(&cos_data, &[max_seq_len, half_dim], device);
        let sin_cache = Tensor::<R>::from_slice(&sin_data, &[max_seq_len, half_dim], device);

        Self::new(cos_cache, sin_cache)
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

    pub fn cos_cache(&self) -> &Var<R> {
        &self.cos_cache
    }

    pub fn sin_cache(&self) -> &Var<R> {
        &self.sin_cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_rope_precompute_shape() {
        let device = CpuDevice::new();
        let rope = RoPE::<CpuRuntime>::precompute_freqs(128, 64, 10000.0, &device);
        assert_eq!(rope.cos_cache().shape(), &[128, 32]);
        assert_eq!(rope.sin_cache().shape(), &[128, 32]);
    }

    #[test]
    fn test_rope_precompute_values() {
        let device = CpuDevice::new();
        let rope = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, &device);

        let cos: Vec<f32> = rope.cos_cache().tensor().to_vec();
        let sin: Vec<f32> = rope.sin_cache().tensor().to_vec();

        // pos=0: all cos=1, sin=0
        for i in 0..4 {
            assert!((cos[i] - 1.0).abs() < 1e-6, "cos[0,{i}]={}", cos[i]);
            assert!(sin[i].abs() < 1e-6, "sin[0,{i}]={}", sin[i]);
        }
    }

    #[test]
    fn test_rope_forward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let rope = RoPE::<CpuRuntime>::precompute_freqs(8, 16, 10000.0, &device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 2 * 4 * 16], &[1, 2, 4, 16], &device),
            false,
        );
        let out = rope.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, 2, 4, 16]);
    }
}
