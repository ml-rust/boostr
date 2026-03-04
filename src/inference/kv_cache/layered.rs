//! LayeredKvCache - Multi-layer KV cache for a full transformer model

use crate::error::Result;
use crate::inference::kv_cache::basic::KvCache;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;

/// Configuration for creating a layered KV cache
#[derive(Debug, Clone)]
pub struct LayeredKvCacheConfig {
    pub batch_size: usize,
    pub num_kv_heads: usize,
    pub initial_capacity: usize,
    pub max_seq_len: usize,
    pub head_dim: usize,
    pub dtype: DType,
}

/// Multi-layer KV cache for a full transformer model
pub struct LayeredKvCache<R: Runtime> {
    layers: Vec<KvCache<R>>,
}

impl<R: Runtime<DType = DType>> LayeredKvCache<R>
where
    R::Client: IndexingOps<R>,
{
    pub fn new(
        num_layers: usize,
        config: &LayeredKvCacheConfig,
        device: &R::Device,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(KvCache::new(
                config.batch_size,
                config.num_kv_heads,
                config.initial_capacity,
                config.max_seq_len,
                config.head_dim,
                config.dtype,
                device,
            )?);
        }
        Ok(Self { layers })
    }

    /// Convenience constructor matching the positional-arg API used by blazr.
    #[allow(clippy::too_many_arguments)]
    pub fn new_positional(
        num_layers: usize,
        batch_size: usize,
        num_kv_heads: usize,
        initial_capacity: usize,
        max_seq_len: usize,
        head_dim: usize,
        dtype: DType,
        device: &R::Device,
    ) -> Result<Self> {
        let config = LayeredKvCacheConfig {
            batch_size,
            num_kv_heads,
            initial_capacity,
            max_seq_len,
            head_dim,
            dtype,
        };
        Self::new(num_layers, &config, device)
    }

    pub fn layer_mut(&mut self, layer_idx: usize) -> Option<&mut KvCache<R>> {
        self.layers.get_mut(layer_idx)
    }

    pub fn layer(&self, layer_idx: usize) -> Option<&KvCache<R>> {
        self.layers.get(layer_idx)
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_layered_kv_cache() {
        let device = numr::runtime::cpu::CpuDevice::new();
        let config = LayeredKvCacheConfig {
            batch_size: 1,
            num_kv_heads: 2,
            initial_capacity: 64,
            max_seq_len: 2048,
            head_dim: 32,
            dtype: DType::F32,
        };
        let cache = LayeredKvCache::<CpuRuntime>::new(4, &config, &device).unwrap();
        assert_eq!(cache.num_layers(), 4);
        assert_eq!(cache.seq_len(), 0);
    }
}
