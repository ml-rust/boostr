//! LayeredKvCache - Multi-layer KV cache for a full transformer model

use crate::error::Result;
use crate::inference::kv_cache::basic::KvCache;
use crate::inference::kv_cache::frozen::FrozenLayeredKvCache;
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
    pub(super) layers: Vec<KvCache<R>>,
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
        self.layers.first().map_or(0, KvCache::seq_len)
    }

    /// Step the first `rows` lanes of every layer — see
    /// [`KvCache::set_active_batch`].
    ///
    /// Every layer of a stack is written and read by the same decode call,
    /// so they move together or a call would meet one width on one layer
    /// and another on the next.
    ///
    /// # Errors
    ///
    /// Propagates the first layer's refusal: `rows` is 0 or past the
    /// allocated batch.
    pub fn set_active_batch(&mut self, rows: usize) -> Result<()> {
        for layer in &mut self.layers {
            layer.set_active_batch(rows)?;
        }
        Ok(())
    }

    /// Lanes a decode call steps, from the first layer;
    /// 0 when the stack holds no layer.
    pub fn active_batch(&self) -> usize {
        self.layers.first().map_or(0, KvCache::active_batch)
    }

    /// Consume this growable cache and hand back a frozen handle over the
    /// same storage.
    ///
    /// After this call no growable handle to these layers exists anywhere:
    /// [`FrozenLayeredKvCache`] has no `grow`, `grow_batch`,
    /// `ensure_capacity`, `update` or `update_fused`, so nothing reachable
    /// through it can move a layer's K or V tensor off the address it was
    /// allocated at — on the key axis or the lane axis.
    pub fn freeze(self) -> FrozenLayeredKvCache<R> {
        FrozenLayeredKvCache::from_growable(self)
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
