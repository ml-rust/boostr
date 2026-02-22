//! KV Cache - Dynamic Key-Value cache for efficient inference
//!
//! Provides a dynamic KV cache that grows as needed, avoiding massive upfront allocation.

use crate::error::{Error, Result};
use crate::ops::traits::KvCacheOps;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Dynamic KV cache for a single attention layer
///
/// Stores K and V tensors with shape [batch, num_kv_heads, capacity, head_dim].
pub struct KvCache<R: Runtime> {
    k_cache: Tensor<R>,
    v_cache: Tensor<R>,
    seq_len: usize,
    capacity: usize,
    max_seq_len: usize,
    batch_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: DType,
    device: R::Device,
}

impl<R: Runtime<DType = DType>> KvCache<R>
where
    R::Client: IndexingOps<R>,
{
    /// Create a new dynamic KV cache
    pub fn new(
        batch_size: usize,
        num_kv_heads: usize,
        initial_capacity: usize,
        max_seq_len: usize,
        head_dim: usize,
        dtype: DType,
        device: &R::Device,
    ) -> Result<Self> {
        let capacity = initial_capacity.min(max_seq_len);
        let shape = [batch_size, num_kv_heads, capacity, head_dim];

        let k_cache = Tensor::<R>::zeros(&shape, dtype, device);
        let v_cache = Tensor::<R>::zeros(&shape, dtype, device);

        Ok(Self {
            k_cache,
            v_cache,
            seq_len: 0,
            capacity,
            max_seq_len,
            batch_size,
            num_kv_heads,
            head_dim,
            dtype,
            device: device.clone(),
        })
    }

    /// Grow the cache to accommodate more tokens
    fn grow(&mut self, min_capacity: usize) -> Result<()> {
        let mut new_capacity = self.capacity * 2;
        while new_capacity < min_capacity {
            new_capacity *= 2;
        }
        new_capacity = new_capacity.min(self.max_seq_len);

        if new_capacity <= self.capacity {
            return Err(Error::InferenceError {
                reason: format!(
                    "cannot grow beyond max_seq_len {}: need {} tokens",
                    self.max_seq_len, min_capacity
                ),
            });
        }

        let new_shape = [
            self.batch_size,
            self.num_kv_heads,
            new_capacity,
            self.head_dim,
        ];
        let mut new_k_cache = Tensor::<R>::zeros(&new_shape, self.dtype, &self.device);
        let mut new_v_cache = Tensor::<R>::zeros(&new_shape, self.dtype, &self.device);

        if self.seq_len > 0 {
            let old_k = self.k_cache.narrow(2, 0, self.seq_len)?;
            let old_v = self.v_cache.narrow(2, 0, self.seq_len)?;

            new_k_cache = new_k_cache.slice_assign(&old_k, 2, 0)?;
            new_v_cache = new_v_cache.slice_assign(&old_v, 2, 0)?;
        }

        self.k_cache = new_k_cache;
        self.v_cache = new_v_cache;
        self.capacity = new_capacity;

        Ok(())
    }

    fn ensure_capacity(&mut self, required: usize) -> Result<()> {
        if required > self.capacity {
            self.grow(required)?;
        }
        Ok(())
    }

    /// Update the cache with new K and V values using slice_assign
    ///
    /// # Arguments
    /// * `k` - New K values [batch, num_kv_heads, new_tokens, head_dim]
    /// * `v` - New V values [batch, num_kv_heads, new_tokens, head_dim]
    pub fn update(&mut self, k: &Tensor<R>, v: &Tensor<R>) -> Result<()> {
        let (_, _, new_tokens, _) = k.dims4()?;
        let required = self.seq_len + new_tokens;

        if required > self.max_seq_len {
            return Err(Error::InferenceError {
                reason: format!(
                    "cache overflow: {} + {} > max_seq_len {}",
                    self.seq_len, new_tokens, self.max_seq_len
                ),
            });
        }

        self.ensure_capacity(required)?;

        self.k_cache = self.k_cache.slice_assign(k, 2, self.seq_len)?;
        self.v_cache = self.v_cache.slice_assign(v, 2, self.seq_len)?;

        self.seq_len += new_tokens;

        Ok(())
    }

    /// Update using a fused CUDA/CPU kernel (single launch instead of slice_assign).
    ///
    /// Requires `R::Client: KvCacheOps<R>`. Falls back to `update` if the
    /// runtime client doesn't support fused KV cache operations.
    pub fn update_fused(&mut self, k: &Tensor<R>, v: &Tensor<R>, client: &R::Client) -> Result<()>
    where
        R::Client: KvCacheOps<R>,
    {
        let (_, _, new_tokens, _) = k.dims4()?;
        let required = self.seq_len + new_tokens;

        if required > self.max_seq_len {
            return Err(Error::InferenceError {
                reason: format!(
                    "cache overflow: {} + {} > max_seq_len {}",
                    self.seq_len, new_tokens, self.max_seq_len
                ),
            });
        }

        self.ensure_capacity(required)?;

        client.kv_cache_update(&self.k_cache, &self.v_cache, k, v, self.seq_len)?;
        self.seq_len += new_tokens;

        Ok(())
    }

    /// Get the current K and V tensors (narrow views of the used portion)
    pub fn get_kv(&self) -> Result<(Tensor<R>, Tensor<R>)> {
        if self.seq_len == 0 {
            return Err(Error::InferenceError {
                reason: "cache is empty".into(),
            });
        }

        let k = self.k_cache.narrow(2, 0, self.seq_len)?;
        let v = self.v_cache.narrow(2, 0, self.seq_len)?;

        Ok((k, v))
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

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

/// Paged KV cache for a single attention layer.
///
/// Uses block-based storage with slot mapping for PagedAttention.
/// Cache layout: `[num_blocks, block_size, num_heads, head_dim]`
pub struct PagedKvCache<R: Runtime> {
    k_cache: Tensor<R>,
    v_cache: Tensor<R>,
    num_blocks: usize,
    block_size: usize,
    num_heads: usize,
    head_dim: usize,
    dtype: DType,
}

impl<R: Runtime<DType = DType>> PagedKvCache<R> {
    /// Create a new paged KV cache.
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &R::Device,
    ) -> Self {
        let shape = [num_blocks, block_size, num_heads, head_dim];
        let k_cache = Tensor::<R>::zeros(&shape, dtype, device);
        let v_cache = Tensor::<R>::zeros(&shape, dtype, device);

        Self {
            k_cache,
            v_cache,
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            dtype,
        }
    }

    /// Write new K/V tokens into cache blocks using slot_mapping.
    ///
    /// - `key`, `value`: `[num_tokens, num_heads, head_dim]`
    /// - `slot_mapping`: `[num_tokens]` (I64) — maps each token to a slot
    ///
    /// Slot `s` maps to block `s / block_size`, offset `s % block_size`.
    pub fn update(
        &self,
        key: &Tensor<R>,
        value: &Tensor<R>,
        slot_mapping: &Tensor<R>,
        client: &R::Client,
    ) -> Result<()>
    where
        R::Client: KvCacheOps<R>,
    {
        client.reshape_and_cache(
            key,
            value,
            &self.k_cache,
            &self.v_cache,
            slot_mapping,
            self.block_size,
        )
    }

    pub fn k_cache(&self) -> &Tensor<R> {
        &self.k_cache
    }

    pub fn v_cache(&self) -> &Tensor<R> {
        &self.v_cache
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_kv_cache_create() {
        let device = numr::runtime::cpu::CpuDevice::new();
        let cache = KvCache::<CpuRuntime>::new(1, 4, 64, 2048, 32, DType::F32, &device).unwrap();
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.capacity(), 64);
        assert_eq!(cache.max_seq_len(), 2048);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_kv_cache_update_and_get() {
        let device = numr::runtime::cpu::CpuDevice::new();
        let mut cache = KvCache::<CpuRuntime>::new(1, 2, 64, 2048, 4, DType::F32, &device).unwrap();

        // Create K and V tensors [1, 2, 3, 4] (3 new tokens)
        let k = Tensor::<CpuRuntime>::zeros(&[1, 2, 3, 4], DType::F32, &device);
        let v = Tensor::<CpuRuntime>::zeros(&[1, 2, 3, 4], DType::F32, &device);

        cache.update(&k, &v).unwrap();
        assert_eq!(cache.seq_len(), 3);

        let (got_k, got_v) = cache.get_kv().unwrap();
        assert_eq!(got_k.shape(), &[1, 2, 3, 4]);
        assert_eq!(got_v.shape(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_kv_cache_grow() {
        let device = numr::runtime::cpu::CpuDevice::new();
        // Small initial capacity of 4
        let mut cache = KvCache::<CpuRuntime>::new(1, 2, 4, 2048, 4, DType::F32, &device).unwrap();
        assert_eq!(cache.capacity(), 4);

        // Fill beyond capacity - should trigger growth
        let k = Tensor::<CpuRuntime>::zeros(&[1, 2, 5, 4], DType::F32, &device);
        let v = Tensor::<CpuRuntime>::zeros(&[1, 2, 5, 4], DType::F32, &device);
        cache.update(&k, &v).unwrap();

        assert_eq!(cache.seq_len(), 5);
        assert!(cache.capacity() >= 5);
    }

    #[test]
    fn test_kv_cache_overflow() {
        let device = numr::runtime::cpu::CpuDevice::new();
        let mut cache = KvCache::<CpuRuntime>::new(1, 2, 4, 8, 4, DType::F32, &device).unwrap();

        let k = Tensor::<CpuRuntime>::zeros(&[1, 2, 10, 4], DType::F32, &device);
        let v = Tensor::<CpuRuntime>::zeros(&[1, 2, 10, 4], DType::F32, &device);
        let result = cache.update(&k, &v);
        assert!(result.is_err());
    }

    #[test]
    fn test_kv_cache_reset() {
        let device = numr::runtime::cpu::CpuDevice::new();
        let mut cache = KvCache::<CpuRuntime>::new(1, 2, 64, 2048, 4, DType::F32, &device).unwrap();

        let k = Tensor::<CpuRuntime>::zeros(&[1, 2, 3, 4], DType::F32, &device);
        let v = Tensor::<CpuRuntime>::zeros(&[1, 2, 3, 4], DType::F32, &device);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.seq_len(), 3);

        cache.reset();
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.is_empty());
    }

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

    #[test]
    fn test_get_kv_empty() {
        let device = numr::runtime::cpu::CpuDevice::new();
        let cache = KvCache::<CpuRuntime>::new(1, 2, 64, 2048, 4, DType::F32, &device).unwrap();
        assert!(cache.get_kv().is_err());
    }

    #[test]
    fn test_update_fused() {
        let (client, device) = crate::test_utils::cpu_setup();
        let mut cache = KvCache::<CpuRuntime>::new(1, 2, 64, 2048, 4, DType::F32, &device).unwrap();

        let k_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let v_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.2).collect();
        let k = Tensor::<CpuRuntime>::from_slice(&k_data, &[1, 2, 3, 4], &device);
        let v = Tensor::<CpuRuntime>::from_slice(&v_data, &[1, 2, 3, 4], &device);

        cache.update_fused(&k, &v, &client).unwrap();
        assert_eq!(cache.seq_len(), 3);

        let (got_k, got_v) = cache.get_kv().unwrap();
        assert_eq!(got_k.shape(), &[1, 2, 3, 4]);
        assert_eq!(got_v.shape(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_paged_kv_cache_update() {
        let (client, device) = crate::test_utils::cpu_setup();

        let num_blocks = 4;
        let block_size = 8;
        let num_heads = 2;
        let head_dim = 4;

        let cache = PagedKvCache::<CpuRuntime>::new(
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            DType::F32,
            &device,
        );

        // 3 tokens to write
        let num_tokens = 3;
        let k_data: Vec<f32> = (0..num_tokens * num_heads * head_dim)
            .map(|i| i as f32 * 0.1)
            .collect();
        let v_data: Vec<f32> = (0..num_tokens * num_heads * head_dim)
            .map(|i| i as f32 * 0.2)
            .collect();
        let key =
            Tensor::<CpuRuntime>::from_slice(&k_data, &[num_tokens, num_heads, head_dim], &device);
        let value =
            Tensor::<CpuRuntime>::from_slice(&v_data, &[num_tokens, num_heads, head_dim], &device);

        // slot_mapping: token 0 → slot 0, token 1 → slot 1, token 2 → slot 9
        let slots: Vec<i32> = vec![0, 1, 9];
        let slot_mapping = Tensor::<CpuRuntime>::from_slice(&slots, &[num_tokens], &device);

        cache.update(&key, &value, &slot_mapping, &client).unwrap();

        // Verify cache shape
        assert_eq!(
            cache.k_cache().shape(),
            &[num_blocks, block_size, num_heads, head_dim]
        );
        assert_eq!(
            cache.v_cache().shape(),
            &[num_blocks, block_size, num_heads, head_dim]
        );

        // Verify written data: slot 0 = block 0, offset 0
        let kc = cache.k_cache().to_vec::<f32>();
        // k_data[0] = 0.0 (token 0, head 0, dim 0)
        assert!((kc[0] - 0.0).abs() < 1e-6);
        // k_data for token 0, head 0, dim 1 = 0.1
        assert!((kc[1] - 0.1).abs() < 1e-6);
    }
}
