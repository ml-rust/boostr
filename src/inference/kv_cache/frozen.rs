//! A [`LayeredKvCache`] with every reallocating method removed.
//!
//! A CUDA graph capture bakes each layer's K/V device pointers into its
//! recorded kernels. Growing the cache after capture reallocates those
//! tensors at new addresses, which the replay never sees: it keeps reading
//! the addresses it captured. [`LayeredKvCache::freeze`] consumes the
//! growable cache and returns [`FrozenLayeredKvCache`], whose type carries
//! no `grow`, `grow_batch`, `ensure_capacity`, `update` or `update_fused` —
//! calling one is a compile error, not a runtime check. The absence of the
//! method is the guarantee.

use crate::error::Result;
use crate::inference::kv_cache::basic::KvCache;
use crate::inference::kv_cache::layered::LayeredKvCache;
use crate::ops::traits::KvCacheOps;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;

/// A layered KV cache that can no longer reallocate.
///
/// The only way to build one is [`LayeredKvCache::freeze`], which consumes
/// the growable cache, so no grow-capable handle to the same layers
/// survives the call. Every method here reads or writes the cache exactly
/// as it was allocated.
pub struct FrozenLayeredKvCache<R: Runtime> {
    inner: LayeredKvCache<R>,
}

impl<R: Runtime<DType = DType>> FrozenLayeredKvCache<R>
where
    R::Client: IndexingOps<R>,
{
    pub(super) fn from_growable(inner: LayeredKvCache<R>) -> Self {
        Self { inner }
    }

    /// Layer count.
    pub fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    /// Layer `layer_idx`'s cache, read-only, or `None` outside the layer
    /// count.
    ///
    /// This never hands out a mutable layer: the raw pointer accessors on
    /// [`KvCache`] are reachable, growth is not.
    pub fn layer(&self, layer_idx: usize) -> Option<&KvCache<R>> {
        self.inner.layer(layer_idx)
    }

    /// Tokens the first layer holds; every layer holds the same count.
    pub fn seq_len(&self) -> usize {
        self.inner.seq_len()
    }

    /// Tokens the first layer was allocated for; every layer holds the
    /// same capacity, fixed for the life of this cache.
    ///
    /// Reads 0 when there is no layer 0, the same convention
    /// [`LayeredKvCache::seq_len`] uses; a caller must check
    /// [`FrozenLayeredKvCache::num_layers`] before reading a 0 here as
    /// "capacity is exhausted" rather than "there are no layers".
    pub fn capacity(&self) -> usize {
        self.inner.layer(0).map_or(0, KvCache::capacity)
    }

    /// Clear every layer's length. Capacity, and every device address, is
    /// unchanged.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Copy a one-sequence cache into lane `lane` of every layer, at
    /// position 0.
    ///
    /// Refuses rather than grows when `src` holds more tokens than this
    /// cache's `capacity()`.
    ///
    /// # Errors
    ///
    /// [`crate::error::Error::InferenceError`] when the two caches hold a different
    /// number of layers, or when the write would run past `capacity()`.
    pub fn install_lane<C: KvCacheOps<R>>(
        &mut self,
        src: &LayeredKvCache<R>,
        lane: usize,
        client: &C,
    ) -> Result<()> {
        self.inner.install_lane_within_capacity(src, lane, client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::kv_cache::layered::LayeredKvCacheConfig;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    const HEADS: usize = 1;
    const HEAD_DIM: usize = 4;

    fn growable(
        initial_capacity: usize,
        max_seq_len: usize,
        device: &CpuDevice,
    ) -> LayeredKvCache<CpuRuntime> {
        let config = LayeredKvCacheConfig {
            batch_size: 1,
            num_kv_heads: HEADS,
            initial_capacity,
            max_seq_len,
            head_dim: HEAD_DIM,
            dtype: DType::F32,
        };
        LayeredKvCache::<CpuRuntime>::new(2, &config, device).expect("cache")
    }

    fn one_sequence(tokens: usize, device: &CpuDevice) -> LayeredKvCache<CpuRuntime> {
        let mut src = growable(tokens, tokens, device);
        let len = HEADS * tokens * HEAD_DIM;
        let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let k = Tensor::<CpuRuntime>::from_slice(&data, &[1, HEADS, tokens, HEAD_DIM], device)
            .expect("k");
        for idx in 0..src.num_layers() {
            src.layer_mut(idx)
                .expect("layer")
                .update(&k, &k)
                .expect("prefill");
        }
        src
    }

    /// A frozen cache built below `max_seq_len` refuses a lane whose
    /// sequence outgrows its capacity, and never reallocates to serve it.
    #[test]
    fn a_frozen_cache_refuses_rather_than_grows() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let growable = growable(2, 8, &device);
        let mut frozen = growable.freeze();
        let addresses_before: Vec<_> = (0..frozen.num_layers())
            .map(|idx| {
                let layer = frozen.layer(idx).expect("layer");
                (layer.k_cache_raw().ptr(), layer.v_cache_raw().ptr())
            })
            .collect();

        let src = one_sequence(5, &device);
        let err = frozen
            .install_lane(&src, 0, &client)
            .expect_err("5 tokens is past a 2-token capacity");
        let text = err.to_string();
        assert!(text.contains("lane 0"), "{text}");
        assert!(text.contains('5'), "{text}");
        assert!(text.contains('2'), "{text}");

        let addresses_after: Vec<_> = (0..frozen.num_layers())
            .map(|idx| {
                let layer = frozen.layer(idx).expect("layer");
                (layer.k_cache_raw().ptr(), layer.v_cache_raw().ptr())
            })
            .collect();
        assert_eq!(
            addresses_before, addresses_after,
            "a refused install must not reallocate"
        );
    }

    /// A frozen cache still installs a lane that fits inside its capacity.
    #[test]
    fn a_frozen_cache_installs_a_lane_that_fits() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut frozen = growable(8, 8, &device).freeze();
        let src = one_sequence(5, &device);

        frozen.install_lane(&src, 0, &client).expect("fits");
        assert_eq!(frozen.capacity(), 8);
    }
}
