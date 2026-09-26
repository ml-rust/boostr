//! Growing a KV cache along its LANE axis.
//!
//! A cache is allocated `[batch, num_kv_heads, capacity, head_dim]`. Growth
//! keeps the reservation at what the caller has seated, rather than every
//! lane up front.
//!
//! Lanes are the OUTERMOST dimension, so the rows a cache already holds are a
//! contiguous prefix of the larger allocation. Growth allocates the wider
//! tensor, copies the old rows into its prefix, leaves the rows above zeroed
//! and swaps. A lane keeps its index, keys and length across the move.
//!
//! The allocation NEVER shrinks.

use crate::error::Result;
use crate::inference::kv_cache::basic::KvCache;
use crate::inference::kv_cache::layered::LayeredKvCache;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> KvCache<R>
where
    R::Client: IndexingOps<R>,
{
    /// Widen the lane axis to `new_batch` lanes, keeping every lane this
    /// cache already holds.
    ///
    /// `[batch_size, H_kv, capacity, D]` is reallocated at `new_batch`
    /// rows, the existing rows are copied into its prefix, and the rows
    /// above them start zeroed and free. Per-lane lengths are extended with
    /// zeros, so a new lane reads as an empty key range until something
    /// seats a sequence in it.
    ///
    /// The STEPPED width is untouched;
    /// [`set_active_batch`](KvCache::set_active_batch) owns it.
    ///
    /// A no-op when `new_batch` is not larger than the current allocation.
    /// The allocation never shrinks.
    ///
    /// # Device addresses
    ///
    /// This REPLACES `k_cache` and `v_cache`. Addresses
    /// [`k_cache_raw`](KvCache::k_cache_raw) and
    /// [`v_cache_raw`](KvCache::v_cache_raw) handed out earlier no longer
    /// describe the buffers a step writes. A caller holding a CUDA graph
    /// captured against them must not grow the cache.
    ///
    /// # Errors
    ///
    /// An allocation or copy error propagates.
    pub fn grow_batch(&mut self, new_batch: usize) -> Result<()> {
        if new_batch <= self.batch_size {
            return Ok(());
        }
        let shape = [new_batch, self.num_kv_heads, self.capacity, self.head_dim];
        let mut k = Tensor::<R>::zeros(&shape, self.dtype, &self.device)?;
        let mut v = Tensor::<R>::zeros(&shape, self.dtype, &self.device)?;

        // The rows already held are the prefix of the wider tensor, so one
        // dim-0 assignment at offset 0 carries them across byte for byte.
        k = k.slice_assign(&self.k_cache, 0, 0)?;
        v = v.slice_assign(&self.v_cache, 0, 0)?;

        self.k_cache = k;
        self.v_cache = v;
        self.lane_lens.resize(new_batch, 0);
        self.batch_size = new_batch;
        // Both buffers are new tensors, so a view of the old ones describes
        // storage nothing writes.
        self.refresh_active_views()
    }
}

impl<R: Runtime<DType = DType>> LayeredKvCache<R>
where
    R::Client: IndexingOps<R>,
{
    /// Widen every layer's lane axis to `new_batch` — see
    /// [`KvCache::grow_batch`].
    ///
    /// One decode call reads and writes every layer, so they grow together.
    ///
    /// A no-op when `new_batch` is not larger than the layers' current
    /// allocation. The same device-address caveat applies: every layer's K
    /// and V buffer is replaced, so a caller holding a graph captured against
    /// them must not grow the stack.
    ///
    /// # Errors
    ///
    /// Propagates the first layer's allocation or copy error.
    pub fn grow_batch(&mut self, new_batch: usize) -> Result<()> {
        for layer in &mut self.layers {
            layer.grow_batch(new_batch)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::kv_cache::LayeredKvCacheConfig;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    const HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const CAPACITY: usize = 8;
    const ROW: usize = HEADS * CAPACITY * HEAD_DIM;

    fn cache(batch: usize, device: &CpuDevice) -> KvCache<CpuRuntime> {
        KvCache::<CpuRuntime>::new(batch, HEADS, CAPACITY, 64, HEAD_DIM, DType::F32, device)
            .expect("cache")
    }

    /// One sequence of `tokens` positions, every element distinct.
    fn kv(
        tokens: usize,
        seed: f32,
        device: &CpuDevice,
    ) -> (Tensor<CpuRuntime>, Tensor<CpuRuntime>) {
        let len = HEADS * tokens * HEAD_DIM;
        let k: Vec<f32> = (0..len).map(|i| seed + i as f32).collect();
        let v: Vec<f32> = (0..len).map(|i| seed - i as f32).collect();
        let shape = [1, HEADS, tokens, HEAD_DIM];
        (
            Tensor::<CpuRuntime>::from_slice(&k, &shape, device).expect("k"),
            Tensor::<CpuRuntime>::from_slice(&v, &shape, device).expect("v"),
        )
    }

    /// Growth carries the rows already seated across byte for byte, and the
    /// rows it adds start empty.
    #[test]
    fn growth_preserves_the_rows_already_seated() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut c = cache(2, &device);
        let (k0, v0) = kv(3, 100.0, &device);
        let (k1, v1) = kv(2, 500.0, &device);
        c.update_lane(&k0, &v0, 0, 0, &client).expect("lane 0");
        c.update_lane(&k1, &v1, 1, 0, &client).expect("lane 1");

        let before_k = c.k_cache.to_vec::<f32>();
        let before_v = c.v_cache.to_vec::<f32>();
        let lens = c.lane_lens.clone();

        c.grow_batch(8).expect("grow");

        assert_eq!(c.batch_size(), 8);
        assert_eq!(c.k_cache.shape(), &[8, HEADS, CAPACITY, HEAD_DIM]);
        let after_k = c.k_cache.to_vec::<f32>();
        let after_v = c.v_cache.to_vec::<f32>();
        assert!(
            before_k.iter().any(|x| *x != 0.0),
            "a zeroed cache would make the comparison vacuous"
        );
        assert_eq!(
            after_k[..2 * ROW],
            before_k[..],
            "the old K rows move as-is"
        );
        assert_eq!(
            after_v[..2 * ROW],
            before_v[..],
            "the old V rows move as-is"
        );
        assert!(
            after_k[2 * ROW..].iter().all(|x| *x == 0.0),
            "the rows growth adds start zeroed"
        );
        assert_eq!(c.lane_lens[..2], lens[..], "the old lengths hold");
        assert_eq!(c.lane_lens[2..], [0; 6], "a new lane holds nothing");
        assert_eq!(c.seq_len(), 3, "the cursor is untouched");
    }

    /// A lane seated after growth holds exactly what the same sequence
    /// writes into a cache allocated wide from the start.
    #[test]
    fn a_lane_seated_after_growth_matches_a_cache_born_wide() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let (k, v) = kv(4, 3.0, &device);

        let mut grown = cache(1, &device);
        grown.grow_batch(4).expect("grow");
        grown.update_lane(&k, &v, 3, 0, &client).expect("seat");

        let mut born = cache(4, &device);
        born.update_lane(&k, &v, 3, 0, &client).expect("seat");

        assert_eq!(grown.k_cache.to_vec::<f32>(), born.k_cache.to_vec::<f32>());
        assert_eq!(grown.v_cache.to_vec::<f32>(), born.v_cache.to_vec::<f32>());
        assert_eq!(grown.lane_lens, born.lane_lens);
    }

    /// A request for the current width, or a narrower one, changes nothing.
    #[test]
    fn growth_to_a_width_that_is_not_larger_is_a_no_op() {
        let device = CpuDevice::new();
        let mut c = cache(4, &device);
        let address = c.k_cache.ptr();

        c.grow_batch(4).expect("same width");
        assert_eq!(c.batch_size(), 4);
        assert_eq!(c.k_cache.ptr(), address, "nothing is reallocated");

        c.grow_batch(1).expect("narrower");
        assert_eq!(c.batch_size(), 4, "the allocation never shrinks");
        assert_eq!(c.k_cache.ptr(), address);
    }

    /// Growing above a narrowed stepped width leaves the stepped width
    /// alone and rebuilds its view against the new buffer.
    #[test]
    fn growth_keeps_the_stepped_width_and_rebuilds_its_view() {
        let device = CpuDevice::new();
        let mut c = cache(2, &device);
        c.set_active_batch(1).expect("narrow");

        c.grow_batch(8).expect("grow");

        assert_eq!(c.active_batch(), 1, "growth does not step more rows");
        assert_eq!(c.k_cache_raw().shape()[0], 1);
        assert_eq!(c.lane_lens().len(), 1);
        assert_eq!(
            c.k_cache_raw().ptr(),
            c.k_cache.ptr(),
            "the narrowed view must sit on the buffer a step writes"
        );
        c.set_active_batch(8).expect("the new lanes are steppable");
        assert_eq!(c.k_cache_raw().shape()[0], 8);
    }

    /// The lane write reaches a lane growth added, and lands in that lane
    /// alone.
    #[test]
    fn a_lane_growth_added_is_writable() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut c = cache(1, &device);
        c.grow_batch(2).expect("grow");
        let (k, v) = kv(2, 9.0, &device);

        c.update_lane(&k, &v, 1, 0, &client).expect("grown lane");

        let all = c.k_cache.to_vec::<f32>();
        let want = k.to_vec::<f32>();
        assert_eq!(all[ROW..ROW + 2 * HEAD_DIM], want[..2 * HEAD_DIM]);
        assert!(all[..ROW].iter().all(|x| *x == 0.0), "lane 0 is untouched");
        assert_eq!(c.lane_lens, vec![0, 2]);
    }

    /// A stack grows every layer, so no decode call meets two widths.
    #[test]
    fn a_layered_stack_grows_every_layer() {
        let device = CpuDevice::new();
        let config = LayeredKvCacheConfig {
            batch_size: 1,
            num_kv_heads: HEADS,
            initial_capacity: CAPACITY,
            max_seq_len: 64,
            head_dim: HEAD_DIM,
            dtype: DType::F32,
        };
        let mut stack = LayeredKvCache::<CpuRuntime>::new(3, &config, &device).expect("stack");

        stack.grow_batch(4).expect("grow");

        for layer in 0..stack.num_layers() {
            let c = stack.layer(layer).expect("layer");
            assert_eq!(c.batch_size(), 4, "layer {layer}");
            assert_eq!(c.k_cache.shape()[0], 4, "layer {layer}");
        }
        stack
            .set_active_batch(4)
            .expect("every layer holds four lanes");
        assert_eq!(stack.active_batch(), 4);
    }
}
