//! KvCache - Dynamic Key-Value cache for a single attention layer
//!
//! Stores K and V tensors with shape [batch, num_kv_heads, capacity, head_dim].
//! Grows dynamically as needed, avoiding massive upfront allocation.

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
    pub(super) k_cache: Tensor<R>,
    pub(super) v_cache: Tensor<R>,
    pub(super) seq_len: usize,
    /// One past each lane's last key. A lane-at-a-time cache holds one
    /// sequence per row, each from position 0 with its own end, so the rows
    /// disagree and `seq_len` — their maximum — describes none of them.
    pub(super) lane_lens: Vec<usize>,
    pub(super) capacity: usize,
    pub(super) max_seq_len: usize,
    /// Lanes ALLOCATED: the row count `k_cache` and `v_cache` were built at.
    /// Every lane index is an index into this width. It moves only through
    /// [`KvCache::grow_batch`], which adds rows above the ones already
    /// there, so a lane keeps its index for the life of the cache.
    pub(super) batch_size: usize,
    /// Lanes STEPPED: the prefix `0..active_batch` a decode call reads and
    /// writes. Defaults to `batch_size` and is lowered by
    /// [`KvCache::set_active_batch`] when the rows above it hold nothing, so
    /// a step pays for the rows it has rather than the rows it could hold.
    /// Never an index bound for seating a lane — that is `batch_size`.
    active_batch: usize,
    /// `k_cache` narrowed to `0..active_batch`, `None` while the two widths
    /// agree. A dim-0 prefix is a contiguous view over the same storage, so
    /// this shares the allocation's bytes and its base address.
    active_k: Option<Tensor<R>>,
    /// `v_cache` narrowed the same way.
    active_v: Option<Tensor<R>>,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) dtype: DType,
    pub(super) device: R::Device,
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

        let k_cache = Tensor::<R>::zeros(&shape, dtype, device)?;
        let v_cache = Tensor::<R>::zeros(&shape, dtype, device)?;

        Ok(Self {
            k_cache,
            v_cache,
            seq_len: 0,
            lane_lens: vec![0; batch_size],
            capacity,
            max_seq_len,
            batch_size,
            active_batch: batch_size,
            active_k: None,
            active_v: None,
            num_kv_heads,
            head_dim,
            dtype,
            device: device.clone(),
        })
    }

    /// Step the first `rows` lanes and leave the rest untouched.
    ///
    /// Every read a decode call makes — [`k_cache_raw`](Self::k_cache_raw),
    /// [`v_cache_raw`](Self::v_cache_raw), [`get_kv`](Self::get_kv),
    /// [`lane_lens`](Self::lane_lens) — then covers `0..rows`, and
    /// [`update_lanes`](Self::update_lanes) wants one row and one position
    /// per stepped lane rather than per allocated lane. Seating a lane is
    /// unaffected: [`update_lane`](Self::update_lane) still addresses the
    /// full allocated width, so a caller can install a sequence above the
    /// stepped prefix and widen it afterwards.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `rows` is 0 or past the allocated
    /// batch; a narrowing error propagates.
    pub fn set_active_batch(&mut self, rows: usize) -> Result<()> {
        if rows == 0 || rows > self.batch_size {
            return Err(Error::InferenceError {
                reason: format!(
                    "active batch {rows} is outside the {} lanes this cache was allocated \
                     with; it must be 1..={}",
                    self.batch_size, self.batch_size
                ),
            });
        }
        self.active_batch = rows;
        self.refresh_active_views()
    }

    /// Lanes a decode call steps — see [`set_active_batch`](Self::set_active_batch).
    /// [`batch_size`](Self::batch_size) is the allocated width instead.
    pub fn active_batch(&self) -> usize {
        self.active_batch
    }

    /// Rebuild the narrowed K/V views.
    ///
    /// Called whenever the active width moves AND whenever `k_cache` or
    /// `v_cache` is replaced by a new tensor, since a view of the old one
    /// would then describe storage nothing writes any more.
    pub(super) fn refresh_active_views(&mut self) -> Result<()> {
        if self.active_batch == self.batch_size {
            self.active_k = None;
            self.active_v = None;
            return Ok(());
        }
        self.active_k = Some(self.k_cache.narrow(0, 0, self.active_batch)?);
        self.active_v = Some(self.v_cache.narrow(0, 0, self.active_batch)?);
        Ok(())
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
        let mut new_k_cache = Tensor::<R>::zeros(&new_shape, self.dtype, &self.device)?;
        let mut new_v_cache = Tensor::<R>::zeros(&new_shape, self.dtype, &self.device)?;

        if self.seq_len > 0 {
            let old_k = self.k_cache.narrow(2, 0, self.seq_len)?;
            let old_v = self.v_cache.narrow(2, 0, self.seq_len)?;

            new_k_cache = new_k_cache.slice_assign(&old_k, 2, 0)?;
            new_v_cache = new_v_cache.slice_assign(&old_v, 2, 0)?;
        }

        self.k_cache = new_k_cache;
        self.v_cache = new_v_cache;
        self.capacity = new_capacity;
        self.refresh_active_views()?;

        Ok(())
    }

    pub(super) fn ensure_capacity(&mut self, required: usize) -> Result<()> {
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
        // `slice_assign` is functional: both buffers are new tensors now, so
        // any view of the old ones describes storage nothing writes.
        self.refresh_active_views()?;

        self.seq_len += new_tokens;
        // Every row was written the same run of slots, so every lane now
        // ends where the cursor does.
        self.lane_lens.fill(self.seq_len);

        Ok(())
    }

    /// Update using a fused CUDA/CPU kernel (single launch instead of slice_assign).
    ///
    /// Requires `R::Client: KvCacheOps<R>`. Falls back to `update` if the
    /// runtime client doesn't support fused KV cache operations.
    pub fn update_fused<C: KvCacheOps<R>>(
        &mut self,
        k: &Tensor<R>,
        v: &Tensor<R>,
        client: &C,
    ) -> Result<()> {
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
        self.lane_lens.fill(self.seq_len);

        Ok(())
    }

    /// Get the current K and V tensors (narrow views of the used portion)
    ///
    /// Covers the stepped lanes and the written positions:
    /// `[active_batch, H_kv, seq_len, D]`.
    pub fn get_kv(&self) -> Result<(Tensor<R>, Tensor<R>)> {
        if self.seq_len == 0 {
            return Err(Error::InferenceError {
                reason: "cache is empty".into(),
            });
        }

        let k = self.k_cache_raw().narrow(2, 0, self.seq_len)?;
        let v = self.v_cache_raw().narrow(2, 0, self.seq_len)?;

        Ok((k, v))
    }

    /// K cache tensor `[active_batch, H_kv, capacity, D]` — device address
    /// stable until the cache is grown.
    ///
    /// Unlike `get_kv()`, this keeps the whole capacity on the key axis
    /// without narrowing or reallocation, so it is safe as a CUDA graph
    /// argument. Both growth paths — growing the key axis to hold more
    /// tokens, and [`grow_batch`](Self::grow_batch) growing the lane axis —
    /// replace this buffer, so a caller that has captured a graph against
    /// it must not grow the cache afterwards.
    ///
    /// The lane axis is the STEPPED prefix: a dim-0 prefix of a dense
    /// tensor is itself dense and starts at the same address, so a kernel
    /// reading it reads the same bytes for the lanes it covers.
    pub fn k_cache_raw(&self) -> &Tensor<R> {
        self.active_k.as_ref().unwrap_or(&self.k_cache)
    }

    /// V cache tensor `[active_batch, H_kv, capacity, D]` — device address
    /// stable until the cache is grown, exactly as
    /// [`k_cache_raw`](Self::k_cache_raw).
    pub fn v_cache_raw(&self) -> &Tensor<R> {
        self.active_v.as_ref().unwrap_or(&self.v_cache)
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// One past each lane's last key, lane by lane.
    ///
    /// [`seq_len`](Self::seq_len) is the maximum of these and moves only up,
    /// so it is the cursor a FUSED write appends at and says nothing about a
    /// cache filled one lane at a time. A caller checking where a lane's next
    /// token belongs reads this instead.
    ///
    /// Covers the STEPPED lanes, so it lines up entry for entry with the
    /// write positions a decode call carries.
    pub fn lane_lens(&self) -> &[usize] {
        &self.lane_lens[..self.active_batch]
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

    /// Increment seq_len without modifying cache contents.
    ///
    /// Used in graph mode where the actual K/V write is done by the `kv_insert`
    /// kernel (graph-captured). The CPU-side seq_len tracks how many tokens have
    /// been inserted so DeviceScalars can be updated correctly before each replay.
    pub fn increment_seq_len(&mut self) {
        debug_assert!(
            self.seq_len < self.max_seq_len,
            "KvCache overflow on increment"
        );
        self.seq_len += 1;
        self.lane_lens.fill(self.seq_len);
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
        self.lane_lens.fill(0);
    }

    /// Lanes ALLOCATED. [`active_batch`](Self::active_batch) is the width a
    /// decode call steps, which is this or less, and
    /// [`grow_batch`](Self::grow_batch) is the one thing that raises it.
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
        let k = Tensor::<CpuRuntime>::zeros(&[1, 2, 3, 4], DType::F32, &device).unwrap();
        let v = Tensor::<CpuRuntime>::zeros(&[1, 2, 3, 4], DType::F32, &device).unwrap();

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
        let k = Tensor::<CpuRuntime>::zeros(&[1, 2, 5, 4], DType::F32, &device).unwrap();
        let v = Tensor::<CpuRuntime>::zeros(&[1, 2, 5, 4], DType::F32, &device).unwrap();
        cache.update(&k, &v).unwrap();

        assert_eq!(cache.seq_len(), 5);
        assert!(cache.capacity() >= 5);
    }

    #[test]
    fn test_kv_cache_overflow() {
        let device = numr::runtime::cpu::CpuDevice::new();
        let mut cache = KvCache::<CpuRuntime>::new(1, 2, 4, 8, 4, DType::F32, &device).unwrap();

        let k = Tensor::<CpuRuntime>::zeros(&[1, 2, 10, 4], DType::F32, &device).unwrap();
        let v = Tensor::<CpuRuntime>::zeros(&[1, 2, 10, 4], DType::F32, &device).unwrap();
        let result = cache.update(&k, &v);
        assert!(result.is_err());
    }

    #[test]
    fn test_kv_cache_reset() {
        let device = numr::runtime::cpu::CpuDevice::new();
        let mut cache = KvCache::<CpuRuntime>::new(1, 2, 64, 2048, 4, DType::F32, &device).unwrap();

        let k = Tensor::<CpuRuntime>::zeros(&[1, 2, 3, 4], DType::F32, &device).unwrap();
        let v = Tensor::<CpuRuntime>::zeros(&[1, 2, 3, 4], DType::F32, &device).unwrap();
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.seq_len(), 3);

        cache.reset();
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_get_kv_empty() {
        let device = numr::runtime::cpu::CpuDevice::new();
        let cache = KvCache::<CpuRuntime>::new(1, 2, 64, 2048, 4, DType::F32, &device).unwrap();
        assert!(cache.get_kv().is_err());
    }

    /// Narrowing the stepped width narrows every per-step read and leaves
    /// the allocation alone, and the views survive a growth that replaces
    /// both buffers.
    #[test]
    fn active_batch_narrows_the_stepped_reads() {
        let device = numr::runtime::cpu::CpuDevice::new();
        let mut cache = KvCache::<CpuRuntime>::new(4, 2, 4, 64, 4, DType::F32, &device).unwrap();
        assert_eq!(cache.active_batch(), 4, "defaults to the allocated width");
        assert_eq!(cache.k_cache_raw().shape()[0], 4);

        cache.set_active_batch(2).expect("narrow");
        assert_eq!(cache.batch_size(), 4, "the allocation never moves");
        assert_eq!(cache.active_batch(), 2);
        assert_eq!(cache.k_cache_raw().shape()[0], 2);
        assert_eq!(cache.v_cache_raw().shape()[0], 2);
        assert_eq!(cache.lane_lens().len(), 2);

        // A write past the initial capacity replaces both buffers.
        let k = Tensor::<CpuRuntime>::zeros(&[4, 2, 5, 4], DType::F32, &device).unwrap();
        let v = Tensor::<CpuRuntime>::zeros(&[4, 2, 5, 4], DType::F32, &device).unwrap();
        cache.update(&k, &v).unwrap();
        assert!(cache.capacity() >= 5);
        assert_eq!(cache.k_cache_raw().shape()[0], 2);
        assert_eq!(cache.k_cache_raw().shape()[2], cache.capacity());
        assert_eq!(cache.k_cache_raw().ptr(), cache.k_cache.ptr());

        let (k_used, _) = cache.get_kv().unwrap();
        assert_eq!(k_used.shape(), &[2, 2, 5, 4]);

        cache.set_active_batch(4).expect("widen");
        assert_eq!(cache.k_cache_raw().shape()[0], 4);
        assert_eq!(cache.lane_lens().len(), 4);
    }

    #[test]
    fn active_batch_outside_the_allocation_is_refused() {
        let device = numr::runtime::cpu::CpuDevice::new();
        let mut cache = KvCache::<CpuRuntime>::new(2, 2, 4, 64, 4, DType::F32, &device).unwrap();
        assert!(cache.set_active_batch(0).is_err());
        assert!(cache.set_active_batch(3).is_err());
        assert_eq!(cache.active_batch(), 2, "a refusal changes nothing");
    }

    #[test]
    fn test_update_fused() {
        let (client, device) = crate::test_utils::cpu_setup();
        let mut cache = KvCache::<CpuRuntime>::new(1, 2, 64, 2048, 4, DType::F32, &device).unwrap();

        let k_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let v_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.2).collect();
        let k = Tensor::<CpuRuntime>::from_slice(&k_data, &[1, 2, 3, 4], &device).unwrap();
        let v = Tensor::<CpuRuntime>::from_slice(&v_data, &[1, 2, 3, 4], &device).unwrap();

        cache.update_fused(&k, &v, &client).unwrap();
        assert_eq!(cache.seq_len(), 3);

        let (got_k, got_v) = cache.get_kv().unwrap();
        assert_eq!(got_k.shape(), &[1, 2, 3, 4]);
        assert_eq!(got_v.shape(), &[1, 2, 3, 4]);
    }
}
