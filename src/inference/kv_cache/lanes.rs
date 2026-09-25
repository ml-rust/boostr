//! Per-lane writes into a batched KV cache.
//!
//! A batched cache holds one sequence per lane, each starting at cache
//! position 0 with its own end. Prefill fills the lanes one sequence at a
//! time; decode then writes each lane's token at that lane's own next
//! position. A lane's own range is what `kv_start` and `kv_end` carry into
//! attention, so nothing outside it is ever read.

use crate::error::{Error, Result};
use crate::inference::kv_cache::{KvCache, LayeredKvCache};
use crate::ops::traits::KvCacheOps;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One layer's lane write: [`KvCache::update_lane`] on the growing path,
/// [`KvCache::update_lane_within_capacity`] on the frozen one. Both take a
/// narrowed K and V, the lane, the offset inside it and the client.
type LaneWrite<R, C> = fn(&mut KvCache<R>, &Tensor<R>, &Tensor<R>, usize, usize, &C) -> Result<()>;

impl<R: Runtime<DType = DType>> KvCache<R>
where
    R::Client: IndexingOps<R>,
{
    /// Write one sequence's K and V into lane `lane`, starting at `offset`.
    ///
    /// Fills a batched cache one sequence at a time.
    /// [`KvCache::update_lanes`] then appends a decode step to every lane,
    /// each at its own position.
    ///
    /// # Arguments
    /// * `k` - New K values [1, num_kv_heads, new_tokens, head_dim]
    /// * `v` - New V values [1, num_kv_heads, new_tokens, head_dim]
    /// * `lane` - Batch row to write
    /// * `offset` - First cache position this sequence occupies
    ///
    /// # Errors
    ///
    /// `lane` is outside the cache's batch, `k` carries more than one row,
    /// or the write would run past `max_seq_len`.
    pub fn update_lane<C: KvCacheOps<R>>(
        &mut self,
        k: &Tensor<R>,
        v: &Tensor<R>,
        lane: usize,
        offset: usize,
        client: &C,
    ) -> Result<()> {
        let new_tokens = self.check_lane_bounds(k, lane)?;
        let end = offset + new_tokens;
        if end > self.max_seq_len {
            return Err(Error::InferenceError {
                reason: format!(
                    "cache overflow: {offset} + {new_tokens} > max_seq_len {}",
                    self.max_seq_len
                ),
            });
        }
        self.ensure_capacity(end)?;
        self.write_lane(k, v, lane, offset, new_tokens, client)
    }

    /// Write one sequence's K and V into lane `lane` at `offset`, refusing
    /// rather than growing when the write would run past [`KvCache::capacity`].
    ///
    /// The frozen path over this cache calls this instead of
    /// [`KvCache::update_lane`]: capacity never moves here, so nothing this
    /// call does can invalidate a device pointer captured earlier.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `lane` is outside the batch, `k`
    /// carries more than one row, or the write would run past `capacity()`.
    pub(super) fn update_lane_within_capacity<C: KvCacheOps<R>>(
        &mut self,
        k: &Tensor<R>,
        v: &Tensor<R>,
        lane: usize,
        offset: usize,
        client: &C,
    ) -> Result<()> {
        let new_tokens = self.check_lane_bounds(k, lane)?;
        let end = offset + new_tokens;
        let capacity = self.capacity();
        if end > capacity {
            return Err(Error::InferenceError {
                reason: format!(
                    "lane {lane}: write end {end} ({offset} + {new_tokens}) is past the \
                     frozen cache's capacity {capacity}; this cache no longer grows, so open \
                     the session with enough capacity for every sequence it will admit"
                ),
            });
        }
        self.write_lane(k, v, lane, offset, new_tokens, client)
    }

    /// `lane` is inside the batch and `k` carries exactly one row; returns
    /// the token count `k` carries.
    fn check_lane_bounds(&self, k: &Tensor<R>, lane: usize) -> Result<usize> {
        let (rows, _, new_tokens, _) = k.dims4()?;
        if lane >= self.batch_size {
            return Err(Error::InferenceError {
                reason: format!("lane {lane} is outside batch {}", self.batch_size),
            });
        }
        if rows != 1 {
            return Err(Error::InferenceError {
                reason: format!("update_lane writes one sequence, got {rows} rows"),
            });
        }
        Ok(new_tokens)
    }

    /// The raw write behind both [`KvCache::update_lane`] and
    /// [`KvCache::update_lane_within_capacity`]: narrow to `lane`'s row,
    /// write `k`/`v` at `offset`, and record the new length.
    fn write_lane<C: KvCacheOps<R>>(
        &mut self,
        k: &Tensor<R>,
        v: &Tensor<R>,
        lane: usize,
        offset: usize,
        new_tokens: usize,
        client: &C,
    ) -> Result<()> {
        let k_lane = self.k_cache.narrow(0, lane, 1)?;
        let v_lane = self.v_cache.narrow(0, lane, 1)?;
        client.kv_cache_update(&k_lane, &v_lane, k, v, offset)?;
        // This lane's own end is ASSIGNED, not raised: a fresh sequence
        // installed at offset 0 replaces whatever the lane held, so a
        // previous occupant's longer run must not survive as this one's end.
        // The cursor still only rises — it is the fused path's append slot.
        self.lane_lens[lane] = offset + new_tokens;
        self.seq_len = self.seq_len.max(offset + new_tokens);
        Ok(())
    }

    /// Write one step's K and V into every lane, each at its own position.
    ///
    /// Lanes hold sequences of different lengths, so a shared write
    /// position would put a lane's token where another lane's tokens are.
    /// Each row of `k` and `v` goes through the same lane write path as
    /// [`KvCache::update_lane`], at the position `positions` gives it.
    ///
    /// # Arguments
    /// * `k` - New K values [batch, num_kv_heads, new_tokens, head_dim]
    /// * `v` - New V values [batch, num_kv_heads, new_tokens, head_dim]
    /// * `positions` - First cache position each lane's row lands on
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `k` carries a different number of
    /// rows than the cache has lanes, when `positions` carries a different
    /// number of entries, or when a lane's write runs past `max_seq_len`.
    pub fn update_lanes<C: KvCacheOps<R>>(
        &mut self,
        k: &Tensor<R>,
        v: &Tensor<R>,
        positions: &[usize],
        client: &C,
    ) -> Result<()> {
        let (rows, _, new_tokens, _) = k.dims4()?;
        if rows != self.batch_size || positions.len() != self.batch_size {
            return Err(Error::InferenceError {
                reason: format!(
                    "update_lanes wants one row and one position per lane: cache holds {} \
                     lanes, k holds {rows} rows, positions holds {}",
                    self.batch_size,
                    positions.len()
                ),
            });
        }
        if !k.is_contiguous() || !v.is_contiguous() {
            return Err(Error::InferenceError {
                reason: "update_lanes reads k and v row by row and needs both contiguous; \
                         call contiguous() on them first"
                    .into(),
            });
        }
        let furthest = positions.iter().copied().max().unwrap_or(0) + new_tokens;
        if furthest > self.max_seq_len {
            return Err(Error::InferenceError {
                reason: format!(
                    "cache overflow: a lane writes {new_tokens} tokens at {} > max_seq_len {}",
                    furthest - new_tokens,
                    self.max_seq_len
                ),
            });
        }
        self.ensure_capacity(furthest)?;

        for (lane, &offset) in positions.iter().enumerate() {
            let k_lane = k.narrow(0, lane, 1)?;
            let v_lane = v.narrow(0, lane, 1)?;
            self.update_lane(&k_lane, &v_lane, lane, offset, client)?;
        }
        Ok(())
    }
}

impl<R: Runtime<DType = DType>> LayeredKvCache<R>
where
    R::Client: IndexingOps<R>,
{
    /// Copy a one-sequence cache into lane `lane` of every layer, at
    /// position 0.
    ///
    /// The source holds `src.seq_len()` tokens from position 0 and they
    /// land at the same positions in the lane, so the lane's length is its
    /// own sequence's length and no other lane constrains it. Whatever a
    /// previous occupant left past that length sits outside the lane's
    /// `[kv_start, kv_end)` range and is never read.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when the two caches hold a different
    /// number of layers, or a layer refuses the write.
    pub fn install_lane<C: KvCacheOps<R>>(
        &mut self,
        src: &LayeredKvCache<R>,
        lane: usize,
        client: &C,
    ) -> Result<()> {
        self.install_lane_via(src, lane, client, KvCache::update_lane)
    }

    /// [`LayeredKvCache::install_lane`], but refusing rather than growing
    /// when a layer's write would run past its own `capacity()`.
    ///
    /// [`FrozenLayeredKvCache`](crate::inference::kv_cache::FrozenLayeredKvCache)
    /// calls this instead of [`LayeredKvCache::install_lane`], so nothing
    /// it does can move a layer's K or V tensor off the address a captured
    /// graph recorded.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when the two caches hold a different
    /// number of layers, or when a layer's write would run past its
    /// `capacity()`.
    pub(super) fn install_lane_within_capacity<C: KvCacheOps<R>>(
        &mut self,
        src: &LayeredKvCache<R>,
        lane: usize,
        client: &C,
    ) -> Result<()> {
        self.install_lane_via(src, lane, client, KvCache::update_lane_within_capacity)
    }

    /// The layer walk behind both [`LayeredKvCache::install_lane`] and
    /// [`LayeredKvCache::install_lane_within_capacity`]: check the layer
    /// counts match, then hand each layer's narrowed K/V to `write_lane`,
    /// which is [`KvCache::update_lane`] on the growing path and
    /// [`KvCache::update_lane_within_capacity`] on the frozen one.
    fn install_lane_via<C: KvCacheOps<R>>(
        &mut self,
        src: &LayeredKvCache<R>,
        lane: usize,
        client: &C,
        write_lane: LaneWrite<R, C>,
    ) -> Result<()> {
        if src.num_layers() != self.num_layers() {
            return Err(Error::InferenceError {
                reason: format!(
                    "kv cache holds {} layers, source holds {}",
                    self.num_layers(),
                    src.num_layers()
                ),
            });
        }
        let tokens = src.seq_len();
        for idx in 0..self.num_layers() {
            let from = src.layer(idx).ok_or_else(|| Error::InferenceError {
                reason: format!("source kv cache missing layer {idx}"),
            })?;
            let k = from.k_cache_raw().narrow(2, 0, tokens)?.contiguous()?;
            let v = from.v_cache_raw().narrow(2, 0, tokens)?.contiguous()?;
            let cache = self.layer_mut(idx).ok_or_else(|| Error::InferenceError {
                reason: format!("kv cache missing layer {idx}"),
            })?;
            write_lane(cache, &k, &v, lane, 0, client)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    const HEADS: usize = 2;
    const HEAD_DIM: usize = 4;

    fn cache(batch: usize, device: &CpuDevice) -> KvCache<CpuRuntime> {
        KvCache::<CpuRuntime>::new(batch, HEADS, 16, 64, HEAD_DIM, DType::F32, device)
            .expect("cache")
    }

    /// K and V for `tokens` positions, each element distinct per `seed`.
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

    /// A lane write lands in its own row at its own offset, and the cursor
    /// tracks the furthest end across lanes.
    #[test]
    fn test_lane_write_places_one_row() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut c = cache(2, &device);
        let (k0, v0) = kv(3, 100.0, &device);
        let (k1, v1) = kv(2, 500.0, &device);

        c.update_lane(&k0, &v0, 0, 1, &client).expect("lane 0");
        assert_eq!(c.seq_len(), 4);
        c.update_lane(&k1, &v1, 1, 2, &client).expect("lane 1");
        assert_eq!(c.seq_len(), 4);

        let k_all = c.k_cache_raw().to_vec::<f32>();
        let row = HEADS * c.capacity() * HEAD_DIM;
        let k0 = k0.to_vec::<f32>();
        let k1 = k1.to_vec::<f32>();
        // Lane 0 head 0 holds its three tokens from position 1.
        assert_eq!(k_all[HEAD_DIM..4 * HEAD_DIM], k0[..3 * HEAD_DIM]);
        // Lane 1 head 0 holds its two tokens from position 2.
        assert_eq!(
            k_all[row + 2 * HEAD_DIM..row + 4 * HEAD_DIM],
            k1[..2 * HEAD_DIM]
        );
        // Position 0 of lane 0 was never written.
        assert!(k_all[..HEAD_DIM].iter().all(|x| *x == 0.0));
    }

    /// A lane of a batched cache holds the same bytes the sequence writes
    /// into a cache of its own.
    #[test]
    fn test_lane_matches_its_own_cache() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut batched = cache(3, &device);
        let mut solo = cache(1, &device);
        let (k, v) = kv(5, 7.0, &device);

        batched.update_lane(&k, &v, 2, 3, &client).expect("lane 2");
        solo.update_lane(&k, &v, 0, 3, &client).expect("solo lane");

        let lane = batched
            .k_cache_raw()
            .narrow(0, 2, 1)
            .expect("lane view")
            .contiguous()
            .expect("contiguous");
        assert_eq!(lane.to_vec::<f32>(), solo.k_cache_raw().to_vec::<f32>());
    }

    /// Every lane's row lands at its own position, so lanes of different
    /// lengths each append where their own sequence ends.
    #[test]
    fn test_update_lanes_writes_each_lane_at_its_own_position() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut c = cache(2, &device);
        let (k, v) = kv(2, 1000.0, &device);
        let two_rows = Tensor::<CpuRuntime>::cat(&[&k, &k], 0).expect("rows");
        let two_v = Tensor::<CpuRuntime>::cat(&[&v, &v], 0).expect("rows");

        c.update_lanes(&two_rows, &two_v, &[0, 3], &client)
            .expect("lanes");

        let k_all = c.k_cache_raw().to_vec::<f32>();
        let row = HEADS * c.capacity() * HEAD_DIM;
        let k = k.to_vec::<f32>();
        assert_eq!(k_all[..2 * HEAD_DIM], k[..2 * HEAD_DIM]);
        assert_eq!(
            k_all[row + 3 * HEAD_DIM..row + 5 * HEAD_DIM],
            k[..2 * HEAD_DIM]
        );
        assert_eq!(c.seq_len(), 5);
    }

    /// One position per lane and one row per lane are both required.
    #[test]
    fn test_update_lanes_refuses_a_count_mismatch() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut c = cache(2, &device);
        let (k, v) = kv(1, 2.0, &device);
        let two_rows = Tensor::<CpuRuntime>::cat(&[&k, &k], 0).expect("rows");
        let two_v = Tensor::<CpuRuntime>::cat(&[&v, &v], 0).expect("rows");

        assert!(c.update_lanes(&two_rows, &two_v, &[0], &client).is_err());
        assert!(c.update_lanes(&k, &v, &[0, 1], &client).is_err());
        assert!(
            c.update_lanes(&two_rows, &two_v, &[0, 64], &client)
                .is_err()
        );
    }

    /// A lane outside the batch, more than one row, and a write past the
    /// cache's own limit are each refused.
    #[test]
    fn test_lane_write_refuses_bad_arguments() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut c = cache(2, &device);
        let (k, v) = kv(2, 1.0, &device);

        assert!(c.update_lane(&k, &v, 2, 0, &client).is_err());
        assert!(c.update_lane(&k, &v, 0, 63, &client).is_err());

        let two_rows = Tensor::<CpuRuntime>::zeros(&[2, HEADS, 2, HEAD_DIM], DType::F32, &device)
            .expect("two rows");
        assert!(c.update_lane(&two_rows, &two_rows, 0, 0, &client).is_err());
    }
}
