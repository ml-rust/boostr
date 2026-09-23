//! Per-lane writes into a batched KV cache.
//!
//! A batched cache holds one sequence per lane. Prefill fills the lanes one
//! sequence at a time, each left-padded so every sequence's tokens end at
//! the shared cursor; decode then advances all lanes together. A lane's own
//! first position is what `kv_start` carries into attention, so the padding
//! is never read.

use crate::error::{Error, Result};
use crate::inference::kv_cache::{KvCache, LayeredKvCache};
use crate::ops::traits::KvCacheOps;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> KvCache<R>
where
    R::Client: IndexingOps<R>,
{
    /// Write one sequence's K and V into lane `lane`, starting at `offset`.
    ///
    /// Fills a batched cache one sequence at a time. Each lane is
    /// left-padded so every sequence's tokens end at the shared cursor;
    /// decode then advances all lanes together through
    /// [`KvCache::update_fused`], and a lane's own start is what
    /// `kv_start` carries into attention.
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

        let k_lane = self.k_cache.narrow(0, lane, 1)?;
        let v_lane = self.v_cache.narrow(0, lane, 1)?;
        client.kv_cache_update(&k_lane, &v_lane, k, v, offset)?;

        self.seq_len = self.seq_len.max(end);

        Ok(())
    }
}

impl<R: Runtime<DType = DType>> LayeredKvCache<R>
where
    R::Client: IndexingOps<R>,
{
    /// Copy a one-sequence cache into lane `lane` of every layer, ending at
    /// the shared cursor.
    ///
    /// The source holds `src.seq_len()` tokens from position 0; they land at
    /// `end - src.seq_len() .. end`, so the sequence's last token sits at
    /// the cursor every lane decodes from. The padding in front of it is
    /// what `kv_start` masks off.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when the two caches hold a different
    /// number of layers, the source holds more tokens than `end`, or a
    /// layer refuses the write.
    pub fn install_lane<C: KvCacheOps<R>>(
        &mut self,
        src: &LayeredKvCache<R>,
        lane: usize,
        end: usize,
        client: &C,
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
        if tokens > end {
            return Err(Error::InferenceError {
                reason: format!("a {tokens}-token sequence cannot end at {end}"),
            });
        }
        let offset = end - tokens;
        for idx in 0..self.num_layers() {
            let from = src.layer(idx).ok_or_else(|| Error::InferenceError {
                reason: format!("source kv cache missing layer {idx}"),
            })?;
            let k = from.k_cache_raw().narrow(2, 0, tokens)?.contiguous()?;
            let v = from.v_cache_raw().narrow(2, 0, tokens)?.contiguous()?;
            let cache = self.layer_mut(idx).ok_or_else(|| Error::InferenceError {
                reason: format!("kv cache missing layer {idx}"),
            })?;
            cache.update_lane(&k, &v, lane, offset, client)?;
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
