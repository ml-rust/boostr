//! Rolling a contiguous KV cache back to a shorter length.
//!
//! Speculative decoding runs one verify forward over several draft tokens.
//! The cache then holds keys for drafts the target rejected. Truncation cuts
//! the logical length back to the accepted prefix. The next write lands at
//! the new length and overwrites the rejected slots.
//!
//! # Stale-slot invariant
//!
//! Truncation never zeroes memory. Slots at and past the logical length keep
//! their old bytes, and no read reaches them:
//!
//! - [`KvCache::get_kv`] narrows the key axis to `0..seq_len`.
//! - Attention over [`KvCache::k_cache_raw`] passes `KvSeqLen::Shared(seq_len)`
//!   or narrows to `seq_len` itself, so keys past it are outside every softmax.
//! - A lane-at-a-time read passes each lane's end from [`KvCache::lane_lens`].
//! - Growth on the key axis copies `0..seq_len` only.
//! - [`LayeredKvCache::install_lane`] copies `0..seq_len` of its source.
//! - The CUDA graph decode path reads its length from device scalars. The
//!   caller must write the new length there after a truncation.
//!
//! A write at the logical length replaces the stale slots before any read
//! covers them. Capacity and every device address are unchanged.

use crate::error::{Error, Result};
use crate::inference::kv_cache::basic::KvCache;
use crate::inference::kv_cache::layered::LayeredKvCache;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> KvCache<R>
where
    R::Client: IndexingOps<R>,
{
    /// Cut the logical length back to `len` positions.
    ///
    /// The next [`update`](Self::update) or
    /// [`update_fused`](Self::update_fused) writes at position `len`. Every
    /// allocated lane ends at `len` or earlier. A lane longer than `len` is
    /// cut to `len`, and a shorter lane keeps its own end. Memory is never
    /// zeroed, see the module's stale-slot invariant. Capacity and device
    /// addresses are unchanged. `len == seq_len()` changes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `len` is past [`seq_len`](Self::seq_len).
    /// The cache is unchanged then.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.seq_len {
            return Err(Error::InferenceError {
                reason: format!(
                    "cannot truncate the KV cache to {len} positions: it holds {}; pass a \
                     length of at most {}",
                    self.seq_len, self.seq_len
                ),
            });
        }
        self.seq_len = len;
        for end in &mut self.lane_lens {
            *end = (*end).min(len);
        }
        Ok(())
    }

    /// Cut lane `lane` back to `len` positions and leave every other lane
    /// alone.
    ///
    /// The lane's next write belongs at position `len`, through
    /// [`update_lane`](Self::update_lane) or
    /// [`update_lanes`](Self::update_lanes) with that position.
    /// [`seq_len`](Self::seq_len) stays where it is. It is the fused append
    /// slot, at or past every lane's end, and other lanes still reach it.
    /// `lane` indexes the allocated width. Memory is never zeroed.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `lane` is outside the allocated batch
    /// or `len` is past the lane's own end. The cache is unchanged then.
    pub fn truncate_lane(&mut self, lane: usize, len: usize) -> Result<()> {
        self.check_lane_truncate(lane, len)
            .map_err(|reason| Error::InferenceError { reason })?;
        if let Some(end) = self.lane_lens.get_mut(lane) {
            *end = len;
        }
        Ok(())
    }

    /// `lane` is inside the allocated batch and holds at least `len`
    /// positions. The error is the refusal's reason text.
    fn check_lane_truncate(&self, lane: usize, len: usize) -> std::result::Result<(), String> {
        let held = *self.lane_lens.get(lane).ok_or_else(|| {
            format!(
                "cannot truncate lane {lane}: the KV cache allocates {} lanes; pass a lane \
                 below {}",
                self.batch_size, self.batch_size
            )
        })?;
        if len > held {
            return Err(format!(
                "cannot truncate lane {lane} to {len} positions: it holds {held}; pass a \
                 length of at most {held}"
            ));
        }
        Ok(())
    }
}

impl<R: Runtime<DType = DType>> LayeredKvCache<R>
where
    R::Client: IndexingOps<R>,
{
    /// Cut every layer back to `len` positions, see [`KvCache::truncate`].
    ///
    /// Every layer is checked before any layer changes. The layers stay in
    /// sync, and a refused call changes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] naming the first layer that holds fewer
    /// than `len` positions.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        for (idx, layer) in self.layers.iter().enumerate() {
            if len > layer.seq_len() {
                return Err(Error::InferenceError {
                    reason: format!(
                        "cannot truncate the layered KV cache to {len} positions: layer {idx} \
                         holds {}; pass a length of at most {}",
                        layer.seq_len(),
                        layer.seq_len()
                    ),
                });
            }
        }
        for layer in &mut self.layers {
            layer.truncate(len)?;
        }
        Ok(())
    }

    /// Cut lane `lane` of every layer back to `len` positions, see
    /// [`KvCache::truncate_lane`].
    ///
    /// Every layer is checked before any layer changes, so a refused call
    /// changes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `lane` is outside a layer's batch or
    /// `len` is past that layer's lane end. The message names the layer.
    pub fn truncate_lane(&mut self, lane: usize, len: usize) -> Result<()> {
        for (idx, layer) in self.layers.iter().enumerate() {
            layer
                .check_lane_truncate(lane, len)
                .map_err(|reason| Error::InferenceError {
                    reason: format!("layer {idx}: {reason}"),
                })?;
        }
        for layer in &mut self.layers {
            layer.truncate_lane(lane, len)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::kv_cache::LayeredKvCacheConfig;
    use crate::ops::traits::{AttnOutLayout, FlashAttentionOps, KvSeqLen};
    use numr::ops::TypeConversionOps;
    use numr::runtime::RuntimeClient;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    const HEADS: usize = 2;
    const HEAD_DIM: usize = 4;

    fn cache(
        batch: usize,
        capacity: usize,
        dtype: DType,
        device: &CpuDevice,
    ) -> KvCache<CpuRuntime> {
        KvCache::<CpuRuntime>::new(batch, HEADS, capacity, 64, HEAD_DIM, dtype, device)
            .expect("cache")
    }

    /// K and V for `rows` lanes and `tokens` positions at `dtype`, every
    /// element distinct per `seed` and exact in half precision.
    fn kv(
        rows: usize,
        tokens: usize,
        seed: f32,
        dtype: DType,
        client: &CpuClient,
    ) -> (Tensor<CpuRuntime>, Tensor<CpuRuntime>) {
        let len = rows * HEADS * tokens * HEAD_DIM;
        let k: Vec<f32> = (0..len).map(|i| seed + i as f32 * 0.5).collect();
        let v: Vec<f32> = (0..len).map(|i| seed - i as f32 * 0.25).collect();
        let shape = [rows, HEADS, tokens, HEAD_DIM];
        let device = client.device();
        let k = Tensor::<CpuRuntime>::from_slice(&k, &shape, device).expect("k");
        let v = Tensor::<CpuRuntime>::from_slice(&v, &shape, device).expect("v");
        if dtype == DType::F32 {
            return (k, v);
        }
        (
            client.cast(&k, dtype).expect("cast k"),
            client.cast(&v, dtype).expect("cast v"),
        )
    }

    /// Positions `start..start + len` of `t` on the key axis, contiguous.
    fn positions(t: &Tensor<CpuRuntime>, start: usize, len: usize) -> Tensor<CpuRuntime> {
        t.narrow(2, start, len)
            .expect("narrow")
            .contiguous()
            .expect("contiguous")
    }

    /// The live range `get_kv` returns, as F32 values.
    fn live(c: &KvCache<CpuRuntime>, client: &CpuClient) -> (Vec<f32>, Vec<f32>) {
        let (k, v) = c.get_kv().expect("live range");
        let read = |t: Tensor<CpuRuntime>| {
            let t = t.contiguous().expect("contiguous");
            client.cast(&t, DType::F32).expect("cast").to_vec::<f32>()
        };
        (read(k), read(v))
    }

    /// Write mode of the rollback case: the slice-assign path or the fused
    /// kernel path.
    #[derive(Clone, Copy, Debug)]
    enum Write {
        Update,
        Fused,
    }

    fn write(
        c: &mut KvCache<CpuRuntime>,
        k: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        mode: Write,
        client: &CpuClient,
    ) {
        match mode {
            Write::Update => c.update(k, v).expect("update"),
            Write::Fused => c.update_fused(k, v, client).expect("update_fused"),
        }
    }

    /// Write 5, truncate to 3, write 2: the cache equals a fresh one that
    /// holds the first write's positions 0..3 then the second write.
    fn rollback_matches_fresh(dtype: DType, mode: Write, batch: usize, capacity: usize) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let (k_a, v_a) = kv(batch, 5, 100.0, dtype, &client);
        let (k_b, v_b) = kv(batch, 2, 900.0, dtype, &client);

        let mut rolled = cache(batch, capacity, dtype, &device);
        write(&mut rolled, &k_a, &v_a, mode, &client);
        rolled.truncate(3).expect("truncate");
        assert_eq!(rolled.seq_len(), 3);
        assert_eq!(rolled.lane_lens(), vec![3; batch].as_slice());
        write(&mut rolled, &k_b, &v_b, mode, &client);

        let mut fresh = cache(batch, capacity, dtype, &device);
        let (k_kept, v_kept) = (positions(&k_a, 0, 3), positions(&v_a, 0, 3));
        write(&mut fresh, &k_kept, &v_kept, mode, &client);
        write(&mut fresh, &k_b, &v_b, mode, &client);

        assert_eq!(rolled.seq_len(), 5);
        assert_eq!(rolled.lane_lens(), fresh.lane_lens());
        let (rk, rv) = live(&rolled, &client);
        let (fk, fv) = live(&fresh, &client);
        assert_eq!(rk, fk, "{dtype:?} {mode:?}: K after rollback");
        assert_eq!(rv, fv, "{dtype:?} {mode:?}: V after rollback");
        // The rolled-back slots held the first write; they must not survive.
        let (ak, _) = {
            let mut first = cache(batch, capacity, dtype, &device);
            write(&mut first, &k_a, &v_a, mode, &client);
            live(&first, &client)
        };
        assert_ne!(rk, ak, "the second write must replace positions 3..5");
    }

    #[test]
    fn rollback_matches_a_fresh_cache_f32() {
        for mode in [Write::Update, Write::Fused] {
            rollback_matches_fresh(DType::F32, mode, 1, 16);
            rollback_matches_fresh(DType::F32, mode, 3, 16);
        }
    }

    /// A capacity of 4 grows during the first write. The rollback stays
    /// exact across that growth.
    #[test]
    fn rollback_matches_a_fresh_cache_across_growth() {
        for mode in [Write::Update, Write::Fused] {
            rollback_matches_fresh(DType::F32, mode, 2, 4);
        }
    }

    #[cfg(feature = "f16")]
    #[test]
    fn rollback_matches_a_fresh_cache_half_precision() {
        for dtype in [DType::F16, DType::BF16] {
            for mode in [Write::Update, Write::Fused] {
                rollback_matches_fresh(dtype, mode, 1, 16);
                rollback_matches_fresh(dtype, mode, 2, 4);
            }
        }
    }

    /// Past the length is refused with both numbers named, and nothing
    /// changes. The current length is a no-op.
    #[test]
    fn truncate_past_the_length_is_refused() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut c = cache(1, 16, DType::F32, &device);
        let (k, v) = kv(1, 4, 1.0, DType::F32, &client);
        c.update(&k, &v).expect("update");
        let before = live(&c, &client);

        let err = c.truncate(5).expect_err("5 is past 4");
        let text = err.to_string();
        assert!(text.contains('5') && text.contains('4'), "{text}");
        assert_eq!(c.seq_len(), 4);
        assert_eq!(c.lane_lens(), &[4]);
        assert_eq!(live(&c, &client), before);

        c.truncate(4).expect("the current length");
        assert_eq!(c.seq_len(), 4);
        assert_eq!(live(&c, &client), before);
    }

    /// Truncating to 0 empties the cache, and a later write starts at 0.
    #[test]
    fn truncate_to_zero_then_reuse() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut c = cache(1, 16, DType::F32, &device);
        let (k_a, v_a) = kv(1, 6, 1.0, DType::F32, &client);
        let (k_b, v_b) = kv(1, 3, 50.0, DType::F32, &client);
        c.update(&k_a, &v_a).expect("first");
        let capacity = c.capacity();

        c.truncate(0).expect("truncate");
        assert!(c.is_empty());
        assert!(c.get_kv().is_err(), "an empty cache has no live range");
        assert_eq!(c.capacity(), capacity, "capacity is unchanged");
        c.update(&k_b, &v_b).expect("reuse");

        let mut fresh = cache(1, 16, DType::F32, &device);
        fresh.update(&k_b, &v_b).expect("fresh");
        assert_eq!(live(&c, &client), live(&fresh, &client));
    }

    /// Truncation keeps every device address and the capacity.
    #[test]
    fn truncate_keeps_addresses() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut c = cache(1, 16, DType::F32, &device);
        let (k, v) = kv(1, 5, 1.0, DType::F32, &client);
        c.update_fused(&k, &v, &client).expect("write");
        let (kp, vp) = (c.k_cache_raw().ptr(), c.v_cache_raw().ptr());

        c.truncate(2).expect("truncate");
        assert_eq!(c.k_cache_raw().ptr(), kp);
        assert_eq!(c.v_cache_raw().ptr(), vp);
        assert_eq!(c.capacity(), 16);
    }

    /// Stale slots past the length are never read. Attention over the raw
    /// buffer with the cache's own length matches attention over a cache
    /// that never held them, and `get_kv` stops at the length.
    #[test]
    fn stale_slots_past_the_length_are_never_read() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let (k_a, v_a) = kv(1, 6, 3.0, DType::F32, &client);

        let mut rolled = cache(1, 16, DType::F32, &device);
        rolled.update_fused(&k_a, &v_a, &client).expect("write");
        rolled.truncate(3).expect("truncate");

        let mut fresh = cache(1, 16, DType::F32, &device);
        let (k_kept, v_kept) = (positions(&k_a, 0, 3), positions(&v_a, 0, 3));
        fresh
            .update_fused(&k_kept, &v_kept, &client)
            .expect("write");

        let (k_live, _) = rolled.get_kv().expect("live");
        assert_eq!(k_live.shape(), &[1, HEADS, 3, HEAD_DIM]);

        let q_data: Vec<f32> = (0..HEADS * 2 * HEAD_DIM)
            .map(|i| (i as f32 * 0.37).sin() * 0.1)
            .collect();
        let q = Tensor::<CpuRuntime>::from_slice(&q_data, &[1, HEADS, 2, HEAD_DIM], &device)
            .expect("q");
        let attend = |c: &KvCache<CpuRuntime>, causal: bool, q: &Tensor<CpuRuntime>| {
            let (out, _) = client
                .flash_attention_fwd(
                    q,
                    c.k_cache_raw(),
                    c.v_cache_raw(),
                    HEADS,
                    HEADS,
                    HEAD_DIM,
                    causal,
                    0,
                    Some(KvSeqLen::Shared(c.seq_len())),
                    None,
                    AttnOutLayout::HeadMajor,
                )
                .expect("attention");
            out.to_vec::<f32>()
        };
        let q_one = positions(&q, 0, 1);
        assert_eq!(
            attend(&rolled, false, &q_one),
            attend(&fresh, false, &q_one)
        );
        assert_eq!(attend(&rolled, true, &q), attend(&fresh, true, &q));
    }

    /// `truncate` caps every lane at the length and keeps a shorter lane's
    /// own end.
    #[test]
    fn truncate_caps_every_lane() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut c = cache(2, 16, DType::F32, &device);
        let (k5, v5) = kv(1, 5, 1.0, DType::F32, &client);
        let (k2, v2) = kv(1, 2, 9.0, DType::F32, &client);
        c.update_lane(&k5, &v5, 0, 0, &client).expect("lane 0");
        c.update_lane(&k2, &v2, 1, 0, &client).expect("lane 1");
        assert_eq!(c.lane_lens(), &[5, 2]);

        c.truncate(3).expect("truncate");
        assert_eq!(c.seq_len(), 3);
        assert_eq!(c.lane_lens(), &[3, 2]);
    }

    /// `truncate_lane` cuts one lane, leaves the rest and the cursor, and a
    /// rewrite of that lane matches a lane that never held the rejected
    /// positions.
    #[test]
    fn truncate_lane_cuts_one_lane() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let (k5, v5) = kv(1, 5, 1.0, DType::F32, &client);
        let (k4, v4) = kv(1, 4, 40.0, DType::F32, &client);
        let (k_new, v_new) = kv(1, 3, 700.0, DType::F32, &client);

        let mut c = cache(2, 16, DType::F32, &device);
        c.update_lane(&k5, &v5, 0, 0, &client).expect("lane 0");
        c.update_lane(&k4, &v4, 1, 0, &client).expect("lane 1");

        c.truncate_lane(0, 2).expect("truncate lane 0");
        assert_eq!(c.lane_lens(), &[2, 4]);
        assert_eq!(
            c.seq_len(),
            5,
            "the cursor is the furthest end across lanes"
        );
        c.update_lane(&k_new, &v_new, 0, 2, &client)
            .expect("rewrite lane 0");
        assert_eq!(c.lane_lens(), &[5, 4]);

        let mut fresh = cache(2, 16, DType::F32, &device);
        let (k_kept, v_kept) = (positions(&k5, 0, 2), positions(&v5, 0, 2));
        fresh
            .update_lane(&k_kept, &v_kept, 0, 0, &client)
            .expect("kept");
        fresh
            .update_lane(&k_new, &v_new, 0, 2, &client)
            .expect("new");
        let lane0 = |c: &KvCache<CpuRuntime>| {
            c.k_cache_raw()
                .narrow(0, 0, 1)
                .and_then(|t| t.narrow(2, 0, 5))
                .expect("lane 0")
                .contiguous()
                .expect("contiguous")
                .to_vec::<f32>()
        };
        assert_eq!(lane0(&c), lane0(&fresh));
    }

    #[test]
    fn truncate_lane_refuses_bad_arguments() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut c = cache(2, 16, DType::F32, &device);
        let (k, v) = kv(1, 4, 1.0, DType::F32, &client);
        c.update_lane(&k, &v, 1, 0, &client).expect("lane 1");

        assert!(c.truncate_lane(2, 0).is_err(), "lane outside the batch");
        let err = c.truncate_lane(1, 5).expect_err("5 is past 4");
        assert!(err.to_string().contains("lane 1"), "{err}");
        assert!(c.truncate_lane(0, 1).is_err(), "lane 0 holds nothing");
        assert_eq!(c.lane_lens(), &[0, 4], "a refusal changes nothing");
        c.truncate_lane(1, 4).expect("the current end is a no-op");
        assert_eq!(c.lane_lens(), &[0, 4]);
    }

    fn stack(layers: usize, device: &CpuDevice) -> LayeredKvCache<CpuRuntime> {
        let config = LayeredKvCacheConfig {
            batch_size: 1,
            num_kv_heads: HEADS,
            initial_capacity: 8,
            max_seq_len: 64,
            head_dim: HEAD_DIM,
            dtype: DType::F32,
        };
        LayeredKvCache::<CpuRuntime>::new(layers, &config, device).expect("stack")
    }

    /// Every layer moves together, and a later write lands in sync.
    #[test]
    fn layered_truncate_keeps_layers_in_sync() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut s = stack(3, &device);
        let (k, v) = kv(1, 6, 1.0, DType::F32, &client);
        let (k2, v2) = kv(1, 2, 80.0, DType::F32, &client);
        for idx in 0..s.num_layers() {
            s.layer_mut(idx)
                .expect("layer")
                .update_fused(&k, &v, &client)
                .expect("write");
        }

        s.truncate(4).expect("truncate");
        for idx in 0..s.num_layers() {
            let layer = s.layer_mut(idx).expect("layer");
            assert_eq!(layer.seq_len(), 4, "layer {idx}");
            assert_eq!(layer.lane_lens(), &[4], "layer {idx}");
            layer.update_fused(&k2, &v2, &client).expect("write");
        }
        assert_eq!(s.seq_len(), 6);
        let first = live(s.layer(0).expect("layer"), &client);
        for idx in 1..s.num_layers() {
            assert_eq!(live(s.layer(idx).expect("layer"), &client), first);
        }
    }

    /// A length past any one layer is refused before any layer changes.
    #[test]
    fn layered_truncate_refusal_changes_no_layer() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut s = stack(2, &device);
        let (k5, v5) = kv(1, 5, 1.0, DType::F32, &client);
        let (k3, v3) = kv(1, 3, 1.0, DType::F32, &client);
        s.layer_mut(0).expect("layer").update(&k5, &v5).expect("w");
        s.layer_mut(1).expect("layer").update(&k3, &v3).expect("w");

        let err = s.truncate(4).expect_err("layer 1 holds 3");
        assert!(err.to_string().contains("layer 1"), "{err}");
        assert_eq!(s.layer(0).expect("layer").seq_len(), 5);
        assert_eq!(s.layer(1).expect("layer").seq_len(), 3);

        let err = s.truncate_lane(0, 4).expect_err("layer 1 lane 0 holds 3");
        assert!(err.to_string().contains("layer 1"), "{err}");
        assert_eq!(s.layer(0).expect("layer").lane_lens(), &[5]);

        s.truncate(0).expect("every layer holds at least 0");
        assert_eq!(s.seq_len(), 0);
        assert!(s.layer(1).expect("layer").is_empty());
    }
}
