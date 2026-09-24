//! Taking a lane for a sequence, giving it back, and stepping every
//! occupant.

use crate::error::{Error, Result};
use crate::inference::gdn_state::LayeredGdnState;
use crate::inference::kv_cache::LayeredKvCache;
use crate::inference::lanes::sink::LaneKvSink;
use crate::inference::lanes::table::LaneTable;
use crate::inference::scheduler::SequenceId;
use crate::ops::traits::KvCacheOps;
use numr::dtype::DType;
use numr::ops::{BinaryOps, HostCopyOps, IndexingOps};
use numr::runtime::Runtime;

/// One prefilled sequence, ready to take a lane.
///
/// `kv` and `gdn` are what the sequence's own prefill produced, a single
/// sequence each. `rope_pos` is the position its first decoded token sits
/// at, which a text-only prefill leaves at the prompt length.
pub struct LaneAdmission<'a, R: Runtime> {
    /// The sequence taking the lane.
    pub seq: SequenceId,
    /// The sequence's own one-lane KV cache.
    pub kv: &'a LayeredKvCache<R>,
    /// The sequence's own one-lane GDN state.
    pub gdn: &'a LayeredGdnState<R>,
    /// The rope position of the sequence's next token.
    pub rope_pos: usize,
}

impl<R: Runtime<DType = DType>> LaneTable<R>
where
    R::Client: IndexingOps<R>,
{
    /// Install a prefilled sequence into a free lane and return that lane.
    ///
    /// The sequence lands at cache position 0 of its lane, so its length is
    /// its own and no other lane constrains it. `kv` and `gdn` are the
    /// batched containers the decode steps read, one lane per sequence.
    /// `kv` is anything implementing [`LaneKvSink`]: a growable
    /// [`LayeredKvCache`] grows to fit the sequence, a
    /// [`FrozenLayeredKvCache`](crate::inference::kv_cache::FrozenLayeredKvCache)
    /// refuses instead.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when every lane is taken, when
    /// `src.rope_pos` is outside the rope table the table was built with,
    /// or when the batched containers refuse the write.
    pub fn admit<C, S>(
        &mut self,
        client: &C,
        mut kv: S,
        gdn: &mut LayeredGdnState<R>,
        src: LaneAdmission<'_, R>,
    ) -> Result<usize>
    where
        C: KvCacheOps<R> + BinaryOps<R> + HostCopyOps<R>,
        S: LaneKvSink<R>,
    {
        let lane = self
            .occupants
            .iter()
            .position(Option::is_none)
            .ok_or_else(|| Error::InferenceError {
                reason: format!(
                    "every one of the {} lanes is taken; release a finished sequence's lane \
                     before admitting sequence {}",
                    self.batch(),
                    src.seq
                ),
            })?;

        self.check_rope_pos(lane, src.rope_pos)?;

        kv.install_lane(src.kv, lane, client)?;
        gdn.install_lane(client, lane, src.gdn)?;

        self.occupants[lane] = Some(src.seq);
        self.lens[lane] = src.kv.seq_len();
        self.rope_pos[lane] = src.rope_pos;
        self.flush(client)?;
        Ok(lane)
    }

    /// Clear lane `lane`'s GDN state and mark the lane free.
    ///
    /// The GDN clear is required: the delta rule carries a lane's state
    /// forward step by step with no mask, so a stale state would leak into
    /// the next occupant's first token.
    ///
    /// The KV cache needs no clear. The next occupant is installed from
    /// cache position 0 and reads only `[kv_start, kv_end)`, so the stale
    /// keys left behind sit past its end and no kernel ever reads them;
    /// the ones below its end are overwritten by the install itself.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `lane` is outside the batch; a clear
    /// error propagates.
    pub fn release<C: BinaryOps<R> + HostCopyOps<R>>(
        &mut self,
        client: &C,
        gdn: &mut LayeredGdnState<R>,
        lane: usize,
    ) -> Result<()> {
        if lane >= self.batch() {
            return Err(Error::InferenceError {
                reason: format!(
                    "lane {lane} is outside batch {}; release a lane the table handed out",
                    self.batch()
                ),
            });
        }
        gdn.reset_lane(client, lane)?;
        self.occupants[lane] = None;
        self.lens[lane] = 0;
        self.rope_pos[lane] = 0;
        self.flush(client)
    }

    /// Count the token every occupied lane just decoded.
    ///
    /// Each occupant's length and rope position go up by one, and both
    /// device tensors are rewritten in place. Every occupied lane is
    /// checked against the rope table first, so a refusal leaves the table
    /// exactly as it was.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when an occupied lane's next rope position
    /// is outside the rope table. A copy error propagates.
    pub fn advance<C: HostCopyOps<R>>(&mut self, client: &C) -> Result<()> {
        for lane in 0..self.batch() {
            if self.occupants[lane].is_some() {
                self.check_rope_pos(lane, self.rope_pos[lane] + 1)?;
            }
        }
        for lane in 0..self.batch() {
            if self.occupants[lane].is_some() {
                self.lens[lane] += 1;
                self.rope_pos[lane] += 1;
            }
        }
        self.flush(client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::kv_cache::layered::LayeredKvCacheConfig;
    use crate::model::config::GdnConfig;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    const LAYERS: usize = 2;
    const H_KV: usize = 1;
    const HD: usize = 4;
    const MAX: usize = 16;

    fn gdn_cfg() -> GdnConfig {
        GdnConfig {
            hidden_size: 8,
            conv_kernel: 4,
            state_size: 4,
            key_heads: 2,
            value_heads: 4,
            inner_size: 16,
            rms_eps: 1e-6,
            chunk_size: 64,
            v_grouped: false,
        }
    }

    fn containers(
        batch: usize,
        device: &CpuDevice,
    ) -> (LayeredKvCache<CpuRuntime>, LayeredGdnState<CpuRuntime>) {
        let cfg = LayeredKvCacheConfig {
            batch_size: batch,
            num_kv_heads: H_KV,
            initial_capacity: MAX,
            max_seq_len: MAX,
            head_dim: HD,
            dtype: DType::F32,
        };
        (
            LayeredKvCache::<CpuRuntime>::new(LAYERS, &cfg, device).expect("kv"),
            LayeredGdnState::<CpuRuntime>::zeros(LAYERS, &gdn_cfg(), batch, DType::F32, device)
                .expect("gdn"),
        )
    }

    /// A one-sequence cache holding `tokens` distinct keys, plus its state.
    fn prefilled(
        client: &CpuClient,
        device: &CpuDevice,
        tokens: usize,
        seed: f32,
    ) -> (LayeredKvCache<CpuRuntime>, LayeredGdnState<CpuRuntime>) {
        let (mut kv, mut gdn) = containers(1, device);
        let len = H_KV * tokens * HD;
        let data: Vec<f32> = (0..len).map(|i| seed + i as f32).collect();
        let shape = [1, H_KV, tokens, HD];
        let k = Tensor::<CpuRuntime>::from_slice(&data, &shape, device).expect("k");
        for idx in 0..LAYERS {
            kv.layer_mut(idx)
                .expect("layer")
                .update_fused(&k, &k, client)
                .expect("prefill");
        }
        let conv = Tensor::<CpuRuntime>::ones(&[1, 32, 3], DType::F32, device).expect("conv");
        let ssm = Tensor::<CpuRuntime>::ones(&[1, 4, 4, 4], DType::F32, device).expect("ssm");
        for idx in 0..LAYERS {
            gdn.layer_mut(idx)
                .expect("layer")
                .update(client, &conv, &ssm)
                .expect("state");
        }
        (kv, gdn)
    }

    /// One admission of `kv`/`gdn` at `rope_pos`.
    fn admission<'a>(
        seq: SequenceId,
        src: &'a (LayeredKvCache<CpuRuntime>, LayeredGdnState<CpuRuntime>),
        rope_pos: usize,
    ) -> LaneAdmission<'a, CpuRuntime> {
        LaneAdmission {
            seq,
            kv: &src.0,
            gdn: &src.1,
            rope_pos,
        }
    }

    /// Lanes are handed out lowest free first, and each one records its own
    /// length and rope position.
    #[test]
    fn admit_takes_the_lowest_free_lane() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let (mut kv, mut gdn) = containers(2, &device);
        let mut table = LaneTable::<CpuRuntime>::new(2, MAX, &device).expect("table");

        let first = prefilled(&client, &device, 3, 10.0);
        let second = prefilled(&client, &device, 5, 20.0);
        let a = table
            .admit(&client, &mut kv, &mut gdn, admission(7, &first, 3))
            .expect("admit a");
        let b = table
            .admit(&client, &mut kv, &mut gdn, admission(9, &second, 5))
            .expect("admit b");

        assert_eq!((a, b), (0, 1));
        assert_eq!(table.occupant(0), Some(7));
        assert_eq!(table.occupant(1), Some(9));
        assert_eq!(table.free_lanes(), 0);
        assert_eq!(table.write_positions(), &[3, 5]);
        assert_eq!(table.kv_end().to_vec::<i32>(), vec![4, 6]);
        assert_eq!(table.kv_start().to_vec::<i32>(), vec![0, 0]);
        assert_eq!(
            table.positions().to_vec::<i32>(),
            vec![3, 5, 3, 5, 3, 5, 0, 0]
        );
    }

    /// A full table refuses the next sequence, and a released lane is the
    /// one the next admit takes.
    #[test]
    fn a_full_table_refuses_and_a_released_lane_comes_back() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let (mut kv, mut gdn) = containers(1, &device);
        let mut table = LaneTable::<CpuRuntime>::new(1, MAX, &device).expect("table");
        let src = prefilled(&client, &device, 2, 1.0);

        assert_eq!(
            table
                .admit(&client, &mut kv, &mut gdn, admission(1, &src, 2))
                .expect("admit"),
            0
        );
        assert!(
            table
                .admit(&client, &mut kv, &mut gdn, admission(2, &src, 2))
                .is_err()
        );

        table.release(&client, &mut gdn, 0).expect("release");
        assert!(table.is_free(0));
        assert_eq!(table.kv_end().to_vec::<i32>(), vec![0]);
        assert!(
            gdn.layer(0)
                .expect("layer")
                .ssm()
                .to_vec::<f32>()
                .iter()
                .all(|x| *x == 0.0)
        );
        assert_eq!(
            table
                .admit(&client, &mut kv, &mut gdn, admission(3, &src, 2))
                .expect("re-admit"),
            0
        );
        assert!(table.release(&client, &mut gdn, 1).is_err());
    }

    /// Advancing bumps only the occupied lanes, and every device tensor
    /// keeps the address it was allocated at.
    #[test]
    fn advance_moves_occupied_lanes_and_keeps_addresses() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let (mut kv, mut gdn) = containers(2, &device);
        let mut table = LaneTable::<CpuRuntime>::new(2, MAX, &device).expect("table");
        let addresses = (
            table.kv_start().ptr(),
            table.kv_end().ptr(),
            table.positions().ptr(),
        );

        let src = prefilled(&client, &device, 4, 3.0);
        table
            .admit(&client, &mut kv, &mut gdn, admission(1, &src, 4))
            .expect("admit");
        table.advance(&client).expect("advance");
        table.advance(&client).expect("advance");

        assert_eq!(table.write_positions(), &[6, 0]);
        assert_eq!(table.lane_rope_pos(0), 6);
        assert_eq!(table.kv_end().to_vec::<i32>(), vec![7, 0]);
        table.release(&client, &mut gdn, 0).expect("release");
        assert_eq!(
            (
                table.kv_start().ptr(),
                table.kv_end().ptr(),
                table.positions().ptr()
            ),
            addresses
        );
    }

    /// The rope table bounds a lane: the last position it holds is
    /// admitted, the one past it is refused, and a lane sitting on the last
    /// one refuses to advance.
    #[test]
    fn the_rope_table_bounds_admit_and_advance() {
        const ROPE: usize = 3;
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let (mut kv, mut gdn) = containers(1, &device);
        let mut table = LaneTable::<CpuRuntime>::new(1, ROPE, &device).expect("table");
        let src = prefilled(&client, &device, 2, 1.0);

        assert!(
            table
                .admit(&client, &mut kv, &mut gdn, admission(1, &src, ROPE))
                .is_err()
        );
        assert!(table.is_free(0));

        table
            .admit(&client, &mut kv, &mut gdn, admission(1, &src, ROPE - 1))
            .expect("admit at the last position the table holds");
        assert_eq!(table.lane_rope_pos(0), ROPE - 1);
        assert_eq!(table.kv_end().to_vec::<i32>(), vec![3]);

        assert!(table.advance(&client).is_err());
        assert_eq!(table.lane_rope_pos(0), ROPE - 1);
        assert_eq!(table.write_positions(), &[2]);
    }
}
