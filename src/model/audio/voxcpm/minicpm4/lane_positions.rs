//! [`LanePositions`]: each row's write position across a batched decode, and
//! the device tensor the attention kernels read its key range from.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::minicpm4::cache_step::CacheStep;
use crate::model::audio::voxcpm::minicpm4::left_pad::LeftPad;
use numr::dtype::DType;
use numr::ops::HostCopyOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Where every row of a batched decode writes next, and where its keys end.
///
/// Row `b` holds its sequence from cache position `pad.starts[b]` to
/// `lens[b]`, so `lens[b]` is both the slot its next step writes and, after
/// subtracting its start, the rope position it rotates at. One table serves
/// both the `base_lm` and the `residual_lm` cache: the two are written at
/// the same position and read over the same range, so they advance together
/// through one [`advance`](Self::advance) after both have stepped.
///
/// `kv_end` is allocated once and rewritten in place through
/// [`HostCopyOps::write_host_slice`], so its device address holds for the
/// life of the table. The host `lens` mirror backs every write, so no step
/// reads a device tensor back.
///
/// `rope_table_len` bounds every row: a position the model's rope table does
/// not hold is refused here, not read out of bounds in a kernel. `None` for
/// a pair of stacks that rotate nothing.
pub struct LanePositions<R: Runtime> {
    pad: Option<LeftPad<R>>,
    lens: Vec<usize>,
    kv_end: Tensor<R>,
    /// Staging for `kv_end`, one entry per row.
    ends: Vec<i32>,
    rope_table_len: Option<usize>,
}

impl<R: Runtime<DType = DType>> LanePositions<R> {
    /// A table of `batch` rows all sitting at `position`, left-padded by
    /// `pad`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `batch` is 0, when `pad` describes a
    /// different number of rows, when `position` is outside
    /// `rope_table_len`, or when a row's end is past what an I32 holds; an
    /// allocation error propagates.
    pub fn new(
        pad: Option<LeftPad<R>>,
        batch: usize,
        position: usize,
        rope_table_len: Option<usize>,
        device: &R::Device,
    ) -> Result<Self> {
        if batch == 0 {
            return Err(Error::InvalidArgument {
                arg: "batch",
                reason: "expected at least 1 row, got 0".to_string(),
            });
        }
        if let Some(pad) = pad.as_ref()
            && pad.batch() != batch
        {
            return Err(Error::InvalidArgument {
                arg: "pad",
                reason: format!("expected {batch} rows, got {}", pad.batch()),
            });
        }
        let ends = vec![as_i32(position + 1, 0)?; batch];
        let kv_end = Tensor::<R>::from_slice(&ends, &[batch], device)?;
        let table = Self {
            pad,
            lens: vec![position; batch],
            kv_end,
            ends,
            rope_table_len,
        };
        for b in 0..batch {
            table.check_rope_pos(b, position)?;
        }
        Ok(table)
    }

    /// Rows in the batch.
    pub fn batch(&self) -> usize {
        self.lens.len()
    }

    /// Row `b`'s next write position.
    pub fn position(&self, b: usize) -> usize {
        self.lens.get(b).copied().unwrap_or(0)
    }

    /// Every row's next write position.
    pub fn positions(&self) -> &[usize] {
        &self.lens
    }

    /// The one position every row writes next, or `None` once they differ.
    pub fn shared_position(&self) -> Option<usize> {
        let first = *self.lens.first()?;
        self.lens.iter().all(|p| *p == first).then_some(first)
    }

    /// The batch's left padding, `None` when no row is padded.
    pub fn pad(&self) -> Option<&LeftPad<R>> {
        self.pad.as_ref()
    }

    /// `[B]` I32, one past each row's last key.
    pub fn kv_end(&self) -> &Tensor<R> {
        &self.kv_end
    }

    /// The view one decode step takes, borrowed from this table.
    pub fn step(&self) -> CacheStep<'_, R> {
        CacheStep::lanes(self.pad.as_ref(), &self.lens, &self.kv_end)
    }

    /// Count the position every row just decoded.
    ///
    /// Call this ONCE per step, after every stack that shares the table has
    /// stepped. Advancing between two of them would rotate and file the
    /// later one a position ahead of the earlier.
    ///
    /// Every row is checked against the rope table first, so a refusal
    /// leaves the table exactly as it was.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when a row's next position is outside the
    /// rope table or past what an I32 holds; a copy error propagates.
    pub fn advance<C: HostCopyOps<R>>(&mut self, client: &C) -> Result<()> {
        for b in 0..self.batch() {
            self.check_rope_pos(b, self.lens[b] + 1)?;
            as_i32(self.lens[b] + 2, b)?;
        }
        for len in &mut self.lens {
            *len += 1;
        }
        self.flush(client)
    }

    /// Write the host mirror into `kv_end`, in place.
    fn flush<C: HostCopyOps<R>>(&mut self, client: &C) -> Result<()> {
        self.ends.clear();
        for (b, len) in self.lens.iter().enumerate() {
            self.ends.push(as_i32(len + 1, b)?);
        }
        client
            .write_host_slice(&self.kv_end, &self.ends)
            .map_err(Error::Numr)
    }

    /// Refuse a rope position the model's rope table does not hold.
    fn check_rope_pos(&self, b: usize, pos: usize) -> Result<()> {
        let Some(limit) = self.rope_table_len else {
            return Ok(());
        };
        let rope_pos = pos.saturating_sub(self.pad.as_ref().map_or(0, |pad| pad.start(b)));
        if rope_pos >= limit {
            return Err(Error::InvalidArgument {
                arg: "position",
                reason: format!(
                    "row {b}: rope position {rope_pos} is outside the rope table, which holds \
                     {limit} positions"
                ),
            });
        }
        Ok(())
    }
}

/// `value` as the I32 the device tensor holds, or an error naming the row.
fn as_i32(value: usize, row: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::InvalidArgument {
        arg: "position",
        reason: format!("row {row}: cache end {value} is past what an I32 holds"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::voxcpm::minicpm4::model::tests::{HIDDEN, filled, tiny_model};
    use crate::test_utils::cpu_setup;
    use numr::autograd::Var;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    /// A fresh table puts every row at the prefill's position and ends one
    /// past it, and `advance` moves both together.
    #[test]
    fn advance_moves_every_row_and_keeps_the_address() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut table =
            LanePositions::<CpuRuntime>::new(None, 2, 4, Some(16), &device).expect("table");
        let address = table.kv_end().ptr();

        assert_eq!(table.positions(), &[4, 4]);
        assert_eq!(table.shared_position(), Some(4));
        assert_eq!(table.kv_end().to_vec::<i32>(), vec![5, 5]);

        table.advance(&client).expect("advance");
        assert_eq!(table.positions(), &[5, 5]);
        assert_eq!(table.kv_end().to_vec::<i32>(), vec![6, 6]);
        assert_eq!(table.kv_end().ptr(), address);
    }

    /// The table hands out a per-lane step whose write positions are its
    /// own rows, and whose bound stays shared while they agree.
    #[test]
    fn the_step_carries_the_tables_own_rows() {
        let device = CpuDevice::new();
        let pad = LeftPad::<CpuRuntime>::new(vec![0, 2], &device).expect("pad");
        let table =
            LanePositions::<CpuRuntime>::new(Some(pad), 2, 5, None, &device).expect("table");
        let step = table.step();

        assert_eq!(step.row_write(0), 5);
        assert_eq!(step.row_write(1), 5);
        assert_eq!(step.row_start(1), 2);
        assert_eq!(step.last_write(), 5);
        step.check_write_order(5).expect("both rows append");
    }

    /// The rope table bounds a row: the last position it holds is accepted,
    /// the one past it is refused, and a refused advance leaves the table
    /// where it was.
    #[test]
    fn the_rope_table_bounds_construction_and_advance() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        assert!(LanePositions::<CpuRuntime>::new(None, 1, 3, Some(3), &device).is_err());

        let mut table =
            LanePositions::<CpuRuntime>::new(None, 1, 2, Some(3), &device).expect("table");
        assert!(table.advance(&client).is_err());
        assert_eq!(table.positions(), &[2]);
        assert_eq!(table.kv_end().to_vec::<i32>(), vec![3]);
    }

    /// A padded row counts its rope position from its own start, so the
    /// bound follows the row rather than the shared slot.
    #[test]
    fn a_padded_row_is_bounded_from_its_own_start() {
        let device = CpuDevice::new();
        let pad = LeftPad::<CpuRuntime>::new(vec![0, 4], &device).expect("pad");
        // Row 1 sits at slot 5 but rope position 1, which a 2-position
        // table holds; row 0 sits at rope position 5, which it does not.
        assert!(LanePositions::<CpuRuntime>::new(Some(pad), 2, 5, Some(2), &device).is_err());

        let pad = LeftPad::<CpuRuntime>::new(vec![4, 4], &device).expect("pad");
        LanePositions::<CpuRuntime>::new(Some(pad), 2, 5, Some(2), &device)
            .expect("every row rotates at position 1");
    }

    /// The lane step the table hands out writes the same cache slots and
    /// returns the same floats as the fused write, while every row sits at
    /// one position.
    ///
    /// Bit equality, not a tolerance: the two differ only in how many
    /// copies carry the K/V into the cache, and both keep the shared key
    /// bound and the same rotation.
    #[test]
    fn a_lane_step_matches_the_fused_step_bit_for_bit() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);
        let (long, short) = (5usize, 3usize);
        let prefix = Var::new(filled(&[2, long, HIDDEN], 2, &device), false);
        let row = Var::new(filled(&[2, HIDDEN], 3, &device), false);
        let starts = vec![0, (long - short) as i32];

        let run = |lanes: bool| {
            let pad = LeftPad::<CpuRuntime>::new(starts.clone(), &device).expect("pad");
            let mut cache = model.new_kv_cache(2, 8).expect("cache");
            model
                .prefill(&client, &prefix, &mut cache, Some(&pad))
                .expect("prefill");
            let out = if lanes {
                let table = LanePositions::<CpuRuntime>::new(Some(pad), 2, long, None, &device)
                    .expect("table");
                model
                    .decode_step(&client, &row, &mut cache, &table.step())
                    .expect("lane step")
            } else {
                let step = CacheStep::prefix(long, Some(&pad));
                model
                    .decode_step(&client, &row, &mut cache, &step)
                    .expect("fused step")
            };
            let keys = cache.layer(0).expect("layer").k_cache_raw().to_vec::<f32>();
            let values = out
                .tensor()
                .contiguous()
                .expect("contiguous")
                .to_vec::<f32>();
            (values, keys)
        };

        let (fused_out, fused_keys) = run(false);
        let (lane_out, lane_keys) = run(true);
        assert!(
            fused_out.iter().any(|v| v.abs() > 1e-6),
            "degenerate output: the comparison below would pass vacuously"
        );
        assert_eq!(lane_out, fused_out);
        assert_eq!(lane_keys, fused_keys);
    }

    /// A table needs at least one row, and a pad must describe every one of
    /// them.
    #[test]
    fn construction_refuses_a_degenerate_batch() {
        let device = CpuDevice::new();
        assert!(LanePositions::<CpuRuntime>::new(None, 0, 0, None, &device).is_err());
        let pad = LeftPad::<CpuRuntime>::new(vec![0], &device).expect("pad");
        assert!(LanePositions::<CpuRuntime>::new(Some(pad), 2, 0, None, &device).is_err());
    }
}
