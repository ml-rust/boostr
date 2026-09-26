//! The lane table's STEPPED WIDTH and the decode view narrowed to it.
//!
//! Every lane index — a claim, a free, a position read — addresses the
//! ALLOCATED width. The STEPPED WIDTH is the lane prefix one decode call
//! covers. The two differ whenever the lanes above the prefix hold nothing.
//!
//! Narrowing is a view, never a move. The stepped lanes are a prefix, so the
//! narrowed `kv_end` and left pad share their full tensors' storage and the
//! in-place writes `flush` makes show through them. No lane changes index.

use super::table::LanePositions;
use crate::error::{Error, Result};
use crate::model::audio::voxcpm::minicpm4::cache_step::CacheStep;
use crate::model::audio::voxcpm::minicpm4::left_pad::LeftPad;
use numr::dtype::DType;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> LanePositions<R> {
    /// Lanes one decode step covers: the prefix `0..active_width()`.
    ///
    /// [`batch`](LanePositions::batch) is the allocated width instead, and
    /// is what a lane index is checked against.
    pub fn active_width(&self) -> usize {
        self.active_width
    }

    /// Step the first `rows` lanes and leave the rest out of the call.
    ///
    /// [`step`](LanePositions::step) then carries `rows` write positions,
    /// `rows` occupancy flags and a `[rows]` key-range tensor, which is what
    /// a decode call over `rows` query rows needs.
    ///
    /// [`claim`](LanePositions::claim) and [`free`](LanePositions::free)
    /// address the allocated width, so a caller seats a lane above the
    /// prefix and widens afterwards.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `rows` is 0 or past the allocated
    /// width; a narrowing error propagates.
    pub fn set_active_width(&mut self, rows: usize) -> Result<()> {
        let width = self.batch();
        if rows == 0 || rows > width {
            return Err(Error::InvalidArgument {
                arg: "rows",
                reason: format!(
                    "expected a stepped width in 1..={width}, the lanes this table was \
                     built with, got {rows}"
                ),
            });
        }
        self.active_width = rows;
        if rows == width {
            self.kv_end_active = None;
            self.pad_active = None;
            return Ok(());
        }
        self.kv_end_active = Some(self.kv_end.narrow(0, 0, rows)?);
        self.pad_active = match self.pad.as_ref() {
            Some(pad) => Some(LeftPad {
                starts: pad.starts.narrow(0, 0, rows)?,
                host: pad.host[..rows].to_vec(),
            }),
            None => None,
        };
        Ok(())
    }

    /// The view one decode step takes, borrowed from this table and covering
    /// the stepped lanes.
    pub fn step(&self) -> CacheStep<'_, R> {
        let rows = self.active_width;
        let pad = self.pad_active.as_ref().or(self.pad.as_ref());
        let kv_end = self.kv_end_active.as_ref().unwrap_or(&self.kv_end);
        CacheStep::lanes(pad, &self.lens[..rows], kv_end, &self.occupied[..rows])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::voxcpm::minicpm4::cache_step::CacheFill;
    use crate::model::audio::voxcpm::minicpm4::model::tests::{HIDDEN, filled, tiny_model};
    use crate::ops::traits::KvSeqLen;
    use crate::test_utils::cpu_setup;
    use numr::autograd::Var;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    /// A fresh table steps every lane it holds.
    #[test]
    fn the_stepped_width_starts_at_the_allocated_one() {
        let device = CpuDevice::new();
        let full = LanePositions::<CpuRuntime>::new(None, 3, 2, None, &device).expect("table");
        assert_eq!(full.active_width(), 3);
        let empty = LanePositions::<CpuRuntime>::empty(4, None, &device).expect("table");
        assert_eq!(empty.active_width(), 4);
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
        step.check_write_order(&CacheFill {
            filled: 5,
            lanes: &[5, 5],
            limit: 8,
        })
        .expect("both rows sit at their own end");
    }

    /// Narrowing the stepped width narrows every part of the step: the
    /// write positions, the occupancy flags and the device key range.
    #[test]
    fn a_narrowed_table_steps_only_its_prefix() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut table = LanePositions::<CpuRuntime>::empty(4, Some(16), &device).expect("table");
        table.claim(&client, 0, 3).expect("claim");
        table.claim(&client, 1, 5).expect("claim");

        table.set_active_width(2).expect("narrow");
        assert_eq!(table.batch(), 4, "the allocation never moves");
        assert_eq!(table.active_width(), 2);

        let step = table.step();
        assert_eq!(step.row_write(0), 3);
        assert_eq!(step.row_write(1), 5);
        assert_eq!(step.last_write(), 5, "only the stepped lanes count");
        match step.kv_seq_len(99) {
            KvSeqLen::PerRow(ends) => assert_eq!(ends.to_vec::<i32>(), vec![4, 6]),
            KvSeqLen::Shared(_) => panic!("the two lanes end at different keys"),
        }
    }

    /// The narrowed key range is a VIEW: an advance writes the full tensor in
    /// place and the stepped view reports the new values unrebuilt.
    #[test]
    fn the_narrowed_key_range_follows_the_table() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut table = LanePositions::<CpuRuntime>::empty(4, Some(16), &device).expect("table");
        table.claim(&client, 0, 3).expect("claim");
        table.claim(&client, 1, 5).expect("claim");
        table.set_active_width(2).expect("narrow");

        table.advance(&client).expect("advance");
        match table.step().kv_seq_len(99) {
            KvSeqLen::PerRow(ends) => assert_eq!(ends.to_vec::<i32>(), vec![5, 7]),
            KvSeqLen::Shared(_) => panic!("the two lanes end at different keys"),
        }
    }

    /// A narrowed padded table hands the kernels the starts of its stepped
    /// lanes alone.
    #[test]
    fn a_narrowed_padded_table_narrows_its_starts() {
        let device = CpuDevice::new();
        let pad = LeftPad::<CpuRuntime>::new(vec![0, 2, 4, 4], &device).expect("pad");
        let mut table =
            LanePositions::<CpuRuntime>::new(Some(pad), 4, 5, None, &device).expect("table");
        table.set_active_width(2).expect("narrow");

        let step = table.step();
        let starts = step.kv_start().expect("padded").to_vec::<i32>();
        assert_eq!(starts, vec![0, 2]);
        assert_eq!(step.row_start(1), 2);
    }

    #[test]
    fn a_stepped_width_outside_the_allocation_is_refused() {
        let device = CpuDevice::new();
        let mut table = LanePositions::<CpuRuntime>::empty(2, None, &device).expect("table");
        assert!(table.set_active_width(0).is_err());
        assert!(table.set_active_width(3).is_err());
        assert_eq!(table.active_width(), 2, "a refusal changes nothing");
    }

    /// Widening back to the allocation drops the narrowed views.
    #[test]
    fn widening_back_restores_the_whole_table() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut table = LanePositions::<CpuRuntime>::empty(3, Some(16), &device).expect("table");
        table.claim(&client, 0, 1).expect("claim");
        table.set_active_width(1).expect("narrow");
        match table.step().kv_seq_len(99) {
            KvSeqLen::Shared(end) => assert_eq!(end, 2),
            KvSeqLen::PerRow(_) => panic!("one stepped lane cannot be ragged"),
        }

        table.claim(&client, 2, 1).expect("claim");
        table.set_active_width(3).expect("widen");
        assert_eq!(table.active_width(), 3);
        let step = table.step();
        assert_eq!(step.row_write(2), 1);
        match step.kv_seq_len(99) {
            KvSeqLen::Shared(end) => assert_eq!(end, 2, "the free lane is not counted"),
            KvSeqLen::PerRow(_) => panic!("both occupied lanes sit at one position"),
        }
    }

    /// While every row sits at one position, the lane step writes the same
    /// cache slots and returns the same floats as the fused write.
    ///
    /// Bit equality, not a tolerance: both keep the shared key bound and the
    /// same rotation.
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
}
