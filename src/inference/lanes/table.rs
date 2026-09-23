//! Lane occupancy and the persistent device tensors a batched decode reads.

use crate::error::{Error, Result};
use crate::inference::lanes::step::LaneKvStep;
use crate::inference::scheduler::SequenceId;
use numr::dtype::DType;
use numr::ops::HostCopyOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Which lanes of a batched decode are taken, and the tensors the forward
/// reads to serve them.
///
/// Every lane holds one sequence from cache position 0 with its own end, so
/// admitting a sequence never depends on what the other lanes hold. The
/// three tensors — `kv_start` `[batch]`, `kv_end` `[batch]` and `positions`
/// `[4, batch, 1]` — are allocated once and written in place through
/// [`HostCopyOps::write_host_slice`], so their device addresses hold for the
/// life of the table and a captured graph can point at them.
///
/// A host mirror of each lane's length and rope position backs every write,
/// so no write reads a device tensor back, and the two staging buffers the
/// writes read from are reused across steps.
///
/// `rope_table_len` bounds every lane: a rope position the model's rope
/// table does not hold is refused at the lane, not read out of bounds in a
/// kernel.
pub struct LaneTable<R: Runtime> {
    /// The sequence in each lane, `None` when the lane is free.
    pub(super) occupants: Vec<Option<SequenceId>>,
    /// Tokens lane `b` holds, which is also the position its next step
    /// writes.
    pub(super) lens: Vec<usize>,
    /// Rope position lane `b`'s next step decodes at.
    pub(super) rope_pos: Vec<usize>,
    /// Rope positions the model's table holds, so `0..rope_table_len`.
    pub(super) rope_table_len: usize,
    kv_start: Tensor<R>,
    kv_end: Tensor<R>,
    positions: Tensor<R>,
    /// Staging for `kv_end`, one entry per lane.
    ends: Vec<i32>,
    /// Staging for `positions`, four streams of one entry per lane.
    streams: Vec<i32>,
}

impl<R: Runtime<DType = DType>> LaneTable<R> {
    /// A table of `batch` free lanes, with its tensors allocated on
    /// `device`.
    ///
    /// `rope_table_len` is how many rope positions the model's table holds,
    /// so a lane may sit at any position in `0..rope_table_len`.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `batch` is 0 or `rope_table_len` is 0;
    /// an allocation error propagates.
    pub fn new(batch: usize, rope_table_len: usize, device: &R::Device) -> Result<Self> {
        if batch == 0 {
            return Err(Error::InferenceError {
                reason: "a lane table needs at least one lane, got batch 0; allocate it with \
                         the number of sequences the decode runs at once"
                    .into(),
            });
        }
        if rope_table_len == 0 {
            return Err(Error::InferenceError {
                reason: "a lane table needs at least one rope position, got rope_table_len 0; \
                         allocate it with the length of the model's rope table"
                    .into(),
            });
        }
        Ok(Self {
            occupants: vec![None; batch],
            lens: vec![0; batch],
            rope_pos: vec![0; batch],
            rope_table_len,
            kv_start: Tensor::<R>::zeros(&[batch], DType::I32, device)?,
            kv_end: Tensor::<R>::zeros(&[batch], DType::I32, device)?,
            positions: Tensor::<R>::zeros(&[4, batch, 1], DType::I32, device)?,
            ends: vec![0; batch],
            streams: vec![0; 4 * batch],
        })
    }

    /// Rope positions the model's table holds.
    pub fn rope_table_len(&self) -> usize {
        self.rope_table_len
    }

    /// Lane count.
    pub fn batch(&self) -> usize {
        self.occupants.len()
    }

    /// The sequence in lane `lane`, or `None` when it is free or outside
    /// the batch.
    pub fn occupant(&self, lane: usize) -> Option<SequenceId> {
        self.occupants.get(lane).copied().flatten()
    }

    /// Whether lane `lane` is free. A lane outside the batch is not.
    pub fn is_free(&self, lane: usize) -> bool {
        self.occupants.get(lane).is_some_and(Option::is_none)
    }

    /// How many lanes are free.
    pub fn free_lanes(&self) -> usize {
        self.occupants.iter().filter(|o| o.is_none()).count()
    }

    /// Tokens lane `lane` holds, and the cache position its next step
    /// writes. A lane outside the batch holds none.
    pub fn lane_len(&self, lane: usize) -> usize {
        self.lens.get(lane).copied().unwrap_or(0)
    }

    /// The rope position lane `lane` decodes at next.
    pub fn lane_rope_pos(&self, lane: usize) -> usize {
        self.rope_pos.get(lane).copied().unwrap_or(0)
    }

    /// `[4, batch, 1]` I32 rope positions: the first three streams carry
    /// each lane's position, the fourth is zero.
    pub fn positions(&self) -> &Tensor<R> {
        &self.positions
    }

    /// `[batch]` I32 first cache position per lane.
    pub fn kv_start(&self) -> &Tensor<R> {
        &self.kv_start
    }

    /// `[batch]` I32 key end per lane, one past the position this step
    /// writes.
    pub fn kv_end(&self) -> &Tensor<R> {
        &self.kv_end
    }

    /// Each lane's write position on the host.
    pub fn write_positions(&self) -> &[usize] {
        &self.lens
    }

    /// The whole KV view one decode step takes, borrowed from the table's
    /// own tensors.
    pub fn step(&self) -> LaneKvStep<'_, R> {
        LaneKvStep {
            kv_start: &self.kv_start,
            kv_end: &self.kv_end,
            write_pos: &self.lens,
        }
    }

    /// Write the host mirrors into the device tensors, in place.
    ///
    /// `kv_start` never moves off zero, so only `kv_end` and `positions`
    /// are written: an occupied lane ends one past the position it is about
    /// to write, and a free lane ends where it starts, an empty range no
    /// kernel reads. Both writes go straight from the table's own staging
    /// buffers into the device tensors, so a step allocates nothing.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when a length or position is past what an
    /// I32 holds; a copy error propagates.
    pub(super) fn flush<C: HostCopyOps<R>>(&mut self, client: &C) -> Result<()> {
        let batch = self.batch();

        self.ends.clear();
        for (lane, (occupant, len)) in self.occupants.iter().zip(&self.lens).enumerate() {
            let end = if occupant.is_some() { len + 1 } else { 0 };
            self.ends.push(as_i32(end, "kv_end", lane)?);
        }
        client
            .write_host_slice(&self.kv_end, &self.ends)
            .map_err(Error::Numr)?;

        self.streams.clear();
        for _ in 0..3 {
            for (lane, pos) in self.rope_pos.iter().enumerate() {
                self.streams.push(as_i32(*pos, "positions", lane)?);
            }
        }
        self.streams.resize(4 * batch, 0);
        client
            .write_host_slice(&self.positions, &self.streams)
            .map_err(Error::Numr)
    }

    /// Point lane `lane` at rope position `pos` and write the table's
    /// device tensors.
    ///
    /// The lane's occupancy and length stay as they were, so a caller that
    /// drives one lane's rope position itself — a captured single-lane
    /// decode graph — writes the positions tensor without an `admit`.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `lane` is outside the batch or `pos`
    /// is outside the rope table; a copy error propagates.
    pub fn set_rope_pos<C: HostCopyOps<R>>(
        &mut self,
        client: &C,
        lane: usize,
        pos: usize,
    ) -> Result<()> {
        if lane >= self.batch() {
            return Err(Error::InferenceError {
                reason: format!(
                    "lane {lane} is outside batch {}; write a rope position to a lane the \
                     table holds",
                    self.batch()
                ),
            });
        }
        self.check_rope_pos(lane, pos)?;
        self.rope_pos[lane] = pos;
        self.flush(client)
    }

    /// Refuse a rope position the model's rope table does not hold.
    pub(super) fn check_rope_pos(&self, lane: usize, pos: usize) -> Result<()> {
        if pos >= self.rope_table_len {
            return Err(Error::InferenceError {
                reason: format!(
                    "lane {lane}: rope position {pos} is outside the rope table, which holds \
                     {} positions; release the lane and admit the sequence again with a \
                     shorter prefix",
                    self.rope_table_len
                ),
            });
        }
        Ok(())
    }
}

/// `value` as the I32 a device tensor holds, or an error naming the lane.
fn as_i32(value: usize, field: &'static str, lane: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::InferenceError {
        reason: format!(
            "lane {lane}: {field} {value} is past what an I32 holds; release the lane and \
             admit the sequence again with a shorter prefix"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    #[test]
    fn new_allocates_every_tensor_and_leaves_lanes_free() {
        let device = CpuDevice::new();
        let table = LaneTable::<CpuRuntime>::new(3, 32, &device).expect("table");

        assert_eq!(table.batch(), 3);
        assert_eq!(table.free_lanes(), 3);
        assert_eq!(table.kv_start().shape(), &[3]);
        assert_eq!(table.kv_end().shape(), &[3]);
        assert_eq!(table.positions().shape(), &[4, 3, 1]);
        assert_eq!(table.kv_start().dtype(), DType::I32);
        assert_eq!(table.kv_end().to_vec::<i32>(), vec![0, 0, 0]);
    }

    #[test]
    fn a_table_needs_at_least_one_lane() {
        let device = CpuDevice::new();
        assert!(LaneTable::<CpuRuntime>::new(0, 32, &device).is_err());
        assert!(LaneTable::<CpuRuntime>::new(2, 0, &device).is_err());
    }

    #[test]
    fn accessors_read_lanes_outside_the_batch_as_empty() {
        let device = CpuDevice::new();
        let table = LaneTable::<CpuRuntime>::new(1, 32, &device).expect("table");

        assert!(table.is_free(0));
        assert!(!table.is_free(1));
        assert_eq!(table.occupant(1), None);
        assert_eq!(table.lane_len(1), 0);
        assert_eq!(table.lane_rope_pos(1), 0);
    }

    #[test]
    fn step_borrows_the_tables_own_tensors() {
        let device = CpuDevice::new();
        let table = LaneTable::<CpuRuntime>::new(2, 32, &device).expect("table");
        let step = table.step();

        assert_eq!(step.kv_start.ptr(), table.kv_start().ptr());
        assert_eq!(step.kv_end.ptr(), table.kv_end().ptr());
        assert_eq!(step.write_pos, &[0, 0]);
    }

    #[test]
    fn the_rope_bound_admits_the_last_position_and_refuses_the_one_past_it() {
        let device = CpuDevice::new();
        let table = LaneTable::<CpuRuntime>::new(1, 8, &device).expect("table");

        assert_eq!(table.rope_table_len(), 8);
        assert!(table.check_rope_pos(0, 7).is_ok());
        assert!(table.check_rope_pos(0, 8).is_err());
    }

    /// Writing one lane's rope position fills that lane's three position
    /// streams and leaves the other lane where it was.
    #[test]
    fn set_rope_pos_writes_one_lanes_streams() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut table = LaneTable::<CpuRuntime>::new(2, 32, &device).expect("table");
        let before = table.positions().ptr();

        table.set_rope_pos(&client, 1, 6).expect("set");

        assert_eq!(table.lane_rope_pos(1), 6);
        assert_eq!(table.lane_rope_pos(0), 0);
        assert_eq!(
            table.positions().to_vec::<i32>(),
            vec![0, 6, 0, 6, 0, 6, 0, 0]
        );
        assert_eq!(table.positions().ptr(), before);
    }

    /// A lane outside the batch and a position outside the rope table are
    /// both refused.
    #[test]
    fn set_rope_pos_refuses_a_lane_or_a_position_the_table_does_not_hold() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut table = LaneTable::<CpuRuntime>::new(1, 8, &device).expect("table");

        let err = table.set_rope_pos(&client, 1, 0).expect_err("lane 1");
        assert!(err.to_string().contains("lane 1"), "{err}");
        assert!(table.set_rope_pos(&client, 0, 8).is_err());
        assert_eq!(table.lane_rope_pos(0), 0);
    }

    #[test]
    fn a_rope_position_past_the_table_names_the_lane_and_the_limit() {
        let device = CpuDevice::new();
        let table = LaneTable::<CpuRuntime>::new(2, 8, &device).expect("table");
        let err = table.check_rope_pos(1, 9).expect_err("past the table");

        let text = err.to_string();
        assert!(text.contains("lane 1"), "{text}");
        assert!(text.contains('9'), "{text}");
        assert!(text.contains('8'), "{text}");
    }
}
