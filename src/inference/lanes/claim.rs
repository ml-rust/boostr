//! Taking a lane whose tokens are already in the batched containers.

use crate::error::{Error, Result};
use crate::inference::lanes::table::LaneTable;
use crate::inference::scheduler::SequenceId;
use numr::dtype::DType;
use numr::ops::HostCopyOps;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> LaneTable<R> {
    /// Record lane `lane` as holding `seq` with `len` tokens already in its
    /// cache lane, decoding next at rope position `rope_pos`, and write the
    /// table's device tensors.
    ///
    /// A caller that prefilled straight into lane `lane` of the batched KV
    /// cache and GDN state claims the lane instead of admitting into it: the
    /// tokens are already where the lane reads them, so there is nothing to
    /// install. Whatever the lane held before is replaced.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `lane` is outside the batch or
    /// `rope_pos` is outside the rope table; a copy error propagates.
    pub fn claim<C: HostCopyOps<R>>(
        &mut self,
        client: &C,
        lane: usize,
        seq: SequenceId,
        len: usize,
        rope_pos: usize,
    ) -> Result<()> {
        if lane >= self.batch() {
            return Err(Error::InferenceError {
                reason: format!(
                    "lane {lane} is outside batch {}; claim a lane the table holds",
                    self.batch()
                ),
            });
        }
        self.check_rope_pos(lane, rope_pos)?;
        self.occupants[lane] = Some(seq);
        self.lens[lane] = len;
        self.rope_pos[lane] = rope_pos;
        self.flush(client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    /// A claimed lane reports its own length, end and positions, and the
    /// other lane stays free.
    #[test]
    fn claim_records_one_lanes_length_and_position() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut table = LaneTable::<CpuRuntime>::new(2, 32, &device).expect("table");

        table.claim(&client, 1, 4, 6, 6).expect("claim");

        assert_eq!(table.occupant(1), Some(4));
        assert_eq!(table.lane_len(1), 6);
        assert_eq!(table.lane_rope_pos(1), 6);
        assert_eq!(table.kv_end().to_vec::<i32>(), vec![0, 7]);
        assert!(table.is_free(0));
        assert_eq!(
            table.positions().to_vec::<i32>(),
            vec![0, 6, 0, 6, 0, 6, 0, 0]
        );
    }

    /// A lane outside the batch and a position outside the rope table are
    /// both refused.
    #[test]
    fn claim_refuses_a_lane_or_a_position_the_table_does_not_hold() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut table = LaneTable::<CpuRuntime>::new(1, 8, &device).expect("table");

        let err = table.claim(&client, 1, 0, 0, 0).expect_err("lane 1");
        assert!(err.to_string().contains("lane 1"), "{err}");
        assert!(table.claim(&client, 0, 0, 2, 8).is_err());
        assert!(table.is_free(0));
    }
}
