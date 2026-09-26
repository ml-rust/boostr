//! Growing the lane table's ALLOCATED width.
//!
//! Growth adds lanes ABOVE the ones already there. The rows the table holds
//! are a prefix of the wider table, so each keeps its index, length and
//! occupancy. Added lanes start free at an empty key range.
//!
//! The allocation never shrinks.

use super::table::LanePositions;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::HostCopyOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> LanePositions<R> {
    /// Widen the table to `new_batch` lanes, keeping every lane it holds.
    ///
    /// The host per-lane vectors extend with a free lane's values. The `[B]`
    /// I32 `kv_end` mirror is reallocated at the new width and rewritten from
    /// them, so an added lane reads as an empty key range until claimed.
    ///
    /// The STEPPED width is untouched;
    /// [`set_active_width`](LanePositions::set_active_width) owns it. Its
    /// narrowed view is rebuilt here against the new tensor.
    ///
    /// A no-op when `new_batch` is not larger than the current width.
    ///
    /// # Device addresses
    ///
    /// This REPLACES `kv_end`. An address
    /// [`kv_end`](LanePositions::kv_end) handed out earlier no longer
    /// describes the tensor a step reads. A caller holding a captured graph
    /// against it must not grow the table.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when the table carries left padding: a
    /// padded lane's first cache position is not 0, so the width it was
    /// padded at is the width it decodes at. An allocation or copy error
    /// propagates.
    pub fn grow_batch<C: HostCopyOps<R>>(&mut self, new_batch: usize, client: &C) -> Result<()> {
        if new_batch <= self.batch() {
            return Ok(());
        }
        if self.pad.is_some() {
            return Err(Error::InvalidArgument {
                arg: "new_batch",
                reason: "a left-padded table cannot be widened: its rows were padded to one \
                         batch's shape, and a lane added afterwards has no padding of its own"
                    .to_string(),
            });
        }
        let device = self.kv_end.device().clone();
        self.lens.resize(new_batch, 0);
        self.occupied.resize(new_batch, false);
        self.ends = vec![0i32; new_batch];
        self.kv_end = Tensor::<R>::from_slice(&self.ends, &[new_batch], &device)?;
        // Rebuild the narrowed view at the same stepped width against the
        // replacement tensor.
        let stepped = self.active_width();
        self.set_active_width(stepped)?;
        // Occupied lanes keep their own ends. Added lanes flush to 0.
        self.flush(client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    /// Growth keeps every seated lane's position and key range, and the
    /// lanes it adds start free.
    #[test]
    fn growth_preserves_the_lanes_already_seated() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut table = LanePositions::<CpuRuntime>::empty(2, Some(16), &device).expect("table");
        table.claim(&client, 0, 5).expect("claim");
        table.advance(&client).expect("advance");

        table.grow_batch(4, &client).expect("grow");

        assert_eq!(table.batch(), 4);
        assert_eq!(table.positions(), &[6, 0, 0, 0]);
        assert_eq!(table.kv_end().to_vec::<i32>(), vec![7, 0, 0, 0]);
        assert!(!table.is_free(0), "the seated lane is still seated");
        assert!((1..4).all(|b| table.is_free(b)), "every other lane is free");
    }

    /// A lane claimed after growth carries exactly what the same claim puts
    /// into a table built wide from the start.
    #[test]
    fn a_lane_claimed_after_growth_matches_a_table_born_wide() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let mut grown = LanePositions::<CpuRuntime>::empty(1, Some(16), &device).expect("table");
        grown.grow_batch(4, &client).expect("grow");
        grown.claim(&client, 3, 6).expect("claim");

        let mut born = LanePositions::<CpuRuntime>::empty(4, Some(16), &device).expect("table");
        born.claim(&client, 3, 6).expect("claim");

        assert_eq!(grown.positions(), born.positions());
        assert_eq!(
            grown.kv_end().to_vec::<i32>(),
            born.kv_end().to_vec::<i32>()
        );
    }

    /// A request for the current width, or a narrower one, changes nothing.
    #[test]
    fn growth_to_a_width_that_is_not_larger_is_a_no_op() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut table = LanePositions::<CpuRuntime>::empty(4, None, &device).expect("table");
        let address = table.kv_end().ptr();

        table.grow_batch(4, &client).expect("same width");
        assert_eq!(table.batch(), 4);
        assert_eq!(table.kv_end().ptr(), address, "nothing is reallocated");

        table.grow_batch(2, &client).expect("narrower");
        assert_eq!(table.batch(), 4, "the allocation never shrinks");
        assert_eq!(table.kv_end().ptr(), address);
    }

    /// Growth leaves the stepped width where it is and rebuilds its view
    /// against the tensor a step now reads.
    #[test]
    fn growth_keeps_the_stepped_width_and_rebuilds_its_view() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let mut table = LanePositions::<CpuRuntime>::empty(2, Some(16), &device).expect("table");
        table.claim(&client, 0, 3).expect("claim");
        table.set_active_width(1).expect("narrow");

        table.grow_batch(8, &client).expect("grow");

        assert_eq!(table.active_width(), 1, "growth steps no more lanes");
        let step = table.step();
        assert_eq!(step.row_write(0), 3, "the step still reads lane 0");
        table.advance(&client).expect("advance");
        assert_eq!(
            table.kv_end().to_vec::<i32>()[0],
            5,
            "the narrowed view must sit on the tensor flush writes"
        );
    }

    /// A left-padded table is refused: a lane added afterwards has no
    /// padding of its own.
    #[test]
    fn a_padded_table_refuses_to_grow() {
        use crate::model::audio::voxcpm::minicpm4::left_pad::LeftPad;

        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let pad = LeftPad::<CpuRuntime>::new(vec![0, 2], &device).expect("pad");
        let mut table =
            LanePositions::<CpuRuntime>::new(Some(pad), 2, 5, None, &device).expect("table");

        assert!(table.grow_batch(4, &client).is_err());
        assert_eq!(table.batch(), 2, "a refusal changes nothing");
    }
}
