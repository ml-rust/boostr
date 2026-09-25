//! [`GenerateState::refresh_active_width`]: the width one iteration steps,
//! recomputed from the slots that hold a request.
//!
//! A state is opened at a fixed CEILING and a slot index addresses it. What
//! moves is the STEPPED width — the prefix of slots one iteration runs over.
//! Requests are seated lowest-free-first and no row ever migrates, so the
//! occupied slots are a prefix and the stepped width is a cut through that
//! prefix, never a gather.
//!
//! The ALLOCATION — the rows the device tensors hold — moves too, but only
//! upward and only in `super::grow`, which widens it to cover a slot before
//! that slot is seated. So the stepped width computed here always fits
//! inside the allocation and this module never has to grow anything.
//!
//! The width is not the occupied count itself. It is that count rounded UP
//! to an entry of the shared allowlist, because the DiT's Euler solver keys
//! its captured graphs by batch size: a width per occupancy would put a
//! fresh key in a bounded cache on every seat and release. The rounding is
//! then capped at the ceiling, and never falls below the occupied count.

use super::super::*;
use crate::model::audio::voxcpm::model::widths::batch_width;

impl<R: Runtime<DType = DType>> GenerateState<R> {
    /// Recompute the stepped width and push it into the lane table and both
    /// cache stacks.
    ///
    /// Call this after any change to which slots hold a request — that is
    /// [`admit`](Self::admit) and [`release`](Self::release), and the state
    /// constructors so one rule decides the width whichever built it.
    ///
    /// # Errors
    ///
    /// Propagates a narrowing or cache refusal. Seating widens the
    /// allocation to the same rounded width before it claims a lane, so the
    /// computed width is always in `1..=alloc_width` and neither the lane
    /// table nor a cache can refuse it for being out of range.
    pub(super) fn refresh_active_width(&mut self) -> Result<()>
    where
        R::Client: IndexingOps<R>,
    {
        let width = self.occupied_width();
        self.active_width = width;
        self.prefill.lanes.set_active_width(width)?;
        self.prefill.base_cache.set_active_batch(width)?;
        self.prefill.residual_cache.set_active_batch(width)?;
        Ok(())
    }

    /// The width the currently occupied slots call for: the smallest
    /// allowlist entry that holds them, capped at the ceiling and never
    /// below the slots themselves.
    ///
    /// An empty state steps one row rather than none: the caches and the
    /// tensors have no zero-row shape, and a state with nothing seated does
    /// not step at all.
    fn occupied_width(&self) -> usize {
        let seated = (0..self.batch)
            .rev()
            .find(|row| self.occupied(*row))
            .map_or(0, |row| row + 1);
        let needed = seated.max(1);
        // `batch_width` saturates at its largest entry, which can sit below
        // a ceiling wider than the list. Clamping to `needed` from below
        // keeps a seated row inside the prefix whatever the list holds.
        batch_width(needed).min(self.batch).max(needed)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::*;
    use super::*;
    use crate::model::audio::voxcpm::local_dit::tests::{FEAT_DIM, PATCH_SIZE, t};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    /// Seat one request into each of `slots`, in the order given.
    fn seated(
        client: &CpuClient,
        m: &VoxCpm2Model<CpuRuntime>,
        device: &CpuDevice,
        width: usize,
        slots: &[usize],
    ) -> GenerateState<CpuRuntime> {
        let ref_feat = t(&[3, PATCH_SIZE, FEAT_DIM], 0.7, device);
        let row = slot_rows(&ref_feat)[0];
        let mut state = GenerateState::open(m, width, SLOT_CAPACITY).expect("open");
        for &slot in slots {
            state
                .admit(client, slot, &prefill_one(client, m, &row), row_opts(2, 7))
                .expect("admit");
        }
        state
    }

    /// The ceiling never moves, and an empty state steps one row.
    #[test]
    fn an_empty_state_steps_one_row() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let state = seated(&client, &m, &device, 8, &[]);
        assert_eq!(state.batch, 8);
        assert_eq!(state.active_width(), 1);
        assert_eq!(state.alloc_width(), 1, "nothing seated, nothing reserved");
    }

    /// The width rounds UP to an allowlist entry: three seated slots step at
    /// four, not at three.
    #[test]
    fn the_width_rounds_up_to_an_allowed_width() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        for (seats, want) in [(1usize, 1usize), (2, 2), (3, 4), (4, 4), (5, 8)] {
            let slots: Vec<usize> = (0..seats).collect();
            let state = seated(&client, &m, &device, 8, &slots);
            assert_eq!(state.active_width(), want, "{seats} seated");
            assert_eq!(state.batch, 8, "the ceiling never moves");
            assert_eq!(
                state.alloc_width(),
                want,
                "the allocation follows the stepped width up"
            );
        }
    }

    /// The rounding is capped at the ceiling: a state narrower than the
    /// next allowlist entry steps its own width and no more.
    #[test]
    fn the_width_never_exceeds_the_allocation() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let state = seated(&client, &m, &device, 3, &[0, 1, 2]);
        assert_eq!(state.active_width(), 3);
    }

    /// Seating a slot above the current stepped width grows it, so the new
    /// request is inside the prefix the loop runs over.
    #[test]
    fn seating_above_the_width_grows_it() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let mut state = seated(&client, &m, &device, 8, &[0]);
        assert_eq!(state.active_width(), 1);

        let ref_feat = t(&[3, PATCH_SIZE, FEAT_DIM], 0.7, &device);
        let row = slot_rows(&ref_feat)[0];
        state
            .admit(&client, 4, &prefill_one(&client, &m, &row), row_opts(2, 7))
            .expect("admit");
        assert_eq!(state.active_width(), 8, "slot 4 needs five rows");
        assert_eq!(
            state.prefill.base_cache.active_batch(),
            8,
            "both stacks follow the width"
        );
        assert_eq!(state.prefill.lanes.active_width(), 8);
    }

    /// Releasing the HIGHEST occupied slot shrinks the width; releasing one
    /// below it does not, because the prefix still has to reach the highest.
    #[test]
    fn releasing_the_highest_slot_shrinks_the_width() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let mut state = seated(&client, &m, &device, 8, &[0, 1, 2]);
        assert_eq!(state.active_width(), 4);

        state.release(&client, 0).expect("release");
        assert_eq!(
            state.active_width(),
            4,
            "slot 2 is still seated, so the prefix still reaches it"
        );

        state.release(&client, 2).expect("release");
        assert_eq!(state.active_width(), 2, "slot 1 is now the highest");
        assert_eq!(state.prefill.residual_cache.active_batch(), 2);

        state.release(&client, 1).expect("release");
        assert_eq!(state.active_width(), 1, "nothing is seated");
    }
}
