//! [`GenerateState::ensure_alloc`]: widening the DEVICE allocation so a slot
//! has a row of its own.
//!
//! A state opens allocated for ONE row whatever its ceiling. Everything a
//! step touches is stored with the slots as the OUTERMOST dimension — both
//! KV caches, both handoff rows, `prefix_feat_cond` and the lane table — and
//! slots are seated lowest-free-first, so the rows in use are a contiguous
//! prefix of any wider allocation. Widening is therefore: allocate the
//! larger tensor, copy the old rows into its prefix, zero the rest, swap.
//! Every row keeps its index and its contents, so no row's audio depends on
//! whether the allocation grew under it.
//!
//! The widths growth can take come from the same allowlist the stepped width
//! rounds to, so a state grows only a handful of times before it reaches its
//! ceiling.
//!
//! The allocation NEVER shrinks. A batch that has seen concurrency keeps the
//! room, because shrinking would only reallocate on the next burst.
//!
//! Only DEVICE tensors grow. The HOST per-row vectors — `patches`,
//! `outcomes`, `patch_len`, `slot_options` — are sized at the CEILING from
//! `open` and stay there: they are small, and their length is what a slot
//! index is checked against. The two row counts genuinely differ, so read a
//! device tensor's rows off `alloc_width` and a slot index's bound off
//! `batch`.
//!
//! # Why growing is safe for this caller
//!
//! Growing replaces the KV cache buffers and the lane table's `kv_end`, so
//! it invalidates any CUDA graph captured against them. The only captured
//! graph in this generation path is the DiT's Euler solve, which takes the
//! noise, the DiT condition and the two projected hidden rows — it never
//! touches a KV cache or the lane table. Nothing else here holds a device
//! address across a call.

use super::super::*;
use crate::model::audio::voxcpm::model::widths::batch_width;

impl<R: Runtime<DType = DType>> GenerateState<R> {
    /// Widen the device allocation until it holds at least `rows` rows.
    ///
    /// Call this BEFORE writing anything into a slot at or above the
    /// current allocation. The target is `rows` rounded up to an allowed
    /// width and capped at the ceiling, which is the same rule
    /// `refresh_active_width` applies, so the stepped width that follows a
    /// seat always fits inside what this reserved.
    ///
    /// A no-op when the allocation already holds `rows` rows.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `rows` is past the ceiling. An
    /// allocation or copy error propagates.
    pub(super) fn ensure_alloc<C>(&mut self, client: &C, rows: usize) -> Result<()>
    where
        C: IndexingOps<R> + HostCopyOps<R>,
        R::Client: IndexingOps<R>,
    {
        if rows > self.batch {
            return Err(Error::InvalidArgument {
                arg: "rows",
                reason: format!(
                    "expected at most {} rows, the ceiling this state was opened at, got {rows}",
                    self.batch
                ),
            });
        }
        // Same rounding as the stepped width, so seating never asks for a
        // width the allocation does not already hold.
        let needed = batch_width(rows).min(self.batch).max(rows);
        if needed <= self.alloc_width {
            return Ok(());
        }
        self.prefill.base_cache.grow_batch(needed)?;
        self.prefill.residual_cache.grow_batch(needed)?;
        self.prefill.lm_hidden = grow_rows(client, &self.prefill.lm_hidden, needed)?;
        self.prefill.residual_hidden = grow_rows(client, &self.prefill.residual_hidden, needed)?;
        self.prefix_feat_cond = grow_rows(client, &self.prefix_feat_cond, needed)?;
        self.prefill.lanes.grow_batch(needed, client)?;
        self.alloc_width = needed;
        // The handoff's own row count is the allocation's, so a later shape
        // check reads one number rather than two that can drift.
        self.prefill.batch = needed;
        Ok(())
    }
}

/// `full` re-allocated at `new_rows` rows, holding its existing rows in the
/// prefix and zeros above them.
///
/// Rows are the outermost dimension, so the old tensor IS the prefix of the
/// new one and one dim-0 assignment at offset 0 carries every row across
/// unchanged.
fn grow_rows<R, C>(client: &C, full: &Var<R>, new_rows: usize) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: IndexingOps<R>,
{
    let stored = full.tensor();
    let mut shape = stored.shape().to_vec();
    shape[0] = new_rows;
    let wider = Tensor::<R>::zeros(&shape, stored.dtype(), stored.device()).map_err(Error::Numr)?;
    Ok(Var::new(
        client
            .slice_assign(&wider, stored, 0, 0)
            .map_err(Error::Numr)?,
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::*;
    use super::*;
    use crate::model::audio::voxcpm::local_dit::tests::{FEAT_DIM, PATCH_SIZE, t};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    /// Every K buffer of both stacks, layer by layer.
    fn kv(state: &GenerateState<CpuRuntime>) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        for stack in [&state.prefill.base_cache, &state.prefill.residual_cache] {
            for layer in 0..stack.num_layers() {
                if let Some(c) = stack.layer(layer) {
                    out.push(c.k_cache_raw().to_vec::<f32>());
                    out.push(c.v_cache_raw().to_vec::<f32>());
                }
            }
        }
        out
    }

    /// A state opened at a ceiling above one row still allocates one row.
    #[test]
    fn a_wide_ceiling_still_opens_at_one_row() {
        let (_client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let state = GenerateState::open(&m, 8, SLOT_CAPACITY).expect("open");

        assert_eq!(state.batch, 8, "the ceiling is what was asked for");
        assert_eq!(state.alloc_width(), 1, "the allocation is one row");
        assert_eq!(state.active_width(), 1);
        assert_eq!(state.prefill.lm_hidden.shape()[0], 1);
        assert_eq!(state.prefill.residual_hidden.shape()[0], 1);
        assert_eq!(state.prefix_feat_cond.shape()[0], 1);
        assert_eq!(state.prefill.lanes.batch(), 1);
        let base = state.prefill.base_cache.layer(0).expect("layer");
        assert_eq!(base.batch_size(), 1);
        assert_eq!(state.patches.len(), 8, "the host vectors hold the ceiling");
        assert_eq!(state.outcomes.len(), 8);
        assert_eq!(state.patch_len.len(), 8);
        assert_eq!(state.slot_options.len(), 8);
    }

    /// Growth keeps the seated row's caches and handoff rows byte for byte,
    /// and the rows it adds start zeroed.
    #[test]
    fn growth_preserves_the_rows_already_seated() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let ref_feat = t(&[3, PATCH_SIZE, FEAT_DIM], 0.7, &device);
        let row = slot_rows(&ref_feat)[0];

        let mut state = GenerateState::open(&m, 8, SLOT_CAPACITY).expect("open");
        state
            .admit(&client, 0, &prefill_one(&client, &m, &row), row_opts(4, 7))
            .expect("admit");
        assert_eq!(state.alloc_width(), 1);

        let before_kv = kv(&state);
        let before_lm = values(&state.prefill.lm_hidden);
        let before_res = values(&state.prefill.residual_hidden);
        let before_ends = state.prefill.lanes.kv_end().to_vec::<i32>();
        assert!(
            before_lm.iter().any(|v| v.abs() > 1e-6),
            "a zeroed handoff would make the comparison vacuous"
        );

        state.ensure_alloc(&client, 4).expect("grow");

        assert_eq!(state.alloc_width(), 4);
        assert_eq!(state.prefill.lanes.batch(), 4);
        assert_eq!(state.prefill.lm_hidden.shape()[0], 4);
        assert_eq!(state.prefix_feat_cond.shape()[0], 4);

        let hidden = before_lm.len();
        assert_eq!(values(&state.prefill.lm_hidden)[..hidden], before_lm[..]);
        assert_eq!(
            values(&state.prefill.residual_hidden)[..hidden],
            before_res[..]
        );
        assert!(
            values(&state.prefill.lm_hidden)[hidden..]
                .iter()
                .all(|v| *v == 0.0),
            "the rows growth adds start zeroed"
        );
        assert_eq!(
            state.prefill.lanes.kv_end().to_vec::<i32>()[..1],
            before_ends[..],
            "the seated lane keeps its key range"
        );
        assert_eq!(
            state.prefill.lanes.kv_end().to_vec::<i32>()[1..],
            [0, 0, 0],
            "the lanes growth adds are empty"
        );
        // Both caches were one row wide, so the old buffers are the prefix
        // of the grown ones.
        for (grown, before) in kv(&state).iter().zip(&before_kv) {
            assert_eq!(&grown[..before.len()], &before[..]);
        }
        assert!(state.occupied(0), "the seated slot is still seated");
        assert!(!state.occupied(1), "the slots growth added are free");
    }

    /// Growth rounds to an allowed width, stops at the ceiling, and never
    /// runs backwards.
    #[test]
    fn growth_rounds_up_caps_and_never_shrinks() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);

        let mut state = GenerateState::open(&m, 8, SLOT_CAPACITY).expect("open");
        state.ensure_alloc(&client, 3).expect("grow");
        assert_eq!(state.alloc_width(), 4, "three rows round up to four");

        state.ensure_alloc(&client, 2).expect("narrower");
        assert_eq!(state.alloc_width(), 4, "the allocation never shrinks");
        state.ensure_alloc(&client, 4).expect("same width");
        assert_eq!(state.alloc_width(), 4);

        assert!(state.ensure_alloc(&client, 9).is_err(), "past the ceiling");
        assert_eq!(state.alloc_width(), 4, "a refusal changes nothing");

        // A ceiling below the next allowed width caps the rounding.
        let mut narrow = GenerateState::open(&m, 3, SLOT_CAPACITY).expect("open");
        narrow.ensure_alloc(&client, 3).expect("grow");
        assert_eq!(narrow.alloc_width(), 3);
    }

    /// Releasing every request leaves the allocation where growth put it,
    /// while the stepped width falls back to its floor.
    #[test]
    fn releasing_shrinks_the_stepped_width_but_not_the_allocation() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let ref_feat = t(&[3, PATCH_SIZE, FEAT_DIM], 0.7, &device);
        let row = slot_rows(&ref_feat)[0];

        let mut state = GenerateState::open(&m, 8, SLOT_CAPACITY).expect("open");
        for slot in 0..3 {
            state
                .admit(
                    &client,
                    slot,
                    &prefill_one(&client, &m, &row),
                    row_opts(2, 7),
                )
                .expect("admit");
        }
        assert_eq!(state.alloc_width(), 4);
        assert_eq!(state.active_width(), 4);

        for slot in 0..3 {
            state.release(&client, slot).expect("release");
        }
        assert_eq!(state.active_width(), 1, "nothing is seated");
        assert_eq!(state.alloc_width(), 4, "the room is kept");
        assert_eq!(state.prefill.lanes.batch(), 4);
    }

    /// A slot seated into a grown state generates exactly what the same
    /// request generates alone, so no row's audio depends on the allocation
    /// having moved under it.
    #[test]
    fn growth_does_not_change_a_rows_audio() {
        let (client, device): (CpuClient, CpuDevice) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let ref_feat = t(&[3, PATCH_SIZE, FEAT_DIM], 0.7, &device);
        let row = slot_rows(&ref_feat)[0];
        let mut opts = options(1, 4);
        opts.rows = vec![row_opts(3, 7); 8];

        // Seat slot 0, grow the allocation under it, then run.
        let mut grown = GenerateState::open(&m, 8, SLOT_CAPACITY).expect("open");
        grown
            .admit(&client, 0, &prefill_one(&client, &m, &row), opts.row(0))
            .expect("admit");
        grown.ensure_alloc(&client, 4).expect("grow");
        m.patch_generator()
            .generate(&client, &mut grown, &opts)
            .expect("generate");

        let mut alone = opts.clone();
        alone.rows = Vec::new();
        alone.max_len = opts.row(0).max_len;
        alone.seed = opts.row(0).seed;
        let mut solo =
            GenerateState::start(prefill_one(&client, &m, &row), m.config).expect("start");
        m.patch_generator()
            .generate(&client, &mut solo, &alone)
            .expect("generate");

        assert_eq!(grown.patch_len[0], solo.patch_len[0]);
        assert!(grown.patch_len[0] > 0, "a silent row proves nothing");
        for (k, want) in solo.row_patches(0).expect("row").iter().enumerate() {
            let want = values(want);
            assert!(want.iter().any(|v| v.abs() > 1e-6), "degenerate patch");
            assert_eq!(values(&grown.patches[0][k]), want, "patch {k}");
        }
    }
}
