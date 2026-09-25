//! [`GenerateState::admit`] and [`GenerateState::release`]: seating one
//! request into a free slot, and giving the slot back.

use super::super::*;
use crate::model::audio::voxcpm::model::prefill::PrefillState;

impl<R: Runtime<DType = DType>> GenerateState<R> {
    /// Seat one prefilled request into free slot `slot`.
    ///
    /// `prefilled` is what a single-row
    /// [`prefill_batch`](VoxCpm2Model::prefill_batch) produced: one row, its
    /// own two caches holding its sequence from cache position 0, and its
    /// own `[1, hidden]` handoff rows.
    ///
    /// `slot` is checked against the CEILING, and the device allocation is
    /// widened to cover it before anything is written. Every check runs
    /// first, so a handoff the slot cannot take leaves the allocation where
    /// it was.
    ///
    /// The lane is claimed LAST. Everything before it writes into slot
    /// `slot` alone, so a refusal part way through leaves the slot free and
    /// the table saying so — the slot can be seated again, and the batch
    /// steps over it meanwhile with an empty key range. A widened
    /// allocation is not a seat: it adds free rows and nothing else, so a
    /// refusal after it leaves the state usable.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `slot` is outside the ceiling, when
    /// it already holds a request, when `prefilled` carries more than one
    /// row or carries left padding, when its two caches disagree on length,
    /// or when its handoff rows are not `[1, hidden]`. A cache, copy or
    /// allocation error propagates.
    pub fn admit<C>(
        &mut self,
        client: &C,
        slot: usize,
        prefilled: &PrefillState<R>,
        options: RowOptions,
    ) -> Result<()>
    where
        C: ModelClient<R> + HostCopyOps<R>,
        R::Client: IndexingOps<R>,
    {
        if slot >= self.batch {
            return Err(Error::InvalidArgument {
                arg: "slot",
                reason: format!("expected a slot below {}, got {slot}", self.batch),
            });
        }
        if self.occupied(slot) {
            return Err(Error::InvalidArgument {
                arg: "slot",
                reason: format!(
                    "slot {slot} already holds a request; release it before seating another"
                ),
            });
        }
        // The handoff rows are stored at the ALLOCATION, which is one row
        // until a slot above it is seated, so that is the width they are
        // checked at rather than the ceiling.
        let hidden = check_row(
            "state.prefill.lm_hidden",
            &self.prefill.lm_hidden,
            self.alloc_width,
        )?;
        check_row(
            "state.prefill.residual_hidden",
            &self.prefill.residual_hidden,
            self.alloc_width,
        )?;
        let len = check_prefilled(prefilled, hidden)?;
        // Every check is done, so widening now cannot leave a grown
        // allocation behind a refusal. This slot needs a row of its own
        // before any of the writes below can address it.
        self.ensure_alloc(client, slot + 1)?;
        let (dtype, device) = (
            self.prefix_feat_cond.tensor().dtype(),
            self.prefix_feat_cond.tensor().device().clone(),
        );

        self.prefill
            .base_cache
            .install_lane(&prefilled.base_cache, slot, client)?;
        self.prefill
            .residual_cache
            .install_lane(&prefilled.residual_cache, slot, client)?;

        self.prefill.lm_hidden =
            write_row(client, &self.prefill.lm_hidden, &prefilled.lm_hidden, slot)?;
        self.prefill.residual_hidden = write_row(
            client,
            &self.prefill.residual_hidden,
            &prefilled.residual_hidden,
            slot,
        )?;

        // The reference conditions a row's FIRST patch on zeros, never on
        // whatever the previous occupant left here.
        let zeros = super::super::types::zero_patches::<R>(1, self.config_of(), dtype, &device)?;
        self.prefix_feat_cond = Var::new(
            client
                .slice_assign(self.prefix_feat_cond.tensor(), &zeros, 0, slot)
                .map_err(Error::Numr)?,
            false,
        );

        self.patches[slot] = Vec::new();
        self.patch_len[slot] = 0;
        self.outcomes[slot] = None;
        self.slot_options[slot] = Some(options);

        self.prefill.lanes.claim(client, slot, len)?;
        // The lane is claimed, so this slot now counts toward the prefix the
        // loop steps. A slot above the current stepped width widens it here,
        // which is what puts the new request inside the rows a step runs.
        self.refresh_active_width()
    }

    /// Give slot `slot` back: drop its patches, clear its bookkeeping and
    /// mark its lane free.
    ///
    /// Neither KV cache is cleared, and none is needed. The next occupant
    /// is installed from cache position 0 and reads only `[0, kv_end)`, so
    /// the keys left above its end are unreachable and the ones below it
    /// are overwritten by the install. VoxCPM2 carries no recurrent state
    /// between steps, so there is nothing else a slot can leak.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `slot` is outside the ceiling; a copy
    /// error propagates.
    pub fn release<C: HostCopyOps<R>>(&mut self, client: &C, slot: usize) -> Result<()>
    where
        R::Client: IndexingOps<R>,
    {
        if slot >= self.batch {
            return Err(Error::InvalidArgument {
                arg: "slot",
                reason: format!("expected a slot below {}, got {slot}", self.batch),
            });
        }
        if slot >= self.alloc_width {
            // The allocation never reached this slot, so nothing was ever
            // seated in it and there is no lane to give back. Releasing it
            // is the no-op releasing any free slot is, and it must not
            // widen the allocation to say so.
            return Ok(());
        }
        // Dropped here rather than at the end of the batch: a finished
        // request's patches are the largest thing a slot holds.
        self.patches[slot] = Vec::new();
        self.patch_len[slot] = 0;
        self.outcomes[slot] = None;
        self.slot_options[slot] = None;
        self.prefill.lanes.free(client, slot)?;
        // Releasing the HIGHEST occupied slot shortens the prefix, so the
        // loop stops paying for it. Releasing one below it changes nothing:
        // the prefix still has to reach the highest slot still seated.
        self.refresh_active_width()
    }

    /// The patch geometry `prefix_feat_cond` was allocated at.
    fn config_of(&self) -> VoxCpm2Config {
        let shape = self.prefix_feat_cond.shape();
        VoxCpm2Config {
            patch_size: shape[1],
            feat_dim: shape[2],
        }
    }
}

/// `prefilled`'s sequence length, after checking it is the one-row handoff
/// a slot `hidden` wide can take.
///
/// Every check runs before `admit` writes anything, so a handoff the slot
/// cannot take leaves both caches untouched.
fn check_prefilled<R: Runtime<DType = DType>>(
    prefilled: &PrefillState<R>,
    hidden: usize,
) -> Result<usize>
where
    R::Client: IndexingOps<R>,
{
    if prefilled.batch != 1 {
        return Err(Error::InvalidArgument {
            arg: "prefilled.batch",
            reason: format!("expected 1 row, got {}", prefilled.batch),
        });
    }
    if prefilled.lanes.pad().is_some() {
        return Err(Error::InvalidArgument {
            arg: "prefilled.lanes",
            reason: "expected an unpadded prefill: a padded row's keys do not start at cache \
                     position 0, so they cannot be installed into a lane"
                .to_string(),
        });
    }
    let len = prefilled.base_cache.seq_len();
    let residual = prefilled.residual_cache.seq_len();
    if residual != len {
        return Err(Error::InvalidArgument {
            arg: "prefilled",
            reason: format!(
                "the two caches hold different lengths: base {len}, residual {residual}"
            ),
        });
    }
    if prefilled.lanes.position(0) != len {
        return Err(Error::InvalidArgument {
            arg: "prefilled.lanes",
            reason: format!(
                "the lane table sits at {} but the caches hold {len} positions",
                prefilled.lanes.position(0)
            ),
        });
    }
    if len == 0 {
        return Err(Error::InvalidArgument {
            arg: "prefilled",
            reason: "expected at least 1 prefilled position, got 0".to_string(),
        });
    }
    for (arg, row) in [
        ("prefilled.lm_hidden", &prefilled.lm_hidden),
        ("prefilled.residual_hidden", &prefilled.residual_hidden),
    ] {
        let width = check_row(arg, row, 1)?;
        if width != hidden {
            return Err(Error::InvalidArgument {
                arg: "prefilled",
                reason: format!(
                    "expected a handoff {hidden} wide, the hidden size this state was \
                     opened at, got {width}"
                ),
            });
        }
    }
    Ok(len)
}

/// `rows` with its row `slot` replaced by `src`'s single row. Both shapes
/// are checked by `check_prefilled` before any write happens.
fn write_row<R, C>(client: &C, rows: &Var<R>, src: &Var<R>, slot: usize) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: IndexingOps<R>,
{
    Ok(Var::new(
        client
            .slice_assign(rows.tensor(), src.tensor(), 0, slot)
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

    /// A released slot drops its patches and can be seated again.
    #[test]
    fn release_frees_a_slot_and_drops_its_patches() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let ref_feat = t(&[3, PATCH_SIZE, FEAT_DIM], 0.7, &device);
        let row = slot_rows(&ref_feat)[0];

        let mut state = GenerateState::open(&m, 2, SLOT_CAPACITY).expect("open");
        state
            .admit(&client, 1, &prefill_one(&client, &m, &row), row_opts(2, 7))
            .expect("admit");
        let mut opts = options(0, 2);
        opts.rows = vec![row_opts(1, 7), row_opts(2, 7)];
        m.patch_generator()
            .generate(&client, &mut state, &opts)
            .expect("generate");
        assert!(!state.patches[1].is_empty());

        state.release(&client, 1).expect("release");
        assert!(state.patches[1].is_empty(), "its patches must be dropped");
        assert_eq!(state.patch_len[1], 0);
        assert_eq!(state.outcomes[1], None);
        assert_eq!(state.slot_options[1], None);
        assert!(!state.occupied(1));
        assert_eq!(state.prefill.lanes.kv_end().to_vec::<i32>(), vec![0, 0]);

        state
            .admit(&client, 1, &prefill_one(&client, &m, &row), row_opts(2, 7))
            .expect("the slot is reusable");
        assert!(state.occupied(1));
        assert_eq!(state.slot_options[1], Some(row_opts(2, 7)));
    }

    /// A seated slot is conditioned on ZEROS whatever the slot held before,
    /// and only that slot's row moves.
    #[test]
    fn admit_zeroes_the_slots_prefix_condition() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let ref_feat = t(&[3, PATCH_SIZE, FEAT_DIM], 0.7, &device);
        let row = slot_rows(&ref_feat)[0];
        let width = PATCH_SIZE * FEAT_DIM;

        let mut state = GenerateState::open(&m, 2, SLOT_CAPACITY).expect("open");
        // Whatever a previous occupant left in slot 1's condition row.
        state.prefix_feat_cond = Var::new(t(&[2, PATCH_SIZE, FEAT_DIM], 3.5, &device), false);
        let before = values(&state.prefix_feat_cond);
        assert!(before[width..].iter().any(|v| v.abs() > 1e-6));

        state
            .admit(&client, 1, &prefill_one(&client, &m, &row), row_opts(2, 7))
            .expect("admit");

        let after = values(&state.prefix_feat_cond);
        assert!(
            after[width..].iter().all(|v| *v == 0.0),
            "the seated slot must be conditioned on zeros"
        );
        assert_eq!(
            &after[..width],
            &before[..width],
            "the other slot must not move"
        );
    }

    /// An occupied slot, a slot outside the width, and a padded handoff are
    /// each refused, and a refusal writes nothing.
    #[test]
    fn a_seat_the_state_cannot_take_is_refused() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let ref_feat = t(&[3, PATCH_SIZE, FEAT_DIM], 0.7, &device);
        let rows = slot_rows(&ref_feat);

        let mut state = GenerateState::open(&m, 1, SLOT_CAPACITY).expect("open");
        state
            .admit(
                &client,
                0,
                &prefill_one(&client, &m, &rows[0]),
                row_opts(4, 7),
            )
            .expect("admit");
        let held = values(&state.prefill.lm_hidden);

        let err = state
            .admit(
                &client,
                0,
                &prefill_one(&client, &m, &rows[1]),
                row_opts(4, 9),
            )
            .expect_err("slot 0 is taken");
        assert!(err.to_string().contains("slot 0"), "{err}");
        assert_eq!(values(&state.prefill.lm_hidden), held);
        assert_eq!(state.slot_options[0], Some(row_opts(4, 7)));
        assert!(
            state
                .admit(
                    &client,
                    1,
                    &prefill_one(&client, &m, &rows[1]),
                    row_opts(4, 9)
                )
                .is_err()
        );
        assert!(state.release(&client, 1).is_err());

        let padded = m
            .prefill_batch(&client, &rows, SLOT_CAPACITY)
            .expect("batch");
        assert!(padded.lanes.pad().is_some(), "rows differ in length");
        let mut wide = GenerateState::open(&m, 2, SLOT_CAPACITY).expect("open");
        assert!(wide.admit(&client, 0, &padded, row_opts(4, 7)).is_err());
        assert!(!wide.occupied(0), "a refused admit leaves the slot free");
    }
}
