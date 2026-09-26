//! [`GenerateState::open`]: an empty state at a slot CEILING, allocated for
//! one row.

use super::super::*;
use crate::model::audio::voxcpm::minicpm4::LanePositions;
use crate::model::audio::voxcpm::model::prefill::PrefillState;

impl<R: Runtime<DType = DType>> GenerateState<R> {
    /// Open an empty state that can hold up to `width` slots, every slot
    /// free.
    ///
    /// `width` is the CEILING, not a reservation. Every DEVICE tensor — both
    /// KV caches, `lm_hidden`, `residual_hidden`, `prefix_feat_cond` and the
    /// lane table — is allocated for ONE row here and widened by
    /// [`admit`](Self::admit) when a slot above the allocation is seated. The
    /// HOST per-row vectors are sized at the ceiling, which is what a slot
    /// index is checked against.
    ///
    /// Both KV caches hold `capacity` cache positions, so `capacity` must
    /// cover the longest prefill this state will host plus that request's
    /// patch budget.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `width` is 0 or `capacity` is
    /// outside what either stack's cache and rope table hold;
    /// [`Error::ModelError`] when the two stacks disagree on their rope
    /// table length. An allocation error propagates.
    pub fn open(model: &VoxCpm2Model<R>, width: usize, capacity: usize) -> Result<Self>
    where
        R::Client: IndexingOps<R>,
    {
        if width == 0 {
            return Err(Error::InvalidArgument {
                arg: "width",
                reason: "expected at least 1 slot, got 0".to_string(),
            });
        }
        let (dtype, device) = model.lm_dtype_device()?;
        let hidden = model.base_lm.hidden_size();
        // ONE row, whatever the ceiling. `admit` widens the allocation to
        // cover a slot before it seats anything into it.
        let alloc = 1;
        let rows = || {
            Tensor::<R>::zeros(&[alloc, hidden], dtype, device)
                .map(|t| Var::new(t, false))
                .map_err(Error::Numr)
        };
        let prefill = PrefillState {
            lm_hidden: rows()?,
            residual_hidden: rows()?,
            base_cache: model.base_lm.new_kv_cache(alloc, capacity)?,
            residual_cache: model.residual_lm.new_kv_cache(alloc, capacity)?,
            lanes: LanePositions::empty(alloc, model.lane_rope_positions()?, device)?,
            batch: alloc,
            intermediates: None,
        };
        let mut state = Self {
            prefill,
            prefix_feat_cond: Var::new(
                super::super::types::zero_patches(alloc, model.config, dtype, device)?,
                false,
            ),
            patches: (0..width).map(|_| Vec::new()).collect(),
            batch: width,
            alloc_width: alloc,
            active_width: alloc,
            outcomes: vec![None; width],
            patch_len: vec![0; width],
            slot_options: vec![None; width],
        };
        // Nothing is seated yet, so the stepped width is the floor. Run the
        // one rule that decides the width rather than repeating its answer.
        state.refresh_active_width()?;
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::*;
    use super::*;
    use crate::model::audio::voxcpm::local_dit::tests::{FEAT_DIM, PATCH_SIZE, t};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// A state seated through `open` + `admit` carries, field for field,
    /// what `start` over the same single-row prefill carries.
    #[test]
    fn an_admitted_slot_matches_what_start_builds() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let ref_feat = t(&[3, PATCH_SIZE, FEAT_DIM], 0.7, &device);
        let row = slot_rows(&ref_feat)[0];

        let mut seated = GenerateState::open(&m, 1, SLOT_CAPACITY).expect("open");
        seated
            .admit(&client, 0, &prefill_one(&client, &m, &row), row_opts(4, 7))
            .expect("admit");
        let started =
            GenerateState::start(prefill_one(&client, &m, &row), m.config).expect("start");

        assert_eq!(seated.batch, started.batch);
        assert_eq!(seated.outcomes, started.outcomes);
        assert_eq!(seated.patch_len, started.patch_len);
        assert_eq!(
            seated.prefill.lanes.positions(),
            started.prefill.lanes.positions()
        );
        assert_eq!(
            seated.prefill.lanes.kv_end().to_vec::<i32>(),
            started.prefill.lanes.kv_end().to_vec::<i32>()
        );
        assert_eq!(
            values(&seated.prefix_feat_cond),
            values(&started.prefix_feat_cond)
        );
        assert!(
            values(&started.prefill.lm_hidden)
                .iter()
                .any(|v| v.abs() > 1e-6),
            "a degenerate handoff would make the comparisons vacuous"
        );
        assert_eq!(
            values(&seated.prefill.lm_hidden),
            values(&started.prefill.lm_hidden)
        );
        assert_eq!(
            values(&seated.prefill.residual_hidden),
            values(&started.prefill.residual_hidden)
        );
        let kv = |st: &GenerateState<CpuRuntime>| -> Vec<Vec<f32>> {
            (0..st.prefill.base_cache.num_layers())
                .filter_map(|l| st.prefill.base_cache.layer(l))
                .flat_map(|c| {
                    [
                        c.k_cache_raw().to_vec::<f32>(),
                        c.v_cache_raw().to_vec::<f32>(),
                    ]
                })
                .collect()
        };
        assert_eq!(
            seated.prefill.base_cache.seq_len(),
            started.prefill.base_cache.seq_len()
        );
        assert_eq!(kv(&seated), kv(&started));
    }

    /// Two requests seated into a four-slot state generate exactly what
    /// each generates alone, and the two free slots neither end the loop
    /// early nor hold it open.
    #[test]
    fn a_free_slot_neither_blocks_nor_perturbs_the_batch() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let ref_feat = t(&[3, PATCH_SIZE, FEAT_DIM], 0.7, &device);
        let rows = slot_rows(&ref_feat);
        let mut opts = options(1, 4);
        opts.rows = vec![
            row_opts(3, 7),
            row_opts(4, 11),
            row_opts(1, 7),
            row_opts(1, 7),
        ];

        let mut wide = GenerateState::open(&m, 4, SLOT_CAPACITY).expect("open");
        for (b, row) in rows.iter().enumerate() {
            wide.admit(&client, b, &prefill_one(&client, &m, row), opts.row(b))
                .expect("admit");
        }
        assert!(wide.occupied(0) && wide.occupied(1));
        assert!(!wide.occupied(2) && !wide.occupied(3));
        assert!(wide.finished(2), "a free slot has nothing to generate");
        assert!(!wide.all_finished(), "two slots are still open");
        // Two seated slots, so two rows allocated. The ceiling's other two
        // cost nothing until something is seated into them.
        assert_eq!(wide.alloc_width(), 2);
        assert_eq!(wide.prefill.lanes.kv_end().to_vec::<i32>().len(), 2);

        m.patch_generator()
            .generate(&client, &mut wide, &opts)
            .expect("generate");
        assert_eq!(wide.patch_len[0], 3);
        assert_eq!(wide.patch_len[1], 4);
        assert_eq!(wide.patch_len[2], 0, "a free slot counts no patch");

        for (b, row) in rows.iter().enumerate() {
            let mut alone = opts.clone();
            alone.rows = Vec::new();
            alone.max_len = opts.row(b).max_len;
            alone.seed = opts.row(b).seed;
            let mut solo =
                GenerateState::start(prefill_one(&client, &m, row), m.config).expect("start");
            m.patch_generator()
                .generate(&client, &mut solo, &alone)
                .expect("generate");
            assert_eq!(solo.patch_len[0], wide.patch_len[b]);
            for (k, want) in solo.row_patches(0).expect("row").iter().enumerate() {
                let want = values(want);
                assert!(want.iter().any(|v| v.abs() > 1e-6), "degenerate patch");
                for (g, w) in values(&wide.patches[b][k]).iter().zip(&want) {
                    assert!((g - w).abs() < 1e-4, "slot {b} patch {k}: {g} vs {w}");
                }
            }
        }
    }
}
