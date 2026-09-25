//! [`PatchGenerator::generate`]: drive the loop until every row of the
//! batch has finished.

use super::*;
use crate::ops::FlashAttentionOps;

impl<R: Runtime<DType = DType>> PatchGenerator<'_, R> {
    /// Run the loop until every row has stopped or hit its cap.
    ///
    /// Steps with [`step`](Self::step), so the noise comes from the row
    /// seeds. The emitted patches stay in `state.patches[b]`, each `[1,
    /// patch_size, feat_dim]`, with `state.patch_len[b]` saying how many
    /// belong to row `b` and `state.outcomes[b]` how that row ended. The
    /// return value is the whole batch's: [`GenerateOutcome::StopToken`]
    /// when every row finished on a stop token, [`GenerateOutcome::MaxLen`]
    /// when any row was TRUNCATED. Does NOT VAE-decode and does NOT write
    /// audio. Errors when `max_len` is 0, and propagates the first step
    /// error (a `position`/cache drift included) rather than continuing.
    pub fn generate<C>(
        &self,
        client: &C,
        state: &mut GenerateState<R>,
        options: &GenerateOptions,
    ) -> Result<GenerateOutcome>
    where
        C: ModelClient<R> + TypeConversionOps<R> + RandomOps<R> + HostCopyOps<R> + 'static,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + TypeConversionOps<R>
            + DequantOps<R>
            + FlashAttentionOps<R>,
    {
        options.check(state.batch)?;
        while !state.all_finished() {
            if self.step(client, state, options)? == StepOutcome::Stopped {
                break;
            }
        }
        let truncated = state.outcomes.contains(&Some(GenerateOutcome::MaxLen));
        Ok(if truncated {
            GenerateOutcome::MaxLen
        } else {
            GenerateOutcome::StopToken
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::model::audio::voxcpm::local_dit::tests::{FEAT_DIM, PATCH_SIZE, t};
    use crate::model::audio::voxcpm::model::config::AUDIO_START_ID;
    use crate::model::audio::voxcpm::model::prefill::PrefillRow;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuRuntime};

    const LONG: [u32; 3] = [11, 22, AUDIO_START_ID];
    const SHORT: [u32; 2] = [7, AUDIO_START_ID];
    /// `S_max` (8) plus the largest row cap (4) fits the 16-row RoPE table.
    const MAX_LENGTH: usize = 12;

    fn rows<'a>(ref_feat: &'a Tensor<CpuRuntime>) -> [PrefillRow<'a, CpuRuntime>; 2] {
        [
            PrefillRow {
                ref_feat: Some(ref_feat),
                text_token_ids: &LONG,
            },
            PrefillRow {
                ref_feat: None,
                text_token_ids: &SHORT,
            },
        ]
    }

    /// Generate one row alone, from its own single-row prefill.
    fn single(
        client: &CpuClient,
        m: &VoxCpm2Model<CpuRuntime>,
        row: &PrefillRow<'_, CpuRuntime>,
        opts: &GenerateOptions,
    ) -> (GenerateOutcome, GenerateState<CpuRuntime>) {
        let prefill = m
            .prefill(client, row.ref_feat, row.text_token_ids, MAX_LENGTH)
            .expect("prefill");
        let mut st = GenerateState::start(prefill, m.config).expect("start");
        let outcome = m
            .patch_generator()
            .generate(client, &mut st, opts)
            .expect("generate");
        (outcome, st)
    }

    /// Two rows of different length and different caps, generated together
    /// from a left-padded prefill, reproduce each row's own single-row run:
    /// same patch counts, same outcomes, patches equal within tolerance.
    /// The shorter-capped row keeps stepping after it finishes and its
    /// extra patch is not counted.
    #[test]
    fn padded_batch_generation_matches_single_rows() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let ref_feat = t(&[3, PATCH_SIZE, FEAT_DIM], 0.7, &device);
        let rows = rows(&ref_feat);
        let mut opts = options(1, 4);
        opts.rows = vec![
            RowOptions {
                max_len: 3,
                seed: 7,
            },
            RowOptions {
                max_len: 4,
                seed: 11,
            },
        ];

        let prefill = m.prefill_batch(&client, &rows, MAX_LENGTH).expect("batch");
        assert!(prefill.lanes.pad().is_some(), "rows differ in length");
        let mut st = GenerateState::start(prefill, m.config).expect("start");
        let outcome = m
            .patch_generator()
            .generate(&client, &mut st, &opts)
            .expect("generate");
        assert_eq!(outcome, GenerateOutcome::MaxLen);
        assert_eq!(st.emitted(), 4, "the loop runs to the longest cap");
        assert_eq!(st.patch_len, [3, 4]);
        assert_eq!(st.outcomes, [Some(GenerateOutcome::MaxLen); 2]);
        assert_eq!(st.prefill.lanes.position(0), 8 + 4);

        for (b, row) in rows.iter().enumerate() {
            let mut one = options(1, opts.rows[b].max_len);
            one.seed = opts.rows[b].seed;
            let (one_outcome, one_st) = single(&client, &m, row, &one);
            assert_eq!(one_outcome, GenerateOutcome::MaxLen);
            assert_eq!(one_st.patch_len[0], st.patch_len[b]);
            for (k, want) in one_st.row_patches(0).expect("row 0").iter().enumerate() {
                let want = values(want);
                let got = values(&st.patches[b][k]);
                assert!(want.iter().any(|v| v.abs() > 1e-6), "degenerate patch");
                for (g, w) in got.iter().zip(&want) {
                    assert!((g - w).abs() < 1e-4, "row {b} patch {k}: {g} vs {w}");
                }
            }
        }
    }

    /// Both rows stop on the same step: the step reports `Stopped`, steps
    /// 6-8 are skipped for the whole batch, and the caches stay where the
    /// previous iteration left them — the one-row semantics, per row.
    #[test]
    fn a_batch_wide_stop_leaves_the_caches_untouched() {
        let (client, device) = cpu_setup();
        let m = model(fixture(true, &device), &device);
        let ref_feat = t(&[3, PATCH_SIZE, FEAT_DIM], 0.7, &device);
        let rows = rows(&ref_feat);
        let opts = options(2, 4);

        let prefill = m.prefill_batch(&client, &rows, MAX_LENGTH).expect("batch");
        let mut st = GenerateState::start(prefill, m.config).expect("start");
        let generator = m.patch_generator();
        assert_eq!(
            generator
                .generate(&client, &mut st, &opts)
                .expect("generate"),
            GenerateOutcome::StopToken
        );
        assert_eq!(st.patch_len, [4, 4]);
        assert_eq!(st.outcomes, [Some(GenerateOutcome::StopToken); 2]);
        assert_eq!(st.prefill.lanes.position(0), 8 + 3);
        assert_eq!(st.prefill.base_cache.seq_len(), 8 + 3);
        assert!(
            generator.step(&client, &mut st, &opts).is_err(),
            "stepping a finished batch must error, not extend it"
        );

        for row in &rows {
            let (outcome, one) = single(&client, &m, row, &opts);
            assert_eq!(outcome, GenerateOutcome::StopToken);
            assert_eq!(one.patch_len[0], 4);
        }
    }

    /// The invariant both the noise index and the stop guard now read per
    /// row: a row still generating has counted exactly one patch per
    /// iteration, so `patch_len[b] == patches[b].len()`. Row 0 caps at 3 and
    /// row 1 at 4, so row 0 is frozen for the last iteration and its count
    /// falls behind while its storage keeps growing.
    #[test]
    fn a_live_rows_patch_count_is_the_loop_index() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let ref_feat = t(&[3, PATCH_SIZE, FEAT_DIM], 0.7, &device);
        let rows = rows(&ref_feat);
        let mut opts = options(1, 4);
        opts.rows = vec![
            RowOptions {
                max_len: 3,
                seed: 7,
            },
            RowOptions {
                max_len: 4,
                seed: 11,
            },
        ];

        let prefill = m.prefill_batch(&client, &rows, MAX_LENGTH).expect("batch");
        let mut st = GenerateState::start(prefill, m.config).expect("start");
        let generator = m.patch_generator();
        let mut steps = 0;
        while !st.all_finished() {
            if generator.step(&client, &mut st, &opts).expect("step") == StepOutcome::Stopped {
                break;
            }
            steps += 1;
            assert_eq!(st.emitted(), steps, "one patch per row per iteration");
            for b in 0..st.batch {
                assert_eq!(st.patches[b].len(), steps, "row {b} storage");
                assert_eq!(
                    st.patches[b][steps - 1].shape(),
                    &[1, PATCH_SIZE, FEAT_DIM],
                    "row {b} stores its own slice, not the batch"
                );
                assert!(
                    st.patches[b][steps - 1].tensor().is_contiguous(),
                    "row {b}'s dim-0 slice of a dense patch is itself dense, \
                     so nothing downstream has to copy it"
                );
                if !st.finished(b) {
                    assert_eq!(
                        st.patch_len[b],
                        st.patches[b].len(),
                        "a live row's count is the loop index"
                    );
                }
            }
        }
        assert_eq!(steps, 4);
        assert_eq!(st.patch_len, [3, 4]);
        assert_eq!(
            st.row_patches(0).expect("row 0").len(),
            3,
            "a frozen row keeps only the patches it counted"
        );
        assert_eq!(st.patches[0].len(), 4, "its storage still grew");
    }

    /// A `rows` list that does not match the batch is rejected before any
    /// device work.
    #[test]
    fn mismatched_row_options_are_rejected() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let mut st = state(&fx, &device);
        let mut opts = options(2, 8);
        opts.rows = vec![
            RowOptions {
                max_len: 8,
                seed: 1
            };
            2
        ];
        assert!(fx.generator().generate(&client, &mut st, &opts).is_err());
        assert_eq!(st.emitted(), 0);
    }
}
