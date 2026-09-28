//! The prefill body every entry point in `entry` runs.

use super::rows::PaddedBatch;
use super::state::{PrefillIntermediates, PrefillRow, PrefillState};
use super::tensors::{batch_audio_feat, last_patch, last_row, mask_var};
use crate::error::{Error, Result};
use crate::model::audio::voxcpm::minicpm4::{LanePositions, LeftPad};
use crate::model::audio::voxcpm::model::loader::VoxCpm2Model;
use crate::model::traits::ModelClient;
use crate::ops::FlashAttentionOps;
use crate::quant::traits::DequantOps;
use numr::autograd::{var_add, var_cat, var_mul};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> VoxCpm2Model<R> {
    /// Prefill both LMs over `rows`, left-padded to the longest row, and
    /// capture the intermediates when `capture` is set.
    pub(super) fn prefill_inner<C>(
        &self,
        client: &C,
        rows: &[PrefillRow<'_, R>],
        max_length: usize,
        capture: bool,
    ) -> Result<PrefillState<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R> + DequantOps<R> + 'static,
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
            + DequantOps<R>
            + FlashAttentionOps<R>,
    {
        let (patch_size, feat_dim) = (self.config.patch_size, self.config.feat_dim);
        let host_rows = rows
            .iter()
            .map(|row| {
                Ok((
                    row.t_ref(patch_size, feat_dim)?,
                    row.text_token_ids,
                    row.t_prompt(patch_size, feat_dim)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let padded = PaddedBatch::build(&host_rows)?;
        let (batch, seq_len) = (padded.batch(), padded.seq_len);
        if max_length < seq_len {
            return Err(Error::InvalidArgument {
                arg: "max_length",
                reason: format!(
                    "expected at least the prefill length S ({seq_len}), got {max_length}"
                ),
            });
        }

        let (dtype, device) = self.lm_dtype_device()?;

        let (audio_feat, context) =
            batch_audio_feat(client, rows, &padded, patch_size, feat_dim, dtype, device)?;
        // The reference's `prefix_feat_cond = feat[:, -1]`: position S_max - 1
        // is every row's last real one under left padding.
        let prefix_feat_cond = last_patch(&audio_feat, seq_len)?;

        // feat_encoder runs over ALL S rows, text rows included — their
        // patches are zeros, not absent.
        let encoded = self.feat_encoder.forward(client, &audio_feat)?;
        let feat_embed = self.aux.enc_to_lm_proj.forward(client, &encoded)?;

        // `scale_emb` is 1.0 on this checkpoint (muP off), so the lookup is
        // UNSCALED — `MiniCpm4Model::embed` already leaves it alone.
        let ids = Tensor::<R>::from_slice(&padded.token_ids, &[batch, seq_len], device)?;
        let text_embed = self.base_lm.embed(client, &ids)?;

        let text_mask = mask_var::<R>(&padded.text_mask, batch, seq_len, dtype, device)?;
        let audio_mask = mask_var::<R>(&padded.audio_mask, batch, seq_len, dtype, device)?;

        // The two masks are complementary at every real position (checked in
        // `SequenceLayout`), so this SUM picks exactly one term there; a pad
        // position is zero in both and contributes nothing.
        let masked_feat = var_mul(&feat_embed, &audio_mask, client)?;
        let combined_embed = var_add(
            &var_mul(&text_embed, &text_mask, client)?,
            &masked_feat,
            client,
        )?;

        let kv_start = padded
            .is_padded()
            .then(|| LeftPad::<R>::new(padded.kv_start.clone(), device))
            .transpose()?;

        let mut base_cache = self.base_lm.new_kv_cache(batch, max_length)?;
        let enc =
            self.base_lm
                .prefill(client, &combined_embed, &mut base_cache, kv_start.as_ref())?;

        // fsq on AUDIO positions, identity on TEXT positions.
        let enc_outputs = var_add(
            &var_mul(&self.fsq.forward(client, &enc)?, &audio_mask, client)?,
            &var_mul(&enc, &text_mask, client)?,
            client,
        )?;

        // LAST row, already blended above: un-fsq'd at a text position,
        // fsq'd at a continuation tail's audio position. Do NOT fsq it again.
        let lm_hidden = last_row(&enc_outputs, seq_len)?;

        // Argument order is (enc_outputs, masked feat_embed) — and
        // `enc_outputs` is NOT masked again here.
        let fused = var_cat(&[&enc_outputs, &masked_feat], 2, client)?;
        let residual_enc_inputs = self.aux.fusion_concat_proj.forward(client, &fused)?;

        let mut residual_cache = self.residual_lm.new_kv_cache(batch, max_length)?;
        let residual_out = self.residual_lm.prefill(
            client,
            &residual_enc_inputs,
            &mut residual_cache,
            kv_start.as_ref(),
        )?;
        let residual_hidden = last_row(&residual_out, seq_len)?;

        let intermediates = capture.then(|| PrefillIntermediates {
            combined_embed,
            enc_outputs,
            feat_embed,
            residual_enc_inputs,
        });

        // ONE table for both caches.
        let rope_table_len = self.lane_rope_positions()?;
        let lanes = LanePositions::new(kv_start, batch, seq_len, rope_table_len, device)?;

        Ok(PrefillState {
            lm_hidden,
            residual_hidden,
            base_cache,
            residual_cache,
            lanes,
            batch,
            intermediates,
            prefix_feat_cond: Some(prefix_feat_cond),
            context,
        })
    }
}

#[cfg(test)]
mod tests {
    //! The no-reference (zero-shot) path, the one-row/batch equivalence and
    //! the left-padded batch, mixed modes included, against its rows' own
    //! single-row prefills.
    //!
    //! The with-reference path is also exercised end to end by
    //! `examples/voxcpm/eval_common.rs`'s `build_prefill_and_target`.

    use super::*;
    use crate::model::audio::voxcpm::local_dit::tests::{FEAT_DIM, PATCH_SIZE, t};
    use crate::model::audio::voxcpm::model::config::AUDIO_START_ID;
    use crate::model::audio::voxcpm::model::generate::test_support::{fixture, model};
    use crate::test_utils::cpu_setup;
    use numr::autograd::Var;
    use numr::runtime::cpu::{CpuClient, CpuRuntime};

    fn values(v: &Var<CpuRuntime>) -> Vec<f32> {
        v.tensor().contiguous().expect("contiguous").to_vec::<f32>()
    }

    fn patch(feat: &Tensor<CpuRuntime>, index: usize) -> Vec<f32> {
        feat.narrow(0, index, 1)
            .expect("narrow")
            .contiguous()
            .expect("contiguous")
            .to_vec()
    }

    fn assert_close(got: &[f32], want: &[f32], what: &str, row: usize) {
        assert_eq!(got.len(), want.len(), "row {row} {what} width");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-5, "row {row} {what}: {g} vs {w}");
        }
    }

    /// Row `b` of `batch` carries what its own single-row prefill carries:
    /// both handoff rows, the first patch's condition and the decode context.
    fn assert_rows_match_own_prefill(
        client: &CpuClient,
        m: &VoxCpm2Model<CpuRuntime>,
        rows: &[PrefillRow<'_, CpuRuntime>],
        batch: &PrefillState<CpuRuntime>,
        max_length: usize,
    ) {
        let cond_of = |state: &PrefillState<CpuRuntime>| {
            values(
                state
                    .prefix_feat_cond
                    .as_ref()
                    .expect("prefill sets the condition"),
            )
        };
        let (lm, res, cond) = (
            values(&batch.lm_hidden),
            values(&batch.residual_hidden),
            cond_of(batch),
        );
        let (width, cond_width) = (lm.len() / rows.len(), cond.len() / rows.len());
        for (b, row) in rows.iter().enumerate() {
            let one = m
                .prefill_batch(client, &[*row], max_length)
                .expect("prefill");
            let want_lm = values(&one.lm_hidden);
            assert!(want_lm.iter().any(|v| v.abs() > 1e-6), "degenerate row {b}");
            let own = |all: &[f32], w: usize| all[b * w..(b + 1) * w].to_vec();
            assert_close(&own(&lm, width), &want_lm, "lm", b);
            assert_close(
                &own(&res, width),
                &values(&one.residual_hidden),
                "residual",
                b,
            );
            assert_close(
                &own(&cond, cond_width),
                &cond_of(&one),
                "prefix_feat_cond",
                b,
            );
            assert_eq!(
                batch.context[b].len(),
                one.context[0].len(),
                "row {b} context"
            );
            for (got, want) in batch.context[b].iter().zip(&one.context[0]) {
                assert_close(&values(got), &values(want), "context", b);
            }
        }
    }

    const LONG: [u32; 3] = [11, 22, AUDIO_START_ID];
    const SHORT: [u32; 2] = [7, AUDIO_START_ID];

    /// `prefill_capturing(client, None, ...)` must succeed, land `position`
    /// exactly at `text_token_ids.len()` (no reference prefix contributes rows),
    /// and always populate `intermediates` — the training loop's
    /// `cfm_loss`/`train_losses_with_noise` require `Some` there.
    #[test]
    fn no_reference_prefill_capturing_succeeds() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let prefill = m
            .prefill_capturing(&client, None, &LONG, LONG.len())
            .expect("no-reference prefill_capturing");
        assert_eq!(prefill.lanes.position(0), LONG.len());
        assert!(prefill.intermediates.is_some());
        assert_eq!(prefill.batch, 1);
        assert!(prefill.lanes.pad().is_none());
    }

    /// The plain (non-capturing) path must agree on `position` and leave
    /// `intermediates` empty, same as the with-reference path already does.
    #[test]
    fn no_reference_prefill_leaves_intermediates_empty() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let prefill = m
            .prefill(&client, None, &SHORT, SHORT.len())
            .expect("no-reference prefill");
        assert_eq!(prefill.lanes.position(0), SHORT.len());
        assert!(prefill.intermediates.is_none());
    }

    /// `prefill` IS `prefill_batch` with one row: same bytes out, no pad, no
    /// `kv_start`, in both modes.
    #[test]
    fn single_row_batch_is_byte_identical_to_prefill() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let ref_feat = t(&[3, PATCH_SIZE, FEAT_DIM], 0.7, &device);
        for ref_feat in [Some(&ref_feat), None] {
            let one = m.prefill(&client, ref_feat, &LONG, 12).expect("prefill");
            let row = PrefillRow {
                ref_feat,
                text_token_ids: &LONG,
                prompt_feat: None,
            };
            let batch = m.prefill_batch(&client, &[row], 12).expect("batch");
            assert_eq!(batch.batch, 1);
            assert!(batch.lanes.pad().is_none());
            assert_eq!(batch.lanes.position(0), one.lanes.position(0));
            assert_eq!(values(&batch.lm_hidden), values(&one.lm_hidden));
            assert_eq!(values(&batch.residual_hidden), values(&one.residual_hidden));
            assert_eq!(batch.base_cache.seq_len(), one.base_cache.seq_len());
        }
    }

    /// Two rows of different length: each row's handoff equals its own
    /// single-row prefill, and dropping the mask over the same padded inputs
    /// moves the short row — so the agreement is the mask's doing.
    #[test]
    fn padded_batch_matches_each_rows_own_prefill() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let ref_feat = t(&[3, PATCH_SIZE, FEAT_DIM], 0.7, &device);
        let rows = [
            PrefillRow {
                ref_feat: Some(&ref_feat),
                text_token_ids: &LONG,
                prompt_feat: None,
            },
            PrefillRow {
                ref_feat: None,
                text_token_ids: &SHORT,
                prompt_feat: None,
            },
        ];
        let max_length = 12;
        let batch = m
            .prefill_batch_capturing(&client, &rows, max_length)
            .expect("batch");
        assert_eq!(batch.batch, 2);
        assert_eq!(batch.lanes.position(0), 3 + 2 + LONG.len());
        let starts: Vec<i32> = batch.lanes.pad().expect("padded").host.clone();
        assert_eq!(starts, [0, (batch.lanes.position(1) - SHORT.len()) as i32]);

        assert_rows_match_own_prefill(&client, &m, &rows, &batch, max_length);
        let lm = values(&batch.lm_hidden);
        let width = lm.len() / 2;

        // Same padded inputs through base_lm with NO kv_start: the short
        // row's last position now attends its zero pad rows and moves.
        let inter = batch.intermediates.as_ref().expect("captured");
        let mut cache = m.base_lm.new_kv_cache(2, max_length).expect("cache");
        let unmasked = m
            .base_lm
            .prefill(&client, &inter.combined_embed, &mut cache, None)
            .expect("unmasked prefill");
        let unmasked_last =
            values(&last_row(&unmasked, batch.lanes.position(0)).expect("last row"));
        assert!(
            unmasked_last[width..]
                .iter()
                .zip(&lm[width..])
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "kv_start changed nothing for the padded row"
        );
        assert_eq!(
            &unmasked_last[..width],
            &lm[..width],
            "the unpadded row must not see any difference"
        );
    }

    /// A batch mixing all four modes — reference, zero-shot,
    /// ref+continuation, continuation-only — gives every row its own
    /// single-row prefill. A tail row conditions its first patch on the
    /// prompt's last patch, and its decode context counts every audio
    /// position, reaching into the reference when the tail is short.
    #[test]
    fn mixed_mode_batch_matches_each_rows_own_prefill() {
        let (client, device) = cpu_setup();
        let m = model(fixture(false, &device), &device);
        let ref_feat = t(&[3, PATCH_SIZE, FEAT_DIM], 0.7, &device);
        let prompt_feat = t(&[2, PATCH_SIZE, FEAT_DIM], 1.9, &device);
        let row = |ref_feat, text_token_ids, prompt_feat| PrefillRow {
            ref_feat,
            text_token_ids,
            prompt_feat,
        };
        let rows = [
            row(Some(&ref_feat), &LONG[..], None),
            row(None, &SHORT[..], None),
            row(Some(&ref_feat), &SHORT[..], Some(&prompt_feat)),
            row(None, &LONG[..], Some(&prompt_feat)),
        ];
        let max_length = 12;
        let batch = m.prefill_batch(&client, &rows, max_length).expect("batch");
        assert_eq!(batch.lanes.position(0), 3 + 2 + SHORT.len() + 2);
        assert_rows_match_own_prefill(&client, &m, &rows, &batch, max_length);

        let cond = values(batch.prefix_feat_cond.as_ref().expect("condition"));
        let width = PATCH_SIZE * FEAT_DIM;
        let last_prompt = patch(&prompt_feat, 1);
        assert!(
            cond[..2 * width].iter().all(|v| *v == 0.0),
            "no tail: zeros"
        );
        assert_eq!(&cond[2 * width..3 * width], last_prompt.as_slice());
        assert_eq!(&cond[3 * width..], last_prompt.as_slice());

        let lens: Vec<usize> = batch.context.iter().map(Vec::len).collect();
        assert_eq!(lens, [0, 0, 3, 2]);
        let want = [patch(&ref_feat, 2), patch(&prompt_feat, 0), last_prompt];
        for (got, want) in batch.context[2].iter().zip(&want) {
            assert_eq!(&values(got), want);
        }
        for (got, want) in batch.context[3].iter().zip(&want[1..]) {
            assert_eq!(&values(got), want);
        }
    }
}
