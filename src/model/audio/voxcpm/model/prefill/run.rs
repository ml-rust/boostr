//! The prefill entry points — one row or a left-padded batch, with or
//! without captured intermediates — and the tensor body they share.

use super::rows::PaddedBatch;
use super::state::{PrefillIntermediates, PrefillRow, PrefillState};
use super::tensors::{last_row, mask_var, row_audio_feat};
use crate::error::{Error, Result};
use crate::model::audio::voxcpm::minicpm4::LeftPad;
use crate::model::audio::voxcpm::model::loader::VoxCpm2Model;
use crate::model::traits::ModelClient;
use crate::ops::FlashAttentionOps;
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_add, var_cat, var_mul};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> VoxCpm2Model<R> {
    /// Prefill both LMs over the reference prefix and the prompt of ONE row.
    ///
    /// `ref_feat` and `text_token_ids` are [`PrefillRow`]'s fields.
    /// `max_length` sizes both KV caches; it must be at least `S` and should
    /// leave room for however many patches the later sampling loop will
    /// generate.
    ///
    /// This is [`prefill_batch`](Self::prefill_batch) with a single row and
    /// computes the same bytes it always has: no pad, no `kv_start`.
    /// [`PrefillState::intermediates`] is `None`; use
    /// [`prefill_capturing`](Self::prefill_capturing) to get them.
    pub fn prefill<C>(
        &self,
        client: &C,
        ref_feat: Option<&Tensor<R>>,
        text_token_ids: &[u32],
        max_length: usize,
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
        let row = PrefillRow {
            ref_feat,
            text_token_ids,
        };
        self.prefill_inner(client, &[row], max_length, false)
    }

    /// [`prefill`](Self::prefill), additionally returning the full-sequence
    /// intermediates in [`PrefillState::intermediates`].
    ///
    /// The values are the SAME tensors the plain path computes and drops —
    /// capturing them keeps them alive, it does not recompute anything.
    pub fn prefill_capturing<C>(
        &self,
        client: &C,
        ref_feat: Option<&Tensor<R>>,
        text_token_ids: &[u32],
        max_length: usize,
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
        let row = PrefillRow {
            ref_feat,
            text_token_ids,
        };
        self.prefill_inner(client, &[row], max_length, true)
    }

    /// Prefill both LMs over `rows` at once, left-padded to the longest row.
    ///
    /// Every row's caches share one `max_length`, which must be at least
    /// `S_max` and must leave room for the LONGEST patch budget any row will
    /// generate: the decode loop steps every row at the same position.
    /// Rows come back in input order, `[B, ...]`, with
    /// [`PrefillState::kv_start`] set when any row is padded.
    ///
    /// A single row, or rows of equal length, run the unpadded kernels and
    /// match [`prefill`](Self::prefill) exactly.
    pub fn prefill_batch<C>(
        &self,
        client: &C,
        rows: &[PrefillRow<'_, R>],
        max_length: usize,
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
        self.prefill_inner(client, rows, max_length, false)
    }

    /// [`prefill_batch`](Self::prefill_batch), additionally returning the
    /// `[B, S_max, _]` intermediates in [`PrefillState::intermediates`].
    pub fn prefill_batch_capturing<C>(
        &self,
        client: &C,
        rows: &[PrefillRow<'_, R>],
        max_length: usize,
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
        self.prefill_inner(client, rows, max_length, true)
    }

    fn prefill_inner<C>(
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
            .map(|row| Ok((row.t_ref(patch_size, feat_dim)?, row.text_token_ids)))
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

        // Row b's audio_feat, `[1, S_b, patch_size, feat_dim]`, its pad in
        // front; rows are stacked only when there is more than one.
        let mut per_row = Vec::with_capacity(batch);
        for (b, (row, layout)) in rows.iter().zip(&padded.layouts).enumerate() {
            let feat = row_audio_feat(
                client,
                row.ref_feat,
                layout.t_ref,
                layout.text_length,
                patch_size,
                feat_dim,
                dtype,
                device,
            )?;
            let pad = padded.pad(b);
            per_row.push(if pad == 0 {
                feat
            } else {
                let front = Var::new(
                    Tensor::<R>::zeros(&[1, pad, patch_size, feat_dim], dtype, device)?,
                    false,
                );
                var_cat(&[&front, &feat], 1, client)?
            });
        }
        let audio_feat = if batch == 1 {
            per_row.swap_remove(0)
        } else {
            let refs: Vec<&Var<R>> = per_row.iter().collect();
            var_cat(&refs, 0, client)?
        };

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

        // LAST row. In both modes this is a text position, hence un-fsq'd
        // by the blend above; do NOT fsq it again.
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

        Ok(PrefillState {
            lm_hidden,
            residual_hidden,
            base_cache,
            residual_cache,
            position: seq_len,
            batch,
            kv_start,
            intermediates,
        })
    }
}

#[cfg(test)]
mod tests {
    //! The no-reference (zero-shot) path, the one-row/batch equivalence and
    //! the left-padded batch against its rows' own single-row prefills.
    //!
    //! The with-reference path is also exercised end to end by
    //! `examples/voxcpm/eval_common.rs`'s `build_prefill_and_target`.

    use super::*;
    use crate::model::audio::voxcpm::local_dit::tests::{FEAT_DIM, PATCH_SIZE, t};
    use crate::model::audio::voxcpm::model::config::AUDIO_START_ID;
    use crate::model::audio::voxcpm::model::generate::test_support::{fixture, model};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn values(v: &Var<CpuRuntime>) -> Vec<f32> {
        v.tensor().contiguous().expect("contiguous").to_vec::<f32>()
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
        assert_eq!(prefill.position, LONG.len());
        assert!(prefill.intermediates.is_some());
        assert_eq!(prefill.batch, 1);
        assert!(prefill.kv_start.is_none());
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
        assert_eq!(prefill.position, SHORT.len());
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
            };
            let batch = m.prefill_batch(&client, &[row], 12).expect("batch");
            assert_eq!(batch.batch, 1);
            assert!(batch.kv_start.is_none());
            assert_eq!(batch.position, one.position);
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
            },
            PrefillRow {
                ref_feat: None,
                text_token_ids: &SHORT,
            },
        ];
        let max_length = 12;
        let batch = m
            .prefill_batch_capturing(&client, &rows, max_length)
            .expect("batch");
        assert_eq!(batch.batch, 2);
        assert_eq!(batch.position, 3 + 2 + LONG.len());
        let starts: Vec<i32> = batch.kv_start.as_ref().expect("padded").host.clone();
        assert_eq!(starts, [0, (batch.position - SHORT.len()) as i32]);

        let lm = values(&batch.lm_hidden);
        let res = values(&batch.residual_hidden);
        let width = lm.len() / 2;
        for (b, row) in rows.iter().enumerate() {
            let one = m
                .prefill(&client, row.ref_feat, row.text_token_ids, max_length)
                .expect("prefill");
            let (want_lm, want_res) = (values(&one.lm_hidden), values(&one.residual_hidden));
            assert!(want_lm.iter().any(|v| v.abs() > 1e-6), "degenerate row {b}");
            for (got, want) in lm[b * width..(b + 1) * width].iter().zip(&want_lm) {
                assert!((got - want).abs() < 1e-5, "row {b} lm: {got} vs {want}");
            }
            for (got, want) in res[b * width..(b + 1) * width].iter().zip(&want_res) {
                assert!(
                    (got - want).abs() < 1e-5,
                    "row {b} residual: {got} vs {want}"
                );
            }
        }

        // Same padded inputs through base_lm with NO kv_start: the short
        // row's last position now attends its zero pad rows and moves.
        let inter = batch.intermediates.as_ref().expect("captured");
        let mut cache = m.base_lm.new_kv_cache(2, max_length).expect("cache");
        let unmasked = m
            .base_lm
            .prefill(&client, &inter.combined_embed, &mut cache, None)
            .expect("unmasked prefill");
        let unmasked_last = values(&last_row(&unmasked, batch.position).expect("last row"));
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
}
