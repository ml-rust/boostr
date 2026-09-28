//! The prefill entry points: one row or a left-padded batch, with or
//! without captured intermediates. Each one runs the shared body in `run`.

use super::state::{PrefillRow, PrefillState};
use crate::error::Result;
use crate::model::audio::voxcpm::model::loader::VoxCpm2Model;
use crate::model::traits::ModelClient;
use crate::ops::FlashAttentionOps;
use crate::quant::traits::DequantOps;
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
    /// `max_length` sizes both KV caches. It must be at least `S` and must
    /// leave room for every patch the later sampling loop generates.
    ///
    /// This is [`prefill_batch`](Self::prefill_batch) with a single row and
    /// no continuation tail: no pad, no `kv_start`. A continuation row goes
    /// through `prefill_batch` with [`PrefillRow::prompt_feat`] set.
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
            prompt_feat: None,
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
            prompt_feat: None,
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
}
