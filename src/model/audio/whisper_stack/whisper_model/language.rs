//! Language detection: the single token following `<|startoftranscript|>`.

use super::mask::{allow_only_mask, greedy_pick_last};
use super::model::WhisperModel;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConditionalOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> WhisperModel<R> {
    /// Pick the most likely token among `candidates` after a `[<|sot|>]`-only
    /// prefill, which is how Whisper detects the spoken language: the decoder
    /// is asked for the token that follows start-of-transcript and every id
    /// outside the language block is masked out of the argmax.
    ///
    /// Returns the winning candidate id. `candidates` must be non-empty and
    /// every id must lie inside the vocabulary.
    pub fn detect_language<C>(
        &self,
        client: &C,
        encoder_out: &Tensor<R>,
        sot_token: u32,
        candidates: &[u32],
    ) -> Result<u32>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + NormalizationOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + UnaryOps<R>
            + ConditionalOps<R>
            + numr::ops::IndexingOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let batch = encoder_out.shape()[0];
        if batch != 1 {
            return Err(Error::ModelError {
                reason: format!("detect_language currently supports batch=1, got batch={batch}"),
            });
        }
        let vocab_size = self.decoder.vocab_size();
        let mask = allow_only_mask::<R>(candidates, vocab_size, encoder_out.device())?;

        let mut cache = self.decoder.new_cache();
        let prefix = Tensor::<R>::from_slice(&[sot_token as i64], &[1, 1], encoder_out.device())?;
        let logits =
            self.decoder
                .forward_with_cache(client, &prefix, encoder_out, 0, &mut cache)?;
        greedy_pick_last(client, &logits, Some(&mask))
    }
}
