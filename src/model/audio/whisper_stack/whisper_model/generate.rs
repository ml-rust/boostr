//! Greedy decoding for [`WhisperModel`], plus a capturing variant that
//! records decoder cross-attention probabilities for word-level timestamp
//! alignment (HuggingFace's `_extract_token_timestamps` runs a DTW over
//! exactly these weights).

use super::mask::{greedy_pick_last, suppression_mask};
use super::model::WhisperModel;
use crate::error::{Error, Result};
use crate::model::audio::whisper_decoder::CrossAttentionCapture;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConditionalOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Options controlling a greedy decode run.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    /// Maximum number of tokens to produce (not counting the prefix).
    pub max_new_tokens: usize,
    /// Token IDs that end generation immediately (e.g. `eos_token_id`,
    /// `<|endoftext|>`). Also used for Whisper's `<|nospeech|>` in callers that want to stop there.
    pub eos_token_ids: Vec<u32>,
    /// Token IDs that are never allowed to be emitted (logit suppression).
    /// Applied at **every** generated position. Whisper's `<|notimestamps|>` /
    /// language tokens are usually part of the prefix, not the output, but some
    /// decoders use suppression to skip them.
    pub suppress_tokens: Vec<u32>,
    /// Token IDs suppressed at the **first** generated position only — the step
    /// right after the prefix. Whisper uses this to forbid a leading space
    /// (`220`) and an immediate `<|endoftext|>`. At step 0 the effective mask is
    /// the union of this list and [`Self::suppress_tokens`]; afterwards only
    /// [`Self::suppress_tokens`] applies.
    pub begin_suppress_tokens: Vec<u32>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 448,
            eos_token_ids: Vec::new(),
            suppress_tokens: Vec::new(),
            begin_suppress_tokens: Vec::new(),
        }
    }
}

/// Cross-attention probabilities captured from selected decoder layers by
/// [`WhisperModel::generate_with_cross_attentions`].
///
/// `layers` pairs a requested (0-indexed) decoder layer with its captured
/// tensor `[B, H, T_total, S]`, in the order the layers were requested. Row
/// `t` along dim 2 is sequence position `t` of the **full** decode: the
/// prefix rows first, in prefix order, then one row per generated token, in
/// generation order. This matches HuggingFace's `_extract_token_timestamps`,
/// which indexes cross-attention by full-sequence position, not just the
/// generated suffix. `S` is the encoder output's sequence length.
pub struct CrossAttentionWeights<R: Runtime> {
    pub layers: Vec<(usize, Tensor<R>)>,
}

impl<R: Runtime<DType = DType>> WhisperModel<R> {
    /// Greedy decode starting from `start_tokens` (e.g. Whisper's SOT prompt
    /// `[<|startoftranscript|>, <|lang|>, <|transcribe|>, <|notimestamps|>]`).
    ///
    /// Assumes `batch = 1`. Returns the **generated** tokens only, not including
    /// the prefix. Stops on any `options.eos_token_ids` or when `max_new_tokens`
    /// is reached.
    pub fn generate<C>(
        &self,
        client: &C,
        encoder_out: &Tensor<R>,
        start_tokens: &[u32],
        options: &GenerateOptions,
    ) -> Result<Vec<u32>>
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
        self.generate_impl(client, encoder_out, start_tokens, options, None)
    }

    /// Same greedy decode as [`Self::generate`], plus the requested decoder
    /// layers' cross-attention probabilities over the full sequence.
    ///
    /// `layers`: 0-indexed decoder layer indices to capture. Every index must
    /// be `< self.decoder.num_layers()`, checked before decoding starts, so a
    /// bad index fails fast instead of after paying for a full decode.
    ///
    /// Returns the same tokens [`Self::generate`] returns, plus one tensor per
    /// requested layer, `cat` once on device from the per-step tensors
    /// collected during decoding — cross-attention never crosses to the
    /// host. See [`CrossAttentionWeights`] for the row layout.
    pub fn generate_with_cross_attentions<C>(
        &self,
        client: &C,
        encoder_out: &Tensor<R>,
        start_tokens: &[u32],
        options: &GenerateOptions,
        layers: &[usize],
    ) -> Result<(Vec<u32>, CrossAttentionWeights<R>)>
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
        let num_layers = self.decoder.num_layers();
        for &l in layers {
            if l >= num_layers {
                return Err(Error::ModelError {
                    reason: format!(
                        "cross-attention layer index {l} is out of range: decoder has {num_layers} layers"
                    ),
                });
            }
        }

        let mut capture = CrossAttentionCapture::new(num_layers, layers);
        let tokens = self.generate_impl(
            client,
            encoder_out,
            start_tokens,
            options,
            Some(&mut capture),
        )?;

        let mut captured = Vec::with_capacity(layers.len());
        for &l in layers {
            // `l < num_layers` was checked above and `CrossAttentionCapture::new`
            // filled every requested slot, so this always holds rows.
            let rows = capture.slots[l].take().unwrap_or_default();
            let refs: Vec<&Tensor<R>> = rows.iter().collect();
            let cat = client.cat(&refs, 2).map_err(Error::Numr)?;
            captured.push((l, cat));
        }
        Ok((tokens, CrossAttentionWeights { layers: captured }))
    }

    /// Shared greedy decode loop behind [`Self::generate`] and
    /// [`Self::generate_with_cross_attentions`]. `capture`, when `Some`,
    /// receives one cross-attention row-batch per requested layer for the
    /// prefill call and for every generated-token call that follows.
    fn generate_impl<C>(
        &self,
        client: &C,
        encoder_out: &Tensor<R>,
        start_tokens: &[u32],
        options: &GenerateOptions,
        mut capture: Option<&mut CrossAttentionCapture<R>>,
    ) -> Result<Vec<u32>>
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
                reason: format!("generate currently supports batch=1, got batch={batch}"),
            });
        }
        if start_tokens.is_empty() {
            return Err(Error::ModelError {
                reason: "generate requires at least one start token".into(),
            });
        }

        let device = encoder_out.device();
        let mut cache = self.decoder.new_cache();
        let mut generated: Vec<u32> = Vec::with_capacity(options.max_new_tokens);
        let mut position: usize = 0;

        // Suppression masks are built once per call and reused across steps —
        // rebuilding a [1, 1, vocab] host buffer and uploading it every token is
        // a pure waste of bandwidth. `step_mask` applies at every position;
        // `first_mask` (the union with `begin_suppress_tokens`) applies only at
        // the first generated position, matching HuggingFace's two processors.
        let vocab_size = self.decoder.vocab_size();
        let step_mask = suppression_mask::<R>(&options.suppress_tokens, vocab_size, device)?;
        let begin_mask = if options.begin_suppress_tokens.is_empty() {
            None
        } else {
            let mut union = options.suppress_tokens.clone();
            union.extend_from_slice(&options.begin_suppress_tokens);
            suppression_mask::<R>(&union, vocab_size, device)?
        };
        let first_mask = begin_mask.as_ref().or(step_mask.as_ref());

        // Prefill: feed the prefix through the decoder once so the cache
        // contains all prefix K/V and we have logits for the last prefix token.
        let prefix_i64: Vec<i64> = start_tokens.iter().map(|&t| t as i64).collect();
        let prefix_tensor = Tensor::<R>::from_slice(&prefix_i64, &[1, prefix_i64.len()], device)?;
        let logits = self.decoder.forward_with_cache_capturing(
            client,
            &prefix_tensor,
            encoder_out,
            position,
            &mut cache,
            capture.as_deref_mut(),
        )?;
        position += start_tokens.len();

        // Predict the first token from the final position of the prefix.
        let mut next_token = greedy_pick_last(client, &logits, first_mask)?;

        // Decode loop.
        for _ in 0..options.max_new_tokens {
            if options.eos_token_ids.contains(&next_token) {
                break;
            }
            generated.push(next_token);

            let step_ids = Tensor::<R>::from_slice(&[next_token as i64], &[1, 1], device)?;
            let logits = self.decoder.forward_with_cache_capturing(
                client,
                &step_ids,
                encoder_out,
                position,
                &mut cache,
                capture.as_deref_mut(),
            )?;
            position += 1;

            next_token = greedy_pick_last(client, &logits, step_mask.as_ref())?;
        }

        // Emit the final predicted token unless it's EOS or we exceeded the budget.
        if !options.eos_token_ids.contains(&next_token) && generated.len() < options.max_new_tokens
        {
            generated.push(next_token);
        }

        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::whisper_model::test_support::{
        DECODER_LAYERS, HEADS, cpu_setup, tiny_encoder_out, tiny_model,
    };

    #[test]
    fn generate_with_cross_attentions_tokens_match_plain_generate() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);
        let encoder_out = tiny_encoder_out(&device, 5, 42);
        let start_tokens = [1u32, 2, 3];
        let options = GenerateOptions {
            max_new_tokens: 4,
            ..GenerateOptions::default()
        };

        let plain = model
            .generate(&client, &encoder_out, &start_tokens, &options)
            .expect("plain generate");
        let (captured_tokens, _weights) = model
            .generate_with_cross_attentions(&client, &encoder_out, &start_tokens, &options, &[0, 1])
            .expect("capturing generate");

        assert_eq!(
            plain, captured_tokens,
            "capturing generate must produce identical tokens to plain generate"
        );
    }

    #[test]
    fn generate_with_cross_attentions_shapes_match_requested_layers_only() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);
        let src_len = 3;
        let encoder_out = tiny_encoder_out(&device, src_len, 7);
        let start_tokens = [1u32];
        let options = GenerateOptions {
            max_new_tokens: 2,
            ..GenerateOptions::default()
        };

        // Request only layer 1: layer 0 must not appear in the output at all.
        let (tokens, weights) = model
            .generate_with_cross_attentions(&client, &encoder_out, &start_tokens, &options, &[1])
            .expect("capturing generate");

        assert_eq!(
            weights.layers.len(),
            1,
            "only the requested layer is returned"
        );
        let (layer_idx, tensor) = &weights.layers[0];
        assert_eq!(*layer_idx, 1);

        let total_rows = start_tokens.len() + tokens.len();
        assert_eq!(tensor.shape(), &[1, HEADS, total_rows, src_len]);
    }

    #[test]
    fn generate_with_cross_attentions_rows_are_probability_distributions() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);
        let src_len = 4;
        let encoder_out = tiny_encoder_out(&device, src_len, 99);
        let start_tokens = [2u32, 5];
        let options = GenerateOptions {
            max_new_tokens: 3,
            ..GenerateOptions::default()
        };

        let (_tokens, weights) = model
            .generate_with_cross_attentions(&client, &encoder_out, &start_tokens, &options, &[0, 1])
            .expect("capturing generate");

        assert_eq!(weights.layers.len(), 2);
        for (_, tensor) in &weights.layers {
            let host: Vec<f32> = tensor
                .try_to_vec()
                .expect("host copy for the assertion only");
            for row in host.chunks(src_len) {
                let sum: f32 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "row {row:?} sums to {sum}, not 1");
            }
        }
    }

    #[test]
    fn generate_with_cross_attentions_rejects_out_of_range_layer() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);
        let encoder_out = tiny_encoder_out(&device, 3, 1);
        let start_tokens = [1u32];
        let options = GenerateOptions {
            max_new_tokens: 1,
            ..GenerateOptions::default()
        };

        let result = model.generate_with_cross_attentions(
            &client,
            &encoder_out,
            &start_tokens,
            &options,
            &[DECODER_LAYERS],
        );
        assert!(
            result.is_err(),
            "an out-of-range layer index must be refused"
        );
    }
}
