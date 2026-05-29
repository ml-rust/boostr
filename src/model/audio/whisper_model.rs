//! End-to-end Whisper model: encoder + decoder with greedy generation.
//!
//! Weight layout matches HuggingFace `WhisperForConditionalGeneration`:
//! - `model.encoder.*` → [`WhisperEncoder`]
//! - `model.decoder.*` → [`WhisperDecoder`]
//! - `proj_out.weight` is tied to `model.decoder.embed_tokens.weight`
//!   (the decoder already takes care of this via its `tied_out_weight`).

use crate::error::{Error, Result};
use crate::model::audio::whisper::WhisperEncoder;
use crate::model::audio::whisper_decoder::WhisperDecoder;
use crate::model::config::AudioConfig;
use crate::nn::VarBuilder;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConditionalOps, ConvOps, IndexingOps, MatmulOps, NormalizationOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps,
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
    /// Whisper's `<|notimestamps|>` / language tokens are usually part of the
    /// prefix, not the output, but some decoders use suppression to skip them.
    pub suppress_tokens: Vec<u32>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 448,
            eos_token_ids: Vec::new(),
            suppress_tokens: Vec::new(),
        }
    }
}

/// Full Whisper model (encoder + decoder).
pub struct WhisperModel<R: Runtime> {
    pub encoder: WhisperEncoder<R>,
    pub decoder: WhisperDecoder<R>,
    config: AudioConfig,
}

impl<R: Runtime<DType = DType>> WhisperModel<R> {
    /// Load from a VarBuilder rooted at the model top level. Expects `model.encoder.*`
    /// and `model.decoder.*` underneath.
    pub fn from_varbuilder(vb: &mut VarBuilder<'_, R>, config: &AudioConfig) -> Result<Self> {
        let mut model_vb = vb.pp("model");
        let mut enc_vb = model_vb.pp("encoder");
        let encoder = WhisperEncoder::from_varbuilder(&mut enc_vb, config)?;
        drop(enc_vb);
        let mut dec_vb = model_vb.pp("decoder");
        let decoder = WhisperDecoder::from_varbuilder(&mut dec_vb, config)?;
        Ok(Self {
            encoder,
            decoder,
            config: config.clone(),
        })
    }

    pub fn config(&self) -> &AudioConfig {
        &self.config
    }

    /// Run the encoder on a mel spectrogram tensor `[B, num_mel_bins, audio_len]`.
    ///
    /// Returns the encoder hidden state `[B, S, hidden]` that both
    /// [`Self::generate`] and any custom decode loop consume.
    pub fn encode<C>(&self, client: &C, mel: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + NormalizationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + ShapeOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ConvOps<R> + ReduceOps<R> + BinaryOps<R>,
    {
        self.encoder.forward_inference(client, mel)
    }

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
            + IndexingOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        assert_eq!(
            encoder_out.shape()[0],
            1,
            "WhisperModel::generate currently supports batch=1"
        );
        if start_tokens.is_empty() {
            return Err(Error::ModelError {
                reason: "generate requires at least one start token".into(),
            });
        }

        let device = encoder_out.device();
        let mut cache = self.decoder.new_cache();
        let mut generated: Vec<u32> = Vec::with_capacity(options.max_new_tokens);
        let mut position: usize = 0;

        // Prefill: feed the prefix through the decoder once so the cache
        // contains all prefix K/V and we have logits for the last prefix token.
        let prefix_i64: Vec<i64> = start_tokens.iter().map(|&t| t as i64).collect();
        let prefix_tensor = Tensor::<R>::from_slice(&prefix_i64, &[1, prefix_i64.len()], device);
        let logits = self.decoder.forward_with_cache(
            client,
            &prefix_tensor,
            encoder_out,
            position,
            &mut cache,
        )?;
        position += start_tokens.len();

        // Predict the first token from the final position of the prefix.
        let mut next_token = greedy_pick_last(
            client,
            &logits,
            self.decoder.vocab_size(),
            &options.suppress_tokens,
        )?;

        // Decode loop.
        for _ in 0..options.max_new_tokens {
            if options.eos_token_ids.contains(&next_token) {
                break;
            }
            generated.push(next_token);

            let step_ids = Tensor::<R>::from_slice(&[next_token as i64], &[1, 1], device);
            let logits = self.decoder.forward_with_cache(
                client,
                &step_ids,
                encoder_out,
                position,
                &mut cache,
            )?;
            position += 1;

            next_token = greedy_pick_last(
                client,
                &logits,
                self.decoder.vocab_size(),
                &options.suppress_tokens,
            )?;
        }

        // Emit the final predicted token unless it's EOS or we exceeded the budget.
        if !options.eos_token_ids.contains(&next_token) && generated.len() < options.max_new_tokens
        {
            generated.push(next_token);
        }

        Ok(generated)
    }
}

/// Greedy-pick the argmax over the vocab dimension at the **last** time step of
/// a logits tensor `[B, T, vocab]` (`B==1` here).
///
/// Suppressed tokens are masked to -inf before the argmax.
fn greedy_pick_last<R, C>(
    client: &C,
    logits: &Tensor<R>,
    vocab_size: usize,
    suppress: &[u32],
) -> Result<u32>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
    R::Client: TensorOps<R>,
{
    let shape = logits.shape();
    let t = shape[1];
    // Slice last time-step: [1, 1, vocab]
    let last = logits.narrow(1, t - 1, 1).map_err(Error::Numr)?;

    if suppress.is_empty() {
        // Pull [vocab] floats and argmax on CPU — cheap since vocab ~51k and we
        // do this once per step.
        let data: Vec<f32> = last.to_vec();
        return Ok(argmax_f32(&data) as u32);
    }

    // Apply suppression by adding a -inf mask. Build a [vocab] mask on CPU.
    let mut mask = vec![0.0f32; vocab_size];
    for &id in suppress {
        if (id as usize) < vocab_size {
            mask[id as usize] = f32::NEG_INFINITY;
        }
    }
    let device = logits.device();
    let mask_t = Tensor::<R>::from_slice(&mask, &[1, 1, vocab_size], device);
    let masked = client.add(&last, &mask_t).map_err(Error::Numr)?;
    let data: Vec<f32> = masked.to_vec();
    Ok(argmax_f32(&data) as u32)
}

fn argmax_f32(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}
