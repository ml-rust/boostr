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
        let logits = self.decoder.forward_with_cache(
            client,
            &prefix_tensor,
            encoder_out,
            position,
            &mut cache,
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
            let logits = self.decoder.forward_with_cache(
                client,
                &step_ids,
                encoder_out,
                position,
                &mut cache,
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

/// Build an additive `[1, 1, vocab]` suppression mask holding `-inf` at every
/// id in `ids` and `0` elsewhere. Returns `None` when `ids` is empty, so the
/// caller can skip the add entirely.
fn suppression_mask<R>(
    ids: &[u32],
    vocab_size: usize,
    device: &R::Device,
) -> Result<Option<Tensor<R>>>
where
    R: Runtime<DType = DType>,
{
    if ids.is_empty() {
        return Ok(None);
    }
    let mut mask = vec![0.0f32; vocab_size];
    for &id in ids {
        if (id as usize) < vocab_size {
            mask[id as usize] = f32::NEG_INFINITY;
        }
    }
    let tensor = Tensor::<R>::from_slice(&mask, &[1, 1, vocab_size], device)?;
    Ok(Some(tensor))
}

/// Greedy-pick the argmax over the vocab dimension at the **last** time step of
/// a logits tensor `[B, T, vocab]` (`B==1` here).
///
/// `mask` is an additive `-inf` suppression mask from [`suppression_mask`], or
/// `None` when nothing is suppressed. The argmax runs on-device; only the
/// resulting index crosses back to the host.
fn greedy_pick_last<R, C>(client: &C, logits: &Tensor<R>, mask: Option<&Tensor<R>>) -> Result<u32>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + BinaryOps<R> + IndexingOps<R>,
{
    let shape = logits.shape();
    if shape.len() != 3 {
        return Err(Error::ModelError {
            reason: format!("greedy_pick_last expects [B, T, vocab] logits, got {shape:?}"),
        });
    }
    let t = shape[1];
    // Unreachable on today's call paths — `generate` rejects an empty prefix and
    // every later step feeds a `[1, 1]` id tensor — but `t - 1` on a `usize` is
    // an underflow, not a caught mistake, so it is refused rather than assumed.
    if t == 0 {
        return Err(Error::ModelError {
            reason: "greedy_pick_last got zero time steps".to_string(),
        });
    }
    // Slice last time-step: [1, 1, vocab]
    let last = logits.narrow(1, t - 1, 1).map_err(Error::Numr)?;

    let scored = match mask {
        Some(mask) => client.add(&last, mask).map_err(Error::Numr)?,
        None => last,
    };

    // Device-side argmax over the vocab dimension → [1, 1] of I64.
    let index = client.argmax(&scored, 2, false).map_err(Error::Numr)?;
    let host: Vec<i64> = index.try_to_vec().map_err(Error::Numr)?;
    let picked = host.first().copied().ok_or_else(|| Error::ModelError {
        reason: "argmax over the vocab dimension returned an empty index tensor".into(),
    })?;
    u32::try_from(picked).map_err(|_| Error::ModelError {
        reason: format!("argmax returned an out-of-range token index {picked}"),
    })
}
