//! The request path: voice lookup, tokenizing, prefill, the per-patch loop,
//! and the VAE decode — whole in [`TtsEngine::synthesize`], chunk by chunk
//! in [`TtsEngine::synthesize_stream`].
//!
//! Both entry points share [`VoxCpm2Engine::prepare`], so a streamed and a
//! buffered render of the same request start from the same generation state
//! and draw the same noise. The streamed path decodes each chunk with
//! `decode_patches_from`, whose windows sit on the same grid as the whole
//! decode, so the chunks concatenate to the buffered waveform.

use std::sync::MutexGuard;

use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, RandomOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use boostr::model::audio::voxcpm::VoxCpmClient;
use boostr::model::audio::voxcpm::model::config::AUDIO_START_ID;
use boostr::model::audio::voxcpm::model::{
    GenerateOptions, GenerateState, StepOutcome, VoxCpm2Model,
};
use boostr::model::audio::voxcpm::vae::decoder::SAMPLE_RATE;
use boostr::quant::traits::DequantOps;

use crate::error::{Error, Result};
use crate::g2p::Lang;
use crate::tts::{TtsEngine, Voice};
use crate::voxcpm::engine::types::{MAX_LEN_CAP, VoxCpm2Engine, ZERO_SHOT_VOICE_ID};
use crate::voxcpm::tokenizer::{normalize_whitespace, tokenize};

/// A request resolved down to what `prefill` and the loop take.
struct Plan<'a, R: Runtime> {
    /// The voice's encoded reference; `None` for zero-shot.
    ref_feat: Option<&'a Tensor<R>>,
    /// Text tokens with `AUDIO_START_ID` appended.
    text_token_ids: Vec<u32>,
    /// KV-cache capacity: the prefix plus the patch budget.
    max_length: usize,
    options: GenerateOptions,
}

impl<R> VoxCpm2Engine<R>
where
    R: Runtime<DType = DType>,
    R::Client: VoxCpmClient<R>
        + TensorOps<R>
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
        + RandomOps<R>
        + DequantOps<R>
        + 'static,
{
    /// Take the render lock. Held by the caller across prefill, generation
    /// and decode, so device work from two requests never interleaves.
    fn lock_render(&self) -> Result<MutexGuard<'_, ()>> {
        self.render.lock().map_err(|_| Error::ModelError {
            reason: "VoxCPM2 render lock poisoned by an earlier panic".into(),
        })
    }

    /// Everything before device work: resolve the voice, tokenize, size the
    /// budget. Runs before the render lock is taken, so a bad request fails
    /// without waiting on another render.
    fn plan(&self, text: &str, voice_id: &str) -> Result<Plan<'_, R>> {
        let ref_feat: Option<&Tensor<R>> = if voice_id == ZERO_SHOT_VOICE_ID {
            None
        } else {
            let voice = self
                .voices
                .get(voice_id)
                .ok_or_else(|| Error::InvalidArgument {
                    arg: "voice",
                    reason: format!(
                        "unknown voice {voice_id:?}; available voices: [{}], or {:?} for \
                         zero-shot rendering",
                        self.voices.keys().cloned().collect::<Vec<_>>().join(", "),
                        ZERO_SHOT_VOICE_ID,
                    ),
                })?;
            Some(&voice.ref_feat)
        };
        let normalized = normalize_whitespace(text);
        if normalized.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "text",
                reason: "input text must not be empty".into(),
            });
        }
        let mut text_token_ids = tokenize(&self.tokenizer, &normalized);
        let text_len = text_token_ids.len();
        text_token_ids.push(AUDIO_START_ID);

        // The clone pipeline's budget: six patches per text token plus ten,
        // capped.
        let max_len = (text_len * 6 + 10).min(MAX_LEN_CAP);
        let ref_len = ref_feat.map(|f| f.shape()[0]);
        let max_length = seq_len_for(ref_len, text_token_ids.len()) + max_len;

        let mut options = GenerateOptions::new(max_len, self.options.seed);
        options.cfm.n_timesteps = self.options.n_timesteps;
        options.cfm.cfg_value = self.options.cfg_value;
        options.min_len = self.options.min_len;

        Ok(Plan {
            ref_feat,
            text_token_ids,
            max_length,
            options,
        })
    }

    /// Prefill and build the state the per-patch loop drives. The caller
    /// holds the render lock.
    fn start(&self, plan: &Plan<'_, R>) -> Result<GenerateState<R>> {
        let prefill = self.model.prefill(
            self.client.as_ref(),
            plan.ref_feat,
            &plan.text_token_ids,
            plan.max_length,
        )?;
        Ok(GenerateState::start(prefill, self.model.config)?)
    }

    /// Whole-utterance render: generate to completion, decode once.
    fn render(&self, text: &str, voice_id: &str) -> Result<Vec<f32>> {
        let plan = self.plan(text, voice_id)?;
        let _render = self.lock_render()?;
        let mut state = self.start(&plan)?;
        let options = plan.options;
        let client = self.client.as_ref();
        self.model
            .patch_generator()
            .generate(client, &mut state, &options)?;
        let decoded = self.model.decode_patches(client, &state.patches)?;
        Ok(decoded.contiguous()?.to_vec())
    }

    /// Chunked render: after every `stream_chunk_patches` new patches, and
    /// once more for the tail, decode the patches not yet handed over and
    /// pass them to `sink`. A `sink` error ends generation at once.
    fn render_stream(
        &self,
        text: &str,
        voice_id: &str,
        sink: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        let chunk = self.options.stream_chunk_patches;
        if chunk == 0 {
            return Err(Error::InvalidArgument {
                arg: "stream_chunk_patches",
                reason: "expected at least 1 patch per chunk, got 0".into(),
            });
        }
        let plan = self.plan(text, voice_id)?;
        let _render = self.lock_render()?;
        let mut state = self.start(&plan)?;
        let options = plan.options;
        let client = self.client.as_ref();
        let generator = self.model.patch_generator();

        let mut emitted = 0usize;
        while state.patches.len() < options.max_len {
            let outcome = generator.step(client, &mut state, &options)?;
            if state.patches.len() - emitted >= chunk {
                emitted = self.emit_from(&state, emitted, sink)?;
            }
            if outcome == StepOutcome::Stopped {
                break;
            }
        }
        if emitted < state.patches.len() {
            self.emit_from(&state, emitted, sink)?;
        }
        Ok(())
    }

    /// Decode `state.patches[from..]`, hand the samples to `sink`, and
    /// return the new emitted count.
    fn emit_from(
        &self,
        state: &GenerateState<R>,
        from: usize,
        sink: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<usize> {
        let decoded = self
            .model
            .decode_patches_from(self.client.as_ref(), &state.patches, from)?;
        let samples: Vec<f32> = decoded.contiguous()?.to_vec();
        sink(&samples)?;
        Ok(state.patches.len())
    }
}

impl<R> TtsEngine for VoxCpm2Engine<R>
where
    R: Runtime<DType = DType>,
    R::Client: VoxCpmClient<R>
        + TensorOps<R>
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
        + RandomOps<R>
        + DequantOps<R>
        + Send
        + Sync
        + 'static,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    fn synthesize(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        check_speed(speed)?;
        self.render(text, voice)
    }

    fn synthesize_stream(
        &self,
        text: &str,
        voice: &str,
        speed: f32,
        sink: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        check_speed(speed)?;
        self.render_stream(text, voice, sink)
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE as u32
    }

    /// [`ZERO_SHOT_VOICE_ID`] first, then one entry per reference recording.
    /// VoxCPM2 takes raw text in any language it was trained on, so every
    /// voice carries the product language tag rather than a per-voice one.
    fn voices(&self) -> Vec<Voice> {
        std::iter::once(Voice::new(ZERO_SHOT_VOICE_ID, Lang::Ms, ZERO_SHOT_VOICE_ID))
            .chain(
                self.voices
                    .keys()
                    .map(|id| Voice::new(id.clone(), Lang::Ms, id.clone())),
            )
            .collect()
    }
}

/// The model has no rate control; a silently ignored `speed` would return
/// audio the caller did not ask for.
fn check_speed(speed: f32) -> Result<()> {
    if speed != 1.0 {
        return Err(Error::InvalidArgument {
            arg: "speed",
            reason: format!("VoxCPM2 renders at its natural pace only; got {speed}"),
        });
    }
    Ok(())
}

/// Sequence length behind `prefill`'s cache sizing. Mirrors `clone.rs`'s
/// zero-shot/reference branch exactly (see its `SequenceLayout` docs): a
/// reference prefix adds `t_ref + 2` positions before the text; zero-shot
/// carries no reference prefix at all, not a reference of zero patches.
/// `text_len` is the token count AFTER `AUDIO_START_ID` is appended.
fn seq_len_for(ref_len: Option<usize>, text_len: usize) -> usize {
    match ref_len {
        Some(t_ref) => t_ref + 2 + text_len,
        None => text_len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seq_len_for_with_reference_adds_t_ref_plus_two() {
        assert_eq!(seq_len_for(Some(40), 12), 40 + 2 + 12);
    }

    #[test]
    fn seq_len_for_zero_shot_is_text_len_only() {
        assert_eq!(seq_len_for(None, 12), 12);
    }

    #[test]
    fn speed_other_than_natural_is_refused() {
        assert!(check_speed(1.0).is_ok());
        let err = check_speed(1.5).unwrap_err();
        assert!(err.to_string().contains("1.5"), "{err}");
    }
}
