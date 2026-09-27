//! The caller's half of a request: resolve the voice, tokenize, size the
//! patch budget, queue the job, and block on the reply.
//!
//! Everything here runs on the caller's thread. No device work happens on
//! it, so a bad request — an unknown voice, empty text, a speed the model
//! cannot serve — fails before it ever reaches a batch. The generation
//! itself runs on the engine's owner thread (`super::batch`, `super::run`),
//! which groups whatever is queued at the same time into ONE generation.
//!
//! A streaming caller keeps its sink here and drains decoded chunks off a
//! channel: the sink is not `Send`, so it must not cross to the owner
//! thread. A sink error sets the job's cancel flag, which retires that row
//! of the running batch.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use boostr::model::audio::voxcpm::model::VoxCpm2Model;
use boostr::model::audio::voxcpm::model::config::AUDIO_START_ID;
use boostr::model::audio::voxcpm::vae::decoder::SAMPLE_RATE;

use crate::error::{Error, Result};
use crate::lang::Lang;
use crate::tts::{TtsEngine, Voice};
use crate::voxcpm::engine::batch::{Job, Reply};
use crate::voxcpm::engine::types::{
    MAX_LEN_CAP, VoxCpm2Engine, VoxCpm2RenderClient, ZERO_SHOT_VOICE_ID,
};
use crate::voxcpm::tokenizer::{normalize_whitespace, tokenize};

impl<R> VoxCpm2Engine<R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    /// Everything before device work: resolve the voice, tokenize, size the
    /// budget. Runs on the caller's thread, so a bad request never joins a
    /// batch.
    ///
    /// The job carries the voice ID, not the voice's tensor: a queued job
    /// must borrow nothing from the engine, and the owner thread looks the
    /// same ID up when it builds the batch.
    fn plan(&self, text: &str, voice_id: &str, reply: Reply) -> Result<Job> {
        let ref_len = if voice_id == ZERO_SHOT_VOICE_ID {
            None
        } else {
            let voice = self
                .inner
                .voices
                .get(voice_id)
                .ok_or_else(|| Error::InvalidArgument {
                    arg: "voice",
                    reason: format!(
                        "unknown voice {voice_id:?}; available voices: [{}], or {:?} for \
                         zero-shot rendering",
                        self.inner
                            .voices
                            .keys()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", "),
                        ZERO_SHOT_VOICE_ID,
                    ),
                })?;
            Some(voice.ref_feat.shape()[0])
        };
        let normalized = normalize_whitespace(text);
        if normalized.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "text",
                reason: "input text must not be empty".into(),
            });
        }
        let mut text_token_ids = tokenize(&self.inner.tokenizer, &normalized);
        let text_len = text_token_ids.len();
        text_token_ids.push(AUDIO_START_ID);

        // The clone pipeline's budget: six patches per text token plus ten,
        // capped.
        let max_len = (text_len * 6 + 10).min(MAX_LEN_CAP);
        let seq_len = seq_len_for(ref_len, text_token_ids.len());

        Ok(Job {
            voice: (voice_id != ZERO_SHOT_VOICE_ID).then(|| voice_id.to_string()),
            text_token_ids,
            max_len,
            seq_len,
            reply,
        })
    }

    /// Queue a job for the owner thread.
    fn submit(&self, job: Job) -> Result<()> {
        let Some(jobs) = self.jobs.as_ref() else {
            return Err(worker_gone());
        };
        jobs.send(job).map_err(|_| worker_gone())
    }

    /// Whole-utterance render: queue the request, wait for its waveform.
    fn render(&self, text: &str, voice_id: &str) -> Result<Vec<f32>> {
        let (done, reply) = channel();
        self.submit(self.plan(text, voice_id, Reply::Buffered(done))?)?;
        reply.recv().map_err(|_| worker_gone())?
    }

    /// Chunked render: queue the request, then drain the decoded chunks into
    /// `sink` as the owner thread produces them. The sender drops when the
    /// batch ends, which is how the last chunk is recognised.
    ///
    /// A `sink` error cancels this row and returns at once; the batch runs
    /// on for the other rows.
    fn render_stream(
        &self,
        text: &str,
        voice_id: &str,
        sink: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        if self.inner.options.stream_chunk_patches == 0 {
            return Err(Error::InvalidArgument {
                arg: "stream_chunk_patches",
                reason: "expected at least 1 patch per chunk, got 0".into(),
            });
        }
        let (chunks, reply) = channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let job = self.plan(
            text,
            voice_id,
            Reply::Streaming {
                chunks,
                cancelled: Arc::clone(&cancelled),
            },
        )?;
        self.submit(job)?;

        let mut delivered = false;
        while let Ok(message) = reply.recv() {
            let samples = message?;
            if let Err(err) = sink(&samples) {
                cancelled.store(true, Ordering::Relaxed);
                return Err(err);
            }
            delivered = true;
        }
        // Every render emits at least one patch, so an empty stream means
        // the owner thread died without reporting anything.
        if delivered {
            Ok(())
        } else {
            Err(worker_gone())
        }
    }
}

impl<R> TtsEngine for VoxCpm2Engine<R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
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
                self.inner
                    .voices
                    .keys()
                    .map(|id| Voice::new(id.clone(), Lang::Ms, id.clone())),
            )
            .collect()
    }
}

/// The owner thread is gone: it panicked, or the engine is shutting down.
/// Every waiting caller gets this instead of blocking forever.
fn worker_gone() -> Error {
    Error::ModelError {
        reason: "VoxCPM2 render thread is not running; an earlier render panicked or the \
                 engine is shutting down"
            .into(),
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

    #[test]
    fn a_dead_worker_is_an_error_not_a_hang() {
        let err = worker_gone();
        assert!(err.to_string().contains("not running"), "{err}");
    }
}
