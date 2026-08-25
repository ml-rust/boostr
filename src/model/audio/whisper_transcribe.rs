//! One-call transcription on top of [`WhisperBundle`].
//!
//! [`WhisperBundle::transcribe`] owns the whole front-to-back path — mel front
//! end, encoder, greedy decode, detokenize — so a caller never hand-assembles
//! it. The mel bin count comes from the bundle, so an 80-bin `whisper-tiny` and
//! a 128-bin `whisper-large-v3` are driven by identical caller code.

use crate::error::{Error, Result};
use crate::model::audio::mel::{MelOptions, compute_mel_spectrogram_with};
use crate::model::audio::vad::SpeechSegment;
use crate::model::audio::whisper_loader::WhisperBundle;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConditionalOps, ConvOps, IndexingOps, MatmulOps, NormalizationOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Whisper's encoder window, in seconds. `MelOptions::whisper` pads or trims
/// every input to exactly this, so longer audio is data loss, not a long decode.
const WHISPER_WINDOW_SECS: usize = 30;

/// What a transcription run may vary. [`Default`] transcribes with no language
/// token and the checkpoint's own token budget.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions<'a> {
    /// BCP-47-ish code, e.g. `Some("ms")`. `None` skips the language token,
    /// which is what an english-only checkpoint wants.
    pub language: Option<&'a str>,
    /// Translate to English instead of transcribing in the source language.
    pub translate: bool,
    /// Overrides the checkpoint's own `max_new_tokens` when set. Use it to bound
    /// a run so a divergent decode cannot spin for the full 448 steps.
    pub max_new_tokens: Option<usize>,
}

/// The result of one transcription: the text and the ids it was decoded from.
///
/// Both are returned because callers need different halves — re-tokenizing with
/// a different tokenizer needs the text, and detecting a degenerate decode (a
/// repetition loop, an empty generation) needs the ids.
#[derive(Debug, Clone)]
pub struct Transcription {
    /// Decoded text, with special tokens skipped.
    pub text: String,
    /// The generated ids only: no SOT prefix, no trailing end-of-text.
    pub tokens: Vec<u32>,
}

impl<R: Runtime<DType = DType>> WhisperBundle<R> {
    /// Transcribe one utterance. `samples` must be mono at `sample_rate`.
    ///
    /// Audio longer than 30 s is **rejected**, not truncated: Whisper's encoder
    /// takes a fixed 30 s window, so a 20-minute recording would otherwise
    /// yield a plausible transcript of its first 30 seconds with nothing to
    /// mark the rest as lost. Segment first — [`SileroVad::speech_timestamps`]
    /// produces exactly the [`SpeechSegment`]s
    /// [`transcribe_segments`](Self::transcribe_segments) consumes.
    ///
    /// [`SileroVad::speech_timestamps`]: crate::model::audio::SileroVad::speech_timestamps
    pub fn transcribe<C>(
        &self,
        client: &C,
        samples: &[f32],
        sample_rate: usize,
        opts: &TranscribeOptions<'_>,
    ) -> Result<Transcription>
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
            + ShapeOps<R>
            + UnaryOps<R>
            + ConditionalOps<R>
            + IndexingOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ConvOps<R> + ReduceOps<R> + BinaryOps<R>,
    {
        if sample_rate == 0 {
            return Err(Error::InvalidArgument {
                arg: "sample_rate",
                reason: "sample rate must be non-zero".into(),
            });
        }
        let max_samples = WHISPER_WINDOW_SECS * sample_rate;
        if samples.len() > max_samples {
            let secs = samples.len() as f64 / sample_rate as f64;
            return Err(Error::InvalidArgument {
                arg: "samples",
                reason: format!(
                    "audio is {secs:.3} s ({} samples at {sample_rate} Hz), over Whisper's fixed \
                     {WHISPER_WINDOW_SECS} s window ({max_samples} samples); segment the audio \
                     first (see SileroVad::speech_timestamps) and transcribe each segment",
                    samples.len()
                ),
            });
        }

        let mel_opts = MelOptions::whisper(self.num_mel_bins, sample_rate);
        let mel = compute_mel_spectrogram_with(samples, sample_rate, &mel_opts)?;
        if self.num_mel_bins == 0 || !mel.len().is_multiple_of(self.num_mel_bins) {
            return Err(Error::InvalidArgument {
                arg: "samples",
                reason: format!(
                    "mel of {} values is not a whole number of {}-bin frames",
                    mel.len(),
                    self.num_mel_bins
                ),
            });
        }
        let num_frames = mel.len() / self.num_mel_bins;

        let shape = [1, self.num_mel_bins, num_frames];
        let mel_t = Tensor::<R>::from_slice(&mel, &shape, client.device()).map_err(Error::Numr)?;
        let encoded = self.model.encode(client, &mel_t)?;

        let prompt = self.sot_prompt(opts.language, opts.translate);
        let mut gen_opts = self.generate_options();
        if let Some(budget) = opts.max_new_tokens {
            gen_opts.max_new_tokens = budget;
        }
        let tokens = self.model.generate(client, &encoded, &prompt, &gen_opts)?;

        // `AnyTokenizer::decode` skips special tokens already, so the SOT
        // prefix and any stray control token cannot leak into the text.
        let text = self
            .tokenizer
            .decode(&tokens)
            .map_err(|e| Error::ModelError {
                reason: format!("decoding {} whisper token ids: {e}", tokens.len()),
            })?;

        Ok(Transcription { text, tokens })
    }

    /// Transcribe each segment of `samples`. Segments come from
    /// [`SileroVad::speech_timestamps`]; each is sliced out of `samples` and
    /// transcribed independently, so each must still fit Whisper's 30 s window.
    ///
    /// A segment whose range falls outside `samples` is an error naming the
    /// offending index and its bounds.
    ///
    /// [`SileroVad::speech_timestamps`]: crate::model::audio::SileroVad::speech_timestamps
    pub fn transcribe_segments<C>(
        &self,
        client: &C,
        samples: &[f32],
        sample_rate: usize,
        segments: &[SpeechSegment],
        opts: &TranscribeOptions<'_>,
    ) -> Result<Vec<Transcription>>
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
            + ShapeOps<R>
            + UnaryOps<R>
            + ConditionalOps<R>
            + IndexingOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ConvOps<R> + ReduceOps<R> + BinaryOps<R>,
    {
        let mut out = Vec::with_capacity(segments.len());
        for (i, seg) in segments.iter().enumerate() {
            if seg.start > seg.end || seg.end > samples.len() {
                return Err(Error::InvalidArgument {
                    arg: "segments",
                    reason: format!(
                        "segment {i} covers samples {}..{}, outside the {} samples given",
                        seg.start,
                        seg.end,
                        samples.len()
                    ),
                });
            }
            out.push(self.transcribe(client, &samples[seg.start..seg.end], sample_rate, opts)?);
        }
        Ok(out)
    }
}
