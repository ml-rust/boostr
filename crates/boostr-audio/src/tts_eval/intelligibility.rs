//! Intelligibility of a render: Whisper transcript scored against the prompt.
//!
//! [`IntelligibilityScorer`] loads one [`WhisperBundle`] and scores any number
//! of clips against the text each was asked to say, with
//! [`word_error_rate`] and [`character_error_rate`] over
//! [`crate::eval::normalize`]d text. The bundle is cast to F32 on load so an
//! fp16 checkpoint (`whisper-large-v3`) runs against the F32 mel.
//!
//! What it tells you: dropped, invented or slurred words. What it cannot
//! tell you: naturalness. Whisper reads clean TTS word-perfect long before
//! the audio sounds right, so a WER of zero across a batch is a floor that
//! has been reached, not a ranking. Signal, pace and timbre carry the rest.
//!
//! # Clips over 30 s
//!
//! Whisper's encoder window is 30 s and [`WhisperBundle::transcribe`] refuses
//! longer input. With the `vad` feature on and a VAD attached through
//! [`IntelligibilityScorer::with_vad`], a long clip is cut at VAD segment
//! boundaries into spans packed under the window and each span is
//! transcribed in turn. Without a VAD the scorer returns an error naming the
//! limit rather than scoring a truncated clip.

use std::path::Path;

use crate::error::{Error, Result};
use crate::eval::{ErrorRate, character_error_rate, word_error_rate};
#[cfg(feature = "vad")]
use crate::vad::{SpeechSegment, VadSegmentOptions, speech_timestamps};
use crate::whisper::{TranscribeOptions, WhisperBundle};
#[cfg(feature = "vad")]
use boostr::model::audio::vad::SileroVad;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConditionalOps, ConvOps, IndexingOps, MatmulOps, NormalizationOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};

/// Sample rate Whisper's mel front end is defined at.
pub const WHISPER_RATE: usize = 16_000;
/// Whisper's encoder window, seconds.
pub const WHISPER_WINDOW_SECS: usize = 30;

/// Client bounds [`WhisperBundle::transcribe`] needs, plus the cast used at
/// load. One alias so callers name one bound instead of thirteen.
pub trait IntelligibilityClient<R: Runtime>:
    RuntimeClient<R>
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
    + IndexingOps<R>
    + TypeConversionOps<R>
{
}

impl<R, C> IntelligibilityClient<R> for C
where
    R: Runtime,
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
        + IndexingOps<R>
        + TypeConversionOps<R>,
{
}

/// Transcript and its error rates against the prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct Intelligibility {
    /// Whisper's text, special tokens skipped. Spans of a long clip are
    /// joined with one space.
    pub transcript: String,
    /// Word error rate counts against the prompt.
    pub wer: ErrorRate,
    /// Character error rate counts against the prompt.
    pub cer: ErrorRate,
}

/// One loaded Whisper, reused across every clip in a batch.
pub struct IntelligibilityScorer<R: Runtime> {
    bundle: WhisperBundle<R>,
    #[cfg(feature = "vad")]
    vad: Option<SileroVad<R>>,
}

impl<R: Runtime<DType = DType>> IntelligibilityScorer<R> {
    /// Load `whisper_dir` (HF layout) as F32 on `device`.
    pub fn from_dir<C: TypeConversionOps<R>>(
        whisper_dir: &Path,
        device: &R::Device,
        client: &C,
    ) -> Result<Self> {
        let bundle =
            WhisperBundle::<R>::from_dir_with_dtype(whisper_dir, device, client, DType::F32)?;
        Ok(Self::new(bundle))
    }

    /// Wrap an already-loaded bundle.
    pub fn new(bundle: WhisperBundle<R>) -> Self {
        Self {
            bundle,
            #[cfg(feature = "vad")]
            vad: None,
        }
    }

    /// Attach a VAD so clips over the 30 s window are split at speech
    /// boundaries instead of refused.
    #[cfg(feature = "vad")]
    pub fn with_vad(mut self, vad: SileroVad<R>) -> Self {
        self.vad = Some(vad);
        self
    }

    /// The bundle, for callers that need the tokenizer or the model directly.
    pub fn bundle(&self) -> &WhisperBundle<R> {
        &self.bundle
    }

    /// Transcribe `samples_16k` (mono, 16 kHz) and score it against `prompt`.
    ///
    /// `language` is Whisper's language token and changes what it emits, not
    /// merely how it labels the output; `None` skips the token.
    pub fn score<C>(
        &self,
        client: &C,
        samples_16k: &[f32],
        prompt: &str,
        language: Option<&str>,
    ) -> Result<Intelligibility>
    where
        C: IntelligibilityClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ConvOps<R> + ReduceOps<R> + BinaryOps<R>,
    {
        let opts = TranscribeOptions {
            language,
            translate: false,
            max_new_tokens: None,
        };
        let transcript = if samples_16k.len() <= WHISPER_WINDOW_SECS * WHISPER_RATE {
            self.bundle
                .transcribe(client, samples_16k, WHISPER_RATE, &opts)?
                .text
        } else {
            self.transcribe_long(client, samples_16k, &opts)?
        };
        let wer = word_error_rate(prompt, &transcript);
        let cer = character_error_rate(prompt, &transcript);
        Ok(Intelligibility {
            transcript,
            wer,
            cer,
        })
    }

    #[cfg(feature = "vad")]
    fn transcribe_long<C>(
        &self,
        client: &C,
        samples_16k: &[f32],
        opts: &TranscribeOptions<'_>,
    ) -> Result<String>
    where
        C: IntelligibilityClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ConvOps<R> + ReduceOps<R> + BinaryOps<R>,
    {
        let vad = self
            .vad
            .as_ref()
            .ok_or_else(|| over_window_error(samples_16k.len()))?;
        let segments = speech_timestamps(vad, client, samples_16k, &VadSegmentOptions::default())?;
        let spans = pack_segments(&segments, WHISPER_WINDOW_SECS * WHISPER_RATE);
        let parts =
            self.bundle
                .transcribe_segments(client, samples_16k, WHISPER_RATE, &spans, opts)?;
        let texts: Vec<&str> = parts.iter().map(|t| t.text.trim()).collect();
        Ok(texts.join(" ").trim().to_string())
    }

    #[cfg(not(feature = "vad"))]
    fn transcribe_long<C>(
        &self,
        _client: &C,
        samples_16k: &[f32],
        _opts: &TranscribeOptions<'_>,
    ) -> Result<String> {
        Err(over_window_error(samples_16k.len()))
    }
}

/// The error for a clip over the window when nothing can split it.
fn over_window_error(len: usize) -> Error {
    Error::InvalidArgument {
        arg: "samples_16k",
        reason: format!(
            "clip is {:.2} s, over Whisper's {WHISPER_WINDOW_SECS} s window, and no VAD is \
             attached to split it (build with the `vad` feature and call `with_vad`)",
            len as f64 / WHISPER_RATE as f64
        ),
    }
}

/// Merge consecutive `segments` into spans no longer than `max_len` samples.
///
/// A span runs from the start of its first segment to the end of its last,
/// so the pauses between merged segments stay in the audio Whisper sees. A
/// single segment longer than `max_len` is cut into `max_len` pieces; that
/// loses the word at each cut, which is still better than dropping the
/// segment.
#[cfg(feature = "vad")]
pub fn pack_segments(segments: &[SpeechSegment], max_len: usize) -> Vec<SpeechSegment> {
    let mut spans: Vec<SpeechSegment> = Vec::new();
    for seg in segments {
        if seg.len() > max_len && max_len > 0 {
            let mut start = seg.start;
            while start < seg.end {
                let end = (start + max_len).min(seg.end);
                spans.push(SpeechSegment { start, end });
                start = end;
            }
            continue;
        }
        match spans.last_mut() {
            Some(last) if seg.end.saturating_sub(last.start) <= max_len => last.end = seg.end,
            _ => spans.push(*seg),
        }
    }
    spans
}

#[cfg(all(test, feature = "vad"))]
mod tests {
    use super::*;

    fn seg(start: usize, end: usize) -> SpeechSegment {
        SpeechSegment { start, end }
    }

    #[test]
    fn packs_neighbours_under_the_cap() {
        let spans = pack_segments(&[seg(0, 10), seg(12, 20), seg(25, 40), seg(41, 50)], 30);
        assert_eq!(spans, vec![seg(0, 20), seg(25, 50)]);
    }

    #[test]
    fn splits_one_oversized_segment() {
        let spans = pack_segments(&[seg(0, 70)], 30);
        assert_eq!(spans, vec![seg(0, 30), seg(30, 60), seg(60, 70)]);
    }

    #[test]
    fn empty_in_empty_out() {
        assert!(pack_segments(&[], 30).is_empty());
    }
}
