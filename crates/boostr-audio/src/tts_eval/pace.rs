//! Speaking pace and silence budget of one render, from VAD segments.
//!
//! `words` counts the PROMPT (what the model was asked to say), normalised
//! with [`crate::eval::normalize`], never the transcript. So the ratio
//! `words_per_speech_s` reads as a render-length check: a render that
//! stopped early packs the prompt's words into too little speech and shows
//! HIGH, a render that stalled or babbled past the prompt shows LOW. It is
//! not a speaking-rate measurement of the audio on its own; pair it with
//! `intelligibility` to know which of the two happened.
//!
//! Leading and trailing silence are the gaps before the first and after the
//! last VAD segment. A render with no speech at all reports the whole clip
//! as leading silence and zero trailing silence, so the two never sum past
//! the clip length.

use crate::error::{Error, Result};
use crate::eval::normalize;
use crate::vad::{SpeechSegment, VadSegmentOptions, speech_timestamps};
use boostr::model::audio::vad::SileroVad;
use numr::dtype::DType;
use numr::ops::{ConvOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Speech-versus-silence layout of one clip against its prompt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pace {
    /// Seconds inside VAD segments.
    pub speech_s: f64,
    /// Seconds before the first segment. The whole clip when nothing is voiced.
    pub leading_silence_s: f64,
    /// Seconds after the last segment. Zero when nothing is voiced.
    pub trailing_silence_s: f64,
    /// `speech_s / clip length`.
    pub speech_ratio: f64,
    /// Normalised word count of the PROMPT, not the transcript.
    pub words: usize,
    /// `words / speech_s`. `None` when there is no speech.
    pub words_per_speech_s: Option<f64>,
    /// VAD segment count.
    pub segments: usize,
}

/// Fold already-computed `segments` over a clip of `total_samples` at
/// `sample_rate` against `prompt`. Pure: this is what [`pace`] returns once
/// the network has run. Testable without weights.
pub fn pace_from_segments(
    segments: &[SpeechSegment],
    total_samples: usize,
    sample_rate: u32,
    prompt: &str,
) -> Pace {
    let rate = f64::from(sample_rate.max(1));
    let total_s = total_samples as f64 / rate;
    let speech_s: f64 = segments.iter().map(|s| s.len() as f64 / rate).sum();
    let (leading_silence_s, trailing_silence_s) = match (segments.first(), segments.last()) {
        (Some(first), Some(last)) => (
            first.start as f64 / rate,
            total_samples.saturating_sub(last.end) as f64 / rate,
        ),
        _ => (total_s, 0.0),
    };
    let words = normalize(prompt).len();
    Pace {
        speech_s,
        leading_silence_s,
        trailing_silence_s,
        speech_ratio: if total_s > 0.0 {
            speech_s / total_s
        } else {
            0.0
        },
        words,
        words_per_speech_s: (speech_s > 0.0).then(|| words as f64 / speech_s),
        segments: segments.len(),
    }
}

/// Segment `samples_16k` with `vad` and fold the result against `prompt`.
///
/// `samples_16k` is mono at the rate `vad` was built for (16 kHz for the
/// shipped weights). `opts` tunes the segmenter; [`VadSegmentOptions::default`]
/// is Silero's own set.
pub fn pace<R, C>(
    vad: &SileroVad<R>,
    client: &C,
    samples_16k: &[f32],
    prompt: &str,
    opts: &VadSegmentOptions,
) -> Result<Pace>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ConvOps<R>,
{
    let segments = speech_timestamps(vad, client, samples_16k, opts)?;
    let rate = u32::try_from(vad.config().sample_rate).map_err(|_| Error::InvalidArgument {
        arg: "vad",
        reason: format!(
            "VAD sample rate {} does not fit in u32",
            vad.config().sample_rate
        ),
    })?;
    Ok(pace_from_segments(
        &segments,
        samples_16k.len(),
        rate,
        prompt,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;

    fn seg(start_s: f64, end_s: f64) -> SpeechSegment {
        SpeechSegment {
            start: (start_s * RATE as f64) as usize,
            end: (end_s * RATE as f64) as usize,
        }
    }

    #[test]
    fn silences_and_ratio_come_from_the_outer_segments() {
        let segments = [seg(0.5, 1.5), seg(2.0, 3.0)];
        let p = pace_from_segments(&segments, 4 * RATE as usize, RATE, "one two three four");
        assert!((p.speech_s - 2.0).abs() < 1e-6);
        assert!((p.leading_silence_s - 0.5).abs() < 1e-6);
        assert!((p.trailing_silence_s - 1.0).abs() < 1e-6);
        assert!((p.speech_ratio - 0.5).abs() < 1e-6);
        assert_eq!(p.words, 4);
        assert!((p.words_per_speech_s.expect("speech") - 2.0).abs() < 1e-6);
        assert_eq!(p.segments, 2);
    }

    #[test]
    fn no_speech_is_all_leading_silence() {
        let p = pace_from_segments(&[], 2 * RATE as usize, RATE, "hello");
        assert_eq!(p.speech_s, 0.0);
        assert!((p.leading_silence_s - 2.0).abs() < 1e-6);
        assert_eq!(p.trailing_silence_s, 0.0);
        assert_eq!(p.speech_ratio, 0.0);
        assert_eq!(p.words_per_speech_s, None);
    }

    #[test]
    fn words_are_counted_from_the_normalised_prompt() {
        let p = pace_from_segments(&[seg(0.0, 1.0)], RATE as usize, RATE, "Ter-delete, okay?");
        assert_eq!(p.words, 3);
    }
}
