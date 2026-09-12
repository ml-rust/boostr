//! Tuning for the segmenter and the range type it produces.

use crate::error::{Error, Result};

/// Tuning for [`segments_from_probabilities`](super::segments_from_probabilities).
///
/// [`Default`] is Silero's own default set, which is what the published
/// Silero examples run: threshold 0.5, 250 ms minimum speech, 100 ms minimum
/// silence, 30 ms pad, no cap on segment length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadSegmentOptions {
    /// Probability at or above which a chunk ENTERS speech.
    pub threshold: f32,
    /// Probability below which a chunk starts counting as silence. `None`
    /// derives it as `(threshold - 0.15).max(0.01)`. Keeping it below
    /// `threshold` is the hysteresis that stops a single noisy chunk from
    /// chopping a word in half.
    pub neg_threshold: Option<f32>,
    /// Segments shorter than this are discarded.
    pub min_speech_duration_ms: u32,
    /// Hard cap on a segment's length. `f32::INFINITY` disables the cap, and
    /// is the default — with it disabled the max-speech branches never run.
    pub max_speech_duration_s: f32,
    /// Silence shorter than this does not close a segment.
    pub min_silence_duration_ms: u32,
    /// Padding grown on both sides of every surviving segment, clamped to the
    /// signal and shared with a neighbour when the gap is too small to hold
    /// two full pads.
    pub speech_pad_ms: u32,
    /// When a segment hits `max_speech_duration_s`, only silences longer than
    /// this are candidate split points.
    pub min_silence_at_max_speech_ms: u32,
    /// `true` splits an over-long segment at its LONGEST candidate silence.
    /// `false` is Silero's older behaviour: split at the most recent
    /// candidate silence instead.
    pub use_max_possible_silence_at_max_speech: bool,
}

impl Default for VadSegmentOptions {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            neg_threshold: None,
            min_speech_duration_ms: 250,
            max_speech_duration_s: f32::INFINITY,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
            min_silence_at_max_speech_ms: 98,
            use_max_possible_silence_at_max_speech: true,
        }
    }
}

impl VadSegmentOptions {
    /// Refuse thresholds and caps that would not panic but would silently
    /// produce nonsense boundaries. Run by the segmenter before any rule fires.
    pub fn validate(&self) -> Result<()> {
        if !self.threshold.is_finite() || self.threshold <= 0.0 || self.threshold > 1.0 {
            return Err(Error::InvalidArgument {
                arg: "opts.threshold",
                reason: format!("must be finite and in (0, 1], got {}", self.threshold),
            });
        }
        if let Some(neg) = self.neg_threshold
            && (!neg.is_finite() || neg <= 0.0 || neg >= self.threshold)
        {
            // A NaN here does NOT panic: every `speech_prob < neg_threshold`
            // comparison silently returns false, so silence never closes a segment
            // and the whole recording comes back as one utterance.
            return Err(Error::InvalidArgument {
                arg: "opts.neg_threshold",
                reason: format!(
                    "must be finite and in (0, threshold), got {neg} against a threshold of {}",
                    self.threshold
                ),
            });
        }
        // Infinity is the documented default and means "no cap"; NaN and negatives
        // are not. A negative cap makes the max-speech split fire on every chunk.
        if self.max_speech_duration_s.is_nan() || self.max_speech_duration_s <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "opts.max_speech_duration_s",
                reason: format!(
                    "must be positive (or infinite for no cap), got {}",
                    self.max_speech_duration_s
                ),
            });
        }
        Ok(())
    }
}

/// One detected utterance, as a half-open sample range `[start, end)` into the
/// ORIGINAL signal — not into the chunk grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSegment {
    /// First sample of the segment.
    pub start: usize,
    /// One past the last sample of the segment.
    pub end: usize,
}

impl SpeechSegment {
    /// Length in samples.
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Whether the segment covers no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Duration in seconds at `sample_rate`. Zero for a zero sample rate
    /// rather than a division by zero.
    pub fn duration_secs(&self, sample_rate: usize) -> f64 {
        if sample_rate == 0 {
            return 0.0;
        }
        self.len() as f64 / sample_rate as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_segment_reports_its_length_and_duration() {
        let seg = SpeechSegment {
            start: 4640,
            end: 15840,
        };
        assert_eq!(seg.len(), 11200);
        assert!(!seg.is_empty());
        assert!((seg.duration_secs(16000) - 0.7).abs() < 1e-9);
        assert_eq!(seg.duration_secs(0), 0.0);

        // An inverted range is empty rather than a wrapped subtraction.
        let inverted = SpeechSegment { start: 10, end: 5 };
        assert_eq!(inverted.len(), 0);
        assert!(inverted.is_empty());
    }

    #[test]
    fn a_threshold_outside_the_unit_interval_is_rejected() {
        for threshold in [0.0, -0.5, 1.5, f32::NAN, f32::INFINITY] {
            let bad = VadSegmentOptions {
                threshold,
                ..VadSegmentOptions::default()
            };
            assert!(
                bad.validate().is_err(),
                "threshold {threshold} must be refused"
            );
        }
        let edge = VadSegmentOptions {
            threshold: 1.0,
            ..VadSegmentOptions::default()
        };
        assert!(edge.validate().is_ok());
    }

    /// A `neg_threshold` of NaN does not panic — every `prob < neg_threshold` test
    /// silently returns false, so no silence ever closes a segment and a whole
    /// recording comes back as one utterance. It has to be refused up front.
    #[test]
    fn a_nonsensical_neg_threshold_is_rejected() {
        for neg in [f32::NAN, 0.0, -0.1, 1.5] {
            let bad = VadSegmentOptions {
                neg_threshold: Some(neg),
                ..VadSegmentOptions::default()
            };
            assert!(
                bad.validate().is_err(),
                "neg_threshold {neg} must be refused"
            );
        }
        // At or above `threshold` is contradictory: the hysteresis inverts.
        let bad = VadSegmentOptions {
            threshold: 0.5,
            neg_threshold: Some(0.5),
            ..VadSegmentOptions::default()
        };
        assert!(bad.validate().is_err());

        // The derived default and any value below the threshold stay accepted.
        let good = VadSegmentOptions {
            neg_threshold: Some(0.35),
            ..VadSegmentOptions::default()
        };
        assert!(good.validate().is_ok());
        assert!(VadSegmentOptions::default().validate().is_ok());
    }

    /// A negative or NaN cap does not panic either: it makes `max_speech_samples`
    /// smaller than any real segment, so the max-speech split fires on nearly every
    /// chunk and shreds the output.
    #[test]
    fn a_nonsensical_max_speech_duration_is_rejected() {
        for cap in [f32::NAN, 0.0, -1.0] {
            let bad = VadSegmentOptions {
                max_speech_duration_s: cap,
                ..VadSegmentOptions::default()
            };
            assert!(
                bad.validate().is_err(),
                "max_speech_duration_s {cap} must be refused"
            );
        }
        // Infinity is the default and means "no cap".
        assert!(
            VadSegmentOptions::default()
                .max_speech_duration_s
                .is_infinite()
        );
        assert!(VadSegmentOptions::default().validate().is_ok());
    }
}
