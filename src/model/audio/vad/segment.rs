//! Segmentation: speech probabilities in, utterance boundaries out.
//!
//! [`super::model::SileroVad`] scores one 512-sample chunk at a time. That
//! per-chunk probability is not a usable answer on its own — a single dip below
//! the threshold in the middle of a word would end an utterance. This layer is
//! the port of upstream's `get_speech_timestamps`: hysteresis (a separate,
//! lower threshold to LEAVE speech), a minimum silence before a segment closes,
//! a minimum duration before a segment counts, an optional hard cap on segment
//! length, and a symmetric pad grown around every surviving segment.
//!
//! [`segments_from_probabilities`] is the whole algorithm and takes no model,
//! no client and no tensors, so it is testable against hand-written probability
//! arrays. [`super::model::SileroVad::speech_timestamps`] is the convenience
//! wrapper that runs the network first.
//!
//! # The rules interact, and the order is upstream's
//!
//! The branches below are ported statement by statement, including two places
//! where upstream's own control flow is asymmetric (one of the three
//! max-speech sub-branches skips the rest of the iteration, the other two fall
//! through) and one where a variable is read and then zeroed a few lines later.
//! Tidying either changes which boundaries come out, so both are reproduced as
//! written and called out in comments where they occur.

use crate::error::{Error, Result};
use crate::model::audio::vad::model::SileroVad;
use numr::dtype::DType;
use numr::ops::{ConvOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Tuning for [`segments_from_probabilities`].
///
/// [`Default`] is upstream's own default set, which is what the published
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
    /// `false` is upstream's older behaviour: split at the most recent
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

/// Turn per-chunk speech probabilities into utterance boundaries.
///
/// `probs[i]` is the probability for the chunk starting at sample
/// `i * window_size`, so exactly `ceil(num_samples / window_size)` values are
/// expected — upstream evaluates a ZERO-PADDED final partial chunk rather than
/// dropping it. Note that [`SileroVad::probabilities`] deliberately drops that
/// trailing partial chunk instead (its own doc comment says so, and a test
/// pins it), so its output is one value short for any signal whose length is
/// not a multiple of the chunk size. Use
/// [`SileroVad::speech_timestamps`], which does the padding.
///
/// Returns segments in ascending order, already padded by
/// [`VadSegmentOptions::speech_pad_ms`].
pub fn segments_from_probabilities(
    probs: &[f32],
    num_samples: usize,
    sample_rate: usize,
    window_size: usize,
    opts: &VadSegmentOptions,
) -> Result<Vec<SpeechSegment>> {
    if sample_rate == 0 {
        return Err(Error::InvalidArgument {
            arg: "sample_rate",
            reason: "must be non-zero".to_string(),
        });
    }
    if window_size == 0 {
        return Err(Error::InvalidArgument {
            arg: "window_size",
            reason: "must be non-zero".to_string(),
        });
    }
    if !opts.threshold.is_finite() || opts.threshold <= 0.0 || opts.threshold > 1.0 {
        return Err(Error::InvalidArgument {
            arg: "opts.threshold",
            reason: format!("must be finite and in (0, 1], got {}", opts.threshold),
        });
    }
    if let Some(neg) = opts.neg_threshold
        && (!neg.is_finite() || neg <= 0.0 || neg >= opts.threshold)
    {
        // A NaN here does NOT panic: every `speech_prob < neg_threshold`
        // comparison silently returns false, so silence never closes a segment
        // and the whole recording comes back as one utterance.
        return Err(Error::InvalidArgument {
            arg: "opts.neg_threshold",
            reason: format!(
                "must be finite and in (0, threshold), got {neg} against a threshold of {}",
                opts.threshold
            ),
        });
    }
    // Infinity is the documented default and means "no cap"; NaN and negatives
    // are not. A negative cap makes the max-speech split fire on every chunk.
    if opts.max_speech_duration_s.is_nan() || opts.max_speech_duration_s <= 0.0 {
        return Err(Error::InvalidArgument {
            arg: "opts.max_speech_duration_s",
            reason: format!(
                "must be positive (or infinite for no cap), got {}",
                opts.max_speech_duration_s
            ),
        });
    }
    let expected = num_samples.div_ceil(window_size);
    if probs.len() != expected {
        return Err(Error::InvalidArgument {
            arg: "probs",
            reason: format!(
                "{} probabilities for {num_samples} samples at a window of \
                 {window_size}: expected {expected} \
                 (ceil({num_samples} / {window_size}))",
                probs.len()
            ),
        });
    }

    let rate = sample_rate as f64;
    let threshold = opts.threshold;
    let neg_threshold = opts
        .neg_threshold
        .unwrap_or_else(|| (threshold - 0.15).max(0.01));
    let min_speech_samples = rate * opts.min_speech_duration_ms as f64 / 1000.0;
    let speech_pad_samples = (rate * opts.speech_pad_ms as f64 / 1000.0) as usize;
    // Infinite `max_speech_duration_s` stays infinite: no finite subtraction
    // can bring it back into range, so the cap never fires.
    let max_speech_samples = rate * opts.max_speech_duration_s as f64
        - window_size as f64
        - 2.0 * speech_pad_samples as f64;
    let min_silence_samples = (rate * opts.min_silence_duration_ms as f64 / 1000.0) as usize;
    let min_silence_samples_at_max_speech =
        (rate * opts.min_silence_at_max_speech_ms as f64 / 1000.0) as usize;

    let mut triggered = false;
    let mut speeches: Vec<SpeechSegment> = Vec::new();
    let mut current: Option<SpeechSegment> = None;
    let mut temp_end: usize = 0;
    let mut prev_end: usize = 0;
    let mut next_start: usize = 0;
    // `(silence start, silence duration)` candidates for splitting an
    // over-long segment.
    let mut possible_ends: Vec<(usize, usize)> = Vec::new();

    for (i, &speech_prob) in probs.iter().enumerate() {
        let cur_sample = window_size * i;

        // Speech resumed while a silence was being timed: bank that silence as
        // a candidate split point and stop timing it.
        if speech_prob >= threshold && temp_end != 0 {
            let sil_dur = cur_sample - temp_end;
            if sil_dur > min_silence_samples_at_max_speech {
                possible_ends.push((temp_end, sil_dur));
            }
            temp_end = 0;
            if next_start < prev_end {
                next_start = cur_sample;
            }
        }

        // Enter speech.
        if speech_prob >= threshold && !triggered {
            triggered = true;
            current = Some(SpeechSegment {
                start: cur_sample,
                end: 0,
            });
            continue;
        }

        // The segment has outgrown `max_speech_duration_s` and must be split.
        // Signed arithmetic throughout: a small cap makes `max_speech_samples`
        // negative, and upstream compares against it as a plain number.
        if triggered
            && let Some(seg) = current
            && (cur_sample as f64 - seg.start as f64) > max_speech_samples
        {
            // Python's `max` keeps the FIRST maximum on ties, so this reduces
            // with a strict `>` rather than using `max_by_key`, which keeps
            // the last. `None` here is upstream's empty-candidate case.
            let longest_silence = possible_ends.iter().copied().reduce(|best, candidate| {
                if candidate.1 > best.1 {
                    candidate
                } else {
                    best
                }
            });

            if opts.use_max_possible_silence_at_max_speech
                && let Some((split_at, dur)) = longest_silence
            {
                // Upstream binds `prev_end` from the tuple here, uses it for
                // the two lines below, and only then zeroes it — so both reads
                // see the TUPLE's value, not the loop's `prev_end`.
                speeches.push(SpeechSegment {
                    start: seg.start,
                    end: split_at,
                });
                next_start = split_at + dur;
                if next_start < split_at + cur_sample {
                    current = Some(SpeechSegment {
                        start: next_start,
                        end: 0,
                    });
                } else {
                    current = None;
                    triggered = false;
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                possible_ends.clear();
            } else if prev_end != 0 {
                speeches.push(SpeechSegment {
                    start: seg.start,
                    end: prev_end,
                });
                // The polarity is the opposite of the branch above. That is
                // upstream's, not a transcription slip.
                if next_start < prev_end {
                    current = None;
                    triggered = false;
                } else {
                    current = Some(SpeechSegment {
                        start: next_start,
                        end: 0,
                    });
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                possible_ends.clear();
            } else {
                // No candidate silence at all: cut at the current chunk. ONLY
                // this sub-branch skips the rest of the iteration; the two
                // above fall through into the silence handling below.
                speeches.push(SpeechSegment {
                    start: seg.start,
                    end: cur_sample,
                });
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
                current = None;
                possible_ends.clear();
                continue;
            }
        }

        // Below the release threshold while in speech: time the silence, and
        // close the segment once it is long enough.
        if speech_prob < neg_threshold
            && triggered
            && let Some(seg) = current
        {
            if temp_end == 0 {
                temp_end = cur_sample;
            }
            let sil_dur_now = cur_sample - temp_end;
            if !opts.use_max_possible_silence_at_max_speech
                && sil_dur_now > min_silence_samples_at_max_speech
            {
                prev_end = temp_end;
            }
            if sil_dur_now < min_silence_samples {
                continue;
            }
            // The segment ends where the silence STARTED, not here.
            if (temp_end as f64 - seg.start as f64) > min_speech_samples {
                speeches.push(SpeechSegment {
                    start: seg.start,
                    end: temp_end,
                });
            }
            current = None;
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            triggered = false;
            possible_ends.clear();
            continue;
        }
    }

    // A segment still open at the end of the signal runs to the last sample.
    if let Some(seg) = current
        && (num_samples as f64 - seg.start as f64) > min_speech_samples
    {
        speeches.push(SpeechSegment {
            start: seg.start,
            end: num_samples,
        });
    }

    pad_segments(&mut speeches, num_samples, speech_pad_samples);
    Ok(speeches)
}

/// Grow every segment by `pad` samples on both sides.
///
/// This MUTATES NEIGHBOURS: iteration `i` moves both `speeches[i].end` and
/// `speeches[i + 1].start`, and iteration `i + 1` then sees that. It is a
/// sequential in-place pass for that reason, not a map. Where a gap cannot
/// hold two full pads, the two segments split the gap and meet in the middle.
fn pad_segments(speeches: &mut [SpeechSegment], num_samples: usize, pad: usize) {
    let n = speeches.len();
    for i in 0..n {
        if i == 0 {
            speeches[i].start = speeches[i].start.saturating_sub(pad);
        }
        if i + 1 != n {
            // Signed: the max-speech split can leave a segment starting at or
            // before its predecessor's end, and a usize subtraction would wrap.
            let silence = speeches[i + 1].start as i64 - speeches[i].end as i64;
            if silence < 2 * pad as i64 {
                // Floor division, matching Python's `//` for negative gaps.
                let half = silence.div_euclid(2);
                speeches[i].end = (speeches[i].end as i64 + half).max(0) as usize;
                speeches[i + 1].start = (speeches[i + 1].start as i64 - half).max(0) as usize;
            } else {
                speeches[i].end = (speeches[i].end + pad).min(num_samples);
                speeches[i + 1].start = speeches[i + 1].start.saturating_sub(pad);
            }
        } else {
            speeches[i].end = (speeches[i].end + pad).min(num_samples);
        }
    }
}

impl<R: Runtime<DType = DType>> SileroVad<R> {
    /// Run the network over `samples` from a fresh state, then segment the
    /// probabilities with [`segments_from_probabilities`].
    ///
    /// The final partial chunk is ZERO-PADDED to a full chunk and evaluated,
    /// because upstream's `get_speech_timestamps` scores `ceil(n / chunk)`
    /// chunks and every boundary is measured off that grid. This is the one
    /// deliberate difference from [`SileroVad::probabilities`], which DROPS a
    /// trailing partial chunk to stay bit-comparable with the ONNX reference
    /// run; dropping it here would shift boundaries on any signal whose length
    /// is not a multiple of the chunk size.
    pub fn speech_timestamps<C>(
        &self,
        client: &C,
        samples: &[f32],
        opts: &VadSegmentOptions,
    ) -> Result<Vec<SpeechSegment>>
    where
        C: RuntimeClient<R> + TensorOps<R> + ConvOps<R>,
    {
        let config = *self.config();
        let window = config.chunk_samples;
        if window == 0 {
            return Err(Error::InvalidArgument {
                arg: "config.chunk_samples",
                reason: "must be non-zero".to_string(),
            });
        }

        let mut state = self.new_state(client.device())?;
        let chunks = samples.len().div_ceil(window);
        let mut probs = Vec::with_capacity(chunks);
        let mut padded = vec![0.0f32; window];
        for i in 0..chunks {
            let start = i * window;
            let end = (start + window).min(samples.len());
            let chunk = &samples[start..end];
            let prob = if chunk.len() == window {
                self.chunk_probability(client, &mut state, chunk)?
            } else {
                padded[..chunk.len()].copy_from_slice(chunk);
                for sample in &mut padded[chunk.len()..] {
                    *sample = 0.0;
                }
                self.chunk_probability(client, &mut state, &padded)?
            };
            probs.push(prob);
        }

        segments_from_probabilities(&probs, samples.len(), config.sample_rate, window, opts)
    }
}

#[cfg(test)]
mod tests;
