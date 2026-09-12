//! The rule engine: per-chunk probabilities in, padded utterance boundaries out.
//!
//! [`segments_from_probabilities`] is the whole algorithm and takes no model,
//! no client and no tensors, so it is testable against hand-written probability
//! arrays. Hysteresis (a separate, lower threshold to LEAVE speech), a minimum
//! silence before a segment closes, a minimum duration before a segment counts,
//! an optional hard cap on segment length, and a symmetric pad grown around
//! every surviving segment.
//!
//! # The rules interact, and the order is Silero's
//!
//! The branches below are ported statement by statement, including two places
//! where Silero's own control flow is asymmetric (one of the three
//! max-speech sub-branches skips the rest of the iteration, the other two fall
//! through) and one where a variable is read and then zeroed a few lines later.
//! Tidying either changes which boundaries come out, so both are reproduced as
//! written and called out in comments where they occur.

use crate::error::{Error, Result};
use crate::vad::options::{SpeechSegment, VadSegmentOptions};

/// Turn per-chunk speech probabilities into utterance boundaries.
///
/// `probs[i]` is the probability for the chunk starting at sample
/// `i * window_size`, so exactly `ceil(num_samples / window_size)` values are
/// expected — Silero evaluates a ZERO-PADDED final partial chunk rather than
/// dropping it. Note that `SileroVad::probabilities` deliberately drops that
/// trailing partial chunk instead (its own doc comment says so, and a test
/// pins it), so its output is one value short for any signal whose length is
/// not a multiple of the chunk size. Use
/// [`speech_timestamps`](crate::vad::speech_timestamps), which does the padding.
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
    opts.validate()?;
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
        // negative, and Silero compares against it as a plain number.
        if triggered
            && let Some(seg) = current
            && (cur_sample as f64 - seg.start as f64) > max_speech_samples
        {
            // Python's `max` keeps the FIRST maximum on ties, so this reduces
            // with a strict `>` rather than using `max_by_key`, which keeps
            // the last. `None` here is Silero's empty-candidate case.
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
                // Silero binds `prev_end` from the tuple here, uses it for
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
                // Silero's, not a transcription slip.
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

/// Unit tests for the segmentation rules, driven by hand-written probability
/// arrays — no checkpoint, no client, no tensors.
///
/// Parity with Silero's `get_speech_timestamps` over real audio lives in
/// `tests/silero_vad_segment_parity.rs`, which needs the weights and the
/// reference JSON. These tests pin the individual rules instead, one per case,
/// so a regression names which rule broke.
///
/// Geometry throughout: 16 kHz, 512-sample chunks. With the default options
/// that makes `min_speech` 4000 samples (7.8 chunks), `min_silence` 1600
/// samples (3.1 chunks) and the pad 480 samples. A silence closes a segment
/// only on a silent chunk where the elapsed silence has ALREADY reached
/// `min_silence`, which is why the multi-chunk gaps below are sized as they
/// are.
#[cfg(test)]
mod tests {
    use super::*;

    const RATE: usize = 16000;
    const WINDOW: usize = 512;

    /// `len` chunks of silence, with every listed half-open chunk range set to a
    /// confident speech probability.
    fn probs(len: usize, speech: &[(usize, usize)]) -> Vec<f32> {
        let mut out = vec![0.0f32; len];
        for &(from, to) in speech {
            for p in &mut out[from..to] {
                *p = 0.9;
            }
        }
        out
    }

    fn segment(probs: &[f32], opts: &VadSegmentOptions) -> Vec<SpeechSegment> {
        segments_from_probabilities(probs, probs.len() * WINDOW, RATE, WINDOW, opts)
            .expect("valid segmentation inputs")
    }

    fn pairs(segments: &[SpeechSegment]) -> Vec<(usize, usize)> {
        segments.iter().map(|s| (s.start, s.end)).collect()
    }

    #[test]
    fn one_speech_run_becomes_one_padded_segment() {
        let p = probs(60, &[(10, 30)]);
        let got = segment(&p, &VadSegmentOptions::default());
        // Speech spans samples [5120, 15360); the pad adds 480 on each side.
        assert_eq!(pairs(&got), vec![(4640, 15840)]);
    }

    #[test]
    fn a_silence_longer_than_min_silence_splits_the_run() {
        let p = probs(70, &[(10, 30), (40, 60)]);
        let got = segment(&p, &VadSegmentOptions::default());
        assert_eq!(pairs(&got), vec![(4640, 15840), (20000, 31200)]);
    }

    #[test]
    fn a_silence_shorter_than_min_silence_does_not_split() {
        // Two chunks of silence is 1024 samples of elapsed gap at the last silent
        // chunk, under the 1600-sample minimum, so the segment stays open.
        let p = probs(60, &[(10, 30), (32, 50)]);
        let got = segment(&p, &VadSegmentOptions::default());
        assert_eq!(pairs(&got), vec![(4640, 26080)]);
    }

    #[test]
    fn a_run_shorter_than_min_speech_is_dropped() {
        // 5 chunks is 2560 samples, under the 4000-sample minimum.
        let p = probs(40, &[(10, 15)]);
        let got = segment(&p, &VadSegmentOptions::default());
        assert!(got.is_empty(), "expected no segments, got {got:?}");
    }

    #[test]
    fn speech_running_to_the_end_of_the_signal_is_closed_at_the_last_sample() {
        let p = probs(30, &[(10, 30)]);
        let got = segment(&p, &VadSegmentOptions::default());
        // The trailing pad clamps to the signal rather than running past it.
        assert_eq!(pairs(&got), vec![(4640, 15360)]);
    }

    #[test]
    fn all_silence_yields_no_segments() {
        let p = probs(50, &[]);
        let got = segment(&p, &VadSegmentOptions::default());
        assert!(got.is_empty(), "expected no segments, got {got:?}");
    }

    #[test]
    fn a_gap_under_two_pads_is_split_between_the_neighbours() {
        // A 100 ms pad is 1600 samples, so two pads need a 3200-sample gap. The
        // gap here is 2560, so each side takes half and the segments meet.
        let opts = VadSegmentOptions {
            speech_pad_ms: 100,
            ..VadSegmentOptions::default()
        };
        let p = probs(65, &[(10, 30), (35, 55)]);
        let got = segment(&p, &opts);
        assert_eq!(pairs(&got), vec![(3520, 16640), (16640, 29760)]);
        assert_eq!(got[0].end, got[1].start, "the padded segments must meet");
    }

    #[test]
    fn a_probability_count_that_does_not_match_the_signal_is_rejected() {
        let opts = VadSegmentOptions::default();
        // 1000 samples at a 512-sample window is 2 chunks, not 1: Silero scores
        // a zero-padded final partial chunk.
        let err = segments_from_probabilities(&[0.9], 1000, RATE, WINDOW, &opts)
            .expect_err("one probability cannot cover 1000 samples");
        let message = err.to_string();
        assert!(message.contains("1000"), "{message}");
        assert!(message.contains("512"), "{message}");
    }

    #[test]
    fn degenerate_geometry_is_rejected() {
        let opts = VadSegmentOptions::default();
        assert!(segments_from_probabilities(&[], 0, 0, WINDOW, &opts).is_err());
        assert!(segments_from_probabilities(&[], 0, RATE, 0, &opts).is_err());
        // Option validation runs here too, not only on `validate()`.
        let bad = VadSegmentOptions {
            threshold: f32::NAN,
            ..VadSegmentOptions::default()
        };
        assert!(segments_from_probabilities(&[], 0, RATE, WINDOW, &bad).is_err());
    }

    /// Pins the `possible_ends` tie-break, which the checkpoint-backed parity cases
    /// do NOT exercise — verified by sabotage: swapping the strict-`>` reduce for
    /// `max_by_key` (which keeps the LAST maximum instead of Python's first) leaves
    /// every case in `silero_vad_timestamps.json` passing.
    ///
    /// Two silences of IDENTICAL duration sit inside one over-long speech run, so
    /// the max-speech cut lands on whichever one the tie-break picks:
    ///   first maximum  -> cut at 2560, resume at 5120
    ///   last maximum   -> cut at 7680, resume at 10240
    ///
    /// Chunks (512 samples each): 0-4 speech, 5-9 silence, 10-14 speech,
    /// 15-19 silence, 20-29 speech. `min_silence_duration_ms` is set far above the
    /// gaps so a silence never CLOSES a segment — it only accumulates a candidate —
    /// and `max_speech_samples` works out to 12288, first exceeded at chunk 25.
    #[test]
    fn the_max_speech_cut_keeps_the_first_of_two_equal_silences() {
        let mut probs = vec![0.9f32; 30];
        probs[5..10].fill(0.1);
        probs[15..20].fill(0.1);
        let num_samples = 30 * WINDOW;

        let opts = VadSegmentOptions {
            // 16000 * 0.8 - 512 - 0 = 12288 samples.
            max_speech_duration_s: 0.8,
            // Far larger than either 2560-sample gap, so neither closes a segment.
            min_silence_duration_ms: 10_000,
            min_speech_duration_ms: 0,
            speech_pad_ms: 0,
            ..VadSegmentOptions::default()
        };

        let got =
            segments_from_probabilities(&probs, num_samples, RATE, WINDOW, &opts).expect("segment");
        assert_eq!(
            pairs(&got),
            vec![(0, 2560), (5120, 15360)],
            "tie-break picked the later silence"
        );
    }
}
