//! Unit tests for the segmentation rules, driven by hand-written probability
//! arrays — no checkpoint, no client, no tensors.
//!
//! Parity with upstream's `get_speech_timestamps` over real audio lives in
//! `tests/silero_vad_segment_parity.rs`, which needs the weights and the
//! reference JSON. These tests pin the individual rules instead, one per case,
//! so a regression names which rule broke.
//!
//! Geometry throughout: 16 kHz, 512-sample chunks. With the default options
//! that makes `min_speech` 4000 samples (7.8 chunks), `min_silence` 1600
//! samples (3.1 chunks) and the pad 480 samples. A silence closes a segment
//! only on a silent chunk where the elapsed silence has ALREADY reached
//! `min_silence`, which is why the multi-chunk gaps below are sized as they
//! are.

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
    assert_eq!(got[0].len(), 11200);
    assert!(!got[0].is_empty());
    assert!((got[0].duration_secs(RATE) - 0.7).abs() < 1e-9);
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
    // 1000 samples at a 512-sample window is 2 chunks, not 1: upstream scores
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
    let bad = VadSegmentOptions {
        threshold: 0.0,
        ..VadSegmentOptions::default()
    };
    assert!(segments_from_probabilities(&[], 0, RATE, WINDOW, &bad).is_err());
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
    let got: Vec<(usize, usize)> = got.iter().map(|s| (s.start, s.end)).collect();
    assert_eq!(
        got,
        vec![(0, 2560), (5120, 15360)],
        "tie-break picked the later silence"
    );
}

/// A `neg_threshold` of NaN does not panic — every `prob < neg_threshold` test
/// silently returns false, so no silence ever closes a segment and a whole
/// recording comes back as one utterance. It has to be refused up front.
#[test]
fn a_nonsensical_neg_threshold_is_rejected() {
    let probs = vec![0.9f32; 4];
    let n = 4 * WINDOW;
    for neg in [f32::NAN, 0.0, -0.1, 1.5] {
        let bad = VadSegmentOptions {
            neg_threshold: Some(neg),
            ..VadSegmentOptions::default()
        };
        assert!(
            segments_from_probabilities(&probs, n, RATE, WINDOW, &bad).is_err(),
            "neg_threshold {neg} must be refused"
        );
    }
    // At or above `threshold` is contradictory: the hysteresis inverts.
    let bad = VadSegmentOptions {
        threshold: 0.5,
        neg_threshold: Some(0.5),
        ..VadSegmentOptions::default()
    };
    assert!(segments_from_probabilities(&probs, n, RATE, WINDOW, &bad).is_err());

    // The derived default and any value below the threshold stay accepted.
    let good = VadSegmentOptions {
        neg_threshold: Some(0.35),
        ..VadSegmentOptions::default()
    };
    assert!(segments_from_probabilities(&probs, n, RATE, WINDOW, &good).is_ok());
    assert!(
        segments_from_probabilities(&probs, n, RATE, WINDOW, &VadSegmentOptions::default()).is_ok()
    );
}

/// A negative or NaN cap does not panic either: it makes `max_speech_samples`
/// smaller than any real segment, so the max-speech split fires on nearly every
/// chunk and shreds the output.
#[test]
fn a_nonsensical_max_speech_duration_is_rejected() {
    let probs = vec![0.9f32; 4];
    let n = 4 * WINDOW;
    for cap in [f32::NAN, 0.0, -1.0] {
        let bad = VadSegmentOptions {
            max_speech_duration_s: cap,
            ..VadSegmentOptions::default()
        };
        assert!(
            segments_from_probabilities(&probs, n, RATE, WINDOW, &bad).is_err(),
            "max_speech_duration_s {cap} must be refused"
        );
    }
    // Infinity is the default and means "no cap".
    assert!(
        VadSegmentOptions::default()
            .max_speech_duration_s
            .is_infinite()
    );
    assert!(
        segments_from_probabilities(&probs, n, RATE, WINDOW, &VadSegmentOptions::default()).is_ok()
    );
}
