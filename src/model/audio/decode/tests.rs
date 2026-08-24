//! Tests for the symphonia-backed decode path.
//!
//! The highest-value test here is [`agrees_with_native_wav_decoder`]: it feeds
//! the same bytes through symphonia and through boostr's own hand-written
//! `decode_wav`, so a bug in either one shows up as a disagreement rather than
//! passing silently.

use super::*;
use crate::model::audio::resample::{resample, to_mono_at_rate};
use crate::model::audio::wav_decode::{decode_wav, to_mono};
use crate::model::audio::wav_encode::{
    encode_wav_f32, encode_wav_pcm16, encode_wav_pcm16_multichannel,
};
use crate::test_utils::corpus_flac;

fn tone(freq: f64, rate: u32, len: usize, channels: u16) -> Vec<f32> {
    let mut out = Vec::with_capacity(len * channels as usize);
    for n in 0..len {
        let s = (std::f64::consts::TAU * freq * n as f64 / rate as f64).sin() as f32;
        for _ in 0..channels {
            out.push(s);
        }
    }
    out
}

#[test]
fn agrees_with_native_wav_decoder() {
    let rate = 22_050u32;
    let samples = tone(440.0, rate, 2000, 1);

    let pcm16 = encode_wav_pcm16(&samples, rate).expect("encode pcm16");
    let via_symphonia = decode_audio(&pcm16, Some("wav")).expect("symphonia decode");
    let via_native = decode_wav(&pcm16).expect("native decode");

    assert_eq!(via_symphonia.sample_rate, via_native.sample_rate);
    assert_eq!(via_symphonia.channels, via_native.channels);
    assert_eq!(via_symphonia.samples.len(), via_native.samples.len());
    // EXACT, not a tolerance. Both paths read the same i16 values and both
    // normalise through `wav_decode::scale`, so any difference means the two
    // decoders have drifted apart on the integer -> float convention. A loose
    // tolerance here would silently absorb exactly that bug: swapping one side
    // to divide by 32768 instead of i16::MAX moves samples by only ~3e-5.
    assert_eq!(via_symphonia.samples, via_native.samples);

    let f32wav = encode_wav_f32(&samples, rate).expect("encode f32");
    let via_symphonia = decode_audio(&f32wav, Some("wav")).expect("symphonia decode f32");
    let via_native = decode_wav(&f32wav).expect("native decode f32");
    assert_eq!(via_symphonia.sample_rate, via_native.sample_rate);
    assert_eq!(via_symphonia.channels, via_native.channels);
    for (a, b) in via_symphonia.samples.iter().zip(via_native.samples.iter()) {
        assert!((a - b).abs() < 1e-6, "f32 mismatch: {a} vs {b}");
    }
}

#[test]
fn mono_at_matches_separate_downmix_and_resample() {
    let rate = 44_100u32;
    let target = 16_000u32;
    let samples = tone(220.0, rate, 4000, 2);
    let wav = encode_wav_pcm16_multichannel(&samples, rate, 2).expect("encode stereo");

    let via_one_step = decode_audio_mono_at(&wav, Some("wav"), target).expect("decode mono_at");

    let decoded = decode_audio(&wav, Some("wav")).expect("decode");
    let mono = to_mono(&decoded.samples, decoded.channels).expect("to_mono");
    let via_two_step = resample(&mono, decoded.sample_rate, target).expect("resample");

    assert_eq!(via_one_step, via_two_step);
}

#[test]
fn extension_hint_parses() {
    assert_eq!(extension_hint("a.wav"), Some("wav"));
    assert_eq!(extension_hint("foo.bar.mp3"), Some("mp3"));
    assert_eq!(extension_hint("noext"), None);
}

#[test]
fn garbage_bytes_error_not_panic() {
    assert!(decode_audio(&[], None).is_err());
    assert!(decode_audio(&[0u8; 4], None).is_err());
    assert!(decode_audio(b"not an audio file at all, just text", None).is_err());
}

/// The file path streams through `MediaSourceStream` instead of an in-memory
/// buffer, so it is a genuinely different read path from `decode_audio` and can
/// diverge on its own. Pin it against the byte path on identical content.
#[test]
fn file_path_matches_the_byte_path() {
    let rate = 32_000u32;
    let target = 16_000u32;
    let samples = tone(1000.0, rate, 3000, 1);
    let wav = encode_wav_pcm16(&samples, rate).expect("encode pcm16");

    let dir = std::env::temp_dir().join(format!("boostr-decode-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    let path = dir.join("tone.wav");
    std::fs::write(&path, &wav).expect("write scratch wav");

    let via_file = decode_audio_file_mono_at(&path, target).expect("decode from file");
    let via_bytes = decode_audio_mono_at(&wav, Some("wav"), target).expect("decode from bytes");
    assert_eq!(via_file, via_bytes);

    std::fs::remove_dir_all(&dir).expect("clean scratch dir");
}

#[test]
fn real_corpus_decodes_at_native_rate_and_resamples() {
    let Some(path) = corpus_flac() else { return };

    // One decode only — a corpus file is tens of megabytes. Everything below is
    // derived from what this file actually declares, so any real .flac works.
    let bytes = std::fs::read(&path).expect("read corpus flac");
    let decoded = decode_audio(&bytes, Some("flac")).expect("decode corpus flac");
    let native_rate = decoded.sample_rate;
    assert!(native_rate > 0, "a decoded file must declare a sample rate");
    assert!(
        decoded.channels >= 1,
        "a decoded file must declare channels"
    );
    let frames = decoded.frames();
    assert!(frames > 0, "the corpus fixture must not be empty");

    let target = 16_000u32;
    let resampled =
        to_mono_at_rate(&decoded, target).expect("downmix + resample the decoded corpus");
    let expected_len = (frames as u64 * target as u64).div_ceil(native_rate as u64) as usize;
    assert!(
        resampled.len().abs_diff(expected_len) <= 1,
        "resampled length {} not within 1 of expected {expected_len}",
        resampled.len()
    );
    assert!(
        resampled.iter().all(|s| s.is_finite() && s.abs() <= 1.5),
        "resampled corpus audio must stay finite and in range"
    );
}
