//! Round-trip, fixture, and malformed-input tests for the WAV decoder.

use super::*;
use crate::model::audio::wav_encode::{
    encode_wav_f32, encode_wav_pcm16, encode_wav_pcm16_multichannel,
};

/// Assemble a RIFF/WAVE file from `(chunk id, body)` pairs, applying the pad rule.
fn build(chunks: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(b"WAVE");
    for (id, body) in chunks {
        payload.extend_from_slice(*id);
        payload.extend_from_slice(&(body.len() as u32).to_le_bytes());
        payload.extend_from_slice(body);
        if body.len() % 2 == 1 {
            payload.push(0);
        }
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

/// A standard 16-byte `fmt ` body.
fn fmt_body(tag: u16, channels: u16, sample_rate: u32, bits: u16) -> Vec<u8> {
    let block_align = channels * (bits / 8);
    let byte_rate = sample_rate * block_align as u32;
    let mut b = Vec::new();
    b.extend_from_slice(&tag.to_le_bytes());
    b.extend_from_slice(&channels.to_le_bytes());
    b.extend_from_slice(&sample_rate.to_le_bytes());
    b.extend_from_slice(&byte_rate.to_le_bytes());
    b.extend_from_slice(&block_align.to_le_bytes());
    b.extend_from_slice(&bits.to_le_bytes());
    b
}

/// A 40-byte `WAVE_FORMAT_EXTENSIBLE` body wrapping `sub_tag`.
fn fmt_body_extensible(sub_tag: u16, channels: u16, sample_rate: u32, bits: u16) -> Vec<u8> {
    let mut b = fmt_body(FORMAT_EXTENSIBLE, channels, sample_rate, bits);
    b.extend_from_slice(&22u16.to_le_bytes()); // cbSize
    b.extend_from_slice(&bits.to_le_bytes()); // validBitsPerSample
    b.extend_from_slice(&4u32.to_le_bytes()); // dwChannelMask
    b.extend_from_slice(&sub_tag.to_le_bytes()); // SubFormat GUID, first two bytes
    b.extend_from_slice(&[0u8; 14]); // rest of the GUID
    b
}

#[test]
fn pcm16_round_trip_within_one_step() {
    let input = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.123_45, -0.987];
    let wav = encode_wav_pcm16(&input, 16_000).unwrap();
    let decoded = decode_wav(&wav).unwrap();
    assert_eq!(decoded.sample_rate, 16_000);
    assert_eq!(decoded.channels, 1);
    assert_eq!(decoded.samples.len(), input.len());
    assert_eq!(decoded.frames(), input.len());
    for (a, b) in input.iter().zip(decoded.samples.iter()) {
        assert!((a - b).abs() <= 1.0 / 32_768.0, "{a} vs {b}");
    }
}

#[test]
fn f32_round_trip_is_exact() {
    let input = vec![0.0, 0.25, -0.125, 1.0, -1.0, 3.5, -2.75, f32::MIN_POSITIVE];
    let wav = encode_wav_f32(&input, 48_000).unwrap();
    let decoded = decode_wav(&wav).unwrap();
    assert_eq!(decoded.sample_rate, 48_000);
    assert_eq!(decoded.channels, 1);
    assert_eq!(decoded.samples, input);
}

#[test]
fn stereo_round_trip_then_downmix() {
    // Frames: (1.0, -1.0), (0.5, 0.5), (0.0, 1.0)
    let input = vec![1.0, -1.0, 0.5, 0.5, 0.0, 1.0];
    let wav = encode_wav_pcm16_multichannel(&input, 44_100, 2).unwrap();
    let decoded = decode_wav(&wav).unwrap();
    assert_eq!(decoded.channels, 2);
    assert_eq!(decoded.sample_rate, 44_100);
    assert_eq!(decoded.frames(), 3);

    let mono = to_mono(&decoded.samples, decoded.channels).unwrap();
    assert_eq!(mono.len(), 3);
    let expected = [0.0, 0.5, 0.5];
    for (a, b) in expected.iter().zip(mono.iter()) {
        assert!((a - b).abs() <= 1.0 / 32_768.0, "{a} vs {b}");
    }
}

#[test]
fn decodes_24_bit_pcm() {
    let mut data = Vec::new();
    for raw in [0x00_0000i32, 0x7F_FFFF, -0x80_0000, 0x40_0000] {
        let bytes = raw.to_le_bytes();
        data.extend_from_slice(&bytes[0..3]);
    }
    let wav = build(&[
        (b"fmt ", fmt_body(FORMAT_PCM, 1, 16_000, 24)),
        (b"data", data),
    ]);
    let decoded = decode_wav(&wav).unwrap();
    assert_eq!(decoded.channels, 1);
    assert_eq!(decoded.samples.len(), 4);
    assert_eq!(decoded.samples[0], 0.0);
    assert_eq!(decoded.samples[1], 1.0);
    assert_eq!(decoded.samples[2], -1.0); // clamped from -1.0000001
    assert!((decoded.samples[3] - 0.5).abs() < 1e-6);
}

#[test]
fn decodes_32_bit_int_pcm() {
    let mut data = Vec::new();
    for raw in [0i32, i32::MAX, i32::MIN, i32::MAX / 2] {
        data.extend_from_slice(&raw.to_le_bytes());
    }
    let wav = build(&[
        (b"fmt ", fmt_body(FORMAT_PCM, 1, 8_000, 32)),
        (b"data", data),
    ]);
    let decoded = decode_wav(&wav).unwrap();
    assert_eq!(decoded.samples.len(), 4);
    assert_eq!(decoded.samples[0], 0.0);
    assert_eq!(decoded.samples[1], 1.0);
    assert_eq!(decoded.samples[2], -1.0);
    assert!((decoded.samples[3] - 0.5).abs() < 1e-6);
}

#[test]
fn parses_wave_format_extensible_pcm16() {
    let data: Vec<u8> = [0i16, i16::MAX, -16_384]
        .iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();
    let wav = build(&[
        (b"fmt ", fmt_body_extensible(FORMAT_PCM, 1, 22_050, 16)),
        (b"data", data),
    ]);
    let decoded = decode_wav(&wav).unwrap();
    assert_eq!(decoded.sample_rate, 22_050);
    assert_eq!(decoded.channels, 1);
    assert_eq!(decoded.samples[0], 0.0);
    assert_eq!(decoded.samples[1], 1.0);
    assert!((decoded.samples[2] + 0.5).abs() < 1e-4);
}

#[test]
fn parses_wave_format_extensible_float() {
    let data: Vec<u8> = [0.25f32, -0.75]
        .iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();
    let wav = build(&[
        (
            b"fmt ",
            fmt_body_extensible(FORMAT_IEEE_FLOAT, 1, 16_000, 32),
        ),
        (b"data", data),
    ]);
    let decoded = decode_wav(&wav).unwrap();
    assert_eq!(decoded.samples, vec![0.25, -0.75]);
}

#[test]
fn skips_list_and_fact_chunks_between_fmt_and_data() {
    let data: Vec<u8> = [i16::MAX, i16::MIN + 1]
        .iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();
    let mut list = b"INFO".to_vec();
    list.extend_from_slice(b"ISFTboostr wav test\0\0");
    let wav = build(&[
        (b"fmt ", fmt_body(FORMAT_PCM, 1, 16_000, 16)),
        (b"fact", 2u32.to_le_bytes().to_vec()),
        (b"LIST", list),
        (b"data", data),
    ]);
    let decoded = decode_wav(&wav).unwrap();
    assert_eq!(decoded.samples.len(), 2);
    assert_eq!(decoded.samples[0], 1.0);
    assert_eq!(decoded.samples[1], -1.0);
}

#[test]
fn skips_odd_sized_chunk_with_its_pad_byte() {
    let data: Vec<u8> = 0.5f32.to_le_bytes().to_vec();
    // `build` appends the pad byte for the odd 5-byte body.
    let wav = build(&[
        (b"fmt ", fmt_body(FORMAT_IEEE_FLOAT, 1, 16_000, 32)),
        (b"junk", vec![1, 2, 3, 4, 5]),
        (b"data", data),
    ]);
    let decoded = decode_wav(&wav).unwrap();
    assert_eq!(decoded.samples, vec![0.5]);
}

#[test]
fn missing_pad_byte_after_odd_chunk_is_rejected_not_panicking() {
    // Hand-built without the pad byte: the following chunk header is misaligned,
    // so the walk must stop with an error rather than index out of bounds.
    let mut payload = Vec::new();
    payload.extend_from_slice(b"WAVE");
    payload.extend_from_slice(b"junk");
    payload.extend_from_slice(&5u32.to_le_bytes());
    payload.extend_from_slice(&[1, 2, 3, 4, 5]);
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    wav.extend_from_slice(&payload);
    assert!(decode_wav(&wav).is_err());
}

// --- malformed inputs: every one returns Err, none panics ---

#[test]
fn rejects_empty_input() {
    assert!(decode_wav(&[]).is_err());
}

#[test]
fn rejects_truncated_header() {
    let wav = encode_wav_pcm16(&[0.0; 8], 16_000).unwrap();
    for len in 0..12 {
        assert!(decode_wav(&wav[..len]).is_err(), "len {len} must fail");
    }
    // Truncated mid-fmt and mid-data too.
    for len in 12..wav.len() {
        // Never panics; a short data chunk fails its declared size check.
        let _ = decode_wav(&wav[..len]);
    }
}

#[test]
fn rejects_bad_magic() {
    let mut wav = encode_wav_pcm16(&[0.0; 4], 16_000).unwrap();
    wav[0] = b'X';
    assert!(decode_wav(&wav).is_err());

    let mut wav = encode_wav_pcm16(&[0.0; 4], 16_000).unwrap();
    wav[8] = b'X';
    assert!(decode_wav(&wav).is_err());
}

#[test]
fn rejects_zero_channels() {
    let wav = build(&[
        (b"fmt ", fmt_body(FORMAT_PCM, 0, 16_000, 16)),
        (b"data", vec![0, 0, 0, 0]),
    ]);
    assert!(decode_wav(&wav).is_err());
}

#[test]
fn rejects_unsupported_format_tag() {
    let wav = build(&[
        (b"fmt ", fmt_body(6, 1, 16_000, 16)), // A-law
        (b"data", vec![0, 0]),
    ]);
    assert!(decode_wav(&wav).is_err());
}

#[test]
fn rejects_bits_not_matching_format_tag() {
    let float_8 = build(&[
        (b"fmt ", fmt_body(FORMAT_IEEE_FLOAT, 1, 16_000, 64)),
        (b"data", vec![0; 8]),
    ]);
    assert!(decode_wav(&float_8).is_err());

    let pcm_8 = build(&[
        (b"fmt ", fmt_body(FORMAT_PCM, 1, 16_000, 8)),
        (b"data", vec![0; 4]),
    ]);
    assert!(decode_wav(&pcm_8).is_err());
}

#[test]
fn rejects_chunk_size_overrunning_buffer() {
    let mut wav = build(&[
        (b"fmt ", fmt_body(FORMAT_PCM, 1, 16_000, 16)),
        (b"data", vec![0, 0, 0, 0]),
    ]);
    // Overwrite the data chunk size (last 4 bytes before the body) with a lie.
    let size_off = wav.len() - 8;
    wav[size_off..size_off + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(decode_wav(&wav).is_err());
}

#[test]
fn rejects_data_length_not_whole_frames() {
    let wav = build(&[
        (b"fmt ", fmt_body(FORMAT_PCM, 2, 16_000, 16)),
        (b"data", vec![0; 6]), // 1.5 stereo frames
    ]);
    assert!(decode_wav(&wav).is_err());
}

#[test]
fn rejects_missing_fmt_or_data_chunk() {
    let no_data = build(&[(b"fmt ", fmt_body(FORMAT_PCM, 1, 16_000, 16))]);
    assert!(decode_wav(&no_data).is_err());

    let no_fmt = build(&[(b"data", vec![0, 0, 0, 0])]);
    assert!(decode_wav(&no_fmt).is_err());
}

#[test]
fn rejects_short_fmt_chunk() {
    let wav = build(&[(b"fmt ", vec![1, 0, 1, 0]), (b"data", vec![0, 0])]);
    assert!(decode_wav(&wav).is_err());

    let short_extensible = build(&[
        (b"fmt ", fmt_body(FORMAT_EXTENSIBLE, 1, 16_000, 16)),
        (b"data", vec![0, 0]),
    ]);
    assert!(decode_wav(&short_extensible).is_err());
}

#[test]
fn empty_data_chunk_decodes_to_no_samples() {
    let wav = build(&[
        (b"fmt ", fmt_body(FORMAT_PCM, 1, 16_000, 16)),
        (b"data", Vec::new()),
    ]);
    let decoded = decode_wav(&wav).unwrap();
    assert!(decoded.samples.is_empty());
    assert_eq!(decoded.frames(), 0);
}

// --- to_mono ---

#[test]
fn to_mono_passes_through_single_channel() {
    let input = vec![0.1, -0.2, 0.3];
    assert_eq!(to_mono(&input, 1).unwrap(), input);
}

#[test]
fn to_mono_averages_channels() {
    let input = vec![1.0, 0.0, 0.0, 1.0, -1.0, 1.0];
    assert_eq!(to_mono(&input, 2).unwrap(), vec![0.5, 0.5, 0.0]);

    let three = vec![0.0, 0.5, 1.0];
    let out = to_mono(&three, 3).unwrap();
    assert!((out[0] - 0.5).abs() < 1e-6);
}

#[test]
fn to_mono_rejects_zero_channels() {
    assert!(to_mono(&[0.0, 1.0], 0).is_err());
}

#[test]
fn to_mono_rejects_indivisible_sample_count() {
    assert!(to_mono(&[0.0, 1.0, 0.5], 2).is_err());
}

/// A parser fed hostile bytes must return `Err`, never panic.
///
/// The hand-written malformed cases above each probe one rejection path. This
/// sweeps mechanically instead: every truncation of a valid file, and every
/// single-byte corruption of its header region — the sizes, counts and offsets
/// the walker does arithmetic on. A panic here is a crash in whatever service
/// decodes a user-supplied file.
#[test]
fn never_panics_on_truncated_or_corrupted_input() {
    let valid = encode_wav_pcm16(&[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], 16_000).unwrap();

    // Every prefix, including the empty one.
    for cut in 0..=valid.len() {
        let _ = decode_wav(&valid[..cut]);
    }

    // Every single-byte value at every header position. The header carries the
    // chunk sizes and the frame geometry, so this is where a lying length turns
    // into an out-of-range read or an overflowing multiply.
    let header_end = valid.len().min(64);
    for pos in 0..header_end {
        for byte in [0x00u8, 0x01, 0x7F, 0x80, 0xFE, 0xFF] {
            let mut mutated = valid.clone();
            mutated[pos] = byte;
            let _ = decode_wav(&mutated);
        }
    }

    // Sizes at the extremes, written directly over the RIFF and data length
    // fields rather than reached by chance through single-byte mutation.
    for size in [u32::MAX, u32::MAX - 1, 0x8000_0000, 0] {
        let mut mutated = valid.clone();
        mutated[4..8].copy_from_slice(&size.to_le_bytes());
        let _ = decode_wav(&mutated);

        let mut mutated = valid.clone();
        let n = mutated.len();
        mutated[n - 4..].copy_from_slice(&size.to_le_bytes());
        let _ = decode_wav(&mutated);
    }
}
