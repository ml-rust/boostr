//! Minimal RIFF/WAVE encoder for raw mono/stereo PCM.
//!
//! The TTS path produces `Vec<f32>` waveform samples in `[-1, 1]` at the model's
//! native sample rate. This module converts those to either 16-bit signed PCM
//! (small, widely supported) or 32-bit float PCM (lossless) inside a WAV container.

use crate::error::{Error, Result};

use super::wav_format::{FORMAT_IEEE_FLOAT, FORMAT_PCM};

fn bad(arg: &'static str, reason: String) -> Error {
    Error::InvalidArgument { arg, reason }
}

/// Encode mono f32 samples as `audio/wav` with signed 16-bit PCM.
///
/// Samples outside `[-1, 1]` are clipped. Returns a standalone byte buffer
/// suitable as an HTTP response body, or [`Error::InvalidArgument`] when the
/// sample count is too large for the WAV container's 32-bit size fields.
pub fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    encode_wav_pcm16_multichannel(samples, sample_rate, 1)
}

/// Encode interleaved multi-channel f32 samples as 16-bit PCM WAV.
///
/// `samples.len()` must be divisible by `channels`. Returns
/// [`Error::InvalidArgument`] when `channels` is 0, when `samples.len()` is not
/// divisible by `channels`, or when the sample count is too large for the WAV
/// container's 32-bit size fields.
pub fn encode_wav_pcm16_multichannel(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<Vec<u8>> {
    if channels == 0 {
        return Err(bad("channels", "channel count is 0".to_string()));
    }
    if !samples.len().is_multiple_of(channels as usize) {
        return Err(bad(
            "samples",
            format!(
                "sample count {} is not divisible by the channel count {channels}",
                samples.len()
            ),
        ));
    }

    let bits_per_sample = 16u16;
    let (byte_rate, block_align) = checked_wave_rates(sample_rate, channels, bits_per_sample)?;
    let data_size = checked_data_size(samples.len(), 2)?; // 2 bytes per i16 sample
    let riff_size = checked_riff_size(samples.len(), data_size)?;

    let mut out = Vec::with_capacity(44 + data_size as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&FORMAT_PCM.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());

    for &s in samples {
        let clipped = s.clamp(-1.0, 1.0);
        let i = (clipped * i16::MAX as f32).round() as i16;
        out.extend_from_slice(&i.to_le_bytes());
    }
    Ok(out)
}

/// Encode mono f32 samples as `audio/wav` with 32-bit float PCM (no clipping).
///
/// Returns [`Error::InvalidArgument`] when the sample count is too large for the
/// WAV container's 32-bit size fields.
pub fn encode_wav_f32(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let channels = 1u16;
    let bits_per_sample = 32u16;
    let (byte_rate, block_align) = checked_wave_rates(sample_rate, channels, bits_per_sample)?;
    let data_size = checked_data_size(samples.len(), 4)?;
    let riff_size = checked_riff_size(samples.len(), data_size)?;

    let mut out = Vec::with_capacity(44 + data_size as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&FORMAT_IEEE_FLOAT.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());

    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    Ok(out)
}

/// Encode mono f32 samples as raw little-endian PCM16 (no WAV header).
///
/// Returned bytes are suitable for `response_format=pcm` streaming.
pub fn encode_pcm16_raw(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let i = (s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        out.extend_from_slice(&i.to_le_bytes());
    }
    out
}

/// `fmt ` chunk rate fields: `byte_rate` (bytes per second) and `block_align`
/// (bytes per frame), both checked against the widths the header stores them in.
///
/// `block_align` is a `u16`, so it overflows above 32767 channels at 16 bits, and
/// `byte_rate` is a `u32`, so a large `sample_rate` and channel count together
/// overflow it. Either would silently emit a header describing a different file.
fn checked_wave_rates(sample_rate: u32, channels: u16, bits_per_sample: u16) -> Result<(u32, u16)> {
    let bytes_per_sample = bits_per_sample / 8;
    let block_align = channels.checked_mul(bytes_per_sample).ok_or_else(|| {
        bad(
            "channels",
            format!(
                "channel count {channels} at {bits_per_sample} bits per sample overflows \
                 the WAV header's u16 block alignment"
            ),
        )
    })?;
    let byte_rate = sample_rate.checked_mul(block_align as u32).ok_or_else(|| {
        bad(
            "sample_rate",
            format!(
                "sample rate {sample_rate} at {block_align} bytes per frame overflows \
                 the WAV header's u32 byte rate"
            ),
        )
    })?;
    Ok((byte_rate, block_align))
}

/// `data` chunk size in bytes: `sample_count * bytes_per_sample`, checked against
/// the WAV container's 32-bit size field.
fn checked_data_size(sample_count: usize, bytes_per_sample: usize) -> Result<u32> {
    sample_count
        .checked_mul(bytes_per_sample)
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or_else(|| {
            bad(
                "samples",
                format!(
                    "sample count {sample_count} is too large to encode as a WAV file \
                     ({sample_count} x {bytes_per_sample} bytes overflows a u32 data size)"
                ),
            )
        })
}

/// `RIFF` chunk size in bytes: 36 (fixed header) + `data_size`, checked against
/// the WAV container's 32-bit size field.
fn checked_riff_size(sample_count: usize, data_size: u32) -> Result<u32> {
    36u32.checked_add(data_size).ok_or_else(|| {
        bad(
            "samples",
            format!(
                "sample count {sample_count} produces a {data_size}-byte data chunk, \
                 too large to encode as a WAV file (36 + data_size overflows a u32 RIFF size)"
            ),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm16_header_shape() {
        let wav = encode_wav_pcm16(&[0.0; 10], 24_000).unwrap();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        // fmt chunk size = 16
        assert_eq!(u32::from_le_bytes(wav[16..20].try_into().unwrap()), 16);
        // PCM format = 1
        assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 1);
        // channels = 1
        assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1);
        // sample rate
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 24_000);
        // bits per sample = 16
        assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 16);
        assert_eq!(&wav[36..40], b"data");
        // data_size = 10 * 2
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 20);
    }

    #[test]
    fn pcm16_clips_out_of_range() {
        let wav = encode_wav_pcm16(&[2.0, -2.0], 8_000).unwrap();
        // Samples live at offset 44, 2 bytes each.
        let s0 = i16::from_le_bytes(wav[44..46].try_into().unwrap());
        let s1 = i16::from_le_bytes(wav[46..48].try_into().unwrap());
        assert_eq!(s0, i16::MAX);
        assert_eq!(s1, -i16::MAX);
    }

    #[test]
    fn pcm16_roundtrips_within_quantization() {
        let input = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let wav = encode_wav_pcm16(&input, 16_000).unwrap();
        let mut decoded = Vec::with_capacity(input.len());
        for chunk in wav[44..].as_chunks::<2>().0 {
            let s = i16::from_le_bytes(*chunk);
            decoded.push(s as f32 / i16::MAX as f32);
        }
        for (a, b) in input.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1.0 / i16::MAX as f32 + 1e-6);
        }
    }

    #[test]
    fn f32_header_uses_ieee_format() {
        let wav = encode_wav_f32(&[0.25], 48_000).unwrap();
        // Format code = 3 (IEEE float)
        assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 3);
        assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 32);
        // The single sample is bit-identical.
        let s = f32::from_le_bytes(wav[44..48].try_into().unwrap());
        assert_eq!(s, 0.25);
    }

    #[test]
    fn raw_pcm16_has_no_header() {
        let raw = encode_pcm16_raw(&[0.0, 1.0]);
        assert_eq!(raw.len(), 4);
        let s1 = i16::from_le_bytes(raw[2..4].try_into().unwrap());
        assert_eq!(s1, i16::MAX);
    }

    #[test]
    fn multichannel_rejects_zero_channels() {
        assert!(encode_wav_pcm16_multichannel(&[0.0, 1.0], 8_000, 0).is_err());
    }

    #[test]
    fn multichannel_rejects_indivisible_sample_count() {
        assert!(encode_wav_pcm16_multichannel(&[0.0, 1.0, 0.5], 8_000, 2).is_err());
    }
    /// `channels * bytes_per_sample` is stored in a `u16`. Above 32767 channels at
    /// 16 bits it wraps, and the header would declare a block alignment smaller
    /// than one frame while the data chunk stays full size.
    #[test]
    fn rejects_a_channel_count_that_overflows_block_align() {
        let err = encode_wav_pcm16_multichannel(&[], 16_000, 40_000)
            .expect_err("40000 channels at 16 bits overflows the u16 block alignment");
        let msg = err.to_string();
        assert!(msg.contains("40000"), "{msg}");
        assert!(msg.contains("block alignment"), "{msg}");
    }

    /// `sample_rate * block_align` is stored in a `u32`, so an extreme sample rate
    /// wraps it and the header would declare a byte rate far below the real one.
    #[test]
    fn rejects_a_sample_rate_that_overflows_byte_rate() {
        let err = encode_wav_pcm16(&[0.0; 4], u32::MAX)
            .expect_err("u32::MAX Hz at 2 bytes per frame overflows the u32 byte rate");
        let msg = err.to_string();
        assert!(msg.contains(&u32::MAX.to_string()), "{msg}");
        assert!(msg.contains("byte rate"), "{msg}");
    }

    /// The largest block alignment that still fits: 32767 channels at 16 bits is
    /// exactly `u16::MAX - 1`, so the guard above must not reject it.
    #[test]
    fn accepts_the_largest_representable_block_align() {
        let wav = encode_wav_pcm16_multichannel(&[], 8_000, 32_767)
            .expect("32767 channels at 16 bits fits a u16 block alignment");
        assert_eq!(u16::from_le_bytes(wav[32..34].try_into().unwrap()), 65_534);
    }
}
