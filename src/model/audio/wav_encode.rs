//! Minimal RIFF/WAVE encoder for raw mono/stereo PCM.
//!
//! The TTS path produces `Vec<f32>` waveform samples in `[-1, 1]` at the model's
//! native sample rate. This module converts those to either 16-bit signed PCM
//! (small, widely supported) or 32-bit float PCM (lossless) inside a WAV container.

/// Encode mono f32 samples as `audio/wav` with signed 16-bit PCM.
///
/// Samples outside `[-1, 1]` are clipped. Returns a standalone byte buffer
/// suitable as an HTTP response body.
pub fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    encode_wav_pcm16_multichannel(samples, sample_rate, 1)
}

/// Encode interleaved multi-channel f32 samples as 16-bit PCM WAV.
///
/// `samples.len()` must be divisible by `channels`.
pub fn encode_wav_pcm16_multichannel(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    assert!(channels >= 1, "channels must be >= 1");
    assert!(
        samples.len().is_multiple_of(channels as usize),
        "sample count not divisible by channel count"
    );

    let bits_per_sample = 16u16;
    let byte_rate = sample_rate * channels as u32 * (bits_per_sample / 8) as u32;
    let block_align = channels * (bits_per_sample / 8);
    let data_size = (samples.len() * 2) as u32; // 2 bytes per i16 sample
    let riff_size = 36 + data_size;

    let mut out = Vec::with_capacity(44 + data_size as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
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
    out
}

/// Encode mono f32 samples as `audio/wav` with 32-bit float PCM (no clipping).
pub fn encode_wav_f32(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let channels = 1u16;
    let bits_per_sample = 32u16;
    let byte_rate = sample_rate * channels as u32 * (bits_per_sample / 8) as u32;
    let block_align = channels * (bits_per_sample / 8);
    let data_size = (samples.len() * 4) as u32;
    let riff_size = 36 + data_size;

    let mut out = Vec::with_capacity(44 + data_size as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&3u16.to_le_bytes()); // IEEE float format code
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
    out
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm16_header_shape() {
        let wav = encode_wav_pcm16(&[0.0; 10], 24_000);
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
        let wav = encode_wav_pcm16(&[2.0, -2.0], 8_000);
        // Samples live at offset 44, 2 bytes each.
        let s0 = i16::from_le_bytes(wav[44..46].try_into().unwrap());
        let s1 = i16::from_le_bytes(wav[46..48].try_into().unwrap());
        assert_eq!(s0, i16::MAX);
        assert_eq!(s1, -i16::MAX);
    }

    #[test]
    fn pcm16_roundtrips_within_quantization() {
        let input = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let wav = encode_wav_pcm16(&input, 16_000);
        let mut decoded = Vec::with_capacity(input.len());
        for chunk in wav[44..].chunks_exact(2) {
            let s = i16::from_le_bytes(chunk.try_into().unwrap());
            decoded.push(s as f32 / i16::MAX as f32);
        }
        for (a, b) in input.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1.0 / i16::MAX as f32 + 1e-6);
        }
    }

    #[test]
    fn f32_header_uses_ieee_format() {
        let wav = encode_wav_f32(&[0.25], 48_000);
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
}
