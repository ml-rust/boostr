//! WAV headers for a stream whose length is unknown when the first byte
//! goes out.
//!
//! The canonical header carries the RIFF and `data` chunk sizes up front. A
//! streamed response cannot know them, so both are written as `u32::MAX`,
//! the convention players and decoders treat as "read to end of stream".
//! The rest of the header is byte-identical to the finished-file header, so
//! a stream header followed by [`super::encode::encode_pcm16_raw`] (or
//! [`super::encode::encode_f32_raw`]) chunks is a finished
//! [`super::encode::encode_wav_pcm16`] (or `encode_wav_f32`) file except
//! for those two fields.

use crate::error::Result;

use super::encode::{HEADER_LEN, WavHeader, write_header};
use super::format::{FORMAT_IEEE_FLOAT, FORMAT_PCM};

/// Size field value meaning "unknown, read to end of stream".
pub const STREAM_SIZE_UNKNOWN: u32 = u32::MAX;

/// Mono 16-bit PCM stream header; follow it with
/// [`super::encode::encode_pcm16_raw`] chunks.
pub fn wav_stream_header_pcm16(sample_rate: u32) -> Result<Vec<u8>> {
    stream_header(FORMAT_PCM, 16, sample_rate)
}

/// Mono 32-bit IEEE float stream header; follow it with
/// [`super::encode::encode_f32_raw`] chunks.
pub fn wav_stream_header_f32(sample_rate: u32) -> Result<Vec<u8>> {
    stream_header(FORMAT_IEEE_FLOAT, 32, sample_rate)
}

fn stream_header(format_tag: u16, bits_per_sample: u16, sample_rate: u32) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(HEADER_LEN);
    write_header(
        &mut out,
        WavHeader {
            format_tag,
            channels: 1,
            sample_rate,
            bits_per_sample,
            riff_size: STREAM_SIZE_UNKNOWN,
            data_size: STREAM_SIZE_UNKNOWN,
        },
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::encode::{encode_f32_raw, encode_pcm16_raw, encode_wav_f32, encode_wav_pcm16};

    fn u32_at(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
    }

    #[test]
    fn pcm16_stream_header_bytes() {
        let h = wav_stream_header_pcm16(48_000).unwrap();
        assert_eq!(h.len(), HEADER_LEN);
        assert_eq!(&h[0..4], b"RIFF");
        assert_eq!(u32_at(&h, 4), u32::MAX);
        assert_eq!(&h[8..12], b"WAVE");
        assert_eq!(&h[12..16], b"fmt ");
        assert_eq!(u32_at(&h, 16), 16);
        assert_eq!(u16::from_le_bytes(h[20..22].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(h[22..24].try_into().unwrap()), 1);
        assert_eq!(u32_at(&h, 24), 48_000);
        assert_eq!(u32_at(&h, 28), 96_000);
        assert_eq!(u16::from_le_bytes(h[32..34].try_into().unwrap()), 2);
        assert_eq!(u16::from_le_bytes(h[34..36].try_into().unwrap()), 16);
        assert_eq!(&h[36..40], b"data");
        assert_eq!(u32_at(&h, 40), u32::MAX);
    }

    #[test]
    fn f32_stream_header_bytes() {
        let h = wav_stream_header_f32(24_000).unwrap();
        assert_eq!(h.len(), HEADER_LEN);
        assert_eq!(u16::from_le_bytes(h[20..22].try_into().unwrap()), 3);
        assert_eq!(u32_at(&h, 28), 96_000);
        assert_eq!(u16::from_le_bytes(h[32..34].try_into().unwrap()), 4);
        assert_eq!(u16::from_le_bytes(h[34..36].try_into().unwrap()), 32);
        assert_eq!(u32_at(&h, 4), u32::MAX);
        assert_eq!(u32_at(&h, 40), u32::MAX);
    }

    /// Outside the two size fields, a stream header plus raw chunks is the
    /// finished file byte for byte.
    #[test]
    fn stream_header_plus_raw_chunks_equals_the_finished_file() {
        let samples: Vec<f32> = (0..300).map(|i| ((i as f32) * 0.05).sin()).collect();
        let (a, b) = samples.split_at(128);

        let mut pcm = wav_stream_header_pcm16(48_000).unwrap();
        pcm.extend(encode_pcm16_raw(a));
        pcm.extend(encode_pcm16_raw(b));
        let whole = encode_wav_pcm16(&samples, 48_000).unwrap();
        assert_eq!(pcm.len(), whole.len());
        assert_eq!(pcm[8..40], whole[8..40]);
        assert_eq!(pcm[HEADER_LEN..], whole[HEADER_LEN..]);
        assert_ne!(pcm[4..8], whole[4..8]);
        assert_ne!(pcm[40..44], whole[40..44]);

        let mut f = wav_stream_header_f32(48_000).unwrap();
        f.extend(encode_f32_raw(a));
        f.extend(encode_f32_raw(b));
        let whole = encode_wav_f32(&samples, 48_000).unwrap();
        assert_eq!(f[8..40], whole[8..40]);
        assert_eq!(f[HEADER_LEN..], whole[HEADER_LEN..]);
    }

    #[test]
    fn stream_header_rejects_an_overflowing_byte_rate() {
        assert!(wav_stream_header_f32(u32::MAX).is_err());
    }
}
