//! Minimal RIFF/WAVE decoder for PCM and IEEE-float audio.
//!
//! The counterpart of [`super::encode`]: it turns a WAV byte buffer back into
//! interleaved `f32` samples in `[-1, 1]` plus the format needed to interpret them.
//! Every WAV this crate writes round-trips through [`decode_wav`], and the integer
//! widths real recorders produce (24-bit, 32-bit) decode as well.
//!
//! The parser reads untrusted file bytes, so it never panics: every read is bounds
//! checked and every malformed input returns a descriptive [`Error::InvalidArgument`].

use crate::error::{Error, Result};

use super::format::{FORMAT_EXTENSIBLE, FORMAT_IEEE_FLOAT, FORMAT_PCM};

/// A decoded WAV file: interleaved samples plus the format needed to interpret them.
#[derive(Debug, Clone, PartialEq)]
pub struct WavData {
    /// Interleaved samples normalised to `[-1.0, 1.0]`, frame-major.
    pub samples: Vec<f32>,
    /// Sample rate in Hz, as declared by the `fmt ` chunk.
    pub sample_rate: u32,
    /// Channel count, at least 1.
    pub channels: u16,
}

impl WavData {
    /// Number of frames, i.e. samples per channel.
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels.max(1) as usize
    }
}

/// Parsed `fmt ` chunk, after resolving `WAVE_FORMAT_EXTENSIBLE`.
struct Format {
    tag: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
}

fn bad(reason: String) -> Error {
    Error::InvalidArgument {
        arg: "wav bytes",
        reason,
    }
}

fn read_u16(bytes: &[u8], off: usize) -> Option<u16> {
    let end = off.checked_add(2)?;
    let raw: [u8; 2] = bytes.get(off..end)?.try_into().ok()?;
    Some(u16::from_le_bytes(raw))
}

fn read_u32(bytes: &[u8], off: usize) -> Option<u32> {
    let end = off.checked_add(4)?;
    let raw: [u8; 4] = bytes.get(off..end)?.try_into().ok()?;
    Some(u32::from_le_bytes(raw))
}

/// Decode a WAV byte buffer.
///
/// Supported formats: 16/24/32-bit signed integer PCM (tag 1), 32-bit IEEE float
/// (tag 3), and `WAVE_FORMAT_EXTENSIBLE` (tag 0xFFFE) wrapping either of those.
/// Integer samples are scaled by the positive full-scale value and clamped to
/// `[-1, 1]`; float samples are returned bit-exact.
///
/// Returns [`Error::InvalidArgument`] for a buffer that is truncated, carries bad
/// RIFF/WAVE magic, declares a chunk size that overruns the buffer, has no `fmt `
/// or `data` chunk, declares zero channels, uses an unsupported format tag or
/// bit width, or holds a `data` chunk that is not a whole number of frames.
pub fn decode_wav(bytes: &[u8]) -> Result<WavData> {
    if bytes.len() < 12 {
        return Err(bad(format!(
            "buffer is {} bytes, too short for the 12-byte RIFF/WAVE header",
            bytes.len()
        )));
    }
    if bytes.get(0..4) != Some(b"RIFF") {
        return Err(bad(
            "missing 'RIFF' magic at offset 0: not a RIFF file".to_string()
        ));
    }
    if bytes.get(8..12) != Some(b"WAVE") {
        return Err(bad(
            "missing 'WAVE' form type at offset 8: RIFF file is not a WAVE".to_string(),
        ));
    }

    let end = bytes.len();
    let mut format: Option<Format> = None;
    let mut data_range: Option<(usize, usize)> = None;
    // Whether the `data` size came from the header (exact) or from the buffer
    // length (streamed, clamped). Only the latter tolerates a partial frame.
    let mut declared_data = false;
    let mut pos = 12usize;

    // Walk the chunk list: every chunk is a 4-byte id, a little-endian u32 size,
    // then that many bytes of body, followed by one pad byte when the size is odd.
    while pos + 8 <= end {
        let id: [u8; 4] = match bytes.get(pos..pos + 4).and_then(|s| s.try_into().ok()) {
            Some(id) => id,
            None => break,
        };
        let size = read_u32(bytes, pos + 4)
            .ok_or_else(|| bad(format!("chunk header at offset {pos} is truncated")))?
            as usize;
        let body = pos + 8;
        let body_end = body.checked_add(size).ok_or_else(|| {
            bad(format!(
                "chunk '{}' at offset {pos} declares size {size}, which overflows",
                tag_name(&id)
            ))
        })?;
        // A writer that streams to a pipe cannot seek back to patch the size,
        // so it leaves `data` at 0xFFFFFFFF (or the pre-write placeholder) and
        // the chunk runs to end of file. `data` is the last chunk of such a
        // file, so its declared size is clamped to the bytes present and any
        // trailing partial frame is dropped. Every other chunk keeps the strict
        // check: a `fmt ` that overruns is corruption, not streaming.
        if body_end > end && &id == b"data" {
            if data_range.is_none() {
                data_range = Some((body, end));
            }
            break;
        }
        if body_end > end {
            return Err(bad(format!(
                "chunk '{}' at offset {pos} declares size {size}, which overruns the {end}-byte buffer",
                tag_name(&id)
            )));
        }

        match &id {
            b"fmt " => format = Some(parse_fmt(bytes, body, size)?),
            b"data" if data_range.is_none() => {
                data_range = Some((body, body_end));
                declared_data = true;
            }
            _ => {}
        }

        // RIFF pads odd-sized chunk bodies to an even boundary.
        let padded = size + (size & 1);
        pos = match body.checked_add(padded) {
            Some(next) => next,
            None => break,
        };
    }

    let format = format.ok_or_else(|| bad("no 'fmt ' chunk found in file".to_string()))?;
    let (data_start, data_end) =
        data_range.ok_or_else(|| bad("no 'data' chunk found in file".to_string()))?;
    let data = bytes.get(data_start..data_end).ok_or_else(|| {
        bad(format!(
            "'data' chunk range {data_start}..{data_end} is outside the buffer"
        ))
    })?;

    let bytes_per_sample = (format.bits_per_sample / 8) as usize;
    let frame_size = bytes_per_sample
        .checked_mul(format.channels as usize)
        .ok_or_else(|| bad("frame size overflows".to_string()))?;
    if frame_size == 0 {
        return Err(bad("frame size is zero".to_string()));
    }
    let whole_frames = data.len() - data.len() % frame_size;
    if whole_frames != data.len() && declared_data {
        return Err(bad(format!(
            "'data' chunk is {} bytes, not a whole number of {frame_size}-byte frames \
             ({} channels x {} bits)",
            data.len(),
            format.channels,
            format.bits_per_sample
        )));
    }
    let data = &data[..whole_frames];

    let samples = decode_samples(data, &format, bytes_per_sample);
    Ok(WavData {
        samples,
        sample_rate: format.sample_rate,
        channels: format.channels,
    })
}

/// Render a chunk id for error messages, replacing non-printable bytes.
fn tag_name(id: &[u8; 4]) -> String {
    id.iter()
        .map(|&b| {
            if b.is_ascii_graphic() || b == b' ' {
                b as char
            } else {
                '?'
            }
        })
        .collect()
}

fn parse_fmt(bytes: &[u8], body: usize, size: usize) -> Result<Format> {
    if size < 16 {
        return Err(bad(format!(
            "'fmt ' chunk is {size} bytes, the minimum is 16"
        )));
    }
    let field = |off: usize| -> Result<u16> {
        read_u16(bytes, body + off)
            .ok_or_else(|| bad(format!("'fmt ' chunk is truncated at offset {off}")))
    };
    let mut tag = field(0)?;
    let channels = field(2)?;
    let sample_rate = read_u32(bytes, body + 4)
        .ok_or_else(|| bad("'fmt ' chunk is truncated at the sample rate".to_string()))?;
    let bits_per_sample = field(14)?;

    if tag == FORMAT_EXTENSIBLE {
        // Extensible layout: 16 standard bytes, cbSize (2), validBitsPerSample (2),
        // dwChannelMask (4), then a 16-byte SubFormat GUID whose first two bytes
        // hold the real format tag.
        if size < 40 {
            return Err(bad(format!(
                "WAVE_FORMAT_EXTENSIBLE 'fmt ' chunk is {size} bytes, the minimum is 40"
            )));
        }
        let cb_size = field(16)?;
        if cb_size < 22 {
            return Err(bad(format!(
                "WAVE_FORMAT_EXTENSIBLE cbSize is {cb_size}, the minimum is 22"
            )));
        }
        tag = field(24)?;
        if tag == FORMAT_EXTENSIBLE {
            return Err(bad(
                "WAVE_FORMAT_EXTENSIBLE SubFormat GUID is itself 0xFFFE".to_string(),
            ));
        }
    }

    if channels == 0 {
        return Err(bad("'fmt ' chunk declares 0 channels".to_string()));
    }
    match tag {
        FORMAT_PCM => {
            if !matches!(bits_per_sample, 16 | 24 | 32) {
                return Err(bad(format!(
                    "integer PCM with {bits_per_sample} bits per sample is unsupported, \
                     expected 16, 24, or 32"
                )));
            }
        }
        FORMAT_IEEE_FLOAT => {
            if bits_per_sample != 32 {
                return Err(bad(format!(
                    "IEEE float with {bits_per_sample} bits per sample is unsupported, \
                     expected 32"
                )));
            }
        }
        other => {
            return Err(bad(format!(
                "unsupported WAVE format tag {other}, expected 1 (PCM), 3 (IEEE float), \
                 or 0xFFFE (extensible)"
            )));
        }
    }

    Ok(Format {
        tag,
        channels,
        sample_rate,
        bits_per_sample,
    })
}

fn decode_samples(data: &[u8], format: &Format, bytes_per_sample: usize) -> Vec<f32> {
    let count = data.len() / bytes_per_sample.max(1);
    let mut out = Vec::with_capacity(count);
    for chunk in data.chunks_exact(bytes_per_sample) {
        out.push(decode_one(chunk, format));
    }
    out
}

/// Convert one sample's bytes to `f32`. `chunk.len()` is `bits_per_sample / 8`,
/// which `parse_fmt` has already restricted to 2, 3, or 4.
fn decode_one(chunk: &[u8], format: &Format) -> f32 {
    match (format.tag, chunk.len()) {
        (FORMAT_IEEE_FLOAT, 4) => match chunk.try_into() {
            Ok(raw) => f32::from_le_bytes(raw),
            Err(_) => 0.0,
        },
        (FORMAT_PCM, 2) => match chunk.try_into() {
            // i16::MAX matches the writer's scale, so the round trip is symmetric.
            Ok(raw) => scale(i16::from_le_bytes(raw) as f64, i16::MAX as f64),
            Err(_) => 0.0,
        },
        (FORMAT_PCM, 3) => {
            let raw: [u8; 3] = match chunk.try_into() {
                Ok(raw) => raw,
                Err(_) => return 0.0,
            };
            let packed = raw[0] as i32 | ((raw[1] as i32) << 8) | ((raw[2] as i32) << 16);
            // Sign-extend from 24 bits.
            let value = if packed & 0x0080_0000 != 0 {
                packed - 0x0100_0000
            } else {
                packed
            };
            scale(value as f64, 8_388_607.0)
        }
        (FORMAT_PCM, 4) => match chunk.try_into() {
            Ok(raw) => scale(i32::from_le_bytes(raw) as f64, i32::MAX as f64),
            Err(_) => 0.0,
        },
        _ => 0.0,
    }
}

/// Normalise an integer sample against positive full scale, clamped to `[-1, 1]`
/// because the negative extreme is one step past `-1`.
///
/// Shared with the `decode` module (feature `decode`) so the symphonia-backed path and this one
/// return identical values for identical bytes.
pub(crate) fn scale(value: f64, full_scale: f64) -> f32 {
    (value / full_scale).clamp(-1.0, 1.0) as f32
}

/// Downmix interleaved samples to mono by averaging channels.
///
/// Returns the input unchanged when `channels == 1`. Returns
/// [`Error::InvalidArgument`] when `channels` is 0 or `samples.len()` is not
/// divisible by `channels`.
pub fn to_mono(samples: &[f32], channels: u16) -> Result<Vec<f32>> {
    if channels == 0 {
        return Err(Error::InvalidArgument {
            arg: "channels",
            reason: "channel count is 0".to_string(),
        });
    }
    if channels == 1 {
        return Ok(samples.to_vec());
    }
    let channels = channels as usize;
    if !samples.len().is_multiple_of(channels) {
        return Err(Error::InvalidArgument {
            arg: "samples",
            reason: format!(
                "sample count {} is not divisible by the channel count {channels}",
                samples.len()
            ),
        });
    }
    let scale = 1.0f32 / channels as f32;
    Ok(samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() * scale)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::encode::encode_wav_pcm16;

    /// Offsets of the two size fields `encode_wav_pcm16` writes: RIFF size at
    /// 4, `data` size at 40 (12-byte header + 24-byte `fmt ` chunk + 4-byte id).
    const RIFF_SIZE_OFF: usize = 4;
    const DATA_SIZE_OFF: usize = 40;

    fn pcm16_with_sizes(samples: &[f32], riff: u32, data: u32) -> Vec<u8> {
        let mut bytes = encode_wav_pcm16(samples, 16_000).expect("encode");
        assert_eq!(&bytes[36..40], b"data");
        bytes[RIFF_SIZE_OFF..RIFF_SIZE_OFF + 4].copy_from_slice(&riff.to_le_bytes());
        bytes[DATA_SIZE_OFF..DATA_SIZE_OFF + 4].copy_from_slice(&data.to_le_bytes());
        bytes
    }

    #[test]
    fn streamed_sizes_clamp_data_to_the_buffer() {
        let samples = [0.0f32, 0.25, -0.5, 0.75, -1.0, 0.5];
        let exact = decode_wav(&encode_wav_pcm16(&samples, 16_000).expect("encode"))
            .expect("exact header decodes");
        let streamed = decode_wav(&pcm16_with_sizes(&samples, u32::MAX, u32::MAX))
            .expect("0xFFFFFFFF sizes decode");
        assert_eq!(streamed, exact);
        assert_eq!(streamed.frames(), samples.len());
    }

    #[test]
    fn streamed_data_drops_a_trailing_partial_frame() {
        let samples = [0.0f32, 0.25, -0.5, 0.75];
        let mut bytes = pcm16_with_sizes(&samples, u32::MAX, u32::MAX);
        bytes.push(0x7f);
        let decoded = decode_wav(&bytes).expect("odd trailing byte is dropped");
        assert_eq!(decoded.frames(), samples.len());
    }

    #[test]
    fn declared_data_size_must_be_whole_frames() {
        let samples = [0.0f32, 0.25, -0.5, 0.75];
        let mut bytes = pcm16_with_sizes(&samples, 0, 7);
        bytes.truncate(44 + 7);
        let err = decode_wav(&bytes).expect_err("7 bytes of 2-byte frames");
        assert!(err.to_string().contains("not a whole number"), "{err}");
    }

    #[test]
    fn fmt_overrun_is_still_an_error() {
        let mut bytes = encode_wav_pcm16(&[0.0f32; 4], 16_000).expect("encode");
        bytes[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        let err = decode_wav(&bytes).expect_err("fmt chunk overruns");
        assert!(err.to_string().contains("overruns"), "{err}");
    }
}
