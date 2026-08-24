//! Compressed-audio decoding (FLAC, MP3, OGG/Vorbis, WAV) via symphonia.
//!
//! Complements [`super::wav_decode`], which parses uncompressed RIFF/WAVE bytes
//! directly. This module hands the same job — and any container symphonia can
//! probe — to symphonia's decoders, and returns the same [`WavData`] shape at
//! the file's native rate. Downmix and rate conversion are NOT done here: the
//! caller reaches for [`super::resample::to_mono_at_rate`], so there is exactly
//! one downmix implementation and one resampler in the crate.
//!
//! **CPU-only, one decode pass.** Every sample format symphonia can hand back
//! (float, signed/unsigned integer, 8 through 32 bits) is converted to `f32` in
//! [`push_frame`] as each packet arrives, so the whole file is visited once.

use std::path::Path;

use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::error::{Error, Result};

use super::resample::to_mono_at_rate;
use super::wav_decode::{WavData, scale};

fn bad(reason: String) -> Error {
    Error::InvalidArgument {
        arg: "audio bytes",
        reason,
    }
}

/// Decode any container symphonia can probe (FLAC, WAV, MP3, OGG/Vorbis) into
/// interleaved f32 samples at the file's native rate.
///
/// `hint` is an optional file-extension hint (`"flac"`, `"mp3"`, ...) that
/// helps symphonia's format probe; pass `None` when the extension is unknown.
/// Returns [`Error::InvalidArgument`] when the bytes cannot be probed, the
/// track carries no codec parameters, or every packet fails to decode.
pub fn decode_audio(bytes: &[u8], hint: Option<&str>) -> Result<WavData> {
    // symphonia's `MediaSource` bound is `'static`, so a stream built over a
    // borrowed slice cannot satisfy it; the byte-slice path must own its copy.
    let cursor = std::io::Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    decode_stream(mss, hint)
}

/// Decode, downmix to mono, and resample to `target_rate` in one step.
///
/// Delegates the downmix and rate conversion to
/// [`super::resample::to_mono_at_rate`], so the behaviour matches every other
/// caller of that function.
pub fn decode_audio_mono_at(
    bytes: &[u8],
    hint: Option<&str>,
    target_rate: u32,
) -> Result<Vec<f32>> {
    let data = decode_audio(bytes, hint)?;
    to_mono_at_rate(&data, target_rate)
}

/// Decode, downmix to mono, and resample to `target_rate`, reading from a file
/// on disk without loading the whole container into memory first.
///
/// The file's extension supplies the format hint. Intended for the training
/// corpus path, where files run tens of megabytes each.
pub fn decode_audio_file_mono_at(path: &Path, target_rate: u32) -> Result<Vec<f32>> {
    let file = std::fs::File::open(path)
        .map_err(|e| bad(format!("opening '{}' failed: {e}", path.display())))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let hint = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(extension_hint);
    let data = decode_stream(mss, hint)?;
    to_mono_at_rate(&data, target_rate)
}

/// Extract a likely file-extension hint from a filename like `"speech.flac"`.
/// Returns `None` for a bare name with no `.`.
pub fn extension_hint(filename: &str) -> Option<&str> {
    filename.rsplit('.').next().filter(|ext| *ext != filename)
}

fn decode_stream(mss: MediaSourceStream, hint: Option<&str>) -> Result<WavData> {
    let mut probe_hint = Hint::new();
    if let Some(ext) = hint {
        probe_hint.with_extension(ext);
    }

    let probe = symphonia::default::get_probe()
        .format(
            &probe_hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| bad(format!("probing audio format: {e}")))?;
    let mut format = probe.format;

    let track = format
        .default_track()
        .ok_or_else(|| bad("audio file has no default track".to_string()))?;
    let codec_params = track.codec_params.clone();
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| bad(format!("creating audio decoder: {e}")))?;

    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| bad("audio file missing sample_rate".to_string()))?;
    let channels = codec_params
        .channels
        .ok_or_else(|| bad("audio file missing channel layout".to_string()))?
        .count();
    let channels_u16 = u16::try_from(channels)
        .map_err(|_| bad(format!("audio file declares {channels} channels, too many")))?;

    // Interleaved samples at the source rate; the caller downmixes/resamples.
    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(bad(format!("reading packet: {e}"))),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let audio = match decoder.decode(&packet) {
            Ok(a) => a,
            Err(SymphoniaError::DecodeError(_)) => continue, // skip corrupt frames
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(e) => return Err(bad(format!("decoding packet: {e}"))),
        };
        push_frame(&audio, channels, &mut samples);
    }

    if samples.is_empty() {
        return Err(bad("decoded audio is empty".to_string()));
    }

    Ok(WavData {
        samples,
        sample_rate,
        channels: channels_u16,
    })
}

/// Append one decoded audio buffer's samples to `out`, interleaved frame-major,
/// converting every symphonia sample format to `f32` in `[-1, 1]`.
fn push_frame(audio: &AudioBufferRef<'_>, channels: usize, out: &mut Vec<f32>) {
    macro_rules! interleave {
        ($buf:expr, $convert:expr) => {{
            let frames = $buf.frames();
            if channels <= 1 {
                out.reserve(frames);
                for i in 0..frames {
                    out.push($convert($buf.chan(0)[i]));
                }
            } else {
                out.reserve(frames * channels);
                for i in 0..frames {
                    for c in 0..channels {
                        out.push($convert($buf.chan(c)[i]));
                    }
                }
            }
        }};
    }
    match audio {
        AudioBufferRef::F32(b) => interleave!(b, |x: f32| x),
        AudioBufferRef::F64(b) => interleave!(b, |x: f64| x as f32),
        // Signed formats go through `wav_decode::scale`, the same helper the
        // native WAV parser uses: divide by positive full scale and clamp, so
        // both decoders return identical values for identical bytes. That scale
        // also matches what `wav_encode` writes, keeping the round trip exact.
        AudioBufferRef::S16(b) => interleave!(b, |x: i16| scale(x as f64, i16::MAX as f64)),
        AudioBufferRef::S32(b) => interleave!(b, |x: i32| scale(x as f64, i32::MAX as f64)),
        AudioBufferRef::U8(b) => interleave!(b, |x: u8| (x as f32 - 128.0) / 128.0),
        // Same exact-bounds scale as U32/U24 below: `(x / MAX) * 2 - 1` maps
        // 0 to -1.0 and MAX to +1.0 with no overshoot. The previous
        // `(x - MAX/2) / (MAX/2)` form overshot +1.0 by one step at the top
        // rail because `u16::MAX` is odd, so `MAX/2.0` is not its own true
        // midpoint divisor.
        AudioBufferRef::U16(b) => {
            interleave!(b, |x: u16| (x as f32 / u16::MAX as f32) * 2.0 - 1.0)
        }
        AudioBufferRef::U32(b) => interleave!(b, |x: u32| (x as f32 / u32::MAX as f32) * 2.0 - 1.0),
        AudioBufferRef::S8(b) => interleave!(b, |x: i8| scale(x as f64, i8::MAX as f64)),
        AudioBufferRef::S24(b) => interleave!(b, |x: symphonia::core::sample::i24| scale(
            x.inner() as f64,
            8_388_607.0
        )),
        AudioBufferRef::U24(b) => interleave!(b, |x: symphonia::core::sample::u24| (x.inner()
            as f32
            / 16_777_215.0)
            * 2.0
            - 1.0),
    }
}

#[cfg(test)]
mod tests;
