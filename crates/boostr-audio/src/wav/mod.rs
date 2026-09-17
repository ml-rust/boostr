//! RIFF/WAVE: the one container this crate reads and writes without a codec
//! dependency. [`decode`] parses PCM and IEEE-float files, [`encode`] writes
//! them, [`stream`] writes the unknown-length header a streamed response
//! starts with, and [`format`] holds the format tags all sides agree on.

pub mod decode;
pub mod encode;
pub mod format;
pub mod stream;

pub use decode::{WavData, decode_wav, to_mono};
pub use encode::{
    HEADER_LEN, encode_f32_raw, encode_pcm16_raw, encode_wav_f32, encode_wav_pcm16,
    encode_wav_pcm16_multichannel,
};
pub use stream::{STREAM_SIZE_UNKNOWN, wav_stream_header_f32, wav_stream_header_pcm16};
