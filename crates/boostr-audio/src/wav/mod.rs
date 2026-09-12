//! RIFF/WAVE: the one container this crate reads and writes without a codec
//! dependency. [`decode`] parses PCM and IEEE-float files, [`encode`] writes
//! them, and [`format`] holds the format tags both sides agree on.

pub mod decode;
pub mod encode;
pub mod format;

pub use decode::{WavData, decode_wav, to_mono};
pub use encode::{
    encode_pcm16_raw, encode_wav_f32, encode_wav_pcm16, encode_wav_pcm16_multichannel,
};
