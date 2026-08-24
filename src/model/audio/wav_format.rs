//! RIFF/WAVE format-tag constants shared by [`super::wav_encode`] and [`super::wav_decode`].

/// Uncompressed integer PCM (`WAVE_FORMAT_PCM`).
pub(crate) const FORMAT_PCM: u16 = 1;
/// 32-bit IEEE float samples (`WAVE_FORMAT_IEEE_FLOAT`).
pub(crate) const FORMAT_IEEE_FLOAT: u16 = 3;
/// Extended header carrying the real format tag in its `SubFormat` GUID.
pub(crate) const FORMAT_EXTENSIBLE: u16 = 0xFFFE;
