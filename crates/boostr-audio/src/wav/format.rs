//! RIFF/WAVE format-tag constants shared by [`super::encode`] and [`super::decode`].

/// Uncompressed integer PCM (`WAVE_FORMAT_PCM`).
pub const FORMAT_PCM: u16 = 1;
/// 32-bit IEEE float samples (`WAVE_FORMAT_IEEE_FLOAT`).
pub const FORMAT_IEEE_FLOAT: u16 = 3;
/// Extended header carrying the real format tag in its `SubFormat` GUID.
pub const FORMAT_EXTENSIBLE: u16 = 0xFFFE;
