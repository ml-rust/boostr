//! Turn a raw recording into a usable reference.
//!
//! A microphone take is not a reference signal. It carries a room, a preamp
//! hiss floor, an arbitrary level set by how close the speaker sat, and a
//! thinned low end from the high-pass every capture chain applies. Cloning a
//! voice from that hands the model the room as part of the voice.
//!
//! Order is fixed and is not a preference: high-pass, denoise, tone, then
//! loudness and the peak ceiling last. Every earlier stage changes the level,
//! so measuring loudness before them measures something that no longer exists.

pub mod biquad;
pub mod denoise;
pub mod limiter;
pub mod loudness;
pub mod pipeline;

pub use biquad::Biquad;
pub use denoise::{DenoiseOptions, denoise, denoise_with_profile, noise_floor_dbfs};
pub use limiter::{LimiterOptions, LimiterReport, limit};
pub use loudness::{integrated_lufs, normalize_to_lufs, peak_dbfs};
pub use pipeline::{EnhanceOptions, EnhanceReport, enhance, enhance_with_noise_profile};
