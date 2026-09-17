//! TTS quality gate: numbers for every rendered clip.
//!
//! [`signal`] and [`summary`] are always compiled. [`pace`] needs `vad`,
//! [`intelligibility`] needs `whisper`, [`timbre`] needs `voxcpm`. The
//! `voxcpm_quality_gate` example wires all of them over a render log.

pub mod signal;
pub mod summary;
pub mod thresholds;

#[cfg(feature = "whisper")]
pub mod intelligibility;
#[cfg(feature = "vad")]
pub mod pace;
#[cfg(feature = "voxcpm")]
pub mod timbre;

#[cfg(feature = "whisper")]
pub use intelligibility::{Intelligibility, IntelligibilityClient, IntelligibilityScorer};
#[cfg(feature = "vad")]
pub use pace::{Pace, pace, pace_from_segments};
pub use signal::{SignalStats, signal_stats};
pub use summary::{MetricStat, RowScore, Summary, UNLABELLED_AXIS};
pub use thresholds::{Thresholds, violations};
#[cfg(feature = "voxcpm")]
pub use timbre::{F0Stats, TimbreProxy, TimbreScorer, cosine, embed_with, timbre_proxy};
