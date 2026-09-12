//! Segmentation: speech probabilities in, utterance boundaries out.
//!
//! [`SileroVad`](boostr::model::audio::SileroVad) scores one 512-sample chunk
//! at a time. That per-chunk probability is not a usable answer on its own — a
//! single dip below the threshold in the middle of a word would end an
//! utterance. This module is the port of Silero's `get_speech_timestamps`:
//! [`options`] holds the tuning and the [`SpeechSegment`] range type,
//! [`segment`] the pure rule engine over a probability array, and
//! [`timestamps`] the wrapper that runs the network first.

pub mod options;
pub mod segment;
pub mod timestamps;

pub use options::{SpeechSegment, VadSegmentOptions};
pub use segment::segments_from_probabilities;
pub use timestamps::speech_timestamps;
