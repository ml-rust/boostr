//! Checks a spoken take against its script: word alignment, the speech
//! check verdict, and script words placed on recognized timings.

mod speech_match;

pub mod align_script;
pub mod heard;
pub mod speech_check;

pub use align_script::{STRAY_PAUSE, StrayPause, align_script, stray_pauses};
pub use heard::HeardWord;
pub use speech_check::{Extra, Verdict, check_speech};
