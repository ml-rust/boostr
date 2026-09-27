//! Narration scripts: phrases and chunks a voice speaks, tags, subtitle and
//! timed script files, the written script, numbers read back from words and
//! digits, the check of a spoken take against its script, and its glitches.

mod js;
mod pattern;

pub mod check;
pub mod glitch;
pub mod text;

pub use check::*;
pub use glitch::*;
pub use text::*;
