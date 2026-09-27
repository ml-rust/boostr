//! Glitches in a generated take: clicks, edge thumps and cut-off ends, and
//! how far its pitch rises above the voice's usual pitch.

pub mod clicks;
pub mod heat;
pub mod repair;

pub use clicks::{ClickRepair, REPAIR_HALF_MS, clicks, declick, repair_clicks};
pub use heat::{HEATED, heat, median_pitch, percentile, pitches};
pub use repair::{CUT_OFF, TidyReport, tidy};
