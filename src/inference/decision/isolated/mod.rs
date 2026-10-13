//! Calibrating isolated Score questions: the samples and their NLL
//! (`objective`), and the bracketed temperature fit (`fit`).

pub mod fit;
pub mod objective;

pub use fit::{SCAN_POINTS, fit_temperature_isolated};
pub use objective::{IsolatedSample, evaluate_isolated, is_flat_isolated, validate_isolated};
