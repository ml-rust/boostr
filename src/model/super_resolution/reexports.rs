//! The flat `super_resolution::*` names, re-exported from each module.

pub use super::client::SrClient;
pub use super::compact::{SrVggCompact, SrVggCompactConfig};
pub use super::conv::SrConv;
pub use super::dir::load_sr_dir;
pub use super::layout::{SrExecution, weight_for_layout};
pub use super::loader::{load_compact, load_rrdb};
pub use super::model::SrModel;
pub use super::rrdb::{RrdbConfig, RrdbNet};
