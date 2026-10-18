//! Cross-entropy loss functions.
//!
//! - `plain`: [`cross_entropy_loss`]
//! - `smooth`: [`cross_entropy_loss_smooth`] (label smoothing)
//! - `masked`: [`cross_entropy_loss_masked`] (per-position mask)
//! - `masked_sum`: [`cross_entropy_loss_masked_sum`] (per-position mask, summed)

mod masked;
mod masked_sum;
mod plain;
mod smooth;

pub use masked::cross_entropy_loss_masked;
pub use masked_sum::cross_entropy_loss_masked_sum;
pub use plain::cross_entropy_loss;
pub use smooth::cross_entropy_loss_smooth;
