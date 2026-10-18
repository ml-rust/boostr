//! Data-parallel collectives issued in caller order, identical on every rank.

pub mod broadcast;
mod buckets;
pub mod grad_sum;

pub use broadcast::{broadcast_params_ordered, broadcast_params_ordered_bucketed};
pub use buckets::DEFAULT_ORDERED_BUCKET_BYTES;
pub use grad_sum::{all_reduce_flat_sum, all_reduce_flat_sum_bucketed};
