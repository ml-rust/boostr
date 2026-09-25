//! [`LanePositions`]: each row's write position across a batched decode, the
//! device tensor the attention kernels read its key range from, and the
//! stepped width a decode call actually covers.
//!
//! `table` holds the table itself — construction, seating, advancing and the
//! device mirror. `active` holds the stepped width and the decode view
//! narrowed to it.

mod active;
mod table;

pub use table::LanePositions;
