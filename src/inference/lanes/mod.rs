//! Lane bookkeeping for a batched decode.
//!
//! - `step`: the KV range and write position a lane-parallel forward reads
//! - `table`: lane occupancy and the persistent device tensors
//! - `admit`: taking a lane, giving it back, and stepping every occupant

pub mod admit;
pub mod step;
pub mod table;

pub use admit::LaneAdmission;
pub use step::LaneKvStep;
pub use table::LaneTable;
