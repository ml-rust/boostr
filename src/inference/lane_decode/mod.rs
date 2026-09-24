//! Driving a fixed set of decode lanes one step at a time.
//!
//! - `driver`: the caches, the lane table and the decode path one step reads
//! - `build`: building a driver at a fixed lane count
//! - `session`: lane occupancy and the next-token mirror over a driver

#![cfg(feature = "cuda")]

pub mod build;
pub mod driver;
pub mod session;

pub use build::{
    LaneDriverBuild, LaneDriverSpec, build_lane_driver, build_lane_driver_with_first_arena_for_test,
};
pub use driver::{EagerLanes, GraphLanes, LaneDecodeMode, LaneDriver};
pub use session::LaneDecodeSession;
