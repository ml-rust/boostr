//! Swarm leader: forms the cluster, assigns layers, and routes request
//! frames through the pipeline.
//!
//! - `swarm_leader`: `SwarmLeader`, request frames and releases
//! - `setup`: cluster formation, assignments, readiness, the shutdown notice
//! - `dispatch`: routes logits and stage errors to waiting requests
//! - `route`: the entry and last worker stage of a schedule

mod dispatch;
mod route;
mod setup;
pub mod swarm_leader;

pub use swarm_leader::SwarmLeader;
