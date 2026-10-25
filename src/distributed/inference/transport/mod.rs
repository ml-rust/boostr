//! Swarm pipeline protocol over nexar.
//!
//! - `link`: tags, one-frame send and receive, client shutdown
//! - `assignment`: `LayerAssignment`
//! - `request`: `GenRequestHeader`, `StageOp`, the request frame
//! - `payload`: `StagePayload`, `Activation`
//! - `reply`: `LogitsFrame`, `StageError`
//! - `version`: the version byte every swarm frame starts with
//! - `tasks`: task guards and listeners the swarm and disaggregated modes share

pub mod assignment;
pub mod link;
pub mod payload;
pub mod reply;
pub mod request;
pub(crate) mod tasks;
pub mod version;

pub use assignment::{LayerAssignment, model_path_frame, model_path_from_frame};
pub use link::{cpu_adapter, recv_frame, send_frame, shutdown_shared, tags};
pub use payload::{Activation, StagePayload};
pub use reply::{LogitsFrame, StageError};
pub use request::{GenRequestHeader, HEADER_LEN, RequestFrame, StageOp};
pub use version::PROTOCOL_VERSION;
