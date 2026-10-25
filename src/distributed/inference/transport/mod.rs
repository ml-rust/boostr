//! Swarm pipeline protocol over nexar.
//!
//! - `link`: tags, one-frame send and receive, client shutdown
//! - `assignment`: `LayerAssignment`
//! - `request`: `GenRequestHeader`, `StageOp`, the request frame
//! - `payload`: `StagePayload`, `Activation`
//! - `reply`: `LogitsFrame`, `StageError`
//! - `tasks`: task guards and listeners the swarm and disaggregated modes share

pub mod assignment;
pub mod link;
pub mod payload;
pub mod reply;
pub mod request;
pub(crate) mod tasks;

pub use assignment::LayerAssignment;
pub use link::{cpu_adapter, recv_frame, send_frame, shutdown_shared, tags};
pub use payload::{Activation, StagePayload};
pub use reply::{LogitsFrame, StageError};
pub use request::{GenRequestHeader, HEADER_LEN, PROTOCOL_VERSION, StageOp};
