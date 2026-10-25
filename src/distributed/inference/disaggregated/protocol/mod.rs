//! Wire protocol of disaggregated prefill/decode inference.
//!
//! - `codec`: the frame header, the version and the field codec
//! - `params`: `GenerationParams`, `FinishReason`, `Stage`
//! - `frames`: `PrefillRequest`, `KvHandoff`, `Cancel`, `HandoffAck`, `Event`
//! - `tags`: the nexar tags each frame travels on

pub mod codec;
pub mod frames;
pub mod params;
pub mod tags;

pub use codec::{FRAME_MAGIC, FrameKind, PROTOCOL_VERSION, peek_request_id};
pub use frames::{Cancel, Event, EventBody, HandoffAck, KvHandoff, PrefillRequest, TokenEvent};
pub use params::{FinishReason, GenerationParams, Stage};
