//! `TcfLoader`: open a `.tcf` file and load its tensors as dense `Tensor<R>`.
//!
//! The file is memory-mapped and the directory is decoded once, so the
//! metadata a placement planner needs is available without touching a
//! payload page (Section 16). Payload pages are read only when a tensor is
//! loaded or verified.
//!
//! Every load verifies the tensor first: `payload_digest`, then the
//! recomputed logical stream against `semantic_digest`, then the proof
//! vector (Section 15). A reader that skips this cannot tell a correct file
//! from a corrupted one, which is the failure the format exists to prevent.

mod directory;
mod read;
mod session;

#[cfg(test)]
mod test_support;

pub use directory::TcfLoader;
pub use session::TcfSession;
