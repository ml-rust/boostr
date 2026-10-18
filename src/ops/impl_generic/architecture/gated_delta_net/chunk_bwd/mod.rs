//! Chunked GDN backward: chunk-parallel gradients of the chunked forward.

mod adjoint;
pub(crate) mod backward;
mod gates;
mod local_grads;
mod states;

pub use backward::gdn_chunk_backward_impl;
