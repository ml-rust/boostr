//! Client trait alias for NeuCodec decoder forward passes.

use crate::model::traits::ModelClient;
use numr::ops::{ConvOps, RandomOps};
use numr::runtime::Runtime;

/// Bounds required by every NeuCodec decoder sub-block: the standard
/// [`ModelClient`] set (linear/attention/RoPE/normalization), [`ConvOps`] for
/// the Conv1d-based `embed` stage and `ResnetBlock`s, and [`RandomOps`] for
/// the `ResnetBlock` dropout (active only in training mode).
pub trait NeuCodecClient<R: Runtime>: ModelClient<R> + ConvOps<R> + RandomOps<R> {}

impl<R, C> NeuCodecClient<R> for C
where
    R: Runtime,
    C: ModelClient<R> + ConvOps<R> + RandomOps<R>,
{
}
