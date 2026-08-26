//! Client trait alias for `AudioVaeDecoder` forward passes.

use crate::model::traits::ModelClient;
use numr::ops::ConvOps;
use numr::runtime::Runtime;

/// Bounds required by every VoxCPM2 decoder sub-block: the standard
/// [`ModelClient`] set (elementwise/activation ops) plus [`ConvOps`] for the
/// causal (transposed) convolutions that make up the whole stack.
pub trait VoxCpmClient<R: Runtime>: ModelClient<R> + ConvOps<R> {}

impl<R, C> VoxCpmClient<R> for C
where
    R: Runtime,
    C: ModelClient<R> + ConvOps<R>,
{
}
