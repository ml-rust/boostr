//! Client trait alias for `AudioVaeDecoder` forward passes.

use crate::model::traits::ModelClient;
use crate::quant::traits::DequantOps;
use numr::ops::ConvOps;
use numr::runtime::Runtime;

/// Bounds required by every VoxCPM2 decoder sub-block: the standard
/// [`ModelClient`] set (elementwise/activation ops), [`ConvOps`] for the
/// causal (transposed) convolutions that make up the whole stack, and
/// [`DequantOps`] for `MiniCpm4Model::embed`'s block-quantized
/// `embed_tokens` path (`ModelClient` covers `QuantMatmulOps` for linear
/// weights, but not the dequant path an embedding gather needs).
pub trait VoxCpmClient<R: Runtime>: ModelClient<R> + ConvOps<R> + DequantOps<R> {}

impl<R, C> VoxCpmClient<R> for C
where
    R: Runtime,
    C: ModelClient<R> + ConvOps<R> + DequantOps<R>,
{
}
