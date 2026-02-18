//! Quantized matmul operations trait

use crate::error::Result;
use crate::quant::QuantTensor;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Mixed-precision matmul: standard activation × quantized weight
///
/// This is THE hot path in quantized inference. The weight matrix stays in
/// quantized format and is dequantized on-the-fly during the multiply,
/// avoiding full materialization of the dequantized weight.
///
/// # Contract
///
/// - `activation` shape: `[..., M, K]` where K matches weight's last-but-one dim
/// - `weight` shape: `[N, K]` (2D quantized weight — N output rows, K input cols; blocks packed along K)
/// - Output shape: `[..., M, N]` with same dtype as `activation`
///
/// The result should match `matmul(activation, dequantize(weight))` within
/// quantization tolerance (see Verification Standards).
pub trait QuantMatmulOps<R: Runtime> {
    fn quant_matmul(&self, activation: &Tensor<R>, weight: &QuantTensor<R>) -> Result<Tensor<R>>;
}
