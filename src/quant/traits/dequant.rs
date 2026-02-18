//! Dequantization operations trait

use crate::error::Result;
use crate::quant::QuantTensor;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Dequantize a quantized tensor to a standard tensor
pub trait DequantOps<R: Runtime> {
    /// Dequantize to standard tensor with the specified dtype
    ///
    /// Converts block-quantized storage back to element-wise floating point.
    /// `target_dtype` must be a floating point type (F32, F16, BF16).
    fn dequantize(&self, qt: &QuantTensor<R>, target_dtype: DType) -> Result<Tensor<R>>;
}
