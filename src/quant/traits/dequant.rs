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

    /// NF4 dequantization: nf4_data + absmax → float tensor
    ///
    /// NF4 (Normal Float 4-bit) stores 4-bit indices into a 16-entry codebook
    /// derived from normal distribution quantiles. Each block of `blocksize`
    /// elements shares one absmax scaling factor.
    ///
    /// # Contract
    ///
    /// - `nf4_data` shape: `[N/2]` u8 (two 4-bit indices per byte), dtype U8
    /// - `absmax` shape: `[N/blocksize]` F32 — per-block scaling factors
    /// - Output shape: `[N]` F32
    ///
    /// Dequant: `output[i] = codebook[nf4_data[i]] * absmax[i / blocksize]`
    fn nf4_dequant(
        &self,
        nf4_data: &Tensor<R>,
        absmax: &Tensor<R>,
        blocksize: usize,
    ) -> Result<Tensor<R>>;

    /// NF4 fused GEMM: input × nf4_weight without materializing dequantized weight
    ///
    /// # Contract
    ///
    /// - `input` shape: `[..., M, K]`, dtype F32
    /// - `nf4_weight` shape: `[N*K/2]` u8 (packed NF4, row-major for [N, K])
    /// - `absmax` shape: `[N*K/blocksize]` F32
    /// - Output shape: `[..., M, N]` F32
    fn nf4_gemm(
        &self,
        input: &Tensor<R>,
        nf4_weight: &Tensor<R>,
        absmax: &Tensor<R>,
        n: usize,
        k: usize,
        blocksize: usize,
    ) -> Result<Tensor<R>>;
}
