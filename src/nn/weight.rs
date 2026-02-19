//! Weight enum for storing standard or quantized tensors in VarMap

use crate::error::{Error, Result};
use crate::quant::QuantFormat;
use crate::quant::tensor::QuantTensor;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A weight that is either a standard tensor or a quantized tensor.
///
/// GGUF models contain a mix of unquantized (norms, embeddings) and quantized
/// (linear weights) tensors. `Weight` provides type-safe storage for both.
pub enum Weight<R: Runtime> {
    /// Standard floating-point tensor
    Standard(Tensor<R>),
    /// Block-quantized tensor (GGUF formats)
    Quantized(QuantTensor<R>),
}

impl<R: Runtime> Weight<R> {
    /// Returns `true` if this weight is quantized.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized(_))
    }

    /// Get as a standard tensor, or error if quantized.
    pub fn as_tensor(&self) -> Result<&Tensor<R>> {
        match self {
            Self::Standard(t) => Ok(t),
            Self::Quantized(_) => Err(Error::ModelError {
                reason: "expected standard tensor, got quantized".into(),
            }),
        }
    }

    /// Get as a quantized tensor, or error if standard.
    pub fn as_quant_tensor(&self) -> Result<&QuantTensor<R>> {
        match self {
            Self::Quantized(q) => Ok(q),
            Self::Standard(_) => Err(Error::ModelError {
                reason: "expected quantized tensor, got standard".into(),
            }),
        }
    }

    /// Consume and return the inner standard tensor, or error if quantized.
    pub fn into_tensor(self) -> Result<Tensor<R>> {
        match self {
            Self::Standard(t) => Ok(t),
            Self::Quantized(_) => Err(Error::ModelError {
                reason: "expected standard tensor, got quantized".into(),
            }),
        }
    }

    /// Consume and return the inner quantized tensor, or error if standard.
    pub fn into_quant_tensor(self) -> Result<QuantTensor<R>> {
        match self {
            Self::Quantized(q) => Ok(q),
            Self::Standard(_) => Err(Error::ModelError {
                reason: "expected quantized tensor, got standard".into(),
            }),
        }
    }
}

impl<R: Runtime<DType = numr::dtype::DType>> Weight<R> {
    /// Logical shape of the weight.
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Standard(t) => t.shape(),
            Self::Quantized(q) => q.shape(),
        }
    }

    /// Get the quantization format if quantized, `None` if standard.
    pub fn quant_format(&self) -> Option<QuantFormat> {
        match self {
            Self::Quantized(q) => Some(q.format()),
            Self::Standard(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn device() -> CpuDevice {
        CpuDevice::new()
    }

    #[test]
    fn test_standard_weight() {
        let d = device();
        let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &d);
        let w = Weight::Standard(t);

        assert!(!w.is_quantized());
        assert_eq!(w.shape(), &[3]);
        assert!(w.as_tensor().is_ok());
        assert!(w.as_quant_tensor().is_err());
        assert!(w.quant_format().is_none());
    }

    #[test]
    fn test_quantized_weight() {
        let d = device();
        let data = vec![0u8; 18]; // 1 Q4_0 block = 32 elements
        let qt =
            QuantTensor::<CpuRuntime>::from_bytes(&data, QuantFormat::Q4_0, &[32], &d).unwrap();
        let w = Weight::Quantized(qt);

        assert!(w.is_quantized());
        assert_eq!(w.shape(), &[32]);
        assert!(w.as_tensor().is_err());
        assert!(w.as_quant_tensor().is_ok());
        assert_eq!(w.quant_format(), Some(QuantFormat::Q4_0));
    }

    #[test]
    fn test_into_tensor() {
        let d = device();
        let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &d);
        let w = Weight::Standard(t);
        assert!(w.into_tensor().is_ok());
    }

    #[test]
    fn test_into_quant_tensor() {
        let d = device();
        let data = vec![0u8; 18];
        let qt =
            QuantTensor::<CpuRuntime>::from_bytes(&data, QuantFormat::Q4_0, &[32], &d).unwrap();
        let w = Weight::Quantized(qt);
        assert!(w.into_quant_tensor().is_ok());
    }
}
