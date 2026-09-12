//! One-shot tensor loading: F32 (with dequantization) and `QuantTensor`.

use super::super::types::GgmlType;
use super::open::Gguf;
use crate::error::{Error, Result};
use crate::quant::QuantTensor;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl Gguf {
    /// Load an F32 tensor (for unquantized F32/F16/BF16 tensors, converted to F32)
    pub fn load_tensor_f32<R: Runtime<DType = DType>>(
        &mut self,
        name: &str,
        device: &R::Device,
    ) -> Result<Tensor<R>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::ModelError {
                reason: format!("GGUF tensor not found: {name}"),
            })?
            .clone();

        let bytes = self.read_tensor_bytes(name)?;

        // GGUF stores shape in GGML order (innermost first), reverse for row-major
        let mut shape = info.shape.clone();
        shape.reverse();

        let data: Vec<f32> = match info.ggml_type {
            GgmlType::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            GgmlType::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect(),
            GgmlType::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect(),
            GgmlType::F64 => bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| {
                    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect(),
            other => {
                // Dequantize quantized types to f32
                let format = other.to_quant_format().ok_or_else(|| Error::ModelError {
                    reason: format!(
                        "tensor '{name}' has type {other:?} which cannot be dequantized"
                    ),
                })?;
                let numel: usize = shape.iter().product();
                let row_k = info.shape[0]; // innermost dim (before reversal) = K per row
                let row_bytes = format.storage_bytes(row_k)?;
                let n_rows = numel / row_k;
                let expected_bytes = n_rows * row_bytes;
                if bytes.len() < expected_bytes {
                    return Err(Error::ModelError {
                        reason: format!(
                            "tensor '{name}': expected at least {expected_bytes} bytes for dequantization, got {}",
                            bytes.len()
                        ),
                    });
                }
                let mut data = vec![0.0f32; numel];
                for row in 0..n_rows {
                    let src = &bytes[row * row_bytes..(row + 1) * row_bytes];
                    let dst = &mut data[row * row_k..(row + 1) * row_k];
                    crate::quant::cpu::kernels::quant_matmul::dequant_row_f32(src, dst, format);
                }
                data
            }
        };

        Tensor::<R>::from_slice(&data, &shape, device).map_err(Error::Numr)
    }

    /// Load a quantized tensor as QuantTensor
    pub fn load_tensor_quantized<R: Runtime<DType = DType>>(
        &mut self,
        name: &str,
        device: &R::Device,
    ) -> Result<QuantTensor<R>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::ModelError {
                reason: format!("GGUF tensor not found: {name}"),
            })?
            .clone();

        let format = info
            .ggml_type
            .to_quant_format()
            .ok_or_else(|| Error::ModelError {
                reason: format!("tensor '{name}' type {:?} is not quantized", info.ggml_type),
            })?;

        let bytes = self.read_tensor_bytes(name)?;

        // GGUF stores shape in GGML order, reverse for row-major
        let mut shape = info.shape.clone();
        shape.reverse();

        QuantTensor::from_bytes(&bytes, format, &shape, device)
    }
}

#[cfg(test)]
mod tests {
    use super::super::open::tests::{create_test_gguf, create_test_gguf_bytes};
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_load_f32_tensor() {
        let (_, device) = cpu_setup();
        let f = create_test_gguf();
        let mut gguf = Gguf::open(f.path()).unwrap();

        let tensor = gguf
            .load_tensor_f32::<CpuRuntime>("weight_f32", &device)
            .unwrap();
        // GGUF 1D tensor: shape reversed is still [4]
        assert_eq!(tensor.shape(), &[4]);
        let data = tensor.to_vec::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_load_quantized_tensor() {
        let f = create_test_gguf();
        let mut gguf = Gguf::open(f.path()).unwrap();

        let device = numr::runtime::cpu::CpuDevice::new();
        let qt = gguf
            .load_tensor_quantized::<numr::runtime::cpu::CpuRuntime>("weight_q4", &device)
            .unwrap();
        assert_eq!(qt.shape(), &[32]);
        assert_eq!(qt.format(), crate::quant::QuantFormat::Q4_0);
    }

    #[test]
    fn test_from_bytes_load_f32() {
        let (_, device) = cpu_setup();
        let buf = create_test_gguf_bytes();
        let mut gguf = Gguf::from_bytes(buf).unwrap();

        let tensor = gguf
            .load_tensor_f32::<CpuRuntime>("weight_f32", &device)
            .unwrap();
        assert_eq!(tensor.shape(), &[4]);
        let data = tensor.to_vec::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_bytes_load_quantized() {
        let buf = create_test_gguf_bytes();
        let mut gguf = Gguf::from_bytes(buf).unwrap();

        let device = numr::runtime::cpu::CpuDevice::new();
        let qt = gguf
            .load_tensor_quantized::<CpuRuntime>("weight_q4", &device)
            .unwrap();
        assert_eq!(qt.shape(), &[32]);
        assert_eq!(qt.format(), crate::quant::QuantFormat::Q4_0);
    }
}
