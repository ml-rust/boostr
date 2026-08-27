//! CPU implementation of QuantizeOps

use crate::error::{Error, Result};
use crate::quant::traits::QuantizeOps;
use crate::quant::{QuantFormat, QuantTensor};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use super::kernels::quantize;

impl QuantizeOps<CpuRuntime> for CpuClient {
    fn quantize(
        &self,
        input: &Tensor<CpuRuntime>,
        format: QuantFormat,
    ) -> Result<QuantTensor<CpuRuntime>> {
        if !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::QuantError {
                reason: format!("quantize input must be float, got {:?}", input.dtype()),
            });
        }

        let shape = input.shape().to_vec();
        let last_dim = shape.last().copied().unwrap_or(0);
        if shape.is_empty() || !last_dim.is_multiple_of(format.block_size()) {
            return Err(Error::QuantError {
                reason: format!(
                    "last dimension {} is not a multiple of {}'s block_size {}",
                    last_dim,
                    format.name(),
                    format.block_size(),
                ),
            });
        }

        // Quantization is elementwise-per-block, so a contiguous f32 view is all
        // the kernels need. Cast first when the source is F16/BF16.
        let cast = if input.dtype() == DType::F32 {
            None
        } else {
            Some(self.cast(input, DType::F32).map_err(Error::Numr)?)
        };
        let base = cast.as_ref().unwrap_or(input);
        // Blocks are packed along the last axis in memory order, so a strided
        // view has to be materialized before the kernels see it.
        let packed = if base.is_contiguous() {
            None
        } else {
            Some(base.contiguous().map_err(Error::Numr)?)
        };
        let src = packed.as_ref().unwrap_or(base);
        // SAFETY: CpuRuntime stores data as host pointers, and `src` is F32.
        let values = unsafe { src.storage().as_host_slice::<f32>() };

        let numel: usize = shape.iter().product();
        let mut blocks = vec![0u8; format.storage_bytes(numel)?];

        match format {
            QuantFormat::Q4_0 => quantize::quantize_q4_0(values, &mut blocks),
            QuantFormat::Q4_1 => quantize::quantize_q4_1(values, &mut blocks),
            QuantFormat::Q8_0 => quantize::quantize_q8_0(values, &mut blocks),
            QuantFormat::Q4K => quantize::quantize_q4k(values, &mut blocks),
            QuantFormat::Q5K => quantize::quantize_q5k(values, &mut blocks),
            QuantFormat::Q6K => quantize::quantize_q6k(values, &mut blocks),
            other => {
                return Err(Error::UnsupportedQuantFormat {
                    format: format!("{} has no CPU quantize kernel", other.name()),
                });
            }
        }

        QuantTensor::<CpuRuntime>::from_bytes(&blocks, format, &shape, input.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::DequantOps;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    /// Deterministic ramp with a sign flip, so no sub-block is constant.
    fn ramp(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.031 * (1.0 + (i / 256) as f32))
            .collect()
    }

    #[test]
    fn quantize_q8_0_round_trips_through_dequant() {
        let (client, device) = setup();
        let values = ramp(512);
        let input = Tensor::<CpuRuntime>::from_slice(&values, &[512], &device).unwrap();

        let qt = client.quantize(&input, QuantFormat::Q8_0).unwrap();
        assert_eq!(qt.shape(), &[512]);
        assert_eq!(qt.storage_bytes(), 16 * 34);

        let back = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
        for (got, want) in back.iter().zip(&values) {
            assert!((got - want).abs() < 0.01, "got {got}, want {want}");
        }
    }

    #[test]
    fn quantize_q4k_round_trips_through_dequant() {
        let (client, device) = setup();
        let values = ramp(768);
        let input = Tensor::<CpuRuntime>::from_slice(&values, &[3, 256], &device).unwrap();

        let qt = client.quantize(&input, QuantFormat::Q4K).unwrap();
        assert_eq!(qt.shape(), &[3, 256]);
        assert_eq!(qt.storage_bytes(), 3 * 144);

        let back = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
        assert!(back.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn quantize_rejects_unaligned_last_dim() {
        let (client, device) = setup();
        let input = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 100], &[100], &device).unwrap();
        assert!(client.quantize(&input, QuantFormat::Q4K).is_err());
    }

    #[test]
    fn quantize_rejects_format_without_kernel() {
        let (client, device) = setup();
        let input = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 256], &[256], &device).unwrap();
        assert!(client.quantize(&input, QuantFormat::Q2K).is_err());
    }
}
