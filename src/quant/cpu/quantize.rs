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
        quantize_cpu(self, input, format, None)
    }

    fn quantize_with_importance(
        &self,
        input: &Tensor<CpuRuntime>,
        format: QuantFormat,
        importance: Option<&[f32]>,
    ) -> Result<QuantTensor<CpuRuntime>> {
        quantize_cpu(self, input, format, importance)
    }
}

/// Checks an importance vector against the tensor it will weight
///
/// A vector of the wrong length is an ERROR and never a fallback to uniform
/// weights: the file that fallback writes is byte-indistinguishable from an
/// unweighted one, so the mistake would survive every check downstream and only
/// show up as a quality result nobody can explain.
///
/// A non-finite entry is rejected because it propagates into every weighted sum
/// the search takes and reaches the file as a NaN scale no reader can use. A
/// negative entry is rejected because an importance is a mean square
/// activation. A ZERO entry is legal and is not rejected: it means that column
/// contributes nothing, the search already guards every division on a positive
/// denominator, and llama.cpp treats it the same way.
///
/// [`ImportanceMatrix`](crate::quant::ImportanceMatrix) applies the same value
/// rule when it parses a file. This check is not that one: the slice reaching
/// a kernel need not have come from a file, and only the tensor being
/// quantized knows how long the vector has to be.
fn check_importance(importance: &[f32], last_dim: usize) -> Result<()> {
    if importance.len() != last_dim {
        return Err(Error::QuantError {
            reason: format!(
                "importance vector has {} entries but the quantized axis has {last_dim}. \
                 It is one entry per column, and a mismatch is never ignored: the output \
                 would be indistinguishable from an unweighted quantization.",
                importance.len(),
            ),
        });
    }
    if let Some((i, v)) = importance
        .iter()
        .enumerate()
        .find(|(_, v)| !v.is_finite() || **v < 0.0)
    {
        return Err(Error::QuantError {
            reason: format!(
                "importance entry {i} is {v}. Every entry must be finite and \
                 non-negative — it is a mean square activation, and a non-finite one \
                 reaches the file as a scale no reader can use."
            ),
        });
    }
    Ok(())
}

/// The whole CPU writer, with or without an importance vector
fn quantize_cpu(
    client: &CpuClient,
    input: &Tensor<CpuRuntime>,
    format: QuantFormat,
    importance: Option<&[f32]>,
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

    if let Some(imatrix) = importance {
        check_importance(imatrix, last_dim)?;
    }

    // Quantization is elementwise-per-block, so a contiguous f32 view is all
    // the kernels need. Cast first when the source is F16/BF16.
    let cast = if input.dtype() == DType::F32 {
        None
    } else {
        Some(client.cast(input, DType::F32).map_err(Error::Numr)?)
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

    match (format, importance) {
        (QuantFormat::Q4_0, None) => quantize::quantize_q4_0(values, &mut blocks),
        (QuantFormat::Q4_1, None) => quantize::quantize_q4_1(values, &mut blocks),
        (QuantFormat::Q8_0, None) => quantize::quantize_q8_0(values, &mut blocks),
        (QuantFormat::Q2K, None) => quantize::quantize_q2k(values, &mut blocks),
        (QuantFormat::Q3K, None) => quantize::quantize_q3k(values, &mut blocks),
        (QuantFormat::Q4K, None) => quantize::quantize_q4k(values, &mut blocks),
        (QuantFormat::Q5K, None) => quantize::quantize_q5k(values, &mut blocks),
        (QuantFormat::Q6K, None) => quantize::quantize_q6k(values, &mut blocks),
        (QuantFormat::Q2K, Some(m)) => quantize::quantize_q2k_imatrix(values, &mut blocks, m),
        (QuantFormat::Q3K, Some(m)) => quantize::quantize_q3k_imatrix(values, &mut blocks, m),
        (QuantFormat::Q4K, Some(m)) => quantize::quantize_q4k_imatrix(values, &mut blocks, m),
        (QuantFormat::Q5K, Some(m)) => quantize::quantize_q5k_imatrix(values, &mut blocks, m),
        (QuantFormat::Q6K, Some(m)) => quantize::quantize_q6k_imatrix(values, &mut blocks, m),
        (other, Some(_)) => {
            // The five K-quants are the formats `ggml-quants.c` gives an
            // `_impl` writer. Silently dropping the importance for anything
            // else would write a file that looks weighted and is not.
            return Err(Error::UnsupportedQuantFormat {
                format: format!(
                    "{} has no importance-weighted CPU quantize kernel",
                    other.name()
                ),
            });
        }
        (other, None) => {
            return Err(Error::UnsupportedQuantFormat {
                format: format!("{} has no CPU quantize kernel", other.name()),
            });
        }
    }

    QuantTensor::<CpuRuntime>::from_bytes(&blocks, format, &shape, input.device())
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
    fn quantize_q2k_round_trips_through_dequant() {
        let (client, device) = setup();
        let values = ramp(768);
        let input = Tensor::<CpuRuntime>::from_slice(&values, &[3, 256], &device).unwrap();

        let qt = client.quantize(&input, QuantFormat::Q2K).unwrap();
        assert_eq!(qt.shape(), &[3, 256]);
        assert_eq!(qt.storage_bytes(), 3 * 84);

        let back = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
        assert!(back.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn quantize_q3k_round_trips_through_dequant() {
        let (client, device) = setup();
        let values = ramp(768);
        let input = Tensor::<CpuRuntime>::from_slice(&values, &[3, 256], &device).unwrap();

        let qt = client.quantize(&input, QuantFormat::Q3K).unwrap();
        assert_eq!(qt.shape(), &[3, 256]);
        assert_eq!(qt.storage_bytes(), 3 * 110);

        let back = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
        assert!(back.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn quantize_rejects_format_without_kernel() {
        let (client, device) = setup();
        let input = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 256], &[256], &device).unwrap();
        assert!(client.quantize(&input, QuantFormat::Q8K).is_err());
    }
}
