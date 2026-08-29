//! CPU implementation of DequantOps

use crate::error::{Error, Result};
use crate::quant::traits::DequantOps;
use crate::quant::{QuantFormat, QuantScheme, QuantTensor};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use super::kernels::{dequant, nf4, tcf};

impl DequantOps<CpuRuntime> for CpuClient {
    fn nf4_dequant(
        &self,
        nf4_data: &Tensor<CpuRuntime>,
        absmax: &Tensor<CpuRuntime>,
        blocksize: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        if nf4_data.dtype() != DType::U8 {
            return Err(Error::QuantError {
                reason: format!("nf4_dequant data must be U8, got {:?}", nf4_data.dtype()),
            });
        }
        let data = unsafe { nf4_data.storage().as_host_slice::<u8>() };
        let abs = unsafe { absmax.storage().as_host_slice::<f32>() };
        let n = data.len() * 2;
        let mut output = vec![0.0f32; n];
        nf4::nf4_dequant_f32(data, abs, blocksize, &mut output);
        Ok(Tensor::<CpuRuntime>::from_slice(
            &output,
            &[n],
            nf4_data.device(),
        )?)
    }

    fn nf4_gemm(
        &self,
        input: &Tensor<CpuRuntime>,
        nf4_weight: &Tensor<CpuRuntime>,
        absmax: &Tensor<CpuRuntime>,
        n_out: usize,
        k: usize,
        blocksize: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        if input.dtype() != DType::F32 {
            return Err(Error::QuantError {
                reason: format!("nf4_gemm input must be F32, got {:?}", input.dtype()),
            });
        }
        let in_shape = input.shape();
        let m: usize = in_shape.iter().product::<usize>() / k;
        let inp = unsafe { input.storage().as_host_slice::<f32>() };
        let wt = unsafe { nf4_weight.storage().as_host_slice::<u8>() };
        let abs = unsafe { absmax.storage().as_host_slice::<f32>() };
        let mut output = vec![0.0f32; m * n_out];
        nf4::nf4_gemm_f32(inp, wt, abs, &mut output, m, k, n_out, blocksize);
        let mut out_shape = in_shape[..in_shape.len() - 1].to_vec();
        out_shape.push(n_out);
        Ok(Tensor::<CpuRuntime>::from_slice(
            &output,
            &out_shape,
            input.device(),
        )?)
    }

    fn dequantize(
        &self,
        qt: &QuantTensor<CpuRuntime>,
        target_dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        // Validate target is floating point
        if !matches!(
            target_dtype,
            DType::F32 | DType::F16 | DType::BF16 | DType::F64
        ) {
            return Err(Error::QuantError {
                reason: format!("dequantize target must be float, got {:?}", target_dtype),
            });
        }

        let numel = qt.numel();

        // Read raw packed bytes from storage (zero-copy for CPU)
        // SAFETY: CpuRuntime stores data as host pointers
        let block_bytes = unsafe { qt.storage().as_host_slice::<u8>() };

        // Dequantize to f32 first
        let f32_output = match qt.scheme() {
            QuantScheme::Gguf(format) => {
                let mut out = vec![0.0f32; numel];
                match format {
                    QuantFormat::Q4_0 => dequant::dequant_q4_0(block_bytes, &mut out),
                    QuantFormat::Q4_1 => dequant::dequant_q4_1(block_bytes, &mut out),
                    QuantFormat::Q5_0 => dequant::dequant_q5_0(block_bytes, &mut out),
                    QuantFormat::Q5_1 => dequant::dequant_q5_1(block_bytes, &mut out),
                    QuantFormat::Q8_0 => dequant::dequant_q8_0(block_bytes, &mut out),
                    QuantFormat::Q8_1 => dequant::dequant_q8_1(block_bytes, &mut out),
                    QuantFormat::Q2K => dequant::dequant_q2k(block_bytes, &mut out),
                    QuantFormat::Q3K => dequant::dequant_q3k(block_bytes, &mut out),
                    QuantFormat::Q4K => dequant::dequant_q4k(block_bytes, &mut out),
                    QuantFormat::Q5K => dequant::dequant_q5k(block_bytes, &mut out),
                    QuantFormat::Q6K => dequant::dequant_q6k(block_bytes, &mut out),
                    QuantFormat::Q8K => dequant::dequant_q8k(block_bytes, &mut out),
                    QuantFormat::IQ4NL => dequant::dequant_iq4_nl(block_bytes, &mut out),
                    QuantFormat::IQ4XS => dequant::dequant_iq4_xs(block_bytes, &mut out),
                    QuantFormat::IQ2XXS => dequant::dequant_iq2_xxs(block_bytes, &mut out),
                    QuantFormat::IQ2XS => dequant::dequant_iq2_xs(block_bytes, &mut out),
                    QuantFormat::IQ2S => dequant::dequant_iq2_s(block_bytes, &mut out),
                    QuantFormat::IQ3XXS => dequant::dequant_iq3_xxs(block_bytes, &mut out),
                    QuantFormat::IQ3S => dequant::dequant_iq3_s(block_bytes, &mut out),
                    QuantFormat::IQ1S => dequant::dequant_iq1_s(block_bytes, &mut out),
                    QuantFormat::IQ1M => dequant::dequant_iq1_m(block_bytes, &mut out),
                    QuantFormat::TQ1_0 => dequant::dequant_tq1_0(block_bytes, &mut out),
                    QuantFormat::TQ2_0 => dequant::dequant_tq2_0(block_bytes, &mut out),
                }
                out
            }
            // TCF is plane-major and its two-level encodings carry a second
            // scale level, so it has its own kernel. See
            // `kernels::tcf` for why that kernel delegates to `tcf-core`.
            QuantScheme::Tcf(encoding) => tcf::dequant_tcf(block_bytes, encoding, qt.shape())?,
        };

        // Create f32 tensor
        let f32_tensor = Tensor::<CpuRuntime>::from_slice(&f32_output, qt.shape(), qt.device())?;

        // Cast to target dtype if needed
        if target_dtype == DType::F32 {
            Ok(f32_tensor)
        } else {
            self.cast(&f32_tensor, target_dtype).map_err(Error::Numr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_dequant_q4_0_roundtrip() {
        let (client, device) = setup();

        // Create a Q4_0 block with known values
        // scale=2.0, all nibbles=9 → value = (9-8)*2.0 = 2.0
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        block[2..18].fill(0x99);

        let qt = QuantTensor::<CpuRuntime>::from_bytes(&block, QuantFormat::Q4_0, &[32], &device)
            .unwrap();

        let result = client.dequantize(&qt, DType::F32).unwrap();
        assert_eq!(result.shape(), &[32]);
        assert_eq!(result.dtype(), DType::F32);

        let data = result.to_vec::<f32>();
        for &v in &data {
            assert!((v - 2.0).abs() < 0.01, "expected 2.0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q8_0_roundtrip() {
        let (client, device) = setup();

        let mut block = [0u8; 34];
        block[0..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        // qs = 6 as i8 → value = 6 * 0.5 = 3.0
        block[2..34].fill(6);

        let qt = QuantTensor::<CpuRuntime>::from_bytes(&block, QuantFormat::Q8_0, &[32], &device)
            .unwrap();

        let result = client.dequantize(&qt, DType::F32).unwrap();
        let data = result.to_vec::<f32>();
        for &v in &data {
            assert!((v - 3.0).abs() < 0.01, "expected 3.0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_q4k_basic() {
        let (client, device) = setup();

        // Minimal test: all zeros → output should be all zeros
        let block = vec![0u8; 144];
        let qt = QuantTensor::<CpuRuntime>::from_bytes(&block, QuantFormat::Q4K, &[256], &device)
            .unwrap();

        let result = client.dequantize(&qt, DType::F32).unwrap();
        assert_eq!(result.shape(), &[256]);
        let data = result.to_vec::<f32>();
        for &v in &data {
            assert!(v.abs() < 1e-5);
        }
    }

    #[test]
    fn test_dequant_q6k_basic() {
        let (client, device) = setup();

        let block = vec![0u8; 210];
        let qt = QuantTensor::<CpuRuntime>::from_bytes(&block, QuantFormat::Q6K, &[256], &device)
            .unwrap();

        let result = client.dequantize(&qt, DType::F32).unwrap();
        assert_eq!(result.shape(), &[256]);
    }

    #[test]
    fn test_dequant_to_f64() {
        let (client, device) = setup();

        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block[2..18].fill(0x99); // value = 1.0

        let qt = QuantTensor::<CpuRuntime>::from_bytes(&block, QuantFormat::Q4_0, &[32], &device)
            .unwrap();

        let result = client.dequantize(&qt, DType::F64).unwrap();
        assert_eq!(result.dtype(), DType::F64);
    }

    #[test]
    fn test_dequant_iq1s_basic() {
        let (client, device) = setup();

        // IQ1S is now supported — test with all zeros
        let block = vec![0u8; 50];
        let qt = QuantTensor::<CpuRuntime>::from_bytes(&block, QuantFormat::IQ1S, &[256], &device)
            .unwrap();

        let result = client.dequantize(&qt, DType::F32);
        assert!(result.is_ok());
        let data = result.unwrap().to_vec::<f32>();
        assert_eq!(data.len(), 256);
        for &v in &data {
            assert!(v.abs() < 1e-5, "expected 0, got {}", v);
        }
    }

    #[test]
    fn test_dequant_invalid_target() {
        let (client, device) = setup();

        let block = vec![0u8; 18];
        let qt = QuantTensor::<CpuRuntime>::from_bytes(&block, QuantFormat::Q4_0, &[32], &device)
            .unwrap();

        // I32 is not a valid dequant target
        let result = client.dequantize(&qt, DType::I32);
        assert!(result.is_err());
    }

    /// A TCF weight reaches `DequantOps` through the SAME trait and the same
    /// tensor type as a GGUF weight, and its values are `tcf-core`'s own,
    /// bit for bit.
    #[test]
    fn a_tcf_tensor_dequantizes_through_the_same_trait() {
        use crate::quant::TcfEncoding;
        use tcf_core::{NativeEncoding, dequantize as tcf_dequantize, pack, quantize, unpack};

        let (client, device) = setup();
        let shape = [3usize, 320];
        let values: Vec<f32> = (0..shape[0] * shape[1])
            .map(|i| (i as f32 * 0.021).sin() * 2.0 - 0.3)
            .collect();

        let native = NativeEncoding::Q4AS32DT64;
        let layout = native.layout();
        let dims: Vec<u64> = shape.iter().map(|d| *d as u64).collect();
        let tiles = quantize(&values, &dims, 2, layout).expect("quantizes");
        let payload = pack(&tiles, layout).expect("packs");

        let encoding = TcfEncoding::new(native);
        let qt = QuantTensor::<CpuRuntime>::from_bytes(&payload, encoding, &shape, &device)
            .expect("packed bytes go to the device as-is");
        assert_eq!(qt.storage_bytes(), payload.len());

        let got = client
            .dequantize(&qt, DType::F32)
            .expect("dequantizes")
            .to_vec::<f32>();
        let reference = tcf_dequantize(&unpack(&payload, 15, layout).expect("unpacks"), layout)
            .expect("dequantizes");

        let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
        let reference_bits: Vec<u32> = reference.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got_bits, reference_bits);
    }
}
