//! CPU implementation of QuantMatmulOps

use crate::error::{Error, Result};
use crate::quant::traits::QuantMatmulOps;
use crate::quant::{QuantFormat, QuantTensor};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use super::kernels::{int4_gemm, int4_gemm_gptq, marlin_gemm, quant_matmul};

/// Validate input is F32 and extract (M, K) from shape.
fn validate_input(input: &Tensor<CpuRuntime>) -> Result<(usize, usize)> {
    if input.dtype() != DType::F32 {
        return Err(Error::QuantError {
            reason: format!("input must be F32, got {:?}", input.dtype()),
        });
    }
    let shape = input.shape();
    if shape.len() < 2 {
        return Err(Error::QuantError {
            reason: format!("input must be at least 2D, got {:?}", shape),
        });
    }
    let k = shape[shape.len() - 1];
    let m: usize = shape.iter().product::<usize>() / k;
    Ok((m, k))
}

/// Build output shape: replace last dim with n.
fn output_shape(input_shape: &[usize], n: usize) -> Vec<usize> {
    let mut s = input_shape[..input_shape.len() - 1].to_vec();
    s.push(n);
    s
}

impl QuantMatmulOps<CpuRuntime> for CpuClient {
    fn int4_gemm(
        &self,
        input: &Tensor<CpuRuntime>,
        qweight: &Tensor<CpuRuntime>,
        scales: &Tensor<CpuRuntime>,
        zeros: &Tensor<CpuRuntime>,
        group_size: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        let (m, k) = validate_input(input)?;
        let qw_shape = qweight.shape();
        if qw_shape.len() != 2 || qw_shape[0] != k {
            return Err(Error::QuantError {
                reason: format!("int4_gemm qweight shape mismatch: {:?}, K={}", qw_shape, k),
            });
        }
        let n = qw_shape[1] * 8;

        let inp = unsafe { input.storage().as_host_slice::<f32>() };
        let qw = unsafe { qweight.storage().as_host_slice::<u32>() };
        let sc = unsafe { scales.storage().as_host_slice::<f32>() };
        let zr = unsafe { zeros.storage().as_host_slice::<f32>() };

        let mut out = vec![0.0f32; m * n];
        int4_gemm::int4_gemm_f32(inp, qw, sc, zr, &mut out, m, k, n, group_size);

        Ok(Tensor::<CpuRuntime>::from_slice(
            &out,
            &output_shape(input.shape(), n),
            input.device(),
        ))
    }

    fn int4_gemm_gptq(
        &self,
        input: &Tensor<CpuRuntime>,
        qweight: &Tensor<CpuRuntime>,
        qzeros: &Tensor<CpuRuntime>,
        scales: &Tensor<CpuRuntime>,
        g_idx: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        let (m, k) = validate_input(input)?;
        let qw_shape = qweight.shape();
        if qw_shape.len() != 2 || qw_shape[0] != k / 8 {
            return Err(Error::QuantError {
                reason: format!(
                    "int4_gemm_gptq qweight shape mismatch: {:?}, K/8={}",
                    qw_shape,
                    k / 8
                ),
            });
        }
        let n = qw_shape[1];

        let inp = unsafe { input.storage().as_host_slice::<f32>() };
        let qw = unsafe { qweight.storage().as_host_slice::<u32>() };
        let qz = unsafe { qzeros.storage().as_host_slice::<u32>() };
        let sc = unsafe { scales.storage().as_host_slice::<f32>() };
        let gi = unsafe { g_idx.storage().as_host_slice::<i32>() };

        let mut out = vec![0.0f32; m * n];
        int4_gemm_gptq::int4_gemm_gptq_f32(inp, qw, qz, sc, gi, &mut out, m, k, n);

        Ok(Tensor::<CpuRuntime>::from_slice(
            &out,
            &output_shape(input.shape(), n),
            input.device(),
        ))
    }

    fn marlin_gemm(
        &self,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        scales: &Tensor<CpuRuntime>,
        zeros: &Tensor<CpuRuntime>,
        group_size: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        let (m, k) = validate_input(input)?;
        let w_shape = weight.shape();
        if w_shape.len() != 2 || w_shape[0] != k / 8 {
            return Err(Error::QuantError {
                reason: format!(
                    "marlin_gemm weight shape mismatch: {:?}, K/8={}",
                    w_shape,
                    k / 8
                ),
            });
        }
        let n = w_shape[1];

        let inp = unsafe { input.storage().as_host_slice::<f32>() };
        let wt = unsafe { weight.storage().as_host_slice::<u32>() };
        let sc = unsafe { scales.storage().as_host_slice::<f32>() };
        let zr = unsafe { zeros.storage().as_host_slice::<f32>() };

        let mut out = vec![0.0f32; m * n];
        marlin_gemm::marlin_gemm_f32(inp, wt, sc, zr, &mut out, m, k, n, group_size);

        Ok(Tensor::<CpuRuntime>::from_slice(
            &out,
            &output_shape(input.shape(), n),
            input.device(),
        ))
    }

    fn quant_matmul_batch(
        &self,
        activation: &Tensor<CpuRuntime>,
        weights: &[&QuantTensor<CpuRuntime>],
    ) -> Result<Vec<Tensor<CpuRuntime>>> {
        if weights.is_empty() {
            return Ok(vec![]);
        }
        if weights.len() == 1 {
            return Ok(vec![self.quant_matmul(activation, weights[0])?]);
        }

        // Validate activation
        if activation.dtype() != DType::F32 {
            return Err(Error::QuantError {
                reason: format!(
                    "quant_matmul_batch activation must be F32, got {:?}",
                    activation.dtype()
                ),
            });
        }

        let a_shape = activation.shape();
        if a_shape.is_empty() {
            return Err(Error::QuantError {
                reason: "quant_matmul_batch activation must be at least 1D".into(),
            });
        }
        let k = a_shape[a_shape.len() - 1];
        let total_elements: usize = a_shape.iter().product();
        let m = total_elements / k;

        // Validate all weights have same format and K
        let format = weights[0].format();
        for (i, w) in weights.iter().enumerate() {
            let ws = w.shape();
            if ws.len() != 2 {
                return Err(Error::QuantError {
                    reason: format!("weight[{}] must be 2D, got {:?}", i, ws),
                });
            }
            if ws[1] != k {
                return Err(Error::QuantError {
                    reason: format!("weight[{}] K={} != activation K={}", i, ws[1], k),
                });
            }
            if w.format() != format {
                // Fall back to sequential if mixed formats
                return weights
                    .iter()
                    .map(|w| self.quant_matmul(activation, w))
                    .collect();
            }
        }

        // Validate format is supported
        match format {
            QuantFormat::Q4_0 | QuantFormat::Q8_0 | QuantFormat::Q4K | QuantFormat::Q6K => {}
            other => {
                return Err(Error::UnsupportedQuantFormat {
                    format: format!("{} (CPU quant_matmul_batch not implemented)", other),
                });
            }
        }

        let act_data = unsafe { activation.storage().as_host_slice::<f32>() };

        // Build weight list and output buffers
        let weight_list: Vec<(&[u8], usize)> = weights
            .iter()
            .map(|w| {
                let bytes = unsafe { w.storage().as_host_slice::<u8>() };
                let n = w.shape()[0];
                (bytes, n)
            })
            .collect();

        let mut output_bufs: Vec<Vec<f32>> = weights
            .iter()
            .map(|w| vec![0.0f32; m * w.shape()[0]])
            .collect();

        {
            let mut output_slices: Vec<&mut [f32]> =
                output_bufs.iter_mut().map(|v| v.as_mut_slice()).collect();
            quant_matmul::quant_matmul_batch_f32(
                act_data,
                &weight_list,
                &mut output_slices,
                m,
                k,
                format,
            );
        }

        // Build output tensors
        let results: Vec<Tensor<CpuRuntime>> = weights
            .iter()
            .zip(output_bufs.into_iter())
            .map(|(w, buf)| {
                let n = w.shape()[0];
                let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
                out_shape.push(n);
                Tensor::<CpuRuntime>::from_slice(&buf, &out_shape, activation.device())
            })
            .collect();

        Ok(results)
    }

    fn quant_matmul(
        &self,
        activation: &Tensor<CpuRuntime>,
        weight: &QuantTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        // Validate activation dtype
        if activation.dtype() != DType::F32 {
            return Err(Error::QuantError {
                reason: format!(
                    "quant_matmul activation must be F32, got {:?}",
                    activation.dtype()
                ),
            });
        }

        // Validate weight is 2D: [N, K]
        let w_shape = weight.shape();
        if w_shape.len() != 2 {
            return Err(Error::QuantError {
                reason: format!("quant_matmul weight must be 2D [N, K], got {:?}", w_shape),
            });
        }
        let n = w_shape[0];
        let k = w_shape[1];

        // Validate activation shape: [..., M, K]
        let a_shape = activation.shape();
        if a_shape.is_empty() {
            return Err(Error::QuantError {
                reason: "quant_matmul activation must be at least 1D".into(),
            });
        }
        let a_k = a_shape[a_shape.len() - 1];
        if a_k != k {
            return Err(Error::QuantError {
                reason: format!(
                    "quant_matmul dimension mismatch: activation K={}, weight K={}",
                    a_k, k
                ),
            });
        }

        // Validate format is supported
        match weight.format() {
            QuantFormat::Q4_0 | QuantFormat::Q8_0 | QuantFormat::Q4K | QuantFormat::Q6K => {}
            other => {
                return Err(Error::UnsupportedQuantFormat {
                    format: format!("{} (CPU quant_matmul not implemented)", other),
                });
            }
        }

        // Compute batch dimensions and M
        let total_elements: usize = a_shape.iter().product();
        let m = total_elements / k;

        // Get raw data (zero-copy for CPU)
        // SAFETY: CpuRuntime stores data as host pointers
        let act_data = unsafe { activation.storage().as_host_slice::<f32>() };
        let weight_bytes = unsafe { weight.storage().as_host_slice::<u8>() };

        // Run kernel
        let mut output = vec![0.0f32; m * n];
        quant_matmul::quant_matmul_f32(
            act_data,
            weight_bytes,
            &mut output,
            m,
            k,
            n,
            weight.format(),
        );

        // Build output shape: [..., M, N] (replace last dim K with N)
        let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
        out_shape.push(n);

        Ok(Tensor::<CpuRuntime>::from_slice(
            &output,
            &out_shape,
            activation.device(),
        ))
    }

    fn quant_swiglu(
        &self,
        activation: &Tensor<CpuRuntime>,
        gate_weight: &QuantTensor<CpuRuntime>,
        up_weight: &QuantTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        // CPU fallback: separate matmuls + fused silu_mul
        let gate = self.quant_matmul(activation, gate_weight)?;
        let up = self.quant_matmul(activation, up_weight)?;
        use numr::ops::ActivationOps;
        self.silu_mul(&gate, &up).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::traits::DequantOps;
    use half::f16;
    use numr::ops::MatmulOps;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_quant_matmul_q4_0_basic() {
        let (client, device) = setup();

        // activation [1, 32], weight [1, 32] → output [1, 1]
        let act = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 32], &[1, 32], &device);

        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        block[2..18].fill(0x99); // dequant value = 2.0

        let qt =
            QuantTensor::<CpuRuntime>::from_bytes(&block, QuantFormat::Q4_0, &[1, 32], &device)
                .unwrap();

        let result = client.quant_matmul(&act, &qt).unwrap();
        assert_eq!(result.shape(), &[1, 1]);

        let data = result.to_vec::<f32>();
        assert!(
            (data[0] - 64.0).abs() < 0.5,
            "expected ~64.0, got {}",
            data[0]
        );
    }

    #[test]
    fn test_quant_matmul_matches_dequant_matmul() {
        let (client, device) = setup();

        // activation [2, 32]
        let act_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let act = Tensor::<CpuRuntime>::from_slice(&act_data, &[2, 32], &device);

        // weight [3, 32] as Q8_0 (3 rows, each 34 bytes)
        let mut weight_bytes = Vec::new();
        for row in 0..3 {
            let mut block = [0u8; 34];
            block[0..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
            // Different qs per row for variety
            block[2..34].fill((row + 1) as u8);
            weight_bytes.extend_from_slice(&block);
        }

        let qt = QuantTensor::<CpuRuntime>::from_bytes(
            &weight_bytes,
            QuantFormat::Q8_0,
            &[3, 32],
            &device,
        )
        .unwrap();

        // Method 1: quant_matmul
        let result_qm = client.quant_matmul(&act, &qt).unwrap();

        // Method 2: dequant then matmul
        let dequant_w = client.dequantize(&qt, DType::F32).unwrap();
        // dequant gives [3, 32], matmul needs act [2, 32] × w^T [32, 3]
        // Our quant_matmul does act × w^T layout, so we need to transpose for standard matmul
        let dequant_w_t = dequant_w.transpose(0isize, 1isize).unwrap();
        let result_dm = MatmulOps::matmul(&client, &act, &dequant_w_t).unwrap();

        assert_eq!(result_qm.shape(), result_dm.shape());

        let qm_data = result_qm.to_vec::<f32>();
        let dm_data = result_dm.to_vec::<f32>();
        for (i, (&a, &b)) in qm_data.iter().zip(dm_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-2,
                "mismatch at index {}: quant_matmul={}, dequant+matmul={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_quant_matmul_dim_mismatch() {
        let (client, device) = setup();

        let act = Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; 64], &[2, 32], &device);

        // Weight K=64 ≠ activation K=32
        let block = vec![0u8; 2 * 34]; // 2 blocks of Q8_0 = 64 elements
        let qt =
            QuantTensor::<CpuRuntime>::from_bytes(&block, QuantFormat::Q8_0, &[1, 64], &device)
                .unwrap();

        let result = client.quant_matmul(&act, &qt);
        assert!(result.is_err());
    }

    #[test]
    fn test_quant_matmul_unsupported_format() {
        let (client, device) = setup();

        let act = Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; 256], &[1, 256], &device);

        let block = vec![0u8; 84]; // Q2K block
        let qt =
            QuantTensor::<CpuRuntime>::from_bytes(&block, QuantFormat::Q2K, &[1, 256], &device)
                .unwrap();

        let result = client.quant_matmul(&act, &qt);
        assert!(result.is_err());
    }
}
