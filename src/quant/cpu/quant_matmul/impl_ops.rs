//! CPU implementation of QuantMatmulOps for CpuClient.

use crate::error::{Error, Result};
use crate::quant::QuantTensor;
use crate::quant::traits::QuantMatmulOps;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use super::super::kernels::{int4_gemm, int4_gemm_gptq, marlin_gemm, quant_matmul};
use super::helpers::{output_shape, validate_input};

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

        let num_groups = k / group_size;
        if inp.len() != m * k {
            return Err(Error::QuantError {
                reason: format!(
                    "int4_gemm input slice len {} != m*k {}*{}={}",
                    inp.len(),
                    m,
                    k,
                    m * k
                ),
            });
        }
        if qw.len() != k * (n / 8) {
            return Err(Error::QuantError {
                reason: format!(
                    "int4_gemm qweight slice len {} != k*(n/8) {}*{}={}",
                    qw.len(),
                    k,
                    n / 8,
                    k * (n / 8)
                ),
            });
        }
        if sc.len() != num_groups * n {
            return Err(Error::QuantError {
                reason: format!(
                    "int4_gemm scales slice len {} != groups*n {}*{}={}",
                    sc.len(),
                    num_groups,
                    n,
                    num_groups * n
                ),
            });
        }
        if zr.len() != num_groups * n {
            return Err(Error::QuantError {
                reason: format!(
                    "int4_gemm zeros slice len {} != groups*n {}*{}={}",
                    zr.len(),
                    num_groups,
                    n,
                    num_groups * n
                ),
            });
        }

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
            .zip(output_bufs)
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

        // Compute batch dimensions and M
        let total_elements: usize = a_shape.iter().product();
        let m = total_elements / k;

        // Ensure activation is contiguous — non-contiguous tensors (from permute/reshape)
        // would cause the raw storage to not match the logical layout.
        let activation = if !activation.is_contiguous() {
            activation.contiguous()
        } else {
            activation.clone()
        };

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
