//! impl `QuantMatmulOps<CudaRuntime>` for CudaClient

use crate::error::{Error, Result};
use crate::quant::traits::QuantMatmulOps;
use crate::quant::{QuantFormat, QuantScheme, QuantTensor};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;
use tcf_core::NativeEncoding;

use super::super::int4_gemm as int4_dispatch;
use super::super::kernels::{
    self, GEMV_Q2_K_MODULE, GEMV_Q3_K_MODULE, GEMV_Q5_K_MODULE, QUANT_GEMV_MODULE,
};
use super::super::tcf::{self as tcf_dispatch, MatmulShape};
use super::batched_gemv::quant_matmul_batch_impl;
use super::fallback::{quant_matmul_via_dequant, quant_swiglu_via_dequant};
use super::format_dispatch::{dispatch_gemv, dispatch_matmul, gemv_max_m};
use super::helpers::{quantize_activation_q8_1, validate_input_cuda};
use super::mmq_feat_major;

/// Smallest token count that takes the feature-major MMQ kernel rather than
/// the f32 GEMV.
///
/// The MMQ kernel is flat in `m` across a token tile, while the f32 GEMV
/// re-reads the weight per token, so the GEMV only wins where the tile would
/// be nearly empty.
///
/// Measured: the MMQ kernel wins at every token count from this constant
/// upward, on both benchmarked projection shapes. `m = 1` stays on the GEMV.
///
/// Re-measure: `cargo bench --features cuda --bench quant_throughput --
/// --backend cuda --filter Q8`, comparing the `tcf` rows against the `gguf`
/// rows at each `m`.
const TCF_FEAT_MAJOR_MIN_M: usize = 2;

impl QuantMatmulOps<CudaRuntime> for CudaClient {
    fn int4_gemm(
        &self,
        input: &Tensor<CudaRuntime>,
        qweight: &Tensor<CudaRuntime>,
        scales: &Tensor<CudaRuntime>,
        zeros: &Tensor<CudaRuntime>,
        group_size: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        let (m, k) = validate_input_cuda(input)?;
        let n = qweight.shape()[1] * 8;
        let act_contig = input.contiguous()?;

        let mut out_shape = input.shape()[..input.shape().len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, input.device())?;
        int4_dispatch::launch_int4_gemm(
            self,
            &act_contig,
            qweight,
            scales,
            zeros,
            &output,
            m as u32,
            k as u32,
            n as u32,
            group_size as u32,
        )?;
        Ok(output)
    }

    fn int4_gemm_gptq(
        &self,
        input: &Tensor<CudaRuntime>,
        qweight: &Tensor<CudaRuntime>,
        qzeros: &Tensor<CudaRuntime>,
        scales: &Tensor<CudaRuntime>,
        g_idx: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let (m, k) = validate_input_cuda(input)?;
        let n = qweight.shape()[1];
        let act_contig = input.contiguous()?;

        let mut out_shape = input.shape()[..input.shape().len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, input.device())?;
        int4_dispatch::launch_int4_gemm_gptq(
            self,
            &act_contig,
            qweight,
            qzeros,
            scales,
            g_idx,
            &output,
            m as u32,
            k as u32,
            n as u32,
        )?;
        Ok(output)
    }

    fn marlin_gemm(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        scales: &Tensor<CudaRuntime>,
        zeros: &Tensor<CudaRuntime>,
        group_size: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        let (m, k) = validate_input_cuda(input)?;
        let n = weight.shape()[1];
        let act_contig = input.contiguous()?;

        let mut out_shape = input.shape()[..input.shape().len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, input.device())?;
        int4_dispatch::launch_marlin_gemm(
            self,
            &act_contig,
            weight,
            scales,
            zeros,
            &output,
            m as u32,
            k as u32,
            n as u32,
            group_size as u32,
        )?;
        Ok(output)
    }

    fn quant_matmul(
        &self,
        activation: &Tensor<CudaRuntime>,
        weight: &QuantTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        if activation.dtype() != DType::F32 {
            return Err(Error::QuantError {
                reason: format!(
                    "quant_matmul activation must be F32, got {:?}",
                    activation.dtype()
                ),
            });
        }

        let w_shape = weight.shape();
        if w_shape.len() != 2 {
            return Err(Error::QuantError {
                reason: format!("quant_matmul weight must be 2D [N, K], got {:?}", w_shape),
            });
        }
        let n = w_shape[0];
        let k = w_shape[1];

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

        let m = a_shape.iter().product::<usize>() / k;
        let act_contig = activation.contiguous()?;

        let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, activation.device())?;
        let output_ptr = output.ptr();

        // TCF weights take their own kernels: a GGUF kernel finds a block's
        // codes and its scale adjacent, while TCF spreads them over
        // whole-tensor planes.
        //
        // TCF has two crossovers, not one, and they are NOT the `M <= 64` the
        // GGUF path below uses:
        // - `TCF_FEAT_MAJOR_MIN_M` (below) gates `Q8S32T64` onto the MMQ
        //   kernel ahead of everything else.
        // - The `m <= 4` check further down chooses `launch_gemv` vs
        //   `launch_gemm` for every encoding that isn't on the MMQ path —
        //   `Q8S32T64` falls back to it too when the MMQ dispatch declines.
        //
        // The mechanism behind the GEMV/GEMM split, which is what carries
        // across devices: GEMV cost grows linearly in M because it re-reads
        // the weights once per row, while the register-blocked GEMM computes
        // a 4x4 output patch per thread and stays nearly flat in M, so the
        // two cross at a small M on every encoding.
        //
        // Re-measure each crossover whenever its kernel changes — a speedup
        // on one side moves it, and both values are measured constants, not
        // derived ones.
        if let QuantScheme::Tcf(encoding) = weight.scheme() {
            let at = MatmulShape { m, k, n };
            let device_index = activation.device().id();
            if m >= TCF_FEAT_MAJOR_MIN_M {
                // `Q8S32T64` stages into the Q8_0 weight row, so it takes the
                // feature-major tensor-core family instead of the f32 FMA tile
                // `launch_gemm` runs. Every other encoding keeps that tile:
                // none of them has a `FeatMajorFormat` yet. `Ok(None)` means no
                // compiled variant fits the device, and `launch_gemm` still
                // serves the shape.
                let feat_major = encoding.native() == NativeEncoding::Q8S32T64
                    && k.is_multiple_of(mmq_feat_major::TCF_Q8S32T64.k_multiple as usize)
                    && numr::runtime::cuda::CudaDevice::new(device_index)
                        .profile()
                        .caps
                        .int8_mma_m16n8k32;
                if feat_major
                    && mmq_feat_major::dispatch(
                        &mmq_feat_major::TCF_Q8S32T64,
                        self,
                        &act_contig,
                        weight,
                        output_ptr,
                        m,
                        k,
                        n,
                    )?
                    .is_some()
                {
                    return Ok(output);
                }
            }
            let launch = if m <= 4 {
                tcf_dispatch::launch_gemv
            } else {
                tcf_dispatch::launch_gemm
            };
            launch(
                self,
                device_index,
                act_contig.ptr(),
                weight.storage().ptr(),
                output_ptr,
                encoding,
                at,
            )?;
            return Ok(output);
        }

        // The GEMV/GEMM crossover is measured per format: a faster GEMM moves
        // it down. `gemv_max_m` holds the current value for each format.
        let format = weight.format()?;
        let device_index = activation.device().id();
        if m <= gemv_max_m(format, device_index) {
            match dispatch_gemv(self, &act_contig, weight, output_ptr, m, k, n)? {
                Some(()) => {}
                None => return quant_matmul_via_dequant(self, activation, weight),
            }
        } else {
            match dispatch_matmul(self, &act_contig, weight, output_ptr, m, k, n)? {
                Some(()) => {}
                None => return quant_matmul_via_dequant(self, activation, weight),
            }
        }

        Ok(output)
    }

    fn quant_matmul_batch(
        &self,
        activation: &Tensor<CudaRuntime>,
        weights: &[&QuantTensor<CudaRuntime>],
    ) -> Result<Vec<Tensor<CudaRuntime>>> {
        quant_matmul_batch_impl(self, activation, weights)
    }

    fn quant_swiglu(
        &self,
        activation: &Tensor<CudaRuntime>,
        gate_weight: &QuantTensor<CudaRuntime>,
        up_weight: &QuantTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let (m, k) = validate_input_cuda(activation)?;
        let n = gate_weight.shape()[0];
        let device_index = activation.device().id();

        if up_weight.shape()[0] != n || up_weight.shape()[1] != k {
            return Err(Error::QuantError {
                reason: format!(
                    "gate_weight shape {:?} vs up_weight shape {:?}",
                    gate_weight.shape(),
                    up_weight.shape()
                ),
            });
        }
        // The fused SwiGLU kernels read a GGUF block layout. A TCF weight goes
        // through two fused matmuls and numr's `silu_mul` instead, which is
        // what the CPU backend does for every codec.
        if !gate_weight.scheme().is_row_blocked() || !up_weight.scheme().is_row_blocked() {
            let gate = self.quant_matmul(activation, gate_weight)?;
            let up = self.quant_matmul(activation, up_weight)?;
            use numr::ops::ActivationOps;
            return self.silu_mul(&gate, &up).map_err(Error::Numr);
        }
        let gate_format = gate_weight.format()?;
        let up_format = up_weight.format()?;
        if gate_format != up_format {
            return Err(Error::QuantError {
                reason: format!("gate format {gate_format:?} != up format {up_format:?}"),
            });
        }

        let act_contig = activation.contiguous()?;
        let a_shape = activation.shape();
        let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, activation.device())?;
        let output_ptr = output.ptr();
        let m_u32 = m as u32;
        let k_u32 = k as u32;
        let n_u32 = n as u32;

        // Use fused kernel for GEMV path (decode + short prefill)
        let use_fused = m <= 64
            && matches!(
                gate_format,
                QuantFormat::Q4K
                    | QuantFormat::Q6K
                    | QuantFormat::Q8_0
                    | QuantFormat::Q5K
                    | QuantFormat::Q3K
                    | QuantFormat::Q2K
            )
            && k % 32 == 0;

        if use_fused {
            let q8_buf = quantize_activation_q8_1(self, &act_contig, m, k)?;
            let q8_ptr = q8_buf.ptr();
            let gate_ptr = gate_weight.storage().ptr();
            let up_ptr = up_weight.storage().ptr();

            let (kernel_name, module_name) = match gate_format {
                QuantFormat::Q4K => ("fused_swiglu_q4k_q8_1_mwr", QUANT_GEMV_MODULE),
                QuantFormat::Q6K => ("fused_swiglu_q6k_q8_1_mwr", QUANT_GEMV_MODULE),
                QuantFormat::Q8_0 => ("fused_swiglu_q8_0_q8_1_mwr", QUANT_GEMV_MODULE),
                QuantFormat::Q5K => ("fused_swiglu_q5k_q8_1_mwr", GEMV_Q5_K_MODULE),
                QuantFormat::Q3K => ("fused_swiglu_q3k_q8_1_mwr", GEMV_Q3_K_MODULE),
                QuantFormat::Q2K => ("fused_swiglu_q2k_q8_1_mwr", GEMV_Q2_K_MODULE),
                _ => unreachable!(),
            };

            let cfg = LaunchConfig {
                grid_dim: (n_u32, m_u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };

            let module = kernels::get_or_load_module(self.context(), device_index, module_name)?;
            let func = kernels::get_kernel_function(&module, kernel_name)?;

            unsafe {
                let mut builder = self.stream().launch_builder(&func);
                builder.arg(&q8_ptr);
                builder.arg(&gate_ptr);
                builder.arg(&up_ptr);
                builder.arg(&output_ptr);
                builder.arg(&m_u32);
                builder.arg(&k_u32);
                builder.arg(&n_u32);
                builder.launch(cfg).map_err(|e| Error::QuantError {
                    reason: format!("CUDA {} launch failed: {:?}", kernel_name, e),
                })?;
            }

            Ok(output)
        } else if m <= 64 {
            // Generic fused SwiGLU: gate+up matmul + silu in one kernel
            quant_swiglu_via_dequant(self, &act_contig, gate_weight, up_weight, &output, m, k, n)
                .map(|_| output)
        } else {
            // Large batch: separate matmuls + fused silu_mul
            let gate = self.quant_matmul(activation, gate_weight)?;
            let up = self.quant_matmul(activation, up_weight)?;
            use numr::ops::ActivationOps;
            self.silu_mul(&gate, &up).map_err(Error::Numr)
        }
    }
}
