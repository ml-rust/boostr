//! Variable-length (ragged) attention CUDA launchers
//!
//! Packed sequences with cu_seqlens indexing. Supports F32 and F16,
//! head dims 64 and 128.

use crate::error::{Error, Result};
use crate::ops::traits::VarLenAttentionOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash::set_smem_attribute;
use crate::ops::cuda::kernels::{self, VARLEN_ATTENTION_BWD_MODULE, VARLEN_ATTENTION_MODULE};

impl VarLenAttentionOps<CudaRuntime> for CudaClient {
    fn varlen_attention_fwd(
        &self,
        q: &Tensor<CudaRuntime>,
        k: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        cu_seqlens_q: &Tensor<CudaRuntime>,
        cu_seqlens_k: &Tensor<CudaRuntime>,
        batch_size: usize,
        num_heads: usize,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        if head_dim != 64 && head_dim != 128 {
            return Err(Error::KernelError {
                reason: format!("varlen attention: unsupported head_dim {head_dim}, only 64/128"),
            });
        }

        let dtype = q.dtype();
        let dtype_suffix = match dtype {
            DType::F32 => "fp32",
            DType::F16 => "fp16",
            _ => {
                return Err(Error::KernelError {
                    reason: format!("varlen attention: unsupported dtype {dtype:?}, only F32/F16"),
                });
            }
        };

        let kernel_name = format!("varlen_flash_attention_fwd_{head_dim}_{dtype_suffix}");
        let device = q.device();
        let device_index = device.id();

        let module =
            kernels::get_or_load_module(self.context(), device_index, VARLEN_ATTENTION_MODULE)?;
        let func = kernels::get_kernel_function(&module, &kernel_name)?;

        // Total tokens from Q shape: [total_tokens_q, num_heads, head_dim]
        let total_tokens_q = q.shape()[0];

        let output =
            Tensor::<CudaRuntime>::empty(&[total_tokens_q, num_heads, head_dim], dtype, device);
        let lse = Tensor::<CudaRuntime>::empty(&[total_tokens_q, num_heads], DType::F32, device);

        const BLOCK_M: usize = 128;
        const BLOCK_N: usize = 64;
        // Grid must cover all batches × heads × Q blocks per batch
        let num_q_blocks_per_batch = (max_seqlen_q + BLOCK_M - 1) / BLOCK_M;
        let num_q_blocks = num_q_blocks_per_batch * batch_size;

        let dtype_size = dtype.size_in_bytes();
        let smem_size = (BLOCK_M * head_dim + BLOCK_N * head_dim + BLOCK_N * head_dim) * dtype_size;
        set_smem_attribute(&func, smem_size)?;

        let cfg = LaunchConfig {
            grid_dim: ((num_q_blocks * num_heads) as u32, 1, 1),
            block_dim: (BLOCK_M as u32, 1, 1),
            shared_mem_bytes: smem_size as u32,
        };

        let q_ptr = q.ptr();
        let k_ptr = k.ptr();
        let v_ptr = v.ptr();
        let cu_q_ptr = cu_seqlens_q.ptr();
        let cu_k_ptr = cu_seqlens_k.ptr();
        let o_ptr = output.ptr();
        let l_ptr = lse.ptr();
        let scale = (head_dim as f32).sqrt().recip();
        let batch_i32 = batch_size as i32;
        let nh_i32 = num_heads as i32;
        let msq_i32 = max_seqlen_q as i32;
        let msk_i32 = max_seqlen_k as i32;
        let causal_i32 = if causal { 1i32 } else { 0i32 };

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&q_ptr);
            builder.arg(&k_ptr);
            builder.arg(&v_ptr);
            builder.arg(&cu_q_ptr);
            builder.arg(&cu_k_ptr);
            builder.arg(&o_ptr);
            builder.arg(&l_ptr);
            builder.arg(&batch_i32);
            builder.arg(&nh_i32);
            builder.arg(&msq_i32);
            builder.arg(&msk_i32);
            builder.arg(&scale);
            builder.arg(&causal_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("varlen attention fwd launch failed: {e:?}"),
            })?;
        }

        self.stream()
            .synchronize()
            .map_err(|e| Error::KernelError {
                reason: format!("varlen attention fwd sync failed: {e:?}"),
            })?;

        Ok((output, lse))
    }

    fn varlen_attention_bwd(
        &self,
        dout: &Tensor<CudaRuntime>,
        q: &Tensor<CudaRuntime>,
        k: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        output: &Tensor<CudaRuntime>,
        lse: &Tensor<CudaRuntime>,
        cu_seqlens_q: &Tensor<CudaRuntime>,
        cu_seqlens_k: &Tensor<CudaRuntime>,
        batch_size: usize,
        num_heads: usize,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        if head_dim != 64 && head_dim != 128 {
            return Err(Error::KernelError {
                reason: format!("varlen attention bwd: unsupported head_dim {head_dim}"),
            });
        }

        let dtype = q.dtype();
        let dtype_suffix = match dtype {
            DType::F32 => "fp32",
            DType::F16 => "fp16",
            _ => {
                return Err(Error::KernelError {
                    reason: format!("varlen attention bwd: unsupported dtype {dtype:?}"),
                });
            }
        };

        let kernel_name = format!("varlen_flash_attention_bwd_{head_dim}_{dtype_suffix}");
        let device = q.device();
        let device_index = device.id();

        let module =
            kernels::get_or_load_module(self.context(), device_index, VARLEN_ATTENTION_BWD_MODULE)?;
        let func = kernels::get_kernel_function(&module, &kernel_name)?;

        let total_tokens_q = q.shape()[0];
        let total_tokens_k = k.shape()[0];

        let dq =
            Tensor::<CudaRuntime>::zeros(&[total_tokens_q, num_heads, head_dim], dtype, device);
        let dk =
            Tensor::<CudaRuntime>::zeros(&[total_tokens_k, num_heads, head_dim], dtype, device);
        let dv =
            Tensor::<CudaRuntime>::zeros(&[total_tokens_k, num_heads, head_dim], dtype, device);

        const BLOCK_M: usize = 128;
        const BLOCK_N: usize = 64;
        let num_q_blocks_per_batch = (max_seqlen_q + BLOCK_M - 1) / BLOCK_M;
        let num_q_blocks = num_q_blocks_per_batch * batch_size;

        // Shared memory: Q + K + V + dO (BLOCK_M*HD + 2*BLOCK_N*HD + BLOCK_M*HD) + row_sum
        let dtype_size = dtype.size_in_bytes();
        let smem_size = (2 * BLOCK_M * head_dim + 2 * BLOCK_N * head_dim + BLOCK_M) * dtype_size;
        set_smem_attribute(&func, smem_size)?;

        let cfg = LaunchConfig {
            grid_dim: ((num_q_blocks * num_heads) as u32, 1, 1),
            block_dim: (BLOCK_M as u32, 1, 1),
            shared_mem_bytes: smem_size as u32,
        };

        let q_ptr = q.ptr();
        let k_ptr = k.ptr();
        let v_ptr = v.ptr();
        let o_ptr = output.ptr();
        let l_ptr = lse.ptr();
        let do_ptr = dout.ptr();
        let cu_q_ptr = cu_seqlens_q.ptr();
        let cu_k_ptr = cu_seqlens_k.ptr();
        let dq_ptr = dq.ptr();
        let dk_ptr = dk.ptr();
        let dv_ptr = dv.ptr();
        let scale = (head_dim as f32).sqrt().recip();
        let batch_i32 = batch_size as i32;
        let nh_i32 = num_heads as i32;
        let msq_i32 = max_seqlen_q as i32;
        let msk_i32 = max_seqlen_k as i32;
        let causal_i32 = if causal { 1i32 } else { 0i32 };

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&q_ptr);
            builder.arg(&k_ptr);
            builder.arg(&v_ptr);
            builder.arg(&o_ptr);
            builder.arg(&l_ptr);
            builder.arg(&do_ptr);
            builder.arg(&cu_q_ptr);
            builder.arg(&cu_k_ptr);
            builder.arg(&dq_ptr);
            builder.arg(&dk_ptr);
            builder.arg(&dv_ptr);
            builder.arg(&batch_i32);
            builder.arg(&nh_i32);
            builder.arg(&msq_i32);
            builder.arg(&msk_i32);
            builder.arg(&scale);
            builder.arg(&causal_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("varlen attention bwd launch failed: {e:?}"),
            })?;
        }

        // Sync stream: BWD uses atomicAdd so must complete before results are read
        self.stream()
            .synchronize()
            .map_err(|e| Error::KernelError {
                reason: format!("varlen attention bwd sync failed: {e:?}"),
            })?;

        Ok((dq, dk, dv))
    }
}
