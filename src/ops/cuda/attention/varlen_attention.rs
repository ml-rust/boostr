//! Variable-length (ragged) attention CUDA launchers
//!
//! Packed sequences with cu_seqlens indexing. Supports F32 and F16,
//! head dims 64, 128, and 256, with GQA (num_kv_heads <= num_heads).
//! Both forward and backward are implemented for all supported head dims.

use crate::error::{Error, Result};
use crate::ops::traits::VarLenAttentionOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash::set_smem_attribute;
use crate::ops::cuda::kernels::{
    self, VARLEN_ATTENTION_BWD_FP16_MODULE, VARLEN_ATTENTION_BWD_MODULE, VARLEN_ATTENTION_MODULE,
};

/// Return `(BLOCK_M, BLOCK_N)` for the given head_dim and dtype.
///
/// Constraints: smem = (BLOCK_M + 2*BLOCK_N) * head_dim * dtype_size ≤ 48 KB,
/// and block_dim = BLOCK_M (one thread per Q row).
///
/// | head_dim | dtype | BLOCK_M | BLOCK_N | smem    |
/// |----------|-------|---------|---------|---------|
/// | 64/128   | any   | 128     | 64      | ≤ 48 KB |
/// | 256      | F32   | 16      | 16      | 48 KB   |
/// | 256      | F16   | 32      | 32      | 48 KB   |
#[inline]
fn block_config(head_dim: usize, dtype: DType) -> (usize, usize) {
    match head_dim {
        256 => match dtype {
            DType::F16 => (32, 32),
            _ => (16, 16), // F32 and any future type default to 16/16
        },
        _ => (128, 64), // 64 and 128 use the proven large-tile config
    }
}

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
        num_kv_heads: usize,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        if head_dim != 64 && head_dim != 128 && head_dim != 256 {
            return Err(Error::KernelError {
                reason: format!(
                    "varlen attention: unsupported head_dim {head_dim}, only 64/128/256"
                ),
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

        // Block sizes are chosen so that shared memory fits within 48 KB
        // (after set_smem_attribute opts in to larger dynamic smem).
        //   head_dim 64/128: BLOCK_M=128, BLOCK_N=64  (proven config)
        //   head_dim 256 fp32: BLOCK_M=16, BLOCK_N=16
        //     smem = (16 + 2*16) * 256 * 4 = 49 152 B = 48 KB
        //   head_dim 256 fp16: BLOCK_M=32, BLOCK_N=32
        //     smem = (32 + 2*32) * 256 * 2 = 49 152 B = 48 KB
        // block_dim == BLOCK_M (one thread per Q row) — invariant kept.
        let (block_m, block_n) = block_config(head_dim, dtype);

        // Grid must cover all batches × heads × Q blocks per batch
        let num_q_blocks_per_batch = max_seqlen_q.div_ceil(block_m);
        let num_q_blocks = num_q_blocks_per_batch * batch_size;

        let dtype_size = dtype.size_in_bytes();
        // smem layout: Q tile + K tile + V tile, each with +1 padding on the column
        // stride to eliminate bank conflicts (HEAD_STRIDE = HEAD_DIM + 1).
        let head_stride = head_dim + 1;
        let smem_size =
            (block_m * head_stride + block_n * head_stride + block_n * head_stride) * dtype_size;
        set_smem_attribute(&func, smem_size)?;

        let cfg = LaunchConfig {
            grid_dim: ((num_q_blocks * num_heads) as u32, 1, 1),
            block_dim: (block_m as u32, 1, 1),
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
        let nkv_i32 = num_kv_heads as i32;
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
            builder.arg(&nkv_i32);
            builder.arg(&msq_i32);
            builder.arg(&msk_i32);
            builder.arg(&scale);
            builder.arg(&causal_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("varlen attention fwd launch failed: {e:?}"),
            })?;
        }

        // No sync needed: same-stream ordering guarantees correctness.

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
        num_kv_heads: usize,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        if head_dim != 64 && head_dim != 128 && head_dim != 256 {
            return Err(Error::KernelError {
                reason: format!(
                    "varlen attention bwd: unsupported head_dim {head_dim}, only 64/128/256"
                ),
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

        // FP16 backward kernels live in their own compiled module (split out to
        // keep each .cu within the file-size budget); FP32 stays in the base module.
        let bwd_module = match dtype {
            DType::F16 => VARLEN_ATTENTION_BWD_FP16_MODULE,
            _ => VARLEN_ATTENTION_BWD_MODULE,
        };
        let module = kernels::get_or_load_module(self.context(), device_index, bwd_module)?;
        let func = kernels::get_kernel_function(&module, &kernel_name)?;

        let total_tokens_q = q.shape()[0];
        let total_tokens_k = k.shape()[0];

        // dq: same head layout as Q (num_heads)
        // dk/dv: kv head layout (num_kv_heads) — GQA: fewer heads than Q
        let dq =
            Tensor::<CudaRuntime>::zeros(&[total_tokens_q, num_heads, head_dim], dtype, device);
        let dk =
            Tensor::<CudaRuntime>::zeros(&[total_tokens_k, num_kv_heads, head_dim], dtype, device);
        let dv =
            Tensor::<CudaRuntime>::zeros(&[total_tokens_k, num_kv_heads, head_dim], dtype, device);

        // Use same block sizes as fwd (matches the kernel template instantiations)
        let (block_m, block_n) = block_config(head_dim, dtype);
        let num_q_blocks_per_batch = max_seqlen_q.div_ceil(block_m);
        let num_q_blocks = num_q_blocks_per_batch * batch_size;

        // Shared memory layout (with +1 HEAD_STRIDE padding, same as the bwd kernel):
        //   Q tile   : BLOCK_M * (HEAD_DIM+1) elements
        //   K tile   : BLOCK_N * (HEAD_DIM+1) elements
        //   V tile   : BLOCK_N * (HEAD_DIM+1) elements
        //   dO tile  : BLOCK_M * (HEAD_DIM+1) elements
        // Total: (2*BLOCK_M + 2*BLOCK_N) * HEAD_STRIDE * dtype_size bytes
        let dtype_size = dtype.size_in_bytes();
        let head_stride = head_dim + 1;
        let smem_size = (2 * block_m + 2 * block_n) * head_stride * dtype_size;
        set_smem_attribute(&func, smem_size)?;

        let cfg = LaunchConfig {
            grid_dim: ((num_q_blocks * num_heads) as u32, 1, 1),
            block_dim: (block_m as u32, 1, 1),
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
        let nkv_i32 = num_kv_heads as i32;
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
            builder.arg(&nkv_i32);
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
