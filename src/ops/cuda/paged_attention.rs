//! CUDA Paged Attention — vLLM-style non-contiguous KV cache
//!
//! Fused kernel — PRIMITIVE op. Block table indirection for KV blocks.
//! Supports: F32, F16, BF16, FP8E4M3, FP8E5M2. Head dimensions: 64, 128.

use crate::error::{Error, Result};
use crate::ops::traits::PagedAttentionOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::kernels::{self, PAGED_ATTENTION_BWD_MODULE, PAGED_ATTENTION_MODULE};

/// Get block sizes for paged attention forward.
/// Uses smaller blocks that fit in 48KB shared memory.
fn fwd_block_config(head_dim: usize, dtype: DType) -> Result<(usize, usize)> {
    match (dtype, head_dim) {
        // FP32: 4 bytes per element
        (DType::F32, 64) => Ok((64, 32)),  // (64+32+32)*64*4 = 32KB
        (DType::F32, 128) => Ok((32, 32)), // (32+32+32)*128*4 = 48KB
        // FP16/BF16: 2 bytes per element
        (DType::F16 | DType::BF16, 64) => Ok((64, 32)), // (64+32+32)*64*2 = 16KB
        (DType::F16 | DType::BF16, 128) => Ok((32, 32)), // (32+32+32)*128*2 = 24KB
        // FP8: uses FP32 smem for compute
        (DType::FP8E4M3 | DType::FP8E5M2, 64) => Ok((64, 32)), // (64+64)*64*4 = 32KB
        (DType::FP8E4M3 | DType::FP8E5M2, 128) => Ok((32, 32)), // (32+64)*128*4 = 49KB
        _ => Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!(
                "unsupported head_dim={} for paged attention. Supported: 64, 128",
                head_dim
            ),
        }),
    }
}

/// Get block sizes for paged attention backward.
/// Backward needs more smem: Q + K + V + dO + O = (3*BLOCK_M + 2*BLOCK_N) * HD * dtype_size
fn bwd_block_config(head_dim: usize, dtype: DType) -> Result<(usize, usize)> {
    match (dtype, head_dim) {
        (DType::F32, 64) => Ok((32, 32)),  // (96+64)*64*4 = 40KB
        (DType::F32, 128) => Ok((16, 16)), // (48+32)*128*4 = 40KB
        (DType::F16 | DType::BF16, 64) => Ok((64, 32)), // (192+64)*64*2 = 32KB
        (DType::F16 | DType::BF16, 128) => Ok((32, 32)), // (96+64)*128*2 = 40KB
        _ => Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!(
                "unsupported head_dim={} for paged attention backward",
                head_dim
            ),
        }),
    }
}

impl PagedAttentionOps<CudaRuntime> for CudaClient {
    fn paged_attention_fwd(
        &self,
        q: &Tensor<CudaRuntime>,
        k_blocks: &Tensor<CudaRuntime>,
        v_blocks: &Tensor<CudaRuntime>,
        block_table: &Tensor<CudaRuntime>,
        num_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        block_size: usize,
        causal: bool,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let q_shape = q.shape();
        if q_shape.len() != 4 {
            return Err(Error::InvalidArgument {
                arg: "q",
                reason: format!("expected 4D [B, H, S, D], got {}D", q_shape.len()),
            });
        }
        let batch_size = q_shape[0];
        let dtype = q.dtype();

        let dtype_suffix = match dtype {
            DType::F32 => "fp32",
            DType::F16 => "fp16",
            DType::BF16 => "bf16",
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "dtype",
                    reason: format!(
                        "unsupported dtype {:?}. Use paged_attention_fwd_fp8 for FP8.",
                        dtype
                    ),
                });
            }
        };

        let (block_m, block_n) = fwd_block_config(head_dim, dtype)?;
        let kernel_name = format!(
            "paged_flash_attention_fwd_{}_{}_small",
            head_dim, dtype_suffix
        );

        let device = q.device();
        let output = Tensor::<CudaRuntime>::empty(
            &[batch_size, num_heads, seq_len_q, head_dim],
            dtype,
            device,
        );
        let lse =
            Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q], DType::F32, device);

        let dtype_size = dtype.size_in_bytes();
        let smem_size = (block_m * head_dim + block_n * head_dim + block_n * head_dim) * dtype_size;

        let max_num_blocks = block_table.shape()[1];
        let device_index = device.id();
        let module =
            kernels::get_or_load_module(self.context(), device_index, PAGED_ATTENTION_MODULE)?;
        let func = kernels::get_kernel_function(&module, &kernel_name)?;

        let cfg = LaunchConfig {
            grid_dim: (
                (batch_size * num_heads) as u32,
                ((seq_len_q + block_m - 1) / block_m) as u32,
                1,
            ),
            block_dim: (block_m as u32, 1, 1),
            shared_mem_bytes: smem_size as u32,
        };

        let q_ptr = q.ptr();
        let kb_ptr = k_blocks.ptr();
        let vb_ptr = v_blocks.ptr();
        let bt_ptr = block_table.ptr();
        let o_ptr = output.ptr();
        let l_ptr = lse.ptr();
        let scale = (head_dim as f32).sqrt().recip();
        let batch_i32 = batch_size as i32;
        let nh_i32 = num_heads as i32;
        let sq_i32 = seq_len_q as i32;
        let sk_i32 = seq_len_k as i32;
        let mnb_i32 = max_num_blocks as i32;
        let bs_i32 = block_size as i32;
        let causal_i32 = if causal { 1i32 } else { 0i32 };

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&q_ptr);
            builder.arg(&kb_ptr);
            builder.arg(&vb_ptr);
            builder.arg(&bt_ptr);
            builder.arg(&o_ptr);
            builder.arg(&l_ptr);
            builder.arg(&batch_i32);
            builder.arg(&nh_i32);
            builder.arg(&sq_i32);
            builder.arg(&sk_i32);
            builder.arg(&mnb_i32);
            builder.arg(&bs_i32);
            builder.arg(&scale);
            builder.arg(&causal_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("Paged attention fwd kernel launch failed: {:?}", e),
            })?;
        }

        Ok((output, lse))
    }

    fn paged_attention_fwd_fp8(
        &self,
        q: &Tensor<CudaRuntime>,
        k_blocks: &Tensor<CudaRuntime>,
        v_blocks: &Tensor<CudaRuntime>,
        block_table: &Tensor<CudaRuntime>,
        num_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        block_size: usize,
        causal: bool,
        q_scale: f32,
        k_scale: f32,
        v_scale: f32,
        o_scale: f32,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let q_shape = q.shape();
        if q_shape.len() != 4 {
            return Err(Error::InvalidArgument {
                arg: "q",
                reason: format!("expected 4D [B, H, S, D], got {}D", q_shape.len()),
            });
        }
        let batch_size = q_shape[0];
        let dtype = q.dtype();

        let dtype_suffix = match dtype {
            DType::FP8E4M3 => "fp8_e4m3",
            DType::FP8E5M2 => "fp8_e5m2",
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "dtype",
                    reason: format!(
                        "paged_attention_fwd_fp8 requires FP8 dtype, got {:?}",
                        dtype
                    ),
                });
            }
        };

        let (block_m, _block_n) = fwd_block_config(head_dim, dtype)?;
        let kernel_name = format!(
            "paged_flash_attention_fwd_{}_{}_small",
            head_dim, dtype_suffix
        );

        let device = q.device();
        let output = Tensor::<CudaRuntime>::empty(
            &[batch_size, num_heads, seq_len_q, head_dim],
            dtype,
            device,
        );
        let lse =
            Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q], DType::F32, device);

        // FP8 uses FP32 smem: (BLOCK_M + 2*BLOCK_N) * HEAD_DIM * 4
        let (bm, bn) = fwd_block_config(head_dim, dtype)?;
        let smem_size = (bm * head_dim + 2 * bn * head_dim) * 4;

        let max_num_blocks = block_table.shape()[1];
        let device_index = device.id();
        let module =
            kernels::get_or_load_module(self.context(), device_index, PAGED_ATTENTION_MODULE)?;
        let func = kernels::get_kernel_function(&module, &kernel_name)?;

        let cfg = LaunchConfig {
            grid_dim: (
                (batch_size * num_heads) as u32,
                ((seq_len_q + block_m - 1) / block_m) as u32,
                1,
            ),
            block_dim: (block_m as u32, 1, 1),
            shared_mem_bytes: smem_size as u32,
        };

        let q_ptr = q.ptr();
        let kb_ptr = k_blocks.ptr();
        let vb_ptr = v_blocks.ptr();
        let bt_ptr = block_table.ptr();
        let o_ptr = output.ptr();
        let l_ptr = lse.ptr();
        let attn_scale = (head_dim as f32).sqrt().recip();
        let batch_i32 = batch_size as i32;
        let nh_i32 = num_heads as i32;
        let sq_i32 = seq_len_q as i32;
        let sk_i32 = seq_len_k as i32;
        let mnb_i32 = max_num_blocks as i32;
        let bs_i32 = block_size as i32;
        let causal_i32 = if causal { 1i32 } else { 0i32 };

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&q_ptr);
            builder.arg(&kb_ptr);
            builder.arg(&vb_ptr);
            builder.arg(&bt_ptr);
            builder.arg(&o_ptr);
            builder.arg(&l_ptr);
            builder.arg(&batch_i32);
            builder.arg(&nh_i32);
            builder.arg(&sq_i32);
            builder.arg(&sk_i32);
            builder.arg(&mnb_i32);
            builder.arg(&bs_i32);
            builder.arg(&attn_scale);
            builder.arg(&q_scale);
            builder.arg(&k_scale);
            builder.arg(&v_scale);
            builder.arg(&o_scale);
            builder.arg(&causal_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("Paged attention FP8 fwd kernel launch failed: {:?}", e),
            })?;
        }

        Ok((output, lse))
    }

    fn paged_attention_bwd(
        &self,
        dout: &Tensor<CudaRuntime>,
        q: &Tensor<CudaRuntime>,
        k_blocks: &Tensor<CudaRuntime>,
        v_blocks: &Tensor<CudaRuntime>,
        output: &Tensor<CudaRuntime>,
        lse: &Tensor<CudaRuntime>,
        block_table: &Tensor<CudaRuntime>,
        num_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        block_size: usize,
        causal: bool,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let q_shape = q.shape();
        if q_shape.len() != 4 {
            return Err(Error::InvalidArgument {
                arg: "q",
                reason: format!("expected 4D [B, H, S, D], got {}D", q_shape.len()),
            });
        }
        let batch_size = q_shape[0];
        let dtype = q.dtype();

        let dtype_suffix = match dtype {
            DType::F32 => "fp32",
            DType::F16 => "fp16",
            DType::BF16 => "bf16",
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "dtype",
                    reason: format!("unsupported dtype {:?} for paged attention backward", dtype),
                });
            }
        };

        let (block_m, block_n) = bwd_block_config(head_dim, dtype)?;
        let kernel_name = format!(
            "paged_flash_attention_bwd_{}_{}_small",
            head_dim, dtype_suffix
        );

        let device = q.device();
        let max_num_blocks = block_table.shape()[1];

        // dQ: contiguous [B, H, S_q, D]
        let dq = Tensor::<CudaRuntime>::empty(
            &[batch_size, num_heads, seq_len_q, head_dim],
            dtype,
            device,
        );
        // dK, dV: same shape as k_blocks/v_blocks — zeroed for atomicAdd
        let dk_blocks = Tensor::<CudaRuntime>::zeros(k_blocks.shape(), dtype, device);
        let dv_blocks = Tensor::<CudaRuntime>::zeros(v_blocks.shape(), dtype, device);

        let dtype_size = dtype.size_in_bytes();
        let smem_size = (3 * block_m * head_dim + 2 * block_n * head_dim) * dtype_size;

        let device_index = device.id();
        let module =
            kernels::get_or_load_module(self.context(), device_index, PAGED_ATTENTION_BWD_MODULE)?;
        let func = kernels::get_kernel_function(&module, &kernel_name)?;

        let cfg = LaunchConfig {
            grid_dim: (
                (batch_size * num_heads) as u32,
                ((seq_len_q + block_m - 1) / block_m) as u32,
                1,
            ),
            block_dim: (block_m as u32, 1, 1),
            shared_mem_bytes: smem_size as u32,
        };

        let q_ptr = q.ptr();
        let kb_ptr = k_blocks.ptr();
        let vb_ptr = v_blocks.ptr();
        let o_ptr = output.ptr();
        let do_ptr = dout.ptr();
        let l_ptr = lse.ptr();
        let bt_ptr = block_table.ptr();
        let dq_ptr = dq.ptr();
        let dkb_ptr = dk_blocks.ptr();
        let dvb_ptr = dv_blocks.ptr();
        let scale = (head_dim as f32).sqrt().recip();
        let batch_i32 = batch_size as i32;
        let nh_i32 = num_heads as i32;
        let sq_i32 = seq_len_q as i32;
        let sk_i32 = seq_len_k as i32;
        let mnb_i32 = max_num_blocks as i32;
        let bs_i32 = block_size as i32;
        let causal_i32 = if causal { 1i32 } else { 0i32 };

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            // Kernel signature: Q, K_blocks, V_blocks, O, dO, L, block_table, dQ, dK_blocks, dV_blocks, ...
            builder.arg(&q_ptr);
            builder.arg(&kb_ptr);
            builder.arg(&vb_ptr);
            builder.arg(&o_ptr);
            builder.arg(&do_ptr);
            builder.arg(&l_ptr);
            builder.arg(&bt_ptr);
            builder.arg(&dq_ptr);
            builder.arg(&dkb_ptr);
            builder.arg(&dvb_ptr);
            builder.arg(&batch_i32);
            builder.arg(&nh_i32);
            builder.arg(&sq_i32);
            builder.arg(&sk_i32);
            builder.arg(&mnb_i32);
            builder.arg(&bs_i32);
            builder.arg(&scale);
            builder.arg(&causal_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("Paged attention bwd kernel launch failed: {:?}", e),
            })?;
        }

        Ok((dq, dk_blocks, dv_blocks))
    }
}
