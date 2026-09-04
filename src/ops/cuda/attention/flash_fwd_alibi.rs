//! CUDA launcher for `flash_attention_fwd_alibi` — fused Flash Attention
//! forward with ALiBi bias computed on-the-fly (online softmax, the
//! `[B, H, S_q, S_k]` scores tensor is never materialized).
//!
//! Kernel: `position/alibi.cu`, `flash_attention_alibi_fp32_impl<HEAD_DIM,
//! BLOCK_M, BLOCK_N>`, entry points `flash_attention_alibi_64_fp32`
//! (`<64,128,128>`) and `flash_attention_alibi_128_fp32` (`<128,128,64>`).
//! F32 only, no GQA — K/V carry `num_heads` heads, like `flash_fwd_fp8_kv.rs`.
//!
//! Launch contract, read off the kernel body:
//! - `blockIdx.x` = `batch_idx * num_heads + head_idx`
//! - `blockIdx.y` = query-tile index, so `grid.y = ceil(seq_len_q / BLOCK_M)`
//! - `threadIdx.x` = query row within the tile, so `block = (BLOCK_M, 1, 1)`
//! - Dynamic shared memory holds a `[BLOCK_M, HEAD_STRIDE]` Q tile plus
//!   `[BLOCK_N, HEAD_STRIDE]` K and V tiles, `HEAD_STRIDE = HEAD_DIM + 1` —
//!   identical layout/formula to `flash_smem::compute_smem`, so this reuses
//!   it rather than recomputing the size.
//!
//! `L` holds the log-sum-exp (`m + log(l)`), not the raw softmax denominator
//! — same convention as `FlashAttentionOps::flash_attention_fwd`.

use crate::error::{Error, Result};
use crate::ops::traits::FlashAlibiOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::ops::cuda::kernels::{self, ALIBI_MODULE};

use super::flash_utils::{compute_smem, device_max_smem, set_smem_attribute};

const BLOCK_M: usize = 128;

/// `BLOCK_N` per supported head dim: `_64` uses `<64,128,128>`, `_128` uses
/// `<128,128,64>` — the two kernel instantiations in `alibi.cu`.
fn block_n_for_head_dim(head_dim: usize) -> Option<usize> {
    match head_dim {
        64 => Some(128),
        128 => Some(64),
        _ => None,
    }
}

impl FlashAlibiOps<CudaRuntime> for CudaClient {
    fn flash_attention_fwd_alibi(
        &self,
        q: &Tensor<CudaRuntime>,
        k: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        num_heads: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        for (name, t) in [("q", q), ("k", k), ("v", v)] {
            if t.dtype() != DType::F32 {
                return Err(Error::InvalidArgument {
                    arg: name,
                    reason: format!(
                        "flash_attention_fwd_alibi requires {name} in F32, got {:?}",
                        t.dtype()
                    ),
                });
            }
        }

        let block_n = match block_n_for_head_dim(head_dim) {
            Some(bn) => bn,
            None => {
                return Err(Error::InvalidArgument {
                    arg: "head_dim",
                    reason: format!(
                        "flash_attention_fwd_alibi supports head_dim 64 or 128, got {head_dim}"
                    ),
                });
            }
        };
        let kernel_name = match head_dim {
            64 => "flash_attention_alibi_64_fp32",
            128 => "flash_attention_alibi_128_fp32",
            _ => unreachable!("validated above"),
        };

        let q_shape = q.shape();
        if q_shape.len() != 4 {
            return Err(Error::InvalidArgument {
                arg: "q",
                reason: format!("expected 4D [B, H, S, D], got {}D", q_shape.len()),
            });
        }
        let (batch_size, q_heads, seq_len_q, q_head_dim) =
            (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
        if q_heads != num_heads || q_head_dim != head_dim {
            return Err(Error::InvalidArgument {
                arg: "q",
                reason: format!(
                    "q shape [{batch_size}, {q_heads}, {seq_len_q}, {q_head_dim}] does not match \
                     num_heads={num_heads} head_dim={head_dim}"
                ),
            });
        }

        let k_shape = k.shape().to_vec();
        if k_shape.len() != 4
            || k_shape[0] != batch_size
            || k_shape[1] != num_heads
            || k_shape[3] != head_dim
        {
            return Err(Error::InvalidArgument {
                arg: "k",
                reason: format!(
                    "expected [{batch_size}, {num_heads}, seq_len_k, {head_dim}], got {k_shape:?}"
                ),
            });
        }
        let seq_len_k = k_shape[2];
        if v.shape() != k_shape.as_slice() {
            return Err(Error::InvalidArgument {
                arg: "v",
                reason: format!("v shape {:?} must match k shape {:?}", v.shape(), k_shape),
            });
        }

        // Both instantiations exceed the 48KB default cap (99,840 bytes for
        // head_dim=64, 132,096 for head_dim=128), so the opt-in path below
        // always runs. Naming the head dim gives a clearer error than the
        // generic `set_smem_attribute` message.
        let smem_size = compute_smem(BLOCK_M, block_n, head_dim, std::mem::size_of::<f32>());
        let max_smem = device_max_smem();
        if smem_size > max_smem {
            return Err(Error::KernelError {
                reason: format!(
                    "flash_attention_fwd_alibi: head_dim={head_dim} needs {}KB shared memory, \
                     device limit is {}KB",
                    smem_size / 1024,
                    max_smem / 1024
                ),
            });
        }

        let device = q.device();
        let output = Tensor::<CudaRuntime>::empty(
            &[batch_size, num_heads, seq_len_q, head_dim],
            DType::F32,
            device,
        )?;
        let lse =
            Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q], DType::F32, device)?;

        let device_index = device.id();
        let module = kernels::get_or_load_module(self.context(), device_index, ALIBI_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;
        set_smem_attribute(&func, smem_size)?;

        let cfg = LaunchConfig {
            grid_dim: (
                (batch_size * num_heads) as u32,
                seq_len_q.div_ceil(BLOCK_M) as u32,
                1,
            ),
            block_dim: (BLOCK_M as u32, 1, 1),
            shared_mem_bytes: smem_size as u32,
        };

        let q_ptr = q.ptr();
        let k_ptr = k.ptr();
        let v_ptr = v.ptr();
        let o_ptr = output.ptr();
        let l_ptr = lse.ptr();
        let scale = (head_dim as f32).sqrt().recip();
        let batch_i32 = batch_size as i32;
        let nh_i32 = num_heads as i32;
        let sq_i32 = seq_len_q as i32;
        let sk_i32 = seq_len_k as i32;
        let causal_i32 = if causal { 1i32 } else { 0i32 };

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&q_ptr);
            builder.arg(&k_ptr);
            builder.arg(&v_ptr);
            builder.arg(&o_ptr);
            builder.arg(&l_ptr);
            builder.arg(&batch_i32);
            builder.arg(&nh_i32);
            builder.arg(&sq_i32);
            builder.arg(&sk_i32);
            builder.arg(&scale);
            builder.arg(&causal_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("Flash Attention ALiBi fwd kernel launch failed: {:?}", e),
            })?;
        }

        Ok((output, lse))
    }
}
