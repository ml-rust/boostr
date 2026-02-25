//! CUDA Flash Attention v2 — forward and backward
//!
//! Fused kernel — this is a PRIMITIVE op (kernel IS the algorithm).
//! Supports: F32, F16, BF16, FP8E4M3 with GQA and sliding window.
//! Head dimensions: 32, 64, 96, 128, 192, 256.

use crate::error::{Error, Result};
use crate::ops::traits::FlashAttentionOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use cudarc::driver::sys;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash_v3;
use crate::ops::cuda::kernels::{self, FLASH_V2_BWD_MODULE, FLASH_V2_MODULE};

/// Validated attention parameters extracted from tensor shapes.
struct AttentionParams {
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len_q: usize,
    seq_len_k: usize,
    head_dim: usize,
    block_m: usize,
    block_n: usize,
}

/// Validate Q/K/V shapes and extract parameters.
fn validate_qkv(
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<AttentionParams> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    if q_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("expected 4D [B, H, S, D], got {}D", q_shape.len()),
        });
    }
    if k_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "k",
            reason: format!("expected 4D, got {}D", k_shape.len()),
        });
    }
    if v_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "v",
            reason: format!("expected 4D, got {}D", v_shape.len()),
        });
    }
    if q_shape[1] != num_heads {
        return Err(Error::InvalidArgument {
            arg: "num_heads",
            reason: format!("num_heads={} but q dim 1 is {}", num_heads, q_shape[1]),
        });
    }
    if k_shape[1] != num_kv_heads {
        return Err(Error::InvalidArgument {
            arg: "num_kv_heads",
            reason: format!(
                "num_kv_heads={} but k dim 1 is {}",
                num_kv_heads, k_shape[1]
            ),
        });
    }
    if q_shape[3] != head_dim || k_shape[3] != head_dim || v_shape[3] != head_dim {
        return Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!(
                "head_dim={} but q.D={}, k.D={}, v.D={}",
                head_dim, q_shape[3], k_shape[3], v_shape[3]
            ),
        });
    }
    if q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] {
        return Err(Error::InvalidArgument {
            arg: "batch_size",
            reason: format!(
                "batch mismatch: q.B={}, k.B={}, v.B={}",
                q_shape[0], k_shape[0], v_shape[0]
            ),
        });
    }
    if k_shape[2] != v_shape[2] {
        return Err(Error::InvalidArgument {
            arg: "v",
            reason: format!("k seq_len={} != v seq_len={}", k_shape[2], v_shape[2]),
        });
    }
    if num_heads % num_kv_heads != 0 {
        return Err(Error::InvalidArgument {
            arg: "num_kv_heads",
            reason: format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                num_heads, num_kv_heads
            ),
        });
    }

    let dtype = q.dtype();
    if k.dtype() != dtype || v.dtype() != dtype {
        return Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!(
                "Q/K/V dtype mismatch: Q={:?}, K={:?}, V={:?}",
                dtype,
                k.dtype(),
                v.dtype()
            ),
        });
    }
    if !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() {
        return Err(Error::InvalidArgument {
            arg: "contiguity",
            reason: "Flash Attention requires contiguous Q, K, V tensors".into(),
        });
    }

    let (block_m, block_n) = block_config(head_dim)?;

    Ok(AttentionParams {
        batch_size: q_shape[0],
        num_heads,
        num_kv_heads,
        seq_len_q: q_shape[2],
        seq_len_k: k_shape[2],
        head_dim,
        block_m,
        block_n,
    })
}

/// Get block configuration for a head dimension.
fn block_config(head_dim: usize) -> Result<(usize, usize)> {
    match head_dim {
        32 => Ok((128, 128)),
        64 => Ok((128, 128)),
        96 => Ok((64, 128)),
        128 => Ok((128, 64)),
        192 => Ok((64, 64)),
        256 => Ok((64, 64)),
        _ => Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!(
                "unsupported head_dim={}. Supported: 32, 64, 96, 128, 192, 256",
                head_dim
            ),
        }),
    }
}

/// Set dynamic shared memory attribute if >48KB.
pub(crate) fn set_smem_attribute(
    func: &cudarc::driver::safe::CudaFunction,
    smem_size: usize,
) -> Result<()> {
    if smem_size <= 48 * 1024 {
        return Ok(());
    }

    let max_shared_mem = unsafe {
        let mut cuda_dev: i32 = 0;
        sys::cuCtxGetDevice(&mut cuda_dev);
        let mut max_smem: i32 = 0;
        sys::cuDeviceGetAttribute(
            &mut max_smem,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            cuda_dev,
        );
        max_smem as usize
    };

    if smem_size > max_shared_mem {
        return Err(Error::KernelError {
            reason: format!(
                "shared memory {}KB exceeds device limit {}KB",
                smem_size / 1024,
                max_shared_mem / 1024
            ),
        });
    }

    // Extract CUfunction handle (second field of CudaFunction)
    let cu_function: sys::CUfunction = unsafe {
        let kernel_ptr = func as *const _ as *const usize;
        std::ptr::read(kernel_ptr.add(1)) as sys::CUfunction
    };

    unsafe {
        let result = sys::cuFuncSetAttribute(
            cu_function,
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem_size as i32,
        );
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(Error::KernelError {
                reason: format!(
                    "failed to set dynamic shared memory to {}KB: {:?}",
                    smem_size / 1024,
                    result
                ),
            });
        }
    }

    Ok(())
}

impl FlashAttentionOps<CudaRuntime> for CudaClient {
    fn flash_attention_fwd(
        &self,
        q: &Tensor<CudaRuntime>,
        k: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        window_size: usize,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let p = validate_qkv(q, k, v, num_heads, num_kv_heads, head_dim)?;

        // Try Flash v3 on Hopper (SM 90+) for supported configs
        if num_kv_heads == num_heads && window_size == 0 && flash_v3::is_hopper(self, &q.device()) {
            if let Some(result) = flash_v3::flash_v3_fwd(
                self,
                q,
                k,
                v,
                p.batch_size,
                p.num_heads,
                p.seq_len_q,
                p.seq_len_k,
                p.head_dim,
                causal,
            )? {
                return Ok(result);
            }
        }

        let dtype = q.dtype();

        let dtype_suffix = match dtype {
            DType::F32 => "fp32",
            DType::F16 => "fp16",
            DType::BF16 => "bf16",
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "dtype",
                    reason: format!(
                        "unsupported dtype {:?} for flash_attention_fwd. Use flash_attention_fwd_fp8 for FP8.",
                        dtype
                    ),
                });
            }
        };
        let kernel_name = format!("flash_attention_fwd_{}_{}", head_dim, dtype_suffix);

        let device = q.device();
        let output = Tensor::<CudaRuntime>::empty(
            &[p.batch_size, p.num_heads, p.seq_len_q, p.head_dim],
            dtype,
            device,
        );
        let lse = Tensor::<CudaRuntime>::empty(
            &[p.batch_size, p.num_heads, p.seq_len_q],
            DType::F32,
            device,
        );

        let head_stride = head_dim + 1;
        let dtype_size = dtype.size_in_bytes();
        let smem_size = (p.block_m * head_stride + 2 * p.block_n * head_stride) * dtype_size;

        let device_index = device.id();
        let module = kernels::get_or_load_module(self.context(), device_index, FLASH_V2_MODULE)?;
        let func = kernels::get_kernel_function(&module, &kernel_name)?;
        set_smem_attribute(&func, smem_size)?;

        let cfg = LaunchConfig {
            grid_dim: (
                (p.batch_size * p.num_heads) as u32,
                ((p.seq_len_q + p.block_m - 1) / p.block_m) as u32,
                1,
            ),
            block_dim: (p.block_m as u32, 1, 1),
            shared_mem_bytes: smem_size as u32,
        };

        let q_ptr = q.ptr();
        let k_ptr = k.ptr();
        let v_ptr = v.ptr();
        let o_ptr = output.ptr();
        let l_ptr = lse.ptr();
        let scale = (head_dim as f32).sqrt().recip();
        let batch_i32 = p.batch_size as i32;
        let nh_i32 = p.num_heads as i32;
        let nkv_i32 = p.num_kv_heads as i32;
        let sq_i32 = p.seq_len_q as i32;
        let sk_i32 = p.seq_len_k as i32;
        let causal_i32 = if causal { 1i32 } else { 0i32 };
        let ws_i32 = window_size as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&q_ptr);
            builder.arg(&k_ptr);
            builder.arg(&v_ptr);
            builder.arg(&o_ptr);
            builder.arg(&l_ptr);
            builder.arg(&batch_i32);
            builder.arg(&nh_i32);
            builder.arg(&nkv_i32);
            builder.arg(&sq_i32);
            builder.arg(&sk_i32);
            builder.arg(&scale);
            builder.arg(&causal_i32);
            builder.arg(&ws_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("Flash Attention fwd kernel launch failed: {:?}", e),
            })?;
        }

        self.stream()
            .synchronize()
            .map_err(|e| Error::KernelError {
                reason: format!("Flash Attention fwd sync failed: {:?}", e),
            })?;

        Ok((output, lse))
    }

    fn flash_attention_fwd_fp8(
        &self,
        q: &Tensor<CudaRuntime>,
        k: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        q_scale: f32,
        k_scale: f32,
        v_scale: f32,
        o_scale: f32,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let p = validate_qkv(q, k, v, num_heads, num_kv_heads, head_dim)?;
        let dtype = q.dtype();

        if !matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!(
                    "flash_attention_fwd_fp8 requires FP8 dtype, got {:?}",
                    dtype
                ),
            });
        }

        // FP8 kernels use E4M3 format (kernel handles both via same entry point)
        let kernel_name = format!("flash_attention_fwd_{}_fp8", head_dim);

        let device = q.device();
        let output = Tensor::<CudaRuntime>::empty(
            &[p.batch_size, p.num_heads, p.seq_len_q, p.head_dim],
            dtype,
            device,
        );
        let lse = Tensor::<CudaRuntime>::empty(
            &[p.batch_size, p.num_heads, p.seq_len_q],
            DType::F32,
            device,
        );

        // FP8 is 1 byte per element
        let head_stride = head_dim + 1;
        let smem_size = (p.block_m * head_stride + 2 * p.block_n * head_stride) * 1;

        let device_index = device.id();
        let module = kernels::get_or_load_module(self.context(), device_index, FLASH_V2_MODULE)?;
        let func = kernels::get_kernel_function(&module, &kernel_name)?;
        set_smem_attribute(&func, smem_size)?;

        let cfg = LaunchConfig {
            grid_dim: (
                (p.batch_size * p.num_heads) as u32,
                ((p.seq_len_q + p.block_m - 1) / p.block_m) as u32,
                1,
            ),
            block_dim: (p.block_m as u32, 1, 1),
            shared_mem_bytes: smem_size as u32,
        };

        let q_ptr = q.ptr();
        let k_ptr = k.ptr();
        let v_ptr = v.ptr();
        let o_ptr = output.ptr();
        let l_ptr = lse.ptr();
        let scale = (head_dim as f32).sqrt().recip();
        let batch_i32 = p.batch_size as i32;
        let nh_i32 = p.num_heads as i32;
        let nkv_i32 = p.num_kv_heads as i32;
        let sq_i32 = p.seq_len_q as i32;
        let sk_i32 = p.seq_len_k as i32;
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
            builder.arg(&nkv_i32);
            builder.arg(&sq_i32);
            builder.arg(&sk_i32);
            builder.arg(&scale);
            builder.arg(&causal_i32);
            builder.arg(&q_scale);
            builder.arg(&k_scale);
            builder.arg(&v_scale);
            builder.arg(&o_scale);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("Flash Attention FP8 fwd kernel launch failed: {:?}", e),
            })?;
        }

        Ok((output, lse))
    }

    fn flash_attention_bwd(
        &self,
        dout: &Tensor<CudaRuntime>,
        q: &Tensor<CudaRuntime>,
        k: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        output: &Tensor<CudaRuntime>,
        lse: &Tensor<CudaRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        window_size: usize,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let p = validate_qkv(q, k, v, num_heads, num_kv_heads, head_dim)?;

        // Try Flash v3 on Hopper (SM 90+) for supported configs
        if num_kv_heads == num_heads && window_size == 0 && flash_v3::is_hopper(self, &q.device()) {
            if let Some(result) = flash_v3::flash_v3_bwd(
                self,
                dout,
                q,
                k,
                v,
                output,
                lse,
                p.batch_size,
                p.num_heads,
                p.seq_len_q,
                p.seq_len_k,
                p.head_dim,
                causal,
            )? {
                return Ok(result);
            }
        }

        let dtype = q.dtype();

        // Validate dout and output shapes
        let expected_o_shape = [p.batch_size, p.num_heads, p.seq_len_q, p.head_dim];
        if dout.shape() != expected_o_shape {
            return Err(Error::InvalidArgument {
                arg: "dout",
                reason: format!(
                    "expected shape {:?}, got {:?}",
                    expected_o_shape,
                    dout.shape()
                ),
            });
        }
        if output.shape() != expected_o_shape {
            return Err(Error::InvalidArgument {
                arg: "output",
                reason: format!(
                    "expected shape {:?}, got {:?}",
                    expected_o_shape,
                    output.shape()
                ),
            });
        }
        let expected_lse_shape = [p.batch_size, p.num_heads, p.seq_len_q];
        if lse.shape() != expected_lse_shape {
            return Err(Error::InvalidArgument {
                arg: "lse",
                reason: format!(
                    "expected shape {:?}, got {:?}",
                    expected_lse_shape,
                    lse.shape()
                ),
            });
        }
        if !dout.is_contiguous() || !output.is_contiguous() || !lse.is_contiguous() {
            return Err(Error::InvalidArgument {
                arg: "contiguity",
                reason: "backward requires contiguous dout, output, lse".into(),
            });
        }

        let dtype_suffix = match dtype {
            DType::F32 => "fp32",
            DType::F16 => "fp16",
            DType::BF16 => "bf16",
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "dtype",
                    reason: format!("unsupported dtype {:?} for flash_attention_bwd", dtype),
                });
            }
        };

        let device = q.device();
        let device_index = device.id();

        let _ = window_size; // TODO: backward kernel doesn't yet support sliding window

        // Allocate gradient tensors (dQ must be zeroed — backward uses atomicAdd)
        let dq = Tensor::<CudaRuntime>::zeros(
            &[p.batch_size, p.num_heads, p.seq_len_q, p.head_dim],
            dtype,
            device,
        );
        let dk = Tensor::<CudaRuntime>::empty(
            &[p.batch_size, p.num_kv_heads, p.seq_len_k, p.head_dim],
            dtype,
            device,
        );
        let dv = Tensor::<CudaRuntime>::empty(
            &[p.batch_size, p.num_kv_heads, p.seq_len_k, p.head_dim],
            dtype,
            device,
        );

        // Step 1: Preprocessing — compute D = rowsum(dO ⊙ O) per query position
        // D shape: [B, num_heads, S_q]
        let d_buf = Tensor::<CudaRuntime>::empty(
            &[p.batch_size, p.num_heads, p.seq_len_q],
            DType::F32,
            device,
        );

        let module =
            kernels::get_or_load_module(self.context(), device_index, FLASH_V2_BWD_MODULE)?;

        {
            // Kernel name: flash_attention_preprocess_bwd_{head_dim}_{dtype}
            let preprocess_name = format!(
                "flash_attention_preprocess_bwd_{}_{}",
                head_dim, dtype_suffix
            );
            let func = kernels::get_kernel_function(&module, &preprocess_name)?;

            // Grid: (batch * num_heads, ceil(seq_len_q / block_size))
            let block_size = 256u32;
            let grid_x = (p.batch_size * p.num_heads) as u32;
            let grid_y = (p.seq_len_q as u32 + block_size - 1) / block_size;

            let cfg = LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            let dout_ptr = dout.ptr();
            let out_ptr = output.ptr();
            let d_ptr = d_buf.ptr();
            let batch_i32 = p.batch_size as i32;
            let nh_i32 = p.num_heads as i32;
            let sq_i32 = p.seq_len_q as i32;

            unsafe {
                let mut builder = self.stream().launch_builder(&func);
                builder.arg(&dout_ptr);
                builder.arg(&out_ptr);
                builder.arg(&d_ptr);
                builder.arg(&batch_i32);
                builder.arg(&nh_i32);
                builder.arg(&sq_i32);
                builder.launch(cfg).map_err(|e| Error::KernelError {
                    reason: format!("Flash Attention bwd preprocess failed: {:?}", e),
                })?;
            }
        }

        // Step 2: Main backward kernel — compute dQ, dK, dV
        // Kernel signature: (Q, K, V, O, dO, LSE, D, dQ, dK, dV,
        //                     batch_size, num_heads, seq_len_q, seq_len_k, scale, causal)
        {
            let bwd_name = format!("flash_attention_bwd_{}_{}", head_dim, dtype_suffix);
            let func = kernels::get_kernel_function(&module, &bwd_name)?;

            // Shared memory: K[BLOCK_N][HD] + V[BLOCK_N][HD] + Q[BLOCK_M][HD] + dO[BLOCK_M][HD]
            let dtype_size = dtype.size_in_bytes();
            let smem_size = (2 * p.block_n * head_dim + 2 * p.block_m * head_dim) * dtype_size;
            set_smem_attribute(&func, smem_size)?;

            // Grid: (batch * num_heads, ceil(seq_len_k / BLOCK_N))
            let grid_x = (p.batch_size * p.num_heads) as u32;
            let grid_y = ((p.seq_len_k + p.block_n - 1) / p.block_n) as u32;

            let cfg = LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (p.block_n as u32, 1, 1),
                shared_mem_bytes: smem_size as u32,
            };

            let q_ptr = q.ptr();
            let k_ptr = k.ptr();
            let v_ptr = v.ptr();
            let o_ptr = output.ptr();
            let dout_ptr = dout.ptr();
            let lse_ptr = lse.ptr();
            let d_ptr = d_buf.ptr();
            let dq_ptr = dq.ptr();
            let dk_ptr = dk.ptr();
            let dv_ptr = dv.ptr();
            let scale = (head_dim as f32).sqrt().recip();
            let batch_i32 = p.batch_size as i32;
            let nh_i32 = p.num_heads as i32;
            let sq_i32 = p.seq_len_q as i32;
            let sk_i32 = p.seq_len_k as i32;
            let causal_i32 = if causal { 1i32 } else { 0i32 };

            unsafe {
                let mut builder = self.stream().launch_builder(&func);
                builder.arg(&q_ptr);
                builder.arg(&k_ptr);
                builder.arg(&v_ptr);
                builder.arg(&o_ptr);
                builder.arg(&dout_ptr);
                builder.arg(&lse_ptr);
                builder.arg(&d_ptr);
                builder.arg(&dq_ptr);
                builder.arg(&dk_ptr);
                builder.arg(&dv_ptr);
                builder.arg(&batch_i32);
                builder.arg(&nh_i32);
                builder.arg(&sq_i32);
                builder.arg(&sk_i32);
                builder.arg(&scale);
                builder.arg(&causal_i32);
                builder.launch(cfg).map_err(|e| Error::KernelError {
                    reason: format!("Flash Attention bwd kernel launch failed: {:?}", e),
                })?;
            }
        }

        // Sync stream: BWD uses atomicAdd so must complete before results are read
        self.stream()
            .synchronize()
            .map_err(|e| Error::KernelError {
                reason: format!("Flash Attention bwd sync failed: {:?}", e),
            })?;

        Ok((dq, dk, dv))
    }

    fn flash_attention_bwd_fp8(
        &self,
        dout: &Tensor<CudaRuntime>,
        q: &Tensor<CudaRuntime>,
        k: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        output: &Tensor<CudaRuntime>,
        lse: &Tensor<CudaRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        q_scale: f32,
        k_scale: f32,
        v_scale: f32,
        do_scale: f32,
        o_scale: f32,
        dq_scale: f32,
        dk_scale: f32,
        dv_scale: f32,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let p = validate_qkv(q, k, v, num_heads, num_kv_heads, head_dim)?;
        let dtype = q.dtype();

        if !matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!(
                    "flash_attention_bwd_fp8 requires FP8 dtype, got {:?}",
                    dtype
                ),
            });
        }

        // Validate dout and output shapes
        let expected_o_shape = [p.batch_size, p.num_heads, p.seq_len_q, p.head_dim];
        if dout.shape() != expected_o_shape || output.shape() != expected_o_shape {
            return Err(Error::InvalidArgument {
                arg: "dout/output",
                reason: format!("expected shape {:?}", expected_o_shape),
            });
        }
        let expected_lse_shape = [p.batch_size, p.num_heads, p.seq_len_q];
        if lse.shape() != expected_lse_shape {
            return Err(Error::InvalidArgument {
                arg: "lse",
                reason: format!("expected shape {:?}", expected_lse_shape),
            });
        }

        let device = q.device();
        let device_index = device.id();

        // Allocate gradient tensors (dQ zeroed for atomicAdd)
        let dq = Tensor::<CudaRuntime>::zeros(
            &[p.batch_size, p.num_heads, p.seq_len_q, p.head_dim],
            dtype,
            device,
        );
        let dk = Tensor::<CudaRuntime>::empty(
            &[p.batch_size, p.num_kv_heads, p.seq_len_k, p.head_dim],
            dtype,
            device,
        );
        let dv = Tensor::<CudaRuntime>::empty(
            &[p.batch_size, p.num_kv_heads, p.seq_len_k, p.head_dim],
            dtype,
            device,
        );

        let d_buf = Tensor::<CudaRuntime>::empty(
            &[p.batch_size, p.num_heads, p.seq_len_q],
            DType::F32,
            device,
        );

        let module =
            kernels::get_or_load_module(self.context(), device_index, FLASH_V2_BWD_MODULE)?;

        // Step 1: FP8 Preprocessing — extra scale args
        {
            let preprocess_name = format!("flash_attention_preprocess_bwd_{}_fp8", head_dim);
            let func = kernels::get_kernel_function(&module, &preprocess_name)?;

            let block_size = 256u32;
            let grid_x = (p.batch_size * p.num_heads) as u32;
            let grid_y = (p.seq_len_q as u32 + block_size - 1) / block_size;

            let cfg = LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            let dout_ptr = dout.ptr();
            let out_ptr = output.ptr();
            let d_ptr = d_buf.ptr();
            let batch_i32 = p.batch_size as i32;
            let nh_i32 = p.num_heads as i32;
            let sq_i32 = p.seq_len_q as i32;

            unsafe {
                let mut builder = self.stream().launch_builder(&func);
                builder.arg(&dout_ptr);
                builder.arg(&out_ptr);
                builder.arg(&d_ptr);
                builder.arg(&batch_i32);
                builder.arg(&nh_i32);
                builder.arg(&sq_i32);
                builder.arg(&do_scale);
                builder.arg(&o_scale);
                builder.launch(cfg).map_err(|e| Error::KernelError {
                    reason: format!("Flash Attention FP8 bwd preprocess failed: {:?}", e),
                })?;
            }
        }

        // Step 2: FP8 Main backward — extra scale args
        {
            let bwd_name = format!("flash_attention_bwd_{}_fp8", head_dim);
            let func = kernels::get_kernel_function(&module, &bwd_name)?;

            // FP8 is 1 byte per element
            let smem_size = 2 * p.block_n * head_dim + 2 * p.block_m * head_dim;
            set_smem_attribute(&func, smem_size)?;

            let grid_x = (p.batch_size * p.num_heads) as u32;
            let grid_y = ((p.seq_len_k + p.block_n - 1) / p.block_n) as u32;

            let cfg = LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (p.block_n as u32, 1, 1),
                shared_mem_bytes: smem_size as u32,
            };

            let q_ptr = q.ptr();
            let k_ptr = k.ptr();
            let v_ptr = v.ptr();
            let o_ptr = output.ptr();
            let dout_ptr = dout.ptr();
            let lse_ptr = lse.ptr();
            let d_ptr = d_buf.ptr();
            let dq_ptr = dq.ptr();
            let dk_ptr = dk.ptr();
            let dv_ptr = dv.ptr();
            let scale = (head_dim as f32).sqrt().recip();
            let batch_i32 = p.batch_size as i32;
            let nh_i32 = p.num_heads as i32;
            let sq_i32 = p.seq_len_q as i32;
            let sk_i32 = p.seq_len_k as i32;
            let causal_i32 = if causal { 1i32 } else { 0i32 };

            unsafe {
                let mut builder = self.stream().launch_builder(&func);
                builder.arg(&q_ptr);
                builder.arg(&k_ptr);
                builder.arg(&v_ptr);
                builder.arg(&o_ptr);
                builder.arg(&dout_ptr);
                builder.arg(&lse_ptr);
                builder.arg(&d_ptr);
                builder.arg(&dq_ptr);
                builder.arg(&dk_ptr);
                builder.arg(&dv_ptr);
                builder.arg(&batch_i32);
                builder.arg(&nh_i32);
                builder.arg(&sq_i32);
                builder.arg(&sk_i32);
                builder.arg(&scale);
                builder.arg(&causal_i32);
                builder.arg(&q_scale);
                builder.arg(&k_scale);
                builder.arg(&v_scale);
                builder.arg(&do_scale);
                builder.arg(&dq_scale);
                builder.arg(&dk_scale);
                builder.arg(&dv_scale);
                builder.launch(cfg).map_err(|e| Error::KernelError {
                    reason: format!("Flash Attention FP8 bwd kernel launch failed: {:?}", e),
                })?;
            }
        }

        Ok((dq, dk, dv))
    }
}
