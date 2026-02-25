//! CUDA implementation of FusedFp8TrainingOps
//!
//! Phase 1: fused_grad_unscale_clip_* — unscale, detect inf/nan, accumulate norm².
//! Phase 2: clip_scale_* — reads norm_sq/found_inf from device memory, computes
//!   clip_coef on-device, applies it. No host roundtrip between phases.
//! Dynamic loss scale update delegates to impl_generic (pure scalar).

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, FUSED_GRAD_UNSCALE_CLIP_MODULE};
use crate::ops::impl_generic::training::dynamic_loss_scale_update_impl;
use crate::ops::traits::FusedFp8TrainingOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

fn kernel_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F64 => Ok("f64"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        _ => Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("fused_grad_unscale_clip: unsupported dtype {:?}", dtype),
        }),
    }
}

impl FusedFp8TrainingOps<CudaRuntime> for CudaClient {
    fn fused_grad_unscale_clip(
        &self,
        grad: &Tensor<CudaRuntime>,
        max_norm: f64,
        loss_scale: f64,
    ) -> Result<(Tensor<CudaRuntime>, f64, bool)> {
        let n: usize = grad.shape().iter().product();
        let dtype = grad.dtype();
        let suffix = kernel_suffix(dtype)?;

        let out = grad.clone();

        let device_index = grad.device().id();
        let module = kernels::get_or_load_module(
            self.context(),
            device_index,
            FUSED_GRAD_UNSCALE_CLIP_MODULE,
        )?;

        let stream = self.stream_arc();

        // Allocate device scalars: found_inf (i32), norm_sq (f32)
        let mut found_inf_dev = unsafe {
            stream.alloc::<i32>(1).map_err(|e| Error::KernelError {
                reason: format!("alloc found_inf: {:?}", e),
            })?
        };
        stream
            .memcpy_htod(&[0i32], &mut found_inf_dev)
            .map_err(|e| Error::KernelError {
                reason: format!("zero found_inf: {:?}", e),
            })?;

        let mut norm_sq_dev = unsafe {
            stream.alloc::<f32>(1).map_err(|e| Error::KernelError {
                reason: format!("alloc norm_sq: {:?}", e),
            })?
        };
        stream
            .memcpy_htod(&[0.0f32], &mut norm_sq_dev)
            .map_err(|e| Error::KernelError {
                reason: format!("zero norm_sq: {:?}", e),
            })?;

        let threads = 256u32;
        let blocks = ((n + 255) / 256) as u32;
        let shared_bytes = threads * 4; // f32 per thread for reduction
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared_bytes,
        };

        // ── Phase 1: unscale + inf/nan detect + norm² accumulation ──────────
        let kernel_name = format!("fused_grad_unscale_clip_{}", suffix);
        let func = kernels::get_kernel_function(&module, &kernel_name)?;

        let out_ptr = out.ptr();
        let grad_ptr = grad.ptr();
        let n_i32 = n as i32;
        let inv_scale_f = (1.0 / loss_scale) as f32;
        let inv_scale_d = 1.0 / loss_scale;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&grad_ptr);
            builder.arg(&found_inf_dev);
            builder.arg(&norm_sq_dev);
            if dtype == DType::F64 {
                builder.arg(&inv_scale_d);
            } else {
                builder.arg(&inv_scale_f);
            }
            builder.arg(&n_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("fused_grad_unscale_clip launch failed: {:?}", e),
            })?;
        }

        // ── Phase 2: clip on-device — no host readback before this ──────────
        // clip_scale_* reads norm_sq and found_inf from device memory directly,
        // computes clip_coef on-device, and applies it conditionally.
        if max_norm > 0.0 {
            let clip_name = format!("clip_scale_{}", suffix);
            let clip_func = kernels::get_kernel_function(&module, &clip_name)?;
            let clip_cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let max_norm_f = max_norm as f32;

            unsafe {
                let mut builder = self.stream().launch_builder(&clip_func);
                builder.arg(&out_ptr);
                builder.arg(&norm_sq_dev);
                builder.arg(&found_inf_dev);
                if dtype == DType::F64 {
                    builder.arg(&max_norm);
                } else {
                    builder.arg(&max_norm_f);
                }
                builder.arg(&n_i32);
                builder.launch(clip_cfg).map_err(|e| Error::KernelError {
                    reason: format!("clip_scale launch failed: {:?}", e),
                })?;
            }
        }

        // ── Single readback after both kernels are enqueued ──────────────────
        let mut found_inf_host = [0i32];
        stream
            .memcpy_dtoh(&found_inf_dev, &mut found_inf_host)
            .map_err(|e| Error::KernelError {
                reason: format!("read found_inf: {:?}", e),
            })?;

        let mut norm_sq_host = [0.0f32];
        stream
            .memcpy_dtoh(&norm_sq_dev, &mut norm_sq_host)
            .map_err(|e| Error::KernelError {
                reason: format!("read norm_sq: {:?}", e),
            })?;

        let found_inf = found_inf_host[0] != 0;
        let norm = (norm_sq_host[0] as f64).sqrt();

        Ok((out, norm, found_inf))
    }

    fn dynamic_loss_scale_update(
        &self,
        found_inf: bool,
        loss_scale: f64,
        growth_tracker: i32,
        growth_interval: i32,
        backoff_factor: f64,
    ) -> Result<(f64, i32)> {
        dynamic_loss_scale_update_impl(
            found_inf,
            loss_scale,
            growth_tracker,
            growth_interval,
            backoff_factor,
        )
    }
}
