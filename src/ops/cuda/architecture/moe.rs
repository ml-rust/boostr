//! CUDA implementation of MoEOps
//!
//! Uses fused CUDA kernels for routing, permutation, and grouped GEMM.
//! All kernels support F32/F16/BF16; routing always outputs F32 weights + I32 indices.
//! Permute/unpermute: argsort + offset computation stays in impl_generic (numr-native),
//! only the final scatter/gather step uses custom kernels.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, MOE_GROUPED_GEMM_MODULE, MOE_ROUTING_MODULE};
use crate::ops::impl_generic::architecture::moe::{
    moe_permute_tokens_impl, moe_unpermute_tokens_impl,
};
use crate::ops::traits::architecture::moe::{MoEActivation, MoEOps};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Select routing kernel name by dtype
fn routing_kernel_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("moe_top_k_routing_f32"),
        DType::F16 => Ok("moe_top_k_routing_f16"),
        DType::BF16 => Ok("moe_top_k_routing_bf16"),
        _ => Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!(
                "MoE routing: unsupported dtype {:?}, expected F32/F16/BF16",
                dtype
            ),
        }),
    }
}

/// Select grouped GEMM kernel name by (dtype, activation)
fn grouped_gemm_kernel_name(dtype: DType, activation: MoEActivation) -> Result<&'static str> {
    match (dtype, activation) {
        (DType::F32, MoEActivation::None) => Ok("moe_grouped_gemm_f32"),
        (DType::F32, MoEActivation::SiLU) => Ok("moe_grouped_gemm_silu_f32"),
        (DType::F32, MoEActivation::GeLU) => Ok("moe_grouped_gemm_gelu_f32"),
        (DType::F16, MoEActivation::None) => Ok("moe_grouped_gemm_f16"),
        (DType::F16, MoEActivation::SiLU) => Ok("moe_grouped_gemm_silu_f16"),
        (DType::F16, MoEActivation::GeLU) => Ok("moe_grouped_gemm_gelu_f16"),
        (DType::BF16, MoEActivation::None) => Ok("moe_grouped_gemm_bf16"),
        (DType::BF16, MoEActivation::SiLU) => Ok("moe_grouped_gemm_silu_bf16"),
        (DType::BF16, MoEActivation::GeLU) => Ok("moe_grouped_gemm_gelu_bf16"),
        _ => Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("MoE grouped GEMM: unsupported dtype {:?}", dtype),
        }),
    }
}

impl MoEOps<CudaRuntime> for CudaClient {
    fn moe_top_k_routing(
        &self,
        logits: &Tensor<CudaRuntime>,
        k: usize,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let shape = logits.shape();
        if shape.len() != 2 {
            return Err(Error::InvalidArgument {
                arg: "logits",
                reason: format!(
                    "expected 2D [num_tokens, num_experts], got {}D",
                    shape.len()
                ),
            });
        }

        let num_tokens = shape[0];
        let num_experts = shape[1];
        let dtype = logits.dtype();

        if k == 0 || k > num_experts {
            return Err(Error::InvalidArgument {
                arg: "k",
                reason: format!("k={} must be in [1, num_experts={}]", k, num_experts),
            });
        }

        let kernel_name = routing_kernel_name(dtype)?;
        let device = logits.device();

        // Output: indices always I32, weights always F32
        let indices = Tensor::<CudaRuntime>::empty(&[num_tokens, k], DType::I32, device);
        let weights = Tensor::<CudaRuntime>::empty(&[num_tokens, k], DType::F32, device);

        let device_index = device.id();
        let module = kernels::get_or_load_module(self.context(), device_index, MOE_ROUTING_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (num_experts * std::mem::size_of::<f32>()) as u32,
        };

        let logits_ptr = logits.ptr();
        let indices_ptr = indices.ptr();
        let weights_ptr = weights.ptr();
        let n_i32 = num_tokens as i32;
        let e_i32 = num_experts as i32;
        let k_i32 = k as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&logits_ptr);
            builder.arg(&indices_ptr);
            builder.arg(&weights_ptr);
            builder.arg(&n_i32);
            builder.arg(&e_i32);
            builder.arg(&k_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("MoE routing kernel launch failed: {:?}", e),
            })?;
        }

        Ok((indices, weights))
    }

    fn moe_permute_tokens(
        &self,
        tokens: &Tensor<CudaRuntime>,
        indices: &Tensor<CudaRuntime>,
        num_experts: usize,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // Argsort + offset computation is already GPU-native via numr.
        // Use impl_generic for the full pipeline — the index_select it uses
        // is already a CUDA kernel in numr. The custom scatter kernel would
        // only help if we fused the argsort+scatter, which we don't.
        // Keep delegating to impl_generic for correctness and simplicity.
        moe_permute_tokens_impl(self, tokens, indices, num_experts)
    }

    fn moe_unpermute_tokens(
        &self,
        expert_output: &Tensor<CudaRuntime>,
        sort_indices: &Tensor<CudaRuntime>,
        weights: &Tensor<CudaRuntime>,
        num_tokens: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        // Same rationale as permute — the impl_generic path composes
        // numr's CUDA-native index_select, reshape, mul, sum.
        moe_unpermute_tokens_impl(self, expert_output, sort_indices, weights, num_tokens)
    }

    fn moe_grouped_gemm(
        &self,
        permuted_tokens: &Tensor<CudaRuntime>,
        expert_weights: &Tensor<CudaRuntime>,
        expert_offsets: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        launch_grouped_gemm(
            self,
            permuted_tokens,
            expert_weights,
            expert_offsets,
            MoEActivation::None,
        )
    }

    fn moe_grouped_gemm_fused(
        &self,
        permuted_tokens: &Tensor<CudaRuntime>,
        expert_weights: &Tensor<CudaRuntime>,
        expert_offsets: &Tensor<CudaRuntime>,
        activation: MoEActivation,
    ) -> Result<Tensor<CudaRuntime>> {
        launch_grouped_gemm(
            self,
            permuted_tokens,
            expert_weights,
            expert_offsets,
            activation,
        )
    }
}

/// Shared grouped GEMM kernel launch for both plain and fused variants.
fn launch_grouped_gemm(
    client: &CudaClient,
    permuted_tokens: &Tensor<CudaRuntime>,
    expert_weights: &Tensor<CudaRuntime>,
    expert_offsets: &Tensor<CudaRuntime>,
    activation: MoEActivation,
) -> Result<Tensor<CudaRuntime>> {
    // Validate shapes
    let pt_shape = permuted_tokens.shape();
    let ew_shape = expert_weights.shape();
    let eo_shape = expert_offsets.shape();

    if pt_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "permuted_tokens",
            reason: format!("expected 2D [total, in_dim], got {}D", pt_shape.len()),
        });
    }
    if ew_shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "expert_weights",
            reason: format!(
                "expected 3D [num_experts, in_dim, out_dim], got {}D",
                ew_shape.len()
            ),
        });
    }
    if eo_shape.len() != 1 {
        return Err(Error::InvalidArgument {
            arg: "expert_offsets",
            reason: format!("expected 1D [num_experts+1], got {}D", eo_shape.len()),
        });
    }
    if pt_shape[1] != ew_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "expert_weights",
            reason: format!(
                "in_dim mismatch: tokens has {}, weights has {}",
                pt_shape[1], ew_shape[1]
            ),
        });
    }
    if eo_shape[0] != ew_shape[0] + 1 {
        return Err(Error::InvalidArgument {
            arg: "expert_offsets",
            reason: format!(
                "expected {} entries (num_experts+1), got {}",
                ew_shape[0] + 1,
                eo_shape[0]
            ),
        });
    }

    let dtype = permuted_tokens.dtype();
    let total_tokens = pt_shape[0];
    let in_dim = pt_shape[1];
    let num_experts = ew_shape[0];
    let out_dim = ew_shape[2];
    let device = permuted_tokens.device();

    let kernel_name = grouped_gemm_kernel_name(dtype, activation)?;

    // Allocate output
    let output = Tensor::<CudaRuntime>::empty(&[total_tokens, out_dim], dtype, device);

    if total_tokens == 0 {
        return Ok(output);
    }

    let device_index = device.id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, MOE_GROUPED_GEMM_MODULE)?;
    let func = kernels::get_kernel_function(&module, kernel_name)?;

    // Grid y uses total_tokens as a conservative upper bound — no CPU readback needed.
    // The kernel reads offsets from device memory and guards: `if (row < count)`.
    const TILE: u32 = 32;
    let grid_x = (out_dim as u32).div_ceil(TILE);
    let grid_y = (total_tokens as u32).div_ceil(TILE);
    let grid_z = num_experts as u32;

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (TILE, TILE, 1),
        shared_mem_bytes: 0,
    };

    let tokens_ptr = permuted_tokens.ptr();
    let weights_ptr = expert_weights.ptr();
    let offsets_ptr = expert_offsets.ptr();
    let output_ptr = output.ptr();
    let in_dim_i32 = in_dim as i32;
    let out_dim_i32 = out_dim as i32;
    let num_experts_i32 = num_experts as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&tokens_ptr);
        builder.arg(&weights_ptr);
        builder.arg(&offsets_ptr);
        builder.arg(&output_ptr);
        builder.arg(&in_dim_i32);
        builder.arg(&out_dim_i32);
        builder.arg(&num_experts_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("MoE grouped GEMM kernel launch failed: {:?}", e),
        })?;
    }

    Ok(output)
}
