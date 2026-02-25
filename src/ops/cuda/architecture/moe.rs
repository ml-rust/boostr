//! CUDA implementation of MoEOps
//!
//! Uses fused CUDA kernels for routing and permutation.
//! Grouped GEMM delegates to impl_generic (per-expert matmul via cuBLAS-free path).

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, MOE_ROUTING_MODULE};
use crate::ops::impl_generic::architecture::moe::{
    moe_grouped_gemm_fused_impl, moe_grouped_gemm_impl, moe_permute_tokens_impl,
    moe_unpermute_tokens_impl,
};
use crate::ops::traits::architecture::moe::{MoEActivation, MoEOps};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

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

        if k == 0 || k > num_experts {
            return Err(Error::InvalidArgument {
                arg: "k",
                reason: format!("k={} must be in [1, num_experts={}]", k, num_experts),
            });
        }

        if logits.dtype() != DType::F32 {
            // Fallback to impl_generic for non-F32
            return moe_top_k_routing_impl(self, logits, k);
        }

        let device = logits.device();
        let indices = Tensor::<CudaRuntime>::empty(&[num_tokens, k], DType::I64, device);
        let weights = Tensor::<CudaRuntime>::empty(&[num_tokens, k], DType::F32, device);

        let device_index = device.id();
        let module = kernels::get_or_load_module(self.context(), device_index, MOE_ROUTING_MODULE)?;
        let func = kernels::get_kernel_function(&module, "moe_top_k_routing_f32")?;

        let block_size = 256u32;
        let grid_size = num_tokens as u32;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
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
        // Delegate to impl_generic — permutation is memory-bound and
        // the impl_generic path uses numr's CUDA-native index_select/argsort
        moe_permute_tokens_impl(self, tokens, indices, num_experts)
    }

    fn moe_unpermute_tokens(
        &self,
        expert_output: &Tensor<CudaRuntime>,
        sort_indices: &Tensor<CudaRuntime>,
        weights: &Tensor<CudaRuntime>,
        num_tokens: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        moe_unpermute_tokens_impl(self, expert_output, sort_indices, weights, num_tokens)
    }

    fn moe_grouped_gemm(
        &self,
        permuted_tokens: &Tensor<CudaRuntime>,
        expert_weights: &Tensor<CudaRuntime>,
        expert_offsets: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        moe_grouped_gemm_impl(self, permuted_tokens, expert_weights, expert_offsets)
    }

    fn moe_grouped_gemm_fused(
        &self,
        permuted_tokens: &Tensor<CudaRuntime>,
        expert_weights: &Tensor<CudaRuntime>,
        expert_offsets: &Tensor<CudaRuntime>,
        activation: MoEActivation,
    ) -> Result<Tensor<CudaRuntime>> {
        moe_grouped_gemm_fused_impl(
            self,
            permuted_tokens,
            expert_weights,
            expert_offsets,
            activation,
        )
    }
}

/// Helper to use impl_generic routing as fallback
fn moe_top_k_routing_impl(
    client: &CudaClient,
    logits: &Tensor<CudaRuntime>,
    k: usize,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    crate::ops::impl_generic::architecture::moe::moe_top_k_routing_impl(client, logits, k)
}
