//! CUDA implementation of MlaOps — fused SDPA kernel dispatch
//!
//! Multi-Head Latent Attention (MLA) scaled dot-product attention.
//! Unlike standard attention, K and V can have different last dimensions.

use super::flash_utils::set_smem_attribute;
use super::mla_block_config::{SDPA_MAX_HEAD_DIM_V, block_config};
use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, SDPA_MODULE};
use crate::ops::traits::MlaOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::autograd::Var;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};

impl MlaOps<CudaRuntime> for CudaClient {
    fn scaled_dot_product_attention(
        &self,
        q: &Var<CudaRuntime>,
        k: &Var<CudaRuntime>,
        v: &Var<CudaRuntime>,
        scale: f64,
        causal: bool,
    ) -> Result<Var<CudaRuntime>> {
        let q_tensor = q.tensor();
        let k_tensor = k.tensor();
        let v_tensor = v.tensor();

        // Validate shapes: all 4D [B, H, S_*, D_*]
        let q_shape = q_tensor.shape();
        let k_shape = k_tensor.shape();
        let v_shape = v_tensor.shape();

        if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
            return Err(Error::InvalidArgument {
                arg: "attention",
                reason: format!(
                    "all inputs must be 4D [B, H, S, D]: q={:?}, k={:?}, v={:?}",
                    q_shape, k_shape, v_shape
                ),
            });
        }

        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let seq_len_q = q_shape[2];
        let head_dim_k = q_shape[3];

        let seq_len_k = k_shape[2];
        let head_dim_v = v_shape[3];

        // Validate batch dimension matches
        if k_shape[0] != batch_size || v_shape[0] != batch_size {
            return Err(Error::InvalidArgument {
                arg: "batch",
                reason: format!(
                    "batch dimension mismatch: q={}, k={}, v={}",
                    q_shape[0], k_shape[0], v_shape[0]
                ),
            });
        }

        // Validate heads dimension matches
        if k_shape[1] != num_heads || v_shape[1] != num_heads {
            return Err(Error::InvalidArgument {
                arg: "heads",
                reason: format!(
                    "num_heads dimension mismatch: q={}, k={}, v={}",
                    q_shape[1], k_shape[1], v_shape[1]
                ),
            });
        }

        // Validate key dimensions match between Q and K
        if k_shape[3] != head_dim_k {
            return Err(Error::InvalidArgument {
                arg: "head_dim",
                reason: format!(
                    "K head_dim must match Q: q={}, k={}",
                    head_dim_k, k_shape[3]
                ),
            });
        }

        // The kernel accumulates the output row in a fixed-length per-thread
        // register array, indexed by `head_dim_v` with no bounds check.
        if head_dim_v > SDPA_MAX_HEAD_DIM_V {
            return Err(Error::InvalidArgument {
                arg: "head_dim",
                reason: format!(
                    "V head_dim must be at most {}: v={}",
                    SDPA_MAX_HEAD_DIM_V, head_dim_v
                ),
            });
        }

        // Verify dtype consistency
        let dtype = q_tensor.dtype();
        if k_tensor.dtype() != dtype || v_tensor.dtype() != dtype {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!(
                    "all inputs must have same dtype: q={:?}, k={:?}, v={:?}",
                    dtype,
                    k_tensor.dtype(),
                    v_tensor.dtype()
                ),
            });
        }

        // Select the kernel and the tile together: `block_config` picks the
        // largest tile whose shared-memory requirement fits this device and
        // returns the matching symbol name, so the name and `block_dim` can
        // never disagree.
        let tile = block_config(dtype, head_dim_k, head_dim_v)?;

        // Create output tensor: [B, H, S_q, D_v]
        let device = q_tensor.device();
        let output_shape = vec![batch_size, num_heads, seq_len_q, head_dim_v];
        let output = numr::tensor::Tensor::<CudaRuntime>::empty(&output_shape, dtype, device)?;

        // Get kernel function
        let device_index = device.id();
        let module = kernels::get_or_load_module(self.context(), device_index, SDPA_MODULE)?;
        let func = kernels::get_kernel_function(&module, &tile.kernel_name)?;

        // Opt in to more than the 48KB default dynamic shared memory before
        // sizing the launch, exactly as the flash/varlen/paged launchers do.
        // `block_config` already checked the request against the device limit.
        set_smem_attribute(&func, tile.smem_bytes)?;

        // Grid: (batch_size * num_heads, ceil(seq_len_q / BLOCK_M), 1)
        // Block: (BLOCK_M, 1, 1) — `sdpa.cu` gives one Q row to each thread,
        // so `block_dim.x` MUST be the selected tile's own BLOCK_M.
        let cfg = LaunchConfig {
            grid_dim: (
                (batch_size * num_heads) as u32,
                seq_len_q.div_ceil(tile.block_m) as u32,
                1,
            ),
            block_dim: (tile.block_m as u32, 1, 1),
            shared_mem_bytes: tile.smem_bytes as u32,
        };

        // Extract pointers
        let q_ptr = q_tensor.ptr();
        let k_ptr = k_tensor.ptr();
        let v_ptr = v_tensor.ptr();
        let out_ptr = output.ptr();

        let b_i32 = batch_size as i32;
        let nh_i32 = num_heads as i32;
        let sq_i32 = seq_len_q as i32;
        let sk_i32 = seq_len_k as i32;
        let hdk_i32 = head_dim_k as i32;
        let hdv_i32 = head_dim_v as i32;
        let scale_f32 = scale as f32;
        let causal_i32 = if causal { 1i32 } else { 0i32 };

        // Launch kernel
        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&q_ptr);
            builder.arg(&k_ptr);
            builder.arg(&v_ptr);
            builder.arg(&out_ptr);
            builder.arg(&b_i32);
            builder.arg(&nh_i32);
            builder.arg(&sq_i32);
            builder.arg(&sk_i32);
            builder.arg(&hdk_i32);
            builder.arg(&hdv_i32);
            builder.arg(&scale_f32);
            builder.arg(&causal_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("SDPA kernel launch failed: {:?}", e),
            })?;
        }

        // No sync needed: same-stream ordering guarantees the kernel
        // completes before any subsequent kernel on this stream.

        Ok(Var::new(output, false))
    }
}
