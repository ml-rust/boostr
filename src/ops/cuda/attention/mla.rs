//! CUDA implementation of MlaOps — fused SDPA kernel dispatch
//!
//! Multi-Head Latent Attention (MLA) scaled dot-product attention.
//! Unlike standard attention, K and V can have different last dimensions.

use super::flash_utils::{device_max_smem, set_smem_attribute};
use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, SDPA_MODULE};
use crate::ops::traits::MlaOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};

/// Q rows per thread block. Must stay in sync with `BLOCK_M` in `sdpa.cu`;
/// it is also the block's thread count.
const SDPA_BLOCK_M: usize = 128;
/// K/V columns staged per iteration. Must stay in sync with `BLOCK_N` in `sdpa.cu`.
const SDPA_BLOCK_N: usize = 128;
/// Length of the per-thread `float O_local[256]` accumulator in `sdpa.cu`.
/// `head_dim_v` indexes it directly, so a larger value corrupts the stack.
const SDPA_MAX_HEAD_DIM_V: usize = 256;

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

        // Select kernel
        let kernel_name = match dtype {
            DType::F32 => "sdpa_f32",
            DType::F16 => "sdpa_f16",
            DType::BF16 => "sdpa_bf16",
            _ => {
                return Err(Error::KernelError {
                    reason: format!("SDPA: unsupported dtype {:?}", dtype),
                });
            }
        };

        // Create output tensor: [B, H, S_q, D_v]
        let device = q_tensor.device();
        let output_shape = vec![batch_size, num_heads, seq_len_q, head_dim_v];
        let output = numr::tensor::Tensor::<CudaRuntime>::empty(&output_shape, dtype, device)?;

        // Get kernel function
        let device_index = device.id();
        let module = kernels::get_or_load_module(self.context(), device_index, SDPA_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        // Opt in to more than the 48KB default dynamic shared memory before
        // sizing the launch, exactly as the flash/varlen/paged launchers do.
        let smem_size = sdpa_smem_size(head_dim_k, head_dim_v);
        let max_smem = device_max_smem();
        if smem_size > max_smem {
            return Err(Error::KernelError {
                reason: format!(
                    "SDPA shared memory requirement ({} bytes) exceeds the device opt-in limit \
                     ({} bytes) for head_dim_k={}, head_dim_v={}: sdpa.cu instantiates only \
                     BLOCK_M={}, BLOCK_N={}, so there is no smaller tile to fall back to",
                    smem_size, max_smem, head_dim_k, head_dim_v, SDPA_BLOCK_M, SDPA_BLOCK_N
                ),
            });
        }
        set_smem_attribute(&func, smem_size)?;

        // Grid: (batch_size * num_heads, ceil(seq_len_q / BLOCK_M), 1)
        // Block: (BLOCK_M, 1, 1)
        let cfg = LaunchConfig {
            grid_dim: (
                (batch_size * num_heads) as u32,
                seq_len_q.div_ceil(SDPA_BLOCK_M) as u32,
                1,
            ),
            block_dim: (SDPA_BLOCK_M as u32, 1, 1),
            shared_mem_bytes: smem_size as u32,
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

/// Shared memory bytes the SDPA kernel needs.
///
/// Layout in `sdpa.cu` (`sdpa_f32` / `sdpa_f16` / `sdpa_bf16`):
/// `[Q: BLOCK_M x head_dim_k][K: BLOCK_N x head_dim_k][V: BLOCK_N x head_dim_v]`,
/// with NO `+1` bank-conflict padding — unlike the flash forward layout in
/// `compute_smem`.
///
/// All three kernels declare the tiles as `float*` and convert F16/BF16 inputs
/// to `float` on load, so the element size is always `f32` and NOT the input
/// dtype.
fn sdpa_smem_size(head_dim_k: usize, head_dim_v: usize) -> usize {
    (SDPA_BLOCK_M * head_dim_k + SDPA_BLOCK_N * head_dim_k + SDPA_BLOCK_N * head_dim_v)
        * size_of::<f32>()
}
