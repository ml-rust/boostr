//! MQA/GQA dedicated attention CUDA forward launcher
//!
//! Used at every GQA ratio the kernel is capable of (see
//! [`super::block_config::should_use_mqa_gqa`] for the capability gate),
//! from true MQA (num_kv_heads=1) through plain MHA (ratio 1). flash_v2 is
//! the fallback only for shapes this kernel cannot handle.
//!
//! Kernel: mqa_gqa.cu

use crate::error::{Error, Result};
use crate::ops::traits::AttnOutLayout;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::super::flash::flash_block_config::register_tile_smem_bytes;
use super::super::flash::flash_utils::set_smem_attribute;
use super::block_config::mqa_fwd_tile;
use crate::ops::cuda::kernels::{self, MQA_GQA_MODULE};
use numr::runtime::cuda::CudaDevice;

/// MQA/GQA forward pass — dedicated kernel, used at every capable ratio.
///
/// `kv_start` is the device pointer of the `[B]` I32 left-padding starts,
/// or `0` for none; the kernel reads it once per block.
///
/// `out_layout` reaches the kernel as a store-address flag; the arithmetic
/// is the same either way.
#[allow(clippy::too_many_arguments)]
pub fn mqa_gqa_fwd(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    kv_start: u64,
    out_layout: AttnOutLayout,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let dtype = q.dtype();

    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(Error::InvalidArgument {
            arg: "num_kv_heads",
            reason: format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                num_heads, num_kv_heads
            ),
        });
    }

    let batch_size = q_shape[0];
    let seq_len_q = q_shape[2];
    let seq_len_k = k_shape[2];

    let dtype_suffix = match dtype {
        DType::F32 => "fp32",
        DType::F16 => "fp16",
        DType::BF16 => "bf16",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("unsupported dtype {:?} for MQA/GQA", dtype),
            });
        }
    };

    let device = q.device();
    let device_index = device.id();

    // The kernel reads Q and K/V and writes O four elements at a time: float4
    // for f32, 8-byte vectors for f16/bf16. A contiguous [B, H, S, D] tensor at
    // a supported head_dim keeps every row aligned as long as its base is;
    // `validate_qkv` already required contiguity, so this only guards the base
    // pointer.
    for (name, ptr) in [("q", q.ptr()), ("k", k.ptr()), ("v", v.ptr())] {
        if !ptr.is_multiple_of(16) {
            return Err(Error::InvalidArgument {
                arg: name,
                reason: "MQA/GQA forward needs 16-byte aligned tensors".into(),
            });
        }
    }

    let compute_units = CudaDevice::new(device_index).profile().compute_units as usize;
    let tile = mqa_fwd_tile(head_dim, seq_len_q, batch_size * num_heads, compute_units)?;

    let variant = if tile.small { "_sm" } else { "" };
    let kernel_name = format!("mqa_gqa_fwd_{}_{}{}", head_dim, dtype_suffix, variant);

    let output = Tensor::<CudaRuntime>::empty(
        &out_layout.shape(batch_size, num_heads, seq_len_q, head_dim),
        dtype,
        device,
    )?;
    let lse =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q], DType::F32, device)?;

    let smem_size = register_tile_smem_bytes(tile, head_dim);

    let module = kernels::get_or_load_module(client.context(), device_index, MQA_GQA_MODULE)?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;
    set_smem_attribute(&func, smem_size)?;

    let cfg = LaunchConfig {
        grid_dim: (
            (batch_size * num_heads) as u32,
            seq_len_q.div_ceil(tile.rows) as u32,
            1,
        ),
        block_dim: (tile.threads as u32, 1, 1),
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
    let nkv_i32 = num_kv_heads as i32;
    let sq_i32 = seq_len_q as i32;
    let sk_i32 = seq_len_k as i32;
    let causal_i32 = if causal { 1i32 } else { 0i32 };
    // Every entry point declares the four trailing FP8 quantization scales.
    // Only the FP8 entries read them, and this launcher rejects FP8 above, so
    // 1.0f is the identity here.
    let one = 1.0f32;
    let token_major_i32 = i32::from(out_layout == AttnOutLayout::TokenMajor);

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
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
        // q_scale, k_scale, v_scale, o_scale
        for _ in 0..4 {
            builder.arg(&one);
        }
        builder.arg(&kv_start);
        builder.arg(&token_major_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("MQA/GQA fwd kernel launch failed: {:?}", e),
        })?;
    }

    Ok((output, lse))
}
