//! Flash Attention v3 (SM 90+) CUDA launchers
//!
//! Warp-specialized forward and backward kernels for SM 90+ GPUs.
//! Supports: F32, F16, BF16, FP8E4M3 with head dims 64 and 128.
//!
//! NOT currently dispatched to. `flash.rs` routes every shape to flash v2
//! instead — see `dispatch_enabled` below for the reason and for what must
//! happen to turn it back on. `tests/flash_v3_bwd_parity_cuda.rs` calls
//! `flash_v3_fwd`/`flash_v3_bwd` directly, so these launchers stay live and
//! stay the validation gate.

use crate::error::{Error, Result};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use cudarc::driver::sys;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash::set_smem_attribute;
use crate::ops::cuda::kernels::{self, FLASH_V3_BWD_MODULE, FLASH_V3_MODULE};

use std::sync::OnceLock;

/// Cached compute capability (major, minor). Queried once.
static COMPUTE_CAP: OnceLock<(i32, i32)> = OnceLock::new();

/// Query GPU compute capability via cuDeviceGetAttribute.
fn get_compute_capability() -> (i32, i32) {
    *COMPUTE_CAP.get_or_init(|| unsafe {
        let mut device: i32 = 0;
        sys::cuCtxGetDevice(&mut device);

        let mut major: i32 = 0;
        sys::cuDeviceGetAttribute(
            &mut major,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        );

        let mut minor: i32 = 0;
        sys::cuDeviceGetAttribute(
            &mut minor,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        );

        (major, minor)
    })
}

/// Master switch for routing production traffic to flash v3.
///
/// Currently `false`: v3 has five defects found by source reading, NONE of
/// which anyone has ever executed, because no machine available to this
/// project is SM 90 and no CI runs these kernels.
///
/// 1. dK/dV attributed to the wrong K positions in the fp16/bf16/fp8
///    backward — accumulators keyed by the thread's Q row, stored at K
///    position `k_start + tid` with a plain `=`, so Q blocks overwrote each
///    other. Fixed, unvalidated.
/// 2. The fp32 backward `atomicAdd`s dK/dV into buffers allocated here with
///    `Tensor::empty` — accumulation into uninitialised memory. Fixed,
///    unvalidated.
/// 3. TOP-LEFT causal convention where the CPU reference, flash v2, varlen
///    and paged all use BOTTOM-RIGHT. Fixed, unvalidated.
/// 4. `float O_local[128]` in the forward is indexed
///    `(i - warp_q_start) * HEAD_DIM + d`, reaching 1023 at `HEAD_DIM = 64`
///    and 2047 at `HEAD_DIM = 128` — an 8x to 16x stack overrun. NOT fixed.
/// 5. The forward's `warp_reduce_max` uses a full `__shfl_down_sync` mask
///    while lanes have diverged on the causal `continue` (undefined
///    behaviour), and all 32 lanes then update the same `m_local`/`l_local`/
///    `O_local` accumulators for one row, racing. NOT fixed.
///
/// Defects 4 and 5 are outstanding memory-safety/UB defects in the forward.
/// Flash v2 runs on SM 90 through the same JIT path and is covered by the
/// parity suite, so v2 is strictly the better route until v3 is executed.
///
/// To re-enable: on an SM 90 device run `tests/flash_v3_bwd_parity_cuda.rs`
/// and the other flash parity tests with `--nocapture`, confirm they PASS
/// rather than print their `FLASH_V3_BWD_SKIPPED` banner, then flip this to
/// `true`.
const V3_DISPATCH_ENABLED: bool = false;

/// THE single decision point for whether `flash.rs` uses flash v3.
///
/// Both the forward and the backward dispatch site consult this and nothing
/// else. They must agree: a v3 forward paired with a v2 backward would mix
/// causal conventions.
pub fn dispatch_enabled(
    client: &CudaClient,
    device: &<CudaRuntime as numr::runtime::Runtime>::Device,
) -> bool {
    if !V3_DISPATCH_ENABLED {
        return false;
    }
    is_hopper(client, device)
}

/// Check if the current device supports SM 90+ (Hopper).
pub fn is_hopper(
    _client: &CudaClient,
    _device: &<CudaRuntime as numr::runtime::Runtime>::Device,
) -> bool {
    let (major, _minor) = get_compute_capability();
    major >= 9
}

/// Flash v3 forward — supports head_dim 64 and 128 only.
/// Returns None if head_dim is not supported by v3 (caller falls back to v2).
#[allow(clippy::too_many_arguments)]
pub fn flash_v3_fwd(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    batch_size: usize,
    num_heads: usize,
    seq_len_q: usize,
    seq_len_k: usize,
    head_dim: usize,
    causal: bool,
) -> Result<Option<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)>> {
    // v3 only supports head_dim 64 and 128
    if head_dim != 64 && head_dim != 128 {
        return Ok(None);
    }

    let dtype = q.dtype();
    let dtype_suffix = match dtype {
        DType::F32 => "fp32",
        DType::F16 => "fp16",
        DType::BF16 => "bf16",
        _ => return Ok(None), // FP8 handled separately
    };

    // v3 kernel naming: flash_attention_v3_fwd_{head_dim} for FP32,
    // flash_attention_v3_fwd_{head_dim}_{dtype} for FP16/BF16
    let kernel_name = if dtype == DType::F32 {
        format!("flash_attention_v3_fwd_{}", head_dim)
    } else {
        format!("flash_attention_v3_fwd_{}_{}", head_dim, dtype_suffix)
    };

    let device = q.device();
    let device_index = device.id();

    // Try to load v3 module — if it fails (no Hopper PTX), return None
    let module = match kernels::get_or_load_module(client.context(), device_index, FLASH_V3_MODULE)
    {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let func = match kernels::get_kernel_function(&module, &kernel_name) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };

    let output =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q, head_dim], dtype, device)?;
    let lse =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q], DType::F32, device)?;

    // v3 uses BLOCK_M=128, BLOCK_N=128, 8 warps (256 threads)
    // Double-buffered shared memory: 2 × (Q + K + V) tiles
    let dtype_size = dtype.size_in_bytes();
    let smem_size = 2 * (128 * head_dim + 128 * head_dim + 128 * head_dim) * dtype_size;
    set_smem_attribute(&func, smem_size)?;

    let cfg = LaunchConfig {
        grid_dim: (
            (batch_size * num_heads) as u32,
            seq_len_q.div_ceil(128) as u32,
            1,
        ),
        block_dim: (256, 1, 1),
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
        let mut builder = client.stream().launch_builder(&func);
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
            reason: format!("Flash Attention v3 fwd kernel launch failed: {:?}", e),
        })?;
    }

    Ok(Some((output, lse)))
}

/// Return type of `flash_v3_bwd`: gradients for q, k, v.
type BwdGrads = Option<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)>;

/// Flash v3 backward — supports head_dim 64 and 128 only.
/// Returns None if head_dim is not supported.
#[allow(clippy::too_many_arguments)]
pub fn flash_v3_bwd(
    client: &CudaClient,
    dout: &Tensor<CudaRuntime>,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    lse: &Tensor<CudaRuntime>,
    batch_size: usize,
    num_heads: usize,
    seq_len_q: usize,
    seq_len_k: usize,
    head_dim: usize,
    causal: bool,
) -> Result<BwdGrads> {
    if head_dim != 64 && head_dim != 128 {
        return Ok(None);
    }

    let dtype = q.dtype();
    let dtype_suffix = match dtype {
        DType::F32 => "",
        DType::F16 => "_fp16",
        DType::BF16 => "_bf16",
        _ => return Ok(None),
    };

    let device = q.device();
    let device_index = device.id();

    let module =
        match kernels::get_or_load_module(client.context(), device_index, FLASH_V3_BWD_MODULE) {
            Ok(m) => m,
            Err(_) => return Ok(None),
        };

    // Allocate gradient tensors (dQ zeroed for atomicAdd)
    let dq =
        Tensor::<CudaRuntime>::zeros(&[batch_size, num_heads, seq_len_q, head_dim], dtype, device)?;
    let dk =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_k, head_dim], dtype, device)?;
    let dv =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_k, head_dim], dtype, device)?;

    // Step 1: Preprocessing — D = rowsum(dO ⊙ O)
    let d_buf =
        Tensor::<CudaRuntime>::empty(&[batch_size, num_heads, seq_len_q], DType::F32, device)?;

    {
        let preprocess_name = format!(
            "flash_attention_v3_preprocess_bwd{}_{}",
            dtype_suffix, head_dim
        );
        let func = match kernels::get_kernel_function(&module, &preprocess_name) {
            Ok(f) => f,
            Err(_) => return Ok(None),
        };

        let block_size = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (
                (batch_size * num_heads) as u32,
                (seq_len_q as u32).div_ceil(block_size),
                1,
            ),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let dout_ptr = dout.ptr();
        let out_ptr = output.ptr();
        let d_ptr = d_buf.ptr();
        let batch_i32 = batch_size as i32;
        let nh_i32 = num_heads as i32;
        let sq_i32 = seq_len_q as i32;

        unsafe {
            let mut builder = client.stream().launch_builder(&func);
            builder.arg(&dout_ptr);
            builder.arg(&out_ptr);
            builder.arg(&d_ptr);
            builder.arg(&batch_i32);
            builder.arg(&nh_i32);
            builder.arg(&sq_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("Flash v3 bwd preprocess failed: {:?}", e),
            })?;
        }
    }

    // Step 2: Main backward
    {
        let bwd_name = format!("flash_attention_v3_bwd{}_{}", dtype_suffix, head_dim);
        let func = match kernels::get_kernel_function(&module, &bwd_name) {
            Ok(f) => f,
            Err(_) => return Ok(None),
        };

        // Block config: head_dim 64 → BLOCK_M=32, BLOCK_N=64; head_dim 128 → BLOCK_M=16, BLOCK_N=32
        let (block_m, block_n) = if head_dim == 64 { (32, 64) } else { (16, 32) };
        let dtype_size = dtype.size_in_bytes();
        let smem_size = (2 * block_n * head_dim + 2 * block_m * head_dim) * dtype_size;
        set_smem_attribute(&func, smem_size)?;

        let cfg = LaunchConfig {
            grid_dim: (
                (batch_size * num_heads) as u32,
                seq_len_k.div_ceil(block_n) as u32,
                1,
            ),
            block_dim: (block_n as u32, 1, 1),
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
        let batch_i32 = batch_size as i32;
        let nh_i32 = num_heads as i32;
        let sq_i32 = seq_len_q as i32;
        let sk_i32 = seq_len_k as i32;
        let causal_i32 = if causal { 1i32 } else { 0i32 };

        unsafe {
            let mut builder = client.stream().launch_builder(&func);
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
                reason: format!("Flash v3 bwd kernel launch failed: {:?}", e),
            })?;
        }
    }

    Ok(Some((dq, dk, dv)))
}
