//! Launcher for the fused IMROPE kernel `mrope_interleaved_f32`.
//!
//! The kernel reads `x` through its own element strides along `[B, S, H]`
//! and requires unit stride along `D`; any other layout is made dense
//! first. The output is a fresh dense `[B, S, H, D]`, the layout the
//! composed op's final `cat` produces. Tables, positions and selector are
//! read from `ptr()`, so a dense row window of a larger table works in
//! place; a strided one is made dense.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, MROPE_INTERLEAVED_MODULE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

const KERNEL_NAME: &str = "mrope_interleaved_f32";

/// `true` when every operand is in the one layout and dtype the kernel
/// takes: `F32` data, `I32` positions. Anything else runs the composed op.
pub(crate) fn kernel_takes(
    x: &Tensor<CudaRuntime>,
    cos_cache: &Tensor<CudaRuntime>,
    sin_cache: &Tensor<CudaRuntime>,
    positions: &Tensor<CudaRuntime>,
    selector: &Tensor<CudaRuntime>,
) -> bool {
    [x, cos_cache, sin_cache, selector]
        .iter()
        .all(|t| t.dtype() == DType::F32)
        && positions.dtype() == DType::I32
}

/// Element strides of `x` along `[B, S, H]` as the kernel reads them; a
/// size-1 dim reads at stride 0.
fn source_strides(x: &Tensor<CudaRuntime>) -> Result<Option<[i32; 3]>> {
    let (shape, strides) = (x.shape(), x.strides());
    if shape[3] > 1 && strides[3] != 1 {
        return Ok(None);
    }
    let mut out = [0i32; 3];
    for (dim, slot) in out.iter_mut().enumerate() {
        if shape[dim] <= 1 {
            continue;
        }
        let stride = strides[dim];
        if stride < 0 {
            return Ok(None);
        }
        *slot = i32::try_from(stride).map_err(|_| Error::InvalidArgument {
            arg: "x",
            reason: format!("stride {stride} exceeds the kernel's i32 index range"),
        })?;
    }
    Ok(Some(out))
}

/// A dense view of `t`: itself when its strides are already row-major
/// (an offset is fine, the kernel reads from `ptr()`), else a copy.
fn dense(t: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    if t.is_contiguous() {
        Ok(t.clone())
    } else {
        t.contiguous().map_err(Error::Numr)
    }
}

/// Validate shapes, then launch one `mrope_interleaved_f32` over a fresh
/// dense output. Same shape rules as the composed op.
///
/// # Errors
///
/// `InvalidArgument` when a shape disagrees with the layout contract or a
/// size exceeds the kernel's `i32` index range; `KernelError` when the
/// launch fails.
pub(crate) fn mrope_interleaved_f32(
    client: &CudaClient,
    x: &Tensor<CudaRuntime>,
    cos_cache: &Tensor<CudaRuntime>,
    sin_cache: &Tensor<CudaRuntime>,
    positions: &Tensor<CudaRuntime>,
    selector: &Tensor<CudaRuntime>,
    n_rot: usize,
) -> Result<Tensor<CudaRuntime>> {
    let x_shape = x.shape().to_vec();
    if x_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!(
                "expected 4D [batch, seq, heads, head_dim], got {}D",
                x_shape.len()
            ),
        });
    }
    let (batch, seq, heads, head_dim) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    if n_rot == 0 || !n_rot.is_multiple_of(2) || n_rot > head_dim {
        return Err(Error::InvalidArgument {
            arg: "n_rot",
            reason: format!("n_rot={n_rot} must be even, nonzero and at most head_dim={head_dim}"),
        });
    }
    let half_rot = n_rot / 2;
    for (arg, cache) in [("cos_cache", cos_cache), ("sin_cache", sin_cache)] {
        let shape = cache.shape();
        if shape.len() != 2 || shape[1] != half_rot {
            return Err(Error::InvalidArgument {
                arg,
                reason: format!("expected [max_pos, {half_rot}], got {shape:?}"),
            });
        }
    }
    if cos_cache.shape()[0] != sin_cache.shape()[0] {
        return Err(Error::InvalidArgument {
            arg: "sin_cache",
            reason: format!(
                "cos_cache has {} rows, sin_cache has {}",
                cos_cache.shape()[0],
                sin_cache.shape()[0]
            ),
        });
    }
    if positions.shape() != [4, seq] {
        return Err(Error::InvalidArgument {
            arg: "positions",
            reason: format!("expected [4, seq={seq}], got {:?}", positions.shape()),
        });
    }
    if selector.shape() != [4, 1, half_rot] {
        return Err(Error::InvalidArgument {
            arg: "selector",
            reason: format!(
                "expected [4, 1, half_rot={half_rot}], got {:?}",
                selector.shape()
            ),
        });
    }
    let max_pos = cos_cache.shape()[0];
    let work = half_rot + (head_dim - n_rot);
    let total = batch * seq * heads * work;
    if batch * seq * heads * head_dim > i32::MAX as usize || max_pos * half_rot > i32::MAX as usize
    {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!(
                "{x_shape:?} with a [{max_pos}, {half_rot}] table exceeds the kernel's i32 index \
                 range"
            ),
        });
    }

    let device = x.device().clone();
    let output = Tensor::<CudaRuntime>::empty(&x_shape, DType::F32, &device)?;
    if total == 0 {
        return Ok(output);
    }

    // A view the kernel cannot address is copied dense once, then read
    // with row-major strides.
    let x_dense;
    let (x_src, [sb, ss, sh]) = match source_strides(x)? {
        Some(strides) => (x, strides),
        None => {
            x_dense = x.contiguous()?;
            let strides = source_strides(&x_dense)?.ok_or_else(|| Error::InvalidArgument {
                arg: "x",
                reason: format!("{x_shape:?} is not addressable after contiguous()"),
            })?;
            (&x_dense, strides)
        }
    };
    let cos = dense(cos_cache)?;
    let sin = dense(sin_cache)?;
    let pos = dense(positions)?;
    let sel = dense(selector)?;

    let module =
        kernels::get_or_load_module(client.context(), device.id(), MROPE_INTERLEAVED_MODULE)?;
    let func = kernels::get_kernel_function(&module, KERNEL_NAME)?;

    let block_size = 256u32;
    let cfg = LaunchConfig {
        grid_dim: ((total as u32).div_ceil(block_size), 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let x_ptr = x_src.ptr();
    let cos_ptr = cos.ptr();
    let sin_ptr = sin.ptr();
    let pos_ptr = pos.ptr();
    let sel_ptr = sel.ptr();
    let out_ptr = output.ptr();
    let batch_i32 = batch as i32;
    let seq_i32 = seq as i32;
    let heads_i32 = heads as i32;
    let head_dim_i32 = head_dim as i32;
    let n_rot_i32 = n_rot as i32;
    let max_pos_i32 = max_pos as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&x_ptr);
        builder.arg(&cos_ptr);
        builder.arg(&sin_ptr);
        builder.arg(&pos_ptr);
        builder.arg(&sel_ptr);
        builder.arg(&out_ptr);
        builder.arg(&batch_i32);
        builder.arg(&seq_i32);
        builder.arg(&heads_i32);
        builder.arg(&head_dim_i32);
        builder.arg(&n_rot_i32);
        builder.arg(&max_pos_i32);
        builder.arg(&sb);
        builder.arg(&ss);
        builder.arg(&sh);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("{KERNEL_NAME} launch failed for x {x_shape:?}: {e:?}"),
        })?;
    }
    Ok(output)
}
