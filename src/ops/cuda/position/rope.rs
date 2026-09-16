//! CUDA implementation of RoPEOps — fused kernel dispatch.
//!
//! The kernels read `x` through its own element strides along `[B, H, S]`
//! and require unit stride along `D`; the output is always a fresh dense
//! `[B, H, S, D]`. So a `[B, S, H, D]`-contiguous projection viewed as
//! `[B, H, S, D]` (the `permute([0, 2, 1, 3])` every attention block does)
//! goes straight in with no `contiguous()` copy, and the attention path
//! still receives the dense head-major tensor it wants. Any view the
//! kernel cannot address — a non-unit `D` stride, a negative stride — is
//! rejected up front instead of being rotated wrong.

use crate::error::{Error, Result};
use crate::ops::autograd_rope::{RopeVariant, attach_rope_backward};
use crate::ops::cuda::kernels::{self, ROPE_INTERLEAVED_MODULE, ROPE_MODULE, ROPE_YARN_MODULE};
use crate::ops::traits::RoPEOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::runtime::{Device, Runtime};
use numr::tensor::Tensor;

/// Element strides of `x` along `[B, H, S]` as the kernel reads them.
/// A size-1 dim reads at stride 0, whatever its stored stride is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RopeSrcStrides {
    pub b: i32,
    pub h: i32,
    pub s: i32,
}

/// Validated inputs: shape, dtype, the caches narrowed to `seq_len` and cast
/// to `x`'s dtype, and the strides the kernel reads `x` through.
struct RopeInputs {
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    dtype: DType,
    cos: Tensor<CudaRuntime>,
    sin: Tensor<CudaRuntime>,
    src: RopeSrcStrides,
    device: CudaDevice,
}

/// Strides the kernel reads a `[B, H, S, D]` view through, or an error
/// naming the layout it cannot address.
pub(crate) fn source_strides(shape: &[usize], strides: &[isize]) -> Result<RopeSrcStrides> {
    let reject = |why: &str| Error::InvalidArgument {
        arg: "x",
        reason: format!(
            "unsupported layout for the fused RoPE kernel: shape {shape:?} strides {strides:?} \
             ({why}); pass a dense [B, H, S, D] tensor or a [B, S, H, D]-contiguous one viewed \
             as [B, H, S, D], or call contiguous() first"
        ),
    };
    if shape.len() != 4 || strides.len() != 4 {
        return Err(reject("expected rank 4"));
    }
    if shape[3] > 1 && strides[3] != 1 {
        return Err(reject("the head dimension must have unit stride"));
    }
    let outer = |dim: usize| -> Result<i32> {
        if shape[dim] <= 1 {
            return Ok(0);
        }
        let stride = strides[dim];
        if stride < 0 {
            return Err(reject("negative strides are not readable"));
        }
        i32::try_from(stride).map_err(|_| reject("stride exceeds the kernel's i32 index range"))
    };
    Ok(RopeSrcStrides {
        b: outer(0)?,
        h: outer(1)?,
        s: outer(2)?,
    })
}

/// Validate shapes and layout, narrow and cast the caches.
fn validate_rope_inputs(
    x: &Var<CudaRuntime>,
    cos_cache: &Var<CudaRuntime>,
    sin_cache: &Var<CudaRuntime>,
) -> Result<RopeInputs> {
    let x_tensor = x.tensor();
    let cos_tensor = cos_cache.tensor();
    let sin_tensor = sin_cache.tensor();

    let x_shape = x_tensor.shape();
    if x_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("expected 4D [B, H, S, D], got {:?}", x_shape),
        });
    }

    let batch_size = x_shape[0];
    let num_heads = x_shape[1];
    let seq_len = x_shape[2];
    let head_dim = x_shape[3];

    if !head_dim.is_multiple_of(2) {
        return Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!("head_dim must be even, got {}", head_dim),
        });
    }
    let src = source_strides(x_shape, x_tensor.strides())?;
    if batch_size * num_heads * seq_len * head_dim > i32::MAX as usize {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("{x_shape:?} exceeds the kernel's i32 index range"),
        });
    }

    let cos_shape = cos_tensor.shape();
    let sin_shape = sin_tensor.shape();

    if cos_shape.len() != 2 || sin_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "cache",
            reason: format!(
                "expected 2D [S, D/2], got cos: {:?}, sin: {:?}",
                cos_shape, sin_shape
            ),
        });
    }

    if cos_shape[1] != head_dim / 2 || sin_shape[1] != head_dim / 2 {
        return Err(Error::InvalidArgument {
            arg: "cache",
            reason: format!(
                "cache second dimension should be {}, got cos: {}, sin: {}",
                head_dim / 2,
                cos_shape[1],
                sin_shape[1]
            ),
        });
    }
    if cos_shape[0] < seq_len || sin_shape[0] < seq_len {
        return Err(Error::InvalidArgument {
            arg: "cache",
            reason: format!(
                "cache covers {} positions (cos) / {} (sin), x has {seq_len}",
                cos_shape[0], sin_shape[0]
            ),
        });
    }

    let dtype = x_tensor.dtype();
    let device = x_tensor.device().clone();
    // Narrow to `seq_len`, cast to `x`'s dtype when the table was built at
    // another one, and make dense: the kernel indexes the caches as
    // row-major `[S, D/2]` from `ptr()`, which already carries a view's
    // offset, so a row window of a dense table is read in place.
    // `Tensor::contiguous` would copy any offset view, hence the explicit
    // layout check. Each step is a no-op when nothing changes.
    let prepare = |cache: &Tensor<CudaRuntime>| -> Result<Tensor<CudaRuntime>> {
        let narrowed = if cache.shape()[0] > seq_len {
            cache.narrow(0, 0, seq_len)?
        } else {
            cache.clone()
        };
        let matched = if narrowed.dtype() != dtype {
            let client = CudaRuntime::default_client(&device);
            client.cast(&narrowed, dtype)?
        } else {
            narrowed
        };
        if matched.is_contiguous() {
            Ok(matched)
        } else {
            Ok(matched.contiguous()?)
        }
    };
    let cos = prepare(cos_tensor)?;
    let sin = prepare(sin_tensor)?;

    Ok(RopeInputs {
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        dtype,
        cos,
        sin,
        src,
        device,
    })
}

fn select_kernel_name(prefix: &str, dtype: DType) -> Result<&'static str> {
    match (prefix, dtype) {
        ("rope_apply", DType::F32) => Ok("rope_apply_f32"),
        ("rope_apply", DType::F16) => Ok("rope_apply_f16"),
        ("rope_apply", DType::BF16) => Ok("rope_apply_bf16"),
        ("rope_interleaved", DType::F32) => Ok("rope_interleaved_f32"),
        ("rope_interleaved", DType::F16) => Ok("rope_interleaved_f16"),
        ("rope_interleaved", DType::BF16) => Ok("rope_interleaved_bf16"),
        ("rope_yarn", DType::F32) => Ok("rope_yarn_f32"),
        ("rope_yarn", DType::F16) => Ok("rope_yarn_f16"),
        ("rope_yarn", DType::BF16) => Ok("rope_yarn_bf16"),
        _ => Err(Error::KernelError {
            reason: format!("RoPE {}: unsupported dtype {:?}", prefix, dtype),
        }),
    }
}

/// Validate, launch one fused RoPE kernel over a fresh dense output, and
/// attach the backward node. `threads_per_pair` is 2 for the split-half
/// kernels (one thread per element) and 1 for interleaved (one per pair).
fn run_rope(
    client: &CudaClient,
    x: &Var<CudaRuntime>,
    cos_cache: &Var<CudaRuntime>,
    sin_cache: &Var<CudaRuntime>,
    variant: RopeVariant,
) -> Result<Var<CudaRuntime>> {
    let (module_name, prefix, threads_per_pair): (&'static str, &str, usize) = match variant {
        RopeVariant::Standard => (ROPE_MODULE, "rope_apply", 2),
        RopeVariant::Interleaved => (ROPE_INTERLEAVED_MODULE, "rope_interleaved", 1),
        RopeVariant::Yarn { .. } => (ROPE_YARN_MODULE, "rope_yarn", 2),
    };
    let inputs = validate_rope_inputs(x, cos_cache, sin_cache)?;
    let kernel_name = select_kernel_name(prefix, inputs.dtype)?;
    let output = Tensor::<CudaRuntime>::empty(x.tensor().shape(), inputs.dtype, &inputs.device)?;

    let module = kernels::get_or_load_module(client.context(), inputs.device.id(), module_name)?;
    let func = kernels::get_kernel_function(&module, kernel_name)?;

    let pairs = inputs.batch_size * inputs.num_heads * inputs.seq_len * (inputs.head_dim / 2);
    let total_threads = (pairs * threads_per_pair) as u32;
    let block_size = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (total_threads.div_ceil(block_size), 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let x_ptr = x.tensor().ptr();
    let cos_ptr = inputs.cos.ptr();
    let sin_ptr = inputs.sin.ptr();
    let out_ptr = output.ptr();
    let b_i32 = inputs.batch_size as i32;
    let nh_i32 = inputs.num_heads as i32;
    let sl_i32 = inputs.seq_len as i32;
    let hd_i32 = inputs.head_dim as i32;
    let RopeSrcStrides {
        b: sb,
        h: sh,
        s: ss,
    } = inputs.src;
    // Hoisted out of the `if`: the launch builder borrows every argument
    // until `launch`.
    let attn_scale = match variant {
        RopeVariant::Yarn { attn_scale } => attn_scale,
        _ => 1.0f32,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&x_ptr);
        builder.arg(&cos_ptr);
        builder.arg(&sin_ptr);
        builder.arg(&out_ptr);
        builder.arg(&b_i32);
        builder.arg(&nh_i32);
        builder.arg(&sl_i32);
        builder.arg(&hd_i32);
        builder.arg(&sb);
        builder.arg(&sh);
        builder.arg(&ss);
        if matches!(variant, RopeVariant::Yarn { .. }) {
            builder.arg(&attn_scale);
        }
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("RoPE kernel {kernel_name} launch failed: {:?}", e),
        })?;
    }

    attach_rope_backward(x, output, &inputs.cos, &inputs.sin, variant)
}

impl RoPEOps<CudaRuntime> for CudaClient {
    fn apply_rope(
        &self,
        x: &Var<CudaRuntime>,
        cos_cache: &Var<CudaRuntime>,
        sin_cache: &Var<CudaRuntime>,
    ) -> Result<Var<CudaRuntime>> {
        run_rope(self, x, cos_cache, sin_cache, RopeVariant::Standard)
    }

    fn apply_rope_interleaved(
        &self,
        x: &Var<CudaRuntime>,
        cos_cache: &Var<CudaRuntime>,
        sin_cache: &Var<CudaRuntime>,
    ) -> Result<Var<CudaRuntime>> {
        run_rope(self, x, cos_cache, sin_cache, RopeVariant::Interleaved)
    }

    fn apply_rope_yarn(
        &self,
        x: &Var<CudaRuntime>,
        cos_cache: &Var<CudaRuntime>,
        sin_cache: &Var<CudaRuntime>,
        attn_scale: f32,
    ) -> Result<Var<CudaRuntime>> {
        run_rope(
            self,
            x,
            cos_cache,
            sin_cache,
            RopeVariant::Yarn { attn_scale },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_layout_reads_row_major_strides() {
        let src = source_strides(&[2, 3, 5, 8], &[120, 40, 8, 1]).unwrap();
        assert_eq!(
            src,
            RopeSrcStrides {
                b: 120,
                h: 40,
                s: 8
            }
        );
    }

    /// `[B, S, H, D]`-contiguous viewed as `[B, H, S, D]`: the head stride is
    /// `D` and the sequence stride is `H * D`.
    #[test]
    fn permuted_seq_major_view_is_readable() {
        let src = source_strides(&[2, 3, 5, 8], &[120, 8, 24, 1]).unwrap();
        assert_eq!(
            src,
            RopeSrcStrides {
                b: 120,
                h: 8,
                s: 24
            }
        );
    }

    #[test]
    fn size_one_dims_read_at_stride_zero() {
        let src = source_strides(&[1, 4, 1, 8], &[999, 8, 777, 1]).unwrap();
        assert_eq!(src, RopeSrcStrides { b: 0, h: 8, s: 0 });
    }

    #[test]
    fn non_unit_head_stride_is_rejected() {
        let err = source_strides(&[1, 2, 4, 8], &[64, 32, 1, 4]).unwrap_err();
        assert!(err.to_string().contains("unit stride"), "{err}");
    }

    #[test]
    fn negative_stride_is_rejected() {
        assert!(source_strides(&[1, 2, 4, 8], &[64, 32, -8, 1]).is_err());
    }
}
