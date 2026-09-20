//! Fused Gated DeltaNet decode step: one launch per call.
//!
//! One block per (batch, head) and column chunk, one thread per state column.
//! The kernel decays the column, takes `k . S1[:, j]`, applies the rank-1
//! delta update and takes `qs . S2[:, j]`, with the state read once and
//! written once. Every column depends only on itself plus the shared `k`,
//! `q`, `exp(g)` and `beta`, so no thread reads another thread's column.
//!
//! Covers F32, `seq == 1` and `S_k` in {32, 64, 128}. `supports` reports
//! whether a call fits; the caller falls back to `gdn_step_impl` otherwise.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, GDN_STEP_MODULE};
use crate::ops::impl_generic::architecture::gated_delta_net::{GdnDims, check_gdn_shapes};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Threads per block; matches `GDN_STEP_BLOCK` in the kernel.
const BLOCK: u32 = 256;

/// Kernel entry point for a supported `S_k`, `None` for any other.
fn kernel_name(s_k: usize) -> Option<&'static str> {
    match s_k {
        32 => Some("gdn_step_f32_sk32"),
        64 => Some("gdn_step_f32_sk64"),
        128 => Some("gdn_step_f32_sk128"),
        _ => None,
    }
}

/// Whether the fused kernel covers this call: F32, one token, `S_k` in
/// {32, 64, 128}.
pub fn supports(dims: &GdnDims, dtype: DType) -> bool {
    dtype == DType::F32 && dims.seq == 1 && kernel_name(dims.s_k).is_some()
}

/// Run the fused step. Shapes follow `GatedDeltaNetOps::gdn_step`.
///
/// Returns `(o: [batch, 1, H, S_v], state: [batch, H, S_k, S_v])`; the
/// state is a new tensor.
pub fn gdn_step_fused(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    g: &Tensor<CudaRuntime>,
    beta: &Tensor<CudaRuntime>,
    state: &Tensor<CudaRuntime>,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let dims = check_gdn_shapes(q, k, v, g, beta, state)?;
    if dims.seq != 1 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("gdn_step_fused takes seq = 1, got shape {:?}", q.shape()),
        });
    }
    for (name, t) in [
        ("q", q),
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("state", state),
    ] {
        if t.dtype() != DType::F32 {
            return Err(Error::InvalidArgument {
                arg: name,
                reason: format!(
                    "gdn_step_fused takes F32, got {:?} for shape {:?}",
                    t.dtype(),
                    t.shape()
                ),
            });
        }
    }
    let name = kernel_name(dims.s_k).ok_or_else(|| Error::InvalidArgument {
        arg: "state",
        reason: format!(
            "gdn_step_fused takes S_k in {{32, 64, 128}}, got state shape {:?}",
            state.shape()
        ),
    })?;
    let GdnDims {
        batch,
        heads,
        s_k,
        s_v,
        ..
    } = dims;
    if s_v > i32::MAX as usize || batch * heads > u32::MAX as usize {
        return Err(Error::InvalidArgument {
            arg: "state",
            reason: format!(
                "gdn_step_fused: state shape {:?} exceeds the launch grid",
                state.shape()
            ),
        });
    }

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let state = state.contiguous()?;

    let device = state.device();
    let o = Tensor::<CudaRuntime>::empty(&[batch, 1, heads, s_v], DType::F32, device)?;
    let state_out = Tensor::<CudaRuntime>::empty(&[batch, heads, s_k, s_v], DType::F32, device)?;

    let module = kernels::get_or_load_module(client.context(), device.id(), GDN_STEP_MODULE)?;
    let func = kernels::get_kernel_function(&module, name)?;

    let cfg = LaunchConfig {
        grid_dim: ((s_v as u32).div_ceil(BLOCK), (batch * heads) as u32, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let q_ptr = q.ptr();
    let k_ptr = k.ptr();
    let v_ptr = v.ptr();
    let g_ptr = g.ptr();
    let beta_ptr = beta.ptr();
    let state_ptr = state.ptr();
    let o_ptr = o.ptr();
    let state_out_ptr = state_out.ptr();
    let s_v_i32 = s_v as i32;
    // Same rounding as `gdn_step_impl`: an f64 scalar cast to F32 at the multiply.
    let q_scale = (1.0 / (s_k as f64).sqrt()) as f32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&k_ptr);
        builder.arg(&v_ptr);
        builder.arg(&g_ptr);
        builder.arg(&beta_ptr);
        builder.arg(&state_ptr);
        builder.arg(&o_ptr);
        builder.arg(&state_out_ptr);
        builder.arg(&s_v_i32);
        builder.arg(&q_scale);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!(
                "{name} launch failed for state shape {:?}: {e:?}",
                state.shape()
            ),
        })?;
    }

    Ok((o, state_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(seq: usize, s_k: usize) -> GdnDims {
        GdnDims {
            batch: 1,
            seq,
            heads: 1,
            s_k,
            s_v: 64,
        }
    }

    #[test]
    fn supports_only_f32_single_token_fixed_sk() {
        assert!(supports(&dims(1, 32), DType::F32));
        assert!(supports(&dims(1, 64), DType::F32));
        assert!(supports(&dims(1, 128), DType::F32));
        assert!(!supports(&dims(1, 96), DType::F32));
        assert!(!supports(&dims(2, 128), DType::F32));
        assert!(!supports(&dims(1, 128), DType::F16));
    }

    #[test]
    fn kernel_name_per_sk() {
        assert_eq!(kernel_name(32), Some("gdn_step_f32_sk32"));
        assert_eq!(kernel_name(64), Some("gdn_step_f32_sk64"));
        assert_eq!(kernel_name(128), Some("gdn_step_f32_sk128"));
        assert_eq!(kernel_name(256), None);
    }
}
