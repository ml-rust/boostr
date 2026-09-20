//! Fused Gated DeltaNet decode step from the conv output: one launch per
//! call for the whole per-token chain.
//!
//! One block per (batch, value head) and column chunk, one thread per state
//! column. The block prologue slices q, k and v out of the conv output,
//! L2-normalizes q and k with numr's reduction order, builds `beta` and the
//! decay from the raw gate projections, then runs the same column update as
//! `gdn_step_fused`. The result equals the primitive chain in
//! `gdn_step_from_conv_impl` bit for bit.
//!
//! Covers F32, `seq == 1` and `S_k` in {32, 64, 128}. `supports_from_conv`
//! reports whether a call fits; the caller falls back to
//! `gdn_step_from_conv_impl` otherwise.

use crate::error::{Error, Result};
use crate::ops::cuda::architecture::gdn_step_support::{launch_config, launch_error, require_f32};
use crate::ops::cuda::kernels::{self, GDN_STEP_MODULE};
use crate::ops::impl_generic::architecture::gated_delta_net::{GdnConvDims, check_gdn_conv_shapes};
use cudarc::driver::PushKernelArg;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Kernel entry point for a supported `S_k`, `None` for any other.
fn kernel_name(s_k: usize) -> Option<&'static str> {
    match s_k {
        32 => Some("gdn_step_v2_f32_sk32"),
        64 => Some("gdn_step_v2_f32_sk64"),
        128 => Some("gdn_step_v2_f32_sk128"),
        _ => None,
    }
}

/// Whether the fused kernel covers this call: F32, one token, `S_k` in
/// {32, 64, 128}.
pub fn supports_from_conv(dims: &GdnConvDims, dtype: DType) -> bool {
    dtype == DType::F32 && dims.seq == 1 && kernel_name(dims.s_k).is_some()
}

/// Run the fused chain. Shapes follow
/// `GatedDeltaNetOps::gdn_step_from_conv`. Non-contiguous operands are
/// copied contiguous first.
///
/// Returns `(o: [batch, 1, H_v, S_v], state: [batch, H_v, S_k, S_v])`; the
/// state is a new tensor.
#[allow(clippy::too_many_arguments)]
pub fn gdn_step_from_conv_fused(
    client: &CudaClient,
    qkv: &Tensor<CudaRuntime>,
    alpha_raw: &Tensor<CudaRuntime>,
    beta_raw: &Tensor<CudaRuntime>,
    dt_bias: &Tensor<CudaRuntime>,
    ssm_a: &Tensor<CudaRuntime>,
    state: &Tensor<CudaRuntime>,
    h_k: usize,
    key_dim: usize,
    value_dim: usize,
    eps: f32,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let dims = check_gdn_conv_shapes(
        qkv, alpha_raw, beta_raw, dt_bias, ssm_a, state, h_k, key_dim, value_dim,
    )?;
    if dims.seq != 1 {
        return Err(Error::InvalidArgument {
            arg: "qkv",
            reason: format!(
                "gdn_step_from_conv_fused takes seq = 1, got shape {:?}",
                qkv.shape()
            ),
        });
    }
    require_f32(
        "gdn_step_from_conv_fused",
        &[
            ("qkv", qkv),
            ("alpha_raw", alpha_raw),
            ("beta_raw", beta_raw),
            ("dt_bias", dt_bias),
            ("ssm_a", ssm_a),
            ("state", state),
        ],
    )?;
    let name = kernel_name(dims.s_k).ok_or_else(|| Error::InvalidArgument {
        arg: "state",
        reason: format!(
            "gdn_step_from_conv_fused takes S_k in {{32, 64, 128}}, got state shape {:?}",
            state.shape()
        ),
    })?;
    let GdnConvDims {
        batch,
        h_v,
        s_k,
        s_v,
        ..
    } = dims;
    let fits = s_v <= i32::MAX as usize
        && h_v <= i32::MAX as usize
        && 2 * key_dim + value_dim <= i32::MAX as usize
        && batch * h_v <= u32::MAX as usize;
    if !fits {
        return Err(Error::InvalidArgument {
            arg: "state",
            reason: format!(
                "gdn_step_from_conv_fused: state shape {:?} exceeds the launch grid",
                state.shape()
            ),
        });
    }

    let qkv = qkv.contiguous()?;
    let alpha_raw = alpha_raw.contiguous()?;
    let beta_raw = beta_raw.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let ssm_a = ssm_a.contiguous()?;
    let state = state.contiguous()?;

    let device = state.device();
    let o = Tensor::<CudaRuntime>::empty(&[batch, 1, h_v, s_v], DType::F32, device)?;
    let state_out = Tensor::<CudaRuntime>::empty(&[batch, h_v, s_k, s_v], DType::F32, device)?;

    let module = kernels::get_or_load_module(client.context(), device.id(), GDN_STEP_MODULE)?;
    let func = kernels::get_kernel_function(&module, name)?;

    let cfg = launch_config(s_v, batch * h_v);

    let qkv_ptr = qkv.ptr();
    let alpha_ptr = alpha_raw.ptr();
    let beta_ptr = beta_raw.ptr();
    let dt_bias_ptr = dt_bias.ptr();
    let ssm_a_ptr = ssm_a.ptr();
    let state_ptr = state.ptr();
    let o_ptr = o.ptr();
    let state_out_ptr = state_out.ptr();
    let s_v_i32 = s_v as i32;
    let h_v_i32 = h_v as i32;
    let h_k_i32 = h_k as i32;
    let key_dim_i32 = key_dim as i32;
    let value_dim_i32 = value_dim as i32;
    // Same rounding as `gdn_step_impl`: an f64 scalar cast to F32 at the multiply.
    let q_scale = (1.0 / (s_k as f64).sqrt()) as f32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&qkv_ptr);
        builder.arg(&alpha_ptr);
        builder.arg(&beta_ptr);
        builder.arg(&dt_bias_ptr);
        builder.arg(&ssm_a_ptr);
        builder.arg(&state_ptr);
        builder.arg(&o_ptr);
        builder.arg(&state_out_ptr);
        builder.arg(&s_v_i32);
        builder.arg(&h_v_i32);
        builder.arg(&h_k_i32);
        builder.arg(&key_dim_i32);
        builder.arg(&value_dim_i32);
        builder.arg(&eps);
        builder.arg(&q_scale);
        builder
            .launch(cfg)
            .map_err(|e| launch_error(name, state.shape(), e))?;
    }

    Ok((o, state_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(seq: usize, s_k: usize) -> GdnConvDims {
        GdnConvDims {
            batch: 1,
            seq,
            h_k: 2,
            h_v: 4,
            s_k,
            s_v: 64,
            key_dim: 2 * s_k,
            value_dim: 256,
        }
    }

    #[test]
    fn supports_only_f32_single_token_fixed_sk() {
        assert!(supports_from_conv(&dims(1, 32), DType::F32));
        assert!(supports_from_conv(&dims(1, 64), DType::F32));
        assert!(supports_from_conv(&dims(1, 128), DType::F32));
        assert!(!supports_from_conv(&dims(1, 96), DType::F32));
        assert!(!supports_from_conv(&dims(2, 128), DType::F32));
        assert!(!supports_from_conv(&dims(1, 128), DType::F16));
    }

    #[test]
    fn kernel_name_per_sk() {
        assert_eq!(kernel_name(32), Some("gdn_step_v2_f32_sk32"));
        assert_eq!(kernel_name(64), Some("gdn_step_v2_f32_sk64"));
        assert_eq!(kernel_name(128), Some("gdn_step_v2_f32_sk128"));
        assert_eq!(kernel_name(256), None);
    }
}
