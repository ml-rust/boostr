//! CUDA implementation of FusedQkvOps
//!
//! The GEMM always runs through numr's CUDA matmul. For F32/F64 the
//! bias/split/transpose epilogue runs as a single fused kernel launch
//! (see `fused_qkv_launch`); every other dtype (F16, BF16, ...) falls back
//! to the generic elementwise path, which has no dtype restriction.
//! `fused_output_projection_residual_bwd` has no fused kernel and always
//! uses the generic path.

use crate::error::{Error, Result};
use crate::ops::impl_generic::attention::fused_qkv::{
    fused_output_projection_residual_bwd_impl, fused_output_projection_residual_impl,
    fused_qkv_projection_bwd_impl, fused_qkv_projection_impl,
};
use crate::ops::traits::attention::fused_qkv::FusedQkvOps;
use numr::ops::{MatmulOps, ReduceOps};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::fused_qkv_launch::{
    fused_qkv_dtype_suffix, launch_bias_split, launch_output_bias_residual, launch_qkv_concat,
};

impl FusedQkvOps<CudaRuntime> for CudaClient {
    fn fused_qkv_projection(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: Option<&Tensor<CudaRuntime>>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let suffix = match fused_qkv_dtype_suffix(input.dtype()) {
            Some(s) => s,
            None => {
                return fused_qkv_projection_impl(
                    self,
                    input,
                    weight,
                    bias,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                );
            }
        };

        let input_shape = input.shape();
        if input_shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: format!("expected 3D [B, S, H], got {}D", input_shape.len()),
            });
        }
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let hidden_dim = input_shape[2];

        let hq = num_heads * head_dim;
        let hkv = num_kv_heads * head_dim;
        let total_proj = hq + 2 * hkv;

        let weight_shape = weight.shape();
        if weight_shape != [total_proj, hidden_dim] {
            return Err(Error::InvalidArgument {
                arg: "weight",
                reason: format!(
                    "expected [{}, {}], got {:?}",
                    total_proj, hidden_dim, weight_shape
                ),
            });
        }
        if let Some(b) = bias
            && b.shape() != [total_proj]
        {
            return Err(Error::InvalidArgument {
                arg: "bias",
                reason: format!("expected [{}], got {:?}", total_proj, b.shape()),
            });
        }

        // qkv: [B*S, total_proj], laid out [Hq | Hkv | Hkv]
        let input_2d = input.reshape(&[batch_size * seq_len, hidden_dim])?;
        let weight_t = weight.transpose(-2, -1)?;
        let qkv = self.matmul(&input_2d, &weight_t).map_err(Error::Numr)?;

        launch_bias_split(
            self,
            &qkv,
            bias,
            suffix,
            batch_size,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            total_proj,
        )
    }

    fn fused_output_projection_residual(
        &self,
        attn_out: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: Option<&Tensor<CudaRuntime>>,
        residual: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let suffix = match fused_qkv_dtype_suffix(attn_out.dtype()) {
            Some(s) => s,
            None => {
                return fused_output_projection_residual_impl(
                    self, attn_out, weight, bias, residual,
                );
            }
        };

        let attn_shape = attn_out.shape();
        if attn_shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "attn_out",
                reason: format!("expected 3D [B, S, Hq*D], got {}D", attn_shape.len()),
            });
        }
        let batch_size = attn_shape[0];
        let seq_len = attn_shape[1];
        let proj_dim = attn_shape[2];

        let weight_shape = weight.shape();
        if weight_shape.len() != 2 || weight_shape[1] != proj_dim {
            return Err(Error::InvalidArgument {
                arg: "weight",
                reason: format!("expected [H, {}], got {:?}", proj_dim, weight_shape),
            });
        }
        let hidden_dim = weight_shape[0];

        if residual.shape() != [batch_size, seq_len, hidden_dim] {
            return Err(Error::InvalidArgument {
                arg: "residual",
                reason: format!(
                    "expected [{}, {}, {}], got {:?}",
                    batch_size,
                    seq_len,
                    hidden_dim,
                    residual.shape()
                ),
            });
        }
        if let Some(b) = bias
            && b.shape() != [hidden_dim]
        {
            return Err(Error::InvalidArgument {
                arg: "bias",
                reason: format!("expected [{}], got {:?}", hidden_dim, b.shape()),
            });
        }

        let attn_2d = attn_out.reshape(&[batch_size * seq_len, proj_dim])?;
        let weight_t = weight.transpose(-2, -1)?;
        let proj = self.matmul(&attn_2d, &weight_t).map_err(Error::Numr)?;
        // contiguous() first: unlike `add`, the fused kernel needs a flat,
        // contiguous residual buffer, and callers may hand in a view.
        let residual_2d = residual
            .contiguous()?
            .reshape(&[batch_size * seq_len, hidden_dim])?;

        let output_2d = launch_output_bias_residual(
            self,
            &proj,
            bias,
            &residual_2d,
            suffix,
            batch_size * seq_len * hidden_dim,
            hidden_dim,
        )?;
        Ok(output_2d.reshape(&[batch_size, seq_len, hidden_dim])?)
    }

    fn fused_qkv_projection_bwd(
        &self,
        dq: &Tensor<CudaRuntime>,
        dk: &Tensor<CudaRuntime>,
        dv: &Tensor<CudaRuntime>,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        has_bias: bool,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Option<Tensor<CudaRuntime>>,
    )> {
        let suffix = match fused_qkv_dtype_suffix(dq.dtype()) {
            Some(s) => s,
            None => {
                return fused_qkv_projection_bwd_impl(
                    self,
                    dq,
                    dk,
                    dv,
                    input,
                    weight,
                    has_bias,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                );
            }
        };

        let input_shape = input.shape();
        if input_shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: format!("expected 3D [B, S, H], got {}D", input_shape.len()),
            });
        }
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let hidden_dim = input_shape[2];

        let hq = num_heads * head_dim;
        let hkv = num_kv_heads * head_dim;
        let total_proj = hq + 2 * hkv;

        if dq.shape() != [batch_size, num_heads, seq_len, head_dim] {
            return Err(Error::InvalidArgument {
                arg: "dq",
                reason: format!(
                    "expected [{}, {}, {}, {}], got {:?}",
                    batch_size,
                    num_heads,
                    seq_len,
                    head_dim,
                    dq.shape()
                ),
            });
        }
        if dk.shape() != [batch_size, num_kv_heads, seq_len, head_dim]
            || dv.shape() != [batch_size, num_kv_heads, seq_len, head_dim]
        {
            return Err(Error::InvalidArgument {
                arg: "dk/dv",
                reason: format!(
                    "expected [{}, {}, {}, {}], got dk: {:?}, dv: {:?}",
                    batch_size,
                    num_kv_heads,
                    seq_len,
                    head_dim,
                    dk.shape(),
                    dv.shape()
                ),
            });
        }

        // fused_qkv_concat reads dq/dk/dv directly in [B, heads, S, D]
        // layout, so it fuses away the transpose(1,2) the generic path does.
        let dq_c = dq.contiguous()?;
        let dk_c = dk.contiguous()?;
        let dv_c = dv.contiguous()?;

        let d_qkv = launch_qkv_concat(
            self,
            &dq_c,
            &dk_c,
            &dv_c,
            suffix,
            batch_size,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            total_proj,
        )?;

        // d_input = d_qkv @ weight -> [B*S, H]
        let d_input_2d = self.matmul(&d_qkv, weight).map_err(Error::Numr)?;
        let d_input = d_input_2d.reshape(&[batch_size, seq_len, hidden_dim])?;

        // d_weight = d_qkv.T @ input_2d -> [total_proj, H]
        let input_2d = input.reshape(&[batch_size * seq_len, hidden_dim])?;
        let d_qkv_t = d_qkv.transpose(-2, -1)?.contiguous()?;
        let d_weight = self.matmul(&d_qkv_t, &input_2d).map_err(Error::Numr)?;

        let d_bias = if has_bias {
            Some(self.sum(&d_qkv, &[0], false).map_err(Error::Numr)?)
        } else {
            None
        };

        Ok((d_input, d_weight, d_bias))
    }

    fn fused_output_projection_residual_bwd(
        &self,
        d_output: &Tensor<CudaRuntime>,
        attn_out: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        has_bias: bool,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Option<Tensor<CudaRuntime>>,
        Tensor<CudaRuntime>,
    )> {
        // No fused kernel covers this backward pass; the generic path is
        // three matmuls plus a sum.
        fused_output_projection_residual_bwd_impl(self, d_output, attn_out, weight, has_bias)
    }
}
