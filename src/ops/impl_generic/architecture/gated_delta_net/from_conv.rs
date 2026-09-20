//! Single-token GDN recurrence from the raw conv output.
//!
//! The per-token chain between the post-SiLU conv output and the delta
//! rule, composed from numr primitives: split q/k/v, L2-normalize q and k,
//! tile the key heads to the value heads, build the gates, then
//! [`GatedDeltaNetOps::gdn_step`]. A backend with a fused kernel for the
//! whole chain skips this function; the fused result must equal it bit for
//! bit.

use super::common::check_gdn_conv_shapes;
use crate::error::{Error, Result};
use crate::ops::traits::architecture::gated_delta_net::GatedDeltaNetOps;
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, NormalizationOps, ShapeOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// `qkv[.., start .. start + len]` reshaped to `shape`.
fn slice_heads<R: Runtime>(
    qkv: &Tensor<R>,
    start: usize,
    len: usize,
    shape: &[usize],
) -> Result<Tensor<R>> {
    Ok(qkv.narrow(2, start, len)?.contiguous()?.reshape(shape)?)
}

/// One token of the gated delta rule from the conv output and the raw gate
/// projections. Shapes follow [`GatedDeltaNetOps::gdn_step_from_conv`].
///
/// ```text
/// q    = l2_normalize(qkv[.., 0 .. key_dim]              as [B, 1, H_k, S_k], eps)
/// k    = l2_normalize(qkv[.., key_dim .. 2 key_dim]      as [B, 1, H_k, S_k], eps)
/// v    = qkv[.., 2 key_dim ..]                           as [B, 1, H_v, S_v]
/// q, k = tile over heads: value head h_v reads key head h_v % H_k
/// beta = sigmoid(beta_raw)
/// g    = ssm_a * softplus(alpha_raw + dt_bias)
/// (o, state_out) = gdn_step(q, k, v, g, beta, state)
/// ```
#[allow(clippy::too_many_arguments)]
pub fn gdn_step_from_conv_impl<R, C>(
    client: &C,
    qkv: &Tensor<R>,
    alpha_raw: &Tensor<R>,
    beta_raw: &Tensor<R>,
    dt_bias: &Tensor<R>,
    ssm_a: &Tensor<R>,
    state: &Tensor<R>,
    h_k: usize,
    key_dim: usize,
    value_dim: usize,
    eps: f32,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: GatedDeltaNetOps<R> + ActivationOps<R> + BinaryOps<R> + NormalizationOps<R> + ShapeOps<R>,
{
    let dims = check_gdn_conv_shapes(
        qkv, alpha_raw, beta_raw, dt_bias, ssm_a, state, h_k, key_dim, value_dim,
    )?;
    if dims.seq != 1 {
        return Err(Error::InvalidArgument {
            arg: "qkv",
            reason: format!(
                "gdn_step_from_conv takes seq = 1, got shape {:?}",
                qkv.shape()
            ),
        });
    }
    let (batch, h_v, s_k, s_v) = (dims.batch, dims.h_v, dims.s_k, dims.s_v);

    let q = slice_heads(qkv, 0, key_dim, &[batch, 1, h_k, s_k])?;
    let k = slice_heads(qkv, key_dim, key_dim, &[batch, 1, h_k, s_k])?;
    let v = slice_heads(qkv, 2 * key_dim, value_dim, &[batch, 1, h_v, s_v])?;

    let q = client.l2_normalize(&q, -1, eps)?;
    let k = client.l2_normalize(&k, -1, eps)?;

    let rep = h_v / h_k;
    let (q, k) = if rep > 1 {
        let tile = [1, 1, rep, 1];
        (client.repeat(&q, &tile)?, client.repeat(&k, &tile)?)
    } else {
        (q, k)
    };

    let beta = client.sigmoid(beta_raw)?;
    let g = client.add(alpha_raw, dt_bias)?;
    let g = client.softplus(&g)?;
    let g = client.mul(&g, ssm_a)?;

    client.gdn_step(&q, &k, &v, &g, &beta, state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn rejects_seq_gt_one() {
        let (client, device) = cpu_setup();
        let qkv = Tensor::<CpuRuntime>::zeros(&[1, 2, 12], DType::F32, &device).unwrap();
        let gate = Tensor::<CpuRuntime>::zeros(&[1, 2, 2], DType::F32, &device).unwrap();
        let per_head = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device).unwrap();
        let state = Tensor::<CpuRuntime>::zeros(&[1, 2, 2, 4], DType::F32, &device).unwrap();
        let r = gdn_step_from_conv_impl(
            &client, &qkv, &gate, &gate, &per_head, &per_head, &state, 1, 2, 8, 1e-6,
        );
        assert!(r.is_err());
    }

    /// Zero state, unit key head tiled to two value heads: `o` for both
    /// value heads reads the same normalized q and k, so with identical v
    /// rows the two outputs agree.
    #[test]
    fn tiled_heads_share_key_head() {
        let (client, device) = cpu_setup();
        // key_dim = 2 (H_k = 1, S_k = 2), value_dim = 4 (H_v = 2, S_v = 2).
        let qkv = Tensor::<CpuRuntime>::from_slice(
            &[3.0f32, 4.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0],
            &[1, 1, 8],
            &device,
        )
        .unwrap();
        let gate = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[1, 1, 2], &device).unwrap();
        let per_head = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device).unwrap();
        let state = Tensor::<CpuRuntime>::zeros(&[1, 2, 2, 2], DType::F32, &device).unwrap();
        let (o, s) = gdn_step_from_conv_impl(
            &client, &qkv, &gate, &gate, &per_head, &per_head, &state, 1, 2, 4, 1e-6,
        )
        .unwrap();
        assert_eq!(o.shape(), &[1, 1, 2, 2]);
        assert_eq!(s.shape(), &[1, 2, 2, 2]);
        let o = o.to_vec::<f32>();
        assert_eq!(&o[..2], &o[2..]);
        // beta = sigmoid(0) = 0.5, k = [1, 0]: S = k ⊗ (0.5 v) = [[0.5, 1], [0, 0]].
        // q = [0.6, 0.8] / sqrt(2): o = 0.6 / sqrt(2) * [0.5, 1].
        let r = 0.6 / 2f32.sqrt();
        assert!((o[0] - 0.5 * r).abs() < 1e-6, "o[0] = {}", o[0]);
        assert!((o[1] - r).abs() < 1e-6, "o[1] = {}", o[1]);
    }
}
