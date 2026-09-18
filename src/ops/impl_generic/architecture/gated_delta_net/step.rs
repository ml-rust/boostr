//! Single-token GDN recurrence.
//!
//! Port of `build_delta_net_autoregressive` (PrismML llama.cpp fork,
//! `src/models/delta-net-base.cpp`), rewritten for the `[batch, H, S_k, S_v]`
//! state orientation documented on the trait.

use super::common::check_gdn_shapes;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{BinaryOps, MatmulOps, ScalarOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// One token of the gated delta rule.
///
/// ```text
/// q  = q / sqrt(S_k)
/// S  = S * exp(g)
/// sk = k @ S                      [1, S_v]
/// d  = (v - sk) * beta            [1, S_v]
/// S  = S + k^T @ d                [S_k, S_v]
/// o  = q @ S                      [1, S_v]
/// ```
///
/// Returns `(o: [batch, 1, H, S_v], state: [batch, H, S_k, S_v])`.
pub fn gdn_step_impl<R, C>(
    client: &C,
    q: &Tensor<R>,
    k: &Tensor<R>,
    v: &Tensor<R>,
    g: &Tensor<R>,
    beta: &Tensor<R>,
    state: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + ScalarOps<R> + MatmulOps<R> + TensorOps<R>,
{
    let dims = check_gdn_shapes(q, k, v, g, beta, state)?;
    if dims.seq != 1 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("gdn_step takes seq = 1, got {}", dims.seq),
        });
    }
    let (batch, heads, s_k, s_v) = (dims.batch, dims.heads, dims.s_k, dims.s_v);

    // seq = 1, so [batch, 1, H, S] and [batch, H, 1, S] share one memory order.
    let q = q.contiguous()?.reshape(&[batch, heads, 1, s_k])?;
    let q = client.mul_scalar(&q, 1.0 / (s_k as f64).sqrt())?;
    let k = k.contiguous()?.reshape(&[batch, heads, 1, s_k])?;
    let v = v.contiguous()?.reshape(&[batch, heads, 1, s_v])?;
    let g = g.contiguous()?.reshape(&[batch, heads, 1, 1])?;
    let beta = beta.contiguous()?.reshape(&[batch, heads, 1, 1])?;

    // S = S * exp(g)
    let decay = client.exp(&g)?;
    let s = client.mul(state, &decay)?;

    // sk = k @ S : [batch, H, 1, S_v]
    let sk = client.matmul(&k, &s)?;

    // d = (v - sk) * beta
    let d = client.sub(&v, &sk)?;
    let d = client.mul(&d, &beta)?;

    // S = S + k^T @ d : [batch, H, S_k, 1] @ [batch, H, 1, S_v]
    let k_t = k.transpose(2, 3)?;
    let kd = client.matmul(&k_t, &d)?;
    let s_new = client.add(&s, &kd)?;

    // o = q @ S : [batch, H, 1, S_v] -> [batch, 1, H, S_v]
    let o = client.matmul(&q, &s_new)?;
    let o = o.contiguous()?.reshape(&[batch, 1, heads, s_v])?;

    Ok((o, s_new))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            assert!((a - b).abs() < tol, "idx={i}: got {a}, want {b}");
        }
    }

    /// Zero state, one token: `S = k ⊗ (beta v)`, `o = (q·k / sqrt(S_k)) beta v`.
    #[test]
    fn step_zero_state_hand_computed() {
        let (client, device) = cpu_setup();
        let q = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 1, 4], &device)
            .unwrap();
        let k = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 1, 1, 4], &device)
            .unwrap();
        let v = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, -1.0, 2.0], &[1, 1, 1, 4], &device)
            .unwrap();
        // g = ln(0.5): decay on a zero state changes nothing.
        let g = Tensor::<CpuRuntime>::from_slice(&[-std::f32::consts::LN_2], &[1, 1, 1], &device)
            .unwrap();
        let beta = Tensor::<CpuRuntime>::from_slice(&[0.5f32], &[1, 1, 1], &device).unwrap();
        let state = Tensor::<CpuRuntime>::zeros(&[1, 1, 4, 4], DType::F32, &device).unwrap();

        let (o, s) = gdn_step_impl(&client, &q, &k, &v, &g, &beta, &state).unwrap();
        assert_eq!(o.shape(), &[1, 1, 1, 4]);
        assert_eq!(s.shape(), &[1, 1, 4, 4]);

        // d = beta * v = [0.5, 0, -0.5, 1]; S[i][j] = k[i] d[j]: every row is d.
        let want_state = [
            0.5f32, 0.0, -0.5, 1.0, 0.5, 0.0, -0.5, 1.0, 0.5, 0.0, -0.5, 1.0, 0.5, 0.0, -0.5, 1.0,
        ];
        assert_close(&s.to_vec::<f32>(), &want_state, 1e-6);

        // q/sqrt(4) = [0.5, 1, 1.5, 2]; o = sum_i (q_i/2) S[i] = 5 * d = [2.5, 0, -2.5, 5].
        assert_close(&o.to_vec::<f32>(), &[2.5, 0.0, -2.5, 5.0], 1e-6);
    }

    /// Non-zero state: decay, the k @ S read, and the delta all take effect.
    #[test]
    fn step_nonzero_state_hand_computed() {
        let (client, device) = cpu_setup();
        // S_k = S_v = 2 for a hand-checkable case.
        let q = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0], &[1, 1, 1, 2], &device).unwrap();
        let k = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0], &[1, 1, 1, 2], &device).unwrap();
        let v = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 4.0], &[1, 1, 1, 2], &device).unwrap();
        let g = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1, 1, 1], &device).unwrap();
        let beta = Tensor::<CpuRuntime>::from_slice(&[0.5f32], &[1, 1, 1], &device).unwrap();
        // S = [[1, 2], [3, 4]] (rows = k index)
        let state =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &device)
                .unwrap();

        let (o, s) = gdn_step_impl(&client, &q, &k, &v, &g, &beta, &state).unwrap();

        // sk = k @ S = row 1 = [3, 4]; d = 0.5 * ([2, 4] - [3, 4]) = [-0.5, 0]
        // S += k^T d: row 1 += d -> [[1, 2], [2.5, 4]]
        assert_close(&s.to_vec::<f32>(), &[1.0, 2.0, 2.5, 4.0], 1e-6);
        // o = (q / sqrt(2)) @ S = row 0 / sqrt(2) = [1, 2] / 1.41421
        let r = 1.0 / 2f32.sqrt();
        assert_close(&o.to_vec::<f32>(), &[r, 2.0 * r], 1e-6);
    }

    #[test]
    fn step_rejects_seq_gt_one() {
        let (client, device) = cpu_setup();
        let q = Tensor::<CpuRuntime>::zeros(&[1, 2, 1, 4], DType::F32, &device).unwrap();
        let g = Tensor::<CpuRuntime>::zeros(&[1, 2, 1], DType::F32, &device).unwrap();
        let state = Tensor::<CpuRuntime>::zeros(&[1, 1, 4, 4], DType::F32, &device).unwrap();
        assert!(gdn_step_impl(&client, &q, &q, &q, &g, &g, &state).is_err());
    }
}
