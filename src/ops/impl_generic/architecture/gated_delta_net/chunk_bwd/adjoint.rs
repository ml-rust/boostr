//! Reverse sweep of the chunked GDN backward.
//!
//! The state adjoint is the only gradient that crosses chunks. Per chunk,
//! with `S` the chunk-start state and `dS'` the chunk-end adjoint:
//!
//! ```text
//! dV_new = P^T dO + K_g dS'
//! dS     = Q_g^T dO + exp(g_last) dS' - W^T dV_new
//!        = X + exp(g_last) dS' - W^T (K_g dS')    X = (Q_g - P W)^T dO
//! ```
//!
//! `dS` is the chunk-end adjoint of the previous chunk. `X` holds no
//! adjoint, so it is built for every chunk in one batched product.

use super::super::chunk_layout::chunk_of;
use super::super::chunk_terms::ChunkTerms;
use crate::error::Result;
use numr::dtype::DType;
use numr::ops::{BinaryOps, MatmulOps, ShapeOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// State adjoints of every chunk.
pub(super) struct ChunkAdjoints<R: Runtime> {
    /// `dL/dS'` at the end of chunk `c`: `[N, n_chunks, S_k, S_v]`.
    pub ds_end: Tensor<R>,
    /// `dL/dS` for the initial state: `[N, S_k, S_v]`.
    pub dstate: Tensor<R>,
}

/// Sweep the state adjoint from the last chunk to the first.
///
/// `d_o` is the chunked output gradient: `[N, n_chunks, cs, S_v]`, zero on
/// padded tokens. `ds_final` is `dL/dS` of the final state: `[N, S_k, S_v]`.
pub(super) fn chunk_end_adjoints<R, C>(
    client: &C,
    terms: &ChunkTerms<R>,
    d_o: &Tensor<R>,
    ds_final: &Tensor<R>,
    n_chunks: usize,
) -> Result<ChunkAdjoints<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + MatmulOps<R> + ShapeOps<R>,
{
    let n = ds_final.shape()[0];
    // X = (Q_g - P W)^T dO : [N, n_chunks, S_k, S_v]
    let pw = client.matmul(&terms.kq, &terms.w)?;
    let x = client.matmul(&client.sub(&terms.q_g, &pw)?.transpose(2, 3)?, d_o)?;

    let mut cur = ds_final.clone();
    let mut ends = Vec::with_capacity(n_chunks);
    for c in (0..n_chunks).rev() {
        let x_c = chunk_of(&x, c)?;
        let w_c = chunk_of(&terms.w, c)?;
        let k_g_t_c = chunk_of(&terms.k_g_t, c)?;
        let gl = chunk_of(&terms.g_last_exp, c)?.reshape(&[n, 1, 1])?;

        let kg_ds = client.matmul(&k_g_t_c.transpose(1, 2)?, &cur)?; // [N, cs, S_v]
        let w_kg_ds = client.matmul(&w_c.transpose(1, 2)?, &kg_ds)?; // [N, S_k, S_v]
        let prev = client.sub(&client.add(&x_c, &client.mul(&cur, &gl)?)?, &w_kg_ds)?;
        ends.push(cur);
        cur = prev;
    }
    ends.reverse();
    let refs: Vec<&Tensor<R>> = ends.iter().collect();
    Ok(ChunkAdjoints {
        ds_end: client.stack(&refs, 1)?,
        dstate: cur,
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::chunk::gdn_chunk_prefill_impl;
    use super::super::super::chunk_layout::{pad_seq, to_chunks};
    use super::super::super::chunk_terms::chunk_inputs;
    use super::super::super::common::check_gdn_shapes;
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Output and final state are affine in the initial state. The adjoint
    /// must therefore match a state perturbation exactly, not to first order.
    #[test]
    fn adjoint_matches_state_perturbation() {
        let (client, device) = cpu_setup();
        // seq 8 over chunks of 3: the last chunk holds one padded token.
        let (batch, seq, heads, s_k, s_v, cs) = (2, 8, 2, 3, 4, 3);
        let wave = |n: usize, f: f64| -> Vec<f64> {
            (0..n).map(|i| ((i as f64 + 1.0) * f).sin()).collect()
        };
        let mk = |data: Vec<f64>, shape: &[usize]| {
            Tensor::<CpuRuntime>::from_slice(&data, shape, &device).unwrap()
        };
        let rows = batch * seq * heads;
        let n_state = batch * heads * s_k * s_v;
        let state_shape = [batch, heads, s_k, s_v];
        let q = mk(wave(rows * s_k, 0.37), &[batch, seq, heads, s_k]);
        let k = mk(wave(rows * s_k, 0.53), &[batch, seq, heads, s_k]);
        let v = mk(wave(rows * s_v, 0.71), &[batch, seq, heads, s_v]);
        let g_data = wave(rows, 0.29).iter().map(|x| -0.1 - x.abs()).collect();
        let g = mk(g_data, &[batch, seq, heads]);
        let beta_data = wave(rows, 0.43).iter().map(|x| 0.5 + 0.4 * x).collect();
        let beta = mk(beta_data, &[batch, seq, heads]);
        let state_data = wave(n_state, 0.61);
        let delta_data = wave(n_state, 0.83);
        let shifted: Vec<f64> = state_data
            .iter()
            .zip(&delta_data)
            .map(|(a, b)| a + b)
            .collect();
        let state = mk(state_data, &state_shape);
        let state_shifted = mk(shifted, &state_shape);
        let d_o_data = wave(rows * s_v, 0.47);
        let ds_data = wave(n_state, 0.19);
        let d_o = mk(d_o_data.clone(), &[batch, seq, heads, s_v]);
        let ds_final = mk(ds_data.clone(), &[batch * heads, s_k, s_v]);

        let run = |s: &Tensor<CpuRuntime>| {
            let (o, s) =
                gdn_chunk_prefill_impl(&client, &q, &k, &v, &g, &beta, s, None, cs).unwrap();
            (o.to_vec::<f64>(), s.to_vec::<f64>())
        };
        let (o0, s0) = run(&state);
        let (o1, s1) = run(&state_shifted);
        let d_out: Vec<f64> = o1.iter().zip(&o0).map(|(a, b)| a - b).collect();
        let d_fin: Vec<f64> = s1.iter().zip(&s0).map(|(a, b)| a - b).collect();
        let want = dot(&d_o_data, &d_out) + dot(&ds_data, &d_fin);

        let dims = check_gdn_shapes(&q, &k, &v, &g, &beta, &state).unwrap();
        let x = chunk_inputs(&client, &q, &k, &v, &g, &beta, None, dims, cs).unwrap();
        let d_o_c = to_chunks(
            &pad_seq(&client, &d_o, seq, x.padded).unwrap(),
            x.n,
            x.n_chunks,
            cs,
        )
        .unwrap();
        let adj = chunk_end_adjoints(&client, &x.terms, &d_o_c, &ds_final, x.n_chunks).unwrap();
        assert_eq!(adj.ds_end.shape(), &[x.n, x.n_chunks, s_k, s_v]);
        assert_eq!(adj.dstate.shape(), &[x.n, s_k, s_v]);
        assert_eq!(
            chunk_of(&adj.ds_end, x.n_chunks - 1)
                .unwrap()
                .to_vec::<f64>(),
            ds_data
        );

        let got = dot(&adj.dstate.to_vec::<f64>(), &delta_data);
        let tol = 1e-10 * want.abs().max(1.0);
        assert!(
            (got - want).abs() < tol,
            "adjoint {got}, perturbation {want}"
        );
    }
}
