//! Forward sweep of the chunked GDN backward.
//!
//! The sweep checkpoints the state at the start of every chunk. The
//! backward never rebuilds a state by dividing out `exp(g)`.

use super::super::chunk_layout::chunk_of;
use super::super::chunk_terms::{ChunkTerms, advance_state};
use crate::error::Result;
use numr::dtype::DType;
use numr::ops::{BinaryOps, MatmulOps, ShapeOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Chunk-start states and the corrected values of every chunk.
pub(super) struct ChunkStates<R: Runtime> {
    /// State before chunk `c`: `[N, n_chunks, S_k, S_v]`.
    pub s: Tensor<R>,
    /// `V_new = U - W @ S_c`: `[N, n_chunks, cs, S_v]`.
    pub v_new: Tensor<R>,
}

/// Run the chunk recurrence of the forward and keep its chunk-start states.
///
/// `state` is the initial state: `[N, S_k, S_v]`. The final state after the
/// last chunk is not kept.
pub(super) fn chunk_start_states<R, C>(
    client: &C,
    terms: &ChunkTerms<R>,
    state: &Tensor<R>,
    n_chunks: usize,
) -> Result<ChunkStates<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + MatmulOps<R> + ShapeOps<R>,
{
    let n = state.shape()[0];
    let mut s = state.clone();
    let mut starts = Vec::with_capacity(n_chunks);
    let mut v_news = Vec::with_capacity(n_chunks);
    for c in 0..n_chunks {
        let w_c = chunk_of(&terms.w, c)?;
        let u_c = chunk_of(&terms.u, c)?;
        let k_g_t_c = chunk_of(&terms.k_g_t, c)?;
        let gl = chunk_of(&terms.g_last_exp, c)?.reshape(&[n, 1, 1])?;
        let (v_new, s_next) = advance_state(client, &s, &w_c, &u_c, &k_g_t_c, &gl)?;
        starts.push(s);
        v_news.push(v_new);
        s = s_next;
    }
    let start_refs: Vec<&Tensor<R>> = starts.iter().collect();
    let v_new_refs: Vec<&Tensor<R>> = v_news.iter().collect();
    Ok(ChunkStates {
        s: client.stack(&start_refs, 1)?,
        v_new: client.stack(&v_new_refs, 1)?,
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::chunk::gdn_chunk_prefill_impl;
    use super::super::super::chunk_terms::chunk_inputs;
    use super::super::super::common::check_gdn_shapes;
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// The start state of chunk `c` equals the prefill state after `c` chunks.
    #[test]
    fn start_states_match_prefix_prefill() {
        let (client, device) = cpu_setup();
        let (batch, seq, heads, s_k, s_v, cs) = (2, 8, 2, 3, 4, 3);
        let wave = |n: usize, f: f64| -> Vec<f64> {
            (0..n).map(|i| ((i as f64 + 1.0) * f).sin()).collect()
        };
        let mk = |data: Vec<f64>, shape: &[usize]| {
            Tensor::<CpuRuntime>::from_slice(&data, shape, &device).unwrap()
        };
        let rows = batch * seq * heads;
        let q = mk(wave(rows * s_k, 0.37), &[batch, seq, heads, s_k]);
        let k = mk(wave(rows * s_k, 0.53), &[batch, seq, heads, s_k]);
        let v = mk(wave(rows * s_v, 0.71), &[batch, seq, heads, s_v]);
        let g_data = wave(rows, 0.29).iter().map(|x| -0.1 - x.abs()).collect();
        let g = mk(g_data, &[batch, seq, heads]);
        let beta_data = wave(rows, 0.43).iter().map(|x| 0.5 + 0.4 * x).collect();
        let beta = mk(beta_data, &[batch, seq, heads]);
        let state = mk(
            wave(batch * heads * s_k * s_v, 0.61),
            &[batch, heads, s_k, s_v],
        );

        let dims = check_gdn_shapes(&q, &k, &v, &g, &beta, &state).unwrap();
        let x = chunk_inputs(&client, &q, &k, &v, &g, &beta, dims, cs).unwrap();
        let n = batch * heads;
        let s0 = state.reshape(&[n, s_k, s_v]).unwrap();
        let st = chunk_start_states(&client, &x.terms, &s0, x.n_chunks).unwrap();
        assert_eq!(st.s.shape(), &[n, x.n_chunks, s_k, s_v]);
        assert_eq!(st.v_new.shape(), &[n, x.n_chunks, cs, s_v]);

        for c in 0..x.n_chunks {
            let want = if c == 0 {
                state.to_vec::<f64>()
            } else {
                let len = c * cs;
                let pre = |t: &Tensor<CpuRuntime>| t.narrow(1, 0, len).unwrap();
                let (_, s) = gdn_chunk_prefill_impl(
                    &client,
                    &pre(&q),
                    &pre(&k),
                    &pre(&v),
                    &pre(&g),
                    &pre(&beta),
                    &state,
                    cs,
                )
                .unwrap();
                s.to_vec::<f64>()
            };
            let got = chunk_of(&st.s, c).unwrap().to_vec::<f64>();
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!((a - b).abs() < 1e-12, "c={c} idx={i}: got {a}, want {b}");
            }
        }
    }
}
