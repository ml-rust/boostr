//! Autograd integration for the chunked Gated DeltaNet prefill.
//!
//! Wraps [`GatedDeltaNetOps::gdn_chunk_prefill`] and
//! [`GatedDeltaNetOps::gdn_chunk_backward`] into a `Var`-level operation.

use crate::error::Result;
use crate::ops::traits::GatedDeltaNetOps;
use numr::autograd::{GradFn, TensorId, Var};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::sync::Arc;

/// Input count of the node: q, k, v, g, beta, state.
const INPUTS: usize = 6;

/// Backward function for the chunked GDN prefill.
///
/// Saved state: the six contiguous inputs, in input order.
/// Computes every input gradient via `GatedDeltaNetOps::gdn_chunk_backward`.
///
/// The final state is not in the autograd graph. `GradFn::backward` takes one
/// output gradient, so `dL/dstate` of the returned state is zero here.
struct GdnChunkBackward<R: Runtime> {
    input_ids: [TensorId; INPUTS],
    saved_tensors: [Tensor<R>; INPUTS],
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; INPUTS],
    /// `requires_grad` of each input. An input with `false` gets no gradient.
    requires_grad: [bool; INPUTS],
    chunk_size: usize,
}

impl<R: Runtime> GradFn<R> for GdnChunkBackward<R>
where
    R::Client: GatedDeltaNetOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        needed: &[bool],
    ) -> numr::error::Result<Vec<Option<Tensor<R>>>> {
        let keep: Vec<bool> = self
            .requires_grad
            .iter()
            .zip(needed)
            .map(|(r, n)| *r && *n)
            .collect();
        if !keep.iter().any(|k| *k) {
            return Ok((0..INPUTS).map(|_| None).collect());
        }
        // One call returns all six gradients, so the mask only drops slots.
        let client = R::default_client(grad_output.device());
        let d_o = grad_output.contiguous()?;
        let [q, k, v, g, beta, state] = &self.saved_tensors;
        let grads = client
            .gdn_chunk_backward(q, k, v, g, beta, state, &d_o, None, self.chunk_size)
            .map_err(|e| numr::error::Error::Internal(format!("gdn_chunk_backward failed: {e}")))?;
        let all = [
            grads.dq,
            grads.dk,
            grads.dv,
            grads.dg,
            grads.dbeta,
            grads.dstate,
        ];
        Ok(all
            .into_iter()
            .zip(keep)
            .map(|(grad, slot)| slot.then_some(grad))
            .collect())
    }

    fn backward_var(&self, grad_output: &Var<R>) -> numr::error::Result<Vec<Option<Var<R>>>> {
        // First-order only: wrap the tensor gradients as detached Vars.
        // Second-order traversal keeps every node, so ask for every gradient.
        let grads = self.backward_all(grad_output.tensor())?;
        Ok(grads
            .into_iter()
            .map(|g| g.map(|t| Var::new(t, false)))
            .collect())
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        &self.saved_tensors
    }

    fn name(&self) -> &'static str {
        "GdnChunkBackward"
    }
}

/// Chunked GDN prefill with autograd tracking.
///
/// Shapes and math follow [`GatedDeltaNetOps::gdn_chunk_prefill`]. `q` and `k`
/// must already have `H` heads, so no gradient is reduced over key heads.
///
/// Returns `(o, state)`. `o` is tracked when any input requires grad. `state`
/// is a detached tensor.
///
/// The gradient of the returned state is not propagated. A loss term on it
/// contributes nothing to the input gradients. A caller that feeds `state` to
/// the next segment will train with truncated backpropagation at that boundary.
#[allow(
    clippy::too_many_arguments,
    reason = "client, six forward inputs and the chunk size"
)]
pub fn var_gdn_chunk<R, C>(
    client: &C,
    q: &Var<R>,
    k: &Var<R>,
    v: &Var<R>,
    g: &Var<R>,
    beta: &Var<R>,
    state: &Var<R>,
    chunk_size: usize,
) -> Result<(Var<R>, Tensor<R>)>
where
    R: Runtime,
    C: GatedDeltaNetOps<R>,
    R::Client: GatedDeltaNetOps<R>,
{
    let inputs = [q, k, v, g, beta, state];
    // `contiguous` is a refcount clone for a contiguous tensor.
    let saved = [
        q.tensor().contiguous()?,
        k.tensor().contiguous()?,
        v.tensor().contiguous()?,
        g.tensor().contiguous()?,
        beta.tensor().contiguous()?,
        state.tensor().contiguous()?,
    ];
    let [sq, sk, sv, sg, sbeta, sstate] = &saved;
    let (output, state_out) =
        client.gdn_chunk_prefill(sq, sk, sv, sg, sbeta, sstate, chunk_size)?;

    let requires_grad = inputs.map(|x| x.requires_grad());
    if !requires_grad.iter().any(|r| *r) {
        return Ok((Var::new(output, false), state_out));
    }
    let grad_fn = GdnChunkBackward {
        input_ids: inputs.map(|x| x.id()),
        saved_tensors: saved,
        input_grad_fns: inputs.map(|x| x.grad_fn().cloned()),
        requires_grad,
        chunk_size,
    };
    Ok((Var::from_op(output, Arc::new(grad_fn)), state_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::impl_generic::architecture::gated_delta_net::gdn_recurrent_backward;
    use crate::test_utils::cpu_setup;
    use numr::autograd::{GradStore, backward, var_mul, var_sum};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    const BATCH: usize = 2;
    const SEQ: usize = 7;
    const HEADS: usize = 2;
    /// `S_k != S_v`, so a swapped transpose fails on shape or value.
    const S_K: usize = 3;
    const S_V: usize = 4;
    /// 7 tokens in chunks of 3: the last chunk is padded.
    const CHUNK: usize = 3;
    /// F64 bound, relative to `max(1, largest oracle entry)`.
    const TOL: f64 = 1e-9;
    const NAMES: [&str; INPUTS] = ["q", "k", "v", "g", "beta", "state"];
    const ALL: [bool; INPUTS] = [true; INPUTS];

    type T = Tensor<CpuRuntime>;

    /// Deterministic 64-bit LCG; no rand dependency.
    struct Lcg(u64);

    impl Lcg {
        fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            lo + (hi - lo) * ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    fn shapes() -> [Vec<usize>; INPUTS] {
        [
            vec![BATCH, SEQ, HEADS, S_K],
            vec![BATCH, SEQ, HEADS, S_K],
            vec![BATCH, SEQ, HEADS, S_V],
            vec![BATCH, SEQ, HEADS],
            vec![BATCH, SEQ, HEADS],
            vec![BATCH, HEADS, S_K, S_V],
        ]
    }

    fn unit_rows(data: Vec<f64>, dim: usize) -> Vec<f64> {
        data.chunks(dim)
            .flat_map(|row| {
                let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
                row.iter().map(move |x| x / norm)
            })
            .collect()
    }

    /// F64 inputs in the order of [`NAMES`], plus the weight `W` of `sum(o * W)`.
    fn problem(device: &CpuDevice) -> ([T; INPUTS], T) {
        let mut rng = Lcg(0x9d4_0001);
        let rows = BATCH * SEQ * HEADS;
        let mut uni = |n: usize, lo: f64, hi: f64| -> Vec<f64> {
            (0..n).map(|_| rng.uniform(lo, hi)).collect()
        };
        let data = [
            unit_rows(uni(rows * S_K, -1.0, 1.0), S_K),
            unit_rows(uni(rows * S_K, -1.0, 1.0), S_K),
            uni(rows * S_V, -1.0, 1.0),
            uni(rows, -1.0, -0.05),
            uni(rows, 0.1, 0.9),
            uni(BATCH * HEADS * S_K * S_V, -0.5, 0.5),
        ];
        let w = uni(rows * S_V, -1.0, 1.0);
        let shapes = shapes();
        let inputs = std::array::from_fn(|i| T::from_slice(&data[i], &shapes[i], device).unwrap());
        let w = T::from_slice(&w, &[BATCH, SEQ, HEADS, S_V], device).unwrap();
        (inputs, w)
    }

    /// `L = sum(o * W)` through the Var path. Returns the input Vars and the gradients.
    fn var_grads(
        client: &CpuClient,
        inputs: &[T; INPUTS],
        w: &T,
        requires: [bool; INPUTS],
    ) -> ([Var<CpuRuntime>; INPUTS], GradStore<CpuRuntime>) {
        let vars: [Var<CpuRuntime>; INPUTS] =
            std::array::from_fn(|i| Var::new(inputs[i].clone(), requires[i]));
        let [q, k, v, g, beta, state] = &vars;
        let (o, _) = var_gdn_chunk(client, q, k, v, g, beta, state, CHUNK).unwrap();
        let w = Var::new(w.clone(), false);
        let prod = var_mul(&o, &w, client).unwrap();
        let loss = var_sum(&prod, &[0, 1, 2, 3], false, client).unwrap();
        let grads = backward(&loss, client).unwrap();
        (vars, grads)
    }

    /// Sequential oracle gradients of `sum(o * W)`.
    fn oracle(client: &CpuClient, inputs: &[T; INPUTS], w: &T) -> [T; INPUTS] {
        let [q, k, v, g, beta, state] = inputs;
        let d = gdn_recurrent_backward(client, q, k, v, g, beta, state, w, None).unwrap();
        [d.dq, d.dk, d.dv, d.dg, d.dbeta, d.dstate]
    }

    fn assert_close(got: &T, want: &T, name: &str) {
        assert_eq!(got.shape(), want.shape(), "{name} shape");
        let (got, want) = (got.to_vec::<f64>(), want.to_vec::<f64>());
        let scale = want.iter().fold(1.0f64, |m, x| m.max(x.abs()));
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            diff <= TOL * scale,
            "{name}: diff {diff} (bound {})",
            TOL * scale
        );
    }

    /// Every input that requires grad gets the oracle gradient. Every other input gets none.
    ///
    /// `strided_q` passes `q` as a non-contiguous view with the same values.
    fn check(requires: [bool; INPUTS], strided_q: bool) {
        let (client, device) = cpu_setup();
        let (reference, w) = problem(&device);
        let mut inputs = reference.clone();
        if strided_q {
            let head_major = inputs[0]
                .permute(&[0, 2, 1, 3])
                .unwrap()
                .contiguous()
                .unwrap();
            inputs[0] = head_major.permute(&[0, 2, 1, 3]).unwrap();
            assert!(!inputs[0].is_contiguous());
        }
        let (vars, grads) = var_grads(&client, &inputs, &w, requires);
        let want = oracle(&client, &reference, &w);
        let shapes = shapes();
        for (i, var) in vars.iter().enumerate() {
            match grads.get(var.id()) {
                Some(got) if requires[i] => {
                    assert_eq!(got.shape(), shapes[i].as_slice(), "{} shape", NAMES[i]);
                    assert_close(got, &want[i], NAMES[i]);
                }
                None if !requires[i] => {}
                got => panic!(
                    "{}: requires_grad {}, gradient present {}",
                    NAMES[i],
                    requires[i],
                    got.is_some()
                ),
            }
        }
    }

    #[test]
    fn var_path_matches_oracle() {
        check(ALL, false);
    }

    #[test]
    fn frozen_input_gets_no_gradient() {
        check([true, true, true, true, false, true], false);
    }

    #[test]
    fn frozen_state() {
        check([true, true, true, true, true, false], false);
    }

    /// A strided `q` view gives the same gradients as its contiguous copy.
    #[test]
    fn non_contiguous_input() {
        check(ALL, true);
    }

    /// No input requires grad: `o` is untracked, and both outputs match the forward op.
    #[test]
    fn untracked_without_requires_grad() {
        let (client, device) = cpu_setup();
        let (inputs, _) = problem(&device);
        let vars: [Var<CpuRuntime>; INPUTS] =
            std::array::from_fn(|i| Var::new(inputs[i].clone(), false));
        let [q, k, v, g, beta, state] = &vars;
        let (o, s) = var_gdn_chunk(&client, q, k, v, g, beta, state, CHUNK).unwrap();
        assert!(!o.requires_grad());
        assert!(o.grad_fn().is_none());
        let [q, k, v, g, beta, state] = &inputs;
        let (want_o, want_s) = client
            .gdn_chunk_prefill(q, k, v, g, beta, state, CHUNK)
            .unwrap();
        assert_eq!(o.tensor().to_vec::<f64>(), want_o.to_vec::<f64>());
        assert_eq!(s.to_vec::<f64>(), want_s.to_vec::<f64>());
    }
}
