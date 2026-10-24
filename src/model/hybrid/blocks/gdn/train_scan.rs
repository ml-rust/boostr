//! The recurrent core of the tracked training forward of a [`GdnBlock`]:
//! causal conv, head split, normalisation, gates and the chunked
//! delta-rule scan, all on the autograd graph.
//!
//! With segment masks, the conv and the scan both isolate packed documents:
//! no conv tap and no recurrent state crosses a document start.

use super::layer::GdnBlock;
use super::train_proj::narrow_heads;
use crate::error::{Error, Result};
use crate::nn::segment_masks::SegmentMasks;
use crate::nn::var_causal_conv1d_silu;
use crate::nn::var_ops::repeat_heads_tiled;
use crate::ops::traits::GatedDeltaNetOps;
use crate::ops::var_gdn_chunk;
use numr::autograd::{Var, var_add, var_l2_normalize, var_mul, var_sigmoid, var_softplus};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, MatmulOps, ScalarOps, ShapeOps, TensorOps,
    TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> GdnBlock<R> {
    /// Conv, split, normalise, gate and scan the projections.
    ///
    /// `qkv` is `[batch, seq, qkv_dim]`, `alpha` and `beta_raw` are
    /// `[batch, seq, H_v]`. Returns `o` as `[batch, seq, H_v, S]`. The scan
    /// starts from a zero state and its final state leaves the graph.
    ///
    /// # Errors
    ///
    /// numr and kernel errors propagate.
    pub(super) fn scan_train<C>(
        &self,
        client: &C,
        qkv: &Var<R>,
        alpha: &Var<R>,
        beta_raw: &Var<R>,
        segments: Option<&SegmentMasks<R>>,
    ) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + CompareOps<R>
            + MatmulOps<R>
            + TypeConversionOps<R>
            + GatedDeltaNetOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + CompareOps<R>
            + MatmulOps<R>
            + GatedDeltaNetOps<R>,
    {
        let cfg = &self.cfg;
        let shape = qkv.shape();
        let (batch, seq) = (shape[0], shape[1]);
        let (h_k, h_v, s) = (cfg.key_heads, cfg.value_heads, cfg.state_size);
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();
        let ssm_a = self.decay_var(client)?;

        // 3. Causal conv over a zero window, then SiLU. No bias.
        let qkv = var_causal_conv1d_silu(client, qkv, &self.conv_weight, None, true, segments)?;

        // 4. Split.
        let q = narrow_heads(&qkv, 0, key_dim, &[batch, seq, h_k, s])?;
        let k = narrow_heads(&qkv, key_dim, key_dim, &[batch, seq, h_k, s])?;
        let v = narrow_heads(&qkv, 2 * key_dim, value_dim, &[batch, seq, h_v, s])?;

        // 5. L2 norm over the head dim, `eps = rms_eps` as in inference.
        let eps = f64::from(cfg.rms_eps);
        let q = var_l2_normalize(&q, -1, eps, client).map_err(Error::Numr)?;
        let k = var_l2_normalize(&k, -1, eps, client).map_err(Error::Numr)?;

        // 6. Tiled repeat: value head h_v reads key head h_v % H_k.
        let q = repeat_heads_tiled(&q, h_v).map_err(Error::Numr)?;
        let k = repeat_heads_tiled(&k, h_v).map_err(Error::Numr)?;

        // 7. Gates: `g = ssm_a * softplus(alpha + dt_bias)`, `ssm_a < 0`.
        let beta = var_sigmoid(beta_raw, client).map_err(Error::Numr)?;
        let g = var_add(alpha, &self.ssm_dt_bias, client).map_err(Error::Numr)?;
        let g = var_softplus(&g, client).map_err(Error::Numr)?;
        let g = var_mul(&g, &ssm_a, client).map_err(Error::Numr)?;

        // 8. Chunked recurrence from a zero state.
        let (dtype, device) = (v.tensor().dtype(), v.tensor().device());
        let state = Tensor::<R>::zeros(&[batch, h_v, s, s], dtype, device).map_err(Error::Numr)?;
        let state = Var::new(state, false);
        let ids = segments.map(SegmentMasks::segment_ids);
        let (o, _) = var_gdn_chunk(client, &q, &k, &v, &g, &beta, &state, ids, cfg.chunk_size)?;
        Ok(o)
    }
}

#[cfg(test)]
mod tests {
    //! Gradients against central finite differences in F64, and packed
    //! documents against separate runs.

    use super::super::train_fixtures::{
        BATCH, HIDDEN, KERNEL, NAMES, SEQ, X, build64, cfg64, f64_tensor, init_params, param_ids,
    };
    use crate::model::config::GdnConfig;
    use crate::nn::segment_masks::SegmentMasks;
    use crate::test_utils::{
        Lcg, check_no_leak_tol, check_packed_vs_separate_tol,
        check_single_document_equals_none_tol, cpu_setup, to_f64, weighted_loss,
    };
    use numr::autograd::{Var, backward};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    /// Central-difference step for the F64 checks.
    const STEP: f64 = 4e-6;
    /// Relative bound, against the larger of `|fd|` and the parameter's
    /// largest analytic gradient.
    const TOL: f64 = 1e-6;
    const PROBES: usize = 8;
    /// Bound of the packed-versus-separate comparisons.
    const PACKED_TOL: f64 = 1e-9;
    const LEAK_TOL: f64 = 1e-12;

    fn loss_value(
        client: &CpuClient,
        device: &CpuDevice,
        cfg: &GdnConfig,
        p: &[Vec<f64>],
        w: &[f64],
    ) -> f64 {
        let block = build64(client, device, cfg, p);
        let x = Var::new(f64_tensor(&p[X], &[BATCH, SEQ, HIDDEN], device), false);
        let y = block.forward_train(client, &x, None).unwrap();
        to_f64(y.tensor()).iter().zip(w).map(|(a, b)| a * b).sum()
    }

    fn check_gradients(key_heads: usize, v_grouped: bool) {
        let (client, device) = cpu_setup();
        let cfg = cfg64(key_heads, v_grouped);
        let params = init_params(&cfg, 0x6d4e_0040 + key_heads as u64);
        let w_data = Lcg(0x6d4e_0041).vec(BATCH * SEQ * HIDDEN, 1.0);

        let block = build64(&client, &device, &cfg, &params);
        let x = Var::new(f64_tensor(&params[X], &[BATCH, SEQ, HIDDEN], &device), true);
        let y = block.forward_train(&client, &x, None).unwrap();
        let w = Var::new(f64_tensor(&w_data, y.shape(), &device), false);
        let grads = backward(&weighted_loss(&client, &y, &w), &client).unwrap();
        let mut ids = vec![x.id()];
        ids.extend(param_ids(&block));

        let rep = cfg.head_repeat();
        let mut rng = Lcg(0x6d4e_0050);
        for (i, id) in ids.iter().enumerate() {
            let name = NAMES[i];
            let grad = grads
                .get(*id)
                .unwrap_or_else(|| panic!("rep={rep} {name}: no gradient"));
            let analytic = to_f64(grad);
            assert_eq!(analytic.len(), params[i].len(), "rep={rep} {name}: size");
            let scale = analytic.iter().fold(1e-3f64, |m, g| m.max(g.abs()));
            for _ in 0..PROBES {
                let j = rng.index(params[i].len());
                let mut plus = params.clone();
                plus[i][j] += STEP;
                let mut minus = params.clone();
                minus[i][j] -= STEP;
                let fd = (loss_value(&client, &device, &cfg, &plus, &w_data)
                    - loss_value(&client, &device, &cfg, &minus, &w_data))
                    / (2.0 * STEP);
                let tol = TOL * fd.abs().max(scale);
                assert!(
                    (analytic[j] - fd).abs() <= tol,
                    "rep={rep} {name}[{j}]: analytic {} vs fd {fd}",
                    analytic[j]
                );
            }
        }
    }

    #[test]
    fn gradients_match_finite_differences_rep2_grouped() {
        check_gradients(2, true);
    }

    #[test]
    fn gradients_match_finite_differences_rep1() {
        check_gradients(4, false);
    }

    // ---- Packed documents ----

    type Seg<'a> = Option<&'a SegmentMasks<CpuRuntime>>;

    /// The block under test, its trainable parameters and a forward closure
    /// over it. Chunks hold four tokens.
    macro_rules! packed_fixture {
        ($block:ident, $fwd:ident, $params:ident) => {
            let (client, device) = cpu_setup();
            let cfg = cfg64(2, false);
            let $block = build64(&client, &device, &cfg, &init_params(&cfg, 0x6d4e_0070));
            let $params = $block.trainable_parameters();
            let $fwd =
                |c: &CpuClient, x: &Var<CpuRuntime>, s: Seg<'_>| $block.forward_train(c, x, s);
        };
    }

    /// Resets fall mid-chunk (6) and, with chunk 4, the first document spans
    /// a chunk boundary.
    #[test]
    fn packed_block_equals_separate_documents() {
        packed_fixture!(block, fwd, params);
        check_packed_vs_separate_tol(&fwd, &params, HIDDEN, KERNEL, &[6, 3], PACKED_TOL);
    }

    /// Three documents, one a single token. Starts at 4 and 8 sit on chunk
    /// boundaries, starts at 3, 5 and 6 sit inside a chunk.
    #[test]
    fn packed_block_three_documents_with_one_token() {
        packed_fixture!(block, fwd, params);
        check_packed_vs_separate_tol(&fwd, &params, HIDDEN, KERNEL, &[3, 1, 4], PACKED_TOL);
        check_packed_vs_separate_tol(&fwd, &params, HIDDEN, KERNEL, &[4, 1, 3, 5], PACKED_TOL);
        check_packed_vs_separate_tol(&fwd, &params, HIDDEN, KERNEL, &[1, 2, 1, 2], PACKED_TOL);
        check_packed_vs_separate_tol(&fwd, &params, HIDDEN, KERNEL, &[1, 5, 2], PACKED_TOL);
    }

    #[test]
    fn packed_block_does_not_leak_across_documents() {
        packed_fixture!(block, fwd, params);
        check_no_leak_tol(&fwd, &params, HIDDEN, KERNEL, 5, 6, LEAK_TOL);
    }

    #[test]
    fn single_document_segments_equal_none() {
        packed_fixture!(block, fwd, params);
        check_single_document_equals_none_tol(&fwd, &params, HIDDEN, KERNEL, LEAK_TOL);
    }
}
