//! Tracked training forward of a [`GdnBlock`]: the inference prefill chain
//! of `forward` and `prefill`, with every step on the autograd graph.
//!
//! Each call starts a fresh sequence: a zero conv window and a zero
//! delta-rule state. No `GdnState` is read or written. The final state
//! leaves the graph, so the `seq` tokens of one call form one
//! backpropagation segment. The conv and scan are in `train_scan`, the
//! projection checks in `train_proj`.

use super::heads::group_heads_var;
use super::layer::GdnBlock;
use super::train_proj::tracked_forward;
use crate::error::{Error, Result};
use crate::nn::segment_masks::SegmentMasks;
use crate::nn::var_contiguous;
use crate::ops::traits::GatedDeltaNetOps;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, MatmulOps, ScalarOps, ShapeOps, TensorOps,
    TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime<DType = DType>> GdnBlock<R> {
    /// Training forward. `x` is the `attn_norm`-ed hidden state
    /// `[batch, seq, hidden_size]`; the result is the `ssm_out` projection
    /// `[batch, seq, hidden_size]`, without the residual.
    ///
    /// The math equals [`Self::forward`] on a zero `GdnState`, with the
    /// chunked recurrence for every `seq`. Gradients reach `x`, every
    /// projection weight or adapter that requires grad, the conv kernel,
    /// `a_log`, `dt_bias` and the norm weight.
    ///
    /// `segments` packs documents into a row. With `Some`, the conv and the
    /// scan isolate them: no conv tap, recurrent state or gradient crosses a
    /// document start. `None` treats each row as one document and gives the
    /// same values as before segments existed.
    ///
    /// # Errors
    ///
    /// - [`Error::ModelError`] when `x` is not `[batch, seq, hidden_size]`.
    /// - [`Error::InvalidArgument`] when a projection is Hadamard-rotated or
    ///   has a decomposed AWQ/GPTQ base. `arg` names the projection.
    /// - [`Error::InvalidArgument`] with `arg` `segments` when the masks
    ///   disagree with `x` in `[batch, seq]` or dtype, or cover fewer conv
    ///   taps than the block's kernel.
    /// - numr and kernel errors propagate.
    pub fn forward_train<C>(
        &self,
        client: &C,
        x: &Var<R>,
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
            + QuantMatmulOps<R>
            + TypeConversionOps<R>
            + GatedDeltaNetOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + CompareOps<R>
            + MatmulOps<R>
            + DequantOps<R>
            + GatedDeltaNetOps<R>,
    {
        self.check_train_input(x, segments)?;
        let cfg = &self.cfg;
        let shape = x.shape();
        let (batch, seq) = (shape[0], shape[1]);
        let (h_k, h_v, s) = (cfg.key_heads, cfg.value_heads, cfg.state_size);
        let value_dim = cfg.value_dim();

        // 1-2. Projections and raw gates, all tracked.
        let qkv = tracked_forward(&self.attn_qkv, "attn_qkv", client, x)?;
        let z = tracked_forward(&self.attn_gate, "attn_gate", client, x)?;
        let (alpha, beta_raw) = self.gates.forward_var(client, x)?;

        // 3-8. Conv, split, normalise, gate, scan.
        let o = self.scan_train(client, &qkv, &alpha, &beta_raw, segments)?;

        // 9. silu(z) * rms_norm(o), per head.
        let z = var_contiguous(&z)?;
        let z = var_reshape(&z, &[batch, seq, h_v, s]).map_err(Error::Numr)?;
        let o = self.norm.forward(client, &o, &z)?;

        // 10. Flatten heads, reorder for a grouped `ssm_out`, project.
        let o = var_contiguous(&o)?;
        let o = var_reshape(&o, &[batch, seq, value_dim]).map_err(Error::Numr)?;
        let o = if cfg.v_grouped {
            group_heads_var(&o, h_k, cfg.head_repeat(), s)?
        } else {
            o
        };
        tracked_forward(&self.ssm_out, "ssm_out", client, &o)
    }
}

#[cfg(test)]
mod tests {
    //! Value parity with the inference forward, adapters over a frozen
    //! base, and a short SGD run.

    use super::super::GdnProjection;
    use super::super::forward::tests::block;
    use super::super::train_fixtures::{
        BATCH, CHUNK, H_V, HIDDEN, NAMES, SEQ, X, build64, cfg64, f32_tensor, f64_tensor,
        init_params, param_ids,
    };
    use crate::inference::GdnState;
    use crate::nn::LoraFactors;
    use crate::test_utils::{Lcg, cpu_setup, to_f64, weighted_loss};
    use numr::autograd::{Var, backward, var_mul, var_mul_scalar, var_sub, var_sum};
    use numr::dtype::DType;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::TensorId;

    const RANK: usize = 2;
    const LR: f64 = 1e-2;

    // ---- Value parity with the inference forward (F32 fixtures) ----

    fn check_value_parity(v_grouped: bool) {
        let (client, device) = cpu_setup();
        for chunk in [64, CHUNK] {
            for seq in [SEQ, 1] {
                let mut block = block(&device, 0x6d4e_0030, v_grouped);
                block.cfg.chunk_size = chunk;
                let cfg = block.config().clone();
                let x = f32_tensor(
                    &Lcg(21).vec(BATCH * seq * HIDDEN, 1.0),
                    &[BATCH, seq, HIDDEN],
                    &device,
                );
                let mut state =
                    GdnState::<CpuRuntime>::zeros(&cfg, BATCH, DType::F32, &device).unwrap();
                let want = block
                    .forward(&client, &Var::new(x.clone(), false), &mut state)
                    .unwrap();
                let got = block
                    .forward_train(&client, &Var::new(x, false), None)
                    .unwrap();
                assert!(!got.requires_grad(), "nothing requires grad");
                assert_eq!(got.shape(), want.shape());
                let (got, want) = (got.tensor().to_vec::<f32>(), want.tensor().to_vec::<f32>());
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() < 1e-5,
                        "v_grouped={v_grouped} chunk={chunk} seq={seq} [{i}]: {g} vs {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn forward_train_matches_inference_tiled() {
        check_value_parity(false);
    }

    #[test]
    fn forward_train_matches_inference_grouped() {
        check_value_parity(true);
    }

    // ---- Adapters over a frozen base ----

    #[test]
    fn frozen_base_trains_only_lora_factors() {
        let (client, device) = cpu_setup();
        let mut block = block(&device, 0x6d4e_0031, false);
        let cfg = block.config().clone();
        let mut rng = Lcg(0x6d4e_0032);
        let mut factors = |out: usize, inp: usize| LoraFactors {
            lora_a: f32_tensor(&rng.vec(RANK * inp, 0.3), &[RANK, inp], &device),
            lora_b: f32_tensor(&rng.vec(out * RANK, 0.3), &[out, RANK], &device),
            scaling: 0.5,
        };
        let qkv = factors(cfg.qkv_dim(), HIDDEN);
        let out = factors(HIDDEN, cfg.value_dim());
        let alpha = factors(H_V, HIDDEN);
        block
            .attach_adapter_trainable(&client, GdnProjection::AttnQkv, qkv, DType::F32, true)
            .unwrap();
        block
            .attach_adapter_trainable(&client, GdnProjection::SsmOut, out, DType::F32, true)
            .unwrap();
        block
            .attach_gate_adapters_trainable(&client, Some(alpha), None, DType::F32, true)
            .unwrap();

        let x_data = Lcg(0x6d4e_0033).vec(BATCH * SEQ * HIDDEN, 1.0);
        let x = Var::new(f32_tensor(&x_data, &[BATCH, SEQ, HIDDEN], &device), false);
        let y = block.forward_train(&client, &x, None).unwrap();
        assert!(y.requires_grad());
        let w_data = Lcg(0x6d4e_0034).vec(BATCH * SEQ * HIDDEN, 1.0);
        let w = Var::new(f32_tensor(&w_data, y.shape(), &device), false);
        let grads = backward(&weighted_loss(&client, &y, &w), &client).unwrap();

        let trainable = block.trainable_parameters();
        assert_eq!(trainable.len(), 6, "two factors on each of three adapters");
        for (id, _) in &trainable {
            assert!(grads.get(*id).is_some(), "a LoRA factor gets a gradient");
        }
        // Small frozen vectors feed `var_add` and `var_mul` directly, and numr
        // can record a gradient for such a leaf. The check covers the projections.
        let is_projection = |name: &str| {
            ["in_proj", "out_proj", "alpha", "beta"]
                .iter()
                .any(|p| name.starts_with(p))
        };
        for (name, var) in block.named_parameters() {
            if !var.requires_grad() && is_projection(&name) {
                assert!(
                    grads.get(var.id()).is_none(),
                    "a frozen base weight gets none: {name}"
                );
            }
        }
    }

    // ---- Two SGD steps ----

    #[test]
    fn sgd_steps_reduce_mse() {
        let (client, device) = cpu_setup();
        let cfg = cfg64(2, false);
        let mut params = init_params(&cfg, 0x6d4e_0060);
        let target = Lcg(0x6d4e_0061).vec(BATCH * SEQ * HIDDEN, 0.5);
        let mut losses = Vec::new();
        for step in 0..3 {
            let block = build64(&client, &device, &cfg, &params);
            let x = Var::new(
                f64_tensor(&params[X], &[BATCH, SEQ, HIDDEN], &device),
                false,
            );
            let y = block.forward_train(&client, &x, None).unwrap();
            let t = Var::new(f64_tensor(&target, y.shape(), &device), false);
            let diff = var_sub(&y, &t, &client).unwrap();
            let sq = var_mul(&diff, &diff, &client).unwrap();
            let sum = var_sum(&sq, &[0, 1, 2], false, &client).unwrap();
            let loss = var_mul_scalar(&sum, 1.0 / target.len() as f64, &client).unwrap();
            losses.push(to_f64(loss.tensor())[0]);
            if step == 2 {
                break;
            }

            let grads = backward(&loss, &client).unwrap();
            let ids = param_ids(&block);
            let trainable: Vec<TensorId> = block
                .trainable_parameters()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            assert_eq!(trainable, ids, "the update covers every trainable weight");
            for (k, id) in ids.iter().enumerate() {
                let name = NAMES[k + 1];
                let g = to_f64(
                    grads
                        .get(*id)
                        .unwrap_or_else(|| panic!("{name}: no gradient")),
                );
                if step == 0 {
                    assert!(g.iter().all(|v| v.is_finite()), "{name}: finite gradient");
                    assert!(g.iter().any(|v| *v != 0.0), "{name}: non-zero gradient");
                }
                for (p, gv) in params[k + 1].iter_mut().zip(&g) {
                    *p -= LR * gv;
                }
            }
        }
        assert!(
            losses[1] < losses[0] && losses[2] < losses[1],
            "loss must fall: {losses:?}"
        );
    }
}
