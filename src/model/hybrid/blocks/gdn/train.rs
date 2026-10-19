//! Tracked training forward of a [`GdnBlock`]: the inference prefill chain
//! of `forward` and `prefill`, with every step on the autograd graph.
//!
//! Each call starts a fresh sequence: a zero conv window and a zero
//! delta-rule state. No `GdnState` is read or written. The final state
//! leaves the graph, so the `seq` tokens of one call form one
//! backpropagation segment.
//!
//! A Hadamard-rotated projection detaches its input, and an AWQ/GPTQ
//! decomposed base gives no input gradient. The forward refuses both
//! before it computes anything.

use super::gate_proj::GateProjections;
use super::heads::group_heads_var;
use super::layer::GdnBlock;
use crate::error::{Error, Result};
use crate::nn::var_ops::repeat_heads_tiled;
use crate::nn::{MaybeQuantLinear, MaybeRotatedLinear, var_causal_conv1d_silu, var_contiguous};
use crate::ops::traits::GatedDeltaNetOps;
use crate::ops::var_gdn_chunk;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{
    Var, var_add, var_l2_normalize, var_mul, var_narrow, var_reshape, var_sigmoid, var_softplus,
};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, MatmulOps, ScalarOps, ShapeOps, TensorOps,
    TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

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
    /// # Errors
    ///
    /// - [`Error::ModelError`] when `x` is not `[batch, seq, hidden_size]`.
    /// - [`Error::InvalidArgument`] when a projection is Hadamard-rotated or
    ///   has a decomposed AWQ/GPTQ base. `arg` names the projection.
    /// - numr and kernel errors propagate.
    pub fn forward_train<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
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
        self.check_train_input(x)?;
        let cfg = &self.cfg;
        let shape = x.shape();
        let (batch, seq) = (shape[0], shape[1]);
        let (h_k, h_v, s) = (cfg.key_heads, cfg.value_heads, cfg.state_size);
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();

        // 1-2. Projections and raw gates, all tracked.
        let qkv = tracked_forward(&self.attn_qkv, "attn_qkv", client, x)?;
        let z = tracked_forward(&self.attn_gate, "attn_gate", client, x)?;
        let (alpha, beta_raw) = self.gates.forward_var(client, x)?;
        let ssm_a = self.decay_var(client)?;

        // 3. Causal conv over a zero window, then SiLU. No bias.
        let qkv = var_causal_conv1d_silu(client, &qkv, &self.conv_weight, None, true)?;

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
        let beta = var_sigmoid(&beta_raw, client).map_err(Error::Numr)?;
        let g = var_add(&alpha, &self.ssm_dt_bias, client).map_err(Error::Numr)?;
        let g = var_softplus(&g, client).map_err(Error::Numr)?;
        let g = var_mul(&g, &ssm_a, client).map_err(Error::Numr)?;

        // 8. Chunked recurrence from a zero state.
        let (dtype, device) = (v.tensor().dtype(), v.tensor().device());
        let state = Tensor::<R>::zeros(&[batch, h_v, s, s], dtype, device).map_err(Error::Numr)?;
        let state = Var::new(state, false);
        let (o, _) = var_gdn_chunk(client, &q, &k, &v, &g, &beta, &state, cfg.chunk_size)?;

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

    /// The checks [`Self::forward_train`] runs before any compute: `x` is
    /// `[batch, seq, hidden_size]`, and every projection has a tracked path.
    ///
    /// # Errors
    ///
    /// As [`Self::forward_train`]: [`Error::ModelError`] for the shape,
    /// [`Error::InvalidArgument`] for a projection.
    pub(super) fn check_train_input(&self, x: &Var<R>) -> Result<()> {
        let shape = x.shape();
        if shape.len() != 3 || shape[2] != self.cfg.hidden_size {
            return Err(Error::ModelError {
                reason: format!(
                    "gdn: expected [batch, seq, {}], got {shape:?}",
                    self.cfg.hidden_size
                ),
            });
        }
        self.check_tracked()
    }

    /// Refuse every projection the tracked forward cannot carry a gradient
    /// through. Runs before any compute.
    fn check_tracked(&self) -> Result<()> {
        for (linear, name) in [
            (&self.attn_qkv, "attn_qkv"),
            (&self.attn_gate, "attn_gate"),
            (&self.ssm_out, "ssm_out"),
        ] {
            if linear.is_rotated() {
                return Err(rotated_error(name));
            }
            check_base(linear.base(), name)?;
        }
        if let GateProjections::Separate(gates) = &self.gates {
            check_base(gates.alpha.base(), "ssm_alpha")?;
            check_base(gates.beta.base(), "ssm_beta")?;
        }
        Ok(())
    }
}

/// The tracked forward of one projection: plain base, or base plus adapter.
fn tracked_forward<R, C>(
    linear: &MaybeRotatedLinear<R>,
    name: &'static str,
    client: &C,
    x: &Var<R>,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + QuantMatmulOps<R>
        + TypeConversionOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + BinaryOps<R> + MatmulOps<R> + DequantOps<R>,
{
    match linear {
        MaybeRotatedLinear::Plain(base) => base.forward(client, x),
        MaybeRotatedLinear::Lora(lora) => lora.forward(client, x),
        MaybeRotatedLinear::Rotated(_) => Err(rotated_error(name)),
    }
}

/// `qkv[.., start .. start + len]` reshaped to `shape`, tracked.
fn narrow_heads<R>(qkv: &Var<R>, start: usize, len: usize, shape: &[usize]) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + ShapeOps<R>,
{
    let part = var_narrow(qkv, 2, start, len).map_err(Error::Numr)?;
    let part = var_contiguous(&part)?;
    var_reshape(&part, shape).map_err(Error::Numr)
}

fn check_base<R: Runtime>(base: &MaybeQuantLinear<R>, name: &'static str) -> Result<()> {
    match base {
        MaybeQuantLinear::DecomposedQuant(_) => Err(Error::InvalidArgument {
            arg: name,
            reason: format!(
                "gdn.{name} has a decomposed AWQ/GPTQ base, which gives no input gradient. \
                 Load it as a dense or GGUF block-quantized weight to train the block"
            ),
        }),
        MaybeQuantLinear::Standard(_) | MaybeQuantLinear::Quantized(_) => Ok(()),
    }
}

fn rotated_error(name: &'static str) -> Error {
    Error::InvalidArgument {
        arg: name,
        reason: format!(
            "gdn.{name} is Hadamard-rotated, and the rotation detaches its input. \
             Load the block without the Hadamard attach to train it"
        ),
    }
}

#[cfg(test)]
mod tests {
    //! Value parity with the inference forward, gradients against central
    //! finite differences in F64, adapters over a frozen base, a short SGD
    //! run, and the refusal of projections without a tracked path.

    use super::super::GdnProjection;
    use super::super::forward::tests::block;
    use super::super::gate_proj::{GateProjections, SeparateGates};
    use super::super::layer::{GdnBlock, GdnWeights};
    use crate::error::Error;
    use crate::inference::GdnState;
    use crate::model::config::GdnConfig;
    use crate::nn::hadamard::HadamardRotation;
    use crate::nn::{Linear, LoraFactors, MaybeQuantLinear, RotatedLinear};
    use crate::quant::decomposed::{
        DecomposedQuantLinear, DecomposedQuantMethod, DecomposedQuantTensor,
    };
    use crate::test_utils::{Lcg, cpu_setup, to_f64, weighted_loss};
    use numr::autograd::{Var, backward, var_mul, var_mul_scalar, var_sub, var_sum};
    use numr::dtype::DType;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::{Tensor, TensorId};

    const HIDDEN: usize = 8;
    const H_V: usize = 4;
    const S: usize = 4;
    const KERNEL: usize = 4;
    const SEQ: usize = 9;
    const BATCH: usize = 2;
    const RANK: usize = 2;
    /// 9 tokens in chunks of 4: the last chunk is padded.
    const CHUNK: usize = 4;
    /// Central-difference step for the F64 checks.
    const STEP: f64 = 4e-6;
    /// Relative bound, against the larger of `|fd|` and the parameter's
    /// largest analytic gradient.
    const TOL: f64 = 1e-6;
    const PROBES: usize = 8;
    const LR: f64 = 1e-2;

    /// Indices into the F64 parameter set.
    const X: usize = 0;
    const QKV: usize = 1;
    const GATE: usize = 2;
    const ALPHA: usize = 3;
    const BETA: usize = 4;
    const OUT: usize = 5;
    const CONV: usize = 6;
    const A_LOG: usize = 7;
    const DT_BIAS: usize = 8;
    const NORM: usize = 9;
    const NAMES: [&str; 10] = [
        "x",
        "attn_qkv",
        "attn_gate",
        "ssm_alpha",
        "ssm_beta",
        "ssm_out",
        "conv",
        "a_log",
        "dt_bias",
        "norm",
    ];

    type T = Tensor<CpuRuntime>;

    fn f32_tensor(data: &[f64], shape: &[usize], device: &CpuDevice) -> T {
        let data: Vec<f32> = data.iter().map(|v| *v as f32).collect();
        T::from_slice(&data, shape, device).unwrap()
    }

    fn f64_tensor(data: &[f64], shape: &[usize], device: &CpuDevice) -> T {
        T::from_slice(data, shape, device).unwrap()
    }

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
                let got = block.forward_train(&client, &Var::new(x, false)).unwrap();
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

    // ---- F64 block with every weight trainable ----

    fn cfg64(key_heads: usize, v_grouped: bool) -> GdnConfig {
        GdnConfig {
            hidden_size: HIDDEN,
            conv_kernel: KERNEL,
            state_size: S,
            key_heads,
            value_heads: H_V,
            inner_size: H_V * S,
            rms_eps: 1e-6,
            chunk_size: CHUNK,
            v_grouped,
        }
    }

    fn shapes(cfg: &GdnConfig) -> [Vec<usize>; 10] {
        let (qkv, value) = (cfg.qkv_dim(), cfg.value_dim());
        [
            vec![BATCH, SEQ, HIDDEN],
            vec![qkv, HIDDEN],
            vec![value, HIDDEN],
            vec![H_V, HIDDEN],
            vec![H_V, HIDDEN],
            vec![HIDDEN, value],
            vec![qkv, KERNEL],
            vec![H_V],
            vec![H_V],
            vec![S],
        ]
    }

    fn init_params(cfg: &GdnConfig, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Lcg(seed);
        let in_scale = 0.5 / (HIDDEN as f64).sqrt();
        let out_scale = 0.5 / (cfg.value_dim() as f64).sqrt();
        let scales = [
            1.0, in_scale, in_scale, in_scale, in_scale, out_scale, 0.5, 0.0, 0.5, 0.0,
        ];
        shapes(cfg)
            .iter()
            .zip(scales)
            .enumerate()
            .map(|(i, (shape, scale))| {
                let n: usize = shape.iter().product();
                match i {
                    A_LOG | NORM => (0..n).map(|_| rng.uniform_f64(0.5, 1.5)).collect(),
                    _ => rng.vec(n, scale),
                }
            })
            .collect()
    }

    /// A trainable F64 block with separate dense gates, so `ssm_alpha` and
    /// `ssm_beta` own their weights.
    fn build64(
        client: &CpuClient,
        device: &CpuDevice,
        cfg: &GdnConfig,
        p: &[Vec<f64>],
    ) -> GdnBlock<CpuRuntime> {
        let sh = shapes(cfg);
        let t = |i: usize| f64_tensor(&p[i], &sh[i], device);
        let lin = |i: usize| MaybeQuantLinear::Standard(Linear::new(t(i), None, true));
        let ssm_a: Vec<f64> = p[A_LOG].iter().map(|a| -a.exp()).collect();
        let weights = GdnWeights {
            attn_qkv: lin(QKV).into(),
            attn_gate: lin(GATE).into(),
            ssm_alpha: lin(ALPHA),
            ssm_beta: lin(BETA),
            ssm_out: lin(OUT).into(),
            ssm_conv1d: t(CONV),
            ssm_a: f64_tensor(&ssm_a, &[H_V], device),
            ssm_dt_bias: t(DT_BIAS),
            ssm_norm: t(NORM),
        };
        let mut block = GdnBlock::new(cfg.clone(), weights).unwrap();
        block.gates = GateProjections::Separate(Box::new(SeparateGates {
            alpha: lin(ALPHA).into(),
            beta: lin(BETA).into(),
        }));
        block.with_trainable(client, true).unwrap()
    }

    /// Weight ids in parameter-index order, `QKV..=NORM`.
    fn param_ids(block: &GdnBlock<CpuRuntime>) -> Vec<TensorId> {
        let GateProjections::Separate(gates) = &block.gates else {
            panic!("build64 makes separate gates");
        };
        vec![
            block.attn_qkv.weight().unwrap().id(),
            block.attn_gate.weight().unwrap().id(),
            gates.alpha.weight().unwrap().id(),
            gates.beta.weight().unwrap().id(),
            block.ssm_out.weight().unwrap().id(),
            block.conv_weight.id(),
            block.a_log.as_ref().unwrap().id(),
            block.ssm_dt_bias.id(),
            block.norm.weight().id(),
        ]
    }

    fn loss_value(
        client: &CpuClient,
        device: &CpuDevice,
        cfg: &GdnConfig,
        p: &[Vec<f64>],
        w: &[f64],
    ) -> f64 {
        let block = build64(client, device, cfg, p);
        let x = Var::new(f64_tensor(&p[X], &[BATCH, SEQ, HIDDEN], device), false);
        let y = block.forward_train(client, &x).unwrap();
        to_f64(y.tensor()).iter().zip(w).map(|(a, b)| a * b).sum()
    }

    fn check_gradients(key_heads: usize, v_grouped: bool) {
        let (client, device) = cpu_setup();
        let cfg = cfg64(key_heads, v_grouped);
        let params = init_params(&cfg, 0x6d4e_0040 + key_heads as u64);
        let w_data = Lcg(0x6d4e_0041).vec(BATCH * SEQ * HIDDEN, 1.0);

        let block = build64(&client, &device, &cfg, &params);
        let x = Var::new(f64_tensor(&params[X], &[BATCH, SEQ, HIDDEN], &device), true);
        let y = block.forward_train(&client, &x).unwrap();
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
        let y = block.forward_train(&client, &x).unwrap();
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
            let y = block.forward_train(&client, &x).unwrap();
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

    // ---- Refusal of projections without a tracked path ----

    fn expect_refusal(block: &GdnBlock<CpuRuntime>, want: &str) {
        let (client, device) = cpu_setup();
        let x = T::zeros(&[1, SEQ, HIDDEN], DType::F32, &device).unwrap();
        match block.forward_train(&client, &Var::new(x, false)) {
            Err(Error::InvalidArgument { arg, reason }) => {
                assert_eq!(arg, want);
                assert!(reason.contains(want), "{reason}");
            }
            Err(other) => panic!("{want}: expected InvalidArgument, got {other}"),
            Ok(_) => panic!("{want}: forward_train must refuse"),
        }
    }

    fn decomposed(device: &CpuDevice, out: usize, inp: usize) -> MaybeQuantLinear<CpuRuntime> {
        let z = |shape: &[usize]| T::zeros(shape, DType::F32, device).unwrap();
        let weight = DecomposedQuantTensor::new(
            z(&[inp, 1]),
            z(&[1, out]),
            z(&[1, out]),
            None,
            DecomposedQuantMethod::Awq { group_size: 128 },
            vec![out, inp],
        );
        MaybeQuantLinear::DecomposedQuant(Box::new(DecomposedQuantLinear::new(weight, None)))
    }

    #[test]
    fn refuses_rotated_projection() {
        let (_, device) = cpu_setup();
        let mut block = block(&device, 0x6d4e_0035, false);
        let signs: Vec<i8> = (0..HIDDEN)
            .map(|i| if i % 2 == 0 { -1 } else { 1 })
            .collect();
        let rotation =
            HadamardRotation::<CpuRuntime>::new(HIDDEN, Some(&signs), DType::F32, &device).unwrap();
        let w = block.attn_qkv.weight().unwrap().tensor().clone();
        let base = MaybeQuantLinear::Standard(Linear::new(w, None, false));
        block.attn_qkv = RotatedLinear::new(base, rotation).unwrap().into();
        expect_refusal(&block, "attn_qkv");
    }

    #[test]
    fn refuses_decomposed_base() {
        let (_, device) = cpu_setup();
        let mut out_block = block(&device, 0x6d4e_0036, false);
        let value_dim = out_block.config().value_dim();
        out_block.ssm_out = decomposed(&device, HIDDEN, value_dim).into();
        expect_refusal(&out_block, "ssm_out");

        let mut gate_block = block(&device, 0x6d4e_0037, false);
        let beta = T::zeros(&[H_V, HIDDEN], DType::F32, &device).unwrap();
        gate_block.gates = GateProjections::Separate(Box::new(SeparateGates {
            alpha: decomposed(&device, H_V, HIDDEN).into(),
            beta: MaybeQuantLinear::Standard(Linear::new(beta, None, false)).into(),
        }));
        expect_refusal(&gate_block, "ssm_alpha");
    }
}
