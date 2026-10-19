//! Tracked training forward of a [`Qwen35AttentionBlock`]: full-sequence
//! causal self-attention with every step on the autograd graph and no KV
//! cache.
//!
//! The input side is the inference `rotated_heads`: projections, the
//! query/gate split, per-head q/k RMS norm and IMROPE, all tracked for a
//! plain or adapted projection. Attention runs through
//! [`var_flash_attention`] with `num_kv_heads` KV heads. Query head `h`
//! reads KV head `h / head_repeat()`, the grouped order of the inference
//! `repeat_kv`. The output side is `attn ⊙ sigmoid(gate)`, then
//! `attn_output`.
//!
//! A Hadamard-rotated projection detaches its input, and an AWQ/GPTQ
//! decomposed base gives no input gradient. The forward refuses both
//! before it computes anything.

use super::layer::Qwen35AttentionBlock;
use super::projections::Heads;
use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use crate::nn::RoPE;
use crate::ops::traits::AttnOutLayout;
use crate::ops::var_flash_attention;
use numr::autograd::{Var, var_reshape, var_sigmoid_mul};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Qwen35AttentionBlock<R> {
    /// Training forward. `x` is the `attn_norm`-ed hidden state
    /// `[batch, seq, hidden_size]`; the result is the `attn_output`
    /// projection `[batch, seq, hidden_size]`, without the residual.
    ///
    /// - `rope`: `[max_pos, rope_dim / 2]` table, as for [`Self::forward`]
    /// - `positions`: `[4, seq]` integer IMROPE streams `t, h, w, e`
    ///
    /// The math equals [`Self::forward`] on an empty KV cache. Gradients
    /// reach `x`, every projection weight or adapter factor that requires
    /// grad, and the q and k norm weights.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when a projection is Hadamard-rotated
    ///   or has a decomposed AWQ/GPTQ base. `arg` names the projection.
    /// - [`Error::ModelError`] when `x` is not `[batch, seq, hidden_size]`
    ///   or the rope table width is not `rope_dim / 2`.
    /// - numr and kernel errors propagate.
    pub fn forward_train<C>(
        &self,
        client: &C,
        x: &Var<R>,
        rope: &RoPE<R>,
        positions: &Tensor<R>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: ModelClient<R>,
    {
        self.check_tracked()?;
        let cfg = &self.cfg;
        let Heads {
            q,
            gate,
            k,
            v,
            batch,
            seq,
        } = self.rotated_heads(client, x, rope, positions)?;

        // Token-major output `[batch, seq, num_heads, head_dim]`, the
        // layout of `gate`.
        let attn = var_flash_attention(
            &q,
            &k,
            &v,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            true,
            0,
            None,
            AttnOutLayout::TokenMajor,
        )?;
        let gated = var_sigmoid_mul(&gate, &attn, client).map_err(Error::Numr)?;
        let gated = var_reshape(&gated, &[batch, seq, cfg.q_dim()]).map_err(Error::Numr)?;
        self.attn_output.forward(client, &gated)
    }

    /// Refuse every projection the tracked forward cannot carry a gradient
    /// through. Runs before any compute.
    pub(super) fn check_tracked(&self) -> Result<()> {
        self.attn_q.check_tracked("qwen35_attention.attn_q")?;
        self.attn_k.check_tracked("qwen35_attention.attn_k")?;
        self.attn_v.check_tracked("qwen35_attention.attn_v")?;
        self.attn_output
            .check_tracked("qwen35_attention.attn_output")
    }
}

#[cfg(test)]
mod tests {
    //! Value parity with the inference forward, GQA head order, gradients
    //! against central finite differences in F64, adapters over a frozen
    //! base, and the refusal of projections without a tracked path.

    use super::super::Qwen35AttentionProjection;
    use super::super::train_fixture::{
        BATCH, Fixture, HIDDEN, K, MAX_POS, NAMES, Q, SEQ, V, cfg_kv, cfg32, positions, rope,
    };
    use super::*;
    use crate::inference::KvCache;
    use crate::nn::hadamard::HadamardRotation;
    use crate::nn::{Linear, LoraFactors, MaybeQuantLinear, RotatedLinear};
    use crate::quant::decomposed::{
        DecomposedQuantLinear, DecomposedQuantMethod, DecomposedQuantTensor,
    };
    use crate::test_utils::{
        Lcg, cpu_setup, max_abs_diff_f64 as max_abs_diff, to_f64, weighted_loss,
    };
    use numr::autograd::backward;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::TensorId;

    /// Central-difference step for the F64 checks.
    const STEP: f64 = 1e-5;
    /// Relative bound, against the larger of `|fd|` and the parameter's
    /// largest analytic gradient.
    const TOL: f64 = 1e-6;
    const PROBES: usize = 8;
    const RANK: usize = 2;

    type Block = Qwen35AttentionBlock<CpuRuntime>;

    fn train(client: &CpuClient, block: &Block, x: &Var<CpuRuntime>) -> Var<CpuRuntime> {
        let device = x.tensor().device();
        let pos = positions(x.shape()[1], device);
        block.forward_train(client, x, &rope(device), &pos).unwrap()
    }

    // ---- Value parity with the inference forward (F32) ----

    #[test]
    fn forward_train_matches_inference_prefill() {
        let (client, device) = cpu_setup();
        for kv_heads in [2, 4] {
            let fixture = Fixture::f32(&cfg_kv(kv_heads), 0x3a7e_0010);
            let block = fixture.frozen_block(&device);
            for seq in [SEQ, 1] {
                let x = fixture
                    .x(&device, false)
                    .tensor()
                    .narrow(1, 0, seq)
                    .unwrap();
                let x = Var::new(x.contiguous().unwrap(), false);
                let mut cache = KvCache::<CpuRuntime>::new(
                    BATCH,
                    kv_heads,
                    MAX_POS,
                    MAX_POS,
                    fixture.cfg.head_dim,
                    DType::F32,
                    &device,
                )
                .unwrap();
                let pos = positions(seq, &device);
                let want = block
                    .forward(&client, &x, &rope(&device), &pos, &mut cache)
                    .unwrap();
                let got = train(&client, &block, &x);
                assert!(!got.requires_grad(), "nothing requires grad");
                assert_eq!(got.shape(), want.shape());
                let diff = max_abs_diff(&to_f64(got.tensor()), &to_f64(want.tensor()));
                assert!(diff < 1e-5, "kv_heads={kv_heads} seq={seq}: diff {diff}");
            }
        }
    }

    /// Four query heads over two KV heads equal four over four with each
    /// KV head's rows copied in grouped order: query head `h` reads KV
    /// head `h / 2`. Tiled order would read `h % 2` and differ.
    #[test]
    fn gqa_reads_kv_heads_in_grouped_order() {
        let (client, device) = cpu_setup();
        let grouped = Fixture::f32(&cfg32(), 0x3a7e_0011);
        let hd = grouped.cfg.head_dim;
        let mut full = grouped.clone();
        full.cfg = cfg_kv(4);
        let widen = |rows: &[f64], order: [usize; 4]| -> Vec<f64> {
            order
                .iter()
                .flat_map(|&head| rows[head * hd * HIDDEN..(head + 1) * hd * HIDDEN].to_vec())
                .collect()
        };
        let mut tiled = full.clone();
        for i in [K, V] {
            full.values[i] = widen(&grouped.values[i], [0, 0, 1, 1]);
            tiled.values[i] = widen(&grouped.values[i], [0, 1, 0, 1]);
        }
        let x = grouped.x(&device, false);
        let run = |f: &Fixture| to_f64(train(&client, &f.frozen_block(&device), &x).tensor());
        let want = run(&grouped);
        assert!(max_abs_diff(&run(&full), &want) < 1e-5, "grouped order");
        assert!(
            max_abs_diff(&run(&tiled), &want) > 1e-4,
            "tiled order differs"
        );
    }

    // ---- Gradients against central finite differences (F64) ----

    fn loss_value(client: &CpuClient, device: &CpuDevice, f: &Fixture, w: &[f64]) -> f64 {
        let y = train(client, &f.block(device, true), &f.x(device, false));
        to_f64(y.tensor()).iter().zip(w).map(|(a, b)| a * b).sum()
    }

    /// Ids in fixture-index order, `X..=K_NORM`.
    fn ids(block: &Block, x: &Var<CpuRuntime>) -> Vec<TensorId> {
        let weight = |l: &crate::nn::MaybeRotatedLinear<CpuRuntime>| l.weight().unwrap().id();
        vec![
            x.id(),
            weight(&block.attn_q),
            weight(&block.attn_k),
            weight(&block.attn_v),
            weight(&block.attn_output),
            block.qk_norm.q_norm().weight().id(),
            block.qk_norm.k_norm().weight().id(),
        ]
    }

    fn check_gradients(kv_heads: usize) {
        let (client, device) = cpu_setup();
        let fixture = Fixture::f64(&cfg_kv(kv_heads), 0x3a7e_0020 + kv_heads as u64);
        let block = fixture.block(&device, true);
        let x = fixture.x(&device, true);
        let y = train(&client, &block, &x);
        let w_data = Lcg(0x3a7e_0021).vec(y.tensor().numel(), 1.0);
        let w = Var::new(
            Tensor::from_slice(&w_data, y.shape(), &device).unwrap(),
            false,
        );
        let grads = backward(&weighted_loss(&client, &y, &w), &client).unwrap();

        let mut rng = Lcg(0x3a7e_0022);
        for (i, id) in ids(&block, &x).iter().enumerate() {
            let name = NAMES[i];
            let grad = grads
                .get(*id)
                .unwrap_or_else(|| panic!("{name}: no gradient"));
            // The block holds `attn_q` rows regrouped; FD perturbs stored rows.
            let grad = if i == Q {
                block.q_rows_to_checkpoint(grad).unwrap()
            } else {
                grad.clone()
            };
            let analytic = to_f64(&grad);
            assert_eq!(analytic.len(), fixture.values[i].len(), "{name}: size");
            let scale = analytic.iter().fold(1e-3f64, |m, g| m.max(g.abs()));
            for _ in 0..PROBES {
                let j = rng.index(fixture.values[i].len());
                let mut plus = fixture.clone();
                plus.values[i][j] += STEP;
                let mut minus = fixture.clone();
                minus.values[i][j] -= STEP;
                let fd = (loss_value(&client, &device, &plus, &w_data)
                    - loss_value(&client, &device, &minus, &w_data))
                    / (2.0 * STEP);
                assert!(
                    (analytic[j] - fd).abs() <= TOL * fd.abs().max(scale),
                    "kv_heads={kv_heads} {name}[{j}]: analytic {} vs fd {fd}",
                    analytic[j]
                );
            }
        }
    }

    #[test]
    fn gradients_match_finite_differences_gqa() {
        check_gradients(2);
    }

    #[test]
    fn gradients_match_finite_differences_mha() {
        check_gradients(4);
    }

    // ---- Adapters over a frozen base ----

    #[test]
    fn frozen_base_trains_only_lora_factors() {
        let (client, device) = cpu_setup();
        let fixture = Fixture::f32(&cfg32(), 0x3a7e_0030);
        let cfg = fixture.cfg.clone();
        let mut block = fixture.frozen_block(&device);
        let mut rng = Lcg(0x3a7e_0031);
        let mut factors = |out: usize, inp: usize| {
            let t = |data: Vec<f64>, shape: &[usize]| {
                let data: Vec<f32> = data.iter().map(|v| *v as f32).collect();
                Tensor::<CpuRuntime>::from_slice(&data, shape, &device).unwrap()
            };
            LoraFactors {
                lora_a: t(rng.vec(RANK * inp, 0.3), &[RANK, inp]),
                lora_b: t(rng.vec(out * RANK, 0.3), &[out, RANK]),
                scaling: 0.5,
            }
        };
        let targets = [
            (Qwen35AttentionProjection::Q, cfg.q_gate_dim(), HIDDEN),
            (Qwen35AttentionProjection::K, cfg.kv_dim(), HIDDEN),
            (Qwen35AttentionProjection::V, cfg.kv_dim(), HIDDEN),
            (Qwen35AttentionProjection::Output, HIDDEN, cfg.q_dim()),
        ];
        for (projection, out, inp) in targets {
            let f = factors(out, inp);
            block
                .attach_adapter_trainable(&client, projection, f, DType::F32, true)
                .unwrap();
        }

        let y = train(&client, &block, &fixture.x(&device, false));
        assert!(y.requires_grad());
        let w_data: Vec<f32> = Lcg(0x3a7e_0032)
            .vec(y.tensor().numel(), 1.0)
            .iter()
            .map(|v| *v as f32)
            .collect();
        let w = Var::new(
            Tensor::from_slice(&w_data, y.shape(), &device).unwrap(),
            false,
        );
        let grads = backward(&weighted_loss(&client, &y, &w), &client).unwrap();

        let trainable = block.trainable_parameters();
        assert_eq!(trainable.len(), 8, "two factors on each of four adapters");
        for (id, _) in &trainable {
            assert!(grads.get(*id).is_some(), "a LoRA factor gets a gradient");
        }
        for (name, var) in block.named_parameters() {
            if name.ends_with("base.weight") {
                assert!(
                    grads.get(var.id()).is_none(),
                    "frozen base gets none: {name}"
                );
            }
        }
    }

    // ---- Refusal of projections without a tracked path ----

    fn expect_refusal(block: &Block, want: &str) {
        let (client, device) = cpu_setup();
        let x = Var::new(
            Tensor::zeros(&[1, SEQ, HIDDEN], DType::F32, &device).unwrap(),
            false,
        );
        let pos = positions(SEQ, &device);
        match block.forward_train(&client, &x, &rope(&device), &pos) {
            Err(Error::InvalidArgument { arg, reason }) => {
                assert_eq!(arg, want);
                assert!(reason.contains(want), "{reason}");
            }
            Err(other) => panic!("{want}: expected InvalidArgument, got {other}"),
            Ok(_) => panic!("{want}: forward_train must refuse"),
        }
    }

    #[test]
    fn refuses_rotated_projection() {
        let (_, device) = cpu_setup();
        let mut block = Fixture::f32(&cfg32(), 0x3a7e_0040).frozen_block(&device);
        let signs: Vec<i8> = (0..HIDDEN)
            .map(|i| if i % 2 == 0 { -1 } else { 1 })
            .collect();
        let rotation =
            HadamardRotation::<CpuRuntime>::new(HIDDEN, Some(&signs), DType::F32, &device).unwrap();
        let w = block.attn_k.weight().unwrap().tensor().clone();
        let base = MaybeQuantLinear::Standard(Linear::new(w, None, false));
        block.attn_k = RotatedLinear::new(base, rotation).unwrap().into();
        expect_refusal(&block, "qwen35_attention.attn_k");
    }

    #[test]
    fn refuses_decomposed_base() {
        let (_, device) = cpu_setup();
        let mut block = Fixture::f32(&cfg32(), 0x3a7e_0041).frozen_block(&device);
        let (out, inp) = (HIDDEN, block.cfg.q_dim());
        let z = |shape: &[usize]| Tensor::<CpuRuntime>::zeros(shape, DType::F32, &device).unwrap();
        let weight = DecomposedQuantTensor::new(
            z(&[inp, 1]),
            z(&[1, out]),
            z(&[1, out]),
            None,
            DecomposedQuantMethod::Awq { group_size: 128 },
            vec![out, inp],
        );
        block.attn_output =
            MaybeQuantLinear::DecomposedQuant(Box::new(DecomposedQuantLinear::new(weight, None)))
                .into();
        expect_refusal(&block, "qwen35_attention.attn_output");
    }
}
