//! Gated DeltaNet inference forward, driven by a per-layer
//! [`GdnState`](crate::inference::GdnState).
//!
//! Ports `build_layer_attn_linear` (`src/models/qwen35.cpp`) of llama.cpp
//! step for step:
//!
//! 1. `qkv = attn_qkv(x)`, `z = attn_gate(x)`
//! 2. `alpha = ssm_alpha(x)`, `beta_raw = ssm_beta(x)`
//! 3. causal depthwise conv over `qkv` with the carried window, then SiLU
//! 4. split q `[H_k, S]`, k `[H_k, S]`, v `[H_v, S]`
//! 5. L2-normalize q and k with `eps = rms_eps`
//! 6. tile q and k to `H_v` heads (`ggml_repeat_4d`)
//! 7. `beta = sigmoid(beta_raw)`, `g = ssm_a * softplus(alpha + ssm_dt_bias)`
//! 8. `gdn_chunk_prefill` for `seq > 1`; for `seq == 1` steps 4 to 8 are one
//!    `gdn_step_from_conv` call, fused into a single kernel where the backend
//!    has one
//! 9. `silu(z) * rms_norm(o)` with `ssm_norm`
//! 10. optional `group_heads`, then `ssm_out`

use super::layer::{GdnBlock, group_heads};
use crate::error::{Error, Result};
use crate::inference::GdnState;
use crate::model::traits::ModelClient;
use crate::nn::MaybeRotatedLinear;
use crate::nn::causal_conv1d;
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ConvOps, FwhtOps, IndexingOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> GdnBlock<R> {
    /// Inference forward. `x` is the `attn_norm`-ed hidden state
    /// `[batch, seq, hidden_size]`; the result is the `ssm_out` projection,
    /// `[batch, seq, hidden_size]`, without the residual.
    ///
    /// `seq > 1` runs the chunked prefill, `seq == 1` one decode step. Both
    /// read `state` as the left context and write it back in place. The
    /// output is a detached `Var`.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `x` is not `[batch, seq, hidden_size]` or
    /// `state` was built for another batch size.
    pub fn forward<C>(&self, client: &C, x: &Var<R>, state: &mut GdnState<R>) -> Result<Var<R>>
    where
        C: ModelClient<R> + ConvOps<R> + FwhtOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + ConvOps<R>
            + DequantOps<R>,
    {
        let (out, window, ssm) = self.forward_core(client, x, state.conv(), state.ssm())?;
        state.update(client, &window, &ssm)?;
        Ok(out)
    }

    /// The block math over an explicit left context: `conv_state`
    /// `[batch, qkv_dim, conv_kernel - 1]` and `ssm_state`
    /// `[batch, value_heads, S, S]`. Returns the output plus the new conv
    /// window and delta-rule state, each a fresh tensor; the caller copies
    /// them into the state in place (`GdnState::update` for the eager
    /// path, `GdnState::update_shared` for graph replay).
    pub(super) fn forward_core<C>(
        &self,
        client: &C,
        x: &Var<R>,
        conv_state: &Tensor<R>,
        ssm_state: &Tensor<R>,
    ) -> Result<(Var<R>, Tensor<R>, Tensor<R>)>
    where
        C: ModelClient<R> + ConvOps<R> + FwhtOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + ConvOps<R>
            + DequantOps<R>,
    {
        let cfg = &self.cfg;
        let shape = x.shape();
        if shape.len() != 3 || shape[2] != cfg.hidden_size {
            return Err(Error::ModelError {
                reason: format!(
                    "gdn: expected [batch, seq, {}], got {shape:?}",
                    cfg.hidden_size
                ),
            });
        }
        let (batch, seq) = (shape[0], shape[1]);
        let state_batch = conv_state.shape().first().copied().unwrap_or(0);
        if state_batch != batch {
            return Err(Error::ModelError {
                reason: format!("gdn: state batch {state_batch} != input batch {batch}"),
            });
        }
        let (h_k, h_v, s) = (cfg.key_heads, cfg.value_heads, cfg.state_size);
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();

        // 1. Projections. `attn_qkv` and `attn_gate` share `x`, so they run
        // through one `forward_batch` call: any Hadamard rotation the two
        // share runs once, and each fuses into one `quant_matmul_batch`.
        let mut projected =
            MaybeRotatedLinear::forward_batch(&[&self.attn_qkv, &self.attn_gate], client, x)?
                .into_iter();
        let qkv = projected.next().ok_or_else(|| Error::ModelError {
            reason: "gdn: forward_batch returned no attn_qkv output".to_string(),
        })?;
        let z = projected.next().ok_or_else(|| Error::ModelError {
            reason: "gdn: forward_batch returned no attn_gate output".to_string(),
        })?;

        // 2. Raw gate projections. `ssm_a` already holds `-exp(A_log)`.
        let alpha = self.ssm_alpha.forward(client, x)?;
        let beta_raw = self.ssm_beta.forward(client, x)?;

        // 3. Causal conv with the carried window, then SiLU.
        let qkv_ncl = qkv
            .tensor()
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()?;
        let (conv_out, window) =
            causal_conv1d(client, &qkv_ncl, &self.conv_weight, None, conv_state)?;
        let conv_out = client.silu(&conv_out).map_err(Error::Numr)?;
        let qkv = conv_out
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()?;

        let (o, ssm) = if seq == 1 {
            // 4-8. One call: split, L2 norm, tiled repeat, gates, step.
            client.gdn_step_from_conv(
                &qkv,
                alpha.tensor(),
                beta_raw.tensor(),
                &self.ssm_dt_bias,
                &self.ssm_a,
                ssm_state,
                h_k,
                key_dim,
                value_dim,
                cfg.rms_eps,
            )?
        } else {
            // 4-8. The same chain as primitives, then the chunked recurrence.
            self.prefill_recurrence(client, &qkv, alpha.tensor(), beta_raw.tensor(), ssm_state)?
        };

        // 9. silu(z) * rms_norm(o), per head.
        let z = z
            .tensor()
            .contiguous()?
            .reshape(&[batch, seq, h_v, s])
            .map_err(Error::Numr)?;
        let o = self
            .norm
            .forward(client, &Var::new(o, false), &Var::new(z, false))?;

        // 10. Flatten heads, reorder for a grouped `ssm_out`, project.
        let o = o
            .tensor()
            .contiguous()?
            .reshape(&[batch, seq, value_dim])
            .map_err(Error::Numr)?;
        let o = if cfg.v_grouped {
            group_heads(&o, h_k, cfg.head_repeat(), s)?
        } else {
            o
        };
        let out = self.ssm_out.forward(client, &Var::new(o, false))?;
        Ok((out, window, ssm))
    }
}

#[cfg(test)]
mod tests {
    use super::super::layer::GdnWeights;
    use super::*;
    use crate::model::config::GdnConfig;
    use crate::nn::hadamard::HadamardRotation;
    use crate::nn::linear::RotatedLinear;
    use crate::nn::{Linear, MaybeQuantLinear, MaybeRotatedLinear};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    const HIDDEN: usize = 8;
    const H_K: usize = 2;
    const H_V: usize = 4;
    const S: usize = 4;
    const KERNEL: usize = 4;
    const SEQ: usize = 9;

    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
        }

        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.next_f32()
        }

        fn tensor(
            &mut self,
            device: &CpuDevice,
            shape: &[usize],
            scale: f32,
        ) -> Tensor<CpuRuntime> {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| self.uniform(-scale, scale)).collect();
            Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
        }
    }

    fn cfg(v_grouped: bool) -> GdnConfig {
        GdnConfig {
            hidden_size: HIDDEN,
            conv_kernel: KERNEL,
            state_size: S,
            key_heads: H_K,
            value_heads: H_V,
            inner_size: H_V * S,
            rms_eps: 1e-6,
            chunk_size: 64,
            v_grouped,
        }
    }

    fn plain(w: Tensor<CpuRuntime>) -> MaybeQuantLinear<CpuRuntime> {
        MaybeQuantLinear::Standard(Linear::new(w, None, false))
    }

    fn rotated_plain(w: Tensor<CpuRuntime>) -> MaybeRotatedLinear<CpuRuntime> {
        MaybeRotatedLinear::Plain(plain(w))
    }

    fn rotated(
        w: Tensor<CpuRuntime>,
        rotation: HadamardRotation<CpuRuntime>,
    ) -> MaybeRotatedLinear<CpuRuntime> {
        RotatedLinear::new(plain(w), rotation).unwrap().into()
    }

    fn block(device: &CpuDevice, seed: u64, v_grouped: bool) -> GdnBlock<CpuRuntime> {
        let cfg = cfg(v_grouped);
        let mut rng = Lcg(seed);
        let in_scale = 0.5 / (HIDDEN as f32).sqrt();
        let a: Vec<f32> = (0..H_V).map(|_| -rng.uniform(0.5, 1.5).exp()).collect();
        let weights = GdnWeights {
            attn_qkv: rotated_plain(rng.tensor(device, &[cfg.qkv_dim(), HIDDEN], in_scale)),
            attn_gate: rotated_plain(rng.tensor(device, &[cfg.value_dim(), HIDDEN], in_scale)),
            ssm_alpha: plain(rng.tensor(device, &[H_V, HIDDEN], in_scale)),
            ssm_beta: plain(rng.tensor(device, &[H_V, HIDDEN], in_scale)),
            ssm_out: rotated_plain(rng.tensor(
                device,
                &[HIDDEN, cfg.value_dim()],
                0.5 / (cfg.value_dim() as f32).sqrt(),
            )),
            ssm_conv1d: rng.tensor(device, &[cfg.qkv_dim(), KERNEL], 0.5),
            ssm_a: Tensor::<CpuRuntime>::from_slice(&a, &[H_V], device).unwrap(),
            ssm_dt_bias: rng.tensor(device, &[H_V], 0.5),
            ssm_norm: rng.tensor(device, &[S], 1.0),
        };
        GdnBlock::new(cfg, weights).unwrap()
    }

    /// Same layout as [`block`], but `attn_qkv` and `attn_gate` are
    /// `Rotated`, sharing one `HadamardRotation`.
    fn block_with_rotated_qkv_gate(device: &CpuDevice, seed: u64) -> GdnBlock<CpuRuntime> {
        let cfg = cfg(false);
        let mut rng = Lcg(seed);
        let in_scale = 0.5 / (HIDDEN as f32).sqrt();
        let a: Vec<f32> = (0..H_V).map(|_| -rng.uniform(0.5, 1.5).exp()).collect();
        let signs: Vec<i8> = (0..HIDDEN)
            .map(|i| if i % 2 == 0 { -1 } else { 1 })
            .collect();
        let rotation =
            HadamardRotation::<CpuRuntime>::new(HIDDEN, Some(&signs), DType::F32, device).unwrap();
        let weights = GdnWeights {
            attn_qkv: rotated(
                rng.tensor(device, &[cfg.qkv_dim(), HIDDEN], in_scale),
                rotation.clone(),
            ),
            attn_gate: rotated(
                rng.tensor(device, &[cfg.value_dim(), HIDDEN], in_scale),
                rotation,
            ),
            ssm_alpha: plain(rng.tensor(device, &[H_V, HIDDEN], in_scale)),
            ssm_beta: plain(rng.tensor(device, &[H_V, HIDDEN], in_scale)),
            ssm_out: rotated_plain(rng.tensor(
                device,
                &[HIDDEN, cfg.value_dim()],
                0.5 / (cfg.value_dim() as f32).sqrt(),
            )),
            ssm_conv1d: rng.tensor(device, &[cfg.qkv_dim(), KERNEL], 0.5),
            ssm_a: Tensor::<CpuRuntime>::from_slice(&a, &[H_V], device).unwrap(),
            ssm_dt_bias: rng.tensor(device, &[H_V], 0.5),
            ssm_norm: rng.tensor(device, &[S], 1.0),
        };
        GdnBlock::new(cfg, weights).unwrap()
    }

    /// `block`'s `attn_qkv`/`attn_gate` (`Rotated`, sharing one rotation) run
    /// through [`MaybeRotatedLinear::forward_batch`] — the same call
    /// [`GdnBlock::forward`] step 1 makes — and through two individual
    /// [`MaybeRotatedLinear::forward`] calls, the pre-batching behavior.
    /// Bit-for-bit agreement here is what makes swapping step 1 to a single
    /// batched call safe.
    #[test]
    fn grouped_qkv_gate_rotated_matches_per_projection_forward() {
        let (client, device) = cpu_setup();
        let block = block_with_rotated_qkv_gate(&device, 0x6d4e_0005);
        let x = Var::new(Lcg(17).tensor(&device, &[1, SEQ, HIDDEN], 1.0), false);

        let mut batched =
            MaybeRotatedLinear::forward_batch(&[&block.attn_qkv, &block.attn_gate], &client, &x)
                .unwrap()
                .into_iter();
        let qkv_batched = batched.next().unwrap();
        let z_batched = batched.next().unwrap();

        let qkv_solo = block.attn_qkv.forward(&client, &x).unwrap();
        let z_solo = block.attn_gate.forward(&client, &x).unwrap();

        assert_eq!(
            qkv_batched.tensor().to_vec::<f32>(),
            qkv_solo.tensor().to_vec::<f32>()
        );
        assert_eq!(
            z_batched.tensor().to_vec::<f32>(),
            z_solo.tensor().to_vec::<f32>()
        );

        // End-to-end sanity: the block's own `forward` (which now calls
        // `forward_batch` for step 1) still runs to a finite result.
        let cfg = block.config().clone();
        let mut state = GdnState::<CpuRuntime>::zeros(&cfg, 1, DType::F32, &device).unwrap();
        let out = run(&client, &block, x.tensor(), &mut state);
        assert!(out.to_vec::<f32>().iter().all(|v| v.is_finite()));
    }

    fn run(
        client: &CpuClient,
        block: &GdnBlock<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
        state: &mut GdnState<CpuRuntime>,
    ) -> Tensor<CpuRuntime> {
        block
            .forward(client, &Var::new(x.clone(), false), state)
            .unwrap()
            .tensor()
            .clone()
    }

    fn max_abs_diff(a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> f32 {
        let a = a.to_vec::<f32>();
        let b = b.to_vec::<f32>();
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn prefill_shapes() {
        let (client, device) = cpu_setup();
        let block = block(&device, 0x6d4e_0001, false);
        let cfg = block.config().clone();
        let x = Lcg(7).tensor(&device, &[1, SEQ, HIDDEN], 1.0);
        let mut state = GdnState::<CpuRuntime>::zeros(&cfg, 1, DType::F32, &device).unwrap();

        let y = run(&client, &block, &x, &mut state);
        assert_eq!(y.shape(), &[1, SEQ, HIDDEN]);
        assert_eq!(state.conv().shape(), &[1, cfg.qkv_dim(), KERNEL - 1]);
        assert_eq!(state.ssm().shape(), &[1, H_V, S, S]);
        assert!(state.is_initialized());
        assert!(y.to_vec::<f32>().iter().all(|v| v.is_finite()));
    }

    /// Prefill of 9 tokens equals prefill of 5 then 4 carried decode steps.
    fn check_state_carry(v_grouped: bool) {
        let (client, device) = cpu_setup();
        let block = block(&device, 0x6d4e_0002, v_grouped);
        let cfg = block.config().clone();
        let x = Lcg(11).tensor(&device, &[1, SEQ, HIDDEN], 1.0);

        let mut full_state = GdnState::<CpuRuntime>::zeros(&cfg, 1, DType::F32, &device).unwrap();
        let full = run(&client, &block, &x, &mut full_state);

        let mut state = GdnState::<CpuRuntime>::zeros(&cfg, 1, DType::F32, &device).unwrap();
        let head = x.narrow(1, 0, 5).unwrap().contiguous().unwrap();
        let mut parts = vec![run(&client, &block, &head, &mut state)];
        for t in 5..SEQ {
            let xt = x.narrow(1, t, 1).unwrap().contiguous().unwrap();
            parts.push(run(&client, &block, &xt, &mut state));
        }
        let refs: Vec<&Tensor<CpuRuntime>> = parts.iter().collect();
        let stepped = client.cat(&refs, 1).unwrap();

        let y_diff = max_abs_diff(&full, &stepped);
        assert!(y_diff < 1e-4, "v_grouped={v_grouped}: output diff {y_diff}");
        let conv_diff = max_abs_diff(full_state.conv(), state.conv());
        assert!(conv_diff < 1e-4, "conv state diff {conv_diff}");
        let ssm_diff = max_abs_diff(full_state.ssm(), state.ssm());
        assert!(ssm_diff < 1e-4, "ssm state diff {ssm_diff}");
    }

    #[test]
    fn prefill_matches_prefill_then_decode() {
        check_state_carry(false);
    }

    #[test]
    fn prefill_matches_prefill_then_decode_grouped() {
        check_state_carry(true);
    }

    #[test]
    fn grouped_and_tiled_differ_only_by_head_order() {
        let (client, device) = cpu_setup();
        let tiled = block(&device, 0x6d4e_0003, false);
        let grouped = block(&device, 0x6d4e_0003, true);
        let cfg = tiled.config().clone();
        let x = Lcg(13).tensor(&device, &[1, 3, HIDDEN], 1.0);
        let mut s1 = GdnState::<CpuRuntime>::zeros(&cfg, 1, DType::F32, &device).unwrap();
        let mut s2 = GdnState::<CpuRuntime>::zeros(&cfg, 1, DType::F32, &device).unwrap();
        let a = run(&client, &tiled, &x, &mut s1);
        let b = run(&client, &grouped, &x, &mut s2);
        // Same weights, permuted `ssm_out` input: outputs differ, states agree.
        assert!(max_abs_diff(&a, &b) > 1e-6);
        assert!(max_abs_diff(s1.ssm(), s2.ssm()) < 1e-7);
    }

    #[test]
    fn rejects_state_batch_mismatch() {
        let (client, device) = cpu_setup();
        let block = block(&device, 0x6d4e_0004, false);
        let cfg = block.config().clone();
        let x = Lcg(3).tensor(&device, &[2, 2, HIDDEN], 1.0);
        let mut state = GdnState::<CpuRuntime>::zeros(&cfg, 1, DType::F32, &device).unwrap();
        assert!(
            block
                .forward(&client, &Var::new(x, false), &mut state)
                .is_err()
        );
    }
}
