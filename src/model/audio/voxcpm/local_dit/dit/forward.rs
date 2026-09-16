//! The public `forward*` entry points of the estimator: input preparation
//! (`mu` tokenization, `cond` projection) around the core in `estimator.rs`.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::local_dit::loader::LocalDit;
use crate::model::traits::ModelClient;
use crate::nn::var_contiguous;
use crate::ops::FlashAttentionOps;
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> LocalDit<R> {
    /// Turn activation checkpointing on or off for every backbone layer.
    ///
    /// `on` trades ~33% extra compute for dropping each layer's
    /// intermediates during the forward pass and recomputing them during
    /// backward, which is what caps training VRAM. Default is `off`, so an
    /// inference path — including [`solve_euler`](Self::solve_euler) — pays
    /// nothing.
    pub fn set_activation_checkpointing(&mut self, on: bool) {
        self.activation_checkpointing = on;
    }

    /// Whether this stack runs its backbone layers with activation
    /// checkpointing.
    pub fn activation_checkpointing(&self) -> bool {
        self.activation_checkpointing
    }

    /// One estimator evaluation.
    ///
    /// - `x`: `[batch, patch_size, feat_dim]` — the current CFM sample.
    /// - `mu`: `[batch, mu_tokens * hidden_dim]` — the global-encoder
    ///   condition, reshaped to `mu_tokens` sequence positions (2 on this
    ///   checkpoint).
    /// - `t`: `[batch]` — the flow timestep, one scalar per sample.
    /// - `cond`: `[batch, patch_size, feat_dim]` — the prefix condition.
    /// - `dt`: `[batch]` — the mean-velocity delta. Zero at inference, but
    ///   still a live input: see the module docs.
    ///
    /// Returns `[batch, patch_size, feat_dim]` — the patch layout every
    /// caller holds (see the module docs for the reference's transposes).
    ///
    /// When
    /// [`set_activation_checkpointing`](Self::set_activation_checkpointing)
    /// is on, every backbone layer runs through
    /// [`BidirectionalLayer::forward_checkpointed`](crate::model::audio::voxcpm::bidirectional::layer::BidirectionalLayer::forward_checkpointed)
    /// — same ops, same order, same output values, at ~33% extra compute.
    pub fn forward<C>(
        &self,
        client: &C,
        x: &Var<R>,
        mu: &Var<R>,
        t: &Var<R>,
        cond: &Var<R>,
        dt: &Var<R>,
    ) -> Result<Var<R>>
    where
        // `'static` is what `forward_checkpointed` adds: the closure numr
        // stores for the backward recompute owns the client.
        C: ModelClient<R> + TypeConversionOps<R> + 'static,
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
            + TypeConversionOps<R>
            + DequantOps<R>
            + FlashAttentionOps<R>,
    {
        let batch = self.check_patch_input("x", x, None)?;
        let mu_tokens = self.check_mu(mu, batch)?;

        // `mu.view(b, -1, hidden)`: mu_tokens sequence positions. Callers
        // that evaluate the estimator many times over one FIXED `mu` (the
        // Euler integrator: see `sampler::euler::solve_euler`) should tokenize
        // it ONCE and call `forward_with_mu_tokens` directly instead of
        // paying this reshape on every step.
        let mu_contig = var_contiguous(mu)?;
        let mu_tok =
            var_reshape(&mu_contig, &[batch, mu_tokens, self.hidden_dim]).map_err(Error::Numr)?;

        self.forward_with_mu_tokens(client, x, &mu_tok, t, cond, dt)
    }

    /// Same estimator evaluation as [`forward`](Self::forward), but `mu`
    /// arrives ALREADY reshaped to `[batch, mu_tokens, hidden_dim]` sequence
    /// tokens instead of the flat `[batch, mu_tokens * hidden_dim]` the
    /// reference passes.
    ///
    /// `mu` is IDENTICAL across every step of one Euler solve
    /// ([`solve_euler`](Self::solve_euler) builds `mu_in` once outside the
    /// step loop), so re-deriving `mu_tok` from it on every estimator call
    /// wastes a reshape + contiguous. This is the entry point that lets a
    /// multi-step caller tokenize once and pass the same `mu_tok` to every
    /// step; [`forward`] itself does the one-shot tokenization and delegates
    /// here so single-call sites (training, tests, `boostr-audio`) keep the
    /// original signature.
    ///
    /// `mu_tok`'s middle dimension IS `mu_tokens` — read from its shape, never
    /// re-derived from a flat width, since there is no flat `mu` here to
    /// derive it from.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_mu_tokens<C>(
        &self,
        client: &C,
        x: &Var<R>,
        mu_tok: &Var<R>,
        t: &Var<R>,
        cond: &Var<R>,
        dt: &Var<R>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R> + 'static,
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
            + TypeConversionOps<R>
            + DequantOps<R>
            + FlashAttentionOps<R>,
    {
        let cond_h = self.project_cond(client, cond)?;
        self.forward_prepared(client, x, mu_tok, t, &cond_h, dt)
    }

    /// `cond_proj(cond)`: the prefix condition as `[batch, patch_size,
    /// hidden_dim]` sequence tokens.
    ///
    /// `cond` is IDENTICAL across every step of one Euler solve, so the
    /// integrators project it ONCE and hand the result to
    /// [`forward_prepared`](Self::forward_prepared) on every step — the same
    /// hoist `mu_tok` gets. [`forward_with_mu_tokens`](Self::forward_with_mu_tokens)
    /// does this projection per call for single-shot callers.
    pub fn project_cond<C>(&self, client: &C, cond: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + DequantOps<R>,
    {
        self.check_patch_input("cond", cond, None)?;
        // `Linear` reshapes its input: a no-op for the dense tensors every
        // caller holds, a copy only for a strided view.
        self.cond_proj.forward(client, &var_contiguous(cond)?)
    }
}

#[cfg(test)]
mod tests {
    //! Tests for [`LocalDit::forward`] — the estimator forward pass.
    //!
    //! Weights are tiny and synthetic; these pin SHAPE and the output SLICE
    //! WINDOW, which are the two things the reference makes easy to get wrong.

    use super::super::super::tests::{FEAT_DIM, HIDDEN_DIM, MU_TOKENS, PATCH_SIZE, model, t};
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    struct Inputs {
        x: Var<CpuRuntime>,
        mu: Var<CpuRuntime>,
        t: Var<CpuRuntime>,
        cond: Var<CpuRuntime>,
        dt: Var<CpuRuntime>,
    }

    fn inputs(batch: usize, x_seed: f32, cond_seed: f32, device: &CpuDevice) -> Inputs {
        Inputs {
            x: Var::new(t(&[batch, PATCH_SIZE, FEAT_DIM], x_seed, device), false),
            mu: Var::new(t(&[batch, MU_TOKENS * HIDDEN_DIM], 1.3, device), false),
            t: Var::new(t(&[batch], 2.1, device), false),
            cond: Var::new(t(&[batch, PATCH_SIZE, FEAT_DIM], cond_seed, device), false),
            // `dt = 0` is the inference value, and it is NOT a no-op branch.
            dt: Var::new(
                Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; batch], &[batch], device).unwrap(),
                false,
            ),
        }
    }

    fn run(client: &CpuClient, model: &LocalDit<CpuRuntime>, i: &Inputs) -> Vec<f32> {
        let out = model
            .forward(client, &i.x, &i.mu, &i.t, &i.cond, &i.dt)
            .unwrap();
        assert_eq!(out.shape(), &[i.x.shape()[0], PATCH_SIZE, FEAT_DIM]);
        out.tensor().contiguous().unwrap().to_vec()
    }

    #[test]
    fn output_shape_is_batch_patch_size_feat_dim() {
        let (client, device) = cpu_setup();
        let m = model(2, &device);
        let out = run(&client, &m, &inputs(3, 0.9, 1.7, &device));
        assert_eq!(out.len(), 3 * FEAT_DIM * PATCH_SIZE);
    }

    /// [`LocalDit::forward_with_mu_tokens`] fed the SAME reshape `forward`
    /// derives internally must return the identical output — this is the
    /// contract `solve_euler` relies on to tokenize `mu` once outside the
    /// step loop instead of once per step.
    #[test]
    fn forward_with_mu_tokens_matches_forward() {
        let (client, device) = cpu_setup();
        let m = model(2, &device);
        let i = inputs(2, 0.9, 1.7, &device);

        let via_forward = run(&client, &m, &i);

        let mu_tok =
            var_reshape(&var_contiguous(&i.mu).unwrap(), &[2, MU_TOKENS, HIDDEN_DIM]).unwrap();
        let out = m
            .forward_with_mu_tokens(&client, &i.x, &mu_tok, &i.t, &i.cond, &i.dt)
            .unwrap();
        let via_pre_tokenized = out.tensor().contiguous().unwrap().to_vec::<f32>();

        assert_eq!(
            via_forward, via_pre_tokenized,
            "pre-tokenized mu must produce the exact same output as forward's own reshape"
        );
    }

    /// Cross-layer deferred-residual fusion in the 2-layer backbone
    /// `model(2, ...)` builds must produce the SAME numbers whether or not
    /// it takes the fused path. `x.requires_grad() == true` propagates
    /// through the assembled `[mu, t, cond, x]` sequence into every layer's
    /// `forward_with_pending_residual`, forcing the whole 2-layer stack AND
    /// the final norm fold down the unfused branch.
    #[test]
    fn cross_layer_fusion_matches_unfused_across_two_layers() {
        let (client, device) = cpu_setup();
        let m = model(2, &device);
        let base = inputs(2, 0.9, 1.7, &device);

        let fused = run(&client, &m, &base);

        let unfused_inputs = Inputs {
            x: Var::new(base.x.tensor().clone(), true),
            mu: base.mu,
            t: base.t,
            cond: base.cond,
            dt: base.dt,
        };
        let unfused = run(&client, &m, &unfused_inputs);

        assert_eq!(fused.len(), unfused.len());
        for (a, b) in fused.iter().zip(&unfused) {
            assert!(
                (a - b).abs() < 1e-5,
                "cross-layer fused vs unfused diverged: {a} vs {b}"
            );
        }
    }

    /// The slice window is `prefix + mu_tokens + 1 ..`, i.e. exactly the trailing
    /// `x` positions. With NO transformer layers nothing mixes across positions,
    /// so the returned window must depend on `x` alone: change `x` and the output
    /// moves, change `cond` and it does not. A wrong window (e.g. starting at the
    /// `cond` block, or including `mu`/`t`) flips both assertions.
    #[test]
    fn slice_window_keeps_only_the_trailing_x_positions() {
        let (client, device) = cpu_setup();
        let m = model(0, &device);

        let base = run(&client, &m, &inputs(2, 0.9, 1.7, &device));
        let other_x = run(&client, &m, &inputs(2, 4.5, 1.7, &device));
        let other_cond = run(&client, &m, &inputs(2, 0.9, 6.2, &device));

        let max_delta = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b.iter())
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(
            max_delta(&base, &other_x) > 1e-4,
            "output must respond to x: base={base:?} other={other_x:?}"
        );
        assert!(
            max_delta(&base, &other_cond) < 1e-6,
            "with no layers the x window cannot see cond: base={base:?} other={other_cond:?}"
        );
    }

    /// With the bidirectional stack in place every position attends every other,
    /// so `cond` DOES reach the `x` window. Guards against a "fix" that drops
    /// `cond` (or `mu`/`t`) from the assembled sequence entirely.
    #[test]
    fn cond_reaches_the_x_window_through_the_bidirectional_stack() {
        let (client, device) = cpu_setup();
        let m = model(2, &device);

        let base = run(&client, &m, &inputs(2, 0.9, 1.7, &device));
        let other_cond = run(&client, &m, &inputs(2, 0.9, 6.2, &device));
        let max_delta = base
            .iter()
            .zip(other_cond.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_delta > 1e-5,
            "cond must influence the x window: base={base:?} other={other_cond:?}"
        );
    }

    /// `dt = 0` is not a dead branch: `SinusoidalPosEmb(0) = [0..0, 1..1]`, so
    /// `delta_time_mlp` adds a real constant. Changing `dt` must change the
    /// output.
    #[test]
    fn dt_branch_contributes() {
        let (client, device) = cpu_setup();
        let m = model(1, &device);

        let mut i = inputs(2, 0.9, 1.7, &device);
        let base = run(&client, &m, &i);
        i.dt = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.25f32, 0.5], &[2], &device).unwrap(),
            false,
        );
        let shifted = run(&client, &m, &i);
        let max_delta = base
            .iter()
            .zip(shifted.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max);
        assert!(max_delta > 1e-5, "dt must change the output");
    }

    #[test]
    fn rejects_wrong_shapes() {
        let (client, device) = cpu_setup();
        let m = model(1, &device);
        let good = inputs(2, 0.9, 1.7, &device);

        // x is 2D, not [batch, patch_size, feat_dim].
        let bad_x = Var::new(t(&[2, FEAT_DIM], 0.9, &device), false);
        assert!(
            m.forward(&client, &bad_x, &good.mu, &good.t, &good.cond, &good.dt)
                .is_err()
        );

        // cond's feature axis is wrong.
        let bad_cond = Var::new(t(&[2, PATCH_SIZE, FEAT_DIM + 1], 1.7, &device), false);
        assert!(
            m.forward(&client, &good.x, &good.mu, &good.t, &bad_cond, &good.dt)
                .is_err()
        );

        // mu's width is not a multiple of hidden_dim.
        let bad_mu = Var::new(t(&[2, MU_TOKENS * HIDDEN_DIM + 1], 1.3, &device), false);
        assert!(
            m.forward(&client, &good.x, &bad_mu, &good.t, &good.cond, &good.dt)
                .is_err()
        );

        // t has the wrong batch.
        let bad_t = Var::new(t(&[3], 2.1, &device), false);
        assert!(
            m.forward(&client, &good.x, &good.mu, &bad_t, &good.cond, &good.dt)
                .is_err()
        );

        // dt is 2D, not [batch].
        let bad_dt = Var::new(t(&[2, 1], 0.0, &device), false);
        assert!(
            m.forward(&client, &good.x, &good.mu, &good.t, &good.cond, &bad_dt)
                .is_err()
        );
    }
}
