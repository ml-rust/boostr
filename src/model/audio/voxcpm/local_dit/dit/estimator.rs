//! The estimator core: [`LocalDit::forward_prepared`], the pass every
//! `forward*` wrapper in `forward.rs` prepares inputs for, and the timestep
//! embedding it uses.
//!
//! Reference (`voxcpm/modules/locdit/local_dit_v2.py:98-115`):
//!
//! ```text
//! x    = in_proj(x.transpose(1, 2).contiguous())      [b, P, H]
//! cond = cond_proj(cond.transpose(1, 2).contiguous()) [b, P, H]
//! prefix = cond.size(1)
//! t  = time_mlp(time_embeddings(t))                   [b, H]
//! dt = delta_time_mlp(time_embeddings(dt))            [b, H]
//! t  = t + dt
//! mu = mu.view(b, -1, H)                              [b, 2, H]
//! seq = cat([mu, t.unsqueeze(1), cond, x], dim=1)     [b, 2+1+P+P, H]
//! hidden = decoder(seq, is_causal=False)              layers -> final norm
//! hidden = hidden[:, prefix + mu.size(1) + 1:, :]     [b, P, H]
//! return out_proj(hidden).transpose(1, 2).contiguous()
//! ```
//!
//! Traps this implementation is pinned against:
//!
//! - The reference's `[b, feat_dim, patch_size]` I/O is a convention, not a
//!   kernel requirement: it transposes to `[b, patch_size, feat_dim]` before
//!   its projections and back after `out_proj`. This port takes and returns
//!   `[b, patch_size, feat_dim]` directly — the layout every caller (the
//!   Euler loop, the per-patch generator, the CFM loss) already holds — so
//!   neither transpose exists here. A caller holding the reference layout
//!   (the fixture gates) transposes at its own boundary.
//! - `mu` is `[b, 2 * hidden_dim]` and reshapes to **two** tokens of
//!   `hidden_dim`, not one. The token count is derived
//!   (`mu_dim / hidden_dim`), never hardcoded.
//! - Sequence order is `[mu, t, cond, x]`. Its length is
//!   `mu_tokens + 1 + patch_size + patch_size`, matching
//!   [`crate::model::audio::voxcpm::local_dit::LocalDitConfig::sequence_len`]
//!   (11 at `patch_size = 4`) — which is exactly the length the RoPE cache
//!   was narrowed to at load time.
//! - The output slice keeps ONLY the trailing `x` positions, starting at
//!   `prefix + mu_tokens + 1` where `prefix` is `cond`'s length. Slicing any
//!   other window returns a wrong answer with a correct shape.
//! - `t` and `dt` share ONE [`crate::nn::SinusoidalPosEmb`] but go through SEPARATE
//!   MLPs (`time_mlp`, `delta_time_mlp`) and are then SUMMED.
//! - `dt` is 0 at inference, yet `SinusoidalPosEmb(0)` is `[0..0, 1..1]`, NOT
//!   zero — so `delta_time_mlp` contributes a real constant bias. The `dt`
//!   branch must NOT be optimized away.
//! - The backbone is BIDIRECTIONAL: no causal mask, no mask at all. That is
//!   what [`crate::model::audio::voxcpm::bidirectional::BidirectionalLayer`] provides.
//! - The final `norm` (RMSNorm) runs after the layer stack and BEFORE the
//!   slice and `out_proj` — it is `MiniCPMModel.norm`, applied inside
//!   `self.decoder` (`voxcpm/modules/minicpm4/model.py:385`).
//!
//! [`crate::nn::SinusoidalPosEmb`] carries no learned weights and is therefore not
//! loaded with the checkpoint; it is built once from `hidden_dim` at load
//! time in `local_dit/loader.rs` and reused here on every call.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::local_dit::loader::LocalDit;
use crate::model::traits::ModelClient;
use crate::nn::{SinusoidalPosEmb, var_contiguous};
use crate::ops::FlashAttentionOps;
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_add, var_cast, var_cat, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> LocalDit<R> {
    /// The estimator core: [`forward_with_mu_tokens`](Self::forward_with_mu_tokens)
    /// with `cond` ALREADY projected by [`project_cond`](Self::project_cond)
    /// to `cond_h: [batch, patch_size, hidden_dim]`.
    ///
    /// Multi-step callers tokenize `mu` and project `cond` once outside the
    /// step loop and call this on every step; the wrappers in `forward.rs`
    /// do the per-call preparation for single-shot sites.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prepared<C>(
        &self,
        client: &C,
        x: &Var<R>,
        mu_tok: &Var<R>,
        t: &Var<R>,
        cond_h: &Var<R>,
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
        let batch = self.check_patch_input("x", x, None)?;
        self.check_cond_hidden(cond_h, batch)?;
        super::validate::check_timestep("t", t, batch)?;
        super::validate::check_timestep("dt", dt, batch)?;
        let mu_tokens = self.check_mu_tokens(mu_tok, batch)?;

        // [b, P, feat_dim] -> [b, P, hidden]. `Linear` reshapes its input: a
        // no-op for the dense tensors every caller holds.
        let x_h = self.in_proj.forward(client, &var_contiguous(x)?)?;
        // `prefix` in the reference: the number of `cond` positions.
        let prefix = cond_h.shape()[1];
        let hidden_dtype = x_h.tensor().dtype();

        // One shared SinusoidalPosEmb, two separate MLPs, summed. The `dt`
        // branch is NOT dead: SinusoidalPosEmb(0) = [0..0, 1..1]. The
        // embedding is built once at load time (see `local_dit/loader.rs`),
        // not reconstructed per call.
        let t_emb = self.embed_time(client, &self.time_embeddings, t, hidden_dtype, true)?;
        let dt_emb = self.embed_time(client, &self.time_embeddings, dt, hidden_dtype, false)?;
        let t_sum = var_add(&t_emb, &dt_emb, client).map_err(Error::Numr)?;
        // `t.unsqueeze(1)`: one sequence position.
        let t_tok = var_reshape(&t_sum, &[batch, 1, self.hidden_dim]).map_err(Error::Numr)?;

        // [mu, t, cond, x] along the sequence axis.
        let seq = var_cat(&[mu_tok, &t_tok, cond_h, &x_h], 1, client).map_err(Error::Numr)?;

        // Final norm BEFORE the slice — it is part of `self.decoder`.
        let h = if self.activation_checkpointing {
            let mut h = seq;
            for layer in &self.layers {
                h = layer.forward_checkpointed(client, &h, &self.rope)?;
            }
            self.norm.forward(client, &h)?
        } else {
            // Deferred-residual fusion across layers: each layer folds the
            // PREVIOUS layer's MLP output into its own input norm instead of
            // a separate add, and hands its own MLP output on unadded. See
            // `BidirectionalLayer::forward_with_pending_residual`.
            let mut h = seq;
            let mut pending: Option<Var<R>> = None;
            for layer in &self.layers {
                let (new_h, mlp_out) = layer.forward_with_pending_residual(
                    client,
                    &h,
                    pending.as_ref(),
                    &self.rope,
                )?;
                h = new_h;
                pending = Some(mlp_out);
            }
            match pending {
                Some(last_mlp) => self.norm.residual_norm(client, &h, &last_mlp)?.0,
                // `self.layers` is empty: nothing was deferred.
                None => self.norm.forward(client, &h)?,
            }
        };

        // Keep only the trailing `x` window: `prefix + mu_tokens + 1 ..`.
        let seq_len = h.shape()[1];
        let start = prefix + mu_tokens + 1;
        if start >= seq_len {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "assembled sequence of {seq_len} positions has no room for the \
                     trailing x window starting at {start}"
                ),
            });
        }
        // The one layout copy of this pass: the window is a strided view and
        // `Linear` reshapes its input. `out_proj` then writes the dense
        // `[b, P, feat_dim]` the caller keeps.
        let window = var_narrow(&h, 1, start, seq_len - start).map_err(Error::Numr)?;
        let tail = var_contiguous(&window)?;
        self.out_proj.forward(client, &tail)
    }

    /// `SinusoidalPosEmb` -> the matching MLP, mirroring the reference's
    /// `time_embeddings(t).to(x.dtype)`. The embedding's frequency table is
    /// `f32`, so the timestep is cast to `f32` going in and the embedding is
    /// cast to the hidden dtype coming out; both casts are no-ops for an
    /// `f32` model. `use_time_mlp` selects `time_mlp` (`t`) over
    /// `delta_time_mlp` (`dt`).
    fn embed_time<C>(
        &self,
        client: &C,
        time_embeddings: &SinusoidalPosEmb<R>,
        step: &Var<R>,
        hidden_dtype: DType,
        use_time_mlp: bool,
    ) -> Result<Var<R>>
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
        let step = var_cast(step, DType::F32, client).map_err(Error::Numr)?;
        let emb = time_embeddings.forward(client, &step)?;
        let emb = var_cast(&emb, hidden_dtype, client).map_err(Error::Numr)?;
        if use_time_mlp {
            self.time_mlp.forward(client, &emb)
        } else {
            self.delta_time_mlp.forward(client, &emb)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::tests::{FEAT_DIM, HIDDEN_DIM, MU_TOKENS, PATCH_SIZE, model, t};
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
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
            dt: Var::new(
                Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; batch], &[batch], device).unwrap(),
                false,
            ),
        }
    }

    /// [`LocalDit::forward_prepared`] fed `project_cond`'s output must return
    /// the identical output — the contract that lets `solve_euler` project
    /// `cond` once outside the step loop.
    #[test]
    fn forward_prepared_matches_forward() {
        let (client, device) = cpu_setup();
        let m = model(2, &device);
        let i = inputs(2, 0.9, 1.7, &device);

        let via_forward = m
            .forward(&client, &i.x, &i.mu, &i.t, &i.cond, &i.dt)
            .unwrap()
            .tensor()
            .contiguous()
            .unwrap()
            .to_vec::<f32>();

        let mu_tok =
            var_reshape(&var_contiguous(&i.mu).unwrap(), &[2, MU_TOKENS, HIDDEN_DIM]).unwrap();
        let cond_h = m.project_cond(&client, &i.cond).unwrap();
        assert_eq!(cond_h.shape(), &[2, PATCH_SIZE, HIDDEN_DIM]);
        let out = m
            .forward_prepared(&client, &i.x, &mu_tok, &i.t, &cond_h, &i.dt)
            .unwrap();
        let via_prepared = out.tensor().contiguous().unwrap().to_vec::<f32>();

        assert_eq!(via_forward, via_prepared);

        // A `cond_h` from another batch is refused, not broadcast.
        let other = m
            .project_cond(&client, &inputs(3, 0.9, 1.7, &device).cond)
            .unwrap();
        assert!(
            m.forward_prepared(&client, &i.x, &mu_tok, &i.t, &other, &i.dt)
                .is_err()
        );
    }
}
