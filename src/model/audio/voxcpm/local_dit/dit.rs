//! Estimator forward pass for VoxCPM2's `feat_decoder` local DiT ("locdit").
//!
//! This is the CFM *estimator* only — it evaluates the DiT once for a given
//! `(x, mu, t, cond, dt)`. Sampling (noise, Euler stepping, classifier-free
//! guidance) is a separate unit and lives nowhere in this file.
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
//! - `x` and `cond` arrive as `[b, feat_dim, patch_size]` and are TRANSPOSED
//!   to `[b, patch_size, feat_dim]` before their projections; the result is
//!   transposed BACK at the end. Skipping either transpose silently projects
//!   the wrong axis.
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
//! - `t` and `dt` share ONE [`SinusoidalPosEmb`] but go through SEPARATE
//!   MLPs (`time_mlp`, `delta_time_mlp`) and are then SUMMED.
//! - `dt` is 0 at inference, yet `SinusoidalPosEmb(0)` is `[0..0, 1..1]`, NOT
//!   zero — so `delta_time_mlp` contributes a real constant bias. The `dt`
//!   branch must NOT be optimized away.
//! - The backbone is BIDIRECTIONAL: no causal mask, no mask at all. That is
//!   what [`BidirectionalLayer`] provides.
//! - The final `norm` (RMSNorm) runs after the layer stack and BEFORE the
//!   slice and `out_proj` — it is `MiniCPMModel.norm`, applied inside
//!   `self.decoder` (`voxcpm/modules/minicpm4/model.py:385`).
//!
//! [`SinusoidalPosEmb`] carries no learned weights and is therefore not
//! loaded with the checkpoint; it is built once from `hidden_dim` at load
//! time in `local_dit/loader.rs` and reused here on every call.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::local_dit::loader::LocalDit;
use crate::model::traits::ModelClient;
use crate::nn::{SinusoidalPosEmb, var_contiguous};
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_add, var_cast, var_cat, var_narrow, var_reshape, var_transpose};
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
    /// - `x`: `[batch, feat_dim, patch_size]` — the current CFM sample.
    /// - `mu`: `[batch, mu_tokens * hidden_dim]` — the global-encoder
    ///   condition, reshaped to `mu_tokens` sequence positions (2 on this
    ///   checkpoint).
    /// - `t`: `[batch]` — the flow timestep, one scalar per sample.
    /// - `cond`: `[batch, feat_dim, patch_size]` — the prefix condition.
    /// - `dt`: `[batch]` — the mean-velocity delta. Zero at inference, but
    ///   still a live input: see the module docs.
    ///
    /// Returns `[batch, feat_dim, patch_size]`.
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
            + DequantOps<R>,
    {
        let batch = self.check_patch_input("x", x, None)?;
        self.check_patch_input("cond", cond, Some(batch))?;
        let mu_tokens = self.check_mu(mu, batch)?;
        check_timestep("t", t, batch)?;
        check_timestep("dt", dt, batch)?;

        // [b, feat_dim, P] -> [b, P, feat_dim] -> [b, P, hidden].
        // `var_transpose` swaps the last two dims and yields a strided view;
        // `Linear` reshapes its input, so materialize first.
        let x_h = self.in_proj.forward(
            client,
            &var_contiguous(&var_transpose(x).map_err(Error::Numr)?)?,
        )?;
        let cond_h = self.cond_proj.forward(
            client,
            &var_contiguous(&var_transpose(cond).map_err(Error::Numr)?)?,
        )?;
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

        // `mu.view(b, -1, hidden)`: mu_tokens sequence positions.
        let mu_contig = var_contiguous(mu)?;
        let mu_tok =
            var_reshape(&mu_contig, &[batch, mu_tokens, self.hidden_dim]).map_err(Error::Numr)?;

        // [mu, t, cond, x] along the sequence axis.
        let seq = var_cat(&[&mu_tok, &t_tok, &cond_h, &x_h], 1, client).map_err(Error::Numr)?;

        let mut h = seq;
        for layer in &self.layers {
            h = if self.activation_checkpointing {
                layer.forward_checkpointed(client, &h, &self.rope)?
            } else {
                layer.forward(client, &h, &self.rope)?
            };
        }
        // Final norm BEFORE the slice — it is part of `self.decoder`.
        let h = self.norm.forward(client, &h)?;

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
        let window = var_narrow(&h, 1, start, seq_len - start).map_err(Error::Numr)?;
        let tail = var_contiguous(&window)?;

        // [b, P, hidden] -> [b, P, feat_dim] -> [b, feat_dim, P].
        let out = self.out_proj.forward(client, &tail)?;
        var_contiguous(&var_transpose(&out).map_err(Error::Numr)?)
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

    /// Validate an `[batch, feat_dim, patch_size]` input, returning its batch.
    /// `expected_batch` pins the batch against an earlier input.
    ///
    /// `pub(super)` so the sibling CFM sampler validates `z`/`cond` up front
    /// with the same rules, instead of waiting for the first estimator call.
    pub(super) fn check_patch_input(
        &self,
        arg: &'static str,
        v: &Var<R>,
        expected_batch: Option<usize>,
    ) -> Result<usize> {
        let shape = v.shape();
        if shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg,
                reason: format!(
                    "expected 3D [batch, feat_dim, patch_size], got {}D {shape:?}",
                    shape.len()
                ),
            });
        }
        if shape[1] != self.feat_dim || shape[2] != self.patch_size {
            return Err(Error::InvalidArgument {
                arg,
                reason: format!(
                    "expected [batch, {}, {}], got {shape:?}",
                    self.feat_dim, self.patch_size
                ),
            });
        }
        if let Some(batch) = expected_batch
            && shape[0] != batch
        {
            return Err(Error::InvalidArgument {
                arg,
                reason: format!("batch {} does not match x's batch {batch}", shape[0]),
            });
        }
        Ok(shape[0])
    }

    /// Validate `mu: [batch, mu_tokens * hidden_dim]` and return `mu_tokens`
    /// — derived from the width, never hardcoded to 2. `pub(super)` for the
    /// same reason as [`Self::check_patch_input`].
    pub(super) fn check_mu(&self, mu: &Var<R>, batch: usize) -> Result<usize> {
        let shape = mu.shape();
        if shape.len() != 2 || shape[0] != batch {
            return Err(Error::InvalidArgument {
                arg: "mu",
                reason: format!(
                    "expected 2D [{batch}, k * {}], got {shape:?}",
                    self.hidden_dim
                ),
            });
        }
        if shape[1] == 0 || !shape[1].is_multiple_of(self.hidden_dim) {
            return Err(Error::InvalidArgument {
                arg: "mu",
                reason: format!(
                    "width {} is not a nonzero multiple of hidden_dim {}",
                    shape[1], self.hidden_dim
                ),
            });
        }
        Ok(shape[1] / self.hidden_dim)
    }
}

/// Validate a `[batch]` scalar-per-sample timestep (`t` or `dt`).
fn check_timestep<R: Runtime<DType = DType>>(
    arg: &'static str,
    v: &Var<R>,
    batch: usize,
) -> Result<()> {
    let shape = v.shape();
    if shape.len() != 1 || shape[0] != batch {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!("expected 1D [{batch}], got {shape:?}"),
        });
    }
    Ok(())
}
