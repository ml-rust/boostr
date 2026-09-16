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

mod forward;
mod validate;
