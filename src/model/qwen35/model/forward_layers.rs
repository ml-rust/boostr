//! The [`Qwen35Model`] layer walk every forward shares.
//!
//! The walk keeps two counters, `attn_idx` into the [`LayeredKvCache`] and
//! `gdn_idx` into the [`LayeredGdnState`], exactly as
//! `HybridModel::forward_hybrid` does: each state container is indexed by
//! position among layers of its kind, not by absolute layer index.

use super::build::{Qwen35Block, Qwen35Model};
use crate::error::{Error, Result};
use crate::inference::{LaneKvStep, LayeredGdnState, LayeredKvCache};
use crate::model::traits::ModelClient;
use crate::nn::{RmsNorm, RotatedMlp};
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ConvOps, IndexingOps, MatmulOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// How the attention layers of [`Qwen35Model::forward_layers`] take their
/// KV step.
pub(super) enum AttentionStep<'a, R: Runtime> {
    /// One `[4, seq]` position set shared by the whole batch; the mixer's
    /// ordinary KV-cached forward.
    Shared,
    /// One lane per sequence: `[4, batch, 1]` positions and each lane's
    /// own key range and write position.
    PerLane { lanes: &'a LaneKvStep<'a, R> },
}

impl<R: Runtime<DType = DType>> Qwen35Model<R>
where
    R::Client: IndexingOps<R>,
{
    /// Walk every block over `hidden` `[batch, seq, hidden]`, then apply
    /// `output_norm`. See [`forward_blocks`](Self::forward_blocks) for the
    /// walk itself.
    pub(super) fn forward_layers<C>(
        &self,
        client: &C,
        hidden: Var<R>,
        positions: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        gdn_state: &mut LayeredGdnState<R>,
        step: &AttentionStep<'_, R>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R>,
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
            + DequantOps<R>
            + MatmulOps<R>,
    {
        let hidden = self.forward_blocks(client, hidden, positions, kv_cache, gdn_state, step)?;
        self.norm.forward(client, &hidden)
    }

    /// Walk every block over `hidden` `[batch, seq, hidden]` and return the
    /// last block's output, before `output_norm`. `positions` is the
    /// `[4, seq]` i32 IMROPE tensor every attention layer reads for
    /// [`AttentionStep::Shared`], or the `[4, batch, 1]` per-lane tensor for
    /// [`AttentionStep::PerLane`].
    ///
    /// Under [`AttentionStep::PerLane`], a GDN block's recurrence runs over
    /// every row, occupied or not: there is no per-lane mask. A free row's
    /// step is harmless — its `kv_start == kv_end` keeps attention from ever
    /// reading it, and the next sequence admitted into that lane overwrites
    /// the GDN state and the KV rows below its own end, so nothing a free
    /// row computed survives to be read.
    pub(super) fn forward_blocks<C>(
        &self,
        client: &C,
        hidden: Var<R>,
        positions: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        gdn_state: &mut LayeredGdnState<R>,
        step: &AttentionStep<'_, R>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R>,
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
            + DequantOps<R>
            + MatmulOps<R>,
    {
        let mut hidden = hidden;
        let mut attn_idx = 0usize;
        let mut gdn_idx = 0usize;

        for (i, block) in self.blocks.iter().enumerate() {
            match block {
                Qwen35Block::Attention(layer) => {
                    let cache = kv_cache
                        .layer_mut(attn_idx)
                        .ok_or_else(|| Error::ModelError {
                            reason: format!(
                                "qwen35: KV cache missing for layer {i} (attn_idx={attn_idx})"
                            ),
                        })?;
                    let normed = layer.attn_norm.forward(client, &hidden)?;
                    let mixed = match step {
                        AttentionStep::Shared => layer
                            .mixer
                            .forward(client, &normed, &self.rope, positions, cache)?,
                        AttentionStep::PerLane { lanes } => layer.mixer.forward_batched_decode(
                            client, &normed, &self.rope, positions, cache, lanes,
                        )?,
                    };
                    hidden = ffn_residual(
                        client,
                        &hidden,
                        &mixed,
                        &layer.post_attention_norm,
                        &layer.mlp,
                    )?;
                    attn_idx += 1;
                }
                Qwen35Block::Gdn(layer) => {
                    let state = gdn_state
                        .layer_mut(gdn_idx)
                        .ok_or_else(|| Error::ModelError {
                            reason: format!(
                                "qwen35: GDN state missing for layer {i} (gdn_idx={gdn_idx})"
                            ),
                        })?;
                    let normed = layer.attn_norm.forward(client, &hidden)?;
                    let mixed = layer.mixer.forward(client, &normed, state)?;
                    hidden = ffn_residual(
                        client,
                        &hidden,
                        &mixed,
                        &layer.post_attention_norm,
                        &layer.mlp,
                    )?;
                    gdn_idx += 1;
                }
            }
        }

        Ok(hidden)
    }
}

/// `h = x + mixed; h + mlp(post_norm(h))`.
pub(super) fn ffn_residual<R, C>(
    client: &C,
    x: &Var<R>,
    mixed: &Var<R>,
    post_norm: &RmsNorm<R>,
    mlp: &RotatedMlp<R>,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + DequantOps<R>
        + MatmulOps<R>,
{
    let h = var_add(x, mixed, client).map_err(Error::Numr)?;
    let normed = post_norm.forward(client, &h)?;
    let ffn = mlp.forward(client, &normed)?;
    var_add(&h, &ffn, client).map_err(Error::Numr)
}
