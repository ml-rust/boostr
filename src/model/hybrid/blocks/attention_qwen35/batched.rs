//! `qwen35` gated attention over a batch of sequences, one decode step each.
//!
//! Every lane of the cache holds one sequence, left-padded so all lanes end
//! at the shared cursor, and `kv_start` names each lane's own first
//! position. A lane therefore attends over exactly its own tokens, whatever
//! the other lanes hold, and the step it takes is the step it would take
//! alone.
//!
//! The decode kernel reads the cache buffer in place through `kv_seq_len`
//! and broadcasts the KV heads itself, so no lane's keys are copied or
//! repeated per step.

use super::layer::Qwen35AttentionBlock;
use super::projections::Projections;
use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::model::traits::ModelClient;
use crate::nn::RoPE;
use crate::ops::traits::attention::{AttnOutLayout, KvSeqLen};
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_permute, var_reshape, var_sigmoid_mul};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, FwhtOps, IndexingOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Qwen35AttentionBlock<R> {
    /// One decode step for `batch` sequences at once.
    ///
    /// - `x`: the `attn_norm`-ed hidden state `[batch, 1, hidden_size]`
    /// - `positions`: `[4, batch, 1]` IMROPE streams, a position per lane
    /// - `kv_cache`: `[batch, ...]`, every lane already prefilled to the
    ///   shared cursor; this step appends one token to each
    /// - `kv_start`: `[batch]` I32, each lane's first real cache position
    ///
    /// The result is the `attn_output` projection `[batch, 1, hidden_size]`,
    /// without the residual, as a detached `Var`.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `x` is not `[batch, 1, hidden_size]`, the
    /// cache holds a different number of lanes, or the rope table width is
    /// not `rope_dim / 2`; cache and op errors propagate.
    pub fn forward_batched_decode<C>(
        &self,
        client: &C,
        x: &Var<R>,
        rope: &RoPE<R>,
        positions: &Tensor<R>,
        kv_cache: &mut KvCache<R>,
        kv_start: &Tensor<R>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + FwhtOps<R>,
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
            + DequantOps<R>,
    {
        let cfg = &self.cfg;
        let shape = x.shape().to_vec();
        if shape.len() != 3 || shape[1] != 1 || shape[2] != cfg.hidden_size {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35_attention_batched: expected [batch, 1, {}], got {shape:?}",
                    cfg.hidden_size
                ),
            });
        }
        let batch = shape[0];
        if kv_cache.batch_size() != batch {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35_attention_batched: cache holds {} lanes, input has {batch}",
                    kv_cache.batch_size()
                ),
            });
        }
        let (h, hd) = (cfg.num_heads, cfg.head_dim);
        let half_rot = cfg.rope_dim / 2;
        if rope.cos_cache().shape().get(1) != Some(&half_rot) {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35_attention_batched: rope table must be [max_pos, {half_rot}], got {:?}",
                    rope.cos_cache().shape()
                ),
            });
        }

        let Projections { q, gate, k, v } = self.project(client, x, batch, 1)?;

        let (cos, sin) = (rope.cos_cache(), rope.sin_cache());
        let n_rot = cfg.rope_dim;
        let q =
            client.mrope_interleaved_fused(&q, cos, sin, positions, &self.mrope_selector, n_rot)?;
        let k =
            client.mrope_interleaved_fused(&k, cos, sin, positions, &self.mrope_selector, n_rot)?;

        // [B, 1, H, D] -> [B, H, 1, D]: the layout the cache and the decode
        // kernel read.
        let head_major = |t: &Var<R>| -> Result<Tensor<R>> {
            Ok(var_permute(t, &[0, 2, 1, 3])
                .map_err(Error::Numr)?
                .tensor()
                .contiguous()?)
        };
        let q = head_major(&q)?;
        let k = head_major(&k)?;
        let v = head_major(&v)?;

        kv_cache.update_fused(&k, &v, client)?;

        let (attn, _lse) = client.flash_attention_fwd(
            &q,
            kv_cache.k_cache_raw(),
            kv_cache.v_cache_raw(),
            h,
            cfg.num_kv_heads,
            hd,
            true,
            0,
            Some(KvSeqLen::Shared(kv_cache.seq_len())),
            Some(kv_start),
            AttnOutLayout::TokenMajor,
        )?;

        let attn = Var::new(attn, false);
        let gated = var_sigmoid_mul(&gate, &attn, client).map_err(Error::Numr)?;
        let gated = var_reshape(&gated, &[batch, 1, h * hd]).map_err(Error::Numr)?;
        let out = self.attn_output.forward(client, &gated)?;
        Ok(Var::new(out.tensor().clone(), false))
    }
}
