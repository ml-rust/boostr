//! Padded self-attention: `[B, S, hidden]` via the classic `[B, H, S, S]` scores.

use super::encoder_layer::EncoderLayer;
use crate::error::{Error, Result};
use crate::ops::RoPEOps;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_permute, var_reshape, var_transpose};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps, TensorOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> EncoderLayer<R> {
    /// Padded self-attention. Input is `[B, S, hidden]`.
    pub(super) fn self_attention_padded<C>(
        &self,
        client: &C,
        x: &Var<R>,
        attention_mask: Option<&Tensor<R>>,
        span_mask: Option<&Tensor<R>>,
    ) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + BinaryOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + UnaryOps<R>
            + QuantMatmulOps<R>
            + TypeConversionOps<R>
            + NormalizationOps<R>
            + RoPEOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + DequantOps<R>,
    {
        use numr::autograd::var_matmul;

        let shape = x.shape().to_vec();
        let batch = shape[0];
        let seq_len = shape[1];

        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        // Whole-hidden QK-norm (jina-bert-v2) runs on the [B, S, hidden]
        // projection output, so every head shares one mean and variance. It
        // MUST happen before the reshape below — after it, the same weights
        // would normalise each head separately and produce different numbers
        // at identical shapes.
        let (q, k) = self.qk_norm_hidden(client, q, k)?;

        // Q: [B, S, num_heads * head_dim] → [B, num_heads, S, head_dim]
        let q = var_reshape(&q, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let q = var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;

        // K, V: [B, S, num_kv_heads * head_dim] → [B, num_kv_heads, S, head_dim]
        let k = var_reshape(&k, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let k = var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let v = var_reshape(&v, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let v = var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;

        // Per-head QK-norm (Gemma/Qwen3): RmsNorm over head_dim after reshape,
        // before RoPE.
        let (q, k) = self.qk_norm_per_head(client, q, k)?;

        // RoPE, using this block's own cache. For an interleaved architecture
        // local and global blocks hold caches built from different bases.
        let q = match &self.rope {
            Some(rope) => rope.forward(client, &q)?,
            None => q,
        };
        let k = match &self.rope {
            Some(rope) => rope.forward(client, &k)?,
            None => k,
        };

        // GQA: repeat K and V heads when num_kv_heads < num_heads.
        let repeats = self.num_heads / self.num_kv_heads;
        let (k, v) = if repeats > 1 {
            let k_rep = client
                .repeat_interleave(&k.tensor().contiguous()?, repeats, Some(1))
                .map_err(Error::Numr)?;
            let v_rep = client
                .repeat_interleave(&v.tensor().contiguous()?, repeats, Some(1))
                .map_err(Error::Numr)?;
            (Var::new(k_rep, false), Var::new(v_rep, false))
        } else {
            (k, v)
        };

        // Fold the attention scale into Q (removes a [B,H,S,D] mul_scalar pass).
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let q_scaled = client.mul_scalar(q.tensor(), scale).map_err(Error::Numr)?;
        let q_scaled = Var::new(q_scaled.contiguous()?, false);
        let k = Var::new(k.tensor().contiguous()?, false);
        let v = Var::new(v.tensor().contiguous()?, false);

        let k_t = var_transpose(&k).map_err(Error::Numr)?;
        let scores = var_matmul(&q_scaled, &k_t, client).map_err(Error::Numr)?;
        let scores_dtype = scores.tensor().dtype();

        // Combine the padding mask [B,1,1,S] and the span mask [1,1,S,S] into a
        // single additive bias; both broadcast against scores [B,H,S,S].
        let padding_bias = match attention_mask {
            Some(mask) => Some(self.padding_bias(client, mask, batch, seq_len, scores_dtype)?),
            None => None,
        };
        let span_bias = match span_mask {
            Some(m) if m.dtype() != scores_dtype => {
                Some(client.cast(m, scores_dtype).map_err(Error::Numr)?)
            }
            Some(m) => Some(m.clone()),
            None => None,
        };

        let bias = match (padding_bias, span_bias) {
            (Some(p), Some(s)) => Some(client.add(&p, &s).map_err(Error::Numr)?),
            (Some(p), None) => Some(p),
            (None, Some(s)) => Some(s),
            (None, None) => None,
        };

        let attn_weights = match &bias {
            Some(b) => Var::new(
                client
                    .softmax_with_bias(scores.tensor(), b, -1)
                    .map_err(Error::Numr)?,
                false,
            ),
            None => Var::new(
                client.softmax(scores.tensor(), -1).map_err(Error::Numr)?,
                false,
            ),
        };

        let attn_out = var_matmul(&attn_weights, &v, client).map_err(Error::Numr)?;
        let attn_out = var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = Var::new(attn_out.tensor().contiguous()?, false);
        let hidden = self.num_heads * self.head_dim;
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, hidden]).map_err(Error::Numr)?;

        self.o_proj.forward(client, &attn_out)
    }

    /// Turn a `[B, S]` 0/1 padding mask into an additive `[B, 1, 1, S]` bias.
    fn padding_bias<C>(
        &self,
        client: &C,
        mask: &Tensor<R>,
        batch: usize,
        seq_len: usize,
        scores_dtype: DType,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + ScalarOps<R> + ShapeOps<R> + TypeConversionOps<R>,
    {
        let mask_shape = mask.shape().to_vec();
        if mask_shape.len() != 2 || mask_shape[0] != batch || mask_shape[1] != seq_len {
            return Err(Error::ModelError {
                reason: format!(
                    "attention_mask shape must be [{batch}, {seq_len}], got {mask_shape:?}"
                ),
            });
        }

        // -30000.0 for F16 (max is ±65504, so -1e9 would overflow to -inf);
        // -1e9 on the F32 path preserves existing behaviour.
        let additive_val = if scores_dtype == DType::F16 {
            -30000.0f64
        } else {
            -1e9f64
        };

        let inv = client.rsub_scalar(mask, 1.0).map_err(Error::Numr)?;
        let additive = client.mul_scalar(&inv, additive_val).map_err(Error::Numr)?;
        let additive = if scores_dtype != DType::F32 {
            client.cast(&additive, scores_dtype).map_err(Error::Numr)?
        } else {
            additive
        };
        additive
            .reshape(&[batch, 1, 1, seq_len])
            .map_err(Error::Numr)
    }
}
