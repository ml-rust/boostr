//! Cross-attention for a Whisper decoder layer: attends decoder queries onto
//! the encoder's output. Holds both call paths — cached (used by `generate`)
//! and full-recompute (used by `forward_inference`) — with an optional
//! capture sink for the attention probabilities.

use super::cache::DecoderLayerCache;
use super::helpers::reshape_heads;
use super::layer::WhisperDecoderLayer;
use crate::error::{Error, Result};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, MatmulOps, ScalarOps, ShapeOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> WhisperDecoderLayer<R> {
    /// Cross-attention with precomputed K/V (populated on first call, reused
    /// on every later step). Used by [`WhisperDecoderLayer::forward_with_cache_capturing`].
    ///
    /// `capture`: `Some` clones this call's attention probabilities
    /// `[B, H, tgt_len, S]` into the sink; `None` skips the clone.
    pub(super) fn cross_attention_cached<C>(
        &self,
        client: &C,
        x: &Tensor<R>,
        encoder_out: &Tensor<R>,
        cache: &mut DecoderLayerCache<R>,
        capture: Option<&mut Vec<Tensor<R>>>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + ShapeOps<R>
            + UnaryOps<R>,
        R::Client: TensorOps<R>,
    {
        let x_shape = x.shape();
        let batch = x_shape[0];
        let tgt_len = x_shape[1];
        let src_len = encoder_out.shape()[1];

        // Lazily compute cross K/V on first call; reuse on every subsequent step.
        if cache.cross_k.is_none() || cache.cross_v.is_none() {
            let kv_in = Var::new(encoder_out.clone(), false);
            let k = self.cross_k_proj.forward(client, &kv_in)?;
            let v = self.cross_v_proj.forward(client, &kv_in)?;
            cache.cross_k = Some(reshape_heads(
                k.tensor(),
                batch,
                src_len,
                self.num_heads,
                self.head_dim,
            )?);
            cache.cross_v = Some(reshape_heads(
                v.tensor(),
                batch,
                src_len,
                self.num_heads,
                self.head_dim,
            )?);
        }
        let k = cache.cross_k.as_ref().ok_or_else(|| Error::ModelError {
            reason: "cross_k cache slot is empty right after being populated".into(),
        })?;
        let v = cache.cross_v.as_ref().ok_or_else(|| Error::ModelError {
            reason: "cross_v cache slot is empty right after being populated".into(),
        })?;

        let q_in = Var::new(x.clone(), false);
        let q = self.cross_q_proj.forward(client, &q_in)?;
        let q = reshape_heads(q.tensor(), batch, tgt_len, self.num_heads, self.head_dim)?;

        let k_t = k.transpose(-2, -1).map_err(Error::Numr)?.contiguous()?;
        let scale = (self.head_dim as f32).sqrt();
        let scores = client.matmul(&q, &k_t).map_err(Error::Numr)?;
        let scores = client
            .div_scalar(&scores, scale as f64)
            .map_err(Error::Numr)?;
        let attn = client.softmax(&scores, -1).map_err(Error::Numr)?;
        if let Some(sink) = capture {
            sink.push(attn.clone());
        }
        let out = client.matmul(&attn, v).map_err(Error::Numr)?;

        let out = out
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()?
            .reshape(&[batch, tgt_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        let projected = self.cross_out_proj.forward(client, &Var::new(out, false))?;
        Ok(projected.tensor().clone())
    }

    /// Cross-attention over a full recompute (no KV cache). Used by
    /// [`WhisperDecoderLayer::forward_inference`]; carries no capture sink
    /// since nothing on that path asks for cross-attention weights today.
    pub(super) fn cross_attention<C>(
        &self,
        client: &C,
        x: &Tensor<R>,
        encoder_out: &Tensor<R>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + ShapeOps<R>
            + UnaryOps<R>,
        R::Client: TensorOps<R>,
    {
        let x_shape = x.shape();
        let batch = x_shape[0];
        let tgt_len = x_shape[1];
        let src_len = encoder_out.shape()[1];

        let q_in = Var::new(x.clone(), false);
        let kv_in = Var::new(encoder_out.clone(), false);

        let q = self.cross_q_proj.forward(client, &q_in)?;
        let k = self.cross_k_proj.forward(client, &kv_in)?;
        let v = self.cross_v_proj.forward(client, &kv_in)?;

        let q = reshape_heads(q.tensor(), batch, tgt_len, self.num_heads, self.head_dim)?;
        let k = reshape_heads(k.tensor(), batch, src_len, self.num_heads, self.head_dim)?;
        let v = reshape_heads(v.tensor(), batch, src_len, self.num_heads, self.head_dim)?;

        let k_t = k.transpose(-2, -1).map_err(Error::Numr)?.contiguous()?;
        let scale = (self.head_dim as f32).sqrt();
        let scores = client.matmul(&q, &k_t).map_err(Error::Numr)?;
        let scores = client
            .div_scalar(&scores, scale as f64)
            .map_err(Error::Numr)?;
        let attn = client.softmax(&scores, -1).map_err(Error::Numr)?;
        let out = client.matmul(&attn, &v).map_err(Error::Numr)?;

        let out = out
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()?
            .reshape(&[batch, tgt_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        let projected = self.cross_out_proj.forward(client, &Var::new(out, false))?;
        Ok(projected.tensor().clone())
    }
}
