//! A single Whisper decoder block (self-attn → cross-attn → MLP).

use super::cache::DecoderLayerCache;
use super::helpers::{apply_causal_mask, load_attn, load_layernorm, reshape_heads};
use crate::error::{Error, Result};
use crate::nn::{LayerNorm, Linear, VarBuilder};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConditionalOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// A single Whisper decoder block:
///   pre-norm → causal self-attn → residual
///   pre-norm → cross-attn(encoder) → residual
///   pre-norm → MLP (GELU) → residual
pub struct WhisperDecoderLayer<R: Runtime> {
    self_attn_ln: LayerNorm<R>,
    self_q_proj: Linear<R>,
    self_k_proj: Linear<R>,
    self_v_proj: Linear<R>,
    self_out_proj: Linear<R>,

    cross_attn_ln: LayerNorm<R>,
    cross_q_proj: Linear<R>,
    cross_k_proj: Linear<R>,
    cross_v_proj: Linear<R>,
    cross_out_proj: Linear<R>,

    final_ln: LayerNorm<R>,
    fc1: Linear<R>,
    fc2: Linear<R>,

    num_heads: usize,
    head_dim: usize,
}

impl<R: Runtime<DType = DType>> WhisperDecoderLayer<R> {
    pub fn from_varbuilder(
        vb: &mut VarBuilder<'_, R>,
        hidden_size: usize,
        num_heads: usize,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        // Self-attention
        let (self_q_proj, self_k_proj, self_v_proj, self_out_proj) = load_attn(vb, "self_attn")?;
        let self_attn_ln = load_layernorm(vb, "self_attn_layer_norm")?;

        // Cross-attention
        let (cross_q_proj, cross_k_proj, cross_v_proj, cross_out_proj) =
            load_attn(vb, "encoder_attn")?;
        let cross_attn_ln = load_layernorm(vb, "encoder_attn_layer_norm")?;

        // FFN
        let mut fc1_vb = vb.pp("fc1");
        let fc1 = Linear::new(
            fc1_vb.take_tensor("weight")?,
            fc1_vb.take_tensor_optional("bias")?,
            false,
        );
        drop(fc1_vb);
        let mut fc2_vb = vb.pp("fc2");
        let fc2 = Linear::new(
            fc2_vb.take_tensor("weight")?,
            fc2_vb.take_tensor_optional("bias")?,
            false,
        );
        drop(fc2_vb);

        let final_ln = load_layernorm(vb, "final_layer_norm")?;

        Ok(Self {
            self_attn_ln,
            self_q_proj,
            self_k_proj,
            self_v_proj,
            self_out_proj,
            cross_attn_ln,
            cross_q_proj,
            cross_k_proj,
            cross_v_proj,
            cross_out_proj,
            final_ln,
            fc1,
            fc2,
            num_heads,
            head_dim,
        })
    }

    /// One forward step (no KV cache — full sequence recompute).
    ///
    /// - `x`: decoder input `[B, T, hidden]`
    /// - `encoder_out`: encoder output `[B, S, hidden]`
    /// - `causal`: apply causal mask to self-attention scores
    ///
    /// Returns `[B, T, hidden]`.
    pub fn forward_inference<C>(
        &self,
        client: &C,
        x: &Tensor<R>,
        encoder_out: &Tensor<R>,
        causal: bool,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + NormalizationOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + UnaryOps<R>
            + ConditionalOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        // Self-attention block (causal).
        let normed = self
            .self_attn_ln
            .forward(client, &Var::new(x.clone(), false))?;
        let sa_out = self.self_attention(client, normed.tensor(), causal)?;
        let x1 = client.add(x, &sa_out).map_err(Error::Numr)?;

        // Cross-attention block.
        let normed2 = self
            .cross_attn_ln
            .forward(client, &Var::new(x1.clone(), false))?;
        let ca_out = self.cross_attention(client, normed2.tensor(), encoder_out)?;
        let x2 = client.add(&x1, &ca_out).map_err(Error::Numr)?;

        // FFN block.
        let normed3 = self
            .final_ln
            .forward(client, &Var::new(x2.clone(), false))?;
        let h = self
            .fc1
            .forward(client, &Var::new(normed3.tensor().clone(), false))?;
        let h = client.gelu(h.tensor()).map_err(Error::Numr)?;
        let h = self.fc2.forward(client, &Var::new(h, false))?;

        client.add(&x2, h.tensor()).map_err(Error::Numr)
    }

    /// Forward pass using a KV cache. Handles both prefill (`cache` empty,
    /// `x` carries multiple tokens, causal mask applied) and incremental steps
    /// (`cache` primed, `x` carries one token, no mask needed — past positions
    /// already precede the current one in the concatenated K/V).
    pub fn forward_with_cache<C>(
        &self,
        client: &C,
        x: &Tensor<R>,
        encoder_out: &Tensor<R>,
        cache: &mut DecoderLayerCache<R>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + NormalizationOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + UnaryOps<R>
            + ConditionalOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        // Self-attention block with KV cache.
        let normed = self
            .self_attn_ln
            .forward(client, &Var::new(x.clone(), false))?;
        let sa_out = self.self_attention_cached(client, normed.tensor(), cache)?;
        let x1 = client.add(x, &sa_out).map_err(Error::Numr)?;

        // Cross-attention with precomputed K/V (populated on first call).
        let normed2 = self
            .cross_attn_ln
            .forward(client, &Var::new(x1.clone(), false))?;
        let ca_out = self.cross_attention_cached(client, normed2.tensor(), encoder_out, cache)?;
        let x2 = client.add(&x1, &ca_out).map_err(Error::Numr)?;

        // FFN.
        let normed3 = self
            .final_ln
            .forward(client, &Var::new(x2.clone(), false))?;
        let h = self
            .fc1
            .forward(client, &Var::new(normed3.tensor().clone(), false))?;
        let h = client.gelu(h.tensor()).map_err(Error::Numr)?;
        let h = self.fc2.forward(client, &Var::new(h, false))?;

        client.add(&x2, h.tensor()).map_err(Error::Numr)
    }

    fn self_attention_cached<C>(
        &self,
        client: &C,
        x: &Tensor<R>,
        cache: &mut DecoderLayerCache<R>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + ShapeOps<R>
            + UnaryOps<R>
            + ConditionalOps<R>,
        R::Client: TensorOps<R>,
    {
        let shape = x.shape();
        let batch = shape[0];
        let cur_len = shape[1];

        let x_var = Var::new(x.clone(), false);
        let q = self.self_q_proj.forward(client, &x_var)?;
        let k_new = self.self_k_proj.forward(client, &x_var)?;
        let v_new = self.self_v_proj.forward(client, &x_var)?;

        let q = reshape_heads(q.tensor(), batch, cur_len, self.num_heads, self.head_dim)?;
        let k_new = reshape_heads(
            k_new.tensor(),
            batch,
            cur_len,
            self.num_heads,
            self.head_dim,
        )?;
        let v_new = reshape_heads(
            v_new.tensor(),
            batch,
            cur_len,
            self.num_heads,
            self.head_dim,
        )?;

        // Concatenate past K/V with the new ones along the time axis (dim=2 in [B,H,T,D]).
        let k_all = match cache.self_k.take() {
            Some(prev) => client.cat(&[&prev, &k_new], 2).map_err(Error::Numr)?,
            None => k_new,
        };
        let v_all = match cache.self_v.take() {
            Some(prev) => client.cat(&[&prev, &v_new], 2).map_err(Error::Numr)?,
            None => v_new,
        };

        let k_t = k_all.transpose(-2, -1).map_err(Error::Numr)?.contiguous();
        let scale = (self.head_dim as f32).sqrt();
        let scores = client.matmul(&q, &k_t).map_err(Error::Numr)?;
        let mut scores = client
            .div_scalar(&scores, scale as f64)
            .map_err(Error::Numr)?;

        // Apply causal mask only for prefill (cur_len > 1 AND cache was empty
        // before this call — in which case k_all's time dim equals cur_len).
        if cur_len > 1 && k_all.shape()[2] == cur_len {
            scores = apply_causal_mask(client, scores, batch, self.num_heads, cur_len)?;
        }

        let attn = client.softmax(&scores, -1).map_err(Error::Numr)?;
        let out = client.matmul(&attn, &v_all).map_err(Error::Numr)?;

        // Write back updated K/V.
        cache.self_k = Some(k_all);
        cache.self_v = Some(v_all);

        let out = out
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()
            .reshape(&[batch, cur_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        let projected = self.self_out_proj.forward(client, &Var::new(out, false))?;
        Ok(projected.tensor().clone())
    }

    fn cross_attention_cached<C>(
        &self,
        client: &C,
        x: &Tensor<R>,
        encoder_out: &Tensor<R>,
        cache: &mut DecoderLayerCache<R>,
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
        let k = cache.cross_k.as_ref().expect("cross_k just populated");
        let v = cache.cross_v.as_ref().expect("cross_v just populated");

        let q_in = Var::new(x.clone(), false);
        let q = self.cross_q_proj.forward(client, &q_in)?;
        let q = reshape_heads(q.tensor(), batch, tgt_len, self.num_heads, self.head_dim)?;

        let k_t = k.transpose(-2, -1).map_err(Error::Numr)?.contiguous();
        let scale = (self.head_dim as f32).sqrt();
        let scores = client.matmul(&q, &k_t).map_err(Error::Numr)?;
        let scores = client
            .div_scalar(&scores, scale as f64)
            .map_err(Error::Numr)?;
        let attn = client.softmax(&scores, -1).map_err(Error::Numr)?;
        let out = client.matmul(&attn, v).map_err(Error::Numr)?;

        let out = out
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()
            .reshape(&[batch, tgt_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        let projected = self.cross_out_proj.forward(client, &Var::new(out, false))?;
        Ok(projected.tensor().clone())
    }

    fn self_attention<C>(&self, client: &C, x: &Tensor<R>, causal: bool) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + ShapeOps<R>
            + UnaryOps<R>
            + ConditionalOps<R>,
        R::Client: TensorOps<R>,
    {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        let x_var = Var::new(x.clone(), false);
        let q = self.self_q_proj.forward(client, &x_var)?;
        let k = self.self_k_proj.forward(client, &x_var)?;
        let v = self.self_v_proj.forward(client, &x_var)?;

        let q = reshape_heads(q.tensor(), batch, seq_len, self.num_heads, self.head_dim)?;
        let k = reshape_heads(k.tensor(), batch, seq_len, self.num_heads, self.head_dim)?;
        let v = reshape_heads(v.tensor(), batch, seq_len, self.num_heads, self.head_dim)?;

        let k_t = k.transpose(-2, -1).map_err(Error::Numr)?.contiguous();
        let scale = (self.head_dim as f32).sqrt();
        let scores = client.matmul(&q, &k_t).map_err(Error::Numr)?;
        let mut scores = client
            .div_scalar(&scores, scale as f64)
            .map_err(Error::Numr)?;

        if causal && seq_len > 1 {
            scores = apply_causal_mask(client, scores, batch, self.num_heads, seq_len)?;
        }

        let attn = client.softmax(&scores, -1).map_err(Error::Numr)?;
        let out = client.matmul(&attn, &v).map_err(Error::Numr)?;

        let out = out
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()
            .reshape(&[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        let projected = self.self_out_proj.forward(client, &Var::new(out, false))?;
        Ok(projected.tensor().clone())
    }

    fn cross_attention<C>(
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

        let k_t = k.transpose(-2, -1).map_err(Error::Numr)?.contiguous();
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
            .contiguous()
            .reshape(&[batch, tgt_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        let projected = self.cross_out_proj.forward(client, &Var::new(out, false))?;
        Ok(projected.tensor().clone())
    }
}
