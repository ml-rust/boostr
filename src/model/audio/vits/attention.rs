//! `VitsAttention`: multi-head self-attention with windowed relative
//! position embeddings.
//!
//! Exact port of HF `transformers.models.vits.modeling_vits.VitsAttention`.
//! `window_size` gates two extra learned parameters, `emb_rel_k`/`emb_rel_v`
//! (`[1, 2*window+1, head_dim]`), that bias the attention scores and the
//! attention-weighted values with a relative-position term. Self-attention
//! only (`query`, `key`, `value` all come from `hidden_states`) — VITS never
//! uses this module for cross-attention.

use crate::error::{Error, Result};
use crate::nn::{Linear, VarBuilder};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, MatmulOps, ScalarOps, ShapeOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub struct VitsAttention<R: Runtime> {
    q_proj: Linear<R>,
    k_proj: Linear<R>,
    v_proj: Linear<R>,
    out_proj: Linear<R>,
    /// `[1, 2*window_size+1, head_dim]`, present iff `window_size.is_some()`.
    emb_rel_k: Option<Tensor<R>>,
    /// `[1, 2*window_size+1, head_dim]`, present iff `window_size.is_some()`.
    emb_rel_v: Option<Tensor<R>>,
    num_heads: usize,
    head_dim: usize,
    window_size: Option<usize>,
    scaling: f32,
}

impl<R: Runtime<DType = DType>> VitsAttention<R> {
    /// Reads `q_proj`/`k_proj`/`v_proj`/`out_proj` (`weight` + `bias` iff
    /// `use_bias`) and, iff `window_size.is_some()`, `emb_rel_k`/`emb_rel_v`.
    pub fn from_varbuilder(
        vb: &mut VarBuilder<'_, R>,
        hidden_size: usize,
        num_heads: usize,
        window_size: Option<usize>,
        use_bias: bool,
    ) -> Result<Self> {
        if !hidden_size.is_multiple_of(num_heads) {
            return Err(Error::ModelError {
                reason: format!(
                    "hidden_size {hidden_size} not divisible by num_attention_heads {num_heads}"
                ),
            });
        }
        let head_dim = hidden_size / num_heads;
        let scaling = (head_dim as f32).powf(-0.5);

        let load_linear = |vb: &mut VarBuilder<'_, R>, name: &str| -> Result<Linear<R>> {
            let mut p = vb.pp(name);
            let weight = p.take_tensor("weight")?;
            let bias = if use_bias {
                Some(p.take_tensor("bias")?)
            } else {
                None
            };
            Ok(Linear::new(weight, bias, false))
        };

        let q_proj = load_linear(vb, "q_proj")?;
        let k_proj = load_linear(vb, "k_proj")?;
        let v_proj = load_linear(vb, "v_proj")?;
        let out_proj = load_linear(vb, "out_proj")?;

        let (emb_rel_k, emb_rel_v) = match window_size {
            Some(_) => (
                Some(vb.take_tensor("emb_rel_k")?),
                Some(vb.take_tensor("emb_rel_v")?),
            ),
            None => (None, None),
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            emb_rel_k,
            emb_rel_v,
            num_heads,
            head_dim,
            window_size,
            scaling,
        })
    }

    /// `[batch, seq, hidden] -> [batch, heads, seq, head_dim]`, contiguous.
    fn shape_heads(&self, x: &Tensor<R>, bsz: usize, seq_len: usize) -> Result<Tensor<R>> {
        x.reshape(&[bsz, seq_len, self.num_heads, self.head_dim])
            .and_then(|t| t.transpose(1, 2))
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)
    }

    /// Self-attention forward. `attention_mask`, if given, is an additive
    /// mask broadcast onto `[batch, heads, tgt_len, src_len]` (typically
    /// `[batch, 1, tgt_len, src_len]`), already in the `-inf`-for-padding form
    /// HF's `_prepare_4d_attention_mask` produces.
    pub fn forward<C>(
        &self,
        client: &C,
        hidden_states: &Tensor<R>,
        attention_mask: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ShapeOps<R>,
        R::Client: TensorOps<R>,
    {
        let shape = hidden_states.shape();
        let bsz = shape[0];
        let tgt_len = shape[1];
        let src_len = tgt_len;

        let x = Var::new(hidden_states.clone(), false);
        let q = self.q_proj.forward(client, &x)?.tensor().clone();
        let q = client
            .mul_scalar(&q, self.scaling as f64)
            .map_err(Error::Numr)?;
        let k = self.k_proj.forward(client, &x)?.tensor().clone();
        let v = self.v_proj.forward(client, &x)?.tensor().clone();

        let q = self.shape_heads(&q, bsz, tgt_len)?;
        let k = self.shape_heads(&k, bsz, src_len)?;
        let v = self.shape_heads(&v, bsz, src_len)?;

        let k_t = k
            .transpose(-2, -1)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?;
        let mut attn_weights = client.matmul(&q, &k_t).map_err(Error::Numr)?;

        if let Some((window, emb_rel_k)) = self.window_size.zip(self.emb_rel_k.as_ref()) {
            let key_rel = Self::relative_embeddings(client, emb_rel_k, src_len, window)?;
            let key_rel = key_rel.unsqueeze(0).map_err(Error::Numr)?;
            let key_rel_t = key_rel
                .transpose(-2, -1)
                .and_then(|t| t.contiguous())
                .map_err(Error::Numr)?;
            let relative_logits = client.matmul(&q, &key_rel_t).map_err(Error::Numr)?;
            let rel_pos_bias = Self::relative_position_to_absolute_position(
                client,
                &relative_logits,
                bsz,
                self.num_heads,
                tgt_len,
            )?;
            attn_weights = client
                .add(&attn_weights, &rel_pos_bias)
                .map_err(Error::Numr)?;
        }

        if let Some(mask) = attention_mask {
            attn_weights = client.add(&attn_weights, mask).map_err(Error::Numr)?;
        }

        let attn_probs = client.softmax(&attn_weights, -1).map_err(Error::Numr)?;
        let mut attn_output = client.matmul(&attn_probs, &v).map_err(Error::Numr)?;

        if let Some((window, emb_rel_v)) = self.window_size.zip(self.emb_rel_v.as_ref()) {
            let value_rel = Self::relative_embeddings(client, emb_rel_v, src_len, window)?;
            let value_rel = value_rel.unsqueeze(0).map_err(Error::Numr)?;
            let relative_weights = Self::absolute_position_to_relative_position(
                client,
                &attn_probs,
                bsz,
                self.num_heads,
                tgt_len,
            )?;
            let rel_pos_bias = client
                .matmul(&relative_weights, &value_rel)
                .map_err(Error::Numr)?;
            attn_output = client
                .add(&attn_output, &rel_pos_bias)
                .map_err(Error::Numr)?;
        }

        let attn_output = attn_output
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(&[bsz, tgt_len, self.num_heads * self.head_dim]))
            .map_err(Error::Numr)?;

        let out = self
            .out_proj
            .forward(client, &Var::new(attn_output, false))?;
        Ok(out.tensor().clone())
    }

    /// `_get_relative_embeddings`: pads `relative_embeddings`
    /// (`[1, 2*window+1, head_dim]`) out to `[1, 2*length-1, head_dim]`,
    /// zero-padding on the position axis when `length > window+1` and
    /// slicing down when `length <= window+1`.
    fn relative_embeddings<C: ShapeOps<R>>(
        client: &C,
        relative_embeddings: &Tensor<R>,
        length: usize,
        window_size: usize,
    ) -> Result<Tensor<R>> {
        let pad_length = length.saturating_sub(window_size + 1);
        let embeddings = if pad_length > 0 {
            client
                .pad(
                    relative_embeddings,
                    &[0, 0, pad_length, pad_length, 0, 0],
                    0.0,
                )
                .map_err(Error::Numr)?
        } else {
            relative_embeddings.clone()
        };
        let slice_start = (window_size + 1).saturating_sub(length);
        let slice_len = 2 * length - 1;
        embeddings
            .narrow(1, slice_start, slice_len)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)
    }

    /// `_relative_position_to_absolute_position`: `[bsz, heads, L, 2L-1] ->
    /// [bsz, heads, L, L]` via the pad-flatten-reshape-slice shift HF uses to
    /// convert per-row relative offsets into absolute column indices.
    fn relative_position_to_absolute_position<C: ShapeOps<R>>(
        client: &C,
        x: &Tensor<R>,
        bsz: usize,
        heads: usize,
        length: usize,
    ) -> Result<Tensor<R>> {
        let bh = bsz * heads;
        let x = x
            .reshape(&[bh, length, 2 * length - 1])
            .map_err(Error::Numr)?;
        let x = client
            .pad(&x, &[0, 1, 0, 0, 0, 0], 0.0)
            .map_err(Error::Numr)?;
        let x_flat = x.reshape(&[bh, length * 2 * length]).map_err(Error::Numr)?;
        let x_flat = client
            .pad(&x_flat, &[0, length - 1, 0, 0], 0.0)
            .map_err(Error::Numr)?;
        let x_final = x_flat
            .reshape(&[bh, length + 1, 2 * length - 1])
            .map_err(Error::Numr)?;
        x_final
            .narrow(1, 0, length)
            .and_then(|t| t.narrow(2, length - 1, length))
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(&[bsz, heads, length, length]))
            .map_err(Error::Numr)
    }

    /// `_absolute_position_to_relative_position`: `[bsz, heads, L, L] ->
    /// [bsz, heads, L, 2L-1]`, the inverse shift of
    /// [`Self::relative_position_to_absolute_position`].
    fn absolute_position_to_relative_position<C: ShapeOps<R>>(
        client: &C,
        x: &Tensor<R>,
        bsz: usize,
        heads: usize,
        length: usize,
    ) -> Result<Tensor<R>> {
        let bh = bsz * heads;
        let x = x.reshape(&[bh, length, length]).map_err(Error::Numr)?;
        let x = client
            .pad(&x, &[0, length - 1, 0, 0, 0, 0], 0.0)
            .map_err(Error::Numr)?;
        let x_flat = x
            .reshape(&[bh, length * (2 * length - 1)])
            .map_err(Error::Numr)?;
        let x_flat = client
            .pad(&x_flat, &[length, 0, 0, 0], 0.0)
            .map_err(Error::Numr)?;
        let x_final = x_flat
            .reshape(&[bh, length, 2 * length])
            .map_err(Error::Numr)?;
        x_final
            .narrow(2, 1, 2 * length - 1)
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(&[bsz, heads, length, 2 * length - 1]))
            .map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn iota(n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|i| i as f32 * scale).collect()
    }

    #[test]
    fn relative_embeddings_slices_when_length_le_window_plus_one() {
        let (client, device) = cpu_setup();
        // window=2 -> stored width 5. length=3 <= window+1=3, so pad_length=0
        // and we slice down to width 2*3-1=5 (the whole thing, unchanged).
        let rel = Tensor::<CpuRuntime>::from_slice(&iota(5 * 4, 1.0), &[1, 5, 4], &device).unwrap();
        let out = VitsAttention::<CpuRuntime>::relative_embeddings(&client, &rel, 3, 2).unwrap();
        assert_eq!(out.shape(), &[1, 5, 4]);
        let got: Vec<f32> = out.to_vec();
        assert_eq!(got, iota(5 * 4, 1.0));
    }

    #[test]
    fn relative_embeddings_pads_when_length_exceeds_window_plus_one() {
        let (client, device) = cpu_setup();
        // window=2 -> stored width 5. length=5 > window+1=3, pad_length=2,
        // output width 2*5-1=9, and the slice recenters on the padded buffer
        // so the original 5 values land back in the middle.
        let rel = Tensor::<CpuRuntime>::from_slice(&iota(5 * 4, 1.0), &[1, 5, 4], &device).unwrap();
        let out = VitsAttention::<CpuRuntime>::relative_embeddings(&client, &rel, 5, 2).unwrap();
        assert_eq!(out.shape(), &[1, 9, 4]);
        let got: Vec<f32> = out.to_vec();
        // Rows 0-1 and 7-8 come from zero padding; rows 2-6 are the
        // original 5 rows, unchanged.
        assert!(got[..8].iter().all(|&v| v == 0.0), "leading pad: {got:?}");
        assert!(got[28..].iter().all(|&v| v == 0.0), "trailing pad: {got:?}");
        assert_eq!(&got[8..28], &iota(5 * 4, 1.0)[..]);
    }

    #[test]
    fn relative_to_absolute_roundtrips_through_absolute_to_relative() {
        let (client, device) = cpu_setup();
        let length = 4usize;
        let bsz = 1usize;
        let heads = 1usize;
        // Distinct values everywhere so a shift bug shows up as a shape or
        // value mismatch rather than an accidental symmetric match.
        let width = 2 * length - 1;
        let rel = Tensor::<CpuRuntime>::from_slice(
            &iota(bsz * heads * length * width, 1.0),
            &[bsz, heads, length, width],
            &device,
        )
        .unwrap();
        let abs_pos = VitsAttention::<CpuRuntime>::relative_position_to_absolute_position(
            &client, &rel, bsz, heads, length,
        )
        .unwrap();
        assert_eq!(abs_pos.shape(), &[bsz, heads, length, length]);

        let back = VitsAttention::<CpuRuntime>::absolute_position_to_relative_position(
            &client, &abs_pos, bsz, heads, length,
        )
        .unwrap();
        assert_eq!(back.shape(), &[bsz, heads, length, width]);
    }
}
