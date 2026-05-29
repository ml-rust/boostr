//! Single transformer encoder layer: self-attention + FFN with residual + LayerNorm.

use crate::error::{Error, Result};
use crate::model::encoder::config::HiddenAct;
use crate::nn::{LayerNorm, MaybeQuantLinear};
use crate::quant::traits::QuantMatmulOps;
use numr::autograd::{Var, var_add, var_matmul, var_permute, var_reshape, var_transpose};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, IndexingOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps,
    TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// A single transformer encoder layer: self-attention + FFN with residual + LayerNorm.
pub(in crate::model::encoder) struct EncoderLayer<R: Runtime> {
    pub(in crate::model::encoder) q_proj: MaybeQuantLinear<R>,
    pub(in crate::model::encoder) k_proj: MaybeQuantLinear<R>,
    pub(in crate::model::encoder) v_proj: MaybeQuantLinear<R>,
    pub(in crate::model::encoder) o_proj: MaybeQuantLinear<R>,
    pub(in crate::model::encoder) attn_norm: LayerNorm<R>,
    pub(in crate::model::encoder) ffn_up: MaybeQuantLinear<R>,
    pub(in crate::model::encoder) ffn_down: MaybeQuantLinear<R>,
    pub(in crate::model::encoder) ffn_norm: LayerNorm<R>,
    pub(in crate::model::encoder) num_heads: usize,
    pub(in crate::model::encoder) head_dim: usize,
    pub(in crate::model::encoder) hidden_act: HiddenAct,
}

impl<R: Runtime<DType = DType>> EncoderLayer<R> {
    pub(super) fn forward<C>(
        &self,
        client: &C,
        x: &Var<R>,
        attention_mask: Option<&Tensor<R>>,
    ) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + BinaryOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + IndexingOps<R>
            + ActivationOps<R>
            + UnaryOps<R>
            + NormalizationOps<R>
            + QuantMatmulOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let attn_out = self.self_attention(client, x, attention_mask)?;
        let x = var_add(x, &attn_out, client).map_err(Error::Numr)?;
        let x = self.attn_norm.forward(client, &x)?;

        let ffn_out = self.ffn(client, &x)?;
        let x = var_add(&x, &ffn_out, client).map_err(Error::Numr)?;
        let x = self.ffn_norm.forward(client, &x)?;

        Ok(x)
    }

    fn self_attention<C>(
        &self,
        client: &C,
        x: &Var<R>,
        attention_mask: Option<&Tensor<R>>,
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
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let shape = x.shape().to_vec();
        let batch = shape[0];
        let seq_len = shape[1];

        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        let q = var_reshape(&q, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let k = var_reshape(&k, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let v = var_reshape(&v, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?;

        let q = var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let k = var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let v = var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;

        let q = Var::new(q.tensor().contiguous(), false);
        let k = Var::new(k.tensor().contiguous(), false);
        let v = Var::new(v.tensor().contiguous(), false);

        let k_t = var_transpose(&k).map_err(Error::Numr)?;
        let scores = var_matmul(&q, &k_t, client).map_err(Error::Numr)?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scores = Var::new(
            client
                .mul_scalar(scores.tensor(), scale as f64)
                .map_err(Error::Numr)?,
            false,
        );

        let scores = if let Some(mask) = attention_mask {
            let mask_shape = mask.shape().to_vec();
            if mask_shape.len() != 2 || mask_shape[0] != batch || mask_shape[1] != seq_len {
                return Err(Error::ModelError {
                    reason: format!(
                        "attention_mask shape must be [{batch}, {seq_len}], got {:?}",
                        mask_shape
                    ),
                });
            }
            let inv = client.rsub_scalar(mask, 1.0).map_err(Error::Numr)?;
            let additive = client.mul_scalar(&inv, -1e9).map_err(Error::Numr)?;
            let additive = additive
                .reshape(&[batch, 1, 1, seq_len])
                .map_err(Error::Numr)?;
            let biased = client
                .add(scores.tensor(), &additive)
                .map_err(Error::Numr)?;
            Var::new(biased, false)
        } else {
            scores
        };

        let attn_weights = Var::new(
            client.softmax(scores.tensor(), -1).map_err(Error::Numr)?,
            false,
        );
        let attn_out = var_matmul(&attn_weights, &v, client).map_err(Error::Numr)?;

        let attn_out = var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = Var::new(attn_out.tensor().contiguous(), false);
        let hidden = self.num_heads * self.head_dim;
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, hidden]).map_err(Error::Numr)?;

        let o = self.o_proj.forward(client, &attn_out)?;
        Ok(o)
    }

    fn ffn<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ActivationOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R>,
    {
        let h = self.ffn_up.forward(client, x)?;
        let h = match self.hidden_act {
            HiddenAct::Gelu => Var::new(client.gelu(h.tensor()).map_err(Error::Numr)?, false),
            HiddenAct::Relu => Var::new(client.relu(h.tensor()).map_err(Error::Numr)?, false),
        };
        let out = self.ffn_down.forward(client, &h)?;
        Ok(out)
    }
}
