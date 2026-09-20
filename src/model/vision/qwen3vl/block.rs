//! One pre-norm transformer block of the vision tower.

use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use crate::nn::{LayerNorm, MaybeQuantLinear, VarBuilder};
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// `LN1 -> fused qkv -> 2D rope -> full attention -> out`, then
/// `LN2 -> up -> gelu -> down`, each with a residual.
pub(super) struct Qwen3VlBlock<R: Runtime> {
    ln1: LayerNorm<R>,
    ln2: LayerNorm<R>,
    qkv: MaybeQuantLinear<R>,
    out: MaybeQuantLinear<R>,
    ffn_up: MaybeQuantLinear<R>,
    ffn_down: MaybeQuantLinear<R>,
    num_heads: usize,
    head_dim: usize,
}

impl<R: Runtime<DType = DType>> Qwen3VlBlock<R> {
    /// Take `v.blk.{index}.*` from `vb`.
    pub(super) fn from_varbuilder(
        vb: &mut VarBuilder<R>,
        index: usize,
        hidden: usize,
        intermediate: usize,
        num_heads: usize,
        eps: f32,
    ) -> Result<Self>
    where
        R::Client: DequantOps<R>,
    {
        let p = format!("v.blk.{index}");
        let ln = |vb: &mut VarBuilder<R>, name: &str| -> Result<LayerNorm<R>> {
            let w = vb.take_tensor_dequant(&format!("{p}.{name}.weight"), DType::F32)?;
            let b = vb.take_tensor_dequant(&format!("{p}.{name}.bias"), DType::F32)?;
            if w.shape() != [hidden].as_slice() || b.shape() != [hidden].as_slice() {
                return Err(Error::ModelError {
                    reason: format!(
                        "qwen3vl vision: {p}.{name} has shape {:?}/{:?}, want [{hidden}]",
                        w.shape(),
                        b.shape()
                    ),
                });
            }
            Ok(LayerNorm::new(w, b, eps, false))
        };
        let ln1 = ln(vb, "ln1")?;
        let ln2 = ln(vb, "ln2")?;
        let lin =
            |vb: &mut VarBuilder<R>, name: &str, want: [usize; 2]| -> Result<MaybeQuantLinear<R>> {
                let weight_name = format!("{p}.{name}.weight");
                let bias_name = format!("{p}.{name}.bias");
                let layer = vb.take_maybe_quant_linear(&weight_name, Some(bias_name.as_str()))?;
                if layer.shape() != want.as_slice() {
                    return Err(Error::ModelError {
                        reason: format!(
                            "qwen3vl vision: {p}.{name}.weight has shape {:?}, want {want:?}",
                            layer.shape()
                        ),
                    });
                }
                Ok(layer)
            };
        let qkv = lin(vb, "attn_qkv", [3 * hidden, hidden])?;
        let out = lin(vb, "attn_out", [hidden, hidden])?;
        let ffn_up = lin(vb, "ffn_up", [intermediate, hidden])?;
        let ffn_down = lin(vb, "ffn_down", [hidden, intermediate])?;
        Ok(Self {
            ln1,
            ln2,
            qkv,
            out,
            ffn_up,
            ffn_down,
            num_heads,
            head_dim: hidden / num_heads,
        })
    }

    /// `x`: `[S, hidden]`. `cos`/`sin`: `[S, head_dim / 2]`.
    pub(super) fn forward<C>(
        &self,
        client: &C,
        x: &Tensor<R>,
        cos: &Var<R>,
        sin: &Var<R>,
    ) -> Result<Tensor<R>>
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
            + DequantOps<R>,
    {
        let x_var = Var::new(x.clone(), false);
        let h = self.ln1.forward(client, &x_var)?;
        let attn = self.attention(client, h.tensor(), cos, sin)?;
        let x1 = client.add(x, &attn)?;

        let h2 = self.ln2.forward(client, &Var::new(x1.clone(), false))?;
        let up = self.ffn_up.forward(client, &h2)?;
        let act = client.gelu(up.tensor())?;
        let down = self.ffn_down.forward(client, &Var::new(act, false))?;
        Ok(client.add(&x1, down.tensor())?)
    }

    fn attention<C>(
        &self,
        client: &C,
        h: &Tensor<R>,
        cos: &Var<R>,
        sin: &Var<R>,
    ) -> Result<Tensor<R>>
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
            + DequantOps<R>,
    {
        let s = h.shape()[0];
        let c = self.num_heads * self.head_dim;
        let qkv = self.qkv.forward(client, &Var::new(h.clone(), false))?;
        let qkv = qkv.tensor();
        // [S, 3C] -> three [1, H, S, D]
        let head_split = |offset: usize| -> Result<Tensor<R>> {
            Ok(qkv
                .narrow(1, offset, c)?
                .contiguous()?
                .reshape(&[1, s, self.num_heads, self.head_dim])?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let q = Var::new(head_split(0)?, false);
        let k = Var::new(head_split(c)?, false);
        let v = head_split(2 * c)?;
        let q = client.apply_rope(&q, cos, sin)?;
        let k = client.apply_rope(&k, cos, sin)?;

        let k_t = k.tensor().transpose(-2, -1)?.contiguous()?;
        let scores = client.matmul(&q.tensor().contiguous()?, &k_t)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = client.mul_scalar(&scores, scale)?;
        let probs = client.softmax(&scores, -1)?;
        let ctx = client.matmul(&probs, &v)?;
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape(&[s, c])?;
        let out = self.out.forward(client, &Var::new(ctx, false))?;
        Ok(out.tensor().clone())
    }
}
