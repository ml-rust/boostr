use super::embeddings::AlbertConfig;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps,
    TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// One ALBERT transformer layer (reused 12 times by `AlbertModel`).
pub struct AlbertLayer<R: Runtime> {
    // Attention
    pub q_weight: Tensor<R>,
    pub q_bias: Tensor<R>,
    pub k_weight: Tensor<R>,
    pub k_bias: Tensor<R>,
    pub v_weight: Tensor<R>,
    pub v_bias: Tensor<R>,
    pub attn_dense_weight: Tensor<R>,
    pub attn_dense_bias: Tensor<R>,
    pub attn_ln_weight: Tensor<R>,
    pub attn_ln_bias: Tensor<R>,
    // FFN
    pub ffn_weight: Tensor<R>,
    pub ffn_bias: Tensor<R>,
    pub ffn_output_weight: Tensor<R>,
    pub ffn_output_bias: Tensor<R>,
    pub full_ln_weight: Tensor<R>,
    pub full_ln_bias: Tensor<R>,
}

impl<R: Runtime> AlbertLayer<R> {
    /// Forward: `x [B, T, H]` → `[B, T, H]`.
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>, config: &AlbertConfig) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + MatmulOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + TensorOps<R>
            + ReduceOps<R>
            + UnaryOps<R>
            + ShapeOps<R>
            + TypeConversionOps<R>
            + UtilityOps<R>,
    {
        let shape = x.shape();
        let (b, t, h) = (shape[0], shape[1], shape[2]);
        let n_heads = config.num_attention_heads;
        let d_head = config.head_dim();

        // Flatten batch-time for matmul.
        let flat = x.reshape(&[b * t, h]).map_err(Error::Numr)?;
        let q = linear(client, &flat, &self.q_weight, &self.q_bias)?;
        let k = linear(client, &flat, &self.k_weight, &self.k_bias)?;
        let v = linear(client, &flat, &self.v_weight, &self.v_bias)?;

        // Reshape to multi-head: [B*T, H] → [B, T, n_heads, d_head] → [B, n_heads, T, d_head].
        let q = q
            .reshape(&[b, t, n_heads, d_head])
            .map_err(Error::Numr)?
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()?;
        let k = k
            .reshape(&[b, t, n_heads, d_head])
            .map_err(Error::Numr)?
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()?;
        let v = v
            .reshape(&[b, t, n_heads, d_head])
            .map_err(Error::Numr)?
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()?;

        // Attention scores: q @ k.T / sqrt(d_head). k.T swaps the last two dims.
        let k_t = k.transpose(2, 3).map_err(Error::Numr)?.contiguous()?;
        let scores = client.matmul(&q, &k_t).map_err(Error::Numr)?;
        let scale = 1.0 / (d_head as f64).sqrt();
        let scaled = client.mul_scalar(&scores, scale).map_err(Error::Numr)?;
        let attn = client.softmax(&scaled, -1).map_err(Error::Numr)?;
        // attn: [B, H, T, T]; v: [B, H, T, d_head] → ctx: [B, H, T, d_head].
        let ctx = client.matmul(&attn, &v).map_err(Error::Numr)?;

        // [B, H, T, d_head] → [B, T, H, d_head] → [B, T, hidden].
        let ctx_merged = ctx
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()?
            .reshape(&[b, t, h])
            .map_err(Error::Numr)?;

        // Attention output projection + residual + LayerNorm.
        let ctx_flat = ctx_merged.reshape(&[b * t, h]).map_err(Error::Numr)?;
        let attn_out = linear(
            client,
            &ctx_flat,
            &self.attn_dense_weight,
            &self.attn_dense_bias,
        )?;
        let attn_out_shaped = attn_out.reshape(&[b, t, h]).map_err(Error::Numr)?;
        let residual1 = client.add(&attn_out_shaped, x).map_err(Error::Numr)?;
        let post_attn = client
            .layer_norm(
                &residual1,
                &self.attn_ln_weight,
                &self.attn_ln_bias,
                config.layer_norm_eps,
            )
            .map_err(Error::Numr)?;

        // FFN: Linear → GELU → Linear → residual → LayerNorm.
        let post_attn_flat = post_attn.reshape(&[b * t, h]).map_err(Error::Numr)?;
        let ffn1 = linear(client, &post_attn_flat, &self.ffn_weight, &self.ffn_bias)?;
        let ffn1_gelu = client.gelu(&ffn1).map_err(Error::Numr)?;
        let ffn2 = linear(
            client,
            &ffn1_gelu,
            &self.ffn_output_weight,
            &self.ffn_output_bias,
        )?;
        let ffn2_shaped = ffn2.reshape(&[b, t, h]).map_err(Error::Numr)?;
        let residual2 = client.add(&ffn2_shaped, &post_attn).map_err(Error::Numr)?;
        client
            .layer_norm(
                &residual2,
                &self.full_ln_weight,
                &self.full_ln_bias,
                config.layer_norm_eps,
            )
            .map_err(Error::Numr)
    }
}

/// Linear helper: `input @ weight.T + bias` for `input [N, in], weight [out, in]`.
pub(super) fn linear<R, C>(
    client: &C,
    input: &Tensor<R>,
    weight: &Tensor<R>,
    bias: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + MatmulOps<R> + TensorOps<R>,
{
    let w_t = weight.transpose(0, 1).map_err(Error::Numr)?;
    client.matmul_bias(input, &w_t, bias).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use crate::model::audio::kokoro::bert::test_support::{build_layer, tiny_config, zeros};
    use crate::test_utils::cpu_setup;

    #[test]
    fn albert_layer_preserves_shape() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let layer = build_layer(&cfg, &device);
        let x = zeros(&[1, 5, cfg.hidden_size], &device);
        let y = layer.forward(&client, &x, &cfg).unwrap();
        assert_eq!(y.shape(), &[1, 5, cfg.hidden_size]);
    }
}
