//! `TransformerBlock` — one of the 12 pre-norm transformer layers in
//! NeuCodec's acoustic decoder.
//!
//! Checkpoint shapes (hidden 1024, 16 heads x head_dim 64):
//! `input_layernorm` (RMSNorm, weight-only) -> self-attention (`q/k/v/o_proj`,
//! weight-only, no biases, no GQA) -> `post_attention_layernorm` (RMSNorm,
//! weight-only) -> plain 2-layer MLP (`mlp.fc1` `[4096,1024]` ->
//! `mlp.fc2` `[1024,4096]`, **not** SwiGLU — there is no gate projection).
//! Pre-norm residual structure throughout. Full (non-causal) attention: this
//! decoder sees the whole latent sequence at once, so no mask is applied.
//!
//! ## No positional encoding — deliberately, to match the released weights
//!
//! `config.json` advertises `rope_parameters`, and upstream's `Attention` does
//! call a `RotaryPositionalEmbeddings` on `q` and `k`. It has no effect:
//!
//! * `RotaryPositionalEmbeddings.forward` documents its input as
//!   `[b, s, n_h, h_d]` and reads the sequence length as `x.size(1)`, but
//!   `Attention.forward` hands it `q`/`k` shaped `(b, h, t, d)`. Axis 1 is the
//!   HEAD axis, so the rotation angle is selected by head index and is
//!   CONSTANT across time.
//! * A per-head rotation `R_h` that does not vary with `t` is orthogonal and
//!   applied to both `q` and `k`, so every score is
//!   `(R_h q_t)·(R_h k_s) = q_t·k_s` — it cancels exactly.
//!
//! Both were verified against the installed `neucodec` package: RoPE input
//! shapes come through as `(2, 16, 7, 64)` for `t = 7`, upstream attention is
//! exactly permutation-equivariant over time (max deviation 2.9e-7), and
//! swapping the rotation for the identity changes the output by 1.5e-7 at an
//! output scale of 0.47 — i.e. float32 noise.
//!
//! So the released decoder has NO effective positional information, and this
//! port applies none. Adding a genuine time-indexed RoPE here would be a
//! silent parity break: it is not a no-op, and the weights were never trained
//! with one.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::nn::{Linear, RmsNorm, var_contiguous};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use numr::autograd::{Var, var_add, var_permute, var_reshape, var_silu};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Bundled, already-built weights for one `TransformerBlock`.
pub struct TransformerBlockWeights<R: Runtime> {
    pub input_layernorm: RmsNorm<R>,
    pub q_proj: Linear<R>,
    pub k_proj: Linear<R>,
    pub v_proj: Linear<R>,
    pub o_proj: Linear<R>,
    pub post_attention_layernorm: RmsNorm<R>,
    pub mlp_fc1: Linear<R>,
    pub mlp_fc2: Linear<R>,
}

/// One pre-norm transformer block with plain (non-gated) MLP.
pub struct TransformerBlock<R: Runtime> {
    input_layernorm: RmsNorm<R>,
    q_proj: Linear<R>,
    k_proj: Linear<R>,
    v_proj: Linear<R>,
    o_proj: Linear<R>,
    post_attention_layernorm: RmsNorm<R>,
    mlp_fc1: Linear<R>,
    mlp_fc2: Linear<R>,
    num_heads: usize,
    head_dim: usize,
}

impl<R: Runtime> TransformerBlock<R> {
    pub fn new(
        weights: TransformerBlockWeights<R>,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        if num_heads == 0 || head_dim == 0 {
            return Err(Error::InvalidArgument {
                arg: "num_heads/head_dim",
                reason: "must both be > 0".into(),
            });
        }
        Ok(Self {
            input_layernorm: weights.input_layernorm,
            q_proj: weights.q_proj,
            k_proj: weights.k_proj,
            v_proj: weights.v_proj,
            o_proj: weights.o_proj,
            post_attention_layernorm: weights.post_attention_layernorm,
            mlp_fc1: weights.mlp_fc1,
            mlp_fc2: weights.mlp_fc2,
            num_heads,
            head_dim,
        })
    }
}

impl<R: Runtime<DType = DType>> TransformerBlock<R> {
    /// Forward: `x [B, T, hidden] -> [B, T, hidden]`. Full (non-causal)
    /// self-attention over the whole sequence.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self.self_attn(client, &normed)?;
        let h = var_add(x, &attn_out, client).map_err(Error::Numr)?;

        let normed = self.post_attention_layernorm.forward(client, &h)?;
        let fc1_out = self.mlp_fc1.forward(client, &normed)?;
        let activated = var_silu(&fc1_out, client).map_err(Error::Numr)?;
        let mlp_out = self.mlp_fc2.forward(client, &activated)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }

    fn self_attn<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let shape = x.shape().to_vec();
        if shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("expected [B, T, hidden], got {shape:?}"),
            });
        }
        let (batch, seq_len, hidden) = (shape[0], shape[1], shape[2]);
        let expected_hidden = self.num_heads * self.head_dim;
        if hidden != expected_hidden {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "last dim {hidden} must equal num_heads*head_dim ({expected_hidden})"
                ),
            });
        }

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

        let q = var_contiguous(&q)?;
        let k = var_contiguous(&k)?;
        let v = var_contiguous(&v)?;

        // No RoPE — see the module doc: upstream's rotation is a verified
        // no-op, so applying one here would break parity with the weights.

        // No mask: NeuCodec's acoustic decoder attends over the full latent
        // sequence (non-causal), unlike an autoregressive LLaMA-style block.
        let attn_out = multi_head_attention_impl(client, &q, &k, &v, None, self.num_heads)?;

        let attn_out = var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out)?;
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, hidden]).map_err(Error::Numr)?;

        self.o_proj.forward(client, &attn_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn linear(out_f: usize, in_f: usize, val: f32, device: &CpuDevice) -> Linear<CpuRuntime> {
        Linear::new(
            Tensor::<CpuRuntime>::from_slice(&vec![val; out_f * in_f], &[out_f, in_f], device),
            None,
            false,
        )
    }

    fn rms_norm(dim: usize, device: &CpuDevice) -> RmsNorm<CpuRuntime> {
        RmsNorm::new(
            Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; dim], &[dim], device),
            1e-6,
            false,
        )
    }

    fn block(
        hidden: usize,
        heads: usize,
        head_dim: usize,
        mlp: usize,
        device: &CpuDevice,
    ) -> TransformerBlock<CpuRuntime> {
        TransformerBlock::new(
            TransformerBlockWeights {
                input_layernorm: rms_norm(hidden, device),
                q_proj: linear(hidden, hidden, 0.01, device),
                k_proj: linear(hidden, hidden, 0.01, device),
                v_proj: linear(hidden, hidden, 0.01, device),
                o_proj: linear(hidden, hidden, 0.01, device),
                post_attention_layernorm: rms_norm(hidden, device),
                mlp_fc1: linear(mlp, hidden, 0.01, device),
                mlp_fc2: linear(hidden, mlp, 0.01, device),
            },
            heads,
            head_dim,
        )
        .unwrap()
    }

    #[test]
    fn forward_preserves_shape() {
        let (client, device) = cpu_setup();
        let (hidden, heads, head_dim, mlp) = (16, 4, 4, 32);
        let b = block(hidden, heads, head_dim, mlp, &device);

        let x_data: Vec<f32> = (0..(2 * 5 * hidden))
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[2, 5, hidden], &device),
            false,
        );
        let out = b.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[2, 5, hidden]);
    }

    #[test]
    fn output_is_finite() {
        let (client, device) = cpu_setup();
        let (hidden, heads, head_dim, mlp) = (16, 4, 4, 32);
        let b = block(hidden, heads, head_dim, mlp, &device);

        let x_data: Vec<f32> = (0..(7 * hidden)).map(|i| (i as f32 * 0.11).cos()).collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[1, 7, hidden], &device),
            false,
        );
        let out = b.forward(&client, &x).unwrap();
        for v in out.tensor().contiguous().unwrap().to_vec::<f32>() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn rejects_hidden_dim_mismatch() {
        let (client, device) = cpu_setup();
        let (hidden, heads, head_dim, mlp) = (16, 4, 4, 32);
        let b = block(hidden, heads, head_dim, mlp, &device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; 2 * 5 * 8], &[2, 5, 8], &device),
            false,
        );
        assert!(b.forward(&client, &x).is_err());
    }

    #[test]
    fn new_rejects_zero_heads() {
        let (_client, device) = cpu_setup();
        let hidden = 8;
        let weights = TransformerBlockWeights {
            input_layernorm: rms_norm(hidden, &device),
            q_proj: linear(hidden, hidden, 0.01, &device),
            k_proj: linear(hidden, hidden, 0.01, &device),
            v_proj: linear(hidden, hidden, 0.01, &device),
            o_proj: linear(hidden, hidden, 0.01, &device),
            post_attention_layernorm: rms_norm(hidden, &device),
            mlp_fc1: linear(16, hidden, 0.01, &device),
            mlp_fc2: linear(hidden, 16, 0.01, &device),
        };
        assert!(TransformerBlock::new(weights, 0, 4).is_err());
    }
}
