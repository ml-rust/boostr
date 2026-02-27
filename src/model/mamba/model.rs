//! Full Mamba2 model — embedding + N layers + norm + LM head
//!
//! Mirrors the Llama model structure but uses Mamba2 SSM layers
//! instead of transformer attention blocks.

use crate::error::{Error, Result};
use crate::inference::{LayeredSsmState, SsmState};
use crate::model::config::UniversalConfig;
use crate::model::mamba::mamba2::{Mamba2, Mamba2Config};
use crate::model::traits::ModelClient;
use crate::nn::{Embedding, Linear, RmsNorm, VarBuilder};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, IndexingOps, NormalizationOps, ReduceOps, ScalarOps,
    TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Full Mamba2 model for inference.
///
/// Structure: `embed_tokens → [Mamba2 layers with RmsNorm] → final_norm → lm_head`
pub struct Mamba2Model<R: Runtime> {
    config: UniversalConfig,
    mamba_config: Mamba2Config,
    embed_tokens: Embedding<R>,
    layers: Vec<Mamba2Block<R>>,
    norm: RmsNorm<R>,
    lm_head: Linear<R>,
}

/// Single Mamba2 block: pre-norm → Mamba2 → residual
struct Mamba2Block<R: Runtime> {
    norm: RmsNorm<R>,
    mamba: Mamba2<R>,
}

impl<R: Runtime<DType = DType>> Mamba2Model<R>
where
    R::Client: IndexingOps<R>,
{
    /// Load from a VarBuilder and UniversalConfig.
    pub fn from_varbuilder(vb: &mut VarBuilder<R>, config: &UniversalConfig) -> Result<Self> {
        let mamba_config = Mamba2Config::from_universal(config)?;
        mamba_config.validate()?;

        let mut model_vb = vb.pp("model");

        // Embedding
        let embed_weight = model_vb.take_tensor("embed_tokens.weight")?;
        let embed_tokens = Embedding::new(embed_weight, false);

        // Mamba2 layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let mut layers_vb = model_vb.pp("layers");
            let mut layer_vb = layers_vb.pp(&i.to_string());

            // Pre-norm
            let norm = RmsNorm::new(
                layer_vb.take_tensor("input_layernorm.weight")?,
                config.rms_norm_eps as f32,
                false,
            );

            // Mamba2 layer
            let mut mixer_vb = layer_vb.pp("mixer");
            let mamba = Mamba2::from_varbuilder(&mamba_config, &mut mixer_vb, false)?;

            layers.push(Mamba2Block { norm, mamba });
        }

        // Final norm
        let norm = RmsNorm::new(
            model_vb.take_tensor("norm.weight")?,
            config.rms_norm_eps as f32,
            false,
        );

        // LM head (may be tied with embedding)
        let lm_head = if config.tie_word_embeddings {
            // Reuse embedding weight
            let embed_w = embed_tokens.weight().tensor().clone();
            Linear::new(embed_w, None, false)
        } else {
            Linear::new(vb.take_tensor("lm_head.weight")?, None, false)
        };

        Ok(Self {
            config: config.clone(),
            mamba_config,
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    /// Inference forward pass with SSM state.
    ///
    /// `input_ids: [batch, seq_len]` → logits `[batch, seq_len, vocab_size]`
    pub fn forward_with_ssm_state<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        ssm_state: &mut LayeredSsmState<R>,
    ) -> Result<Tensor<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + IndexingOps<R>,
    {
        // Embed tokens: [B, S] -> Var<R> [B, S, hidden]
        let mut hidden = self.embed_tokens.forward(client, input_ids)?;

        // Mamba2 blocks with SSM state
        for (i, layer) in self.layers.iter().enumerate() {
            let state = ssm_state.layer_mut(i).ok_or_else(|| Error::ModelError {
                reason: format!("SSM state missing for layer {i}"),
            })?;
            hidden = layer.forward_inference(client, &hidden, state)?;
        }

        // Final norm
        hidden = self.norm.forward(client, &hidden)?;

        // LM head: [B, S, hidden] -> [B, S, vocab]
        let logits = self.lm_head.forward(client, &hidden)?;

        Ok(logits.tensor().clone())
    }

    /// Get the universal config.
    pub fn config(&self) -> &UniversalConfig {
        &self.config
    }

    /// Get the Mamba2 layer config.
    pub fn mamba_config(&self) -> &Mamba2Config {
        &self.mamba_config
    }
}

impl<R: Runtime<DType = DType>> Mamba2Block<R> {
    /// Inference forward: pre-norm → Mamba2 → residual
    fn forward_inference<C>(
        &self,
        client: &C,
        x: &Var<R>,
        state: &mut SsmState<R>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + IndexingOps<R>,
    {
        // Pre-norm
        let normed = self.norm.forward(client, x)?;

        // Mamba2 layer (inference path on raw tensors)
        let out_tensor = self
            .mamba
            .forward_inference(client, normed.tensor(), state)?;

        // Residual connection
        let out = Var::new(out_tensor, false);
        numr::autograd::var_add(x, &out, client).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mamba2_model_config() {
        // Just verify Mamba2Config::from_universal works
        let config = UniversalConfig {
            model_type: "mamba2".into(),
            vocab_size: 1000,
            hidden_size: 64,
            num_layers: 2,
            max_seq_len: 512,
            intermediate_size: None,
            rms_norm_eps: 1e-5,
            attention: None,
            ssm: Some(crate::model::config::SsmConfig {
                variant: "mamba2".into(),
                state_size: 16,
                num_heads: 2,
                head_dim: 64,
                expand: 2,
                conv_kernel: 4,
                chunk_size: 64,
                n_groups: 1,
                complex_rope: None,
                mimo_rank: None,
                use_conv: None,
            }),
            moe: None,
            hybrid_layers: None,
            tie_word_embeddings: false,
        };
        let mamba_config = Mamba2Config::from_universal(&config).unwrap();
        assert_eq!(mamba_config.d_model, 64);
        assert_eq!(mamba_config.nheads, 2);
        assert_eq!(mamba_config.d_state, 16);
    }
}
