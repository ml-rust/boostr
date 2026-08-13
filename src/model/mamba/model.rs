//! Full Mamba2 model — embedding + N layers + norm + LM head
//!
//! Mirrors the Llama model structure but uses Mamba2 SSM layers
//! instead of transformer attention blocks.

use crate::error::{Error, Result};
use crate::inference::{LayeredSsmState, SsmState};
use crate::model::config::UniversalConfig;
use crate::model::mamba::mamba1::{Mamba1, Mamba1Config};
use crate::model::mamba::mamba2::{Mamba2, Mamba2Config};
use crate::model::mamba::mamba3::{Mamba3, Mamba3Config};
use crate::model::traits::ModelClient;
use crate::nn::{Embedding, Linear, RmsNorm, VarBuilder};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConvOps, IndexingOps, NormalizationOps, ReduceOps,
    ScalarOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub struct Mamba1Model<R: Runtime> {
    config: UniversalConfig,
    mamba_config: Mamba1Config,
    embed_tokens: Embedding<R>,
    layers: Vec<Mamba1Block<R>>,
    norm: RmsNorm<R>,
    lm_head: Linear<R>,
}

struct Mamba1Block<R: Runtime> {
    norm: RmsNorm<R>,
    mamba: Mamba1<R>,
}

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

pub struct Mamba3Model<R: Runtime> {
    config: UniversalConfig,
    mamba_config: Mamba3Config,
    embed_tokens: Embedding<R>,
    layers: Vec<Mamba3Block<R>>,
    norm: RmsNorm<R>,
    lm_head: Linear<R>,
}

/// Single Mamba3 block: pre-norm → Mamba3 → residual.
struct Mamba3Block<R: Runtime> {
    norm: RmsNorm<R>,
    mamba: Mamba3<R>,
}

impl<R: Runtime<DType = DType>> Mamba1Model<R>
where
    R::Client: IndexingOps<R>,
{
    pub fn from_varbuilder(vb: &mut VarBuilder<R>, config: &UniversalConfig) -> Result<Self> {
        let mamba_config = Mamba1Config::from_universal(config)?;
        mamba_config.validate()?;
        let mut model_vb = vb.pp("model");
        let embed_tokens = Embedding::new(model_vb.take_tensor("embed_tokens.weight")?, false);
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let mut layers_vb = model_vb.pp("layers");
            let mut layer_vb = layers_vb.pp(&i.to_string());
            let norm_name = if layer_vb.contains("input_layernorm.weight") {
                "input_layernorm.weight"
            } else {
                "norm.weight"
            };
            let norm = RmsNorm::new(
                layer_vb.take_tensor(norm_name)?,
                config.rms_norm_eps as f32,
                false,
            );
            let mamba = if layer_vb.contains("mamba1.mixer.in_proj.weight") {
                let mut mamba1_vb = layer_vb.pp("mamba1");
                let mut mixer_vb = mamba1_vb.pp("mixer");
                Mamba1::from_varbuilder(&mamba_config, &mut mixer_vb, false)?
            } else {
                let mut mixer_vb = layer_vb.pp("mixer");
                Mamba1::from_varbuilder(&mamba_config, &mut mixer_vb, false)?
            };
            layers.push(Mamba1Block { norm, mamba });
        }
        let norm = RmsNorm::new(
            model_vb.take_tensor("norm.weight")?,
            config.rms_norm_eps as f32,
            false,
        );
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_tokens.weight().tensor().clone(), None, false)
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

    pub fn forward_full<C>(&self, client: &C, input_ids: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + IndexingOps<R>
            + CompareOps<R>,
    {
        let hidden = self.forward_hidden(client, input_ids)?;
        let logits = self.lm_head.forward(client, &hidden)?;
        Ok(logits.tensor().clone())
    }

    pub fn forward_hidden<C>(&self, client: &C, input_ids: &Tensor<R>) -> Result<Var<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + IndexingOps<R>
            + CompareOps<R>,
    {
        let mut hidden = self.embed_tokens.forward(client, input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(client, &hidden)?;
        }
        self.norm.forward(client, &hidden)
    }

    pub fn config(&self) -> &UniversalConfig {
        &self.config
    }

    pub fn mamba_config(&self) -> &Mamba1Config {
        &self.mamba_config
    }
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

    /// Contextualized hidden states for embedding extraction.
    ///
    /// Runs embed + all Mamba2 layers + final norm (no `lm_head`) with a fresh,
    /// throwaway SSM state. Returns shape `[batch, seq_len, hidden]`.
    pub fn forward_hidden<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
    ) -> Result<numr::autograd::Var<R>>
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
        let batch = input_ids.shape()[0];
        let device = input_ids.device();
        let dtype = self.embed_tokens.weight().tensor().dtype();
        let mut ssm_state =
            LayeredSsmState::<R>::new(self.layers.len(), batch, &self.mamba_config, dtype, device);

        let mut hidden = self.embed_tokens.forward(client, input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let state = ssm_state.layer_mut(i).ok_or_else(|| Error::ModelError {
                reason: format!("SSM state missing for layer {i}"),
            })?;
            hidden = layer.forward_inference(client, &hidden, state)?;
        }
        hidden = self.norm.forward(client, &hidden)?;
        Ok(hidden)
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

impl<R: Runtime<DType = DType>> Mamba3Model<R>
where
    R::Client: IndexingOps<R>,
{
    /// Load from a VarBuilder and UniversalConfig.
    pub fn from_varbuilder(vb: &mut VarBuilder<R>, config: &UniversalConfig) -> Result<Self> {
        let mamba_config = Mamba3Config::from_universal(config)?;
        mamba_config.validate()?;

        let mut model_vb = vb.pp("model");

        // Embedding
        let embed_weight = model_vb.take_tensor("embed_tokens.weight")?;
        let embed_tokens = Embedding::new(embed_weight, false);

        // Mamba3 layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let mut layers_vb = model_vb.pp("layers");
            let mut layer_vb = layers_vb.pp(&i.to_string());

            let norm_name = if layer_vb.contains("input_layernorm.weight") {
                "input_layernorm.weight"
            } else {
                "norm.weight"
            };
            let norm = RmsNorm::new(
                layer_vb.take_tensor(norm_name)?,
                config.rms_norm_eps as f32,
                false,
            );

            let mamba = if layer_vb.contains("mamba3.mixer.in_proj.weight") {
                let mut mamba3_vb = layer_vb.pp("mamba3");
                let mut mixer_vb = mamba3_vb.pp("mixer");
                Mamba3::from_varbuilder(&mamba_config, &mut mixer_vb, false)?
            } else {
                let mut mixer_vb = layer_vb.pp("mixer");
                Mamba3::from_varbuilder(&mamba_config, &mut mixer_vb, false)?
            };

            layers.push(Mamba3Block { norm, mamba });
        }

        // Final norm
        let norm = RmsNorm::new(
            model_vb.take_tensor("norm.weight")?,
            config.rms_norm_eps as f32,
            false,
        );

        // LM head (may be tied with embedding)
        let lm_head = if config.tie_word_embeddings {
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

    /// Full-sequence forward pass without a persistent recurrent cache.
    pub fn forward_full<C>(&self, client: &C, input_ids: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + IndexingOps<R>
            + CompareOps<R>,
    {
        let hidden = self.forward_hidden(client, input_ids)?;
        let logits = self.lm_head.forward(client, &hidden)?;
        Ok(logits.tensor().clone())
    }

    /// Contextualized hidden states for embedding extraction.
    pub fn forward_hidden<C>(&self, client: &C, input_ids: &Tensor<R>) -> Result<Var<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + IndexingOps<R>
            + CompareOps<R>,
    {
        let mut hidden = self.embed_tokens.forward(client, input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(client, &hidden)?;
        }
        self.norm.forward(client, &hidden)
    }

    /// Get the universal config.
    pub fn config(&self) -> &UniversalConfig {
        &self.config
    }

    /// Get the Mamba3 layer config.
    pub fn mamba_config(&self) -> &Mamba3Config {
        &self.mamba_config
    }
}

impl<R: Runtime<DType = DType>> Mamba1Block<R> {
    fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + IndexingOps<R>
            + CompareOps<R>,
    {
        let normed = self.norm.forward(client, x)?;
        let out = self.mamba.forward(client, &normed)?;
        numr::autograd::var_add(x, &out, client).map_err(Error::Numr)
    }
}

impl<R: Runtime<DType = DType>> Mamba3Block<R> {
    /// Full-sequence forward: pre-norm → Mamba3 → residual.
    fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + IndexingOps<R>
            + CompareOps<R>,
    {
        let normed = self.norm.forward(client, x)?;
        let out = self.mamba.forward(client, &normed)?;
        numr::autograd::var_add(x, &out, client).map_err(Error::Numr)
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
