//! Full Mamba3 model — embedding + N layers + norm + LM head.

use crate::error::{Error, Result};
use crate::model::config::UniversalConfig;
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

    /// The dtype every forward pass computes and returns logits in.
    ///
    /// Reads the final norm weight's dtype: an owned, never-quantized tensor
    /// that matches the compute dtype the checkpoint loaded at.
    pub fn dtype(&self) -> DType {
        self.norm.weight().tensor().dtype()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::{SsmConfig, UniversalConfig};
    use crate::nn::VarMap;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    fn tiny_config() -> UniversalConfig {
        UniversalConfig {
            model_type: "mamba3".into(),
            vocab_size: 12,
            hidden_size: 4,
            num_layers: 1,
            max_seq_len: 16,
            intermediate_size: None,
            rms_norm_eps: 1e-5,
            attention: None,
            ssm: Some(SsmConfig {
                variant: "mamba3".into(),
                num_heads: 1,
                head_dim: 8,
                state_size: 4,
                chunk_size: 4,
                n_groups: 1,
                conv_kernel: 4,
                expand: 2,
                complex_rope: Some(false),
                mimo_rank: Some(0),
                use_conv: Some(false),
            }),
            moe: None,
            hybrid_layers: None,
            tie_word_embeddings: true,
            grow_vocab: false,
            vision: None,
            audio: None,
            gdn: None,
            qwen35_attention: None,
            hadamard: None,
            quant_formats: Vec::new(),
        }
    }

    /// Weight names and shapes matching [`tiny_config`]: one Mamba3 layer,
    /// complex RoPE / conv / MIMO all off, tied embedding so no separate
    /// `lm_head.weight` is needed.
    fn insert_weights(
        varmap: &mut VarMap<CpuRuntime>,
        config: &UniversalConfig,
        device: &numr::runtime::cpu::CpuDevice,
    ) {
        let mamba_config = Mamba3Config::from_universal(config).unwrap();
        let hidden = config.hidden_size;
        let d_inner = mamba_config.d_inner();
        let nheads = mamba_config.nheads;
        let d_state = mamba_config.d_state;
        let shapes: Vec<(String, Vec<usize>)> = vec![
            (
                "model.embed_tokens.weight".into(),
                vec![config.vocab_size, hidden],
            ),
            ("model.layers.0.norm.weight".into(), vec![hidden]),
            (
                "model.layers.0.mixer.in_proj.weight".into(),
                vec![mamba_config.proj_dim(), hidden],
            ),
            (
                "model.layers.0.mixer.out_proj.weight".into(),
                vec![hidden, d_inner],
            ),
            (
                "model.layers.0.mixer.lambda_proj.weight".into(),
                vec![nheads, hidden],
            ),
            ("model.layers.0.mixer.b_bias".into(), vec![nheads, d_state]),
            ("model.layers.0.mixer.c_bias".into(), vec![nheads, d_state]),
            ("model.layers.0.mixer.dt_bias".into(), vec![nheads]),
            ("model.layers.0.mixer.A_log".into(), vec![nheads]),
            ("model.layers.0.mixer.D".into(), vec![nheads]),
            ("model.layers.0.mixer.bc_norm.weight".into(), vec![d_state]),
            ("model.layers.0.mixer.norm.weight".into(), vec![d_inner]),
            ("model.norm.weight".into(), vec![hidden]),
        ];
        for (name, shape) in shapes {
            varmap.insert(
                name,
                Tensor::<CpuRuntime>::zeros(&shape, DType::F32, device).unwrap(),
            );
        }
    }

    #[test]
    fn mamba3_dtype_matches_the_varmap_dtype_it_is_built_with() {
        let (_client, device) = cpu_setup();
        let config = tiny_config();
        let mut varmap = VarMap::<CpuRuntime>::new();
        insert_weights(&mut varmap, &config, &device);
        let mut vb = VarBuilder::new(&mut varmap, &device);
        let model = Mamba3Model::<CpuRuntime>::from_varbuilder(&mut vb, &config).unwrap();
        assert_eq!(model.dtype(), DType::F32);
    }
}
