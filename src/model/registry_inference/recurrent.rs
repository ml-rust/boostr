//! Dispatchers for the forwards that carry recurrent state:
//! [`LoadedModel::forward_with_ssm_state`], [`LoadedModel::forward_hybrid`],
//! [`LoadedModel::forward_qwen35`] and [`LoadedModel::forward_qwen35_embeds`].

use crate::error::{Error, Result};
use crate::inference::{LayeredGdnState, LayeredKvCache, LayeredSsmState};
use crate::model::registry::LoadedModel;
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, IndexingOps, NormalizationOps, ScalarOps, ShapeOps, TensorOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> LoadedModel<R>
where
    R::Client: IndexingOps<R>,
{
    /// Forward pass with SSM state for Mamba2 inference
    pub fn forward_with_ssm_state(
        &self,
        input_ids: &Tensor<R>,
        ssm_state: &mut LayeredSsmState<R>,
    ) -> Result<Tensor<R>>
    where
        R::Client: ModelClient<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
    {
        let device = input_ids.device();
        let client = R::default_client(device);
        match self {
            LoadedModel::Mamba1(_) => Err(Error::ModelError {
                reason: "Mamba1 needs a Mamba1-specific recurrent cache (per-channel/per-state A and depthwise-conv state); LayeredSsmState is Mamba2-shaped".into(),
            }),
            LoadedModel::Mamba2(m) => m.forward_with_ssm_state(&client, input_ids, ssm_state),
            LoadedModel::Mamba3(_) => Err(Error::ModelError {
                reason: "Mamba3 needs a Mamba3-specific recurrent cache (trapezoidal prev x/B plus optional MIMO state); LayeredSsmState is Mamba2-shaped".into(),
            }),
            LoadedModel::Llama(_) | LoadedModel::LlamaTp(_) => Err(Error::ModelError {
                reason: "Llama does not use SSM state — use forward_with_kv_cache() instead".into(),
            }),
            LoadedModel::Hybrid(_) => Err(Error::ModelError {
                reason: "Hybrid model does not support forward_with_ssm_state — use forward_hybrid() instead"
                    .into(),
            }),
            LoadedModel::Qwen35(_) => Err(Error::ModelError {
                reason: "qwen35 model does not support forward_with_ssm_state — use forward_qwen35() instead"
                    .into(),
            }),
            LoadedModel::Multimodal(m) => m.llm().forward_with_ssm_state(input_ids, ssm_state),
        }
    }

    /// Forward pass for hybrid model with both KV cache and SSM state
    pub fn forward_hybrid(
        &self,
        input_ids: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        ssm_state: &mut LayeredSsmState<R>,
        position: usize,
    ) -> Result<Tensor<R>>
    where
        R::Client: ModelClient<R>
            + NormalizationOps<R>
            + UnaryOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + ShapeOps<R>
            + TensorOps<R>
            + ScalarOps<R>,
    {
        let device = input_ids.device();
        let client = R::default_client(device);
        match self {
            LoadedModel::Hybrid(m) => {
                m.forward_hybrid(&client, input_ids, kv_cache, ssm_state, position)
            }
            LoadedModel::Llama(_) | LoadedModel::LlamaTp(_) => Err(Error::ModelError {
                reason: "Llama model does not support forward_hybrid — use forward_with_kv_cache()"
                    .into(),
            }),
            LoadedModel::Mamba1(_) | LoadedModel::Mamba2(_) | LoadedModel::Mamba3(_) => {
                Err(Error::ModelError {
                    reason: "SSM-only model does not support forward_hybrid".into(),
                })
            }
            LoadedModel::Qwen35(_) => Err(Error::ModelError {
                reason:
                    "qwen35 model does not support forward_hybrid — use forward_qwen35() instead"
                        .into(),
            }),
            LoadedModel::Multimodal(m) => m
                .llm()
                .forward_hybrid(input_ids, kv_cache, ssm_state, position),
        }
    }

    /// Forward pass for `qwen35` with KV cache (attention layers) and GDN
    /// state (linear-attention layers). `rope_pos` is the IMROPE position
    /// of the first token; the KV slot comes from `kv_cache.seq_len()`.
    /// Text-only callers pass `kv_cache.seq_len()`.
    pub fn forward_qwen35(
        &self,
        input_ids: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        gdn_state: &mut LayeredGdnState<R>,
        rope_pos: usize,
    ) -> Result<Tensor<R>>
    where
        R::Client: ModelClient<R> + BinaryOps<R> + ShapeOps<R> + TensorOps<R> + ScalarOps<R>,
    {
        let device = input_ids.device();
        let client = R::default_client(device);
        match self {
            LoadedModel::Qwen35(m) => {
                m.forward_qwen35(&client, input_ids, kv_cache, gdn_state, rope_pos)
            }
            LoadedModel::Llama(_) | LoadedModel::LlamaTp(_) => Err(Error::ModelError {
                reason: "Llama model does not support forward_qwen35 — use forward_with_kv_cache()"
                    .into(),
            }),
            LoadedModel::Mamba1(_) | LoadedModel::Mamba2(_) | LoadedModel::Mamba3(_) => {
                Err(Error::ModelError {
                    reason: "SSM-only model does not support forward_qwen35".into(),
                })
            }
            LoadedModel::Hybrid(_) => Err(Error::ModelError {
                reason: "Hybrid model does not support forward_qwen35 — use forward_hybrid()"
                    .into(),
            }),
            LoadedModel::Multimodal(m) => m
                .llm()
                .forward_qwen35(input_ids, kv_cache, gdn_state, rope_pos),
        }
    }

    /// Input embeddings `[batch, seq, hidden]` of `input_ids` for a `qwen35`
    /// model, rotation applied. See `Qwen35Model::embed_tokens`.
    pub fn embed_tokens_qwen35(&self, input_ids: &Tensor<R>) -> Result<Tensor<R>>
    where
        R::Client: ModelClient<R>,
    {
        let device = input_ids.device();
        let client = R::default_client(device);
        match self {
            LoadedModel::Qwen35(m) => m.embed_tokens(&client, input_ids),
            LoadedModel::Multimodal(m) => m.llm().embed_tokens_qwen35(input_ids),
            _ => Err(Error::ModelError {
                reason: format!("{} does not support embed_tokens_qwen35", self.model_type()),
            }),
        }
    }

    /// Forward pass for `qwen35` over input embeddings `[batch, seq, hidden]`
    /// with explicit `[4, seq]` i32 IMROPE positions. See
    /// `Qwen35Model::forward_qwen35_embeds`.
    pub fn forward_qwen35_embeds(
        &self,
        embeds: &Tensor<R>,
        positions: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        gdn_state: &mut LayeredGdnState<R>,
    ) -> Result<Tensor<R>>
    where
        R::Client: ModelClient<R> + BinaryOps<R> + ShapeOps<R> + TensorOps<R> + ScalarOps<R>,
    {
        let device = embeds.device();
        let client = R::default_client(device);
        match self {
            LoadedModel::Qwen35(m) => {
                m.forward_qwen35_embeds(&client, embeds, positions, kv_cache, gdn_state)
            }
            LoadedModel::Multimodal(m) => m
                .llm()
                .forward_qwen35_embeds(embeds, positions, kv_cache, gdn_state),
            _ => Err(Error::ModelError {
                reason: format!(
                    "{} does not support forward_qwen35_embeds",
                    self.model_type()
                ),
            }),
        }
    }
}
