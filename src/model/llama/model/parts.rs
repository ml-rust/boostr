//! The pieces a Llama is built from and runs through: the RoPE tables, the
//! embedding, one layer, the final norm, and the layer loop. The full model
//! and a pipeline stage share them, so both read a checkpoint alike.

use crate::error::{Error, Result};
use crate::inference::LayeredKvCache;
use crate::model::config::ModelConfig;
use crate::model::llama::model::blocks::{BlockNorm, LlamaBlock, build_block_from_varbuilder};
use crate::model::traits::ModelClient;
use crate::model::vocab_growth::fit_vocab_rows;
use crate::nn::{Embedding, RoPE, VarBuilder};
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;

/// Head counts and head size of a Llama's attention.
#[derive(Debug, Clone, Copy)]
pub(super) struct Dims {
    pub(super) num_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
}

impl Dims {
    /// # Errors
    ///
    /// A config with no attention section, or an MLA config
    /// ([`mla_refusal`](crate::model::llama::mla_refusal)).
    pub(super) fn of(config: &ModelConfig) -> Result<Self> {
        if let Some(reason) = crate::model::llama::mla_refusal(config) {
            return Err(Error::ModelError { reason });
        }
        let attn = config.attention.as_ref().ok_or_else(|| Error::ModelError {
            reason: "LLaMA requires attention config".into(),
        })?;
        Ok(Self {
            num_heads: attn.num_heads,
            num_kv_heads: attn.kv_heads(),
            head_dim: attn.head_dim(config.hidden_size),
        })
    }
}

/// The RoPE tables of `config`. A partial rotary head rotates its leading
/// `rotary_dim` dims only.
pub(super) fn rope_of<R: Runtime<DType = DType>>(
    config: &ModelConfig,
    dims: Dims,
    device: &R::Device,
) -> Result<RoPE<R>> {
    let attn = config.attention.as_ref().ok_or_else(|| Error::ModelError {
        reason: "LLaMA requires attention config".into(),
    })?;
    RoPE::<R>::precompute_freqs_partial(
        config.max_seq_len,
        dims.head_dim,
        config.block.rotary_dim(dims.head_dim),
        attn.rope_theta,
        attn.rope_scaling.as_ref(),
        device,
    )
}

/// `rope` with its tables cast to the dtype of the first layer's weights,
/// so no forward casts them per token.
pub(super) fn cast_rope<R: Runtime<DType = DType>>(
    mut rope: RoPE<R>,
    layers: &[LlamaBlock<R>],
) -> Result<RoPE<R>>
where
    R::Client: numr::ops::TypeConversionOps<R>,
{
    if let Some(first) = layers.first() {
        rope.cast_caches(first.input_layernorm.weight().tensor().dtype())?;
    }
    Ok(rope)
}

/// The token embedding under `model_vb`, grown to `config.vocab_size`
/// rows, and the checkpoint key it was read from.
///
/// A tied head multiplies by this tensor, and an importance collection
/// records it under the returned key. The key comes from the builder that
/// reads it, so it cannot drift from the loader.
pub(super) fn take_embedding<R: Runtime<DType = DType>>(
    model_vb: &mut VarBuilder<R>,
    config: &ModelConfig,
) -> Result<(Embedding<R>, String)>
where
    R::Client: DequantOps<R> + numr::ops::TypeConversionOps<R> + ReduceOps<R> + ShapeOps<R>,
{
    let name = model_vb.full_name("embed_tokens.weight");
    let weight = model_vb.take_tensor_dequant("embed_tokens.weight", DType::F32)?;
    let weight = fit_vocab_rows(weight, config, model_vb.device(), "embed_tokens.weight")?;
    Ok((Embedding::new(weight, false), name))
}

/// Layer `layer` under `model_vb`.
pub(super) fn take_layer<R: Runtime<DType = DType>>(
    model_vb: &mut VarBuilder<R>,
    config: &ModelConfig,
    layer: usize,
    dims: Dims,
) -> Result<LlamaBlock<R>> {
    let mut layers_vb = model_vb.pp("layers");
    let mut layer_vb = layers_vb.pp(&layer.to_string());
    build_block_from_varbuilder(
        &mut layer_vb,
        config,
        layer,
        dims.num_heads,
        dims.num_kv_heads,
        dims.head_dim,
    )
}

/// The final norm under `model_vb`.
pub(super) fn take_final_norm<R: Runtime<DType = DType>>(
    model_vb: &mut VarBuilder<R>,
    config: &ModelConfig,
) -> Result<BlockNorm<R>> {
    let weight = model_vb.take_tensor("norm.weight")?;
    let bias = model_vb.take_tensor_optional("norm.bias")?;
    BlockNorm::build(
        weight,
        bias,
        config.rms_norm_eps as f32,
        &config.block,
        false,
    )
}

/// Runs `layers`, the model's layers `first..first + layers.len()`, with
/// KV cache. Cache slot `0` holds layer `first`. Returns the hidden state
/// and the deferred MLP residual of the last layer run.
///
/// # Errors
///
/// A cache whose layer count differs from `layers.len()`, or a layer that
/// fails.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_layers<R, C>(
    client: &C,
    layers: &[LlamaBlock<R>],
    first: usize,
    rope: &RoPE<R>,
    hidden: Var<R>,
    prev_mlp_out: Option<Var<R>>,
    kv_cache: &mut LayeredKvCache<R>,
    position: usize,
) -> Result<(Var<R>, Option<Var<R>>)>
where
    R: Runtime<DType = DType>,
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
    let end = first + layers.len();
    if kv_cache.num_layers() != layers.len() {
        return Err(Error::ModelError {
            reason: format!(
                "layer range {first}..{end} needs a KV cache of {} layers, and this one \
                 holds {}",
                layers.len(),
                kv_cache.num_layers()
            ),
        });
    }
    let mut hidden = hidden;
    let mut prev_mlp_out = prev_mlp_out;
    for (slot, layer) in layers.iter().enumerate() {
        let cache = kv_cache.layer_mut(slot).ok_or_else(|| Error::ModelError {
            reason: format!("KV cache missing for layer {}", first + slot),
        })?;
        let (h, mlp_out) = layer.forward_with_kv_cache(
            client,
            &hidden,
            prev_mlp_out.as_ref(),
            rope,
            cache,
            position,
        )?;
        hidden = h;
        prev_mlp_out = Some(mlp_out);
    }
    Ok((hidden, prev_mlp_out))
}
