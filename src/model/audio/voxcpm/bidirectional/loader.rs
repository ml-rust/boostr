//! Shared per-layer loader for VoxCPM2's bidirectional transformer block,
//! reused by `local_dit` (`feat_decoder.estimator.decoder.layers.{i}`) and
//! `local_encoder` (`feat_encoder.encoder.layers.{i}`) — same key pattern,
//! same shapes, different prefixes and config sources:
//!
//! ```text
//! {layer_prefix}.input_layernorm.weight          [hidden_dim]
//! {layer_prefix}.self_attn.q_proj.weight         [num_heads*head_dim, hidden_dim]
//! {layer_prefix}.self_attn.k_proj.weight         [num_kv_heads*head_dim, hidden_dim]
//! {layer_prefix}.self_attn.v_proj.weight         [num_kv_heads*head_dim, hidden_dim]
//! {layer_prefix}.self_attn.o_proj.weight         [hidden_dim, num_heads*head_dim]
//! {layer_prefix}.post_attention_layernorm.weight [hidden_dim]
//! {layer_prefix}.mlp.gate_proj.weight            [ffn_dim, hidden_dim]
//! {layer_prefix}.mlp.up_proj.weight              [ffn_dim, hidden_dim]
//! {layer_prefix}.mlp.down_proj.weight            [hidden_dim, ffn_dim]
//! ```
//! Every projection and norm is bias-free. Knows nothing about
//! `LocalDitConfig`/`LocalEncoderConfig` — callers pass dimensions in.

use crate::error::Result;
use crate::model::audio::voxcpm::bidirectional::attention::BidirectionalAttention;
use crate::model::audio::voxcpm::bidirectional::layer::BidirectionalLayer;
use crate::model::audio::voxcpm::bidirectional::mlp::BidirectionalMlp;
use crate::model::audio::voxcpm::loader::support::{TensorLoader, WeightSource};
use crate::nn::RmsNorm;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;

/// Dimensions the shared bidirectional-layer loader needs, read out of
/// whichever config type the caller owns (`LocalDitConfig` or
/// `LocalEncoderConfig`) — this module never references either.
pub(crate) struct BidirectionalLayerDims {
    pub(crate) hidden_dim: usize,
    pub(crate) ffn_dim: usize,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) rms_norm_eps: f32,
}

/// Load one bidirectional transformer layer at `layer_prefix` (e.g.
/// `decoder.layers.3` or `encoder.layers.3`).
pub(crate) fn load_bidirectional_layer<R: Runtime<DType = DType>, S: WeightSource<R>>(
    tl: &mut TensorLoader<'_, R, S>,
    layer_prefix: &str,
    dims: &BidirectionalLayerDims,
) -> Result<BidirectionalLayer<R>>
where
    R::Client: TypeConversionOps<R>,
{
    let q_dim = dims.num_heads * dims.head_dim;
    let kv_dim = dims.num_kv_heads * dims.head_dim;

    // DENSE, deliberately: an RmsNorm weight is an element-wise scale, not a
    // matmul weight — there is no packed kernel to run it against, and a
    // GGUF stores it unquantized anyway. `tl.tensor` (the dequantizing read)
    // is the only correct call here, for this norm and the one below.
    let input_layernorm = RmsNorm::new(
        tl.tensor(
            &format!("{layer_prefix}.input_layernorm.weight"),
            &[dims.hidden_dim],
        )?,
        dims.rms_norm_eps,
        false,
    );

    // `TensorLoader::linear` keeps a block-quantized weight PACKED, so on a
    // GGUF these four (and the three below) hold K-quant blocks and multiply
    // through `quant_matmul`. On safetensors they are the same dense
    // `Linear` they always were. Every projection is bias-free — hence
    // `with_bias: false` throughout.
    let self_attn = {
        let attn_prefix = format!("{layer_prefix}.self_attn");
        let q_proj = tl.linear(
            &format!("{attn_prefix}.q_proj"),
            q_dim,
            dims.hidden_dim,
            false,
        )?;
        let k_proj = tl.linear(
            &format!("{attn_prefix}.k_proj"),
            kv_dim,
            dims.hidden_dim,
            false,
        )?;
        let v_proj = tl.linear(
            &format!("{attn_prefix}.v_proj"),
            kv_dim,
            dims.hidden_dim,
            false,
        )?;
        let o_proj = tl.linear(
            &format!("{attn_prefix}.o_proj"),
            dims.hidden_dim,
            q_dim,
            false,
        )?;
        BidirectionalAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: dims.num_heads,
            num_kv_heads: dims.num_kv_heads,
            head_dim: dims.head_dim,
        }
    };

    let post_attention_layernorm = RmsNorm::new(
        tl.tensor(
            &format!("{layer_prefix}.post_attention_layernorm.weight"),
            &[dims.hidden_dim],
        )?,
        dims.rms_norm_eps,
        false,
    );

    let mlp = {
        let mlp_prefix = format!("{layer_prefix}.mlp");
        let gate_proj = tl.linear(
            &format!("{mlp_prefix}.gate_proj"),
            dims.ffn_dim,
            dims.hidden_dim,
            false,
        )?;
        let up_proj = tl.linear(
            &format!("{mlp_prefix}.up_proj"),
            dims.ffn_dim,
            dims.hidden_dim,
            false,
        )?;
        let down_proj = tl.linear(
            &format!("{mlp_prefix}.down_proj"),
            dims.hidden_dim,
            dims.ffn_dim,
            false,
        )?;
        BidirectionalMlp {
            gate_proj,
            up_proj,
            down_proj,
        }
    };

    Ok(BidirectionalLayer {
        input_layernorm,
        self_attn,
        post_attention_layernorm,
        mlp,
    })
}
