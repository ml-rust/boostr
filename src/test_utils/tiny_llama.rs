//! A tiny Llama with random weights, loaded through the normal loader path.

use crate::format::restructure_hf_checkpoint;
#[cfg(feature = "distributed")]
use crate::model::LoadedModel;
use crate::model::config::{HuggingFaceConfig, UniversalConfig};
#[cfg(feature = "distributed")]
use crate::nn::VarBuilder;
use crate::nn::VarMap;
use numr::dtype::DType;
use numr::ops::{RandomOps, ScalarOps};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Vocabulary size of [`random_llama`].
pub(crate) const TINY_VOCAB: usize = 16;

/// Hidden size of [`random_llama`].
pub(crate) const TINY_HIDDEN: usize = 16;

/// Positions [`random_llama`] supports.
pub(crate) const TINY_MAX_SEQ_LEN: usize = 32;

fn random(client: &CpuClient, shape: &[usize], scale: f64, offset: f64) -> Tensor<CpuRuntime> {
    let t = client.randn(shape, DType::F32).unwrap();
    let t = client.mul_scalar(&t, scale).unwrap();
    client.add_scalar(&t, offset).unwrap()
}

/// A GQA Llama of `num_layers` layers with random weights, untied head.
#[cfg(feature = "distributed")]
pub(crate) fn random_llama(
    client: &CpuClient,
    device: &CpuDevice,
    num_layers: usize,
) -> LoadedModel<CpuRuntime> {
    let (mut map, config) = random_llama_checkpoint(client, device, num_layers, false);
    let mut vb = VarBuilder::new(&mut map, device);
    LoadedModel::load(&config, &mut vb).unwrap()
}

/// The restructured tensors and config of a GQA Llama of `num_layers`
/// layers with random weights. A `tied` checkpoint holds no `lm_head`.
pub(crate) fn random_llama_checkpoint(
    client: &CpuClient,
    device: &CpuDevice,
    num_layers: usize,
    tied: bool,
) -> (VarMap<CpuRuntime>, UniversalConfig) {
    let json = format!(
        r#"{{"model_type": "llama", "vocab_size": {TINY_VOCAB}, "hidden_size": {TINY_HIDDEN},
            "num_hidden_layers": {num_layers}, "max_position_embeddings": {TINY_MAX_SEQ_LEN},
            "num_attention_heads": 2, "num_key_value_heads": 1, "intermediate_size": 32,
            "tie_word_embeddings": {tied}}}"#
    );
    let hf: HuggingFaceConfig = serde_json::from_str(&json).unwrap();
    let config = hf.to_universal().unwrap();
    let attn = config.attention.as_ref().unwrap();
    let (h, d) = (config.hidden_size, attn.head_dim(config.hidden_size));
    let (q, kv) = (attn.num_heads * d, attn.kv_heads() * d);
    let inter = config.intermediate_size();
    let mut map = VarMap::<CpuRuntime>::new();
    let mut put = |name: String, shape: &[usize], scale: f64, offset: f64| {
        map.insert(name, random(client, shape, scale, offset));
    };
    put(
        "model.embed_tokens.weight".into(),
        &[TINY_VOCAB, h],
        0.5,
        0.0,
    );
    for l in 0..num_layers {
        let p = format!("model.layers.{l}");
        for (name, rows, cols) in [("q", q, h), ("k", kv, h), ("v", kv, h), ("o", h, q)] {
            put(
                format!("{p}.self_attn.{name}_proj.weight"),
                &[rows, cols],
                0.3,
                0.0,
            );
        }
        put(format!("{p}.input_layernorm.weight"), &[h], 0.1, 1.0);
        put(
            format!("{p}.post_attention_layernorm.weight"),
            &[h],
            0.1,
            1.0,
        );
        put(format!("{p}.mlp.gate_proj.weight"), &[inter, h], 0.3, 0.0);
        put(format!("{p}.mlp.up_proj.weight"), &[inter, h], 0.3, 0.0);
        put(format!("{p}.mlp.down_proj.weight"), &[h, inter], 0.3, 0.0);
    }
    put("model.norm.weight".into(), &[h], 0.1, 1.0);
    if !tied {
        put("lm_head.weight".into(), &[TINY_VOCAB, h], 0.3, 0.0);
    }
    restructure_hf_checkpoint(&mut map, &config, device).unwrap();
    (map, config)
}

/// The tensors of `map` that a load of `span` reads, sharing storage.
pub(crate) fn span_map(
    map: &VarMap<CpuRuntime>,
    config: &UniversalConfig,
    span: &crate::model::llama::LayerSpan,
) -> VarMap<CpuRuntime> {
    let mut out = VarMap::new();
    for (name, t) in map.all_tensors() {
        if span.reads(config, name) {
            out.insert(name.to_string(), t.clone());
        }
    }
    out
}
