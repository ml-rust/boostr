//! Every block variant computes the same logits on every forward path.
//!
//! Each case builds a tiny random checkpoint in Hugging Face names for one
//! architecture, runs `restructure_hf_checkpoint`, and loads it. The
//! cache-free forward, a one-shot KV-cache prefill, and a prefill followed by
//! single-token decodes must agree on the last position's logits. The paths
//! share the block halves but differ in attention kernels, the deferred
//! residual and the fused add+norm, so a variant wired into one path only
//! fails here.
//!
//! Parity with `transformers` lives in blazr's `tests/arch_reference.rs`.

use super::Llama;
use crate::format::restructure_hf_checkpoint;
use crate::inference::LayeredKvCache;
use crate::model::config::{HuggingFaceConfig, ModelConfig};
use crate::model::traits::Model;
use crate::nn::{VarBuilder, VarMap};
use crate::test_utils::cpu_setup;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{RandomOps, ScalarOps};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

const PROMPT: [i64; 6] = [3, 1, 4, 1, 5, 9];

/// Inserts one random tensor: name, shape, scale, offset.
type Put<'a> = dyn FnMut(String, &[usize], f64, f64) + 'a;

fn random(client: &CpuClient, shape: &[usize], scale: f64, offset: f64) -> Tensor<CpuRuntime> {
    let t = client.randn(shape, DType::F32).unwrap();
    let t = client.mul_scalar(&t, scale).unwrap();
    client.add_scalar(&t, offset).unwrap()
}

/// Which optional tensors the checkpoint carries.
#[derive(Default)]
struct Layout {
    attn_bias: bool,
    norm_bias: bool,
    gated_mlp: bool,
    post_norm: bool,
    sandwich: bool,
    experts: usize,
    shared_expert: bool,
    tie: bool,
}

/// Tensors in the canonical names `restructure_hf_checkpoint` reads after
/// renaming, with experts one tensor per expert.
fn checkpoint(client: &CpuClient, config: &ModelConfig, layout: &Layout) -> VarMap<CpuRuntime> {
    let h = config.hidden_size;
    let attn = config.attention.as_ref().unwrap();
    let d = attn.head_dim(h);
    let (q, kv) = (attn.num_heads * d, attn.kv_heads() * d);
    let inter = config.intermediate_size();
    let v = config.vocab_size;
    let mut map = VarMap::<CpuRuntime>::new();
    let mut put = |name: String, shape: &[usize], scale: f64, offset: f64| {
        map.insert(name, random(client, shape, scale, offset));
    };
    put("model.embed_tokens.weight".into(), &[v, h], 0.5, 0.0);
    let norm = |put: &mut Put<'_>, name: String| {
        put(format!("{name}.weight"), &[h], 0.1, 1.0);
        if layout.norm_bias {
            put(format!("{name}.bias"), &[h], 0.1, 0.0);
        }
    };
    for l in 0..config.num_layers {
        let p = format!("model.layers.{l}");
        for (name, rows, cols) in [("q", q, h), ("k", kv, h), ("v", kv, h), ("o", h, q)] {
            put(
                format!("{p}.self_attn.{name}_proj.weight"),
                &[rows, cols],
                0.3,
                0.0,
            );
            if layout.attn_bias {
                put(format!("{p}.self_attn.{name}_proj.bias"), &[rows], 0.1, 0.0);
            }
        }
        norm(&mut put, format!("{p}.input_layernorm"));
        if layout.post_norm {
            norm(&mut put, format!("{p}.post_attention_layernorm"));
        }
        if layout.sandwich {
            norm(&mut put, format!("{p}.pre_feedforward_layernorm"));
            norm(&mut put, format!("{p}.post_feedforward_layernorm"));
        }
        if layout.experts > 0 {
            let m = format!("{p}.block_sparse_moe");
            put(format!("{m}.gate.weight"), &[layout.experts, h], 0.5, 0.0);
            for e in 0..layout.experts {
                put(
                    format!("{m}.experts.{e}.gate_proj.weight"),
                    &[inter, h],
                    0.3,
                    0.0,
                );
                put(
                    format!("{m}.experts.{e}.up_proj.weight"),
                    &[inter, h],
                    0.3,
                    0.0,
                );
                put(
                    format!("{m}.experts.{e}.down_proj.weight"),
                    &[h, inter],
                    0.3,
                    0.0,
                );
            }
            if layout.shared_expert {
                put(
                    format!("{m}.shared_expert.gate_proj.weight"),
                    &[inter, h],
                    0.3,
                    0.0,
                );
                put(
                    format!("{m}.shared_expert.up_proj.weight"),
                    &[inter, h],
                    0.3,
                    0.0,
                );
                put(
                    format!("{m}.shared_expert.down_proj.weight"),
                    &[h, inter],
                    0.3,
                    0.0,
                );
                put(format!("{m}.shared_expert_gate.weight"), &[1, h], 0.5, 0.0);
            }
        } else {
            let bias = layout.attn_bias;
            if layout.gated_mlp {
                put(format!("{p}.mlp.gate_proj.weight"), &[inter, h], 0.3, 0.0);
            }
            put(format!("{p}.mlp.up_proj.weight"), &[inter, h], 0.3, 0.0);
            put(format!("{p}.mlp.down_proj.weight"), &[h, inter], 0.3, 0.0);
            if bias {
                put(format!("{p}.mlp.up_proj.bias"), &[inter], 0.1, 0.0);
                put(format!("{p}.mlp.down_proj.bias"), &[h], 0.1, 0.0);
            }
        }
    }
    norm(&mut put, "model.norm".into());
    if !layout.tie {
        put("lm_head.weight".into(), &[v, h], 0.3, 0.0);
    }
    map
}

fn load(json: &str, layout: &Layout) -> (Llama<CpuRuntime>, CpuClient, CpuDevice) {
    let (client, device) = cpu_setup();
    let hf: HuggingFaceConfig = serde_json::from_str(json).unwrap();
    let config = hf.to_universal().unwrap();
    assert_eq!(config.tie_word_embeddings, layout.tie, "layout tie flag");
    let mut map = checkpoint(&client, &config, layout);
    restructure_hf_checkpoint(&mut map, &config, &device).unwrap();
    let mut vb = VarBuilder::new(&mut map, &device);
    let model = Llama::<CpuRuntime>::from_varbuilder(&mut vb, &config).unwrap();
    (model, client, device)
}

fn last_row(logits: &Tensor<CpuRuntime>) -> Vec<f32> {
    let v = logits.shape()[2];
    let all: Vec<f32> = logits.contiguous().unwrap().to_vec();
    all[all.len() - v..].to_vec()
}

fn cache(model: &Llama<CpuRuntime>, device: &CpuDevice) -> LayeredKvCache<CpuRuntime> {
    let attn = model.config().attention.as_ref().unwrap();
    LayeredKvCache::new_positional(
        model.config().num_layers,
        1,
        attn.kv_heads(),
        16,
        16,
        attn.head_dim(model.config().hidden_size),
        DType::F32,
        device,
    )
    .unwrap()
}

fn ids(tokens: &[i64], device: &CpuDevice) -> Tensor<CpuRuntime> {
    Tensor::from_slice(tokens, &[1, tokens.len()], device).unwrap()
}

/// The cache-free, prefill and decode logits of the last prompt token.
fn assert_paths_agree(json: &str, layout: Layout) {
    let (model, client, device) = load(json, &layout);

    let full = model
        .forward(&client, &Var::new(ids(&PROMPT, &device), false))
        .unwrap();
    let full = last_row(full.tensor());

    let mut kv = cache(&model, &device);
    let prefill = model
        .forward_with_kv_cache(&client, &ids(&PROMPT, &device), &mut kv, 0)
        .unwrap();
    let prefill = last_row(&prefill);

    let mut kv = cache(&model, &device);
    let split = PROMPT.len() - 2;
    model
        .forward_with_kv_cache(&client, &ids(&PROMPT[..split], &device), &mut kv, 0)
        .unwrap();
    let mut decode = Vec::new();
    for (i, &t) in PROMPT[split..].iter().enumerate() {
        let out = model
            .forward_with_kv_cache(&client, &ids(&[t], &device), &mut kv, split + i)
            .unwrap();
        decode = last_row(&out);
    }

    let scale = full.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1.0);
    for (name, other) in [("prefill", &prefill), ("decode", &decode)] {
        for (i, (a, b)) in full.iter().zip(other.iter()).enumerate() {
            assert!(
                a.is_finite() && (a - b).abs() <= 1e-4 * scale,
                "{name} logit {i}: {b}, cache-free forward gives {a}"
            );
        }
    }
}

#[test]
fn gpt_neox_paths_agree() {
    assert_paths_agree(
        r#"{"model_type": "gpt_neox", "vocab_size": 16, "hidden_size": 16,
            "num_hidden_layers": 2, "max_position_embeddings": 16,
            "num_attention_heads": 2, "intermediate_size": 32, "rotary_pct": 0.25,
            "hidden_act": "gelu", "use_parallel_residual": true,
            "tie_word_embeddings": false}"#,
        Layout {
            attn_bias: true,
            norm_bias: true,
            post_norm: true,
            ..Layout::default()
        },
    );
}

#[test]
fn gemma2_paths_agree() {
    assert_paths_agree(
        r#"{"model_type": "gemma2", "vocab_size": 16, "hidden_size": 16,
            "num_hidden_layers": 2, "max_position_embeddings": 16,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
            "intermediate_size": 32, "query_pre_attn_scalar": 4,
            "attn_logit_softcapping": 2.0, "final_logit_softcapping": 3.0,
            "sliding_window": 3, "hidden_activation": "gelu_pytorch_tanh"}"#,
        Layout {
            gated_mlp: true,
            post_norm: true,
            sandwich: true,
            tie: true,
            ..Layout::default()
        },
    );
}

#[test]
fn cohere_paths_agree() {
    assert_paths_agree(
        r#"{"model_type": "cohere", "vocab_size": 16, "hidden_size": 16,
            "num_hidden_layers": 2, "max_position_embeddings": 16,
            "num_attention_heads": 2, "intermediate_size": 32,
            "logit_scale": 0.5, "use_qk_norm": false}"#,
        Layout {
            gated_mlp: true,
            tie: true,
            ..Layout::default()
        },
    );
}

#[test]
fn falcon_alibi_paths_agree() {
    assert_paths_agree(
        r#"{"model_type": "falcon", "vocab_size": 16, "hidden_size": 16,
            "n_layer": 2, "n_head": 2, "alibi": true, "parallel_attn": false,
            "intermediate_size": 32, "tie_word_embeddings": false}"#,
        Layout {
            attn_bias: true,
            norm_bias: true,
            post_norm: true,
            ..Layout::default()
        },
    );
}

#[test]
fn dbrx_like_clip_and_moe_paths_agree() {
    assert_paths_agree(
        r#"{"model_type": "dbrx", "vocab_size": 16, "d_model": 16, "n_heads": 2,
            "n_layers": 2, "max_seq_len": 16,
            "attn_config": {"kv_n_heads": 1, "clip_qkv": 0.5},
            "ffn_config": {"ffn_hidden_size": 32, "moe_num_experts": 4, "moe_top_k": 2,
                           "moe_normalize_expert_weights": 1.0}}"#,
        Layout {
            gated_mlp: true,
            post_norm: true,
            experts: 4,
            ..Layout::default()
        },
    );
}

#[test]
fn qwen2_moe_shared_gate_paths_agree() {
    assert_paths_agree(
        r#"{"model_type": "qwen2_moe", "vocab_size": 16, "hidden_size": 16,
            "num_hidden_layers": 2, "max_position_embeddings": 16,
            "num_attention_heads": 2, "intermediate_size": 32, "num_experts": 4,
            "num_experts_per_tok": 2, "norm_topk_prob": false, "hidden_act": "gelu"}"#,
        Layout {
            attn_bias: true,
            post_norm: true,
            experts: 4,
            shared_expert: true,
            ..Layout::default()
        },
    );
}
