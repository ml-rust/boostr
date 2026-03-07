//! HuggingFace SafeTensors tensor name normalization.
//!
//! Maps non-standard HF tensor names to canonical Llama-style names at load time.
//! Same pattern as `gguf_to_hf_name()` in the GGUF module.
//!
//! The canonical naming convention is:
//! ```text
//! model.embed_tokens.weight
//! model.layers.{N}.self_attn.{q,k,v,o}_proj.weight
//! model.layers.{N}.mlp.{gate,up,down}_proj.weight
//! model.layers.{N}.input_layernorm.weight
//! model.layers.{N}.post_attention_layernorm.weight
//! model.norm.weight
//! lm_head.weight
//! ```

/// Normalize a HuggingFace SafeTensors tensor name to canonical Llama-style naming.
///
/// If `model_type` is not a known non-standard architecture, the name is returned as-is.
/// This is a zero-cost passthrough for Llama-family models.
pub fn normalize_hf_name(model_type: &str, name: &str) -> String {
    // Audio-language models: normalize audio encoder/projector prefixes first,
    // then fall through to standard LLM backbone normalization.
    let name = normalize_audio_language(name);

    match model_type {
        "falcon" => normalize_falcon(&name),
        "gpt_neox" => normalize_gpt_neox(&name),
        "dbrx" => normalize_dbrx(&name),
        _ => name,
    }
}

/// Falcon: `transformer.` prefix, fused QKV, different MLP naming.
///
/// Falcon naming:
/// - `transformer.word_embeddings.weight` → `model.embed_tokens.weight`
/// - `transformer.h.{N}.self_attention.query_key_value.weight` → fused QKV (kept as-is, split at load)
/// - `transformer.h.{N}.self_attention.dense.weight` → `model.layers.{N}.self_attn.o_proj.weight`
/// - `transformer.h.{N}.mlp.dense_h_to_4h.weight` → `model.layers.{N}.mlp.up_proj.weight`
/// - `transformer.h.{N}.mlp.dense_4h_to_h.weight` → `model.layers.{N}.mlp.down_proj.weight`
/// - `transformer.h.{N}.ln_attn.weight` → `model.layers.{N}.input_layernorm.weight`
/// - `transformer.h.{N}.ln_mlp.weight` → `model.layers.{N}.post_attention_layernorm.weight`
/// - `transformer.ln_f.weight` → `model.norm.weight`
/// - `lm_head.weight` → `lm_head.weight` (unchanged)
fn normalize_falcon(name: &str) -> String {
    // Embeddings
    if name == "transformer.word_embeddings.weight" {
        return "model.embed_tokens.weight".to_string();
    }
    // Final norm
    if let Some(suffix) = name.strip_prefix("transformer.ln_f.") {
        return format!("model.norm.{suffix}");
    }
    // lm_head passes through
    if name.starts_with("lm_head.") {
        return name.to_string();
    }
    // Layer tensors: transformer.h.{N}.{rest}
    if let Some(rest) = name.strip_prefix("transformer.h.") {
        if let Some((layer_num, layer_rest)) = split_layer_num(rest) {
            // Attention
            if let Some(suffix) = layer_rest.strip_prefix("self_attention.") {
                if suffix.starts_with("query_key_value.") {
                    // Fused QKV — keep with canonical prefix for later splitting
                    return format!(
                        "model.layers.{layer_num}.self_attn.query_key_value.{rest_suffix}",
                        rest_suffix = suffix.strip_prefix("query_key_value.").unwrap()
                    );
                }
                if let Some(s) = suffix.strip_prefix("dense.") {
                    return format!("model.layers.{layer_num}.self_attn.o_proj.{s}");
                }
            }
            // MLP
            if let Some(suffix) = layer_rest.strip_prefix("mlp.") {
                if let Some(s) = suffix.strip_prefix("dense_h_to_4h.") {
                    return format!("model.layers.{layer_num}.mlp.up_proj.{s}");
                }
                if let Some(s) = suffix.strip_prefix("dense_4h_to_h.") {
                    return format!("model.layers.{layer_num}.mlp.down_proj.{s}");
                }
            }
            // Layer norms
            if let Some(s) = layer_rest.strip_prefix("ln_attn.") {
                return format!("model.layers.{layer_num}.input_layernorm.{s}");
            }
            if let Some(s) = layer_rest.strip_prefix("ln_mlp.") {
                return format!("model.layers.{layer_num}.post_attention_layernorm.{s}");
            }
            // Falcon v2 uses input_layernorm directly
            if let Some(s) = layer_rest.strip_prefix("input_layernorm.") {
                return format!("model.layers.{layer_num}.input_layernorm.{s}");
            }
        }
    }
    name.to_string()
}

/// GPT-NeoX/Pythia: `gpt_neox.` prefix, fused QKV, different MLP naming.
///
/// GPT-NeoX naming:
/// - `gpt_neox.embed_in.weight` → `model.embed_tokens.weight`
/// - `gpt_neox.layers.{N}.attention.query_key_value.weight` → fused QKV
/// - `gpt_neox.layers.{N}.attention.dense.weight` → `model.layers.{N}.self_attn.o_proj.weight`
/// - `gpt_neox.layers.{N}.mlp.dense_h_to_4h.weight` → `model.layers.{N}.mlp.up_proj.weight`
/// - `gpt_neox.layers.{N}.mlp.dense_4h_to_h.weight` → `model.layers.{N}.mlp.down_proj.weight`
/// - `gpt_neox.layers.{N}.input_layernorm.weight` → `model.layers.{N}.input_layernorm.weight`
/// - `gpt_neox.layers.{N}.post_attention_layernorm.weight` → same
/// - `gpt_neox.final_layer_norm.weight` → `model.norm.weight`
/// - `embed_out.weight` → `lm_head.weight`
fn normalize_gpt_neox(name: &str) -> String {
    // Embeddings
    if name == "gpt_neox.embed_in.weight" {
        return "model.embed_tokens.weight".to_string();
    }
    // LM head
    if name == "embed_out.weight" {
        return "lm_head.weight".to_string();
    }
    if let Some(suffix) = name.strip_prefix("embed_out.") {
        return format!("lm_head.{suffix}");
    }
    // Final norm
    if let Some(suffix) = name.strip_prefix("gpt_neox.final_layer_norm.") {
        return format!("model.norm.{suffix}");
    }
    // Layer tensors: gpt_neox.layers.{N}.{rest}
    if let Some(rest) = name.strip_prefix("gpt_neox.layers.") {
        if let Some((layer_num, layer_rest)) = split_layer_num(rest) {
            // Attention
            if let Some(suffix) = layer_rest.strip_prefix("attention.") {
                if suffix.starts_with("query_key_value.") {
                    return format!(
                        "model.layers.{layer_num}.self_attn.query_key_value.{rest_suffix}",
                        rest_suffix = suffix.strip_prefix("query_key_value.").unwrap()
                    );
                }
                if let Some(s) = suffix.strip_prefix("dense.") {
                    return format!("model.layers.{layer_num}.self_attn.o_proj.{s}");
                }
            }
            // MLP
            if let Some(suffix) = layer_rest.strip_prefix("mlp.") {
                if let Some(s) = suffix.strip_prefix("dense_h_to_4h.") {
                    return format!("model.layers.{layer_num}.mlp.up_proj.{s}");
                }
                if let Some(s) = suffix.strip_prefix("dense_4h_to_h.") {
                    return format!("model.layers.{layer_num}.mlp.down_proj.{s}");
                }
            }
            // Layer norms — already canonical names
            if layer_rest.starts_with("input_layernorm.")
                || layer_rest.starts_with("post_attention_layernorm.")
            {
                return format!("model.layers.{layer_num}.{layer_rest}");
            }
        }
    }
    name.to_string()
}

/// DBRX: `transformer.` prefix, fused Wqkv, MoE with v1/w1/w2 expert naming.
///
/// DBRX naming:
/// - `transformer.wte.weight` → `model.embed_tokens.weight`
/// - `transformer.blocks.{N}.norm_attn_norm.attn.Wqkv.weight` → fused QKV
/// - `transformer.blocks.{N}.norm_attn_norm.attn.out_proj.weight` → o_proj
/// - `transformer.blocks.{N}.norm_attn_norm.norm_1.weight` → input_layernorm
/// - `transformer.blocks.{N}.norm_attn_norm.norm_2.weight` → post_attention_layernorm
/// - `transformer.blocks.{N}.ffn.router.layer.weight` → MoE gate
/// - `transformer.blocks.{N}.ffn.experts.mlp.{E}.v1.weight` → expert gate_proj
/// - `transformer.blocks.{N}.ffn.experts.mlp.{E}.w1.weight` → expert up_proj
/// - `transformer.blocks.{N}.ffn.experts.mlp.{E}.w2.weight` → expert down_proj
/// - `transformer.norm_f.weight` → `model.norm.weight`
fn normalize_dbrx(name: &str) -> String {
    // Embeddings
    if name == "transformer.wte.weight" {
        return "model.embed_tokens.weight".to_string();
    }
    // Final norm
    if let Some(suffix) = name.strip_prefix("transformer.norm_f.") {
        return format!("model.norm.{suffix}");
    }
    // lm_head passes through
    if name.starts_with("lm_head.") {
        return name.to_string();
    }
    // Layer tensors: transformer.blocks.{N}.{rest}
    if let Some(rest) = name.strip_prefix("transformer.blocks.") {
        if let Some((layer_num, layer_rest)) = split_layer_num(rest) {
            // Attention block
            if let Some(suffix) = layer_rest.strip_prefix("norm_attn_norm.attn.") {
                if suffix.starts_with("Wqkv.") {
                    return format!(
                        "model.layers.{layer_num}.self_attn.query_key_value.{rest_suffix}",
                        rest_suffix = suffix.strip_prefix("Wqkv.").unwrap()
                    );
                }
                if let Some(s) = suffix.strip_prefix("out_proj.") {
                    return format!("model.layers.{layer_num}.self_attn.o_proj.{s}");
                }
            }
            // Norms
            if let Some(s) = layer_rest.strip_prefix("norm_attn_norm.norm_1.") {
                return format!("model.layers.{layer_num}.input_layernorm.{s}");
            }
            if let Some(s) = layer_rest.strip_prefix("norm_attn_norm.norm_2.") {
                return format!("model.layers.{layer_num}.post_attention_layernorm.{s}");
            }
            // MoE FFN
            if let Some(s) = layer_rest.strip_prefix("ffn.router.layer.") {
                return format!("model.layers.{layer_num}.block_sparse_moe.gate.{s}");
            }
            // Expert weights: ffn.experts.mlp.{E}.{v1,w1,w2}.weight
            if let Some(suffix) = layer_rest.strip_prefix("ffn.experts.mlp.") {
                // We keep expert-level naming for now; the MoE loader handles stacking
                let mapped = suffix
                    .replace(".v1.", ".gate_proj.")
                    .replace(".w1.", ".up_proj.")
                    .replace(".w2.", ".down_proj.");
                return format!("model.layers.{layer_num}.block_sparse_moe.experts.{mapped}");
            }
        }
    }
    name.to_string()
}

/// Normalize audio-language model weight prefixes to canonical boostr names.
///
/// Ultravox uses:
/// - `audio_tower.*` → `audio_encoder.*` (Whisper encoder)
/// - `multi_modal_projector.*` → kept as-is (shared with vision)
///
/// Qwen2-Audio uses:
/// - `audio_model.*` → `audio_encoder.*`
/// - `audio_projector.*` → kept as-is
///
/// Both conventions are normalized so that `MultimodalModel::load_audio()`
/// can load weights with the `audio_encoder.` and `audio_projector.` prefixes.
fn normalize_audio_language(name: &str) -> String {
    // Ultravox: audio_tower.* → audio_encoder.*
    if let Some(rest) = name.strip_prefix("audio_tower.") {
        return format!("audio_encoder.{rest}");
    }
    // Qwen2-Audio: audio_model.* → audio_encoder.*
    if let Some(rest) = name.strip_prefix("audio_model.") {
        return format!("audio_encoder.{rest}");
    }
    // Ultravox uses multi_modal_projector for audio too; when audio_config is
    // present without vision_config, map it to audio_projector.
    // However, we can't know the config here, so we keep multi_modal_projector
    // as-is — the VarBuilder prefix-based loading handles this correctly.
    name.to_string()
}

/// Split "42.rest_of_path" into (42, "rest_of_path").
fn split_layer_num(s: &str) -> Option<(usize, &str)> {
    let dot_pos = s.find('.')?;
    let num: usize = s[..dot_pos].parse().ok()?;
    Some((num, &s[dot_pos + 1..]))
}

/// Check if a model type uses fused QKV tensors that need splitting.
pub fn uses_fused_qkv(model_type: &str) -> bool {
    matches!(model_type, "falcon" | "gpt_neox" | "dbrx")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Falcon ────────────────────────────────────────────────────────

    #[test]
    fn falcon_embeddings() {
        assert_eq!(
            normalize_hf_name("falcon", "transformer.word_embeddings.weight"),
            "model.embed_tokens.weight"
        );
    }

    #[test]
    fn falcon_final_norm() {
        assert_eq!(
            normalize_hf_name("falcon", "transformer.ln_f.weight"),
            "model.norm.weight"
        );
    }

    #[test]
    fn falcon_attention_qkv() {
        assert_eq!(
            normalize_hf_name(
                "falcon",
                "transformer.h.5.self_attention.query_key_value.weight"
            ),
            "model.layers.5.self_attn.query_key_value.weight"
        );
    }

    #[test]
    fn falcon_attention_dense() {
        assert_eq!(
            normalize_hf_name("falcon", "transformer.h.3.self_attention.dense.weight"),
            "model.layers.3.self_attn.o_proj.weight"
        );
    }

    #[test]
    fn falcon_mlp() {
        assert_eq!(
            normalize_hf_name("falcon", "transformer.h.0.mlp.dense_h_to_4h.weight"),
            "model.layers.0.mlp.up_proj.weight"
        );
        assert_eq!(
            normalize_hf_name("falcon", "transformer.h.0.mlp.dense_4h_to_h.weight"),
            "model.layers.0.mlp.down_proj.weight"
        );
    }

    #[test]
    fn falcon_layernorms() {
        assert_eq!(
            normalize_hf_name("falcon", "transformer.h.2.ln_attn.weight"),
            "model.layers.2.input_layernorm.weight"
        );
        assert_eq!(
            normalize_hf_name("falcon", "transformer.h.2.ln_mlp.weight"),
            "model.layers.2.post_attention_layernorm.weight"
        );
    }

    #[test]
    fn falcon_lm_head_passthrough() {
        assert_eq!(
            normalize_hf_name("falcon", "lm_head.weight"),
            "lm_head.weight"
        );
    }

    // ── GPT-NeoX ─────────────────────────────────────────────────────

    #[test]
    fn gpt_neox_embeddings() {
        assert_eq!(
            normalize_hf_name("gpt_neox", "gpt_neox.embed_in.weight"),
            "model.embed_tokens.weight"
        );
    }

    #[test]
    fn gpt_neox_lm_head() {
        assert_eq!(
            normalize_hf_name("gpt_neox", "embed_out.weight"),
            "lm_head.weight"
        );
    }

    #[test]
    fn gpt_neox_final_norm() {
        assert_eq!(
            normalize_hf_name("gpt_neox", "gpt_neox.final_layer_norm.weight"),
            "model.norm.weight"
        );
    }

    #[test]
    fn gpt_neox_attention_qkv() {
        assert_eq!(
            normalize_hf_name(
                "gpt_neox",
                "gpt_neox.layers.7.attention.query_key_value.weight"
            ),
            "model.layers.7.self_attn.query_key_value.weight"
        );
    }

    #[test]
    fn gpt_neox_attention_dense() {
        assert_eq!(
            normalize_hf_name("gpt_neox", "gpt_neox.layers.0.attention.dense.weight"),
            "model.layers.0.self_attn.o_proj.weight"
        );
    }

    #[test]
    fn gpt_neox_mlp() {
        assert_eq!(
            normalize_hf_name("gpt_neox", "gpt_neox.layers.1.mlp.dense_h_to_4h.weight"),
            "model.layers.1.mlp.up_proj.weight"
        );
        assert_eq!(
            normalize_hf_name("gpt_neox", "gpt_neox.layers.1.mlp.dense_4h_to_h.weight"),
            "model.layers.1.mlp.down_proj.weight"
        );
    }

    #[test]
    fn gpt_neox_layernorms() {
        assert_eq!(
            normalize_hf_name("gpt_neox", "gpt_neox.layers.3.input_layernorm.weight"),
            "model.layers.3.input_layernorm.weight"
        );
        assert_eq!(
            normalize_hf_name(
                "gpt_neox",
                "gpt_neox.layers.3.post_attention_layernorm.weight"
            ),
            "model.layers.3.post_attention_layernorm.weight"
        );
    }

    // ── DBRX ─────────────────────────────────────────────────────────

    #[test]
    fn dbrx_embeddings() {
        assert_eq!(
            normalize_hf_name("dbrx", "transformer.wte.weight"),
            "model.embed_tokens.weight"
        );
    }

    #[test]
    fn dbrx_final_norm() {
        assert_eq!(
            normalize_hf_name("dbrx", "transformer.norm_f.weight"),
            "model.norm.weight"
        );
    }

    #[test]
    fn dbrx_attention_qkv() {
        assert_eq!(
            normalize_hf_name(
                "dbrx",
                "transformer.blocks.0.norm_attn_norm.attn.Wqkv.weight"
            ),
            "model.layers.0.self_attn.query_key_value.weight"
        );
    }

    #[test]
    fn dbrx_attention_out() {
        assert_eq!(
            normalize_hf_name(
                "dbrx",
                "transformer.blocks.1.norm_attn_norm.attn.out_proj.weight"
            ),
            "model.layers.1.self_attn.o_proj.weight"
        );
    }

    #[test]
    fn dbrx_norms() {
        assert_eq!(
            normalize_hf_name("dbrx", "transformer.blocks.2.norm_attn_norm.norm_1.weight"),
            "model.layers.2.input_layernorm.weight"
        );
        assert_eq!(
            normalize_hf_name("dbrx", "transformer.blocks.2.norm_attn_norm.norm_2.weight"),
            "model.layers.2.post_attention_layernorm.weight"
        );
    }

    #[test]
    fn dbrx_moe_router() {
        assert_eq!(
            normalize_hf_name("dbrx", "transformer.blocks.0.ffn.router.layer.weight"),
            "model.layers.0.block_sparse_moe.gate.weight"
        );
    }

    #[test]
    fn dbrx_expert_weights() {
        assert_eq!(
            normalize_hf_name("dbrx", "transformer.blocks.0.ffn.experts.mlp.0.v1.weight"),
            "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight"
        );
        assert_eq!(
            normalize_hf_name("dbrx", "transformer.blocks.0.ffn.experts.mlp.0.w1.weight"),
            "model.layers.0.block_sparse_moe.experts.0.up_proj.weight"
        );
        assert_eq!(
            normalize_hf_name("dbrx", "transformer.blocks.0.ffn.experts.mlp.0.w2.weight"),
            "model.layers.0.block_sparse_moe.experts.0.down_proj.weight"
        );
    }

    // ── Standard models pass through unchanged ──────────────────────

    #[test]
    fn standard_passthrough() {
        let name = "model.layers.0.self_attn.q_proj.weight";
        assert_eq!(normalize_hf_name("llama", name), name);
        assert_eq!(normalize_hf_name("mistral", name), name);
        assert_eq!(normalize_hf_name("qwen2", name), name);
    }

    // ── Audio-language models ────────────────────────────────────────

    #[test]
    fn ultravox_audio_tower_normalized() {
        assert_eq!(
            normalize_hf_name(
                "llama",
                "audio_tower.encoder.layers.0.self_attn.q_proj.weight"
            ),
            "audio_encoder.encoder.layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn qwen2_audio_model_normalized() {
        assert_eq!(
            normalize_hf_name(
                "qwen2",
                "audio_model.encoder.layers.0.self_attn.q_proj.weight"
            ),
            "audio_encoder.encoder.layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn audio_projector_passthrough() {
        assert_eq!(
            normalize_hf_name("llama", "audio_projector.linear.weight"),
            "audio_projector.linear.weight"
        );
    }

    #[test]
    fn multi_modal_projector_passthrough() {
        assert_eq!(
            normalize_hf_name("llama", "multi_modal_projector.linear_1.weight"),
            "multi_modal_projector.linear_1.weight"
        );
    }

    // ── uses_fused_qkv ──────────────────────────────────────────────

    #[test]
    fn fused_qkv_detection() {
        assert!(uses_fused_qkv("falcon"));
        assert!(uses_fused_qkv("gpt_neox"));
        assert!(uses_fused_qkv("dbrx"));
        assert!(!uses_fused_qkv("llama"));
        assert!(!uses_fused_qkv("mistral"));
    }
}
