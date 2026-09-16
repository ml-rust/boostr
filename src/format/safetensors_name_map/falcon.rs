//! Falcon: `transformer.` prefix, fused QKV, different MLP naming.

use super::dispatch::split_layer_num;

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
pub(super) fn normalize_falcon(name: &str) -> String {
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
    if let Some(rest) = name.strip_prefix("transformer.h.")
        && let Some((layer_num, layer_rest)) = split_layer_num(rest)
    {
        // Attention
        if let Some(suffix) = layer_rest.strip_prefix("self_attention.") {
            if let Some(rest_suffix) = suffix.strip_prefix("query_key_value.") {
                // Fused QKV — keep with canonical prefix for later splitting
                return format!("model.layers.{layer_num}.self_attn.query_key_value.{rest_suffix}");
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
    name.to_string()
}

#[cfg(test)]
mod tests {
    use super::super::dispatch::normalize_hf_name;

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
}
