//! GPT-NeoX/Pythia: `gpt_neox.` prefix, fused QKV, different MLP naming.

use super::dispatch::split_layer_num;

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
pub(super) fn normalize_gpt_neox(name: &str) -> String {
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
    if let Some(rest) = name.strip_prefix("gpt_neox.layers.")
        && let Some((layer_num, layer_rest)) = split_layer_num(rest)
    {
        // Attention
        if let Some(suffix) = layer_rest.strip_prefix("attention.") {
            if let Some(rest_suffix) = suffix.strip_prefix("query_key_value.") {
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
        // Layer norms — already canonical names
        if layer_rest.starts_with("input_layernorm.")
            || layer_rest.starts_with("post_attention_layernorm.")
        {
            return format!("model.layers.{layer_num}.{layer_rest}");
        }
    }
    name.to_string()
}

#[cfg(test)]
mod tests {
    use super::super::dispatch::normalize_hf_name;

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
}
