//! DBRX: `transformer.` prefix, fused Wqkv, MoE with v1/w1/w2 expert naming.

use super::dispatch::split_layer_num;

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
pub(super) fn normalize_dbrx(name: &str) -> String {
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
    if let Some(rest) = name.strip_prefix("transformer.blocks.")
        && let Some((layer_num, layer_rest)) = split_layer_num(rest)
    {
        // Attention block
        if let Some(suffix) = layer_rest.strip_prefix("norm_attn_norm.attn.") {
            if let Some(rest_suffix) = suffix.strip_prefix("Wqkv.") {
                return format!("model.layers.{layer_num}.self_attn.query_key_value.{rest_suffix}");
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
    name.to_string()
}

#[cfg(test)]
mod tests {
    use super::super::dispatch::normalize_hf_name;

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
}
