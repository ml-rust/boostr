//! GGUF to HuggingFace tensor name mapping.
//!
//! GGUF files use short names like `blk.0.attn_q.weight` while HuggingFace
//! models use `model.layers.0.self_attn.q_proj.weight`. This module translates
//! between the two conventions so that `VarMap::from_gguf()` produces tensors
//! with HF-style names that `LoadedModel::load()` expects.
//!
//! Covers: standard transformer (LLaMA), MLA (DeepSeek-V2/V3), MoE,
//! and Mamba2/3 SSM architectures.

/// Map a GGUF tensor name to its HuggingFace equivalent.
///
/// Unrecognized names are returned unchanged.
pub fn gguf_to_hf_name(name: &str) -> String {
    // Global (non-layer) tensors
    match name {
        "token_embd.weight" => return "model.embed_tokens.weight".to_string(),
        "output_norm.weight" => return "model.norm.weight".to_string(),
        "output.weight" => return "lm_head.weight".to_string(),
        _ => {}
    }

    // Layer tensors: blk.N.suffix -> model.layers.N.hf_suffix
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot_pos) = rest.find('.') {
            let layer_num = &rest[..dot_pos];
            let suffix = &rest[dot_pos + 1..];

            if let Some(hf_suffix) = map_layer_suffix(suffix) {
                return format!("model.layers.{layer_num}.{hf_suffix}");
            }

            // MoE expert tensors: ffn_{gate,up,down}.{expert_id}.weight
            if let Some(hf_suffix) = map_moe_expert(suffix) {
                return format!("model.layers.{layer_num}.{hf_suffix}");
            }
        }
    }

    name.to_string()
}

/// Map a GGUF layer suffix to its HuggingFace equivalent.
fn map_layer_suffix(suffix: &str) -> Option<&'static str> {
    Some(match suffix {
        // Standard attention
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",

        // Attention biases
        "attn_q.bias" => "self_attn.q_proj.bias",
        "attn_k.bias" => "self_attn.k_proj.bias",
        "attn_v.bias" => "self_attn.v_proj.bias",
        "attn_output.bias" => "self_attn.o_proj.bias",

        // MLA (Multi-Head Latent Attention) - DeepSeek-V2/V3
        "attn_q_a.weight" => "self_attn.q_a_proj.weight",
        "attn_q_a_norm.weight" => "self_attn.q_a_layernorm.weight",
        "attn_q_b.weight" => "self_attn.q_b_proj.weight",
        "attn_kv_a.weight" => "self_attn.kv_a_proj_with_mqa.weight",
        "attn_kv_a_norm.weight" => "self_attn.kv_a_layernorm.weight",
        "attn_kv_b.weight" => "self_attn.kv_b_proj.weight",

        // Standard MLP (SwiGLU)
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",

        // MoE router
        "ffn_gate_inp.weight" => "moe.gate.weight",
        "ffn_gate_inp.bias" => "moe.gate.bias",

        // MoE shared expert
        "ffn_gate_shexp.weight" => "moe.shared_expert.gate_proj.weight",
        "ffn_up_shexp.weight" => "moe.shared_expert.up_proj.weight",
        "ffn_down_shexp.weight" => "moe.shared_expert.down_proj.weight",

        // Layer norms
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",

        // Mamba2 SSM
        "ssm_in.weight" => "mamba2.mixer.in_proj.weight",
        "ssm_in.bias" => "mamba2.mixer.in_proj.bias",
        "ssm_out.weight" => "mamba2.mixer.out_proj.weight",
        "ssm_out.bias" => "mamba2.mixer.out_proj.bias",
        "ssm_conv1d.weight" => "mamba2.mixer.conv1d.weight",
        "ssm_conv1d.bias" => "mamba2.mixer.conv1d.bias",
        "ssm_a" => "mamba2.mixer.A_log",
        "ssm_dt.bias" => "mamba2.mixer.dt_bias",
        "ssm_d" => "mamba2.mixer.D",
        "ssm_norm.weight" => "mamba2.mixer.norm.weight",

        _ => return None,
    })
}

/// Map a GGUF MoE expert tensor name to its HuggingFace equivalent.
///
/// GGUF uses `blk.N.ffn_gate.E.weight` while HF uses
/// `model.layers.N.moe.experts.E.gate_proj.weight`.
///
/// Returns `None` if the name doesn't match the MoE expert pattern.
pub(crate) fn map_moe_expert(suffix: &str) -> Option<String> {
    // Pattern: ffn_{gate,up,down}.{expert_id}.weight
    let (prefix, rest) = suffix.split_once('.')?;
    let (expert_id, weight_suffix) = rest.split_once('.')?;

    // Validate expert_id is numeric
    if !expert_id.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }

    let proj = match prefix {
        "ffn_gate" => "gate_proj",
        "ffn_up" => "up_proj",
        "ffn_down" => "down_proj",
        _ => return None,
    };

    Some(format!("moe.experts.{expert_id}.{proj}.{weight_suffix}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_mappings() {
        assert_eq!(
            gguf_to_hf_name("token_embd.weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(gguf_to_hf_name("output_norm.weight"), "model.norm.weight");
        assert_eq!(gguf_to_hf_name("output.weight"), "lm_head.weight");
    }

    #[test]
    fn test_standard_attention() {
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_q.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.31.attn_output.weight"),
            "model.layers.31.self_attn.o_proj.weight"
        );
    }

    #[test]
    fn test_attention_biases() {
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_q.bias"),
            "model.layers.0.self_attn.q_proj.bias"
        );
    }

    #[test]
    fn test_mla_mappings() {
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_q_a.weight"),
            "model.layers.0.self_attn.q_a_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_q_a_norm.weight"),
            "model.layers.0.self_attn.q_a_layernorm.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_q_b.weight"),
            "model.layers.0.self_attn.q_b_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_kv_a.weight"),
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_kv_a_norm.weight"),
            "model.layers.0.self_attn.kv_a_layernorm.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_kv_b.weight"),
            "model.layers.0.self_attn.kv_b_proj.weight"
        );
    }

    #[test]
    fn test_mlp_mappings() {
        assert_eq!(
            gguf_to_hf_name("blk.5.ffn_gate.weight"),
            "model.layers.5.mlp.gate_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.5.ffn_down.weight"),
            "model.layers.5.mlp.down_proj.weight"
        );
    }

    #[test]
    fn test_moe_router() {
        assert_eq!(
            gguf_to_hf_name("blk.0.ffn_gate_inp.weight"),
            "model.layers.0.moe.gate.weight"
        );
    }

    #[test]
    fn test_moe_shared_expert() {
        assert_eq!(
            gguf_to_hf_name("blk.0.ffn_gate_shexp.weight"),
            "model.layers.0.moe.shared_expert.gate_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.ffn_up_shexp.weight"),
            "model.layers.0.moe.shared_expert.up_proj.weight"
        );
    }

    #[test]
    fn test_moe_experts() {
        assert_eq!(
            map_moe_expert("ffn_gate.0.weight"),
            Some("moe.experts.0.gate_proj.weight".to_string())
        );
        assert_eq!(
            map_moe_expert("ffn_up.7.weight"),
            Some("moe.experts.7.up_proj.weight".to_string())
        );
        assert_eq!(
            map_moe_expert("ffn_down.63.weight"),
            Some("moe.experts.63.down_proj.weight".to_string())
        );
        assert_eq!(map_moe_expert("ffn_norm.weight"), None);
        assert_eq!(map_moe_expert("ffn_gate.abc.weight"), None);
    }

    #[test]
    fn test_layer_norms() {
        assert_eq!(
            gguf_to_hf_name("blk.5.attn_norm.weight"),
            "model.layers.5.input_layernorm.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.5.ffn_norm.weight"),
            "model.layers.5.post_attention_layernorm.weight"
        );
    }

    #[test]
    fn test_mamba2_mappings() {
        assert_eq!(
            gguf_to_hf_name("blk.0.ssm_in.weight"),
            "model.layers.0.mamba2.mixer.in_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.ssm_out.weight"),
            "model.layers.0.mamba2.mixer.out_proj.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.ssm_conv1d.weight"),
            "model.layers.0.mamba2.mixer.conv1d.weight"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.ssm_conv1d.bias"),
            "model.layers.0.mamba2.mixer.conv1d.bias"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.ssm_a"),
            "model.layers.0.mamba2.mixer.A_log"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.ssm_dt.bias"),
            "model.layers.0.mamba2.mixer.dt_bias"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.ssm_d"),
            "model.layers.0.mamba2.mixer.D"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.ssm_norm.weight"),
            "model.layers.0.mamba2.mixer.norm.weight"
        );
    }

    #[test]
    fn test_unknown_passthrough() {
        assert_eq!(
            gguf_to_hf_name("some.unknown.tensor"),
            "some.unknown.tensor"
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.unknown_suffix.weight"),
            "blk.0.unknown_suffix.weight"
        );
    }
}
