//! Architecture detection entry points and helper functions.

use super::detectors::{
    detect_mamba2_params, detect_mamba3_params, detect_mla_params, detect_moe_params,
    detect_transformer_params,
};
use super::types::{DetectedConfig, LayerType, ModelFormat};
use crate::error::{Error, Result};
use crate::format::SafeTensors;
use std::collections::HashSet;

/// Detect model architecture from tensor names and shapes using boostr's SafeTensors
pub fn detect_architecture(safetensors: &SafeTensors) -> Result<DetectedConfig> {
    let tensor_names: Vec<&str> = safetensors.tensor_names().collect();
    let mut config = DetectedConfig::default();

    // Detect model format (HuggingFace vs oxidizr)
    let format = detect_format(&tensor_names);
    config.format = format;
    let prefix = match format {
        ModelFormat::HuggingFace => "model.",
        ModelFormat::Oxidizr => "",
    };

    // Detect hidden_size and vocab_size from embedding layer
    let embed_key = format!("{}embed_tokens.weight", prefix);
    if let Ok(info) = safetensors.tensor_info(&embed_key) {
        if info.shape.len() == 2 {
            config.vocab_size = info.shape[0];
            config.hidden_size = info.shape[1];
        }
    } else {
        return Err(Error::ModelError {
            reason: format!("Cannot find {} tensor", embed_key),
        });
    }

    // Check if embeddings are tied (no separate lm_head)
    config.tie_word_embeddings = safetensors.tensor_info("lm_head.weight").is_err();

    // Find all layer indices
    let layer_indices = detect_layer_indices(&tensor_names, prefix);
    config.num_layers = layer_indices.len();

    if config.num_layers == 0 {
        return Err(Error::ModelError {
            reason: "No layers detected in checkpoint".into(),
        });
    }

    // Detect layer types and parameters
    for &layer_idx in &layer_indices {
        let layer_type = detect_layer_type(&tensor_names, layer_idx, prefix);
        config.layer_types.push(layer_type);

        match layer_type {
            LayerType::Mamba3 => {
                if config.mamba2_num_heads.is_none() {
                    detect_mamba2_params(safetensors, layer_idx, &mut config, prefix);
                }
                if config.mamba3_enabled.is_none() {
                    detect_mamba3_params(
                        safetensors,
                        &tensor_names,
                        layer_idx,
                        &mut config,
                        prefix,
                    );
                }
            }
            LayerType::Mamba2 => {
                if config.mamba2_num_heads.is_none() {
                    detect_mamba2_params(safetensors, layer_idx, &mut config, prefix);
                }
            }
            LayerType::MlaWithMoe | LayerType::MlaWithMlp => {
                if config.num_attention_heads.is_none() {
                    detect_mla_params(safetensors, layer_idx, &mut config, prefix);
                }
                if layer_type == LayerType::MlaWithMoe && config.num_experts.is_none() {
                    detect_moe_params(safetensors, &tensor_names, layer_idx, &mut config, prefix);
                }
            }
            LayerType::StandardTransformer => {
                if config.num_attention_heads.is_none() {
                    detect_transformer_params(
                        safetensors,
                        &tensor_names,
                        layer_idx,
                        &mut config,
                        prefix,
                    );
                }
            }
        }
    }

    Ok(config)
}

/// Detect model architecture from tensor names only (for quick detection)
pub fn detect_architecture_from_names(tensor_names: &[String]) -> Result<DetectedConfig> {
    let tensor_name_refs: Vec<&str> = tensor_names.iter().map(|s| s.as_str()).collect();
    let mut config = DetectedConfig::default();

    let format = detect_format(&tensor_name_refs);
    config.format = format;
    let prefix = match format {
        ModelFormat::HuggingFace => "model.",
        ModelFormat::Oxidizr => "",
    };

    let layer_indices = detect_layer_indices(&tensor_name_refs, prefix);
    config.num_layers = layer_indices.len();

    if config.num_layers == 0 {
        return Err(Error::ModelError {
            reason: "No layers detected in checkpoint".into(),
        });
    }

    config.tie_word_embeddings = !tensor_names.iter().any(|n| n == "lm_head.weight");

    for &layer_idx in &layer_indices {
        let layer_type = detect_layer_type(&tensor_name_refs, layer_idx, prefix);
        config.layer_types.push(layer_type);
    }

    Ok(config)
}

/// Detect model format from tensor names
pub(super) fn detect_format(tensor_names: &[&str]) -> ModelFormat {
    if tensor_names.iter().any(|k| k.starts_with("model.")) {
        ModelFormat::HuggingFace
    } else {
        ModelFormat::Oxidizr
    }
}

/// Find all layer indices from tensor names
pub(super) fn detect_layer_indices(tensor_names: &[&str], prefix: &str) -> Vec<usize> {
    let mut indices: HashSet<usize> = HashSet::new();
    let layer_prefix = format!("{}layers.", prefix);

    for name in tensor_names {
        if name.starts_with(&layer_prefix) {
            if let Some(rest) = name.strip_prefix(&layer_prefix) {
                if let Some(dot_pos) = rest.find('.') {
                    if let Ok(idx) = rest[..dot_pos].parse::<usize>() {
                        indices.insert(idx);
                    }
                }
            }
        }
    }

    let mut indices: Vec<usize> = indices.into_iter().collect();
    indices.sort();
    indices
}

/// Detect the type of a specific layer
pub(super) fn detect_layer_type(
    tensor_names: &[&str],
    layer_idx: usize,
    model_prefix: &str,
) -> LayerType {
    let prefix = format!("{}layers.{}.", model_prefix, layer_idx);

    let has_mamba3 = tensor_names
        .iter()
        .any(|k| k.starts_with(&format!("{}mamba3.", prefix)));
    let has_mamba2 = tensor_names
        .iter()
        .any(|k| k.starts_with(&format!("{}mamba2.", prefix)));
    let has_mla = tensor_names
        .iter()
        .any(|k| k.starts_with(&format!("{}self_attn.w_dkv", prefix)));
    let has_moe = tensor_names
        .iter()
        .any(|k| k.starts_with(&format!("{}moe.", prefix)));
    let has_standard_attn = tensor_names
        .iter()
        .any(|k| k.starts_with(&format!("{}self_attn.q_proj", prefix)));
    let has_mlp = tensor_names
        .iter()
        .any(|k| k.starts_with(&format!("{}mlp.", prefix)));

    if has_mamba3 {
        LayerType::Mamba3
    } else if has_mamba2 {
        LayerType::Mamba2
    } else if has_mla && has_moe {
        LayerType::MlaWithMoe
    } else if has_mla && has_mlp {
        LayerType::MlaWithMlp
    } else if has_standard_attn {
        LayerType::StandardTransformer
    } else if has_mla {
        // MLA without MoE or MLP - assume MoE for now
        LayerType::MlaWithMoe
    } else {
        // Default to standard transformer
        LayerType::StandardTransformer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer_names(layer_idx: usize, suffixes: &[&str]) -> Vec<String> {
        suffixes
            .iter()
            .map(|s| format!("layers.{}.", layer_idx) + s)
            .collect()
    }

    fn refs(names: &[String]) -> Vec<&str> {
        names.iter().map(|s| s.as_str()).collect()
    }

    #[test]
    fn test_detect_mamba3_layer() {
        let names = layer_names(0, &["mamba3.mixer.A_log", "mamba3.mixer.conv1d.weight"]);
        assert_eq!(detect_layer_type(&refs(&names), 0, ""), LayerType::Mamba3);
    }

    #[test]
    fn test_detect_mamba2_layer() {
        let names = layer_names(0, &["mamba2.mixer.A_log", "mamba2.mixer.conv1d.weight"]);
        assert_eq!(detect_layer_type(&refs(&names), 0, ""), LayerType::Mamba2);
    }

    #[test]
    fn test_detect_mla_with_moe_layer() {
        let names = layer_names(
            0,
            &[
                "self_attn.w_dkv.weight",
                "moe.gate.weight",
                "moe.experts.0.up_proj.weight",
            ],
        );
        assert_eq!(
            detect_layer_type(&refs(&names), 0, ""),
            LayerType::MlaWithMoe
        );
    }

    #[test]
    fn test_detect_mla_with_mlp_layer() {
        let names = layer_names(
            0,
            &[
                "self_attn.w_dkv.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
            ],
        );
        assert_eq!(
            detect_layer_type(&refs(&names), 0, ""),
            LayerType::MlaWithMlp
        );
    }

    #[test]
    fn test_detect_standard_transformer_layer() {
        let names = layer_names(
            0,
            &[
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "mlp.gate_proj.weight",
            ],
        );
        assert_eq!(
            detect_layer_type(&refs(&names), 0, ""),
            LayerType::StandardTransformer
        );
    }

    #[test]
    fn test_detect_mla_alone_defaults_to_moe() {
        let names = layer_names(0, &["self_attn.w_dkv.weight"]);
        assert_eq!(
            detect_layer_type(&refs(&names), 0, ""),
            LayerType::MlaWithMoe
        );
    }

    #[test]
    fn test_detect_format_huggingface() {
        let names = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ];
        assert_eq!(detect_format(&names), ModelFormat::HuggingFace);
    }

    #[test]
    fn test_detect_format_oxidizr() {
        let names = ["embed_tokens.weight", "layers.0.self_attn.q_proj.weight"];
        assert_eq!(detect_format(&names), ModelFormat::Oxidizr);
    }

    #[test]
    fn test_detect_layer_indices_ordered() {
        let names = [
            "layers.2.mlp.weight",
            "layers.0.mlp.weight",
            "layers.5.mlp.weight",
            "layers.1.mlp.weight",
        ];
        let indices = detect_layer_indices(&names, "");
        assert_eq!(indices, vec![0, 1, 2, 5]);
    }

    #[test]
    fn test_detect_layer_indices_huggingface_prefix() {
        let names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
        ];
        let indices = detect_layer_indices(&names, "model.");
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn test_detect_from_names_mamba2_only() {
        let names: Vec<String> = vec![
            "embed_tokens.weight".into(),
            "layers.0.mamba2.mixer.A_log".into(),
            "layers.0.mamba2.mixer.conv1d.weight".into(),
            "layers.1.mamba2.mixer.A_log".into(),
            "norm.weight".into(),
            "lm_head.weight".into(),
        ];
        let config = detect_architecture_from_names(&names).unwrap();
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.format, ModelFormat::Oxidizr);
        assert!(!config.tie_word_embeddings);
        assert!(config.layer_types.iter().all(|&t| t == LayerType::Mamba2));
    }

    #[test]
    fn test_detect_from_names_hybrid() {
        let names: Vec<String> = vec![
            "embed_tokens.weight".into(),
            "layers.0.mamba2.mixer.A_log".into(),
            "layers.1.self_attn.w_dkv.weight".into(),
            "layers.1.moe.gate.weight".into(),
            "norm.weight".into(),
            "lm_head.weight".into(),
        ];
        let config = detect_architecture_from_names(&names).unwrap();
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.layer_types[0], LayerType::Mamba2);
        assert_eq!(config.layer_types[1], LayerType::MlaWithMoe);
    }

    #[test]
    fn test_detect_from_names_no_layers_errors() {
        let names: Vec<String> = vec!["embed_tokens.weight".into(), "norm.weight".into()];
        assert!(detect_architecture_from_names(&names).is_err());
    }

    #[test]
    fn test_mamba3_takes_priority_over_mamba2() {
        let names = layer_names(0, &["mamba3.mixer.A_log", "mamba2.mixer.A_log"]);
        assert_eq!(detect_layer_type(&refs(&names), 0, ""), LayerType::Mamba3);
    }
}
