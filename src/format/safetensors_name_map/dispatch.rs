//! Entry point: routes a tensor name to the per-architecture normalizer, and
//! the audio-language prefix normalization shared by all architectures.

use super::dbrx::normalize_dbrx;
use super::falcon::normalize_falcon;
use super::gpt_neox::normalize_gpt_neox;

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
pub(super) fn split_layer_num(s: &str) -> Option<(usize, &str)> {
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
