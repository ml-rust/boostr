//! Hugging Face Qwen3.5 tensor names to the names the `qwen35` loader reads.
//!
//! | Hugging Face | `qwen35` loader |
//! | --- | --- |
//! | `model.language_model.*` | `model.*` |
//! | `linear_attn.in_proj_a.weight` | `linear_attn.alpha_proj.weight` |
//! | `linear_attn.in_proj_b.weight` | `linear_attn.beta_proj.weight` |
//! | `linear_attn.A_log` | `linear_attn.a_neg_exp` |
//! | `mtp.*`, `model.mtp.*`, `model.visual.*` | dropped |
//!
//! Every other name is the same in both. The `A_log` rename marks the slot
//! only: the value becomes `-exp(A_log)` in `restructure`.

/// Prefixes of tensors the text model does not read: the multi-token
/// prediction head and the vision tower.
const DROPPED_PREFIXES: [&str; 3] = ["mtp.", "model.mtp.", "model.visual."];

/// Wrapper prefix of the text model in `Qwen3_5ForConditionalGeneration`.
const LANGUAGE_MODEL_PREFIX: &str = "model.language_model.";

/// Suffix renames, Hugging Face to `qwen35` loader.
const SUFFIX_RENAMES: [(&str, &str); 3] = [
    (
        ".linear_attn.in_proj_a.weight",
        ".linear_attn.alpha_proj.weight",
    ),
    (
        ".linear_attn.in_proj_b.weight",
        ".linear_attn.beta_proj.weight",
    ),
    (".linear_attn.A_log", ".linear_attn.a_neg_exp"),
];

/// `true` for a tensor the text model does not read.
pub fn is_dropped(name: &str) -> bool {
    DROPPED_PREFIXES.iter().any(|p| name.starts_with(p))
}

/// The `qwen35` loader name of the Hugging Face tensor `name`, or `None`
/// when the text model does not read it.
pub fn qwen35_name(name: &str) -> Option<String> {
    if is_dropped(name) {
        return None;
    }
    let name = match name.strip_prefix(LANGUAGE_MODEL_PREFIX) {
        Some(rest) => format!("model.{rest}"),
        None => name.to_string(),
    };
    for (from, to) in SUFFIX_RENAMES {
        if let Some(stem) = name.strip_suffix(from) {
            return Some(format!("{stem}{to}"));
        }
    }
    Some(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapped(name: &str) -> String {
        qwen35_name(name).unwrap()
    }

    #[test]
    fn wrapper_prefix_strips_to_model() {
        assert_eq!(
            mapped("model.language_model.embed_tokens.weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            mapped("model.language_model.layers.3.self_attn.q_norm.weight"),
            "model.layers.3.self_attn.q_norm.weight"
        );
        assert_eq!(mapped("lm_head.weight"), "lm_head.weight");
    }

    #[test]
    fn gate_and_decay_tensors_take_loader_names() {
        let p = "model.language_model.layers.0.linear_attn";
        let q = "model.layers.0.linear_attn";
        assert_eq!(
            mapped(&format!("{p}.in_proj_a.weight")),
            format!("{q}.alpha_proj.weight")
        );
        assert_eq!(
            mapped(&format!("{p}.in_proj_b.weight")),
            format!("{q}.beta_proj.weight")
        );
        assert_eq!(mapped(&format!("{p}.A_log")), format!("{q}.a_neg_exp"));
        assert_eq!(mapped(&format!("{p}.dt_bias")), format!("{q}.dt_bias"));
        assert_eq!(
            mapped(&format!("{p}.in_proj_qkv.weight")),
            format!("{q}.in_proj_qkv.weight")
        );
    }

    /// A text-only `Qwen3_5ForCausalLM` checkpoint has no wrapper prefix.
    #[test]
    fn unwrapped_names_rename_too() {
        assert_eq!(
            mapped("model.layers.2.linear_attn.in_proj_a.weight"),
            "model.layers.2.linear_attn.alpha_proj.weight"
        );
        assert_eq!(
            mapped("model.layers.2.input_layernorm.weight"),
            "model.layers.2.input_layernorm.weight"
        );
    }

    #[test]
    fn mtp_and_vision_tensors_drop() {
        assert_eq!(qwen35_name("mtp.fc.weight"), None);
        assert_eq!(qwen35_name("mtp.layers.0.self_attn.q_proj.weight"), None);
        assert_eq!(qwen35_name("model.mtp.norm.weight"), None);
        assert_eq!(qwen35_name("model.visual.patch_embed.proj.weight"), None);
        assert!(!is_dropped("model.language_model.norm.weight"));
    }
}
