//! PEFT adapter tensor keys: split a key into its module and LoRA factor, and
//! normalize the module to the base checkpoint's `model.layers.N.<sub>` name.

use crate::error::{Error, Result};

/// Which LoRA factor a tensor holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraPart {
    /// `lora_A`: `[r, in_features]`.
    A,
    /// `lora_B`: `[out_features, r]`.
    B,
}

/// One adapter tensor key, split and normalized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterKey {
    /// Base-checkpoint module name, e.g. `model.layers.3.self_attn.q_proj`.
    pub module: String,
    /// The module name PEFT matched patterns against: the key without its
    /// `base_model.model.` prefix and LoRA suffix, e.g.
    /// `model.language_model.layers.3.self_attn.q_proj` or, for an adapter
    /// trained on a bare text model, `layers.3.self_attn.q_proj`.
    pub peft_module: String,
    /// The LoRA factor.
    pub part: LoraPart,
}

const PEFT_PREFIX: &str = "base_model.model.";
const LAYER_PREFIXES: [&str; 3] = ["model.layers.", "model.language_model.layers.", "layers."];

/// Parse a PEFT adapter tensor key.
///
/// Accepts `base_model.model.<module>.lora_A.weight` and `lora_B`, with an
/// optional adapter-name segment before `weight` (`lora_A.default.weight`).
/// `<module>` must start with `model.layers.N.`,
/// `model.language_model.layers.N.`, or `layers.N.` (an adapter trained on a
/// bare text model). All three normalize to `model.layers.N.<sub>`.
pub fn parse_adapter_key(key: &str) -> Result<AdapterKey> {
    let not_lora = || Error::ModelError {
        reason: format!(
            "adapter tensor `{key}` is not a LoRA factor: expected \
             `<module>.lora_A[.<adapter>].weight` or `<module>.lora_B[.<adapter>].weight`"
        ),
    };
    let body = key.strip_suffix(".weight").ok_or_else(not_lora)?;
    let segments: Vec<&str> = body.split('.').collect();
    let (part, module_len) = match segments.as_slice() {
        [.., last] if lora_part(last).is_some() => (lora_part(last), segments.len() - 1),
        [.., factor, adapter] if !adapter.is_empty() && lora_part(factor).is_some() => {
            (lora_part(factor), segments.len() - 2)
        }
        _ => (None, 0),
    };
    let part = part.ok_or_else(not_lora)?;
    let full_module = segments[..module_len].join(".");

    let peft_module = full_module
        .strip_prefix(PEFT_PREFIX)
        .ok_or_else(|| Error::ModelError {
            reason: format!(
                "adapter tensor `{key}` has an unknown prefix: PEFT keys start with \
                 `{PEFT_PREFIX}`"
            ),
        })?;
    let module = normalize_module(peft_module).ok_or_else(|| Error::ModelError {
        reason: format!(
            "adapter tensor `{key}` has an unknown module prefix: expected \
             `{PEFT_PREFIX}model.layers.N.<module>`, \
             `{PEFT_PREFIX}model.language_model.layers.N.<module>` or \
             `{PEFT_PREFIX}layers.N.<module>`"
        ),
    })?;
    Ok(AdapterKey {
        module,
        peft_module: peft_module.to_string(),
        part,
    })
}

fn lora_part(segment: &str) -> Option<LoraPart> {
    match segment {
        "lora_A" => Some(LoraPart::A),
        "lora_B" => Some(LoraPart::B),
        _ => None,
    }
}

/// `model.layers.N.<sub>` for a PEFT module under any layer prefix, if
/// `N` is a layer index and `<sub>` is a non-empty dotted path.
fn normalize_module(peft_module: &str) -> Option<String> {
    let rest = LAYER_PREFIXES
        .iter()
        .find_map(|prefix| peft_module.strip_prefix(prefix))?;
    let (index, sub) = rest.split_once('.')?;
    let index_ok = !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit());
    let sub_ok = sub.split('.').all(|s| !s.is_empty());
    (index_ok && sub_ok).then(|| format!("model.layers.{index}.{sub}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(key: &str) -> AdapterKey {
        parse_adapter_key(key).unwrap()
    }

    #[test]
    fn plain_causal_lm_key() {
        let k = parsed("base_model.model.model.layers.4.self_attn.q_proj.lora_A.weight");
        assert_eq!(k.module, "model.layers.4.self_attn.q_proj");
        assert_eq!(k.peft_module, "model.layers.4.self_attn.q_proj");
        assert_eq!(k.part, LoraPart::A);
    }

    #[test]
    fn conditional_generation_key() {
        let k =
            parsed("base_model.model.model.language_model.layers.11.mlp.down_proj.lora_B.weight");
        assert_eq!(k.module, "model.layers.11.mlp.down_proj");
        assert_eq!(
            k.peft_module,
            "model.language_model.layers.11.mlp.down_proj"
        );
        assert_eq!(k.part, LoraPart::B);
    }

    #[test]
    fn bare_text_model_key() {
        let k = parsed("base_model.model.layers.7.linear_attn.in_proj_a.lora_A.weight");
        assert_eq!(k.module, "model.layers.7.linear_attn.in_proj_a");
        assert_eq!(k.peft_module, "layers.7.linear_attn.in_proj_a");
        assert_eq!(k.part, LoraPart::A);
        let b = parsed("base_model.model.layers.0.mlp.gate_proj.lora_B.default.weight");
        assert_eq!(b.module, "model.layers.0.mlp.gate_proj");
        assert_eq!(b.peft_module, "layers.0.mlp.gate_proj");
        assert_eq!(b.part, LoraPart::B);
    }

    #[test]
    fn adapter_name_infix() {
        let a = parsed(
            "base_model.model.model.layers.0.linear_attn.in_proj_qkvz.lora_A.default.weight",
        );
        assert_eq!(a.module, "model.layers.0.linear_attn.in_proj_qkvz");
        assert_eq!(a.part, LoraPart::A);
        let b = parsed(
            "base_model.model.model.language_model.layers.2.self_attn.o_proj.lora_B.default.weight",
        );
        assert_eq!(b.module, "model.layers.2.self_attn.o_proj");
        assert_eq!(b.part, LoraPart::B);
    }

    #[test]
    fn non_lora_keys_error() {
        for key in [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.bias",
            "base_model.model.model.embed_tokens.lora_embedding_A",
            "base_model.model.model.layers.0.self_attn.q_proj.weight",
        ] {
            let text = parse_adapter_key(key).unwrap_err().to_string();
            assert!(text.contains("is not a LoRA factor"), "{key}: {text}");
            assert!(text.contains(key), "{text}");
        }
    }

    #[test]
    fn unknown_prefixes_error() {
        for key in [
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.lm_head.lora_A.weight",
            "base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight",
            "base_model.model.model.layers.x.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.3.lora_A.weight",
            "base_model.model.model.layers.3..q_proj.lora_A.weight",
            "base_model.model.layers.x.self_attn.q_proj.lora_A.weight",
            "base_model.model.layers.3.lora_A.weight",
            "base_model.model.model.model.layers.3.self_attn.q_proj.lora_A.weight",
            "base_model.model.language_model.layers.3.self_attn.q_proj.lora_A.weight",
        ] {
            let text = parse_adapter_key(key).unwrap_err().to_string();
            assert!(text.contains("unknown"), "{key}: {text}");
            assert!(text.contains(key), "{text}");
        }
    }
}
