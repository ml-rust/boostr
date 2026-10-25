//! [`unsupported_model_refusal`]: the one error for a model type boostr
//! names but has no model for.

/// Why `model_type` loads no model, or `None` when it is not refused.
///
/// DeepSeek-V2 and V3 (`deepseek_v2`, `deepseek_v3`, and GGUF `deepseek2`)
/// pair MLA with DeepSeek's own MoE routing and layer layout. boostr has the
/// MLA block, not the model. The GGUF `deepseek` architecture is the
/// Llama-layout DeepSeek and stays loadable.
pub fn unsupported_model_refusal(model_type: &str) -> Option<String> {
    matches!(model_type, "deepseek_v2" | "deepseek_v3" | "deepseek2").then(|| {
        format!("model type '{model_type}' is not supported yet: no DeepSeek model implementation")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek_v2_and_v3_are_refused() {
        for model_type in ["deepseek_v2", "deepseek_v3", "deepseek2"] {
            let reason = unsupported_model_refusal(model_type).expect(model_type);
            assert!(
                reason.contains("no DeepSeek model implementation"),
                "{reason}"
            );
            assert!(reason.contains(model_type), "{reason}");
        }
        for model_type in ["deepseek", "llama", "hybrid"] {
            assert!(
                unsupported_model_refusal(model_type).is_none(),
                "{model_type}"
            );
        }
    }
}
