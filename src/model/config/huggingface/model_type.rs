//! `model_type` inference from `architectures[0]` when `model_type` is absent.

use super::types::HuggingFaceConfig;

impl HuggingFaceConfig {
    pub(super) fn infer_model_type(&self) -> String {
        if let Some(mt) = &self.model_type {
            // Map audio-language composite model types to their LLM backbone
            return match mt.as_str() {
                "ultravox" => "llama".to_string(),
                "qwen2_audio" | "qwen2.5_omni" => "qwen2".to_string(),
                _ => mt.clone(),
            };
        }
        if let Some(archs) = &self.architectures
            && let Some(arch) = archs.first()
        {
            let arch_lower = arch.to_lowercase();
            // Order matters: check specific variants before generic ones
            // (e.g. "qwen2moe" before "qwen", "phi3" before "phi",
            //  "gemma2" before "gemma").
            if arch_lower.contains("llava") {
                return "llava".to_string();
            } else if arch_lower.contains("qwen_vl") || arch_lower.contains("qwenvl") {
                return "qwen_vl".to_string();
            } else if arch_lower.contains("llama") {
                return "llama".to_string();
            } else if arch_lower.contains("mixtral") {
                return "mixtral".to_string();
            } else if arch_lower.contains("mistral") {
                return "mistral".to_string();
            } else if arch_lower.contains("mamba") {
                return "mamba2".to_string();
            } else if arch_lower.contains("qwen2moe") {
                return "qwen2_moe".to_string();
            } else if arch_lower.contains("qwen") {
                return "qwen2".to_string();
            } else if arch_lower.contains("phi3") {
                return "phi3".to_string();
            } else if arch_lower.contains("phi") {
                return "phi".to_string();
            } else if arch_lower.contains("gemma2") {
                return "gemma2".to_string();
            } else if arch_lower.contains("gemma") {
                return "gemma".to_string();
            } else if arch_lower.contains("starcoder") {
                return "starcoder2".to_string();
            } else if arch_lower.contains("internlm") {
                return "internlm2".to_string();
            } else if arch_lower.contains("falcon") {
                return "falcon".to_string();
            } else if arch_lower.contains("neox") || arch_lower.contains("pythia") {
                return "gpt_neox".to_string();
            } else if arch_lower.contains("dbrx") {
                return "dbrx".to_string();
            } else if arch_lower.contains("cohere") || arch_lower.contains("command") {
                return "command_r".to_string();
            }
        }
        "llama".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{config_with_arch, config_with_model_type};

    // -- model_type passthrough (primary path) --

    #[test]
    fn model_type_passthrough() {
        // When model_type is present, infer_model_type returns it verbatim.
        for mt in &[
            "llama",
            "mistral",
            "qwen2",
            "qwen2_moe",
            "phi3",
            "phi",
            "gemma",
            "gemma2",
            "starcoder2",
            "internlm2",
        ] {
            let c = config_with_model_type(mt);
            assert_eq!(c.infer_model_type(), *mt, "passthrough failed for {mt}");
        }
    }

    // -- architecture fallback (when model_type is absent) --
    // Uses real HuggingFace architectures[0] values.

    #[test]
    fn arch_fallback_llama() {
        // LlamaForCausalLM covers Llama, CodeLlama, Yi, Solar
        assert_eq!(
            config_with_arch("LlamaForCausalLM").infer_model_type(),
            "llama"
        );
    }

    #[test]
    fn arch_fallback_mistral() {
        assert_eq!(
            config_with_arch("MistralForCausalLM").infer_model_type(),
            "mistral"
        );
    }

    #[test]
    fn arch_fallback_qwen2() {
        assert_eq!(
            config_with_arch("Qwen2ForCausalLM").infer_model_type(),
            "qwen2"
        );
    }

    #[test]
    fn arch_fallback_qwen2_moe() {
        // Qwen2MoeForCausalLM must map to "qwen2_moe", NOT "qwen2"
        assert_eq!(
            config_with_arch("Qwen2MoeForCausalLM").infer_model_type(),
            "qwen2_moe"
        );
    }

    #[test]
    fn arch_fallback_phi3() {
        assert_eq!(
            config_with_arch("Phi3ForCausalLM").infer_model_type(),
            "phi3"
        );
    }

    #[test]
    fn arch_fallback_phi() {
        // Phi-2 uses "PhiForCausalLM", must map to "phi" not "phi3"
        assert_eq!(config_with_arch("PhiForCausalLM").infer_model_type(), "phi");
    }

    #[test]
    fn arch_fallback_gemma2() {
        assert_eq!(
            config_with_arch("Gemma2ForCausalLM").infer_model_type(),
            "gemma2"
        );
    }

    #[test]
    fn arch_fallback_gemma() {
        // Gemma (v1) must map to "gemma", not "gemma2"
        assert_eq!(
            config_with_arch("GemmaForCausalLM").infer_model_type(),
            "gemma"
        );
    }

    #[test]
    fn arch_fallback_starcoder2() {
        assert_eq!(
            config_with_arch("Starcoder2ForCausalLM").infer_model_type(),
            "starcoder2"
        );
    }

    #[test]
    fn arch_fallback_internlm2() {
        assert_eq!(
            config_with_arch("InternLM2ForCausalLM").infer_model_type(),
            "internlm2"
        );
    }

    #[test]
    fn arch_fallback_mamba() {
        assert_eq!(
            config_with_arch("MambaForCausalLM").infer_model_type(),
            "mamba2"
        );
    }

    #[test]
    fn arch_fallback_unknown_defaults_to_llama() {
        assert_eq!(
            config_with_arch("SomeNewModelForCausalLM").infer_model_type(),
            "llama"
        );
    }

    #[test]
    fn arch_fallback_falcon() {
        assert_eq!(
            config_with_arch("FalconForCausalLM").infer_model_type(),
            "falcon"
        );
    }

    #[test]
    fn arch_fallback_gpt_neox() {
        assert_eq!(
            config_with_arch("GPTNeoXForCausalLM").infer_model_type(),
            "gpt_neox"
        );
    }

    #[test]
    fn arch_fallback_dbrx() {
        assert_eq!(
            config_with_arch("DbrxForCausalLM").infer_model_type(),
            "dbrx"
        );
    }

    #[test]
    fn arch_fallback_mixtral() {
        assert_eq!(
            config_with_arch("MixtralForCausalLM").infer_model_type(),
            "mixtral"
        );
    }

    #[test]
    fn arch_fallback_command_r() {
        assert_eq!(
            config_with_arch("CohereForCausalLM").infer_model_type(),
            "command_r"
        );
    }

    #[test]
    fn arch_fallback_llava() {
        assert_eq!(
            config_with_arch("LlavaForConditionalGeneration").infer_model_type(),
            "llava"
        );
    }

    #[test]
    fn arch_fallback_qwen_vl() {
        assert_eq!(
            config_with_arch("QwenVLForConditionalGeneration").infer_model_type(),
            "qwen_vl"
        );
    }
}
