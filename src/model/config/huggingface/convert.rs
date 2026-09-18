//! Converting a `HuggingFaceConfig` into boostr's `UniversalConfig`.
//! `model_type` inference lives in `super::model_type`.

use super::types::HuggingFaceConfig;
use crate::model::config::attention::{AttentionConfig, RopeScalingConfig};
use crate::model::config::audio::AudioConfig;
use crate::model::config::universal::UniversalConfig;
use crate::model::config::vision::VisionConfig;

impl HuggingFaceConfig {
    /// Convert to UniversalConfig
    pub fn to_universal(&self) -> UniversalConfig {
        let model_type = self.infer_model_type();

        // Determine num_kv_heads: Falcon multi_query means 1 KV head
        let effective_kv_heads = if self.multi_query == Some(true) && self.num_kv_heads.is_none() {
            Some(1)
        } else {
            self.num_kv_heads
        };

        let attention = self.num_attention_heads.map(|num_heads| {
            let rope_scaling = self.rope_scaling.as_ref().and_then(|rs| {
                rs.scaling_type.as_ref().map(|t| RopeScalingConfig {
                    scaling_type: t.clone(),
                    factor: rs.factor.unwrap_or(1.0),
                    original_max_position_embeddings: rs.original_max_position_embeddings,
                    low_freq_factor: rs.low_freq_factor,
                    high_freq_factor: rs.high_freq_factor,
                    attention_factor: rs.attention_factor,
                    beta_fast: rs.beta_fast,
                    beta_slow: rs.beta_slow,
                    short_factor: rs.short_factor.clone(),
                    long_factor: rs.long_factor.clone(),
                })
            });

            AttentionConfig {
                num_heads,
                num_kv_heads: effective_kv_heads,
                head_dim: self.head_dim,
                rope_theta: self.rope_theta,
                rope_scaling,
                kv_latent_dim: None,
                q_latent_dim: None,
                d_rope: None,
                sliding_window: self.sliding_window,
                use_alibi: self.alibi.unwrap_or(false),
            }
        });

        // Auto-detect MoE from num_local_experts
        let moe = self
            .num_local_experts
            .map(|num_experts| crate::model::config::moe::MoeConfig {
                num_experts,
                experts_per_tok: self.num_experts_per_tok.unwrap_or(2),
                shared_expert: None,
                intermediate_size: None,
                load_balance_alpha: 0.01,
                z_loss_alpha: 1e-3,
            });

        // Parse vision_config JSON into VisionConfig if present
        let vision = self
            .vision_config
            .as_ref()
            .and_then(|vc| serde_json::from_value::<VisionConfig>(vc.clone()).ok());

        // Parse audio_config if present (Ultravox, Qwen2-Audio, Qwen2.5-Omni)
        let audio = self.audio_config.as_ref().map(|audio_cfg| AudioConfig {
            encoder_type: audio_cfg
                .get("model_type")
                .and_then(|v| v.as_str())
                .unwrap_or("whisper")
                .to_string(),
            hidden_size: audio_cfg
                .get("d_model")
                .or_else(|| audio_cfg.get("hidden_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(512) as usize,
            num_layers: audio_cfg
                .get("encoder_layers")
                .or_else(|| audio_cfg.get("num_hidden_layers"))
                .and_then(|v| v.as_u64())
                .unwrap_or(6) as usize,
            num_heads: audio_cfg
                .get("encoder_attention_heads")
                .or_else(|| audio_cfg.get("num_attention_heads"))
                .and_then(|v| v.as_u64())
                .unwrap_or(8) as usize,
            num_mel_bins: audio_cfg
                .get("num_mel_bins")
                .and_then(|v| v.as_u64())
                .unwrap_or(128) as usize,
            max_audio_len: audio_cfg
                .get("max_source_positions")
                .and_then(|v| v.as_u64())
                .unwrap_or(3000) as usize,
            projector_type: "linear".to_string(),
            vocab_size: audio_cfg
                .get("vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(51865) as usize,
            decoder_layers: audio_cfg
                .get("decoder_layers")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize),
            max_target_positions: audio_cfg
                .get("max_target_positions")
                .and_then(|v| v.as_u64())
                .unwrap_or(448) as usize,
            intermediate_size: audio_cfg
                .get("decoder_ffn_dim")
                .or_else(|| audio_cfg.get("encoder_ffn_dim"))
                .and_then(|v| v.as_u64())
                .map(|n| n as usize),
        });

        UniversalConfig {
            model_type,
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            max_seq_len: self.max_seq_len,
            intermediate_size: self.intermediate_size,
            rms_norm_eps: self.rms_norm_eps,
            attention,
            ssm: None,
            moe,
            hybrid_layers: None,
            tie_word_embeddings: self.tie_word_embeddings,
            grow_vocab: false,
            vision,
            audio,
            gdn: None,
            qwen35_attention: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::config_with_model_type;
    use super::super::types::HuggingFaceConfig;

    // -- MoE auto-detection from HF fields --

    #[test]
    fn moe_auto_detection_from_num_local_experts() {
        let mut c = config_with_model_type("mixtral");
        c.num_local_experts = Some(8);
        c.num_experts_per_tok = Some(2);
        let uc = c.to_universal();
        let moe = uc.moe.as_ref().expect("MoE config should be populated");
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.experts_per_tok, 2);
    }

    #[test]
    fn no_moe_when_field_absent() {
        let c = config_with_model_type("llama");
        let uc = c.to_universal();
        assert!(uc.moe.is_none());
    }

    #[test]
    fn multi_query_sets_one_kv_head() {
        let mut c = config_with_model_type("falcon");
        c.multi_query = Some(true);
        let uc = c.to_universal();
        let attn = uc.attention.as_ref().unwrap();
        assert_eq!(attn.num_kv_heads, Some(1));
    }

    #[test]
    fn alibi_propagated_to_universal() {
        let mut c = config_with_model_type("falcon");
        c.alibi = Some(true);
        let uc = c.to_universal();
        let attn = uc.attention.as_ref().unwrap();
        assert!(attn.use_alibi);
    }

    #[test]
    fn vision_config_parsed_from_hf() {
        let json = r#"{
            "model_type": "llava",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "max_position_embeddings": 4096,
            "num_attention_heads": 32,
            "vision_config": {
                "encoder_type": "clip",
                "hidden_size": 1024,
                "num_layers": 24,
                "num_heads": 16,
                "patch_size": 14,
                "image_size": 336,
                "intermediate_size": 4096
            }
        }"#;
        let hf: HuggingFaceConfig = serde_json::from_str(json).unwrap();
        let uc = hf.to_universal();
        assert_eq!(uc.model_type, "llava");
        let vision = uc.vision.as_ref().expect("vision config should be parsed");
        assert_eq!(vision.encoder_type, "clip");
        assert_eq!(vision.hidden_size, 1024);
        assert_eq!(vision.patch_size, 14);
        assert_eq!(vision.projector_type, "linear"); // default
    }

    #[test]
    fn no_vision_when_absent() {
        let c = config_with_model_type("llama");
        let uc = c.to_universal();
        assert!(uc.vision.is_none());
        assert!(uc.audio.is_none());
    }

    #[test]
    fn alibi_defaults_to_false() {
        let c = config_with_model_type("llama");
        let uc = c.to_universal();
        let attn = uc.attention.as_ref().unwrap();
        assert!(!attn.use_alibi);
    }

    // -- Audio-language model support --

    #[test]
    fn test_audio_config_ultravox() {
        let json = r#"{
            "model_type": "ultravox",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "max_position_embeddings": 4096,
            "num_attention_heads": 32,
            "audio_config": {
                "model_type": "whisper",
                "d_model": 512,
                "encoder_layers": 6,
                "encoder_attention_heads": 8,
                "num_mel_bins": 80,
                "max_source_positions": 1500
            }
        }"#;
        let hf: HuggingFaceConfig = serde_json::from_str(json).unwrap();
        let uc = hf.to_universal();
        // Ultravox maps to llama backbone
        assert_eq!(uc.model_type, "llama");
        assert!(uc.audio.is_some());
        let audio = uc.audio.unwrap();
        assert_eq!(audio.encoder_type, "whisper");
        assert_eq!(audio.hidden_size, 512);
        assert_eq!(audio.num_layers, 6);
        assert_eq!(audio.num_heads, 8);
        assert_eq!(audio.num_mel_bins, 80);
        assert_eq!(audio.max_audio_len, 1500);
    }

    #[test]
    fn test_audio_config_qwen2_audio() {
        let json = r#"{
            "model_type": "qwen2_audio",
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "max_position_embeddings": 32768,
            "num_attention_heads": 16,
            "audio_config": {
                "hidden_size": 1280,
                "num_hidden_layers": 32,
                "num_attention_heads": 20,
                "num_mel_bins": 128,
                "max_source_positions": 1500
            }
        }"#;
        let hf: HuggingFaceConfig = serde_json::from_str(json).unwrap();
        let uc = hf.to_universal();
        // qwen2_audio maps to qwen2 backbone
        assert_eq!(uc.model_type, "qwen2");
        assert!(uc.audio.is_some());
        let audio = uc.audio.unwrap();
        assert_eq!(audio.encoder_type, "whisper"); // default
        assert_eq!(audio.hidden_size, 1280);
        assert_eq!(audio.num_layers, 32);
        assert_eq!(audio.num_heads, 20);
        assert_eq!(audio.num_mel_bins, 128);
        assert_eq!(audio.max_audio_len, 1500);
    }

    #[test]
    fn test_audio_config_qwen25_omni() {
        let mut c = config_with_model_type("qwen2.5_omni");
        c.audio_config = Some(serde_json::json!({
            "d_model": 1024,
            "encoder_layers": 24,
            "encoder_attention_heads": 16,
            "num_mel_bins": 128
        }));
        let uc = c.to_universal();
        assert_eq!(uc.model_type, "qwen2");
        let audio = uc.audio.as_ref().unwrap();
        assert_eq!(audio.hidden_size, 1024);
        assert_eq!(audio.num_layers, 24);
        assert_eq!(audio.num_heads, 16);
        assert_eq!(audio.num_mel_bins, 128);
        assert_eq!(audio.max_audio_len, 3000); // default
    }

    #[test]
    fn no_audio_when_absent() {
        let c = config_with_model_type("llama");
        let uc = c.to_universal();
        assert!(uc.audio.is_none());
    }
}
