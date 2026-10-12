//! Converting a `HuggingFaceConfig` into boostr's `UniversalConfig`.
//! `model_type` inference lives in `super::model_type`.

use super::types::HuggingFaceConfig;
use crate::error::{Error, Result};
use crate::model::config::attention::{AttentionConfig, RopeScalingConfig};
use crate::model::config::audio::AudioConfig;
use crate::model::config::block::{BlockConfig, MlpActivation, NormKind, ResidualLayout};
use crate::model::config::universal::UniversalConfig;
use crate::model::config::vision::VisionConfig;
use serde_json::Value;

/// A number from a nested JSON object such as DBRX's `attn_config`.
fn nested_f64(block: Option<&Value>, key: &str) -> Option<f64> {
    block?.get(key)?.as_f64()
}

impl HuggingFaceConfig {
    /// The RoPE base: `rope_theta`, then a transformers v5
    /// `rope_parameters.rope_theta`, then DBRX `attn_config.rope_theta`,
    /// then 10000.
    fn resolved_rope_theta(&self) -> f32 {
        self.rope_theta
            .or_else(|| nested_f64(self.rope_parameters.as_ref(), "rope_theta").map(|v| v as f32))
            .or_else(|| nested_f64(self.attn_config.as_ref(), "rope_theta").map(|v| v as f32))
            .unwrap_or(10000.0)
    }

    /// The KV head count: DBRX `attn_config.kv_n_heads`, then
    /// `num_key_value_heads`, then 1 for a Falcon `multi_query` model.
    fn resolved_kv_heads(&self, model_type: &str) -> Option<usize> {
        if model_type == "dbrx" {
            return nested_f64(self.attn_config.as_ref(), "kv_n_heads").map(|v| v as usize);
        }
        if model_type == "falcon" {
            if self.new_decoder_architecture == Some(true) {
                return self.num_kv_heads;
            }
            if self.multi_query == Some(true) {
                return Some(1);
            }
            return None;
        }
        if self.multi_query == Some(true) && self.num_kv_heads.is_none() {
            return Some(1);
        }
        self.num_kv_heads
    }

    /// The MLP activation the config names, with each architecture's
    /// default when it names none.
    fn resolved_activation(&self, model_type: &str) -> Result<MlpActivation> {
        let dbrx_act = self
            .ffn_config
            .as_ref()
            .and_then(|f| f.get("ffn_act_fn"))
            .and_then(|a| a.get("name"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let named = match model_type {
            // Gemma names it `hidden_activation`. Old Gemma configs carry a
            // stale `hidden_act: gelu` that transformers ignores.
            "gemma" | "gemma2" => self
                .hidden_activation
                .clone()
                .or_else(|| Some("gelu_pytorch_tanh".to_string())),
            "dbrx" => dbrx_act.or_else(|| Some("silu".to_string())),
            // FalconMLP runs `nn.GELU()` whatever the config says.
            "falcon" => Some("gelu".to_string()),
            _ => self.hidden_act.clone(),
        };
        match named {
            Some(name) => MlpActivation::from_hf(&name),
            None => Ok(MlpActivation::Silu),
        }
    }

    /// The per-layer sliding flags, or `None` when every layer slides the
    /// same way.
    fn resolved_sliding_layers(&self, model_type: &str) -> Option<Vec<bool>> {
        if let Some(types) = &self.layer_types {
            let flags: Vec<bool> = types.iter().map(|t| t == "sliding_attention").collect();
            return (!flags.iter().all(|&f| f)).then_some(flags);
        }
        match model_type {
            // Gemma 2 slides on even layers.
            "gemma2" => Some((0..self.num_layers).map(|i| i % 2 == 0).collect()),
            "qwen2" | "qwen2_moe" => {
                let from = self.max_window_layers.unwrap_or(0);
                (from > 0).then(|| (0..self.num_layers).map(|i| i >= from).collect())
            }
            _ => None,
        }
    }

    /// The sliding window, `None` when the config turns it off.
    fn resolved_sliding_window(&self, model_type: &str) -> Option<usize> {
        if self.use_sliding_window == Some(false) {
            return None;
        }
        if let Some(types) = &self.layer_types
            && !types.iter().any(|t| t == "sliding_attention")
        {
            return None;
        }
        match model_type {
            // Gemma (v1) and Falcon have no window.
            "gemma" | "falcon" => None,
            _ => self.sliding_window.filter(|&w| w > 0),
        }
    }

    /// The block shape of `model_type`.
    fn block_config(&self, model_type: &str, head_dim: usize) -> Result<BlockConfig> {
        let mut block = BlockConfig {
            mlp_activation: self.resolved_activation(model_type)?,
            ..BlockConfig::default()
        };
        if let Some(f) = self.partial_rotary_factor.or(self.rotary_pct) {
            block.partial_rotary_factor = f;
        }
        match model_type {
            "gemma" | "gemma2" => {
                block.norm_weight_offset = 1.0;
                block.embedding_multiplier = Some((self.hidden_size as f32).sqrt());
                if let Some(s) = self.query_pre_attn_scalar {
                    block.attention_scale = Some(1.0 / s.sqrt());
                }
                block.attn_logit_softcap = self.attn_logit_softcapping;
                block.final_logit_softcap = self.final_logit_softcapping;
            }
            "starcoder2" => block.norm = NormKind::Layer,
            "gpt_neox" => {
                block.norm = NormKind::Layer;
                if self.use_parallel_residual.unwrap_or(true) {
                    block.residual = ResidualLayout::Parallel;
                }
                // GPTNeoXConfig defaults `rotary_pct` to 0.25.
                block.partial_rotary_factor = self.rotary_pct.unwrap_or(0.25);
            }
            "falcon" => {
                block.norm = NormKind::Layer;
                block.residual = if self.new_decoder_architecture == Some(true) {
                    ResidualLayout::Parallel
                } else if self.parallel_attn.unwrap_or(true) {
                    ResidualLayout::ParallelSharedNorm
                } else {
                    ResidualLayout::Sequential
                };
                if self.alibi == Some(true) {
                    // Falcon adds `alibi / sqrt(head_dim)` to the scaled scores.
                    block.alibi_bias_scale = Some(1.0 / (head_dim as f32).sqrt());
                }
            }
            "dbrx" => {
                block.norm = NormKind::Layer;
                block.clip_qkv =
                    nested_f64(self.attn_config.as_ref(), "clip_qkv").map(|v| v as f32);
            }
            "cohere" | "command_r" => {
                if self.use_qk_norm == Some(true) {
                    return Err(Error::ModelError {
                        reason: "Cohere use_qk_norm: true needs per-head LayerNorm weights, \
                                 which boostr does not load; this checkpoint is refused"
                            .into(),
                    });
                }
                block.norm = NormKind::Layer;
                block.residual = ResidualLayout::ParallelSharedNorm;
                block.rope_interleaved = true;
                block.logit_scale = self.logit_scale;
            }
            _ => {}
        }
        block.sliding_window_layers = self.resolved_sliding_layers(model_type);
        Ok(block)
    }

    /// Norm epsilon: the LayerNorm key for LayerNorm architectures, else
    /// `rms_norm_eps`.
    fn resolved_norm_eps(&self, block: &BlockConfig) -> f64 {
        match block.norm {
            NormKind::Layer => self.layer_norm_eps.unwrap_or(1e-5),
            NormKind::Rms => self.rms_norm_eps,
        }
    }

    /// `tie_word_embeddings`, with the architecture default when absent.
    fn resolved_tie(&self, model_type: &str) -> bool {
        self.tie_word_embeddings.unwrap_or(matches!(
            model_type,
            "gemma" | "gemma2" | "cohere" | "command_r" | "falcon" | "starcoder2"
        ))
    }

    /// The RoPE scaling config. LongRoPE gets its attention factor from
    /// `max_position_embeddings / original_max_position_embeddings`, as
    /// transformers computes it, so the cache length cannot change it.
    fn resolved_rope_scaling(&self) -> Option<RopeScalingConfig> {
        let rs = self.rope_scaling.as_ref()?;
        let scaling_type = rs.scaling_type.clone()?;
        if scaling_type == "default" {
            return None;
        }
        let original_max = rs
            .original_max_position_embeddings
            .or(self.original_max_position_embeddings);
        let mut attention_factor = rs.attention_factor;
        if scaling_type == "longrope"
            && attention_factor.is_none()
            && let Some(original) = original_max.filter(|&o| o > 1)
        {
            let factor = self.max_seq_len as f64 / original as f64;
            attention_factor = Some(if factor <= 1.0 {
                1.0
            } else {
                (1.0 + factor.ln() / (original as f64).ln()).sqrt() as f32
            });
        }
        Some(RopeScalingConfig {
            scaling_type,
            factor: rs.factor.unwrap_or(1.0),
            original_max_position_embeddings: original_max,
            low_freq_factor: rs.low_freq_factor,
            high_freq_factor: rs.high_freq_factor,
            attention_factor,
            beta_fast: rs.beta_fast,
            beta_slow: rs.beta_slow,
            short_factor: rs.short_factor.clone(),
            long_factor: rs.long_factor.clone(),
        })
    }

    /// The MoE config: Mixtral and Qwen2-MoE keys, or DBRX `ffn_config`.
    fn resolved_moe(&self, model_type: &str) -> Option<crate::model::config::moe::MoeConfig> {
        let (num_experts, top_k, renorm, inter) = if model_type == "dbrx" {
            let ffn = self.ffn_config.as_ref()?;
            let experts = ffn.get("moe_num_experts")?.as_u64()? as usize;
            let top_k = ffn.get("moe_top_k").and_then(Value::as_u64).unwrap_or(1) as usize;
            // `moe_normalize_expert_weights` is the p of an Lp norm over the
            // top-k weights; p = 1 equals sum renormalization.
            let renorm = ffn
                .get("moe_normalize_expert_weights")
                .and_then(Value::as_f64)
                .is_some();
            let inter = ffn
                .get("ffn_hidden_size")
                .and_then(Value::as_u64)
                .map(|v| v as usize);
            (experts, top_k, Some(renorm), inter)
        } else {
            (
                self.num_local_experts?,
                self.num_experts_per_tok.unwrap_or(2),
                self.norm_topk_prob,
                self.moe_intermediate_size,
            )
        };
        Some(crate::model::config::moe::MoeConfig {
            num_experts,
            experts_per_tok: top_k,
            shared_expert: None,
            intermediate_size: inter,
            load_balance_alpha: 0.01,
            z_loss_alpha: 1e-3,
            norm_topk_prob: renorm,
        })
    }

    /// Fail on DBRX settings boostr does not run.
    fn check_unsupported(&self, model_type: &str) -> Result<()> {
        if model_type == "dbrx"
            && let Some(p) = nested_f64(self.ffn_config.as_ref(), "moe_normalize_expert_weights")
            && (p - 1.0).abs() > f64::EPSILON
        {
            return Err(Error::ModelError {
                reason: format!(
                    "DBRX moe_normalize_expert_weights {p} is an L{p} norm; boostr renormalizes \
                     with p = 1 only"
                ),
            });
        }
        Ok(())
    }

    /// Convert to UniversalConfig
    ///
    /// # Errors
    ///
    /// An activation, RoPE scaling or architecture option boostr does not
    /// run, named in the error.
    pub fn to_universal(&self) -> Result<UniversalConfig> {
        let model_type = self.infer_model_type();
        self.check_unsupported(&model_type)?;
        let effective_kv_heads = self.resolved_kv_heads(&model_type);
        let head_dim = self.head_dim.unwrap_or_else(|| {
            self.num_attention_heads
                .map_or(0, |h| self.hidden_size / h.max(1))
        });
        let block = self.block_config(&model_type, head_dim)?;
        let rms_norm_eps = self.resolved_norm_eps(&block);

        let attention = self.num_attention_heads.map(|num_heads| AttentionConfig {
            num_heads,
            num_kv_heads: effective_kv_heads,
            head_dim: self.head_dim,
            rope_theta: self.resolved_rope_theta(),
            rope_scaling: self.resolved_rope_scaling(),
            kv_latent_dim: None,
            q_latent_dim: None,
            d_rope: None,
            sliding_window: self.resolved_sliding_window(&model_type),
            use_alibi: self.alibi.unwrap_or(false),
        });

        let moe = self.resolved_moe(&model_type);

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

        let intermediate_size = if model_type == "dbrx" {
            moe.as_ref().and_then(|m| m.intermediate_size)
        } else {
            self.intermediate_size
        };

        Ok(UniversalConfig {
            tie_word_embeddings: self.resolved_tie(&model_type),
            model_type,
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            max_seq_len: self.max_seq_len,
            intermediate_size,
            rms_norm_eps,
            attention,
            ssm: None,
            moe,
            hybrid_layers: None,
            grow_vocab: false,
            vision,
            audio,
            gdn: None,
            qwen35_attention: None,
            hadamard: None,
            block,
            quant_formats: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::config_with_model_type;
    use super::super::types::HuggingFaceConfig;
    use crate::model::config::{MlpActivation, NormKind, ResidualLayout};

    // -- MoE auto-detection from HF fields --

    #[test]
    fn moe_auto_detection_from_num_local_experts() {
        let mut c = config_with_model_type("mixtral");
        c.num_local_experts = Some(8);
        c.num_experts_per_tok = Some(2);
        let uc = c.to_universal().unwrap();
        let moe = uc.moe.as_ref().expect("MoE config should be populated");
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.experts_per_tok, 2);
    }

    #[test]
    fn no_moe_when_field_absent() {
        let c = config_with_model_type("llama");
        let uc = c.to_universal().unwrap();
        assert!(uc.moe.is_none());
    }

    #[test]
    fn multi_query_sets_one_kv_head() {
        let mut c = config_with_model_type("falcon");
        c.multi_query = Some(true);
        let uc = c.to_universal().unwrap();
        let attn = uc.attention.as_ref().unwrap();
        assert_eq!(attn.num_kv_heads, Some(1));
    }

    #[test]
    fn alibi_propagated_to_universal() {
        let mut c = config_with_model_type("falcon");
        c.alibi = Some(true);
        let uc = c.to_universal().unwrap();
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
        let uc = hf.to_universal().unwrap();
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
        let uc = c.to_universal().unwrap();
        assert!(uc.vision.is_none());
        assert!(uc.audio.is_none());
    }

    #[test]
    fn alibi_defaults_to_false() {
        let c = config_with_model_type("llama");
        let uc = c.to_universal().unwrap();
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
        let uc = hf.to_universal().unwrap();
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
        let uc = hf.to_universal().unwrap();
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
        let uc = c.to_universal().unwrap();
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
        let uc = c.to_universal().unwrap();
        assert!(uc.audio.is_none());
    }

    // -- Architecture block mapping --

    fn parse(json: &str) -> crate::model::config::universal::UniversalConfig {
        let hf: HuggingFaceConfig = serde_json::from_str(json).unwrap();
        hf.to_universal().unwrap()
    }

    #[test]
    fn qwen2_window_is_off_unless_enabled() {
        let uc = parse(
            r#"{"model_type": "qwen2", "vocab_size": 8, "hidden_size": 8,
                "num_hidden_layers": 2, "max_position_embeddings": 64,
                "num_attention_heads": 2, "sliding_window": 32,
                "use_sliding_window": false}"#,
        );
        assert_eq!(uc.attention.unwrap().sliding_window, None);
        assert!(uc.block.is_llama());
        assert!(!uc.tie_word_embeddings);
    }

    #[test]
    fn gemma2_block_carries_softcaps_and_alternating_windows() {
        let uc = parse(
            r#"{"model_type": "gemma2", "vocab_size": 8, "hidden_size": 16,
                "num_hidden_layers": 3, "max_position_embeddings": 64,
                "num_attention_heads": 2, "head_dim": 8, "sliding_window": 4,
                "query_pre_attn_scalar": 64, "attn_logit_softcapping": 50.0,
                "final_logit_softcapping": 30.0,
                "hidden_activation": "gelu_pytorch_tanh", "hidden_act": "gelu"}"#,
        );
        let b = &uc.block;
        assert_eq!(b.norm_weight_offset, 1.0);
        assert_eq!(b.mlp_activation, MlpActivation::GeluTanh);
        assert_eq!(b.embedding_multiplier, Some(4.0));
        assert_eq!(b.attention_scale, Some(0.125));
        assert_eq!(b.attn_logit_softcap, Some(50.0));
        assert_eq!(b.final_logit_softcap, Some(30.0));
        assert_eq!(b.sliding_window_layers, Some(vec![true, false, true]));
        assert!(uc.tie_word_embeddings, "Gemma ties by default");
        assert_eq!(uc.attention.unwrap().sliding_window, Some(4));
    }

    #[test]
    fn gpt_neox_block_is_parallel_layernorm_with_partial_rope() {
        let uc = parse(
            r#"{"model_type": "gpt_neox", "vocab_size": 8, "hidden_size": 16,
                "num_hidden_layers": 1, "max_position_embeddings": 64,
                "num_attention_heads": 2, "rotary_pct": 0.25,
                "rotary_emb_base": 5000, "layer_norm_eps": 1e-5,
                "hidden_act": "gelu", "use_parallel_residual": true}"#,
        );
        assert_eq!(uc.block.norm, NormKind::Layer);
        assert_eq!(uc.block.residual, ResidualLayout::Parallel);
        assert_eq!(uc.block.partial_rotary_factor, 0.25);
        assert_eq!(uc.block.mlp_activation, MlpActivation::GeluErf);
        assert_eq!(uc.attention.unwrap().rope_theta, 5000.0);
    }

    #[test]
    fn falcon_rw_reads_falcon_keys_and_scales_alibi() {
        let uc = parse(
            r#"{"model_type": "falcon", "vocab_size": 8, "hidden_size": 16,
                "n_layer": 2, "n_head": 4, "alibi": true, "bias": true,
                "multi_query": false, "parallel_attn": false,
                "layer_norm_epsilon": 1e-5}"#,
        );
        assert_eq!(uc.num_layers, 2);
        assert_eq!(uc.max_seq_len, 2048);
        assert_eq!(uc.block.residual, ResidualLayout::Sequential);
        assert_eq!(uc.block.alibi_bias_scale, Some(0.5));
        let attn = uc.attention.unwrap();
        assert!(attn.use_alibi);
        assert_eq!(attn.num_kv_heads, None);
        assert!(uc.tie_word_embeddings);
    }

    #[test]
    fn dbrx_reads_nested_attn_and_ffn_configs() {
        let uc = parse(
            r#"{"model_type": "dbrx", "vocab_size": 8, "d_model": 24, "n_heads": 6,
                "n_layers": 2, "max_seq_len": 64, "num_key_value_heads": 8,
                "attn_config": {"kv_n_heads": 2, "clip_qkv": 8, "rope_theta": 500000},
                "ffn_config": {"ffn_hidden_size": 24, "moe_num_experts": 16,
                               "moe_top_k": 4, "moe_normalize_expert_weights": 1.0,
                               "ffn_act_fn": {"name": "silu"}}}"#,
        );
        let attn = uc.attention.as_ref().unwrap();
        assert_eq!(attn.num_kv_heads, Some(2));
        assert_eq!(attn.rope_theta, 500000.0);
        assert_eq!(uc.block.clip_qkv, Some(8.0));
        assert_eq!(uc.block.norm, NormKind::Layer);
        let moe = uc.moe.unwrap();
        assert_eq!((moe.num_experts, moe.experts_per_tok), (16, 4));
        assert!(moe.renormalizes_top_k());
        assert_eq!(uc.intermediate_size, Some(24));
    }

    #[test]
    fn cohere_is_parallel_interleaved_and_refuses_qk_norm() {
        let json = r#"{"model_type": "cohere", "vocab_size": 8, "hidden_size": 8,
                "num_hidden_layers": 1, "max_position_embeddings": 64,
                "num_attention_heads": 4, "logit_scale": 0.125,
                "use_qk_norm": false}"#;
        let uc = parse(json);
        assert_eq!(uc.block.residual, ResidualLayout::ParallelSharedNorm);
        assert!(uc.block.rope_interleaved);
        assert_eq!(uc.block.logit_scale, Some(0.125));
        assert!(uc.tie_word_embeddings);
        let hf: HuggingFaceConfig = serde_json::from_str(&json.replace("false", "true")).unwrap();
        assert!(hf.to_universal().is_err());
    }

    #[test]
    fn v5_rope_parameters_give_the_theta() {
        let uc = parse(
            r#"{"model_type": "mixtral", "vocab_size": 8, "hidden_size": 8,
                "num_hidden_layers": 1, "max_position_embeddings": 64,
                "num_attention_heads": 2, "head_dim": null,
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
                "num_local_experts": 4, "num_experts_per_tok": 2}"#,
        );
        assert_eq!(uc.attention.unwrap().rope_theta, 10000.0);
    }

    #[test]
    fn longrope_attention_factor_follows_the_config_lengths() {
        let uc = parse(
            r#"{"model_type": "phi3", "vocab_size": 8, "hidden_size": 8,
                "num_hidden_layers": 1, "max_position_embeddings": 131072,
                "original_max_position_embeddings": 4096,
                "num_attention_heads": 2,
                "rope_scaling": {"type": "longrope", "short_factor": [1.0, 1.0],
                                 "long_factor": [2.0, 2.0]}}"#,
        );
        let rs = uc.attention.unwrap().rope_scaling.unwrap();
        assert_eq!(rs.original_max_position_embeddings, Some(4096));
        let want = (1.0 + (32.0f64).ln() / (4096.0f64).ln()).sqrt() as f32;
        assert!((rs.attention_factor.unwrap() - want).abs() < 1e-6);
    }
}
