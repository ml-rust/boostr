//! PEFT LoRA `adapter_config.json`: the fields an adapter reader needs,
//! refusal of variants that a plain `(A, B, scaling)` pair cannot express,
//! and per-module rank, alpha and scaling with `rank_pattern` and
//! `alpha_pattern` overrides.

use super::patterns::{OrderedMap, PatternOverride, first_match};
use crate::error::{Error, Result};
use serde::Deserialize;
use std::fmt;
use std::path::Path;

/// `target_modules` as PEFT stores it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetModules {
    /// Module names. PEFT matches each name exactly or as a `.`-separated suffix.
    Names(Vec<String>),
    /// One regex. PEFT full-matches it against the module name.
    Regex(String),
}

/// A parsed `adapter_config.json` for a plain LoRA adapter.
#[derive(Debug, Clone)]
pub struct PeftConfig {
    /// Default LoRA rank (`r`).
    pub r: usize,
    /// Default LoRA alpha (`lora_alpha`).
    pub lora_alpha: f32,
    /// Rank-stabilized scaling: `alpha / sqrt(r)` in place of `alpha / r`.
    pub use_rslora: bool,
    /// Per-module rank overrides, in file order.
    pub rank_pattern: Vec<PatternOverride<usize>>,
    /// Per-module alpha overrides, in file order.
    pub alpha_pattern: Vec<PatternOverride<f32>>,
    /// The modules the adapter was trained on.
    pub target_modules: Option<TargetModules>,
    /// The base checkpoint the adapter was trained against.
    pub base_model_name_or_path: Option<String>,
    /// PEFT task type, such as `CAUSAL_LM`.
    pub task_type: Option<String>,
}

impl PeftConfig {
    /// Read and parse `adapter_config.json` at `path`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| Error::ModelError {
            reason: format!("cannot read PEFT adapter config {}: {e}", path.display()),
        })?;
        Self::parse(&text).map_err(|e| prefix_reason(&path.display().to_string(), e))
    }

    /// Parse the text of an `adapter_config.json`.
    pub fn from_json_str(text: &str) -> Result<Self> {
        Self::parse(text).map_err(|e| prefix_reason("adapter_config.json", e))
    }

    fn parse(text: &str) -> Result<Self> {
        let raw: RawConfig = serde_json::from_str(text)
            .map_err(|e| config_error(format!("not a valid PEFT adapter config: {e}")))?;
        raw.into_config()
    }

    /// LoRA rank of PEFT module `module`: its first `rank_pattern` match, else `r`.
    pub fn rank_for(&self, module: &str) -> usize {
        first_match(&self.rank_pattern, module).map_or(self.r, |p| p.value)
    }

    /// The `rank_pattern` key that sets the rank of PEFT module `module`, if any.
    pub fn rank_pattern_key(&self, module: &str) -> Option<&str> {
        first_match(&self.rank_pattern, module).map(|p| p.pattern.as_str())
    }

    /// LoRA alpha of PEFT module `module`: its first `alpha_pattern` match, else `lora_alpha`.
    pub fn alpha_for(&self, module: &str) -> f32 {
        first_match(&self.alpha_pattern, module).map_or(self.lora_alpha, |p| p.value)
    }

    /// Output scale of PEFT module `module`: `alpha / r`, or `alpha / sqrt(r)` with rsLoRA.
    pub fn scaling_for(&self, module: &str) -> f32 {
        let rank = self.rank_for(module) as f64;
        let alpha = f64::from(self.alpha_for(module));
        let scaling = if self.use_rslora {
            alpha / rank.sqrt()
        } else {
            alpha / rank
        };
        scaling as f32
    }
}

fn config_error(reason: String) -> Error {
    Error::ModelError { reason }
}

/// `e` with `origin: ` in front of its reason, when it is a model error.
pub(super) fn prefix_reason(origin: &str, e: Error) -> Error {
    match e {
        Error::ModelError { reason } => Error::ModelError {
            reason: format!("{origin}: {reason}"),
        },
        other => other,
    }
}

fn unsupported(field: &str, value: impl fmt::Display, why: &str) -> Error {
    config_error(format!("{field} = {value} is unsupported: {why}"))
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawTargetModules {
    Regex(String),
    Names(Vec<String>),
}

#[derive(Deserialize)]
struct RawConfig {
    peft_type: Option<String>,
    r: Option<usize>,
    lora_alpha: Option<f64>,
    use_rslora: Option<bool>,
    use_dora: Option<bool>,
    fan_in_fan_out: Option<bool>,
    lora_bias: Option<bool>,
    use_qalora: Option<bool>,
    rank_pattern: Option<OrderedMap<usize>>,
    alpha_pattern: Option<OrderedMap<f64>>,
    target_modules: Option<RawTargetModules>,
    modules_to_save: Option<Vec<String>>,
    base_model_name_or_path: Option<String>,
    task_type: Option<String>,
    target_parameters: Option<Vec<String>>,
    alora_invocation_tokens: Option<serde_json::Value>,
    trainable_token_indices: Option<serde_json::Value>,
    layer_replication: Option<serde_json::Value>,
}

impl RawConfig {
    fn reject_unsupported(&self) -> Result<()> {
        if let Some(kind) = &self.peft_type
            && kind != "LORA"
        {
            return Err(unsupported("peft_type", kind, "only LORA adapters load"));
        }
        if self.use_dora == Some(true) {
            return Err(unsupported(
                "use_dora",
                true,
                "DoRA adds a per-module magnitude vector that an (A, B, scaling) pair cannot hold",
            ));
        }
        if self.fan_in_fan_out == Some(true) {
            return Err(unsupported(
                "fan_in_fan_out",
                true,
                "the base layer stores its weight transposed (Conv1D), and A/B would need transposing",
            ));
        }
        if self.lora_bias == Some(true) {
            return Err(unsupported(
                "lora_bias",
                true,
                "lora_B carries a bias term that an (A, B, scaling) pair cannot hold",
            ));
        }
        if self.use_qalora == Some(true) {
            return Err(unsupported(
                "use_qalora",
                true,
                "QA-LoRA pools the input before lora_A, which changes the adapter math",
            ));
        }
        if let Some(modules) = &self.modules_to_save
            && !modules.is_empty()
        {
            return Err(unsupported(
                "modules_to_save",
                format!("{modules:?}"),
                "the adapter replaces whole modules, and only lora_A/lora_B pairs load",
            ));
        }
        if let Some(params) = &self.target_parameters
            && !params.is_empty()
        {
            return Err(unsupported(
                "target_parameters",
                format!("{params:?}"),
                "LoRA on raw parameters is not a per-module A/B pair",
            ));
        }
        for (field, value) in [
            ("alora_invocation_tokens", &self.alora_invocation_tokens),
            ("trainable_token_indices", &self.trainable_token_indices),
            ("layer_replication", &self.layer_replication),
        ] {
            if let Some(value) = value {
                return Err(unsupported(
                    field,
                    value,
                    "it changes the adapter beyond per-module A/B pairs",
                ));
            }
        }
        Ok(())
    }

    fn into_config(self) -> Result<PeftConfig> {
        self.reject_unsupported()?;
        let r = self.r.ok_or_else(|| {
            config_error("field `r` is missing; the LoRA rank is required".into())
        })?;
        if r == 0 {
            return Err(config_error("r = 0 is not a valid LoRA rank".into()));
        }
        let lora_alpha = self.lora_alpha.ok_or_else(|| {
            config_error("field `lora_alpha` is missing; the LoRA alpha is required".into())
        })?;
        check_alpha("lora_alpha", lora_alpha)?;

        let mut rank_pattern = Vec::new();
        for (pattern, rank) in self.rank_pattern.map(|m| m.0).unwrap_or_default() {
            if rank == 0 {
                return Err(config_error(format!(
                    "rank_pattern key `{pattern}` sets rank 0, which is not a valid LoRA rank"
                )));
            }
            rank_pattern.push(PatternOverride::new("rank_pattern", pattern, rank)?);
        }
        let mut alpha_pattern = Vec::new();
        for (pattern, alpha) in self.alpha_pattern.map(|m| m.0).unwrap_or_default() {
            check_alpha(&format!("alpha_pattern key `{pattern}`"), alpha)?;
            alpha_pattern.push(PatternOverride::new(
                "alpha_pattern",
                pattern,
                alpha as f32,
            )?);
        }

        let target_modules = self.target_modules.map(|t| match t {
            RawTargetModules::Regex(re) => TargetModules::Regex(re),
            RawTargetModules::Names(names) => TargetModules::Names(names),
        });

        Ok(PeftConfig {
            r,
            lora_alpha: lora_alpha as f32,
            use_rslora: self.use_rslora.unwrap_or(false),
            rank_pattern,
            alpha_pattern,
            target_modules,
            base_model_name_or_path: self.base_model_name_or_path,
            task_type: self.task_type,
        })
    }
}

fn check_alpha(field: &str, alpha: f64) -> Result<()> {
    if alpha.is_finite() {
        Ok(())
    } else {
        Err(config_error(format!(
            "{field} = {alpha} is not a finite alpha"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(extra: &str) -> Result<PeftConfig> {
        PeftConfig::from_json_str(&format!(
            r#"{{"peft_type": "LORA", "r": 8, "lora_alpha": 16{extra}}}"#
        ))
    }

    fn err_text(extra: &str) -> String {
        match parse(extra) {
            Ok(_) => panic!("config with `{extra}` parsed but must error"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn parses_plain_config() {
        let cfg = parse(
            r#", "use_rslora": false, "use_dora": false, "fan_in_fan_out": false,
               "lora_bias": false, "modules_to_save": null, "rank_pattern": {},
               "alpha_pattern": {}, "target_modules": ["q_proj", "v_proj"],
               "base_model_name_or_path": "Qwen/Qwen3.5-0.8B", "task_type": "CAUSAL_LM""#,
        )
        .unwrap();
        assert_eq!(cfg.r, 8);
        assert_eq!(cfg.lora_alpha, 16.0);
        assert_eq!(
            cfg.target_modules,
            Some(TargetModules::Names(vec!["q_proj".into(), "v_proj".into()]))
        );
        assert_eq!(
            cfg.base_model_name_or_path.as_deref(),
            Some("Qwen/Qwen3.5-0.8B")
        );
        assert_eq!(cfg.task_type.as_deref(), Some("CAUSAL_LM"));
        assert_eq!(cfg.scaling_for("model.layers.0.self_attn.q_proj"), 2.0);
    }

    #[test]
    fn target_modules_regex_string() {
        let cfg = parse(r#", "target_modules": ".*\\.(q|v)_proj""#).unwrap();
        assert_eq!(
            cfg.target_modules,
            Some(TargetModules::Regex(r".*\.(q|v)_proj".into()))
        );
    }

    #[test]
    fn missing_rank_or_alpha_errors() {
        let no_r = PeftConfig::from_json_str(r#"{"lora_alpha": 16}"#)
            .unwrap_err()
            .to_string();
        assert!(no_r.contains("`r` is missing"), "{no_r}");
        let no_alpha = PeftConfig::from_json_str(r#"{"r": 8}"#)
            .unwrap_err()
            .to_string();
        assert!(no_alpha.contains("`lora_alpha` is missing"), "{no_alpha}");
        let zero = PeftConfig::from_json_str(r#"{"r": 0, "lora_alpha": 16}"#)
            .unwrap_err()
            .to_string();
        assert!(zero.contains("r = 0"), "{zero}");
    }

    #[test]
    fn rejects_each_unsupported_field() {
        for (extra, needle) in [
            (r#", "use_dora": true"#, "use_dora = true"),
            (r#", "fan_in_fan_out": true"#, "fan_in_fan_out = true"),
            (r#", "lora_bias": true"#, "lora_bias = true"),
            (r#", "use_qalora": true"#, "use_qalora = true"),
            (
                r#", "modules_to_save": ["lm_head"]"#,
                "modules_to_save = [\"lm_head\"]",
            ),
            (
                r#", "target_parameters": ["experts.w1"]"#,
                "target_parameters",
            ),
            (
                r#", "alora_invocation_tokens": [1, 2]"#,
                "alora_invocation_tokens",
            ),
            (
                r#", "trainable_token_indices": [5]"#,
                "trainable_token_indices",
            ),
            (r#", "layer_replication": [[0, 2]]"#, "layer_replication"),
        ] {
            let text = err_text(extra);
            assert!(
                text.contains(needle),
                "`{extra}` error lacks `{needle}`: {text}"
            );
            assert!(text.contains("unsupported"), "{text}");
        }
        let other = PeftConfig::from_json_str(r#"{"peft_type": "IA3", "r": 8, "lora_alpha": 8}"#)
            .unwrap_err()
            .to_string();
        assert!(other.contains("peft_type = IA3"), "{other}");
    }

    #[test]
    fn empty_modules_to_save_is_accepted() {
        assert!(parse(r#", "modules_to_save": []"#).is_ok());
        assert!(parse(r#", "modules_to_save": null"#).is_ok());
    }

    #[test]
    fn rslora_scales_by_sqrt_rank() {
        let cfg = PeftConfig::from_json_str(r#"{"r": 16, "lora_alpha": 32, "use_rslora": true}"#)
            .unwrap();
        assert_eq!(cfg.scaling_for("model.layers.0.mlp.up_proj"), 8.0);
    }

    #[test]
    fn rank_and_alpha_patterns_follow_peft_matching() {
        let cfg = parse(
            r#", "rank_pattern": {"q_proj": 4, "layers.1.self_attn.v_proj": 2},
               "alpha_pattern": {"v_proj": 6}"#,
        )
        .unwrap();
        assert_eq!(cfg.rank_for("model.layers.0.self_attn.q_proj"), 4);
        assert_eq!(cfg.rank_for("model.layers.1.self_attn.v_proj"), 2);
        assert_eq!(cfg.rank_for("model.layers.0.self_attn.v_proj"), 8);
        assert_eq!(
            cfg.rank_pattern_key("model.layers.1.self_attn.v_proj"),
            Some("layers.1.self_attn.v_proj")
        );
        assert_eq!(
            cfg.rank_pattern_key("model.layers.0.self_attn.v_proj"),
            None
        );
        // A match starts right after a `.`, never inside a segment.
        assert_eq!(cfg.rank_for("model.layers.0.self_attn.xq_proj"), 8);
        // The pattern must reach the end of the name.
        assert_eq!(cfg.rank_for("model.layers.0.self_attn.q_proj.base"), 8);
        // A whole-name match needs no leading `.`.
        assert_eq!(cfg.rank_for("q_proj"), 4);
        assert_eq!(cfg.scaling_for("model.layers.1.self_attn.v_proj"), 3.0);
        assert_eq!(cfg.scaling_for("model.layers.0.self_attn.q_proj"), 4.0);
        assert_eq!(cfg.alpha_for("model.layers.0.mlp.down_proj"), 16.0);
    }

    #[test]
    fn rslora_uses_pattern_rank_and_alpha() {
        let cfg = PeftConfig::from_json_str(
            r#"{"r": 8, "lora_alpha": 8, "use_rslora": true,
                "rank_pattern": {"o_proj": 4}, "alpha_pattern": {"o_proj": 10}}"#,
        )
        .unwrap();
        assert_eq!(cfg.scaling_for("model.layers.3.self_attn.o_proj"), 5.0);
    }

    #[test]
    fn first_pattern_in_file_order_wins() {
        let specific_first =
            parse(r#", "rank_pattern": {"self_attn.q_proj": 4, "q_proj": 2}"#).unwrap();
        assert_eq!(
            specific_first.rank_for("model.layers.0.self_attn.q_proj"),
            4
        );
        let general_first =
            parse(r#", "rank_pattern": {"q_proj": 2, "self_attn.q_proj": 4}"#).unwrap();
        assert_eq!(general_first.rank_for("model.layers.0.self_attn.q_proj"), 2);
    }

    #[test]
    fn regex_pattern_keys_match() {
        let cfg = parse(r#", "rank_pattern": {"layers\\.[0-3]\\.mlp\\..*": 2}"#).unwrap();
        assert_eq!(cfg.rank_for("model.layers.2.mlp.gate_proj"), 2);
        assert_eq!(cfg.rank_for("model.layers.12.mlp.gate_proj"), 8);
    }

    #[test]
    fn invalid_pattern_and_zero_pattern_rank_error() {
        let bad = err_text(r#", "rank_pattern": {"(": 4}"#);
        assert!(bad.contains("rank_pattern key `(`"), "{bad}");
        let zero = err_text(r#", "rank_pattern": {"q_proj": 0}"#);
        assert!(zero.contains("rank 0"), "{zero}");
    }
}
