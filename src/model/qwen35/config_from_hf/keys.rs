//! Key lookup over a Hugging Face Qwen3.5 `config.json`, its error
//! messages, and the check that rejects options the `qwen35` model does not
//! run.

use crate::error::{Error, Result};
use serde_json::{Map, Value};
use std::fmt::Display;

pub(super) fn missing(key: &str) -> Error {
    Error::ModelError {
        reason: format!("Qwen3.5 config.json: missing {key}"),
    }
}

pub(super) fn bad(key: impl Display, value: &Value, want: &str) -> Error {
    Error::ModelError {
        reason: format!("Qwen3.5 config.json: {key} is {value}; expected {want}"),
    }
}

fn unsupported(key: &str, value: &Value, want: &str) -> Error {
    Error::ModelError {
        reason: format!(
            "Qwen3.5 config.json: {key} = {value} is not supported; the qwen35 model needs {want}"
        ),
    }
}

fn non_null(v: Option<&Value>) -> Option<&Value> {
    v.filter(|v| !v.is_null())
}

/// `v` as an object, `None` when absent or null.
fn object<'a>(key: &str, v: Option<&'a Value>) -> Result<Option<&'a Map<String, Value>>> {
    match non_null(v) {
        None => Ok(None),
        Some(Value::Object(m)) => Ok(Some(m)),
        Some(v) => Err(bad(key, v, "an object")),
    }
}

/// Key lookup: `text_config` first, then the top level. RoPE keys look in
/// `rope_parameters` (or `rope_scaling`) before both. A `null` value counts
/// as absent.
pub(super) struct Keys<'a> {
    text: Option<&'a Map<String, Value>>,
    top: &'a Map<String, Value>,
    rope: Option<&'a Map<String, Value>>,
}

impl<'a> Keys<'a> {
    /// Lookup over `config`, which must be a JSON object.
    pub(super) fn new(config: &'a Value) -> Result<Self> {
        let top = config
            .as_object()
            .ok_or_else(|| bad("config.json", config, "a JSON object"))?;
        let mut keys = Self {
            text: object("text_config", top.get("text_config"))?,
            top,
            rope: None,
        };
        keys.rope = match keys.get("rope_parameters") {
            Some(v) => object("rope_parameters", Some(v))?,
            None => object("rope_scaling", keys.get("rope_scaling"))?,
        };
        Ok(keys)
    }

    pub(super) fn get(&self, key: &str) -> Option<&'a Value> {
        non_null(self.text.and_then(|t| t.get(key))).or_else(|| non_null(self.top.get(key)))
    }

    pub(super) fn rope_get(&self, key: &str) -> Option<&'a Value> {
        non_null(self.rope.and_then(|r| r.get(key))).or_else(|| self.get(key))
    }

    pub(super) fn opt_usize(&self, key: &str) -> Result<Option<usize>> {
        self.get(key)
            .map(|v| {
                v.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| bad(key, v, "a non-negative integer"))
            })
            .transpose()
    }

    pub(super) fn usize(&self, key: &str) -> Result<usize> {
        self.opt_usize(key)?.ok_or_else(|| missing(key))
    }
}

/// `v` as a number, `None` when absent.
pub(super) fn opt_f64(key: &str, v: Option<&Value>) -> Result<Option<f64>> {
    v.map(|v| v.as_f64().ok_or_else(|| bad(key, v, "a number")))
        .transpose()
}

/// `Err` when `v` is present and not `want`.
fn require_bool(key: &str, v: Option<&Value>, want: bool) -> Result<()> {
    match v {
        None => Ok(()),
        Some(Value::Bool(b)) if *b == want => Ok(()),
        Some(v) => Err(unsupported(key, v, &want.to_string())),
    }
}

/// `Err` when `v` is present and not the string `want`.
fn require_str(key: &str, v: Option<&Value>, want: &str) -> Result<()> {
    match v {
        None => Ok(()),
        Some(Value::String(s)) if s == want => Ok(()),
        Some(v) => Err(unsupported(key, v, &format!("\"{want}\""))),
    }
}

/// Reject every option the `qwen35` model does not run. The module docs of
/// `build` list them.
pub(super) fn check_supported(keys: &Keys<'_>) -> Result<()> {
    require_bool("attn_output_gate", keys.get("attn_output_gate"), true)?;
    require_bool("attention_bias", keys.get("attention_bias"), false)?;
    require_bool(
        "mrope_interleaved",
        keys.rope_get("mrope_interleaved"),
        true,
    )?;
    require_str("hidden_act", keys.get("hidden_act"), "silu")?;
    require_str("rope_type", keys.rope_get("rope_type"), "default")?;
    match keys.get("mlp_only_layers") {
        None => {}
        Some(Value::Array(a)) if a.is_empty() => {}
        Some(v @ Value::Array(_)) => return Err(unsupported("mlp_only_layers", v, "[]")),
        Some(v) => return Err(bad("mlp_only_layers", v, "an array")),
    }
    let key_dim = keys.usize("linear_key_head_dim")?;
    let value_dim = keys.usize("linear_value_head_dim")?;
    if key_dim != value_dim {
        return Err(unsupported(
            "linear_value_head_dim",
            &Value::from(value_dim),
            &format!("linear_key_head_dim ({key_dim})"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn text_config_wins_over_the_top_level() {
        let v = json!({"a": 1, "b": 2, "text_config": {"a": 10, "c": null}});
        let keys = Keys::new(&v).unwrap();
        assert_eq!(keys.usize("a").unwrap(), 10);
        assert_eq!(keys.usize("b").unwrap(), 2);
        assert_eq!(keys.opt_usize("c").unwrap(), None);
        let err = keys.usize("d").unwrap_err().to_string();
        assert!(err.contains("missing d"), "{err}");
    }

    #[test]
    fn rope_parameters_win_for_rope_keys() {
        let v = json!({"text_config": {"partial_rotary_factor": 0.5,
            "rope_parameters": {"partial_rotary_factor": 0.25}}});
        let keys = Keys::new(&v).unwrap();
        let f = opt_f64("p", keys.rope_get("partial_rotary_factor")).unwrap();
        assert_eq!(f, Some(0.25));
        let v = json!({"rope_scaling": {"rope_theta": 5.0}});
        let keys = Keys::new(&v).unwrap();
        assert_eq!(
            opt_f64("t", keys.rope_get("rope_theta")).unwrap(),
            Some(5.0)
        );
    }

    #[test]
    fn wrong_types_name_the_key() {
        let v = json!({"a": "x", "text_config": 3});
        let err = Keys::new(&v).err().unwrap().to_string();
        assert!(err.contains("text_config"), "{err}");
        let v = json!({"a": -1});
        let err = Keys::new(&v).unwrap().usize("a").unwrap_err().to_string();
        assert!(err.contains("a is -1"), "{err}");
        assert!(Keys::new(&json!([1])).is_err());
    }
}
