//! Derived `qwen35` fields of a Hugging Face Qwen3.5 `config.json`: the
//! rotated width, the IMROPE sections, and the layer roles.

use super::keys::{Keys, bad, opt_f64};
use crate::error::{Error, Result};
use crate::model::config::HybridConfig;
use serde_json::Value;

/// Converter defaults.
const DEFAULT_PARTIAL_ROTARY_FACTOR: f64 = 0.25;
const DEFAULT_MROPE_SECTION: [usize; 3] = [11, 11, 10];
const DEFAULT_FULL_ATTENTION_INTERVAL: usize = 4;

/// `head_dim * partial_rotary_factor`, which must be a whole number.
pub(super) fn rope_dim(keys: &Keys<'_>, head_dim: usize) -> Result<usize> {
    const KEY: &str = "partial_rotary_factor";
    let factor = opt_f64(KEY, keys.rope_get(KEY))?.unwrap_or(DEFAULT_PARTIAL_ROTARY_FACTOR);
    let dim = head_dim as f64 * factor;
    if !dim.is_finite() || dim < 0.0 || dim.fract() != 0.0 {
        return Err(bad(
            KEY,
            &Value::from(factor),
            &format!("a factor that makes head_dim ({head_dim}) a whole rotated width"),
        ));
    }
    Ok(dim as usize)
}

/// `mrope_section` padded with zeros to four streams.
pub(super) fn rope_sections(keys: &Keys<'_>) -> Result<[usize; 4]> {
    const KEY: &str = "mrope_section";
    let mut out = [0usize; 4];
    let Some(v) = keys.rope_get(KEY) else {
        out[..3].copy_from_slice(&DEFAULT_MROPE_SECTION);
        return Ok(out);
    };
    let want = "an array of at most 4 non-negative integers";
    let items = v
        .as_array()
        .filter(|a| a.len() <= 4)
        .ok_or_else(|| bad(KEY, v, want))?;
    for (slot, item) in out.iter_mut().zip(items) {
        *slot = item.as_u64().ok_or_else(|| bad(KEY, v, want))? as usize;
    }
    Ok(out)
}

/// Layer roles from `layer_types`, checked against `full_attention_interval`
/// when both are present. With neither, every fourth layer is attention.
pub(super) fn hybrid_layers(keys: &Keys<'_>, num_layers: usize) -> Result<HybridConfig> {
    let interval = keys.opt_usize("full_attention_interval")?;
    let Some(types) = keys.get("layer_types") else {
        let interval = interval.unwrap_or(DEFAULT_FULL_ATTENTION_INTERVAL);
        return Ok(HybridConfig::from_full_attention_interval(
            num_layers, interval,
        ));
    };
    let types = types
        .as_array()
        .filter(|a| a.len() == num_layers)
        .ok_or_else(|| {
            bad(
                "layer_types",
                types,
                &format!("an array of num_hidden_layers ({num_layers}) entries"),
            )
        })?;
    let mut layers = HybridConfig {
        ssm_layers: Vec::new(),
        attention_layers: Vec::new(),
    };
    for (i, t) in types.iter().enumerate() {
        match t.as_str() {
            Some("linear_attention") => layers.ssm_layers.push(i),
            Some("full_attention") => layers.attention_layers.push(i),
            _ => {
                return Err(bad(
                    format!("layer_types[{i}]"),
                    t,
                    "\"linear_attention\" or \"full_attention\"",
                ));
            }
        }
    }
    if let Some(interval) = interval {
        let rule = HybridConfig::from_full_attention_interval(num_layers, interval);
        if rule.attention_layers != layers.attention_layers {
            return Err(Error::ModelError {
                reason: format!(
                    "Qwen3.5 config.json: layer_types puts full attention at {:?}; \
                     full_attention_interval {interval} puts it at {:?}",
                    layers.attention_layers, rule.attention_layers
                ),
            });
        }
    }
    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rope_dim_needs_a_whole_width() {
        let v = json!({"partial_rotary_factor": 0.25});
        assert_eq!(rope_dim(&Keys::new(&v).unwrap(), 256).unwrap(), 64);
        let v = json!({});
        assert_eq!(rope_dim(&Keys::new(&v).unwrap(), 128).unwrap(), 32);
        let v = json!({"partial_rotary_factor": 0.33});
        let err = rope_dim(&Keys::new(&v).unwrap(), 10)
            .unwrap_err()
            .to_string();
        assert!(err.contains("partial_rotary_factor"), "{err}");
    }

    #[test]
    fn sections_pad_to_four_and_default_to_the_converter() {
        let v = json!({"rope_parameters": {"mrope_section": [8, 8]}});
        assert_eq!(
            rope_sections(&Keys::new(&v).unwrap()).unwrap(),
            [8, 8, 0, 0]
        );
        let v = json!({});
        assert_eq!(
            rope_sections(&Keys::new(&v).unwrap()).unwrap(),
            [11, 11, 10, 0]
        );
        let v = json!({"mrope_section": [1, 1, 1, 1, 1]});
        assert!(rope_sections(&Keys::new(&v).unwrap()).is_err());
    }

    #[test]
    fn interval_alone_sets_the_roles() {
        let v = json!({"full_attention_interval": 2});
        let layers = hybrid_layers(&Keys::new(&v).unwrap(), 4).unwrap();
        assert_eq!(layers.attention_layers, vec![1, 3]);
        assert_eq!(layers.ssm_layers, vec![0, 2]);
    }

    #[test]
    fn layer_types_set_the_roles_and_reject_unknown_kinds() {
        let v = json!({"layer_types": ["linear_attention", "full_attention"]});
        let layers = hybrid_layers(&Keys::new(&v).unwrap(), 2).unwrap();
        assert_eq!(layers.attention_layers, vec![1]);
        assert!(hybrid_layers(&Keys::new(&v).unwrap(), 3).is_err());
        let v = json!({"layer_types": ["linear_attention", "sliding_attention"]});
        let err = hybrid_layers(&Keys::new(&v).unwrap(), 2)
            .unwrap_err()
            .to_string();
        assert!(err.contains("layer_types[1]"), "{err}");
    }
}
