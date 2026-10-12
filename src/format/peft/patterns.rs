//! PEFT `rank_pattern` / `alpha_pattern` entries: PEFT's module-name
//! matching rule, and reading the pattern object in file order.

use crate::error::{Error, Result};
use serde::Deserialize;
use serde::de::{Deserializer, MapAccess, Visitor};
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

/// One `rank_pattern` or `alpha_pattern` entry.
///
/// PEFT's `get_pattern_key` matches `pattern` against a module name `m` as
/// `re.match(rf"(.*\.)?({pattern})$", m)`: the pattern covers the whole name,
/// or a suffix that starts right after a `.`. The first entry in file order
/// that matches wins.
#[derive(Debug, Clone)]
pub struct PatternOverride<T> {
    /// The key as written in the config.
    pub pattern: String,
    /// The rank or alpha this key sets.
    pub value: T,
    regex: Arc<regexr::Regex>,
}

impl<T> PatternOverride<T> {
    /// Compile `pattern` from config field `field` under PEFT's matching rule.
    pub fn new(field: &str, pattern: String, value: T) -> Result<Self> {
        let anchored = format!("^(.*\\.)?({pattern})$");
        let regex = regexr::Regex::new(&anchored).map_err(|e| Error::ModelError {
            reason: format!("{field} key `{pattern}` is not a valid regex: {e}"),
        })?;
        Ok(Self {
            pattern,
            value,
            regex: Arc::new(regex),
        })
    }

    /// Whether this entry applies to the PEFT module name `module`.
    pub fn matches(&self, module: &str) -> bool {
        self.regex.is_match(module)
    }
}

/// The first entry in `patterns` that matches `module`.
pub fn first_match<'a, T>(
    patterns: &'a [PatternOverride<T>],
    module: &str,
) -> Option<&'a PatternOverride<T>> {
    patterns.iter().find(|p| p.matches(module))
}

/// A JSON object read as key/value pairs in file order.
pub(super) struct OrderedMap<T>(pub(super) Vec<(String, T)>);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for OrderedMap<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct OrderedVisitor<T>(PhantomData<T>);

        impl<'de, T: Deserialize<'de>> Visitor<'de> for OrderedVisitor<T> {
            type Value = OrderedMap<T>;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON object")
            }

            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry::<String, T>()? {
                    entries.push(entry);
                }
                Ok(OrderedMap(entries))
            }
        }

        deserializer.deserialize_map(OrderedVisitor(PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(pattern: &str) -> PatternOverride<usize> {
        PatternOverride::new("rank_pattern", pattern.to_string(), 1).unwrap()
    }

    #[test]
    fn matches_whole_name_or_dotted_suffix() {
        let q = entry("q_proj");
        assert!(q.matches("q_proj"));
        assert!(q.matches("model.layers.0.self_attn.q_proj"));
        assert!(!q.matches("model.layers.0.self_attn.xq_proj"));
        assert!(!q.matches("model.layers.0.self_attn.q_proj.base_layer"));
        let layer = entry("layers.1.mlp.up_proj");
        assert!(layer.matches("model.layers.1.mlp.up_proj"));
        assert!(!layer.matches("model.layers.11.mlp.up_proj"));
    }

    #[test]
    fn first_match_keeps_file_order() {
        let patterns = [entry("self_attn.q_proj"), entry("q_proj")];
        let hit = first_match(&patterns, "model.layers.0.self_attn.q_proj").unwrap();
        assert_eq!(hit.pattern, "self_attn.q_proj");
        assert!(first_match(&patterns, "model.layers.0.mlp.up_proj").is_none());
    }

    #[test]
    fn ordered_map_keeps_file_order() {
        let map: OrderedMap<usize> = serde_json::from_str(r#"{"b": 2, "a": 1, "c": 3}"#).unwrap();
        let keys: Vec<&str> = map.0.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["b", "a", "c"]);
    }

    #[test]
    fn invalid_regex_errors() {
        let text = PatternOverride::new("alpha_pattern", "(".to_string(), 1.0f32)
            .unwrap_err()
            .to_string();
        assert!(text.contains("alpha_pattern key `(`"), "{text}");
    }
}
