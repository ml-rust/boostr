//! The object schema walk: properties, `required` and `additionalProperties`.
//!
//! Properties appear in declared order, as `run` builds them. Extra keys from
//! `additionalProperties` follow the declared properties. An extra key can
//! repeat a declared name.

use std::collections::HashSet;

use serde_json::{Map, Value};

use super::super::generic::{self, member};
use super::super::limits::check_members;
use super::super::refs::child;
use super::super::string;
use super::super::walk::{Lang, Walker, invalid};
use super::run::{Prop, run};
use crate::inference::grammar::ast::Expr;
use crate::inference::grammar::error::{GrammarError, GrammarErrorKind, excerpt};

/// The object schema `map` at `path`.
pub(in super::super) fn schema<'s>(
    w: &mut Walker<'s>,
    map: &'s Map<String, Value>,
    path: &str,
) -> Result<Lang, GrammarError> {
    let required = required(map, path)?;
    let required_set: HashSet<&str> = required.iter().copied().collect();
    let extras = extras(w, map, path)?;
    let mut props: Vec<Prop<'s>> = Vec::new();
    let declared = match map.get("properties") {
        None => None,
        Some(Value::Object(declared)) => Some(declared),
        Some(_) => return Err(invalid(&child(path, "properties"), "must be an object")),
    };
    let base = child(path, "properties");
    check_members("properties", &base, declared.map_or(0, Map::len))?;
    let absent = required
        .iter()
        .filter(|key| !declared.is_some_and(|d| d.contains_key(**key)))
        .count();
    let total = declared.map_or(0, Map::len).saturating_add(absent);
    check_members("required", &child(path, "required"), total)?;
    for (key, sub) in declared.into_iter().flatten() {
        let is_required = required_set.contains(key.as_str());
        match w.schema(sub, &child(&base, key))? {
            Lang::Expr(value) => props.push(Prop {
                key: key.as_str(),
                value,
                required: is_required,
            }),
            Lang::Empty(reason) if is_required => return Ok(Lang::Empty(reason)),
            Lang::Empty(_) => {}
        }
    }
    for key in required {
        if declared.is_some_and(|d| d.contains_key(key)) {
            continue;
        }
        let Some(value) = extras.clone() else {
            return Err(GrammarError::new(
                GrammarErrorKind::EmptyLanguage,
                None,
                format!(
                    "required property `{}` at `{}` is not in `properties`, \
                     and `additionalProperties` forbids it. \
                     Declare the property or allow additional properties.",
                    excerpt(key),
                    child(path, "required")
                ),
            ));
        };
        props.push(Prop {
            key,
            value,
            required: true,
        });
    }
    Ok(Lang::Expr(body(w, props, extras, path)?))
}

/// The `required` names of `map`, without repeats.
fn required<'s>(map: &'s Map<String, Value>, path: &str) -> Result<Vec<&'s str>, GrammarError> {
    let Some(value) = map.get("required") else {
        return Ok(Vec::new());
    };
    let at = child(path, "required");
    let Value::Array(items) = value else {
        return Err(invalid(&at, "must be an array of property names"));
    };
    let mut names = Vec::with_capacity(items.len());
    let mut seen: HashSet<&str> = HashSet::with_capacity(items.len());
    for item in items {
        let Value::String(name) = item else {
            return Err(invalid(&at, "must list property names as strings"));
        };
        if seen.insert(name.as_str()) {
            names.push(name.as_str());
        }
    }
    Ok(names)
}

/// The value grammar of extra keys, or `None` when `map` allows no extra key.
fn extras<'s>(
    w: &mut Walker<'s>,
    map: &'s Map<String, Value>,
    path: &str,
) -> Result<Option<Expr>, GrammarError> {
    match map.get("additionalProperties") {
        None if w.opts.additional_properties_default => Ok(Some(generic::any_value(w))),
        None | Some(Value::Bool(false)) => Ok(None),
        Some(sub) => match w.schema(sub, &child(path, "additionalProperties"))? {
            Lang::Expr(value) => Ok(Some(value)),
            Lang::Empty(_) => Ok(None),
        },
    }
}

/// `{ ws body ws }` over `props` in order, then the extra pairs.
fn body(
    w: &mut Walker<'_>,
    props: Vec<Prop<'_>>,
    extras: Option<Expr>,
    path: &str,
) -> Result<Expr, GrammarError> {
    // `can_be_empty`: no property is required, so the declared run can be empty.
    let can_be_empty = props.iter().all(|prop| !prop.required);
    let prefix = run(w, props, path)?;
    let ws = w.ws();
    let inner = match (prefix, extras) {
        (None, None) => None,
        (Some(prefix), None) => Some(prefix),
        (None, Some(value)) => {
            let key = string::any_string(w);
            let pair = member(w, key, value);
            return Ok(generic::list(w, "{", "}", pair));
        }
        (Some(prefix), Some(value)) => {
            let key = string::any_string(w);
            let value = w.share(value);
            let pair = member(w, key, value);
            let pair = w.share(pair);
            let first = if can_be_empty {
                Expr::alt(vec![prefix, pair.clone()])
            } else {
                prefix
            };
            let comma = w.comma();
            let more = Expr::repeat(Expr::seq(vec![comma, pair]), 0, None);
            Some(Expr::seq(vec![first, more]))
        }
    };
    let open = Expr::literal("{");
    let close = Expr::literal("}");
    Ok(match inner {
        None => Expr::seq(vec![open, ws, close]),
        Some(inner) => {
            let full = Expr::seq(vec![inner, ws.clone(), close.clone()]);
            let rest = if can_be_empty {
                Expr::alt(vec![close, full])
            } else {
                full
            };
            Expr::seq(vec![open, ws, rest])
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::grammar::json_schema::SchemaOptions;
    use crate::inference::grammar::json_schema::fixture::{check, dfa, error, opts};
    use serde_json::json;

    #[test]
    fn extra_keys_need_additional_properties() {
        let base = r#""properties": {"a": {"type": "integer"}}, "required": ["a"]"#;
        let closed = dfa(&format!("{{{base}}}"), &opts());
        check(
            &closed,
            &[r#"{"a":1}"#],
            &[r#"{"a":1,"z":2}"#, r#"{"z":2,"a":1}"#],
        );
        let open = dfa(
            &format!(r#"{{{base}, "additionalProperties": true}}"#),
            &shallow(),
        );
        check(
            &open,
            &[r#"{"a":1}"#, r#"{"a":1,"z":[2],"y":{"k":null}}"#],
            &[r#"{"z":2}"#, r#"{"z":2,"a":1}"#, r#"{"a":1,}"#],
        );
        let typed = dfa(
            &format!(r#"{{{base}, "additionalProperties": {{"type": "string"}}}}"#),
            &opts(),
        );
        check(&typed, &[r#"{"a":1,"z":"s"}"#], &[r#"{"a":1,"z":2}"#]);
        let by_default = SchemaOptions {
            additional_properties_default: true,
            ..shallow()
        };
        let d = dfa(&format!("{{{base}}}"), &by_default);
        check(&d, &[r#"{"a":1,"z":true}"#], &[r#"{"a":1,"z":}"#]);
    }

    #[test]
    fn extras_without_declared_properties_follow_list_commas() {
        let d = dfa(
            r#"{"type": "object", "additionalProperties": {"type": "integer"}}"#,
            &opts(),
        );
        check(
            &d,
            &["{}", r#"{"x":1}"#, r#"{"x":1,"y":2}"#],
            &[r#"{"x":1,}"#, r#"{,"x":1}"#, r#"{"x":"1"}"#],
        );
        let d = dfa(
            r#"{"properties": {"a": {"type": "null"}}, "additionalProperties": {"type": "integer"}}"#,
            &opts(),
        );
        check(
            &d,
            &[
                "{}",
                r#"{"a":null}"#,
                r#"{"x":1}"#,
                r#"{"a":null,"x":1,"y":2}"#,
            ],
            &[r#"{"x":1,"a":null}"#, r#"{"a":null,,"x":1}"#],
        );
    }

    #[test]
    fn whitespace_variants_respect_the_bound() {
        let schema = r#"{"properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}"#;
        let compact = SchemaOptions {
            max_whitespace: 0,
            ..opts()
        };
        check(
            &dfa(schema, &compact),
            &[r#"{"a":1,"b":2}"#],
            &[r#"{"a": 1,"b":2}"#, r#" {"a":1,"b":2}"#],
        );
        check(
            &dfa(schema, &opts()),
            &[r#"{"a": 1, "b": 2}"#, "{\n \"a\" : 1 ,\n \"b\":2\n}"],
            &["{\n  \"a\": 1, \"b\": 2}", r#"{"a":   1,"b":2}"#],
        );
        let wide = SchemaOptions {
            max_whitespace: 4,
            ..opts()
        };
        check(
            &dfa(schema, &wide),
            &[
                "{\n  \"a\": 1,\n  \"b\": 2\n}",
                "{\n\t\t\"a\": 1, \"b\": 2}",
            ],
            &["{\n     \"a\": 1, \"b\": 2}"],
        );
    }

    #[test]
    fn required_names_outside_properties_use_additional_properties() {
        let d = dfa(
            r#"{"properties": {"a": {"type": "integer"}}, "required": ["z"], "additionalProperties": {"type": "boolean"}}"#,
            &opts(),
        );
        check(
            &d,
            &[
                r#"{"z":true}"#,
                r#"{"a":1,"z":false}"#,
                r#"{"a":1,"z":true,"q":false}"#,
            ],
            &[r#"{"a":1}"#, r#"{"z":1}"#],
        );
        let err = error(r#"{"properties": {}, "required": ["ghost"]}"#, &opts());
        assert_eq!(err.kind(), GrammarErrorKind::EmptyLanguage);
        assert!(err.message().contains("`ghost`"), "{}", err.message());
        assert!(err.message().contains("`#/required`"), "{}", err.message());
    }

    #[test]
    fn keys_are_escaped_literals() {
        let d = dfa(
            r#"{"properties": {"a\"b": {"type": "null"}, "é/": {"type": "null"}}, "required": ["a\"b", "é/"]}"#,
            &opts(),
        );
        check(
            &d,
            &[r#"{"a\"b":null,"é/":null}"#],
            &[r#"{"a"b":null,"é/":null}"#],
        );
    }

    #[test]
    fn an_impossible_property_drops_or_empties_the_object() {
        let d = dfa(
            r#"{"properties": {"a": false, "b": {"type": "null"}}}"#,
            &opts(),
        );
        check(&d, &["{}", r#"{"b":null}"#], &[r#"{"a":null}"#]);
        let err = error(
            r#"{"properties": {"a": false}, "required": ["a"]}"#,
            &opts(),
        );
        assert_eq!(err.kind(), GrammarErrorKind::EmptyLanguage);
        assert!(
            err.message().contains("`#/properties/a`"),
            "{}",
            err.message()
        );
    }

    fn shallow() -> SchemaOptions {
        SchemaOptions {
            max_value_depth: 2,
            ..opts()
        }
    }

    fn props(n: usize) -> serde_json::Map<String, Value> {
        (0..n)
            .map(|i| (format!("p{i}"), json!({"type": "integer"})))
            .collect()
    }

    #[test]
    fn a_thousand_properties_compile_and_one_more_does_not() {
        use crate::inference::grammar::json_schema::fixture::{limit_message, lowered};
        lowered(&json!({"type": "object", "properties": props(1_000)}));
        let msg = limit_message(&json!({"type": "object", "properties": props(1_001)}));
        assert!(msg.contains("`properties` at `#/properties`"), "{msg}");
        assert!(msg.contains("1001 members"), "{msg}");
        assert!(msg.contains("limit is 1000"), "{msg}");
    }

    #[test]
    fn required_names_outside_properties_count_toward_the_limit() {
        use crate::inference::grammar::json_schema::fixture::limit_message;
        let names: Vec<String> = (0..1_001).map(|i| format!("r{i}")).collect();
        let msg = limit_message(&json!({
            "type": "object",
            "required": names,
            "additionalProperties": true
        }));
        assert!(msg.contains("`required` at `#/required`"), "{msg}");
    }
}
