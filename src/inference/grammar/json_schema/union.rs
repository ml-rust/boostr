//! Unions: `type` arrays, `anyOf`, `oneOf`, single-member `allOf`, `enum` and `const`.
//!
//! `oneOf` reads as `anyOf`: the grammar does not enforce that exactly one
//! member matches. An `enum` or `const` member matches only its compact
//! serialization by `serde_json::to_string`. An object or array member thus
//! matches with no whitespace and with keys in the schema order.

use std::collections::HashSet;

use serde_json::{Map, Value};

use super::keywords::unsupported;
use super::limits::check_members;
use super::refs::child;
use super::walk::{Lang, Walker, compact, invalid};
use super::{array, generic, object, scalar, string};
use crate::inference::grammar::ast::Expr;
use crate::inference::grammar::error::{GrammarError, excerpt};

/// The JSON Schema type names.
const TYPES: &[&str] = &[
    "null", "boolean", "integer", "number", "string", "object", "array",
];

/// The union of `members`. It is empty only when every member is empty.
pub(super) fn union(members: Vec<Lang>) -> Lang {
    let mut exprs = Vec::with_capacity(members.len());
    let mut reason = None;
    for member in members {
        match member {
            Lang::Expr(expr) => exprs.push(expr),
            Lang::Empty(why) => {
                reason.get_or_insert(why);
            }
        }
    }
    if exprs.is_empty() {
        Lang::Empty(reason.unwrap_or_else(|| "the union has no member".to_string()))
    } else {
        Lang::Expr(Expr::alt(exprs))
    }
}

/// The schema `map` by its `type`, or by its keywords when `type` is absent.
///
/// With no `type`, `properties`, `required` and `additionalProperties` imply
/// an object. `items`, `prefixItems`, `minItems` and `maxItems` imply an array.
/// `minLength` and `maxLength` imply a string. No such keyword gives any value.
pub(super) fn typed<'s>(
    w: &mut Walker<'s>,
    map: &'s Map<String, Value>,
    path: &str,
) -> Result<Lang, GrammarError> {
    let names = match type_names(map, path)? {
        Some(names) => names,
        None => {
            let inferred = inferred(map);
            if inferred.is_empty() {
                return Ok(Lang::Expr(generic::any_value(w)));
            }
            inferred
        }
    };
    let mut members = Vec::with_capacity(names.len());
    for name in names {
        let lang = match name {
            "null" => Lang::Expr(scalar::null()),
            "boolean" => Lang::Expr(scalar::boolean(w)),
            "integer" => Lang::Expr(scalar::integer(w)),
            "number" => Lang::Expr(scalar::number(w)),
            "string" => string::schema(w, map, path)?,
            "object" => object::schema(w, map, path)?,
            _ => array::schema(w, map, path)?,
        };
        members.push(lang);
    }
    Ok(union_or(members, || {
        format!("`type` at `{path}` lists no type")
    }))
}

/// The `anyOf` or `oneOf` array `value` at `path`.
pub(super) fn any_of<'s>(
    w: &mut Walker<'s>,
    value: &'s Value,
    path: &str,
) -> Result<Lang, GrammarError> {
    let Value::Array(items) = value else {
        return Err(invalid(path, "must be an array of schemas"));
    };
    if items.is_empty() {
        return Err(invalid(path, "must hold at least one schema"));
    }
    check_members(
        path.rsplit('/').next().unwrap_or("anyOf"),
        path,
        items.len(),
    )?;
    let mut members = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        members.push(w.schema(item, &child(path, &i.to_string()))?);
    }
    Ok(union(members))
}

/// The `allOf` array `value` at `path`. Only one member is supported.
///
/// An empty `allOf` is any value. In loose mode a longer one reads as its
/// first member, a superset of the intersection.
pub(super) fn all_of<'s>(
    w: &mut Walker<'s>,
    value: &'s Value,
    path: &str,
) -> Result<Lang, GrammarError> {
    let Value::Array(items) = value else {
        return Err(invalid(path, "must be an array of schemas"));
    };
    let Some(first) = items.first() else {
        return Ok(Lang::Expr(generic::any_value(w)));
    };
    if items.len() > 1 && w.opts.strict_keywords {
        let what = format!("`allOf` with {} members", items.len());
        return Err(unsupported(path, &what));
    }
    w.schema(first, &child(path, "0"))
}

/// The `enum` and `const` members of `map`, filtered by its `type`.
///
/// With both keywords the members are the `enum` members equal to `const`.
pub(super) fn enum_const(map: &Map<String, Value>, path: &str) -> Result<Lang, GrammarError> {
    let mut members: Vec<&Value> = match map.get("enum") {
        Some(Value::Array(items)) => {
            check_members("enum", &child(path, "enum"), items.len())?;
            items.iter().collect()
        }
        Some(_) => return Err(invalid(&child(path, "enum"), "must be an array")),
        None => Vec::new(),
    };
    if let Some(value) = map.get("const") {
        if map.contains_key("enum") {
            members.retain(|member| *member == value);
        } else {
            members.push(value);
        }
    }
    if let Some(names) = type_names(map, path)? {
        members.retain(|member| names.iter().any(|name| type_matches(member, name)));
    }
    let mut texts: Vec<String> = Vec::with_capacity(members.len());
    let mut seen: HashSet<String> = HashSet::with_capacity(members.len());
    for member in members {
        let text = compact(member, path)?;
        if seen.insert(text.clone()) {
            texts.push(text);
        }
    }
    if texts.is_empty() {
        return Ok(Lang::Empty(format!(
            "`enum` or `const` at `{path}` has no member of the allowed types"
        )));
    }
    Ok(Lang::Expr(Expr::alt(
        texts.into_iter().map(Expr::literal).collect(),
    )))
}

/// The type names in the `type` keyword of `map`, without repeats, if present.
fn type_names<'s>(
    map: &'s Map<String, Value>,
    path: &str,
) -> Result<Option<Vec<&'s str>>, GrammarError> {
    let at = child(path, "type");
    let names: Vec<&str> = match map.get("type") {
        None => return Ok(None),
        Some(Value::String(name)) => vec![name.as_str()],
        Some(Value::Array(items)) => {
            let mut names = Vec::with_capacity(items.len());
            let mut seen: HashSet<&str> = HashSet::with_capacity(items.len());
            for item in items {
                let Value::String(name) = item else {
                    return Err(invalid(&at, "must list type names as strings"));
                };
                if seen.insert(name.as_str()) {
                    names.push(name.as_str());
                }
            }
            names
        }
        Some(_) => return Err(invalid(&at, "must be a string or an array of strings")),
    };
    if let Some(bad) = names.iter().find(|name| !TYPES.contains(name)) {
        let problem = format!(
            "names the unknown type `{}`, and the types are {}",
            excerpt(bad),
            TYPES.join(", ")
        );
        return Err(invalid(&at, &problem));
    }
    Ok(Some(names))
}

/// The types that the keywords of `map` imply when `type` is absent.
fn inferred(map: &Map<String, Value>) -> Vec<&'static str> {
    let has = |keys: &[&str]| keys.iter().any(|key| map.contains_key(*key));
    let mut names = Vec::new();
    if has(&["properties", "required", "additionalProperties"]) {
        names.push("object");
    }
    if has(&["items", "prefixItems", "minItems", "maxItems"]) {
        names.push("array");
    }
    if has(&["minLength", "maxLength"]) {
        names.push("string");
    }
    names
}

/// True when `value` has JSON Schema type `name`. A whole number is an integer.
fn type_matches(value: &Value, name: &str) -> bool {
    match name {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "integer" => {
            value.is_i64()
                || value.is_u64()
                || value
                    .as_f64()
                    .is_some_and(|f| f.is_finite() && f.fract() == 0.0)
        }
        "number" => value.is_number(),
        "string" => value.is_string(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        _ => false,
    }
}

/// The union of `members`, or an empty language with `reason` when there are none.
fn union_or(members: Vec<Lang>, reason: impl FnOnce() -> String) -> Lang {
    if members.is_empty() {
        Lang::Empty(reason())
    } else {
        union(members)
    }
}

#[cfg(test)]
mod tests {
    use crate::inference::grammar::GrammarErrorKind;
    use crate::inference::grammar::json_schema::fixture::{check, dfa, error, opts};

    #[test]
    fn enum_and_const_match_compact_members_of_any_type() {
        let schema = r#"{"enum": ["red", "a\"b", 1, -2.5, true, null, {"k": [1, 2]}, [1, "x"]]}"#;
        let d = dfa(schema, &opts());
        check(
            &d,
            &[
                r#""red""#,
                r#""a\"b""#,
                "1",
                "-2.5",
                "true",
                "null",
                r#"{"k":[1,2]}"#,
                r#"[1,"x"]"#,
                r#" "red" "#,
            ],
            &[
                r#""blue""#,
                "2",
                "1.0",
                "false",
                r#"{"k": [1, 2]}"#,
                r#"{"k":[2,1]}"#,
                r#"[1, "x"]"#,
                r#""a"b""#,
            ],
        );
        let d = dfa(r#"{"const": {"b": 1, "a": 2}}"#, &opts());
        check(&d, &[r#"{"b":1,"a":2}"#], &[r#"{"a":2,"b":1}"#, "1"]);
    }

    #[test]
    fn enum_with_type_keeps_members_of_that_type() {
        let d = dfa(
            r#"{"type": "string", "enum": ["a", 1, "b", null]}"#,
            &opts(),
        );
        check(&d, &[r#""a""#, r#""b""#], &["1", "null", r#""c""#]);
        let d = dfa(
            r#"{"type": "integer", "enum": [1, 1.0, 1.5, "1"]}"#,
            &opts(),
        );
        check(&d, &["1", "1.0"], &["1.5", r#""1""#]);
        let d = dfa(r#"{"enum": ["x", "y", "x"], "const": "y"}"#, &opts());
        check(&d, &[r#""y""#], &[r#""x""#]);
    }

    #[test]
    fn an_empty_enum_is_an_empty_language_error() {
        for schema in [
            r#"{"enum": []}"#,
            r#"{"type": "string", "enum": [1, 2]}"#,
            r#"{"enum": ["a"], "const": "b"}"#,
        ] {
            let err = error(schema, &opts());
            assert_eq!(err.kind(), GrammarErrorKind::EmptyLanguage, "{schema}");
            assert!(
                err.message().contains("`enum` or `const` at `#`"),
                "{}",
                err.message()
            );
        }
        assert_eq!(
            error(r#"{"enum": 3}"#, &opts()).kind(),
            GrammarErrorKind::Syntax
        );
    }

    #[test]
    fn any_of_and_one_of_are_alternations() {
        for keyword in ["anyOf", "oneOf"] {
            let schema = format!(
                r#"{{"{keyword}": [{{"type": "integer"}}, {{"type": "string", "maxLength": 1}}, {{"type": "integer"}}]}}"#
            );
            let d = dfa(&schema, &opts());
            check(&d, &["3", r#""a""#], &["3.5", r#""ab""#, "null"]);
        }
        let err = error(r#"{"anyOf": []}"#, &opts());
        assert_eq!(err.kind(), GrammarErrorKind::Syntax);
        let d = dfa(r#"{"anyOf": [false, {"type": "null"}]}"#, &opts());
        check(&d, &["null"], &["false"]);
    }

    #[test]
    fn type_arrays_nullable_and_single_all_of() {
        let d = dfa(r#"{"type": ["integer", "null", "integer"]}"#, &opts());
        check(&d, &["7", "null"], &["7.5", r#""7""#]);
        let d = dfa(r#"{"type": "string", "nullable": true}"#, &opts());
        check(&d, &[r#""s""#, "null"], &["1"]);
        let d = dfa(r#"{"type": "string", "nullable": false}"#, &opts());
        check(&d, &[r#""s""#], &["null"]);
        let d = dfa(
            r#"{"allOf": [{"type": "boolean"}], "description": "flag"}"#,
            &opts(),
        );
        check(&d, &["true"], &["1"]);
        let d = dfa(r#"{"enum": ["a"], "nullable": true}"#, &opts());
        check(&d, &[r#""a""#, "null"], &[r#""b""#]);
    }

    #[test]
    fn bad_type_values_are_syntax_errors() {
        for schema in [r#"{"type": "float"}"#, r#"{"type": 1}"#, r#"{"type": [1]}"#] {
            let err = error(schema, &opts());
            assert_eq!(err.kind(), GrammarErrorKind::Syntax, "{schema}");
            assert!(err.message().contains("`#/type`"), "{}", err.message());
        }
        let err = error(r#"{"type": []}"#, &opts());
        assert_eq!(err.kind(), GrammarErrorKind::EmptyLanguage);
    }

    #[test]
    fn enum_and_any_of_member_counts_are_limited() {
        use crate::inference::grammar::json_schema::fixture::{limit_message, lowered};
        let members = |n: usize| (0..n).map(|i| serde_json::json!(i)).collect::<Vec<_>>();
        lowered(&serde_json::json!({"enum": members(1_000)}));
        let msg = limit_message(&serde_json::json!({"enum": members(1_001)}));
        assert!(msg.contains("`enum` at `#/enum`"), "{msg}");
        let schemas = |n: usize| vec![serde_json::json!({"type": "null"}); n];
        lowered(&serde_json::json!({"anyOf": schemas(1_000)}));
        let msg = limit_message(&serde_json::json!({"anyOf": schemas(1_001)}));
        assert!(msg.contains("`anyOf` at `#/anyOf`"), "{msg}");
        let msg = limit_message(&serde_json::json!({"oneOf": schemas(1_001)}));
        assert!(msg.contains("`oneOf` at `#/oneOf`"), "{msg}");
    }
}
