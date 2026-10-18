//! Keyword policy: which schema keywords the compiler enforces, ignores or refuses.

use serde_json::{Map, Value};

use super::refs::child;
use crate::inference::grammar::error::{GrammarError, GrammarErrorKind};

/// Validation keywords that a grammar cannot enforce.
///
/// With `strict_keywords` each one is an `Unsupported` error. Without it the
/// compiler ignores them, so the language becomes a superset of the schema.
/// A multi-member `allOf`, an external `$ref` and an anchor `$ref` follow the same policy.
pub const UNSUPPORTED_KEYWORDS: &[&str] = &[
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "pattern",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "minProperties",
    "maxProperties",
    "patternProperties",
    "propertyNames",
    "not",
    "if",
    "then",
    "else",
    "dependentRequired",
    "dependentSchemas",
    "unevaluatedProperties",
    "unevaluatedItems",
];

/// Annotation keywords. The compiler ignores them, as it ignores every unknown keyword.
pub const ANNOTATION_KEYWORDS: &[&str] = &[
    "$schema",
    "$id",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "format",
    "name",
    "strict",
];

/// Keywords whose language replaces the rest of the schema object.
pub(super) const COMBINATORS: &[&str] = &["$ref", "allOf", "anyOf", "oneOf"];

/// Keywords that build a language from types.
///
/// In strict mode one of them beside a combinator is an error,
/// because the compiler does not intersect languages.
pub(super) const TYPE_KEYWORDS: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "prefixItems",
    "additionalItems",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "enum",
    "const",
];

/// In strict mode, an `Unsupported` error for the first unsupported keyword of `map`.
pub(super) fn check(
    map: &Map<String, Value>,
    path: &str,
    strict: bool,
) -> Result<(), GrammarError> {
    if !strict {
        return Ok(());
    }
    match map
        .keys()
        .find(|key| UNSUPPORTED_KEYWORDS.contains(&key.as_str()))
    {
        Some(key) => Err(unsupported(&child(path, key), &format!("`{key}`"))),
        None => Ok(()),
    }
}

/// The `Unsupported` error for `what` at the JSON pointer `pointer`.
pub(super) fn unsupported(pointer: &str, what: &str) -> GrammarError {
    GrammarError::new(
        GrammarErrorKind::Unsupported,
        None,
        format!(
            "schema keyword {what} at `{pointer}` is not supported, \
             because the grammar cannot enforce it. \
             Remove it or set strict_keywords to false."
        ),
    )
}

/// The `Unsupported` error for combinator `first` beside keyword `other` at `path`.
pub(super) fn combined(path: &str, first: &str, other: &str) -> GrammarError {
    GrammarError::new(
        GrammarErrorKind::Unsupported,
        None,
        format!(
            "schema at `{path}` combines `{first}` with `{other}`, \
             and the compiler does not intersect them. \
             Move `{other}` into the `{first}` schema or set strict_keywords to false."
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::grammar::json_schema::SchemaOptions;
    use crate::inference::grammar::json_schema::fixture::{dfa, error, opts};

    fn loose() -> SchemaOptions {
        SchemaOptions {
            strict_keywords: false,
            ..opts()
        }
    }

    /// A schema with `keyword` on the property `age`, and the value it takes.
    fn with_keyword(keyword: &str) -> String {
        let value = match keyword {
            "not" | "if" | "then" | "else" | "contains" | "propertyNames" => "{}",
            "unevaluatedProperties" | "unevaluatedItems" | "uniqueItems" => "false",
            "pattern" => "\"^a\"",
            "patternProperties" | "dependentRequired" | "dependentSchemas" => "{}",
            _ => "1",
        };
        format!(
            r#"{{"type": "object", "properties": {{"age": {{"type": "integer", "{keyword}": {value}}}}}, "required": ["age"]}}"#
        )
    }

    #[test]
    fn every_unsupported_keyword_errors_in_strict_mode_with_its_pointer() {
        for keyword in UNSUPPORTED_KEYWORDS {
            let schema = with_keyword(keyword);
            let err = error(&schema, &opts());
            assert_eq!(err.kind(), GrammarErrorKind::Unsupported, "{keyword}");
            let pointer = format!("`#/properties/age/{keyword}`");
            assert!(err.message().contains(&pointer), "{}", err.message());
            assert!(
                err.message().contains("strict_keywords"),
                "{}",
                err.message()
            );
        }
    }

    #[test]
    fn every_unsupported_keyword_compiles_in_loose_mode() {
        for keyword in UNSUPPORTED_KEYWORDS {
            let d = dfa(&with_keyword(keyword), &loose());
            assert!(d.accepts(br#"{"age": 7}"#), "{keyword}");
            assert!(!d.accepts(br#"{"age": "x"}"#), "{keyword}");
        }
    }

    #[test]
    fn annotation_keywords_are_ignored() {
        for keyword in ANNOTATION_KEYWORDS {
            let schema = format!(r#"{{"type": "integer", "{keyword}": "text"}}"#);
            let d = dfa(&schema, &opts());
            assert!(d.accepts(b"12"), "{keyword}");
            assert!(!d.accepts(b"\"12\""), "{keyword}");
        }
        let d = dfa(r#"{"type": "boolean", "x-vendor": {"a": 1}}"#, &opts());
        assert!(d.accepts(b"true"));
    }

    #[test]
    fn multi_member_all_of_and_external_refs_follow_the_policy() {
        let all_of = r#"{"allOf": [{"type": "integer"}, {"type": "number"}]}"#;
        let err = error(all_of, &opts());
        assert_eq!(err.kind(), GrammarErrorKind::Unsupported);
        assert!(err.message().contains("`#/allOf`"), "{}", err.message());
        assert!(dfa(all_of, &loose()).accepts(b"3"));
        let external = r#"{"properties": {"a": {"$ref": "other.json#/x"}}}"#;
        let err = error(external, &opts());
        assert_eq!(err.kind(), GrammarErrorKind::Unsupported);
        assert!(
            err.message().contains("`#/properties/a/$ref`"),
            "{}",
            err.message()
        );
        let shallow = SchemaOptions {
            max_value_depth: 2,
            ..loose()
        };
        let d = dfa(external, &shallow);
        assert!(d.accepts(br#"{"a": [1, {"b": null}]}"#));
    }
}
