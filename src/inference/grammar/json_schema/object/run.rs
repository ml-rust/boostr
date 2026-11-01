//! Comma-correct runs over the declared properties of an object, in order.
//!
//! A required property must appear, and an optional one can be absent.
//!
//! With a required property the run is flat. Each optional property before
//! the first required one is `(pair ws , ws)?`. Each property after it is
//! `ws , ws pair`, optional for an optional property.
//!
//! With no required property the run can be empty, and `chain` builds its
//! non-empty forms left to right: `run(j + 1) = run(j) (ws , ws pair(j))? | pair(j)`.
//! Each pair appears twice, so the grammar stays linear in the property count.
//! Each step nests the tree two levels deeper. So past `BLOCK` properties,
//! `optional_run` chains each block of pairs, then chains the block runs.

use super::super::generic::member;
use super::super::limits::{BLOCK, blocks};
use super::super::string;
use super::super::walk::Walker;
use crate::inference::grammar::ast::Expr;
use crate::inference::grammar::error::GrammarError;

/// One property in emission order.
pub(super) struct Prop<'s> {
    /// The property name.
    pub(super) key: &'s str,
    /// The value grammar.
    pub(super) value: Expr,
    /// True when the property must appear.
    pub(super) required: bool,
}

/// The non-empty runs over `props` in order, at the object pointer `path`.
///
/// `None` when `props` is empty. When no property is required, the caller
/// adds the empty run.
pub(super) fn run(
    w: &mut Walker<'_>,
    props: Vec<Prop<'_>>,
    path: &str,
) -> Result<Option<Expr>, GrammarError> {
    if props.iter().any(|prop| prop.required) {
        required_run(w, props, path).map(Some)
    } else {
        optional_run(w, props, path)
    }
}

/// The flat run over `props`, which hold at least one required property.
fn required_run(
    w: &mut Walker<'_>,
    props: Vec<Prop<'_>>,
    path: &str,
) -> Result<Expr, GrammarError> {
    let mut items = Vec::with_capacity(props.len());
    let mut after_required = false;
    for prop in props {
        let key = string::literal(prop.key, path)?;
        let pair = member(w, key, prop.value);
        let item = if after_required {
            let comma = w.comma();
            Expr::seq(vec![comma, pair])
        } else if prop.required {
            pair
        } else {
            let comma = w.comma();
            Expr::seq(vec![pair, comma])
        };
        items.push(if prop.required {
            item
        } else {
            Expr::repeat(item, 0, Some(1))
        });
        after_required |= prop.required;
    }
    Ok(Expr::seq(items))
}

/// The non-empty runs over `props`, all optional. `None` when `props` is empty.
fn optional_run(
    w: &mut Walker<'_>,
    props: Vec<Prop<'_>>,
    path: &str,
) -> Result<Option<Expr>, GrammarError> {
    let mut pairs = Vec::with_capacity(props.len());
    for prop in props {
        let key = string::literal(prop.key, path)?;
        let value = w.share(prop.value);
        pairs.push(member(w, key, value));
    }
    if pairs.len() <= BLOCK {
        return Ok(chain(w, pairs));
    }
    let mut runs = Vec::with_capacity(pairs.len().div_ceil(BLOCK));
    for block in blocks(pairs) {
        if let Some(run) = chain(w, block) {
            runs.push(w.share(run));
        }
    }
    Ok(chain(w, runs))
}

/// The non-empty comma-separated runs over `items` in order, each item optional.
///
/// Every item after the first appears twice, so callers pass small or shared items.
fn chain(w: &mut Walker<'_>, items: Vec<Expr>) -> Option<Expr> {
    let mut run: Option<Expr> = None;
    for item in items {
        run = Some(match run {
            None => item,
            Some(before) => {
                let comma = w.comma();
                let link = Expr::repeat(Expr::seq(vec![comma, item.clone()]), 0, Some(1));
                Expr::alt(vec![Expr::seq(vec![before, link]), item])
            }
        });
    }
    run
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::grammar::ExpandLimits;
    use crate::inference::grammar::json_schema::SchemaOptions;
    use crate::inference::grammar::json_schema::fixture::{check, dfa, opts};
    use serde_json::{Map, Value, json};

    /// `a`, `c`, `d`, `e` optional and `b` required, in that order.
    const FLAT: &str = r#"{
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "string"},
            "c": {"type": "boolean"},
            "d": {"type": "null"},
            "e": {"type": "number"}
        },
        "required": ["b"]
    }"#;

    const MEMBERS: [(&str, &str); 5] = [
        ("a", "1"),
        ("b", "\"x\""),
        ("c", "true"),
        ("d", "null"),
        ("e", "2.5"),
    ];

    /// The object text of `keys` in the given order.
    fn text(keys: &[&str]) -> String {
        let parts: Vec<String> = keys
            .iter()
            .filter_map(|key| MEMBERS.iter().find(|(k, _)| k == key))
            .map(|(k, v)| format!("\"{k}\":{v}"))
            .collect();
        format!("{{{}}}", parts.join(","))
    }

    #[test]
    fn every_subset_of_optional_properties_matches_in_order_only() {
        let d = dfa(FLAT, &opts());
        let optional = ["a", "c", "d", "e"];
        for mask in 0..16u32 {
            let chosen: Vec<&str> = MEMBERS
                .iter()
                .map(|(k, _)| *k)
                .filter(|k| {
                    *k == "b"
                        || optional
                            .iter()
                            .position(|o| o == k)
                            .is_some_and(|i| mask & (1 << i) != 0)
                })
                .collect();
            let ordered = text(&chosen);
            assert!(d.accepts(ordered.as_bytes()), "must accept {ordered}");
            if chosen.len() > 1 {
                let reversed: Vec<&str> = chosen.iter().rev().copied().collect();
                let wrong = text(&reversed);
                assert!(!d.accepts(wrong.as_bytes()), "must reject {wrong}");
            }
            let without: Vec<&str> = chosen.iter().copied().filter(|k| *k != "b").collect();
            let missing = text(&without);
            assert!(!d.accepts(missing.as_bytes()), "must reject {missing}");
            let extra = ordered.replacen('{', "{\"z\":1,", 1).replace(",}", "}");
            assert!(!d.accepts(extra.as_bytes()), "must reject {extra}");
        }
    }

    #[test]
    fn commas_stay_correct_when_the_first_property_is_optional() {
        let d = dfa(
            r#"{"properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["b"]}"#,
            &opts(),
        );
        check(
            &d,
            &[r#"{"b":1}"#, r#"{"a":1,"b":2}"#, r#"{"a": 1 , "b":2}"#],
            &[
                r#"{,"b":1}"#,
                r#"{"a":1}"#,
                r#"{"a":1,,"b":2}"#,
                r#"{"a":1"b":2}"#,
                r#"{"a":1,}"#,
                "{}",
            ],
        );
    }

    #[test]
    fn an_object_with_no_required_property_accepts_empty() {
        let d = dfa(
            r#"{"properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}"#,
            &opts(),
        );
        check(
            &d,
            &["{}", "{ }", r#"{"a":1}"#, r#"{"b":2}"#, r#"{"a":1,"b":2}"#],
            &[r#"{"b":2,"a":1}"#, "{,}", r#"{"a":1,}"#],
        );
        check(
            &dfa(r#"{"type": "object"}"#, &opts()),
            &["{}"],
            &[r#"{"a":1}"#],
        );
    }

    /// `n` properties of type `ty` named `p000`, `p001`, ... so that name order is index order.
    fn props(n: usize, ty: &str) -> Map<String, Value> {
        (0..n)
            .map(|i| (format!("p{i:03}"), json!({ "type": ty })))
            .collect()
    }

    /// The object text with a `null` member for each index in `keys`, in that order.
    fn nulls(keys: &[usize]) -> String {
        let parts: Vec<String> = keys.iter().map(|i| format!("\"p{i:03}\":null")).collect();
        format!("{{{}}}", parts.join(","))
    }

    fn compact() -> SchemaOptions {
        SchemaOptions {
            max_whitespace: 0,
            ..opts()
        }
    }

    #[test]
    fn optional_runs_past_one_block_keep_their_language() {
        let n = 2 * BLOCK + 6;
        let schema = json!({"type": "object", "properties": props(n, "null")});
        let d = dfa(&schema.to_string(), &compact());
        let last = n - 1;
        let all: Vec<usize> = (0..n).collect();
        let accept = [
            nulls(&[]),
            nulls(&[0]),
            nulls(&[last]),
            nulls(&[BLOCK - 1, BLOCK]),
            nulls(&[0, BLOCK - 1, BLOCK, 2 * BLOCK, last]),
            nulls(&[BLOCK, 2 * BLOCK - 1]),
            nulls(&all),
        ];
        let reject = [
            nulls(&[BLOCK, BLOCK - 1]),
            nulls(&[BLOCK, BLOCK]),
            nulls(&[last, 0]),
            nulls(&[BLOCK - 1]).replace('}', ",}"),
            nulls(&[BLOCK]).replacen('{', "{,", 1),
            nulls(&[BLOCK - 1, BLOCK]).replace(',', ""),
        ];
        for text in &accept {
            assert!(d.accepts(text.as_bytes()), "must accept {text}");
        }
        for text in &reject {
            assert!(!d.accepts(text.as_bytes()), "must reject {text}");
        }
    }

    #[test]
    fn a_late_required_property_keeps_every_optional_one_before_it() {
        let n = BLOCK + 8;
        let required = BLOCK + 3;
        let mut schema = json!({"type": "object", "properties": props(n, "null")});
        schema["required"] = json!([format!("p{required:03}")]);
        let d = dfa(&schema.to_string(), &compact());
        let accept = [
            nulls(&[required]),
            nulls(&[0, required]),
            nulls(&[0, 1, BLOCK, required, required + 1, n - 1]),
            nulls(&(0..n).collect::<Vec<_>>()),
        ];
        let reject = [
            nulls(&[]),
            nulls(&[0]),
            nulls(&[required, 0]),
            nulls(&[1, 0, required]),
            nulls(&[0, required]).replace(',', ",,"),
        ];
        for text in &accept {
            assert!(d.accepts(text.as_bytes()), "must accept {text}");
        }
        for text in &reject {
            assert!(!d.accepts(text.as_bytes()), "must reject {text}");
        }
    }

    #[test]
    fn a_deep_optional_object_builds_clones_and_drops_without_overflow() {
        use crate::inference::grammar::json_schema::fixture::lowered;
        let grammar = lowered(&json!({"type": "object", "properties": props(1_000, "integer")}));
        let copy = grammar.clone();
        assert_eq!(copy, grammar);
        drop(copy);
        let expanded = match grammar.expand(&ExpandLimits::default()) {
            Ok(expr) => expr,
            Err(err) => panic!("expansion error: {err}"),
        };
        let copy = expanded.clone();
        assert_eq!(copy, expanded);
        drop(copy);
        drop(expanded);
        drop(grammar);
    }
}
