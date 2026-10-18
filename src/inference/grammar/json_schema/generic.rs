//! Any-JSON-value and JSON-object grammars with bounded container nesting.
//!
//! A level-`d` value is a scalar, or an object or array of level `d - 1`
//! values. Level 0 holds scalars only.

use super::walk::Walker;
use super::{scalar, string};
use crate::inference::grammar::ast::Expr;

/// Any JSON value, nesting at most `max_value_depth` containers.
pub(super) fn any_value(w: &mut Walker<'_>) -> Expr {
    let depth = w.opts.max_value_depth;
    value(w, depth)
}

/// Any JSON value nesting at most `depth` containers.
fn value(w: &mut Walker<'_>, depth: u32) -> Expr {
    w.rule(&format!("value-{depth}"), |w| {
        let scalar = any_scalar(w);
        if depth == 0 {
            return scalar;
        }
        let object = object(w, depth);
        let array = array(w, depth);
        Expr::alt(vec![scalar, object, array])
    })
}

/// Any JSON string, number, boolean or null.
fn any_scalar(w: &mut Walker<'_>) -> Expr {
    w.rule("scalar", |w| {
        let items = vec![
            string::any_string(w),
            scalar::number(w),
            scalar::boolean(w),
            scalar::null(),
        ];
        Expr::alt(items)
    })
}

/// Any JSON object nesting at most `depth` containers, itself included.
///
/// `depth` must be 1 or more. A `depth` of 0 counts as 1.
pub(super) fn object(w: &mut Walker<'_>, depth: u32) -> Expr {
    let depth = depth.max(1);
    w.rule(&format!("object-{depth}"), |w| {
        let inner = value(w, depth.saturating_sub(1));
        let key = string::any_string(w);
        let member = member(w, key, inner);
        list(w, "{", "}", member)
    })
}

/// Any JSON array nesting at most `depth` containers, itself included.
fn array(w: &mut Walker<'_>, depth: u32) -> Expr {
    w.rule(&format!("array-{depth}"), |w| {
        let inner = value(w, depth.saturating_sub(1));
        list(w, "[", "]", inner)
    })
}

/// `key ws : ws value`: one object member.
pub(super) fn member(w: &mut Walker<'_>, key: Expr, value: Expr) -> Expr {
    let ws = w.ws();
    Expr::seq(vec![key, ws.clone(), Expr::literal(":"), ws, value])
}

/// `open ws ( close | ( item ws , ws )* item ws close )`: any count of `item`.
///
/// The loop copy and the last copy of `item` start at the same input
/// positions, so determinization tracks both in one state set. A form with
/// the first item outside the loop doubles the automaton per nesting level.
pub(super) fn list(w: &mut Walker<'_>, open: &str, close: &str, item: Expr) -> Expr {
    let item = w.share(item);
    let ws = w.ws();
    let comma = w.comma();
    let body = Expr::seq(vec![
        Expr::repeat(Expr::seq(vec![item.clone(), comma]), 0, None),
        item,
        ws.clone(),
        Expr::literal(close),
    ]);
    Expr::seq(vec![
        Expr::literal(open),
        ws,
        Expr::alt(vec![Expr::literal(close), body]),
    ])
}

#[cfg(test)]
mod tests {
    use crate::inference::grammar::json_schema::fixture::{check, compiled, dfa, opts};
    use crate::inference::grammar::json_schema::{SchemaOptions, json_object_grammar};
    use crate::inference::grammar::{CompileLimits, GrammarErrorKind};

    fn depth(n: u32) -> SchemaOptions {
        SchemaOptions {
            max_value_depth: n,
            ..opts()
        }
    }

    /// `[[...[1]...]]` with `n` arrays.
    fn nested(n: u32) -> String {
        let mut text = "1".to_string();
        for _ in 0..n {
            text = format!("[{text}]");
        }
        text
    }

    const INVALID_JSON: &[&str] = &[
        "[1,]",
        r#"{"a":1,}"#,
        "{'a':1}",
        "{a:1}",
        r#"{"a" 1}"#,
        "[1 2]",
        "01",
        "\"tab\tin string\"",
        "nul",
        "",
        "[",
        r#"{"a":1}}"#,
    ];

    #[test]
    fn any_value_nests_to_the_depth_limit() {
        for n in [0, 1, 3] {
            let d = dfa("{}", &depth(n));
            assert!(d.accepts(nested(n).as_bytes()), "{n}");
            assert!(!d.accepts(nested(n + 1).as_bytes()), "{n}");
        }
        let d = dfa("true", &depth(3));
        check(
            &d,
            &[
                "null",
                "-0.5e+3",
                r#""aé\n""#,
                r#"{"a": [1, {"b": null}], "c": {}}"#,
                "[]",
                " [ true , false ] ",
            ],
            INVALID_JSON,
        );
    }

    #[test]
    fn a_schema_without_type_keywords_is_any_value() {
        let d = dfa(r#"{"description": "free form"}"#, &depth(2));
        check(&d, &["1", r#"{"k": [2]}"#, r#""s""#], INVALID_JSON);
    }

    #[test]
    fn json_object_accepts_objects_only() {
        let o = depth(3);
        let grammar = json_object_grammar(&o);
        let d = compiled(grammar.and_then(|g| g.compile(&CompileLimits::default())));
        check(
            &d,
            &[
                "{}",
                r#"{"a": 1}"#,
                r#"{"a": {"b": [1, 2]}, "c": "d"}"#,
                "\n{\"x\": [true, null]}\n",
            ],
            &[
                "[]",
                "[1]",
                "1",
                r#""s""#,
                "null",
                r#"{"a": {"b": {"c": {"d": 1}}}}"#,
            ],
        );
        for bad in INVALID_JSON {
            assert!(!d.accepts(bad.as_bytes()), "{bad}");
        }
    }

    #[test]
    fn json_object_at_the_default_depth_fits_the_default_limits() {
        let grammar = json_object_grammar(&SchemaOptions::default());
        let d = compiled(grammar.and_then(|g| g.compile(&CompileLimits::default())));
        assert!(d.accepts(br#"{"a": {"b": [{"c": 1}]}}"#));
        assert!(!d.accepts(br#"{"a": {"b": [{"c": [1]}]}}"#));
        assert!(d.num_states() < 4_000, "{} states", d.num_states());
    }

    #[test]
    fn json_object_needs_a_value_depth_of_one() {
        let err = json_object_grammar(&depth(0)).err().map(|e| e.kind());
        assert_eq!(err, Some(GrammarErrorKind::EmptyLanguage));
    }
}
