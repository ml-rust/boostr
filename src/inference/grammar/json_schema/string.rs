//! JSON string grammars: any string, length bounds and exact literals.
//!
//! A character is a Unicode scalar other than `"`, `\` and U+0000 to U+001F,
//! or one escape: `\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t` or `\uXXXX`.
//! Length bounds count an escape as one character. A surrogate pair written
//! as two `\uXXXX` escapes therefore counts as two.

use serde_json::{Map, Value};

use super::scalar::class;
use super::walk::{Lang, Walker, compact, count};
use crate::inference::grammar::ast::Expr;
use crate::inference::grammar::error::{GrammarError, GrammarErrorKind};

/// Any JSON string.
pub(super) fn any_string(w: &mut Walker<'_>) -> Expr {
    w.rule("string", |w| {
        let unit = char_unit(w);
        quoted(Expr::repeat(unit, 0, None))
    })
}

/// The string schema `map` at `path`: `minLength` and `maxLength` bound the length.
pub(super) fn schema(
    w: &mut Walker<'_>,
    map: &Map<String, Value>,
    path: &str,
) -> Result<Lang, GrammarError> {
    let min = count(map, "minLength", path)?;
    let max = count(map, "maxLength", path)?;
    if let (Some(lo), Some(hi)) = (min, max)
        && lo > hi
    {
        return Err(GrammarError::new(
            GrammarErrorKind::EmptyLanguage,
            None,
            format!(
                "`minLength` {lo} is above `maxLength` {hi} at `{path}`. \
                 Set minLength to at most maxLength."
            ),
        ));
    }
    let lo = min.unwrap_or(0);
    if lo == 0 && max.is_none() {
        return Ok(Lang::Expr(any_string(w)));
    }
    let name = match max {
        Some(hi) => format!("string-{lo}-{hi}"),
        None => format!("string-{lo}-"),
    };
    Ok(Lang::Expr(w.rule(&name, |w| {
        let unit = char_unit(w);
        quoted(Expr::repeat(unit, lo, max))
    })))
}

/// The exact compact JSON text of the string `text`, quotes and escapes included.
pub(super) fn literal(text: &str, path: &str) -> Result<Expr, GrammarError> {
    Ok(Expr::literal(compact(
        &Value::String(text.to_string()),
        path,
    )?))
}

/// One string character: a plain scalar or one escape.
fn char_unit(w: &mut Walker<'_>) -> Expr {
    w.rule("char", |_| {
        let plain = Expr::Class {
            negated: true,
            ranges: vec![('"', '"'), ('\\', '\\'), ('\u{0}', '\u{1f}')],
        };
        let simple: Vec<(char, char)> = "\"\\/bfnrt".chars().map(|c| (c, c)).collect();
        let hex = class(&[('0', '9'), ('a', 'f'), ('A', 'F')]);
        let unicode = Expr::seq(vec![Expr::literal("u"), Expr::repeat(hex, 4, Some(4))]);
        let escape = Expr::seq(vec![
            Expr::literal("\\"),
            Expr::alt(vec![class(&simple), unicode]),
        ]);
        Expr::alt(vec![plain, escape])
    })
}

/// `"` `inner` `"`.
fn quoted(inner: Expr) -> Expr {
    Expr::seq(vec![Expr::literal("\""), inner, Expr::literal("\"")])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::grammar::json_schema::fixture::{check, dfa, error, opts};

    const PLAIN: &[&str] = &[
        r#""""#,
        r#""hello""#,
        r#""é 世 😀""#,
        r#""\"\\\/\b\f\n\r\t""#,
        r#""é😀\u0000""#,
        "\"\u{7f}\u{10ffff}\"",
    ];

    #[test]
    fn strings_accept_scalars_and_escapes() {
        let d = dfa(r#"{"type": "string"}"#, &opts());
        check(
            &d,
            PLAIN,
            &[
                "\"",
                "\"abc",
                "abc\"",
                "\"a\nb\"",
                "\"tab\there\"",
                "\"\u{1f}\"",
                r#""\x41""#,
                r#""\u00g0""#,
                r#""\u12""#,
                r#""\""#,
                "'single'",
                r#""a""b""#,
            ],
        );
        assert!(!d.accepts(b"\"\xff\""));
        assert!(!d.accepts(b"\"\xed\xa0\x80\""));
        assert!(!d.accepts(b"\"\xc0\xaf\""));
    }

    #[test]
    fn length_bounds_count_an_escape_as_one_character() {
        let d = dfa(
            r#"{"type": "string", "minLength": 2, "maxLength": 3}"#,
            &opts(),
        );
        check(
            &d,
            &[r#""ab""#, r#""abc""#, r#""\nA""#, r#""é世""#, r#""a\"b""#],
            &[r#""a""#, r#""""#, r#""abcd""#, r#""\n\n\n\n""#, r#""A""#],
        );
        let d = dfa(r#"{"type": "string", "minLength": 1}"#, &opts());
        check(&d, &[r#""a""#, r#""abcdefgh""#], &[r#""""#]);
        let d = dfa(r#"{"type": "string", "maxLength": 0}"#, &opts());
        check(&d, &[r#""""#], &[r#""a""#]);
        let d = dfa(r#"{"maxLength": 1}"#, &opts());
        check(&d, &[r#""a""#], &[r#""ab""#, "1"]);
    }

    #[test]
    fn min_length_above_max_length_is_an_error() {
        let err = error(
            r#"{"type": "string", "minLength": 4, "maxLength": 3}"#,
            &opts(),
        );
        assert_eq!(err.kind(), GrammarErrorKind::EmptyLanguage);
        assert!(err.message().contains("`minLength` 4"), "{}", err.message());
    }

    #[test]
    fn literals_escape_per_json() {
        let at = |text: &str| literal(text, "#").ok();
        assert_eq!(at("a\"b\\c"), Some(Expr::literal(r#""a\"b\\c""#)));
        assert_eq!(at("\n\u{1}"), Some(Expr::literal(r#""\n\u0001""#)));
        assert_eq!(at("é/"), Some(Expr::literal("\"é/\"")));
    }
}
