//! Null, boolean, integer and number grammars per RFC 8259.

use super::walk::Walker;
use crate::inference::grammar::ast::Expr;

/// `null`.
pub(super) fn null() -> Expr {
    Expr::literal("null")
}

/// `true` or `false`.
pub(super) fn boolean(w: &mut Walker<'_>) -> Expr {
    w.rule("boolean", |_| {
        Expr::alt(vec![Expr::literal("true"), Expr::literal("false")])
    })
}

/// `-?(0|[1-9][0-9]*)`: no fraction and no exponent. `-0` matches.
pub(super) fn integer(w: &mut Walker<'_>) -> Expr {
    w.rule("integer", |_| {
        let magnitude = Expr::alt(vec![
            Expr::literal("0"),
            Expr::seq(vec![
                class(&[('1', '9')]),
                Expr::repeat(class(&[('0', '9')]), 0, None),
            ]),
        ]);
        Expr::seq(vec![
            Expr::repeat(Expr::literal("-"), 0, Some(1)),
            magnitude,
        ])
    })
}

/// `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`.
pub(super) fn number(w: &mut Walker<'_>) -> Expr {
    w.rule("number", |w| {
        let int = integer(w);
        let digits = digits(w);
        let fraction = Expr::seq(vec![Expr::literal("."), digits.clone()]);
        let exponent = Expr::seq(vec![
            class(&[('e', 'e'), ('E', 'E')]),
            Expr::repeat(class(&[('+', '+'), ('-', '-')]), 0, Some(1)),
            digits,
        ]);
        Expr::seq(vec![
            int,
            Expr::repeat(fraction, 0, Some(1)),
            Expr::repeat(exponent, 0, Some(1)),
        ])
    })
}

/// `[0-9]+`.
fn digits(w: &mut Walker<'_>) -> Expr {
    w.rule("digits", |_| Expr::repeat(class(&[('0', '9')]), 1, None))
}

/// One code point inside the inclusive `ranges`.
pub(super) fn class(ranges: &[(char, char)]) -> Expr {
    Expr::Class {
        negated: false,
        ranges: ranges.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use crate::inference::grammar::json_schema::fixture::{check, dfa, opts};

    #[test]
    fn numbers_follow_rfc_8259() {
        let d = dfa(r#"{"type": "number"}"#, &opts());
        check(
            &d,
            &[
                "0", "-0", "7", "-12", "1.5", "-0.25", "1e5", "1E+5", "2.5e-10", "0e0", "10",
                " 3 ", "1.0",
            ],
            &[
                "01", "-01", "1.", ".5", "+1", "1e", "1e+", "--1", "1.5.2", "0x1", "NaN",
                "Infinity", "\"1\"", "1 2", "", "-",
            ],
        );
    }

    #[test]
    fn integers_reject_fractions_and_exponents() {
        let d = dfa(r#"{"type": "integer"}"#, &opts());
        check(
            &d,
            &["0", "-0", "42", "-1234567890123456789012"],
            &["1.5", "1.0", "1e5", "01", "-", "true", "\"3\""],
        );
    }

    #[test]
    fn booleans_and_null_match_their_literals() {
        let d = dfa(r#"{"type": "boolean"}"#, &opts());
        check(
            &d,
            &["true", "false", "  true  "],
            &["True", "1", "null", "tru", "\"true\""],
        );
        let d = dfa(r#"{"type": "null"}"#, &opts());
        check(
            &d,
            &["null", "\nnull\n"],
            &["Null", "nil", "0", "\"null\"", "   null"],
        );
    }

    #[test]
    fn whitespace_runs_stay_within_the_bound() {
        let compact = crate::inference::grammar::json_schema::SchemaOptions {
            max_whitespace: 0,
            ..opts()
        };
        check(
            &dfa(r#"{"type": "null"}"#, &compact),
            &["null"],
            &[" null", "null "],
        );
        check(
            &dfa(r#"{"type": "null"}"#, &opts()),
            &["\t\nnull \n"],
            &["\r\nnull", "null   "],
        );
    }
}
