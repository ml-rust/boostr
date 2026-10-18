//! Postfix repeat operators: `*`, `+`, `?` and `{n}`, `{n,}`, `{,m}`, `{n,m}`.

use super::ast::Span;
use super::error::{GrammarError, GrammarErrorKind};
use super::lexer::{Cursor, describe};

/// Consume the postfix operator `op` at the cursor. Returns `(min, max)`.
///
/// `{,m}` reads as `{0,m}`. A count above `max_repeat` is an error.
pub(crate) fn postfix(
    cur: &mut Cursor,
    op: char,
    max_repeat: u32,
) -> Result<(u32, Option<u32>), GrammarError> {
    let open = cur.span();
    cur.bump();
    match op {
        '*' => return Ok((0, None)),
        '+' => return Ok((1, None)),
        '?' => return Ok((0, Some(1))),
        _ => {}
    }
    cur.skip_inline_space();
    let min = match cur.peek() {
        Some(c) if c.is_ascii_digit() => count(cur, max_repeat)?,
        Some(',') => 0,
        found => return Err(brace_error(open, "a number or `,`", found)),
    };
    cur.skip_inline_space();
    let max = if cur.eat('}') {
        Some(min)
    } else if cur.eat(',') {
        cur.skip_inline_space();
        let max = match cur.peek() {
            Some(c) if c.is_ascii_digit() => Some(count(cur, max_repeat)?),
            _ => None,
        };
        cur.skip_inline_space();
        if !cur.eat('}') {
            return Err(brace_error(open, "`}`", cur.peek()));
        }
        max
    } else {
        return Err(brace_error(open, "`}` or `,`", cur.peek()));
    };
    if let Some(max) = max
        && max < min
    {
        return Err(GrammarError::syntax(
            open,
            format!(
                "repeat `{{{min},{max}}}` has a maximum below its minimum. \
                 Write `{{n,m}}` with n <= m."
            ),
        ));
    }
    Ok((min, max))
}

fn brace_error(open: Span, expected: &str, found: Option<char>) -> GrammarError {
    GrammarError::syntax(
        open,
        format!(
            "repeat expects {expected}, found {}. \
             Write `{{n}}`, `{{n,}}`, `{{,m}}` or `{{n,m}}`.",
            describe(found)
        ),
    )
}

/// Consume a repeat count and check it against `max_repeat`.
fn count(cur: &mut Cursor, max_repeat: u32) -> Result<u32, GrammarError> {
    let at = cur.span();
    let number = cur.number();
    match number.value {
        Some(value) if value <= max_repeat => Ok(value),
        _ => Err(GrammarError::at(
            GrammarErrorKind::LimitExceeded,
            at,
            format!(
                "repeat count `{}` exceeds max_repeat {max_repeat}. \
                 Use `*` or `+` for an unbounded count.",
                number.text
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &str) -> Result<(u32, Option<u32>), GrammarError> {
        let mut cur = Cursor::new(src);
        let op = cur.peek().unwrap_or(' ');
        postfix(&mut cur, op, 1_000)
    }

    #[test]
    fn every_form_reads_its_bounds() {
        for (src, want) in [
            ("*", (0, None)),
            ("+", (1, None)),
            ("?", (0, Some(1))),
            ("{2}", (2, Some(2))),
            ("{2,}", (2, None)),
            ("{2,4}", (2, Some(4))),
            ("{,3}", (0, Some(3))),
            ("{ 2 , 4 }", (2, Some(4))),
            ("{,}", (0, None)),
            ("{0}", (0, Some(0))),
            ("{1000}", (1000, Some(1000))),
        ] {
            assert_eq!(run(src), Ok(want), "{src}");
        }
    }

    #[test]
    fn malformed_braces_are_errors() {
        for src in ["{", "{2", "{2,", "{2,4", "{x}", "{2x}", "{-1}", "{2,,3}"] {
            let err = run(src).err().map(|e| e.kind());
            assert_eq!(err, Some(GrammarErrorKind::Syntax), "{src}");
        }
    }

    #[test]
    fn reversed_bounds_name_the_repeat() {
        let msg = run("{3,2}").err().map(|e| e.message().to_string());
        assert!(msg.unwrap_or_default().contains("`{3,2}`"));
    }

    #[test]
    fn counts_above_the_limit_are_errors() {
        for (src, column) in [
            ("{1001}", 2),
            ("{0,1001}", 4),
            ("{ 2 , 5000 }", 7),
            ("{99999999999999999999999}", 2),
        ] {
            let err = run(src).err();
            assert_eq!(
                err.as_ref().map(GrammarError::kind),
                Some(GrammarErrorKind::LimitExceeded),
                "{src}"
            );
            assert_eq!(
                err.and_then(|e| e.span()).map(|s| s.column),
                Some(column),
                "{src}"
            );
        }
    }
}
