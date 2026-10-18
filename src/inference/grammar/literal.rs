//! Scanners for string literals, character classes and backslash escapes.
//!
//! A literal or class must close on the line it opens on.

use super::ast::{Expr, Span};
use super::error::GrammarError;
use super::lexer::{Cursor, describe};

/// The escapes literals and classes accept.
const ESCAPES: &str = r#"\n \r \t \\ \" \[ \] \- \^ \xHH \uHHHH \UHHHHHHHH"#;

impl Cursor {
    /// Consume a `"..."` literal and return its decoded text.
    pub fn literal(&mut self) -> Result<String, GrammarError> {
        let open = self.span();
        self.bump();
        let mut text = String::new();
        loop {
            match self.peek() {
                None | Some('\n') => {
                    return Err(GrammarError::syntax(
                        open,
                        format!(
                            "literal has no closing `\"` before {}. \
                             Close it on its own line and write a newline as `\\n`.",
                            describe(self.peek())
                        ),
                    ));
                }
                Some('"') => {
                    self.bump();
                    return Ok(text);
                }
                Some('\\') => text.push(self.escape()?),
                Some(c) => {
                    self.bump();
                    text.push(c);
                }
            }
        }
    }

    /// Consume a `[...]` class.
    pub fn class(&mut self) -> Result<Expr, GrammarError> {
        let open = self.span();
        self.bump();
        let negated = self.eat('^');
        let mut ranges = Vec::new();
        while !self.eat(']') {
            let start = self.span();
            let lo = self.class_char(open)?;
            let ranged = self.peek() == Some('-') && !matches!(self.peek_at(1), Some(']') | None);
            let hi = if ranged {
                self.bump();
                self.class_char(open)?
            } else {
                lo
            };
            if hi < lo {
                return Err(GrammarError::syntax(
                    start,
                    format!(
                        "class range `{}-{}` ends before it starts. \
                         Write the lower code point first.",
                        lo.escape_debug(),
                        hi.escape_debug()
                    ),
                ));
            }
            ranges.push((lo, hi));
        }
        if ranges.is_empty() && !negated {
            return Err(GrammarError::syntax(
                open,
                "class `[]` is empty and matches nothing. Write `.` to match any character.",
            ));
        }
        Ok(Expr::Class { negated, ranges })
    }

    /// Consume one class member: a raw character or an escape.
    fn class_char(&mut self, open: Span) -> Result<char, GrammarError> {
        match self.peek() {
            None | Some('\n') => Err(GrammarError::syntax(
                open,
                format!(
                    "class has no closing `]` before {}. Write a literal `]` as `\\]`.",
                    describe(self.peek())
                ),
            )),
            Some('\\') => self.escape(),
            Some(c) => {
                self.bump();
                Ok(c)
            }
        }
    }

    /// Consume a backslash escape and return the character it stands for.
    fn escape(&mut self) -> Result<char, GrammarError> {
        let at = self.span();
        self.bump();
        let Some(c) = self.bump() else {
            return Err(GrammarError::syntax(
                at,
                format!("escape `\\` has no character after it. Supported escapes: {ESCAPES}."),
            ));
        };
        match c {
            'n' => Ok('\n'),
            'r' => Ok('\r'),
            't' => Ok('\t'),
            '\\' | '"' | '[' | ']' | '-' | '^' => Ok(c),
            'x' => self.hex(at, c, 2),
            'u' => self.hex(at, c, 4),
            'U' => self.hex(at, c, 8),
            other => Err(GrammarError::syntax(
                at,
                format!(
                    "unknown escape `\\{}`. Supported escapes: {ESCAPES}.",
                    other.escape_debug()
                ),
            )),
        }
    }

    /// Consume exactly `digits` hex digits and return that code point.
    fn hex(&mut self, at: Span, letter: char, digits: usize) -> Result<char, GrammarError> {
        let mut text = String::new();
        let mut value = Some(0u32);
        for _ in 0..digits {
            let Some((c, digit)) = self.peek().and_then(|c| c.to_digit(16).map(|d| (c, d))) else {
                return Err(GrammarError::syntax(
                    at,
                    format!("escape `\\{letter}{text}` needs {digits} hex digits."),
                ));
            };
            self.bump();
            text.push(c);
            value = value
                .and_then(|v| v.checked_mul(16))
                .and_then(|v| v.checked_add(digit));
        }
        value.and_then(char::from_u32).ok_or_else(|| {
            GrammarError::syntax(
                at,
                format!(
                    "escape `\\{letter}{text}` is not a Unicode scalar value. \
                     Use a code point up to 10FFFF outside D800-DFFF."
                ),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(src: &str) -> Result<Expr, GrammarError> {
        Cursor::new(src).class()
    }

    fn message(result: Result<impl std::fmt::Debug, GrammarError>) -> String {
        result
            .err()
            .map(|e| e.message().to_string())
            .unwrap_or_default()
    }

    #[test]
    fn literal_decodes_every_escape() {
        let src = r#""a\n\r\t\\\"\[\]\-\^\x41\u00e9\U0001F600""#;
        let text = Cursor::new(src).literal();
        assert_eq!(text, Ok("a\n\r\t\\\"[]-^Aé\u{1F600}".to_string()));
    }

    #[test]
    fn literal_keeps_non_ascii_and_hash() {
        let text = Cursor::new("\"é # ü\"").literal();
        assert_eq!(text, Ok("é # ü".to_string()));
    }

    #[test]
    fn class_reads_ranges_and_escapes() {
        let ranges = vec![('a', 'z'), (']', ']'), ('\0', '\u{1f}'), ('-', '-')];
        assert_eq!(
            class(r"[a-z\]\u0000-\u001f-]"),
            Ok(Expr::Class {
                negated: false,
                ranges
            })
        );
        assert_eq!(
            class("[^]"),
            Ok(Expr::Class {
                negated: true,
                ranges: vec![]
            })
        );
        assert_eq!(
            class("[-α-ω]"),
            Ok(Expr::Class {
                negated: false,
                ranges: vec![('-', '-'), ('α', 'ω')]
            })
        );
    }

    #[test]
    fn bad_escapes_name_the_escape() {
        for (src, needle) in [
            (r#""\q""#, r"`\q`"),
            (r#""\x1""#, r"`\x1`"),
            (r#""\u12"#, r"`\u12`"),
            (r#""\uD800""#, r"`\uD800`"),
            (r#""\UFFFFFFFF""#, r"`\UFFFFFFFF`"),
            (r#""\U00110000""#, r"`\U00110000`"),
        ] {
            let msg = message(Cursor::new(src).literal());
            assert!(msg.contains(needle), "{src}: {msg}");
        }
    }

    #[test]
    fn unterminated_literal_and_class_are_errors() {
        for src in ["\"abc", "\"abc\ndef\"", "\"esc\\"] {
            assert!(Cursor::new(src).literal().is_err(), "{src:?}");
        }
        for src in ["[a-z", "[a-", "[^", "[", "[a\n]", "[\\"] {
            assert!(class(src).is_err(), "{src:?}");
        }
        assert!(message(class("[a-z")).contains("no closing `]`"));
    }

    #[test]
    fn reversed_range_and_empty_class_are_errors() {
        assert!(message(class("[z-a]")).contains("`z-a`"));
        assert!(message(class("[]")).contains("`[]`"));
    }
}
