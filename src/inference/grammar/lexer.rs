//! Character cursor and the small token scanners for GBNF text.
//!
//! Every scanner consumes at least one character or reports that none matched.

use super::ast::Span;

/// Longest digit run a message quotes before it truncates.
const QUOTE_DIGITS: usize = 20;

/// A position in GBNF text, tracked by line and column.
pub(crate) struct Cursor {
    chars: Vec<char>,
    pos: usize,
    line: u32,
    column: u32,
}

/// A run of decimal digits.
pub(crate) struct Number {
    /// The digits as written, truncated for quoting.
    pub text: String,
    /// The value, or `None` if it overflows `u32`.
    pub value: Option<u32>,
}

/// True for a character a rule name can hold.
pub(crate) fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '-' || c == '_'
}

/// A character quoted for a message, or `end of input`.
pub(crate) fn describe(c: Option<char>) -> String {
    match c {
        Some(c) => format!("`{}`", c.escape_debug()),
        None => "end of input".to_string(),
    }
}

impl Cursor {
    /// A cursor at the start of `src`.
    pub fn new(src: &str) -> Self {
        Self {
            chars: src.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
        }
    }

    /// The current character.
    pub fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    /// The character `n` places ahead.
    pub fn peek_at(&self, n: usize) -> Option<char> {
        self.pos
            .checked_add(n)
            .and_then(|i| self.chars.get(i))
            .copied()
    }

    /// Consume and return the current character.
    pub fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos = self.pos.saturating_add(1);
        if c == '\n' {
            self.line = self.line.saturating_add(1);
            self.column = 1;
        } else {
            self.column = self.column.saturating_add(1);
        }
        Some(c)
    }

    /// Consume `c` if it is the current character.
    pub fn eat(&mut self, c: char) -> bool {
        let found = self.peek() == Some(c);
        if found {
            self.bump();
        }
        found
    }

    /// Consume `s` if the input continues with it.
    pub fn eat_str(&mut self, s: &str) -> bool {
        let found = self.continues_with(0, s);
        if found {
            for _ in s.chars() {
                self.bump();
            }
        }
        found
    }

    /// True if the input `offset` places ahead continues with `s`.
    fn continues_with(&self, offset: usize, s: &str) -> bool {
        s.chars()
            .enumerate()
            .all(|(i, c)| self.peek_at(offset.saturating_add(i)) == Some(c))
    }

    /// The current position.
    pub fn span(&self) -> Span {
        Span {
            line: self.line,
            column: self.column,
        }
    }

    /// Skip spaces and tabs.
    pub fn skip_inline_space(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t')) {
            self.bump();
        }
    }

    /// Skip whitespace, newlines and `#` comments. Returns true if a newline was skipped.
    pub fn skip_trivia(&mut self) -> bool {
        let mut crossed = false;
        loop {
            match self.peek() {
                Some(' ' | '\t' | '\r') => {}
                Some('\n') => crossed = true,
                Some('#') => {
                    while self.peek().is_some_and(|c| c != '\n') {
                        self.bump();
                    }
                    continue;
                }
                _ => return crossed,
            }
            self.bump();
        }
    }

    /// True if the input continues with `name ::=`, the start of a rule.
    pub fn at_rule_start(&self) -> bool {
        let mut i = 0usize;
        while self.peek_at(i).is_some_and(is_name_char) {
            i = i.saturating_add(1);
        }
        if i == 0 {
            return false;
        }
        while matches!(self.peek_at(i), Some(' ' | '\t')) {
            i = i.saturating_add(1);
        }
        self.continues_with(i, "::=")
    }

    /// Consume a rule name. Returns an empty string if none starts here.
    pub fn name(&mut self) -> String {
        let mut name = String::new();
        while let Some(c) = self.peek().filter(|&c| is_name_char(c)) {
            self.bump();
            name.push(c);
        }
        name
    }

    /// Consume a run of decimal digits.
    pub fn number(&mut self) -> Number {
        let mut text = String::new();
        let mut value = Some(0u32);
        while let Some((c, digit)) = self.peek().and_then(|c| c.to_digit(10).map(|d| (c, d))) {
            self.bump();
            if text.len() < QUOTE_DIGITS {
                text.push(c);
            } else if text.len() == QUOTE_DIGITS {
                text.push_str("...");
            }
            value = value
                .and_then(|v| v.checked_mul(10))
                .and_then(|v| v.checked_add(digit));
        }
        Number { text, value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_start_needs_name_and_assign() {
        assert!(Cursor::new("ab-c_1 ::= x").at_rule_start());
        assert!(Cursor::new("a::=").at_rule_start());
        assert!(!Cursor::new("ab :: x").at_rule_start());
        assert!(!Cursor::new("::= x").at_rule_start());
        assert!(!Cursor::new("ab").at_rule_start());
    }

    #[test]
    fn number_caps_overflow_and_quote_length() {
        let mut cur = Cursor::new("4294967295");
        assert_eq!(cur.number().value, Some(u32::MAX));
        let mut cur = Cursor::new("4294967296");
        assert_eq!(cur.number().value, None);
        let long = "9".repeat(100);
        let num = Cursor::new(&long).number();
        assert_eq!(num.value, None);
        assert_eq!(num.text.len(), QUOTE_DIGITS + 3);
    }

    #[test]
    fn trivia_reports_newlines_and_skips_comments() {
        let mut cur = Cursor::new("  # note\n\t x");
        assert!(cur.skip_trivia());
        assert_eq!(cur.peek(), Some('x'));
        assert_eq!(cur.span(), Span { line: 2, column: 3 });
        let mut cur = Cursor::new("  x");
        assert!(!cur.skip_trivia());
    }

    #[test]
    fn bump_tracks_lines_and_stops_at_end() {
        let mut cur = Cursor::new("a\nb");
        assert_eq!(cur.bump(), Some('a'));
        assert_eq!(cur.bump(), Some('\n'));
        assert_eq!(cur.span(), Span { line: 2, column: 1 });
        assert_eq!(cur.bump(), Some('b'));
        assert_eq!(cur.bump(), None);
        assert_eq!(cur.span(), Span { line: 2, column: 2 });
    }

    #[test]
    fn eat_str_consumes_only_a_full_match() {
        let mut cur = Cursor::new("::x");
        assert!(!cur.eat_str("::="));
        assert_eq!(cur.peek(), Some(':'));
        let mut cur = Cursor::new("::=x");
        assert!(cur.eat_str("::="));
        assert_eq!(cur.peek(), Some('x'));
    }
}
