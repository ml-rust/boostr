//! misaki `subtokenize`: `regex.findall` of misaki's `SUBTOKEN_REGEX`.
//!
//! The pattern is
//! `^['‘’]+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|['‘’]{2,}|`
//! `\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})|\p{L}+(?:['‘’]\p{L})*|`
//! `[^-_\p{L}'‘’\d]|['‘’]+$`.
//! Each branch below is a direct matcher that returns the end of the first
//! match a backtracking engine finds at the position, and the branches run in
//! pattern order. `findall` then advances past each match, or one character
//! when nothing matches.

use super::chars::{QUOTES, is_decimal, is_letter, is_lower_letter, is_upper_letter};

struct Cursor<'a> {
    chars: &'a [char],
}

impl Cursor<'_> {
    fn at(&self, i: usize, f: impl Fn(char) -> bool) -> bool {
        self.chars.get(i).is_some_and(|&c| f(c))
    }

    fn run(&self, mut i: usize, f: impl Fn(char) -> bool) -> usize {
        while self.at(i, &f) {
            i += 1;
        }
        i
    }
}

fn is_quote(c: char) -> bool {
    QUOTES.contains(c)
}

fn is_sep(c: char) -> bool {
    c == ',' || c == '.'
}

/// `(?:\d?[,.]?\d)+` from `q`: the end after greedy repetition, or `None`
/// when not even one repetition matches.
fn digit_groups(cur: &Cursor, mut q: usize) -> Option<usize> {
    let mut count = 0usize;
    loop {
        if cur.at(q, is_decimal) {
            if cur.at(q + 1, is_sep) && cur.at(q + 2, is_decimal) {
                q += 3;
            } else if cur.at(q + 1, is_decimal) {
                q += 2;
            } else {
                q += 1;
            }
        } else if cur.at(q, is_sep) && cur.at(q + 1, is_decimal) {
            q += 2;
        } else {
            break;
        }
        count += 1;
    }
    (count > 0).then_some(q)
}

/// `\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})` from `p`, lazy on both loops.
fn camel_split(cur: &Cursor, p: usize) -> Option<usize> {
    let mut i = p;
    loop {
        let mut q = i;
        loop {
            if cur.at(q, is_lower_letter) && cur.at(q + 1, is_upper_letter) {
                return Some(q + 1);
            }
            if cur.at(q, is_quote) && cur.at(q + 1, is_letter) {
                q += 2;
            } else {
                break;
            }
        }
        if cur.at(i, is_letter) {
            i += 1;
        } else {
            return None;
        }
    }
}

/// End of the first branch that matches at `p`, if any.
fn match_at(cur: &Cursor, p: usize) -> Option<usize> {
    let n = cur.chars.len();
    let c = cur.chars[p];
    // ^['‘’]+
    if p == 0 && is_quote(c) {
        return Some(cur.run(p, is_quote));
    }
    // \p{Lu}(?=\p{Lu}\p{Ll})
    if is_upper_letter(c) && cur.at(p + 1, is_upper_letter) && cur.at(p + 2, is_lower_letter) {
        return Some(p + 1);
    }
    // (?:^-)?(?:\d?[,.]?\d)+
    let signed = if p == 0 && c == '-' {
        digit_groups(cur, 1)
    } else {
        None
    };
    if let Some(e) = signed.or_else(|| digit_groups(cur, p)) {
        return Some(e);
    }
    // [-_]+
    if c == '-' || c == '_' {
        return Some(cur.run(p, |c| c == '-' || c == '_'));
    }
    // ['‘’]{2,}
    if is_quote(c) && cur.at(p + 1, is_quote) {
        return Some(cur.run(p, is_quote));
    }
    // \p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})
    if let Some(e) = camel_split(cur, p) {
        return Some(e);
    }
    // \p{L}+(?:['‘’]\p{L})*
    if is_letter(c) {
        let mut e = cur.run(p, is_letter);
        while cur.at(e, is_quote) && cur.at(e + 1, is_letter) {
            e += 2;
        }
        return Some(e);
    }
    // [^-_\p{L}'‘’\d]
    if !(c == '-' || c == '_' || is_quote(c) || is_decimal(c)) {
        return Some(p + 1);
    }
    // ['‘’]+$ ; `$` also matches before a final newline.
    if is_quote(c) {
        let e = cur.run(p, is_quote);
        if e == n || (e + 1 == n && cur.chars[e] == '\n') {
            return Some(e);
        }
    }
    None
}

/// Splits one token's text the way misaki's `subtokenize` does.
pub(crate) fn subtokenize(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let cur = Cursor { chars: &chars };
    let mut out = Vec::new();
    let mut p = 0usize;
    while p < chars.len() {
        match match_at(&cur, p) {
            Some(e) => {
                out.push(chars[p..e].iter().collect());
                p = e;
            }
            None => p += 1,
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(s: &str) -> Vec<String> {
        subtokenize(s)
    }

    #[test]
    fn matches_misaki_subtokenize() {
        let cases: &[(&str, &[&str])] = &[
            ("CI/CD", &["CI", "/", "CD"]),
            ("iPhone", &["i", "Phone"]),
            ("PostgreSQL", &["Postgre", "SQL"]),
            ("$1.5M", &["$", "1.5", "M"]),
            ("7:30am", &["7", ":", "30", "am"]),
            ("-12,000.5", &["-12,000.5"]),
            ("rock'n'roll", &["rock'n'r", "oll"]),
            ("'tis", &["'", "tis"]),
            ("don't", &["don't"]),
            ("well--known", &["well", "--", "known"]),
            ("ab''", &["ab", "''"]),
            ("1'2", &["1", "2"]),
            ("B2B", &["B", "2", "B"]),
            ("", &[]),
        ];
        for &(input, want) in cases {
            assert_eq!(split(input), want, "{input:?}");
        }
    }
}
