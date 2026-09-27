//! The module's regular expressions, compiled once, and the JavaScript string
//! semantics they run under: the `\s` set, `trim`, UTF-16 lengths, `split`
//! and `Number(string)`.
//!
//! The patterns spell out what JavaScript means where regexr differs: `\s` is
//! [`WS`], `\d` is `[0-9]`, `^`/`$` are `\A`/`\z`, and a case-insensitive
//! word is an ASCII class per letter ([`ci`]), since JavaScript's `/i`
//! without `u` does not fold `ſ` to `s` the way Unicode folding does.

use std::sync::OnceLock;

use regexr::{Captures, Regex};

use crate::error::{Error, Result};

/// Members of JavaScript's `\s` class, to place inside a character class.
pub(crate) const WS: &str = r"\t\n\x0B\x0C\r \x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}\x{FEFF}";

/// A character in JavaScript's `\s` class. Agrees with [`WS`].
pub(crate) fn is_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r' | ' ' | '\u{A0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200A}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202F}'
                | '\u{205F}'
                | '\u{3000}'
                | '\u{FEFF}'
    )
}

/// `String.prototype.trim`.
pub(crate) fn trim(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `String.prototype.length`: UTF-16 code units.
pub(crate) fn utf16_len(s: &str) -> usize {
    s.encode_utf16().count()
}

/// `word` matched case-insensitively over ASCII letters only, as JavaScript's
/// `/i` does for these words.
pub(crate) fn ci(word: &str) -> String {
    let mut out = String::new();
    for c in word.chars() {
        if c.is_ascii_alphabetic() {
            out.push('[');
            out.push(c.to_ascii_lowercase());
            out.push(c.to_ascii_uppercase());
            out.push(']');
        } else {
            out.push_str(&regexr::escape(c.encode_utf8(&mut [0; 4])));
        }
    }
    out
}

/// `text.split(re)` for a pattern without capture groups and without empty
/// matches: the pieces between matches, empty ones included.
pub(crate) fn split<'t>(re: &Regex, text: &'t str) -> Vec<&'t str> {
    let mut out = Vec::new();
    let mut last = 0;
    for m in re.find_iter(text) {
        out.push(&text[last..m.start()]);
        last = m.end();
    }
    out.push(&text[last..]);
    out
}

/// `text.replace(re /g, fn)`: every match replaced by what `with` returns.
pub(crate) fn replace_all(
    re: &Regex,
    text: &str,
    mut with: impl FnMut(&Captures<'_>) -> String,
) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for caps in re.captures_iter(text) {
        let Some(m) = caps.get(0) else { continue };
        out.push_str(&text[last..m.start()]);
        out.push_str(&with(&caps));
        last = m.end();
    }
    out.push_str(&text[last..]);
    out
}

/// `text.replace(re /g, rep)` with a fixed replacement.
pub(crate) fn replace_all_with(re: &Regex, text: &str, rep: &str) -> String {
    replace_all(re, text, |_| rep.to_string())
}

/// Group `i` of a match, `None` when it did not take part.
pub(crate) fn group<'t>(caps: &Captures<'t>, i: usize) -> Option<&'t str> {
    caps.get(i).map(|m| m.as_str())
}

/// `Number(s)` for a string: JavaScript's StringToNumber. Whitespace around
/// is ignored, an empty string is 0, and anything else that is not a decimal,
/// `0x`/`0o`/`0b` or `Infinity` literal is NaN.
pub(crate) fn number(s: &str) -> f64 {
    let t = trim(s);
    if t.is_empty() {
        return 0.0;
    }
    match t {
        "Infinity" | "+Infinity" => return f64::INFINITY,
        "-Infinity" => return f64::NEG_INFINITY,
        _ => {}
    }
    for (prefix, radix) in [
        ("0x", 16),
        ("0X", 16),
        ("0o", 8),
        ("0O", 8),
        ("0b", 2),
        ("0B", 2),
    ] {
        if let Some(digits) = t.strip_prefix(prefix) {
            if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
                return f64::NAN;
            }
            // u128 rounds once to the nearest f64, as JavaScript does; a
            // longer literal accumulates in f64.
            return u128::from_str_radix(digits, radix).map_or_else(
                |_| {
                    digits.chars().fold(0.0, |v, c| {
                        v * f64::from(radix) + f64::from(c.to_digit(radix).unwrap_or(0))
                    })
                },
                |v| v as f64,
            );
        }
    }
    if is_decimal_literal(t) {
        t.parse().unwrap_or(f64::NAN)
    } else {
        f64::NAN
    }
}

/// `[+-]? (digits [. digits?] | . digits) ([eE] [+-]? digits)?`. Rust's
/// float parser also takes `inf` and `nan`, which JavaScript does not.
fn is_decimal_literal(t: &str) -> bool {
    let b = t.as_bytes();
    let mut i = usize::from(matches!(b.first(), Some(b'+' | b'-')));
    let digits = |i: &mut usize| {
        let from = *i;
        while b.get(*i).is_some_and(u8::is_ascii_digit) {
            *i += 1;
        }
        *i - from
    };
    let mut mantissa = digits(&mut i);
    if b.get(i) == Some(&b'.') {
        i += 1;
        mantissa += digits(&mut i);
    }
    if mantissa == 0 {
        return false;
    }
    if matches!(b.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(b.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        if digits(&mut i) == 0 {
            return false;
        }
    }
    i == b.len()
}

/// Every pattern the module runs, compiled on first use.
pub(crate) struct Patterns {
    /// `\s+`
    pub(crate) spaces: Regex,
    /// `\n\s*\n`: a blank line, between beats and between subtitle blocks.
    pub(crate) blank_line: Regex,
    /// `[ \t]+`
    pub(crate) blanks: Regex,
    /// ` ?\n ?`
    pub(crate) padded_newline: Regex,
    /// `TAGS`: VoxCPM2's non-verbal tags.
    pub(crate) tags: Regex,
    /// `LEAD`: a delivery note opening the text. Group 1 is the note; its
    /// 2 to 120 UTF-16 unit bound is checked by the caller.
    pub(crate) lead: Regex,
    /// `LEAD` with the `m` flag: a delivery note opening any line.
    pub(crate) lead_lines: Regex,
    /// `[pause]` or `[pause 0.6]`; group 1 is the value.
    pub(crate) pause: Regex,
    /// `SAID`: `{written|spoken}`.
    pub(crate) said: Regex,
    /// One sentence of whitespace-collapsed text.
    pub(crate) sentence: Regex,
    /// `[^\p{L}\p{N}]+`: what a word key drops.
    pub(crate) not_letters: Regex,
    /// `.srt` or `.vtt` at the end of a file name.
    pub(crate) subtitle_file: Regex,
    /// `<i>` and `{\an8}` subtitle markup.
    pub(crate) markup: Regex,
    /// `STAMP`: a time, an optional range end, a separator, then the text.
    pub(crate) stamp: Regex,
    /// A unit after a time, removed before the time is read.
    pub(crate) unit_suffix: Regex,
    /// A colon or a unit in a time: `/[:s]|sec|saat/i`.
    pub(crate) time_marked: Regex,
    /// `^\s*[[(]`
    pub(crate) opens_bracket: Regex,
    /// `^[\s\d.,]+[|\-–—.):]`: a number then a separator.
    pub(crate) number_then_separator: Regex,
    /// `[^\p{L}\p{N}\s-]`: what a number token drops.
    pub(crate) not_token: Regex,
    /// `[\s-]+`: what separates number tokens.
    pub(crate) token_break: Regex,
    /// `\d`
    pub(crate) digit: Regex,
    /// A number written as digits, commas grouping thousands.
    pub(crate) written_number: Regex,
    /// `(\d)\s+([.,])\s*(\d)`: a decimal a recognizer spaced out.
    pub(crate) spaced_decimal: Regex,
    /// `[^\p{L}\p{N}\s%]`: what a heard or script word drops.
    pub(crate) not_word: Regex,
    /// `(\d)(\p{L}|%)`: a unit written onto its number.
    pub(crate) digit_unit: Regex,
    /// `^\s*(-\p{L}|,\d)`: a recognizer fragment of the word before.
    pub(crate) fragment: Regex,
    /// `^(.*\d)\s*(\p{L}+|%)([.,!?]*)$`: a number, its unit, then end
    /// punctuation.
    pub(crate) number_unit: Regex,
    /// `[\p{L}\p{N}%]`: a script token that holds a word.
    pub(crate) word_char: Regex,
    /// `[,.;:!?…—–-]["'”’)\]]*$`: a word ending in a break.
    pub(crate) break_end: Regex,
}

static PATTERNS: OnceLock<std::result::Result<Patterns, String>> = OnceLock::new();

fn compile(name: &str, pattern: &str) -> std::result::Result<Regex, String> {
    Regex::new(pattern)
        .map_err(|e| format!("narration pattern `{name}` ({pattern}) does not compile: {e}"))
}

impl Patterns {
    /// The process-wide patterns, compiled on first use.
    pub(crate) fn get() -> Result<&'static Self> {
        PATTERNS
            .get_or_init(Self::build)
            .as_ref()
            .map_err(|reason| Error::ModelError {
                reason: reason.clone(),
            })
    }

    fn build() -> std::result::Result<Self, String> {
        let tags = format!(
            r"\[(?:{}|{}|{}|{}|{}(?:{}|{}|{}|{})|{}(?:{}|{})|{})\]",
            ci("laughing"),
            ci("sigh"),
            ci("uhm"),
            ci("shh"),
            ci("question-"),
            ci("ah"),
            ci("ei"),
            ci("en"),
            ci("oh"),
            ci("surprise-"),
            ci("wa"),
            ci("yo"),
            ci("dissatisfaction-hnn"),
        );
        let note = r"[ \t]*\(([^()\n]+)\)[ \t]*";
        let units = format!(
            "{}|{}|{}|{}?|{}",
            ci("s"),
            ci("sec"),
            ci("secs"),
            ci("seconds"),
            ci("saat")
        );
        let time = format!(r"[0-9]+(?::[0-9]{{1,2}}){{0,2}}(?:[.,][0-9]+)?[{WS}]*(?:{units})?");
        let stamp = format!(
            r"\A[{WS}]*[\[(]?[{WS}]*({time})[{WS}]*(?:(?:-|–|—|{}|{}|{})[{WS}]*({time}))?[{WS}]*[\])]?[{WS}]*(?:[:|\-–—.)][{WS}]*|[{WS}]+)([^\n\r\x{{2028}}\x{{2029}}]*)\z",
            ci("to"),
            ci("hingga"),
            ci("sampai"),
        );
        Ok(Self {
            spaces: compile("spaces", &format!("[{WS}]+"))?,
            blank_line: compile("blank_line", &format!(r"\n[{WS}]*\n"))?,
            blanks: compile("blanks", r"[ \t]+")?,
            padded_newline: compile("padded_newline", r" ?\n ?")?,
            tags: compile("tags", &tags)?,
            lead: compile("lead", &format!(r"\A{note}"))?,
            lead_lines: compile(
                "lead_lines",
                &format!(r"(?:\A|(?<=[\n\r\x{{2028}}\x{{2029}}])){note}"),
            )?,
            pause: compile(
                "pause",
                &format!(r"\[{}(?:[{WS}]+([0-9.]+))?\]", ci("pause")),
            )?,
            said: compile("said", r"\{([^{}|\n]+)\|([^{}\n]+)\}")?,
            sentence: compile(
                "sentence",
                &format!(r#".+?(?:[.!?…]+["')\]」』]*(?=[{WS}]|\z)|[。！？]+[」』]*|\z)"#),
            )?,
            not_letters: compile("not_letters", r"[^\p{L}\p{N}]+")?,
            subtitle_file: compile(
                "subtitle_file",
                &format!(r"\.(?:{}|{})\z", ci("srt"), ci("vtt")),
            )?,
            markup: compile("markup", r"<[^>]+>|\{[^}]*\}")?,
            stamp: compile("stamp", &stamp)?,
            unit_suffix: compile("unit_suffix", &format!(r"[{WS}]*(?:{units})\z"))?,
            time_marked: compile(
                "time_marked",
                &format!("[:sS]|{}|{}", ci("sec"), ci("saat")),
            )?,
            opens_bracket: compile("opens_bracket", &format!(r"\A[{WS}]*[\[(]"))?,
            number_then_separator: compile(
                "number_then_separator",
                &format!(r"\A[{WS}0-9.,]+[|\-–—.):]"),
            )?,
            not_token: compile("not_token", &format!(r"[^\p{{L}}\p{{N}}{WS}\-]"))?,
            token_break: compile("token_break", &format!(r"[{WS}\-]+"))?,
            digit: compile("digit", "[0-9]")?,
            written_number: compile(
                "written_number",
                r"[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?",
            )?,
            spaced_decimal: compile(
                "spaced_decimal",
                &format!(r"([0-9])[{WS}]+([.,])[{WS}]*([0-9])"),
            )?,
            not_word: compile("not_word", &format!(r"[^\p{{L}}\p{{N}}{WS}%]"))?,
            digit_unit: compile("digit_unit", r"([0-9])(\p{L}|%)")?,
            fragment: compile("fragment", &format!(r"\A[{WS}]*(?:-\p{{L}}|,[0-9])"))?,
            // JavaScript's `.` stops at the four line terminators.
            number_unit: compile(
                "number_unit",
                &format!(r"\A([^\n\r\x{{2028}}\x{{2029}}]*[0-9])[{WS}]*(\p{{L}}+|%)([.,!?]*)\z"),
            )?,
            word_char: compile("word_char", r"[\p{L}\p{N}%]")?,
            break_end: compile("break_end", r#"[,.;:!?…—–\-]["'”’)\]]*\z"#)?,
        })
    }
}

/// `text.replace(/\s+/g, ' ')`
pub(crate) fn collapse(p: &Patterns, text: &str) -> String {
    replace_all_with(&p.spaces, text, " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_class_agrees_with_predicate() -> Result<()> {
        let p = Patterns::get()?;
        for cp in 0..=0xFFFFu32 {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            let s = c.to_string();
            assert_eq!(p.spaces.is_match(&s), is_space(c), "U+{cp:04X}");
        }
        Ok(())
    }

    #[test]
    fn number_follows_javascript() {
        assert_eq!(number(""), 0.0);
        assert_eq!(number("  12 "), 12.0);
        assert_eq!(number("1.5"), 1.5);
        assert_eq!(number("5."), 5.0);
        assert_eq!(number(".5"), 0.5);
        assert_eq!(number("1e3"), 1000.0);
        assert_eq!(number("0x10"), 16.0);
        assert_eq!(number("-Infinity"), f64::NEG_INFINITY);
        assert!(number("1.2.3").is_nan());
        assert!(number(".").is_nan());
        assert!(number("inf").is_nan());
        assert!(number("nan").is_nan());
        assert!(number("12abc").is_nan());
    }

    #[test]
    fn ascii_case_only() -> Result<()> {
        let p = Patterns::get()?;
        assert!(p.pause.is_match("[PAUSE]"));
        assert!(p.pause.is_match("[Pause 1.2]"));
        // JavaScript's `/i` without `u` does not fold U+017F to `s`.
        assert!(!p.pause.is_match("[pau\u{17F}e]"));
        assert!(!p.tags.is_match("[\u{17F}igh]"));
        Ok(())
    }

    #[test]
    fn split_keeps_empty_pieces() -> Result<()> {
        let p = Patterns::get()?;
        assert_eq!(split(&p.spaces, " a  b "), vec!["", "a", "b", ""]);
        assert_eq!(split(&p.spaces, ""), vec![""]);
        Ok(())
    }
}
