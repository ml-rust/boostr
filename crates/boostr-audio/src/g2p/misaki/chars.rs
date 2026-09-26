//! Character classes and Python string predicates the misaki port relies on.
//!
//! misaki runs on CPython `str` methods and the `regex` module. These helpers
//! reproduce the exact classes it uses: `str.isalpha` is General Category
//! `L*`, `str.isdigit` is Numeric_Type Digit or Decimal, `\d` in `regex` is
//! `Nd`, and `str.isspace` is White_Space plus U+001C..U+001F.

use unicode_general_category::{GeneralCategory, get_general_category};

pub(crate) use crate::g2p::pystr::{is_letter, is_py_space, is_upper_letter};

/// Primary stress mark.
pub(crate) const PRIMARY_STRESS: char = 'ˈ';
/// Secondary stress mark.
pub(crate) const SECONDARY_STRESS: char = 'ˌ';

/// Phoneme characters misaki treats as vowels.
pub(crate) const VOWELS: &str = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
/// Phoneme characters misaki treats as consonants.
pub(crate) const CONSONANTS: &str = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
/// Phonemes that weigh two in `stress_weight`.
pub(crate) const DIPHTHONGS: &str = "AIOQWYʤʧ";
/// Vowels before a final `t` that turn it into a flap in US English.
pub(crate) const US_TAUS: &str = "AIOWYiuæɑəɛɪɹʊʌ";
/// Punctuation misaki passes through as phonemes.
pub(crate) const PUNCTS: &str = ";:,.!?—…\"“”";
/// `PUNCTS` without the quote characters.
pub(crate) const NON_QUOTE_PUNCTS: &str = ";:,.!?—…";
/// Characters a sub-token can consist of and still read as silent.
pub(crate) const SUBTOKEN_JUNKS: &str = "',-._‘’/";
/// Apostrophe variants the sub-tokenizer groups together.
pub(crate) const QUOTES: &str = "'‘’";

/// `c in VOWELS`.
pub(crate) fn is_vowel(c: char) -> bool {
    VOWELS.contains(c)
}

/// `regex` `\p{Ll}`.
pub(crate) fn is_lower_letter(c: char) -> bool {
    get_general_category(c) == GeneralCategory::LowercaseLetter
}

/// `regex` `\d`: General Category Nd.
pub(crate) fn is_decimal(c: char) -> bool {
    get_general_category(c) == GeneralCategory::DecimalNumber
}

/// `str.isalpha`: non-empty and every character a letter.
pub(crate) fn is_alpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_letter)
}

/// misaki `is_digit` on one character: `re.match(r'^[0-9]+$', c)`.
pub(crate) fn is_ascii_digit(c: char) -> bool {
    c.is_ascii_digit()
}

/// misaki `is_digit`: a non-empty run of ASCII digits.
pub(crate) fn is_digit_str(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// Numeric_Type=Digit characters outside Nd that CPython `str.isdigit`
/// accepts and NFKC leaves unchanged, as (first, last, value of first).
const OTHER_DIGITS: &[(u32, u32, u32)] = &[
    (0x1369, 0x1371, 1),
    (0x19DA, 0x19DA, 1),
    (0x24F5, 0x24FD, 1),
    (0x24FF, 0x24FF, 0),
    (0x2776, 0x277E, 1),
    (0x2780, 0x2788, 1),
    (0x278A, 0x2792, 1),
    (0x10A40, 0x10A43, 1),
    (0x10E60, 0x10E68, 1),
    (0x11052, 0x1105A, 1),
];

/// The value of `c` when `c.isdigit()` holds after NFKC, else `None`.
///
/// Nd digits sit in aligned runs of ten starting at zero, so the value is
/// the offset from the start of the run, modulo ten.
pub(crate) fn digit_value(c: char) -> Option<u32> {
    if c.is_ascii_digit() {
        return Some(c as u32 - '0' as u32);
    }
    if is_decimal(c) {
        let mut offset = 0u32;
        let mut cp = c as u32;
        while cp > 0 {
            cp -= 1;
            match char::from_u32(cp) {
                Some(prev) if is_decimal(prev) => offset += 1,
                _ => break,
            }
        }
        return Some(offset % 10);
    }
    let cp = c as u32;
    OTHER_DIGITS
        .iter()
        .find(|(lo, hi, _)| (*lo..=*hi).contains(&cp))
        .map(|(lo, _, first)| first + (cp - lo))
}

/// `str.lower`.
pub(crate) fn lower(s: &str) -> String {
    s.to_lowercase()
}

/// `str.upper`.
pub(crate) fn upper(s: &str) -> String {
    s.to_uppercase()
}

/// `c == c.lower()` for one character.
pub(crate) fn char_is_lower_fixed(c: char) -> bool {
    let mut it = c.to_lowercase();
    it.next() == Some(c) && it.next().is_none()
}

/// `s[1:]`: everything after the first character.
pub(crate) fn tail(s: &str) -> &str {
    let mut it = s.chars();
    it.next();
    it.as_str()
}

/// `s[:-n]` for `n` characters.
pub(crate) fn drop_last(s: &str, n: usize) -> &str {
    if n == 0 {
        return s;
    }
    match s.char_indices().rev().nth(n - 1) {
        Some((i, _)) => &s[..i],
        None => "",
    }
}

/// `len(s)` in characters.
pub(crate) fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// misaki `LEXICON_ORDS`: apostrophe, hyphen, and ASCII letters.
pub(crate) fn is_lexicon_char(c: char) -> bool {
    c == '\'' || c == '-' || c.is_ascii_alphabetic()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letters_follow_general_category() {
        assert!(is_letter('a') && is_letter('É') && is_letter('ǅ'));
        assert!(!is_letter('1') && !is_letter('\'') && !is_letter('\u{0301}'));
        assert!(is_upper_letter('Q') && !is_upper_letter('ǅ'));
        assert!(is_lower_letter('q'));
        assert!(is_decimal('٣') && !is_decimal('²'));
    }

    #[test]
    fn digit_values_match_python_isdigit() {
        assert_eq!(digit_value('7'), Some(7));
        assert_eq!(digit_value('٣'), Some(3));
        assert_eq!(digit_value('\u{1D7D5}'), Some(7));
        assert_eq!(digit_value('\u{136B}'), Some(3));
        assert_eq!(digit_value('\u{24FF}'), Some(0));
        assert_eq!(digit_value('a'), None);
    }

    #[test]
    fn python_slices() {
        assert_eq!(tail("héllo"), "éllo");
        assert_eq!(drop_last("héllo", 2), "hél");
        assert_eq!(drop_last("ab", 5), "");
        assert!(is_py_space('\u{1f}') && !is_py_space('x'));
        assert!(char_is_lower_fixed('a') && !char_is_lower_fixed('A'));
    }
}
