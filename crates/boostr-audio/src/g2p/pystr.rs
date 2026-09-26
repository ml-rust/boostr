//! CPython `str` character predicates shared by the spaCy tokenizer and the
//! misaki port.
//!
//! `str.isalpha` is General Category `L*`. `str.isspace` is White_Space plus
//! U+001C..U+001F. `str.isdigit` is Numeric_Type Digit or Decimal.

use unicode_general_category::{GeneralCategory, get_general_category};

/// `str.isalpha` for one character: General Category L*.
pub(crate) fn is_letter(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

/// `regex` `\p{Lu}`. For a letter it equals `str.isupper`.
pub(crate) fn is_upper_letter(c: char) -> bool {
    get_general_category(c) == GeneralCategory::UppercaseLetter
}

/// `str.isspace` for one character.
pub(crate) fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Numeric_Type=Digit characters: `str.isdigit` holds and `str.isdecimal`
/// does not. Inclusive code point ranges.
const DIGIT_NOT_DECIMAL: &[(u32, u32)] = &[
    (0xB2, 0xB3),
    (0xB9, 0xB9),
    (0x1369, 0x1371),
    (0x19DA, 0x19DA),
    (0x2070, 0x2070),
    (0x2074, 0x2079),
    (0x2080, 0x2089),
    (0x2460, 0x2468),
    (0x2474, 0x247C),
    (0x2488, 0x2490),
    (0x24EA, 0x24EA),
    (0x24F5, 0x24FD),
    (0x24FF, 0x24FF),
    (0x2776, 0x277E),
    (0x2780, 0x2788),
    (0x278A, 0x2792),
    (0x10A40, 0x10A43),
    (0x10E60, 0x10E68),
    (0x11052, 0x1105A),
    (0x1F100, 0x1F10A),
];

/// `str.isdigit` for one character.
pub(crate) fn is_py_digit(c: char) -> bool {
    if get_general_category(c) == GeneralCategory::DecimalNumber {
        return true;
    }
    let cp = c as u32;
    DIGIT_NOT_DECIMAL
        .iter()
        .any(|&(lo, hi)| (lo..=hi).contains(&cp))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicates_follow_cpython() {
        assert!(is_letter('a') && is_letter('É') && is_letter('ǅ'));
        assert!(!is_letter('1') && !is_letter('\u{0301}'));
        assert!(is_upper_letter('Q') && !is_upper_letter('ǅ'));
        assert!(is_py_space('\u{1f}') && is_py_space('\u{a0}') && !is_py_space('\u{200b}'));
        assert!(is_py_digit('7') && is_py_digit('٣') && is_py_digit('²') && is_py_digit('①'));
        assert!(!is_py_digit('½') && !is_py_digit('a'));
    }
}
