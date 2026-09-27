//! Numbers spoken as words, read back as values, so a check can compare a
//! script's "lapan belas sembilan puluh enam" with the "1896" a recognizer
//! writes.
//!
//! Covers English and Malay/Indonesian number words, including years read
//! in pairs ("eighteen fifty-seven", "sembilan belas tujuh puluh empat").
//! Words of other languages are not read.

use unicode_normalization::UnicodeNormalization;

use crate::error::Result;
use crate::narration::pattern::{Patterns, group, number, replace_all, replace_all_with, split};

fn unit(w: &str) -> Option<f64> {
    Some(match w {
        "zero" | "oh" | "kosong" | "sifar" => 0.0,
        "one" | "satu" => 1.0,
        "two" | "dua" => 2.0,
        "three" | "tiga" => 3.0,
        "four" | "empat" => 4.0,
        "five" | "lima" => 5.0,
        "six" | "enam" => 6.0,
        "seven" | "tujuh" => 7.0,
        "eight" | "lapan" | "delapan" => 8.0,
        "nine" | "sembilan" => 9.0,
        _ => return None,
    })
}

fn teen(w: &str) -> Option<f64> {
    Some(match w {
        "ten" | "sepuluh" => 10.0,
        "eleven" | "sebelas" => 11.0,
        "twelve" => 12.0,
        "thirteen" => 13.0,
        "fourteen" => 14.0,
        "fifteen" => 15.0,
        "sixteen" => 16.0,
        "seventeen" => 17.0,
        "eighteen" => 18.0,
        "nineteen" => 19.0,
        _ => return None,
    })
}

fn tens(w: &str) -> Option<f64> {
    Some(match w {
        "twenty" => 20.0,
        "thirty" => 30.0,
        "forty" => 40.0,
        "fifty" => 50.0,
        "sixty" => 60.0,
        "seventy" => 70.0,
        "eighty" => 80.0,
        "ninety" => 90.0,
        _ => return None,
    })
}

/// A Malay/Indonesian word after a unit: "belas" makes a teen, the others
/// multiply.
#[derive(Debug, Clone, Copy, PartialEq)]
enum MsScale {
    Teen,
    Times(f64),
}

fn scale_ms(w: &str) -> Option<MsScale> {
    Some(match w {
        "belas" => MsScale::Teen,
        "puluh" => MsScale::Times(10.0),
        "ratus" => MsScale::Times(100.0),
        "ribu" => MsScale::Times(1000.0),
        "juta" => MsScale::Times(1e6),
        _ => return None,
    })
}

/// "se-" words: one of a scale.
fn one_ms(w: &str) -> Option<f64> {
    Some(match w {
        "seratus" => 100.0,
        "seribu" => 1000.0,
        "sejuta" => 1e6,
        _ => return None,
    })
}

fn scale_en(w: &str) -> Option<f64> {
    Some(match w {
        "hundred" => 100.0,
        "thousand" => 1000.0,
        "million" => 1e6,
        _ => return None,
    })
}

/// Lowercased, NFKD-normalized words: anything but letters, digits,
/// whitespace and hyphens becomes a space, then whitespace and hyphens split.
fn tokens(p: &Patterns, text: &str) -> Vec<String> {
    let text: String = text.to_lowercase().nfkd().collect();
    let text = replace_all_with(&p.not_token, &text, " ");
    split(&p.token_break, &text)
        .into_iter()
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

/// A number read from the tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Read {
    value: f64,
    /// The token after the number.
    next: usize,
    /// The value is under 100 and could be half of a year.
    small: bool,
}

/// Reads one number starting at token `i`: a value under 100, or a larger
/// one built with hundred/thousand words.
fn read_number(t: &[&str], i: usize) -> Option<Read> {
    let at = |k: usize| t.get(k).copied().unwrap_or_default();
    let mut total = 0.0;
    let mut group = 0.0;
    let mut k = i;
    let mut read = false;
    let mut small = true;
    while k < t.len() {
        let w = t[k];
        if let (Some(u), Some(scale)) = (unit(w), scale_ms(at(k + 1))) {
            match scale {
                MsScale::Teen => {
                    if group % 100.0 >= 10.0 {
                        break;
                    }
                    group += 10.0 + u;
                    k += 2;
                    read = true;
                    // A teen closes a small number: "lapan belas" then "lima
                    // puluh tujuh" is two halves of a year.
                    let n = at(k);
                    if scale_ms(n).is_none() && one_ms(n).is_none() && scale_en(n).is_none() {
                        break;
                    }
                    continue;
                }
                MsScale::Times(10.0) => {
                    if group % 100.0 >= 10.0 {
                        break;
                    }
                    group += u * 10.0;
                    k += 2;
                    read = true;
                    if let Some(v) = unit(at(k))
                        && scale_ms(at(k + 1)).is_none()
                    {
                        group += v;
                        k += 1;
                    }
                    break;
                }
                MsScale::Times(s) => {
                    group = (group + u) * s;
                    if s >= 1000.0 {
                        total += group;
                        group = 0.0;
                    }
                    small = false;
                    k += 2;
                    read = true;
                    continue;
                }
            }
        }
        // A bare scale word multiplies what came before: "dua ratus ribu" is
        // 200 000.
        if read
            && let Some(MsScale::Times(s)) = scale_ms(w)
            && s >= 100.0
        {
            group *= s;
            if s >= 1000.0 {
                total += group;
                group = 0.0;
            }
            small = false;
            k += 1;
            continue;
        }
        if let Some(s) = one_ms(w) {
            if s >= 1000.0 {
                total += s;
            } else {
                group += s;
            }
            small = false;
            k += 1;
            read = true;
            continue;
        }
        if read && let Some(s) = scale_en(w) {
            group = group.max(1.0) * s;
            if s >= 1000.0 {
                total += group;
                group = 0.0;
            }
            small = false;
            k += 1;
            if at(k) == "and" {
                k += 1;
            }
            continue;
        }
        if let Some(v) = tens(w) {
            if group % 100.0 >= 10.0 {
                break;
            }
            group += v;
            k += 1;
            read = true;
            if let Some(u) = unit(at(k)) {
                group += u;
                k += 1;
            }
            // "twenty-five hundred" is 2500.
            if at(k) == "hundred" && group < 100.0 {
                continue;
            }
            break;
        }
        if let Some(v) = teen(w) {
            if group % 100.0 >= 10.0 {
                break;
            }
            group += v;
            k += 1;
            read = true;
            // "eleven hundred" is 1100, "nineteen hundred and five" 1905.
            if at(k) == "hundred" && group < 100.0 {
                continue;
            }
            break;
        }
        if let Some(u) = unit(w) {
            if read && group % 10.0 != 0.0 {
                break;
            }
            group += u;
            k += 1;
            read = true;
            if scale_en(at(k)).is_none() {
                break;
            }
            continue;
        }
        break;
    }
    if !read {
        return None;
    }
    let value = total + group;
    Some(Read {
        value,
        next: k,
        small: small && value < 100.0,
    })
}

/// `Number(`${value}.${digits}`)`: JavaScript writes 1e21 and over in
/// exponent form, which then reads as NaN.
fn decimal(value: f64, digits: &str) -> f64 {
    if value >= 1e21 {
        return f64::NAN;
    }
    number(&format!("{value}.{digits}"))
}

/// Every number spoken as words in a text, in order. Two small numbers in a
/// row, the first from 10 to 99, are a year read in pairs: "nineteen
/// eighty-five" is 1985, "lapan belas lima puluh tujuh" is 1857. "oh" and
/// "kosong" join as a zero tens: "nineteen oh five" is 1905.
pub fn spoken_numbers(text: &str) -> Result<Vec<f64>> {
    let p = Patterns::get()?;
    let owned = tokens(p, text);
    let t: Vec<&str> = owned.iter().map(String::as_str).collect();
    let at = |k: usize| t.get(k).copied().unwrap_or_default();
    let mut out = Vec::new();
    let mut i = 0;
    while i < t.len() {
        let Some(a) = read_number(&t, i) else {
            i += 1;
            continue;
        };
        // A fraction or decimal after a number: "tiga setengah" and "three
        // and a half" are 3.5, "tiga perpuluhan lima" and "three point five"
        // are 3.5.
        let after = a.next;
        if at(after) == "setengah"
            || (at(after) == "and" && at(after + 1) == "a" && at(after + 2) == "half")
        {
            out.push(a.value + 0.5);
            i = after + if at(after) == "setengah" { 1 } else { 3 };
            continue;
        }
        if matches!(at(after), "perpuluhan" | "point" | "koma") {
            let mut digits = String::new();
            let mut k = after + 1;
            while let Some(d) = unit(at(k)) {
                // Units are 0 to 9: one decimal digit each.
                digits.push(char::from(b'0' + d as u8));
                k += 1;
            }
            if !digits.is_empty() {
                out.push(decimal(a.value, &digits));
                i = k;
                continue;
            }
        }
        let zero = matches!(at(a.next), "oh" | "kosong");
        let b = if a.small && a.value >= 10.0 {
            read_number(&t, if zero { a.next + 1 } else { a.next })
        } else {
            None
        };
        match b {
            Some(b)
                if b.small
                    && (if zero {
                        b.value < 10.0
                    } else {
                        b.value >= 10.0
                    }) =>
            {
                out.push(a.value * 100.0 + b.value);
                i = b.next;
            }
            _ => {
                out.push(a.value);
                i = a.next;
            }
        }
    }
    Ok(out)
}

/// Numbers written as digits in a text: "1,896" and "13.5" read as 1896 and
/// 13.5.
pub fn written_numbers(text: &str) -> Result<Vec<f64>> {
    let p = Patterns::get()?;
    Ok(p.written_number
        .find_iter(text)
        .map(|m| number(&m.as_str().replace(',', "")))
        .collect())
}

/// A token of a number spoken as words.
fn is_number_token(t: &str) -> bool {
    unit(t).is_some()
        || teen(t).is_some()
        || tens(t).is_some()
        || scale_ms(t).is_some()
        || one_ms(t).is_some()
        || scale_en(t).is_some()
        || matches!(
            t,
            "and" | "setengah" | "perpuluhan" | "koma" | "point" | "half"
        )
}

/// A word that is part of a number: it holds a digit, or every token of it
/// is a number word. A word with no tokens (".") counts.
pub fn is_number_word(word: &str) -> Result<bool> {
    let p = Patterns::get()?;
    Ok(p.digit.is_match(word) || tokens(p, word).iter().all(|t| is_number_token(t)))
}

/// All numbers in a text, spoken or written, sorted: the multiset two texts
/// must share. A decimal a recognizer spaced out ("3 .5") is one number.
pub fn numbers_in(text: &str) -> Result<Vec<f64>> {
    let p = Patterns::get()?;
    let t = replace_all(&p.spaced_decimal, text, |caps| {
        (1..=3).filter_map(|i| group(caps, i)).collect()
    });
    let mut out = spoken_numbers(&t)?;
    out.extend(written_numbers(&t)?);
    // Stable, as JavaScript's sort is. Values are never -0; a NaN sorts last
    // here, where JavaScript's order for it is unspecified.
    out.sort_by(f64::total_cmp);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_cardinals() -> Result<()> {
        assert_eq!(spoken_numbers("one hundred and five")?, vec![105.0]);
        // Two numbers from 10 to 99 in a row read as a year.
        assert_eq!(spoken_numbers("twenty-one, forty two")?, vec![2142.0]);
        // A tens word closes the number, so "thousand" after it starts none.
        assert_eq!(
            spoken_numbers("three hundred forty-two thousand six hundred")?,
            vec![342.0, 600.0]
        );
        assert_eq!(spoken_numbers("twenty-five hundred")?, vec![2500.0]);
        assert_eq!(spoken_numbers("nineteen hundred and five")?, vec![1905.0]);
        Ok(())
    }

    #[test]
    fn malay_cardinals() -> Result<()> {
        assert_eq!(spoken_numbers("sebelas dua belas")?, vec![1112.0]);
        assert_eq!(spoken_numbers("dua ratus ribu")?, vec![200000.0]);
        assert_eq!(
            spoken_numbers("seribu sembilan ratus lapan puluh lima")?,
            vec![1985.0]
        );
        assert_eq!(spoken_numbers("delapan")?, vec![8.0]);
        Ok(())
    }

    #[test]
    fn paired_years() -> Result<()> {
        assert_eq!(spoken_numbers("nineteen eighty-five")?, vec![1985.0]);
        assert_eq!(spoken_numbers("nineteen oh five")?, vec![1905.0]);
        assert_eq!(
            spoken_numbers("lapan belas lima puluh tujuh")?,
            vec![1857.0]
        );
        assert_eq!(spoken_numbers("sembilan belas kosong lima")?, vec![1905.0]);
        Ok(())
    }

    #[test]
    fn decimals_append_digits() -> Result<()> {
        assert_eq!(spoken_numbers("three point one two")?, vec![3.12]);
        assert_eq!(spoken_numbers("dua koma tujuh lima")?, vec![2.75]);
        assert_eq!(spoken_numbers("tiga setengah")?, vec![3.5]);
        assert_eq!(spoken_numbers("three and a half")?, vec![3.5]);
        // Digits append as text, so "oh five" after the point is ".05".
        assert_eq!(spoken_numbers("two point oh five")?, vec![2.05]);
        Ok(())
    }

    #[test]
    fn digits_and_gluing() -> Result<()> {
        assert_eq!(written_numbers("In 1,896 and 13.5")?, vec![1896.0, 13.5]);
        assert_eq!(numbers_in("Version 3 .5 and two")?, vec![2.0, 3.5]);
        Ok(())
    }

    #[test]
    fn number_words() -> Result<()> {
        assert!(is_number_word("twenty-one")?);
        assert!(is_number_word("1857")?);
        assert!(is_number_word(".")?);
        assert!(!is_number_word("twenty-abc")?);
        // Full-width letters decompose under NFKD.
        assert!(is_number_word("ｔｗｏ")?);
        Ok(())
    }
}
