//! English number words, output-identical to `num2words` 0.5.14 (`lang='en'`)
//! for the calls misaki makes: cardinal, ordinal, year, and float.
//!
//! Integers arrive as ASCII digit strings, so values up to num2words' limit
//! of 10^306 need no big-integer type. Every scale word from hundred up is a
//! power of ten, so splitting a value at a scale is a split of its digits.

const LOW: [&str; 21] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const ILLION_UNITS: [&str; 10] = [
    "", "un", "duo", "tre", "quattuor", "quin", "sex", "sept", "octo", "novem",
];
const ILLION_TENS: [&str; 9] = [
    "dec",
    "vigint",
    "trigint",
    "quadragint",
    "quinquagint",
    "sexagint",
    "septuagint",
    "octogint",
    "nonagint",
];
const ILLION_LOWS: [&str; 9] = [
    "m", "b", "tr", "quadr", "quint", "sext", "sept", "oct", "non",
];
const ORDS: [(&str, &str); 12] = [
    ("one", "first"),
    ("two", "second"),
    ("three", "third"),
    ("four", "fourth"),
    ("five", "fifth"),
    ("six", "sixth"),
    ("seven", "seventh"),
    ("eight", "eighth"),
    ("nine", "ninth"),
    ("ten", "tenth"),
    ("eleven", "eleventh"),
    ("twelve", "twelfth"),
];
/// num2words refuses values at or above 10^306.
const MAX_DIGITS: usize = 306;

/// Name of 10^k for k a multiple of 3 in 6..=303.
fn illion(k: usize) -> String {
    let i = k / 3 - 2;
    if i < 9 {
        return format!("{}illion", ILLION_LOWS[i]);
    }
    if i == 99 {
        return "centillion".to_string();
    }
    let j = i - 9;
    format!("{}{}illion", ILLION_UNITS[j % 10], ILLION_TENS[j / 10])
}

fn strip_zeros(d: &str) -> &str {
    let s = d.trim_start_matches('0');
    if s.is_empty() { "0" } else { s }
}

fn below_100(v: usize) -> String {
    if v <= 20 {
        return LOW[v].to_string();
    }
    let (t, u) = (v / 10, v % 10);
    if u == 0 {
        TENS[t].to_string()
    } else {
        format!("{}-{}", TENS[t], LOW[u])
    }
}

/// `num2words(int(digits))`. `digits` is ASCII digits, leading zeros
/// allowed. `None` when the value reaches 10^306 or `digits` holds another
/// character.
pub(crate) fn cardinal(digits: &str) -> Option<String> {
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let d = strip_zeros(digits);
    let n = d.len();
    if n > MAX_DIGITS {
        return None;
    }
    if n <= 2 {
        return d.parse::<usize>().ok().map(below_100);
    }
    let k = if n == 3 {
        2
    } else {
        ((n - 1) / 3 * 3).min(303)
    };
    let scale = match k {
        2 => "hundred".to_string(),
        3 => "thousand".to_string(),
        _ => illion(k),
    };
    let div = strip_zeros(&d[..n - k]);
    let rem = strip_zeros(&d[n - k..]);
    let left = if div == "1" {
        "one".to_string()
    } else {
        cardinal(div)?
    };
    let mut out = format!("{left} {scale}");
    if rem != "0" {
        out.push_str(if rem.len() <= 2 { " and " } else { ", " });
        out.push_str(&cardinal(rem)?);
    }
    Some(out)
}

/// `num2words(int(digits), to='ordinal')`.
pub(crate) fn ordinal(digits: &str) -> Option<String> {
    let card = cardinal(digits)?;
    let (head, last) = match card.rfind(' ') {
        Some(i) => card.split_at(i + 1),
        None => ("", card.as_str()),
    };
    let (prefix, word) = match last.rfind('-') {
        Some(i) => last.split_at(i + 1),
        None => ("", last),
    };
    let word = match ORDS.iter().find(|(c, _)| *c == word) {
        Some((_, o)) => (*o).to_string(),
        None => match word.strip_suffix('y') {
            Some(stem) => format!("{stem}ieth"),
            None => format!("{word}th"),
        },
    };
    Some(format!("{head}{prefix}{word}"))
}

/// `num2words(int(digits), to='year')` for a non-negative value.
pub(crate) fn year(digits: &str) -> Option<String> {
    let d = strip_zeros(digits);
    if d.len() > 4 {
        return cardinal(d);
    }
    let v: u32 = d.parse().ok()?;
    let (high, low) = (v / 100, v % 100);
    if high == 0 || (high % 10 == 0 && low < 10) || high >= 100 {
        return cardinal(d);
    }
    let high_text = cardinal(&high.to_string())?;
    let low_text = if low == 0 {
        "hundred".to_string()
    } else if low < 10 {
        format!("oh-{}", cardinal(&low.to_string())?)
    } else {
        cardinal(&low.to_string())?
    };
    Some(format!("{high_text} {low_text}"))
}

/// `num2words(float(text))` for `text` of ASCII digits and one `.`.
///
/// An integral value reads as a cardinal. Otherwise the digits after the
/// point come from the shortest round-trip form of the `f64`, as in
/// num2words' `float2tuple`. An integral value at or above 2^53 reads from
/// its exact decimal expansion, where num2words runs its splitting on
/// Python floats.
pub(crate) fn float(text: &str) -> Option<String> {
    let v: f64 = text.parse().ok()?;
    if !v.is_finite() || v < 0.0 {
        return None;
    }
    if v.fract() == 0.0 {
        return cardinal(&format!("{v:.0}"));
    }
    let pre = v.trunc();
    let shortest = format!("{v}");
    let precision = shortest.split_once('.').map_or(0, |(_, f)| f.len());
    let scale: f64 = format!("1e{precision}").parse().ok()?;
    let post = (v - pre).abs() * scale;
    let post = if (post.round_ties_even() - post).abs() < 0.01 {
        post.round_ties_even()
    } else {
        post.floor()
    };
    let mut post_digits = format!("{post:.0}");
    if post_digits.len() < precision {
        post_digits = "0".repeat(precision - post_digits.len()) + &post_digits;
    }
    let mut out = vec![cardinal(&format!("{pre:.0}"))?];
    if precision > 0 {
        out.push("point".to_string());
    }
    for c in post_digits.chars().take(precision) {
        out.push(cardinal(c.encode_utf8(&mut [0u8; 4]))?);
    }
    Some(out.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: Option<String>) -> String {
        v.unwrap_or_else(|| "<none>".to_string())
    }

    #[test]
    fn cardinals() {
        assert_eq!(s(cardinal("0")), "zero");
        assert_eq!(s(cardinal("007")), "seven");
        assert_eq!(s(cardinal("45")), "forty-five");
        assert_eq!(s(cardinal("101")), "one hundred and one");
        assert_eq!(
            s(cardinal("1234")),
            "one thousand, two hundred and thirty-four"
        );
        assert_eq!(s(cardinal("1034")), "one thousand and thirty-four");
        assert_eq!(s(cardinal("2500000")), "two million, five hundred thousand");
        assert_eq!(s(cardinal("1000000001")), "one billion and one");
        assert_eq!(
            s(cardinal(&format!("1{}", "0".repeat(33)))),
            "one decillion"
        );
        assert_eq!(
            s(cardinal(&format!("1{}", "0".repeat(303)))),
            "one centillion"
        );
        assert_eq!(cardinal(&format!("1{}", "0".repeat(306))), None);
    }

    #[test]
    fn ordinals_and_years() {
        assert_eq!(s(ordinal("1")), "first");
        assert_eq!(s(ordinal("21")), "twenty-first");
        assert_eq!(s(ordinal("20")), "twentieth");
        assert_eq!(s(ordinal("100")), "one hundredth");
        assert_eq!(s(ordinal("0")), "zeroth");
        assert_eq!(s(year("1990")), "nineteen ninety");
        assert_eq!(s(year("2005")), "two thousand and five");
        assert_eq!(s(year("1905")), "nineteen oh-five");
        assert_eq!(s(year("1900")), "nineteen hundred");
        assert_eq!(s(year("0123")), "one twenty-three");
    }

    #[test]
    fn floats() {
        assert_eq!(s(float("3.14")), "three point one four");
        assert_eq!(s(float("3.05")), "three point zero five");
        assert_eq!(s(float("1.10")), "one point one");
        assert_eq!(s(float("12.0")), "twelve");
        assert_eq!(s(float("5.")), "five");
        assert_eq!(
            s(float("0.0000001")),
            "zero point zero zero zero zero zero zero one"
        );
        assert_eq!(
            s(float("0.30000000000000004")),
            "zero point three zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero four"
        );
    }
}
