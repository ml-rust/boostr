//! JavaScript number semantics the narration checks reproduce: `toFixed` as
//! text and read back as a number, `Math.round`, and
//! `Number.prototype.toString`.

/// `+x.toFixed(digits)`: `x` rounded to `digits` decimals, read back as a
/// number.
///
/// NaN, infinities and magnitudes of 1e21 or more come back unchanged, as
/// their `toFixed` text reads back as the same value.
pub(crate) fn js_to_fixed(x: f64, digits: usize) -> f64 {
    if !x.is_finite() || x.abs() >= 1e21 {
        return x;
    }
    js_to_fixed_text(x, digits).parse().unwrap_or(f64::NAN)
}

/// `x.toFixed(digits)`: the text of `x` rounded to `digits` decimals.
///
/// `toFixed` works on the exact value of the double and rounds an exact tie
/// away from zero: `0.125` gives `0.13`. Rust's `{:.N}` rounds an exact tie
/// to even, so the rounding happens here on the exact decimal expansion.
/// NaN, infinities and magnitudes of 1e21 or more are written as
/// `String(x)`.
pub(crate) fn js_to_fixed_text(x: f64, digits: usize) -> String {
    if !x.is_finite() || x.abs() >= 1e21 {
        return js_string(x);
    }
    if x == 0.0 {
        // `(-0).toFixed(n)` has no sign.
        return format!("{:.*}", digits, 0.0);
    }
    // A double is m * 2^e: its exact expansion has at most -e decimals.
    let bits = x.abs().to_bits();
    let biased = ((bits >> 52) & 0x7FF) as i64;
    let exp = if biased == 0 { -1074 } else { biased - 1075 };
    let exact = usize::try_from(-exp).unwrap_or(0).max(digits);
    let text = format!("{:.*}", exact, x.abs());
    let (int, frac) = text.split_once('.').unwrap_or((text.as_str(), ""));
    let mut kept: Vec<u8> = int.bytes().chain(frac.bytes().take(digits)).collect();
    // The first dropped digit decides: 5 or more is at least half a unit.
    if frac.as_bytes().get(digits).is_some_and(|&d| d >= b'5') {
        let mut carry = true;
        for d in kept.iter_mut().rev() {
            if *d == b'9' {
                *d = b'0';
            } else {
                *d += 1;
                carry = false;
                break;
            }
        }
        if carry {
            kept.insert(0, b'1');
        }
    }
    let split = kept.len() - digits;
    let mut rounded = String::with_capacity(kept.len() + 2);
    if x < 0.0 {
        rounded.push('-');
    }
    rounded.extend(kept[..split].iter().map(|&b| char::from(b)));
    if digits > 0 {
        rounded.push('.');
        rounded.extend(kept[split..].iter().map(|&b| char::from(b)));
    }
    rounded
}

/// `Math.round(x)`: the nearest integer, an exact half rounded up (toward
/// positive infinity). Rust's `round` takes a negative half away from zero.
pub(crate) fn js_round(x: f64) -> f64 {
    let floor = x.floor();
    // `x - floor` is exact for every double.
    if x - floor >= 0.5 { floor + 1.0 } else { floor }
}

/// `String(x)`: the shortest digits that read back as `x`, in JavaScript's
/// layout. Plain decimals from 1e-6 up to 1e21, exponent form (`1e+21`,
/// `1.5e-7`) outside that.
pub(crate) fn js_string(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x == 0.0 {
        return "0".to_string();
    }
    if x < 0.0 {
        return format!("-{}", js_string(-x));
    }
    if x.is_infinite() {
        return "Infinity".to_string();
    }
    // `{:e}` writes the shortest round-trip digits: `1.985e3`.
    let sci = format!("{x:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let k = digits.len() as i64;
    let n = exp.parse::<i64>().unwrap_or(0) + 1;
    if k <= n && n <= 21 {
        format!("{digits}{}", "0".repeat((n - k) as usize))
    } else if 0 < n && n <= 21 {
        let (a, b) = digits.split_at(n as usize);
        format!("{a}.{b}")
    } else if -6 < n && n <= 0 {
        format!("0.{}{digits}", "0".repeat((-n) as usize))
    } else {
        let e = n - 1;
        let sign = if e < 0 { '-' } else { '+' };
        let (first, rest) = digits.split_at(1);
        if rest.is_empty() {
            format!("{first}e{sign}{}", e.abs())
        } else {
            format!("{first}.{rest}e{sign}{}", e.abs())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_fixed_rounds_exact_ties_up() {
        // 0.125, 0.375 and 2.5 are exact in binary: ties round away from zero.
        assert_eq!(js_to_fixed(0.125, 2), 0.13);
        assert_eq!(js_to_fixed(0.375, 2), 0.38);
        assert_eq!(js_to_fixed(2.5, 0), 3.0);
        assert_eq!(js_to_fixed(1.0625, 3), 1.063);
        assert_eq!(js_to_fixed(0.625, 2), 0.63);
        // `{:.2}` alone rounds the tie to even.
        assert_eq!(format!("{:.2}", 0.125), "0.12");
    }

    #[test]
    fn to_fixed_uses_the_exact_binary_value() {
        // 1.005 is 1.00499999999999989... in binary.
        assert_eq!(js_to_fixed(1.005, 2), 1.0);
        // 0.0005 is 0.000500000000000000010... in binary.
        assert_eq!(js_to_fixed(0.0005, 3), 0.001);
        assert_eq!(js_to_fixed(1.0005, 3), 1.0);
        assert_eq!(js_to_fixed(0.1 + 0.2, 3), 0.3);
        assert_eq!(js_to_fixed(9.9995, 3), 9.999);
        assert_eq!(js_to_fixed(9.99951, 3), 10.0);
    }

    #[test]
    fn to_fixed_negatives_and_zero() {
        assert_eq!(js_to_fixed(-0.125, 2), -0.13);
        assert_eq!(js_to_fixed(-1.005, 2), -1.0);
        // `(-0.0001).toFixed(3)` is "-0.000", which reads as -0.
        let tiny = js_to_fixed(-0.0001, 3);
        assert!(tiny == 0.0 && tiny.is_sign_negative());
        assert!(js_to_fixed(-0.0, 3).is_sign_positive());
    }

    #[test]
    fn to_fixed_large_and_special() {
        assert_eq!(js_to_fixed(123_456_789.000_5, 3), 123_456_789.0);
        assert_eq!(js_to_fixed(1e20 + 16384.0, 2), 1e20 + 16384.0);
        assert_eq!(js_to_fixed(1e21, 2), 1e21);
        assert_eq!(js_to_fixed(4.5e15 + 0.5, 0), 4.5e15 + 1.0);
        assert!(js_to_fixed(f64::NAN, 3).is_nan());
        assert_eq!(js_to_fixed(f64::INFINITY, 3), f64::INFINITY);
        assert_eq!(js_to_fixed(5e-324, 3), 0.0);
    }

    #[test]
    fn to_fixed_text_keeps_trailing_zeros() {
        assert_eq!(js_to_fixed_text(0.04, 2), "0.04");
        assert_eq!(js_to_fixed_text(1.0, 2), "1.00");
        assert_eq!(js_to_fixed_text(0.125, 2), "0.13");
        assert_eq!(js_to_fixed_text(0.0, 2), "0.00");
        assert_eq!(js_to_fixed_text(-0.0, 1), "0.0");
        assert_eq!(js_to_fixed_text(-0.0001, 3), "-0.000");
        assert_eq!(js_to_fixed_text(2.5, 0), "3");
        assert_eq!(js_to_fixed_text(f64::NAN, 2), "NaN");
        assert_eq!(js_to_fixed_text(1e21, 2), "1e+21");
    }

    #[test]
    fn round_takes_halves_up() {
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0);
        assert_eq!(js_round(-2.6), -3.0);
        assert_eq!(js_round(0.49999999999999994), 0.0);
        assert_eq!(js_round(480.00000000000006), 480.0);
        assert_eq!(js_round(-0.4), 0.0);
    }

    #[test]
    fn string_follows_javascript() {
        assert_eq!(js_string(1985.0), "1985");
        assert_eq!(js_string(3.5), "3.5");
        assert_eq!(js_string(-0.0), "0");
        assert_eq!(js_string(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(js_string(1e21), "1e+21");
        assert_eq!(js_string(1.5e21), "1.5e+21");
        assert_eq!(js_string(1e20), "100000000000000000000");
        assert_eq!(js_string(0.000001), "0.000001");
        assert_eq!(js_string(1e-7), "1e-7");
        assert_eq!(js_string(-2.5e-7), "-2.5e-7");
        assert_eq!(js_string(123.456), "123.456");
        assert_eq!(js_string(f64::NAN), "NaN");
        assert_eq!(js_string(f64::NEG_INFINITY), "-Infinity");
    }
}
