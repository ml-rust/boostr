//! A known script's words put on recognized timings, and the pauses a voice
//! took where the script has no break.

use serde::Serialize;

use super::heard::HeardWord;
use super::speech_match::{key, match_words, script_words, split_units};
use crate::error::Result;
use crate::narration::js::js_to_fixed;
use crate::narration::pattern::{Patterns, split};

/// A silence over this many seconds between two words with no punctuation
/// between them is a stray pause.
pub const STRAY_PAUSE: f64 = 0.45;

/// Decimals of the times [`align_script`] spreads.
const TIME_DECIMALS: usize = 3;
/// Decimals of a stray pause's length.
const GAP_DECIMALS: usize = 2;

/// Letters that weigh a word when time is shared: its pieces run together,
/// in UTF-16 code units, at least 1.
fn letters(p: &Patterns, text: &str) -> usize {
    key(p, text).encode_utf16().count().max(1)
}

/// Puts the script's words on the recognized timings.
///
/// Recognition misspells words and drops some, but a narration read from a
/// script should show the script's exact words. A pair of alike words is a
/// sure match and keeps its recognized time. The rest (a word misheard
/// beyond recognition, a number spoken as words but written as digits, a
/// dropped word) is spread by letter count over the speech heard between
/// the sure matches around it, times rounded to milliseconds. Spoken "lapan
/// belas lima puluh tujuh" then shares the span of "1857".
///
/// With no script words or no heard words, returns the heard words, units
/// split off and fragments rejoined.
pub fn align_script(heard: &[HeardWord], script: &str) -> Result<Vec<HeardWord>> {
    let p = Patterns::get()?;
    let words = split_units(p, heard);
    let said = script_words(p, script);
    if words.is_empty() || said.is_empty() {
        return Ok(words);
    }
    let first_start = words[0].start;
    let last_end = words[words.len() - 1].end;
    let matched = match_words(p, &words, &said);
    let sure_at = |i: usize| matched.pair[i].filter(|_| matched.sure[i]);
    let mut out: Vec<HeardWord> = said
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let (start, end) =
                sure_at(i).map_or((f64::NAN, f64::NAN), |j| (words[j].start, words[j].end));
            HeardWord {
                text: text.to_string(),
                start,
                end,
            }
        })
        .collect();
    // The heard word after the last sure match, as `lastSure + 1`.
    let mut after_sure = 0;
    let mut i = 0;
    while i < out.len() {
        if let Some(j) = sure_at(i) {
            after_sure = j + 1;
            i += 1;
            continue;
        }
        let mut k = i;
        while k < out.len() && sure_at(k).is_none() {
            k += 1;
        }
        let next_sure = if k < out.len() {
            sure_at(k).unwrap_or(words.len())
        } else {
            words.len()
        };
        let gap = &words[after_sure.min(next_sure)..next_sure];
        let prev_end = if i > 0 { out[i - 1].end } else { first_start };
        let next_start = if k < out.len() {
            out[k].start
        } else {
            last_end
        };
        let (from, to) = match (gap.first(), gap.last()) {
            (Some(g0), Some(gn)) => {
                let from = prev_end.max(g0.start);
                (from, next_start.min(gn.end.max(from)))
            }
            _ => (prev_end, next_start),
        };
        let weight: Vec<f64> = out[i..k]
            .iter()
            .map(|w| letters(p, &w.text) as f64)
            .collect();
        let total: f64 = weight.iter().sum();
        let mut t = from;
        for (w, share) in out[i..k].iter_mut().zip(&weight) {
            let d = ((to - from).max(0.0) * share) / total;
            w.start = js_to_fixed(t, TIME_DECIMALS);
            w.end = js_to_fixed(t + d, TIME_DECIMALS);
            t += d;
        }
        i = k;
    }
    Ok(out)
}

/// A pause the script does not ask for.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StrayPause {
    /// The script word before the pause.
    pub after: String,
    /// The script word after it.
    pub before: String,
    /// The silence in seconds, rounded to hundredths.
    pub gap: f64,
}

/// Pauses the script does not ask for: a silence over [`STRAY_PAUSE`]
/// between two aligned script words with no break between them.
///
/// A break after a word is its own trailing punctuation, or a dash or
/// ellipsis standing alone before the next word. A voice phrases by how it
/// reads the grammar, so a word that can be a noun or a verb can pull the
/// pause to the wrong place while every word is still said.
pub fn stray_pauses(heard: &[HeardWord], text: &str) -> Result<Vec<StrayPause>> {
    let p = Patterns::get()?;
    let aligned = align_script(heard, text)?;
    let mut breaks: Vec<bool> = Vec::new();
    for tok in split(&p.spaces, text) {
        if p.word_char.is_match(tok) {
            breaks.push(p.break_end.is_match(tok));
        } else if tok.chars().any(|c| matches!(c, '—' | '–' | '…' | '-'))
            && let Some(last) = breaks.last_mut()
        {
            *last = true;
        }
    }
    let mut out = Vec::new();
    for (i, pair) in aligned.windows(2).enumerate() {
        let [a, b] = pair else { continue };
        if a.end.is_nan() || b.start.is_nan() || breaks.get(i).copied().unwrap_or(false) {
            continue;
        }
        let gap = b.start - a.end;
        if gap > STRAY_PAUSE {
            out.push(StrayPause {
                after: a.text.clone(),
                before: b.text.clone(),
                gap: js_to_fixed(gap, GAP_DECIMALS),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start: f64, end: f64) -> HeardWord {
        HeardWord {
            text: text.to_string(),
            start,
            end,
        }
    }

    #[test]
    fn spoken_number_shares_the_digits_span() -> Result<()> {
        let heard = [
            word("Pada", 6.2, 6.5),
            word("tahun", 6.5, 7.0),
            word("1857,", 7.08, 7.97),
            word("Raja", 8.2, 8.5),
        ];
        let out = align_script(&heard, "Pada tahun lapan belas lima puluh tujuh, Raja")?;
        assert_eq!(out.len(), 8);
        assert_eq!(out[2].start, 7.08);
        assert_eq!(out[6].end, 7.97);
        assert!(out[2..7].iter().all(|w| w.end - w.start > 0.1));
        assert_eq!(out[7].start, 8.2);
        Ok(())
    }

    #[test]
    fn spread_times_round_ties_up() -> Result<()> {
        let heard = [
            word("alpha", 0.5, 1.0625),
            word("zzzz", 1.0625, 1.3125),
            word("gamma", 1.3125, 1.8),
        ];
        let out = align_script(&heard, "alpha beta gamma")?;
        assert_eq!((out[1].start, out[1].end), (1.063, 1.313));
        Ok(())
    }

    #[test]
    fn empty_sides_return_heard_words() -> Result<()> {
        assert!(align_script(&[], "a b")?.is_empty());
        let out = align_script(&[word("5L", 0.0, 1.0)], "...")?;
        assert_eq!(out.len(), 2);
        Ok(())
    }

    #[test]
    fn stray_pause_needs_no_break() -> Result<()> {
        let heard = [
            word("the", 0.0, 0.2),
            word("team", 0.2, 1.0),
            word("calls", 1.625, 2.0),
            word("them", 2.6, 3.0),
        ];
        let out = stray_pauses(&heard, "the team calls — them")?;
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].after.as_str(), out[0].gap), ("team", 0.63));
        assert!(stray_pauses(&heard, "the team, calls, them")?.is_empty());
        Ok(())
    }
}
