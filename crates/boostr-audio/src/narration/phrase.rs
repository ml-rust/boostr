//! A narration script split into the phrases a voice speaks, each with the
//! silence after it.
//!
//! The script marks its own pacing. A blank line is a long pause (a new
//! beat), a line break a breath, a sentence end inside a line a short stop,
//! and `[pause]` or `[pause 0.6]` an exact pause in seconds. A delivery note
//! opening a line, `(curious tone) Where does it go?`, attaches to the next
//! phrase and is never spoken.

use serde::Serialize;

use super::pattern::{Patterns, collapse, group, number, split, trim};
use super::tags::is_note;
use crate::error::Result;

/// The silence after each kind of break, in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Pauses {
    /// After a beat: a blank line.
    pub beat: f64,
    /// After a line break.
    pub line: f64,
    /// After a sentence end inside a line.
    pub sentence: f64,
    /// A `[pause]` mark without a usable value.
    pub mark: f64,
}

/// The default pauses.
pub const PAUSES: Pauses = Pauses {
    beat: 0.8,
    line: 0.45,
    sentence: 0.3,
    mark: 0.6,
};

/// What ends a phrase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PhraseKind {
    /// A sentence end inside a line.
    Sentence,
    /// A line break.
    Line,
    /// A blank line.
    Beat,
    /// A `[pause]` mark.
    Mark,
    /// The end of the script.
    End,
}

/// One phrase to speak.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Phrase {
    /// The words, whitespace collapsed.
    pub text: String,
    /// Seconds of silence after the phrase.
    pub pause: f64,
    /// What ends the phrase.
    pub kind: PhraseKind,
    /// The delivery note opening the line the phrase starts, parentheses
    /// included: `(calm, slow)`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delivery: Option<String>,
}

/// Collects phrases. A break with no words before it lengthens the break
/// after the previous phrase instead.
struct Builder {
    out: Vec<Phrase>,
    delivery: Option<String>,
}

impl Builder {
    fn add(&mut self, p: &Patterns, text: &str, pause: f64, kind: PhraseKind) {
        let t = trim(&collapse(p, text)).to_string();
        if !t.is_empty() {
            self.out.push(Phrase {
                text: t,
                pause,
                kind,
                delivery: self.delivery.take(),
            });
        } else if let Some(last) = self.out.last_mut() {
            // A [pause] mark is exact, so it wins over the break it sits on.
            if kind == PhraseKind::Mark || (last.kind != PhraseKind::Mark && pause > last.pause) {
                last.kind = kind;
            }
            last.pause = if kind == PhraseKind::Mark {
                pause
            } else {
                last.pause.max(pause)
            };
        }
    }
}

/// The seconds a `[pause]` mark asks for. A missing value, `0`, and a value
/// that is not a number (`1.2.3`) all give [`PAUSES`]`.mark`.
fn mark_seconds(value: Option<&str>) -> f64 {
    let v = value.map_or(PAUSES.mark, number);
    if v == 0.0 || v.is_nan() {
        PAUSES.mark
    } else {
        v
    }
}

/// The line split at its pause marks: each piece of text with the mark that
/// follows it, the last piece with none.
fn pause_pieces<'t>(p: &Patterns, line: &'t str) -> Vec<(&'t str, Option<f64>)> {
    let mut out = Vec::new();
    let mut last = 0;
    for caps in p.pause.captures_iter(line) {
        let Some(m) = caps.get(0) else { continue };
        out.push((&line[last..m.start()], Some(mark_seconds(group(&caps, 1)))));
        last = m.end();
    }
    out.push((&line[last..], None));
    out
}

/// Sentences of one line. A stop inside a number (`13.5`) or a name
/// (`Node.js`) is no sentence end: a Latin stop ends one only before a space
/// or the end of the text. A CJK stop always does.
pub(crate) fn sentences(p: &Patterns, text: &str) -> Vec<String> {
    let text = collapse(p, text);
    p.sentence
        .find_iter(&text)
        .map(|m| trim(m.as_str()))
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// The phrases to speak, each with the silence after it and its kind. The
/// last phrase always ends the script: pause 0, kind [`PhraseKind::End`].
pub fn phrases(script: &str) -> Result<Vec<Phrase>> {
    let p = Patterns::get()?;
    let mut b = Builder {
        out: Vec::new(),
        delivery: None,
    };
    let script = script.replace('\r', "");
    let beats = split(&p.blank_line, &script);
    for (bi, beat) in beats.iter().enumerate() {
        let lines: Vec<&str> = beat.split('\n').collect();
        for (li, raw) in lines.iter().enumerate() {
            let lead = p
                .lead
                .captures(raw)
                .and_then(|caps| Some((group(&caps, 0)?, group(&caps, 1)?)))
                .filter(|(_, note)| is_note(note));
            if let Some((_, note)) = lead {
                b.delivery = Some(format!("({})", trim(note)));
            }
            let line = lead.map_or(*raw, |(whole, _)| &raw[whole.len()..]);
            let (end, end_kind) = if li + 1 < lines.len() {
                (PAUSES.line, PhraseKind::Line)
            } else if bi + 1 < beats.len() {
                (PAUSES.beat, PhraseKind::Beat)
            } else {
                (0.0, PhraseKind::End)
            };
            for (piece, mark) in pause_pieces(p, line) {
                let said = sentences(p, piece);
                for (si, s) in said.iter().enumerate() {
                    match mark {
                        _ if si + 1 < said.len() => {
                            b.add(p, s, PAUSES.sentence, PhraseKind::Sentence);
                        }
                        Some(m) => b.add(p, s, m, PhraseKind::Mark),
                        None => b.add(p, s, end, end_kind),
                    }
                }
                if said.is_empty()
                    && let Some(m) = mark
                {
                    b.add(p, "", m, PhraseKind::Mark);
                }
            }
        }
    }
    if let Some(last) = b.out.last_mut() {
        last.pause = 0.0;
        last.kind = PhraseKind::End;
    }
    Ok(b.out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(script: &str) -> Result<Vec<String>> {
        Ok(sentences(Patterns::get()?, script))
    }

    #[test]
    fn stops_inside_numbers_and_names_do_not_split() -> Result<()> {
        assert_eq!(
            texts("It rose 13.5 percent. Built with Node.js today!")?,
            vec!["It rose 13.5 percent.", "Built with Node.js today!"]
        );
        assert_eq!(
            texts("He said \"Stop.\" Then left.")?,
            vec!["He said \"Stop.\"", "Then left."]
        );
        assert_eq!(texts("你好。今天好！")?, vec!["你好。", "今天好！"]);
        assert_eq!(texts("   ")?, Vec::<String>::new());
        Ok(())
    }

    #[test]
    fn breaks_set_pause_and_kind() -> Result<()> {
        let got = phrases("One. Two.\nThree.\n\nFour.")?;
        let kinds: Vec<(f64, PhraseKind)> = got.iter().map(|p| (p.pause, p.kind)).collect();
        assert_eq!(
            kinds,
            vec![
                (0.3, PhraseKind::Sentence),
                (0.45, PhraseKind::Line),
                (0.8, PhraseKind::Beat),
                (0.0, PhraseKind::End),
            ]
        );
        Ok(())
    }

    #[test]
    fn pause_values() -> Result<()> {
        // `[pause]`, `[pause 0]` and a value that is no number all give 0.6.
        let got = phrases("a [pause] b [pause 0] c [pause 1.2.3] d [pause 1.2] e")?;
        let pauses: Vec<f64> = got.iter().map(|p| p.pause).collect();
        assert_eq!(pauses, vec![0.6, 0.6, 0.6, 1.2, 0.0]);
        assert!(got[..4].iter().all(|p| p.kind == PhraseKind::Mark));
        Ok(())
    }

    #[test]
    fn a_mark_alone_sets_the_previous_pause() -> Result<()> {
        let got = phrases("One.\n[pause 2]\nTwo.")?;
        assert_eq!(got.len(), 2);
        assert_eq!((got[0].pause, got[0].kind), (2.0, PhraseKind::Mark));
        // A mark closing a line replaces the break after it.
        let got = phrases("One. [pause 0.2]\n\nTwo.")?;
        assert_eq!((got[0].pause, got[0].kind), (0.2, PhraseKind::Mark));
        Ok(())
    }

    #[test]
    fn delivery_note_attaches_to_next_phrase() -> Result<()> {
        let got = phrases("(calm tone) One. Two.\n(x) Three.")?;
        assert_eq!(got[0].delivery.as_deref(), Some("(calm tone)"));
        assert_eq!(got[1].delivery, None);
        assert_eq!(got[2].text, "(x) Three.");
        Ok(())
    }

    #[test]
    fn empty_script_has_no_phrases() -> Result<()> {
        assert!(phrases("")?.is_empty());
        assert!(phrases("\n\n  \n")?.is_empty());
        Ok(())
    }
}
