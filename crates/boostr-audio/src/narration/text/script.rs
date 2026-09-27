//! Script files: subtitles (SRT, WebVTT) and timed scripts.
//!
//! A subtitle file gives both the exact words and where each is spoken. A
//! timed script is a plain script whose lines carry rough times, as written
//! in a brief: `0-5s: Words.`, `[00:05] Words.`, `10 sec - Words.`.

use std::path::Path;

use serde::Serialize;

use crate::error::{Error, Result};
use crate::narration::pattern::{Patterns, collapse, group, number, replace_all_with, split, trim};

/// A subtitle cue.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Cue {
    /// Start, in seconds.
    pub start: f64,
    /// End, in seconds.
    pub end: f64,
    /// The words, markup removed, on one line.
    pub text: String,
}

/// A line of a timed script.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TimedCue {
    /// The time the line asks to start at, in seconds.
    pub start: f64,
    /// The end of a range (`0-5s`), else `None`.
    pub end: Option<f64>,
    /// The line and the untimed lines after it, joined by line breaks.
    pub text: String,
}

/// A script file's spoken words and timing.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScriptSource {
    /// The words: the file as is, the timed lines joined by line breaks, or
    /// the subtitle cues joined by spaces. Time stamps are never part of it.
    pub text: String,
    /// Subtitles only: each cue's exact span. `None` for a timed or plain
    /// script.
    pub phrases: Option<Vec<Cue>>,
    /// Timed scripts only: the time each line asks for. `None` for
    /// subtitles and for a plain script.
    pub targets: Option<Vec<TimedCue>>,
}

/// A file name ending in `.srt` or `.vtt`, any case.
pub(crate) fn is_subtitles(p: &Patterns, file: &str) -> bool {
    p.subtitle_file.is_match(file)
}

/// `00:01:02,500` or `01:02.500` in seconds. NaN when a part is no number.
pub(crate) fn seconds(stamp: &str) -> f64 {
    trim(stamp)
        .replacen(',', ".", 1)
        .split(':')
        .map(number)
        .fold(0.0, |t, v| t * 60.0 + v)
}

/// A time as written in a script (`5s`, `5 sec`, `0:05`) in seconds.
fn to_seconds(p: &Patterns, time: &str) -> f64 {
    match p.unit_suffix.find(time) {
        Some(m) => seconds(&format!("{}{}", &time[..m.start()], &time[m.end()..])),
        None => seconds(time),
    }
}

/// Subtitle cues in file order. Markup (`<i>`, `{\an8}`) stays out of the
/// text, and a cue without words is dropped.
pub fn parse_subtitles(text: &str) -> Result<Vec<Cue>> {
    let p = Patterns::get()?;
    let text = text.replace('\r', "");
    let mut cues = Vec::new();
    for block in split(&p.blank_line, &text) {
        let lines: Vec<&str> = block.split('\n').filter(|l| !trim(l).is_empty()).collect();
        let Some(at) = lines.iter().position(|l| l.contains("-->")) else {
            continue;
        };
        let mut sides = lines[at].split("-->");
        let a = sides.next().unwrap_or_default();
        let b = sides.next().unwrap_or_default();
        let words = replace_all_with(&p.markup, &lines[at + 1..].join(" "), "");
        let words = trim(&collapse(p, &words)).to_string();
        if !words.is_empty() {
            let end = split(&p.spaces, trim(b))
                .first()
                .copied()
                .unwrap_or_default();
            cues.push(Cue {
                start: seconds(a),
                end: seconds(end),
                text: words,
            });
        }
    }
    Ok(cues)
}

/// A timed script's lines with their times. A line without a time continues
/// the line before it. `None` when fewer than two lines carry a time: the
/// text is a plain script.
///
/// A bare number opening a sentence (`5 people came`) is no time: a time
/// needs a unit, a colon, a range, brackets, or a separator after it. A year
/// opening a line (`1857: ...`) is no time either: a plain number over ten
/// minutes needs a colon to be one.
pub fn parse_timed_script(text: &str) -> Result<Option<Vec<TimedCue>>> {
    let p = Patterns::get()?;
    let text = text.replace('\r', "");
    let mut cues: Vec<TimedCue> = Vec::new();
    let mut timed = 0;
    for line in text.split('\n') {
        if trim(line).is_empty() {
            if let Some(last) = cues.last_mut() {
                last.text.push('\n');
            }
            continue;
        }
        let stamp = p
            .stamp
            .captures(line)
            .and_then(|caps| Some((group(&caps, 1)?, group(&caps, 2), group(&caps, 3)?)));
        let looks_timed = stamp.filter(|&(time, range_end, rest)| {
            let head = &line[..line.len() - rest.len()];
            let marked = p.time_marked.is_match(time)
                || range_end.is_some_and(|e| !e.is_empty())
                || p.opens_bracket.is_match(line)
                || p.number_then_separator.is_match(head);
            let plausible = time.contains(':') || to_seconds(p, time) <= 600.0;
            marked && plausible && !trim(rest).is_empty()
        });
        if let Some((time, range_end, rest)) = looks_timed {
            timed += 1;
            cues.push(TimedCue {
                start: to_seconds(p, time),
                end: range_end.map(|e| to_seconds(p, e)),
                text: rest.to_string(),
            });
        } else if let Some(last) = cues.last_mut() {
            last.text.push('\n');
            last.text.push_str(line);
        } else {
            cues.push(TimedCue {
                start: 0.0,
                end: None,
                text: line.to_string(),
            });
        }
    }
    if timed < 2 {
        return Ok(None);
    }
    for c in &mut cues {
        c.text = trim(&c.text).to_string();
    }
    cues.retain(|c| !c.text.is_empty());
    Ok(Some(cues))
}

/// The script held in `raw`, read by the kind `file` names: subtitles for
/// `.srt` and `.vtt`, else a timed script when two lines carry a time, else
/// a plain script.
pub fn script_source(file: &str, raw: &str) -> Result<ScriptSource> {
    let p = Patterns::get()?;
    if !is_subtitles(p, file) {
        return Ok(match parse_timed_script(raw)? {
            Some(timed) => ScriptSource {
                text: timed
                    .iter()
                    .map(|c| c.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
                phrases: None,
                targets: Some(timed),
            },
            None => ScriptSource {
                text: raw.to_string(),
                phrases: None,
                targets: None,
            },
        });
    }
    let cues = parse_subtitles(raw)?;
    if cues.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "file",
            reason: format!(
                "{file} holds no subtitle cues (\"00:00:01,000 --> 00:00:03,000\" lines with text)."
            ),
        });
    }
    Ok(ScriptSource {
        text: cues
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        phrases: Some(cues),
        targets: None,
    })
}

/// [`script_source`] for a file on disk. Bytes that are not UTF-8 read as
/// U+FFFD.
pub fn read_script(file: &Path) -> Result<ScriptSource> {
    let bytes = std::fs::read(file).map_err(|e| Error::InvalidArgument {
        arg: "file",
        reason: format!(
            "reading script {}: {e}. Pass a readable script file.",
            file.display()
        ),
    })?;
    script_source(&file.to_string_lossy(), &String::from_utf8_lossy(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamps_to_seconds() {
        assert_eq!(seconds("00:01:02,500"), 62.5);
        assert_eq!(seconds(" 01:02.500 "), 62.5);
        assert_eq!(seconds(""), 0.0);
        assert!(seconds("1:xx").is_nan());
    }

    #[test]
    fn srt_cues_drop_markup() -> Result<()> {
        let srt = "1\r\n00:00:01,000 --> 00:00:03,500\r\n<i>Every</i> {\\an8}word\r\n\r\n2\r\n00:00:04,000 --> 00:00:05,000\r\n\r\n";
        let cues = parse_subtitles(srt)?;
        assert_eq!(
            cues,
            vec![Cue {
                start: 1.0,
                end: 3.5,
                text: "Every word".into()
            }]
        );
        Ok(())
    }

    #[test]
    fn vtt_cue_settings_are_ignored() -> Result<()> {
        let cues = parse_subtitles("WEBVTT\n\n00:01.000 --> 00:02.000 align:start\nHi.")?;
        assert_eq!((cues[0].start, cues[0].end), (1.0, 2.0));
        Ok(())
    }

    #[test]
    fn timed_script_ranges_and_continuations() -> Result<()> {
        let cues = parse_timed_script(
            "0-5s: Korang tahu tak?\n[00:05] Kuala Lumpur.\nMore.\n15 hingga 20 saat | End.",
        )?;
        let cues = cues.ok_or_else(|| Error::DataError {
            reason: "expected a timed script".into(),
        })?;
        let got: Vec<(f64, Option<f64>, &str)> = cues
            .iter()
            .map(|c| (c.start, c.end, c.text.as_str()))
            .collect();
        assert_eq!(
            got,
            vec![
                (0.0, Some(5.0), "Korang tahu tak?"),
                (5.0, None, "Kuala Lumpur.\nMore."),
                (15.0, Some(20.0), "End."),
            ]
        );
        Ok(())
    }

    #[test]
    fn years_and_bare_numbers_are_not_times() -> Result<()> {
        assert_eq!(
            parse_timed_script("1857: Founded.\n5 people came.\n0:10 Grows.")?,
            None
        );
        assert_eq!(parse_timed_script("0:00 Only one stamp.\nPlain.")?, None);
        Ok(())
    }

    #[test]
    fn empty_subtitles_name_the_file() {
        let err = script_source("a.SRT", "1\nno stamp\n")
            .err()
            .map(|e| e.to_string());
        assert_eq!(
            err.as_deref(),
            Some(
                "invalid argument 'file': a.SRT holds no subtitle cues (\"00:00:01,000 --> 00:00:03,000\" lines with text)."
            )
        );
    }

    #[test]
    fn reads_from_disk() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("brief.txt");
        std::fs::write(&path, "0:00 One.\n0:05 Two.\n")?;
        let got = read_script(&path)?;
        assert_eq!(got.text, "One.\nTwo.");
        assert_eq!(got.phrases, None);
        assert_eq!(got.targets.map(|t| t.len()), Some(2));
        let missing = read_script(&dir.path().join("missing.srt"));
        assert!(matches!(
            missing,
            Err(Error::InvalidArgument { arg: "file", .. })
        ));
        Ok(())
    }
}
