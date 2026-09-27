//! Phrases grouped into what VoxCPM2 speaks in one generation.
//!
//! A chunk is a beat: the phrases between blank lines and pause marks,
//! joined so the model sets the pauses inside it from the meaning. Each
//! generation is a separate draw of the voice, so fewer, longer chunks keep
//! it steadier. A chunk over `max` words splits at a sentence end, since
//! long inputs make the model unstable. A delivery note covers its own line
//! as one generation, so the lines after it return to the default style.

use serde::Serialize;

use super::pattern::Patterns;
use super::phrase::{Phrase, PhraseKind, phrases};
use crate::error::Result;

/// How [`chunks`] groups phrases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkOptions {
    /// Every line break also ends a chunk: a timed script places each line
    /// at its own time.
    pub lines: bool,
    /// The most words a chunk holds before it splits at a sentence end.
    pub max: usize,
    /// Blank lines and pause marks stay inside a chunk (their silence is set
    /// after generation), so the script is one take up to `max` words.
    pub whole: bool,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            lines: false,
            max: 40,
            whole: false,
        }
    }
}

/// One generation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Chunk {
    /// The phrases' text joined by spaces.
    pub text: String,
    /// The silence after the last phrase.
    pub pause: f64,
    /// What ends the last phrase.
    pub kind: PhraseKind,
    /// The phrases, to speak one by one when the whole chunk keeps failing.
    pub sentences: Vec<Phrase>,
    /// The delivery note the chunk is spoken with.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delivery: Option<String>,
}

/// The chunk being filled and the chunks done.
struct Grouper {
    out: Vec<Chunk>,
    group: Vec<Phrase>,
    words: usize,
    delivery: Option<String>,
}

impl Grouper {
    fn flush(&mut self) {
        let Some(last) = self.group.last() else {
            return;
        };
        let (pause, kind) = (last.pause, last.kind);
        let group = std::mem::take(&mut self.group);
        self.words = 0;
        self.out.push(Chunk {
            text: group
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            pause,
            kind,
            sentences: group,
            delivery: self.delivery.clone(),
        });
    }
}

/// The script's phrases grouped into chunks. See the module docs.
pub fn chunks(script: &str, options: ChunkOptions) -> Result<Vec<Chunk>> {
    let p = Patterns::get()?;
    let mut g = Grouper {
        out: Vec::new(),
        group: Vec::new(),
        words: 0,
        delivery: None,
    };
    // The note in force: set by a phrase that carries one, cleared once its
    // line is done.
    let mut delivery: Option<String> = None;
    for phrase in phrases(script)? {
        if let Some(note) = &phrase.delivery {
            g.flush();
            delivery = Some(note.clone());
        }
        // The word count a split would give, without allocating its pieces.
        let words = p.spaces.find_iter(&phrase.text).count() + 1;
        if !g.group.is_empty() && g.words + words > options.max {
            g.flush();
        }
        if g.group.is_empty() {
            g.delivery = delivery.clone();
        }
        let kind = phrase.kind;
        g.group.push(phrase);
        g.words += words;
        if delivery.is_some() && kind != PhraseKind::Sentence {
            g.flush();
            delivery = None;
        }
        let open = kind == PhraseKind::Sentence
            || (kind == PhraseKind::Line && !options.lines)
            || (options.whole && matches!(kind, PhraseKind::Beat | PhraseKind::Mark));
        if !open {
            g.flush();
        }
    }
    g.flush();
    Ok(g.out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beats_and_marks_end_chunks() -> Result<()> {
        let got = chunks(
            "One. Two.\nThree.\n\nFour [pause] five.",
            ChunkOptions::default(),
        )?;
        let texts: Vec<&str> = got.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, vec!["One. Two. Three.", "Four", "five."]);
        assert_eq!(got[0].sentences.len(), 3);
        Ok(())
    }

    #[test]
    fn lines_option_splits_every_line() -> Result<()> {
        let options = ChunkOptions {
            lines: true,
            ..ChunkOptions::default()
        };
        assert_eq!(chunks("One line here.\nAnother line.", options)?.len(), 2);
        Ok(())
    }

    #[test]
    fn max_splits_at_a_sentence_end() -> Result<()> {
        let options = ChunkOptions {
            max: 10,
            ..ChunkOptions::default()
        };
        let got = chunks(
            "One two three four five. Six seven eight nine ten. Eleven twelve.",
            options,
        )?;
        let texts: Vec<&str> = got.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                "One two three four five. Six seven eight nine ten.",
                "Eleven twelve."
            ]
        );
        Ok(())
    }

    #[test]
    fn whole_keeps_beats_together() -> Result<()> {
        let options = ChunkOptions {
            whole: true,
            ..ChunkOptions::default()
        };
        let got = chunks("One.\n\nTwo [pause 1] three.", options)?;
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].kind, PhraseKind::End);
        Ok(())
    }

    #[test]
    fn a_note_covers_its_own_line() -> Result<()> {
        let got = chunks("Before.\n(calm) One. Two.\nAfter.", ChunkOptions::default())?;
        let texts: Vec<(&str, Option<&str>)> = got
            .iter()
            .map(|c| (c.text.as_str(), c.delivery.as_deref()))
            .collect();
        assert_eq!(
            texts,
            vec![
                ("Before.", None),
                ("One. Two.", Some("(calm)")),
                ("After.", None)
            ]
        );
        Ok(())
    }
}
