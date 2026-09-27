//! Script markup and the words left once it is removed.
//!
//! A narration script carries markup beside its words: VoxCPM2 non-verbal
//! tags (`[sigh]`, `[laughing]`), delivery notes opening a line
//! (`(calm, slow) Words.`), pause marks (`[pause]`, `[pause 1.2]`) and
//! `{written|spoken}` pairs. Captions show the written form of a pair, the
//! voice says the spoken form, and nothing else is spoken.

use crate::error::Result;
use crate::narration::pattern::{
    Patterns, collapse, group, replace_all, replace_all_with, trim, utf16_len,
};

/// The shortest and longest delivery note, in UTF-16 code units.
const NOTE_UNITS: std::ops::RangeInclusive<usize> = 2..=120;

/// A delivery note of acceptable length.
pub(crate) fn is_note(note: &str) -> bool {
    NOTE_UNITS.contains(&utf16_len(note))
}

/// The text with VoxCPM2's non-verbal tags removed: runs of spaces and tabs
/// become one space, spaces beside a line break go, and the ends are
/// trimmed. Line breaks stay.
pub fn without_tags(text: &str) -> Result<String> {
    let p = Patterns::get()?;
    Ok(without_tags_with(p, text))
}

fn without_tags_with(p: &Patterns, text: &str) -> String {
    let s = replace_all_with(&p.tags, text, " ");
    let s = replace_all_with(&p.blanks, &s, " ");
    let s = replace_all_with(&p.padded_newline, &s, "\n");
    trim(&s).to_string()
}

/// The words of a script as spoken, on one line: delivery notes, pause
/// marks and tags removed.
fn clean(p: &Patterns, script: &str) -> String {
    let s = replace_all(&p.lead_lines, script, |caps| match group(caps, 1) {
        Some(note) if is_note(note) => String::new(),
        _ => group(caps, 0).unwrap_or_default().to_string(),
    });
    let s = replace_all_with(&p.pause, &s, " ");
    let s = without_tags_with(p, &s);
    trim(&collapse(p, &s)).to_string()
}

/// Each `{written|spoken}` pair replaced by its spoken form.
pub fn say(text: &str) -> Result<String> {
    let p = Patterns::get()?;
    Ok(said_part(p, text, 2))
}

/// Each `{written|spoken}` pair replaced by its written form.
pub fn shown(text: &str) -> Result<String> {
    let p = Patterns::get()?;
    Ok(said_part(p, text, 1))
}

fn said_part(p: &Patterns, text: &str, part: usize) -> String {
    replace_all(&p.said, text, |caps| {
        group(caps, part).unwrap_or_default().to_string()
    })
}

/// The words captions show: written forms, markup removed, one line.
pub fn spoken(script: &str) -> Result<String> {
    let p = Patterns::get()?;
    Ok(clean(p, &said_part(p, script, 1)))
}

/// The words the voice says: spoken forms, markup removed, one line. A
/// take is checked against this text.
pub fn sayable(script: &str) -> Result<String> {
    let p = Patterns::get()?;
    Ok(clean(p, &said_part(p, script, 2)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_go_line_breaks_stay() -> Result<()> {
        assert_eq!(
            without_tags("[sigh]  Not \t again. \n [LAUGHING] Ha")?,
            "Not again.\nHa"
        );
        assert_eq!(without_tags("[sighs] stay")?, "[sighs] stay");
        Ok(())
    }

    #[test]
    fn written_and_spoken_forms() -> Result<()> {
        let s = "Run {SQL|sequel} at {km/h|kilometres per hour}.";
        assert_eq!(say(s)?, "Run sequel at kilometres per hour.");
        assert_eq!(shown(s)?, "Run SQL at km/h.");
        Ok(())
    }

    #[test]
    fn spoken_drops_notes_marks_and_tags() -> Result<()> {
        let s = "(calm) One [pause 1.2] two.\n  (slow)  [sigh] Three {SQL|sequel}.";
        assert_eq!(spoken(s)?, "One two. Three SQL.");
        assert_eq!(sayable(s)?, "One two. Three sequel.");
        Ok(())
    }

    #[test]
    fn note_length_bounds() -> Result<()> {
        // One unit is too short to be a note; a note over 120 units is spoken.
        assert_eq!(spoken("(x) Words.")?, "(x) Words.");
        let long = format!("({}) Words.", "a".repeat(121));
        assert_eq!(spoken(&long)?, long);
        let edge = format!("({}) Words.", "a".repeat(120));
        assert_eq!(spoken(&edge)?, "Words.");
        // An astral character counts as two units, as in JavaScript.
        let astral = format!("({}) Words.", "\u{1F600}".repeat(61));
        assert_eq!(spoken(&astral)?, astral);
        Ok(())
    }

    #[test]
    fn nested_parentheses_are_no_note() -> Result<()> {
        assert_eq!(
            spoken("(outer (inner) note) Words.")?,
            "(outer (inner) note) Words."
        );
        Ok(())
    }
}
