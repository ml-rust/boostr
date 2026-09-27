//! Narration scripts: beats, lines, pause marks, delivery notes, VoxCPM2
//! tags and `{written|spoken}` pairs, split into the phrases and chunks a
//! voice speaks; subtitle and timed script files; the written script
//! captions show; and numbers read back from words and digits.

mod pattern;

pub mod chunk;
pub mod numbers;
pub mod phrase;
pub mod script;
pub mod tags;
pub mod written;

pub use chunk::{Chunk, ChunkOptions, chunks};
pub use numbers::{is_number_word, numbers_in, spoken_numbers, written_numbers};
pub use phrase::{PAUSES, Pauses, Phrase, PhraseKind, phrases};
pub use script::{
    Cue, ScriptSource, TimedCue, parse_subtitles, parse_timed_script, read_script, script_source,
};
pub use tags::{say, sayable, shown, spoken, without_tags};
pub use written::{WrittenScript, written_script};
