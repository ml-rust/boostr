//! Narration scripts: phrases and chunks a voice speaks, tags, subtitle and
//! timed script files, the written script, numbers read back from words and
//! digits, and the check of a spoken take and its timings against its script.

mod js;
mod pattern;
mod speech_match;

pub mod align_script;
pub mod chunk;
pub mod heard;
pub mod numbers;
pub mod phrase;
pub mod script;
pub mod speech_check;
pub mod tags;
pub mod written;

pub use align_script::{STRAY_PAUSE, StrayPause, align_script, stray_pauses};
pub use chunk::{Chunk, ChunkOptions, chunks};
pub use heard::HeardWord;
pub use numbers::{is_number_word, numbers_in, spoken_numbers, written_numbers};
pub use phrase::{PAUSES, Pauses, Phrase, PhraseKind, phrases};
pub use script::{
    Cue, ScriptSource, TimedCue, parse_subtitles, parse_timed_script, read_script, script_source,
};
pub use speech_check::{Extra, Verdict, check_speech};
pub use tags::{say, sayable, shown, spoken, without_tags};
pub use written::{WrittenScript, written_script};
