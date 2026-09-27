//! VITS (MMS-TTS): text tokenizer and the ready-to-call synthesis engine.

pub mod engine;
mod load;
pub mod tokenizer;

pub use engine::{VitsEngine, is_vits_config, load_vits_from_dir};
pub use tokenizer::VitsTokenizer;
