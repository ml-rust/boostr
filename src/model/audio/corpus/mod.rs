//! Speech-corpus preparation: a decoded recording in, a packed speech-LM token
//! stream out.
//!
//! The pipeline composes four already-verified pieces and adds no signal
//! processing of its own:
//!
//! ```text
//! samples (16 kHz mono)
//!   -> SileroVad::speech_timestamps      -> Vec<SpeechSegment>
//!   -> WhisperBundle::transcribe         -> text, per segment
//!   -> NeuCodecEncoder::encode_frames    -> Vec<Vec<usize>>, per segment
//!   -> speech_lm::pack::pack_records     -> Vec<u32>
//! ```
//!
//! Tokenization lives here rather than in the driving CLI: the transcript text
//! must be tokenized with the SAME vocabulary the speech LM trains on, and
//! [`SpeechVocab`](crate::model::speech_lm::SpeechVocab)'s text region is that
//! tokenizer's vocabulary.

pub mod builder;
pub mod options;
pub mod utterance;

pub use builder::SpeechCorpusBuilder;
pub use options::{
    CorpusOptions, MAX_UTTERANCE_SECS, PRETRAINED_TOKENIZER_NAMES, TextTokenizer,
    check_max_speech_duration,
};
pub use utterance::{Utterance, pack_utterances};

#[cfg(test)]
mod tests;
