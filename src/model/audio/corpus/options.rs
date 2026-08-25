//! What a corpus run may vary, and where its text tokenizer comes from.

use std::path::Path;

use splintr::{AnyTokenizer, PretrainedVocab, from_json_path, from_vocab};

use crate::error::{Error, Result};
use crate::model::audio::vad::VadSegmentOptions;

/// Whisper's fixed encoder window, in seconds.
///
/// The binding cap on an utterance. NeuCodec accepts 60 s
/// ([`MAX_ENCODE_SAMPLES`](crate::model::audio::neucodec::MAX_ENCODE_SAMPLES)),
/// but every segment is transcribed before it is encoded, so 30 s is the limit
/// that actually applies.
pub const MAX_UTTERANCE_SECS: f32 = 30.0;

/// The pretrained vocabulary names an error lists back when a name does not
/// resolve. Splintr accepts more names than these (see
/// [`PretrainedVocab::from_name`]); these are the four this crate documents,
/// and the ones a speech-LM base model is normally trained against.
pub const PRETRAINED_TOKENIZER_NAMES: [&str; 4] =
    ["cl100k_base", "o200k_base", "llama3", "deepseek_v3"];

/// Where the BASE model's text tokenizer comes from.
///
/// The transcript text must be tokenized with the same vocabulary the speech LM
/// trains on: [`SpeechVocab`](crate::model::speech_lm::SpeechVocab) keeps text
/// ids at their original values so the pretrained embedding rows still line up,
/// so a different tokenizer here silently points every token at another token's
/// learned row.
#[derive(Debug, Clone, Copy)]
pub enum TextTokenizer<'a> {
    /// A splintr built-in, by name — `"cl100k_base"`, `"o200k_base"`,
    /// `"llama3"`, `"deepseek_v3"`.
    ///
    /// A name splintr knows but whose vocabulary this build did not bundle
    /// fails with splintr's own `VocabNotBundled`, which names the cargo
    /// feature to enable.
    Pretrained(&'a str),
    /// A `tokenizer.json` on disk, in the HuggingFace `tokenizers` layout.
    JsonFile(&'a Path),
}

impl TextTokenizer<'_> {
    /// Load the tokenizer this variant names.
    pub fn resolve(&self) -> Result<AnyTokenizer> {
        match self {
            Self::Pretrained(name) => {
                let vocab =
                    PretrainedVocab::from_name(name).ok_or_else(|| Error::InvalidArgument {
                        arg: "tokenizer",
                        reason: format!(
                            "unknown pretrained tokenizer name {name:?}; accepted names are {}",
                            PRETRAINED_TOKENIZER_NAMES.join(", ")
                        ),
                    })?;
                from_vocab(vocab).map_err(|e| Error::ModelError {
                    reason: format!("loading pretrained tokenizer {name:?}: {e}"),
                })
            }
            Self::JsonFile(path) => from_json_path(path).map_err(|e| Error::ModelError {
                reason: format!("loading tokenizer from {}: {e}", path.display()),
            }),
        }
    }
}

/// Tuning for one corpus run.
///
/// [`Default`] is the 16 kHz setup this pipeline is built for, with
/// `vad.max_speech_duration_s` already capped at [`MAX_UTTERANCE_SECS`] — the
/// VAD's own default is `f32::INFINITY`, which
/// [`check_max_speech_duration`] refuses.
#[derive(Debug, Clone)]
pub struct CorpusOptions<'a> {
    /// Sample rate of the decoded recording. NeuCodec and the Silero VAD are
    /// both 16 kHz-only, so this is 16_000 in every working setup.
    pub sample_rate: usize,
    /// Segmentation tuning. `max_speech_duration_s` must be finite and at most
    /// [`MAX_UTTERANCE_SECS`].
    pub vad: VadSegmentOptions,
    /// Whisper language token, e.g. `Some("ms")`. `None` skips it.
    pub language: Option<&'a str>,
    /// Bound on Whisper's decode length, overriding the checkpoint's own.
    pub max_new_tokens: Option<usize>,
    /// Utterances shorter than this are dropped BEFORE transcription —
    /// transcription is the expensive step and a 0.2 s blip yields nothing.
    pub min_utterance_secs: f32,
    /// Pad the packed stream to a multiple of this. `None` is no padding.
    pub pad_to_multiple: Option<usize>,
}

impl Default for CorpusOptions<'_> {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            vad: VadSegmentOptions {
                max_speech_duration_s: MAX_UTTERANCE_SECS,
                ..VadSegmentOptions::default()
            },
            language: None,
            max_new_tokens: None,
            min_utterance_secs: 0.5,
            pad_to_multiple: None,
        }
    }
}

/// Refuse a segmentation cap Whisper cannot transcribe.
///
/// Checked up front, before any model runs. [`VadSegmentOptions::default`] caps
/// nothing (`f32::INFINITY`), so without this guard the common case fails deep
/// inside Whisper on the first long utterance, after a full VAD pass, with an
/// error naming a sample count rather than the option that produced it.
pub fn check_max_speech_duration(max_speech_duration_s: f32) -> Result<()> {
    if !max_speech_duration_s.is_finite() {
        return Err(Error::InvalidArgument {
            arg: "opts.vad.max_speech_duration_s",
            reason: format!(
                "is {max_speech_duration_s}, which sets no cap; cap it at \
                 {MAX_UTTERANCE_SECS} s or less, because every segment is transcribed by \
                 Whisper, whose encoder window is fixed at {MAX_UTTERANCE_SECS} s"
            ),
        });
    }
    if max_speech_duration_s > MAX_UTTERANCE_SECS {
        return Err(Error::InvalidArgument {
            arg: "opts.vad.max_speech_duration_s",
            reason: format!(
                "is {max_speech_duration_s} s; cap it at {MAX_UTTERANCE_SECS} s or less, \
                 because every segment is transcribed by Whisper, whose encoder window is \
                 fixed at {MAX_UTTERANCE_SECS} s"
            ),
        });
    }
    if max_speech_duration_s <= 0.0 {
        return Err(Error::InvalidArgument {
            arg: "opts.vad.max_speech_duration_s",
            reason: format!("is {max_speech_duration_s} s; it must be greater than zero"),
        });
    }
    Ok(())
}
