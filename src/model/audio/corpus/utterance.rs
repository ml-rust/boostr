//! One prepared utterance, and the flattening of a batch of them.

use crate::error::Result;
use crate::model::audio::corpus::options::CorpusOptions;
use crate::model::audio::vad::SpeechSegment;
use crate::model::speech_lm::layout::SpeechLayout;
use crate::model::speech_lm::pack::{SpeechRecord, pack_records, pack_records_padded};
use crate::model::speech_lm::vocab::SpeechVocab;

/// One prepared utterance: where it sits in the recording, what was said, and
/// the codec frames realising it.
///
/// [`text`](Self::text) is the transcript with surrounding whitespace trimmed,
/// and [`text_tokens`](Self::text_tokens) are that exact string's ids under the
/// builder's tokenizer — the two never disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Utterance {
    /// Half-open sample range `[start, end)` into the recording this came from.
    pub segment: SpeechSegment,
    /// Trimmed transcript, prefixed with `"{speaker}: "` when
    /// [`CorpusOptions::speaker`] is set. Never empty: an utterance that
    /// transcribed to nothing is dropped rather than prepared.
    pub text: String,
    /// `text` under the base tokenizer.
    ///
    /// When [`CorpusOptions::speaker`] is set, `text` already carries the
    /// `"{speaker}: "` prefix, so these ids do too — which is exactly what the
    /// ExpressiveTTS layout needs and why that layout takes no separate speaker
    /// field.
    pub text_tokens: Vec<u32>,
    /// Per-frame codec codes, exactly as
    /// [`SpeechRecord::frames`] expects them.
    pub frames: Vec<Vec<usize>>,
}

/// Build one borrowing [`SpeechRecord`] per utterance, with speaker and style
/// omitted.
///
/// Shared by [`pack_utterances`] and [`pack_utterances_with_layout`]: nothing
/// in this pipeline knows who is speaking or in what style, and a fabricated
/// value for either would train the model to condition on a label that means
/// nothing.
fn build_records(utterances: &[Utterance]) -> Vec<SpeechRecord<'_>> {
    utterances
        .iter()
        .map(|utterance| SpeechRecord {
            speaker: None,
            style: None,
            text: &utterance.text_tokens,
            frames: &utterance.frames,
        })
        .collect()
}

/// Flatten prepared utterances into one speech-LM token stream.
///
/// Speaker and style are omitted: nothing in this pipeline knows who is
/// speaking, and a fabricated speaker name would train the model to condition
/// on a label that means nothing.
///
/// Padding comes from `opts.pad_to_multiple`: `Some` uses
/// [`pack_records_padded`], `None` uses [`pack_records`].
pub fn pack_utterances(
    vocab: &SpeechVocab,
    utterances: &[Utterance],
    opts: &CorpusOptions<'_>,
) -> Result<Vec<u32>> {
    let records = build_records(utterances);
    match opts.pad_to_multiple {
        Some(pad_to_multiple) => pack_records_padded(vocab, &records, pad_to_multiple),
        None => pack_records(vocab, &records),
    }
}

/// Flatten prepared utterances into whichever layout `layout` names.
///
/// Speaker and style are `None` on every record, and that is correct for BOTH
/// layouts:
///
/// - the ExpressiveTTS layout has no speaker delimiter at all — the name is a
///   `"{speaker}: "` prefix already inside [`Utterance::text_tokens`], put there
///   by [`CorpusOptions::speaker`],
/// - nothing in this pipeline knows a style, and a fabricated one would train
///   the model to condition on a label that means nothing.
pub fn pack_utterances_with_layout(
    layout: &SpeechLayout,
    utterances: &[Utterance],
    opts: &CorpusOptions<'_>,
) -> Result<Vec<u32>> {
    let records = build_records(utterances);
    match opts.pad_to_multiple {
        Some(pad_to_multiple) => layout.pack_records_padded(&records, pad_to_multiple),
        None => layout.pack_records(&records),
    }
}
