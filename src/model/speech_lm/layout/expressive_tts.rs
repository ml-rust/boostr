//! The token layout of `Scicom-intl/Multilingual-Expressive-TTS-1.7B`.
//!
//! This base is a Qwen3 causal LM continued-pretrained for TTS. It already has a
//! trained token layout, so a fine-tune MUST emit that layout and not a layout of
//! boostr's own: every id addresses a specific embedding row, and packing our
//! corpus in [`SpeechVocab`](crate::model::speech_lm::SpeechVocab)'s scheme would
//! point each token at a row that means something else.
//!
//! # The sequence
//!
//! From the model card, verbatim
//! (`prompt = f"<|im_start|>{speaker}: {text}<|speech_start|>"`):
//!
//! ```text
//! [151644] ++ tokenize("{speaker}: {text}") ++ [151669] ++ [151670 + c ...] ++ [151645]
//! ```
//!
//! and the optional description variant:
//!
//! ```text
//! <|im_start|>{speaker}: {text}<|description|>{description}<|speech_start|>
//! ```
//!
//! # The speaker is TEXT, not a token
//!
//! The speaker is free text (the card uses `DisfluencySpeech` and `Rahman`),
//! rendered as a literal `"{speaker}: "` prefix INSIDE the tokenized text. There
//! is no speaker delimiter to put it behind. So the caller renders
//! `"{speaker}: {text}"` and tokenizes THAT — this layout cannot tokenize, because
//! it is handed ids, not strings, and joining two separately-tokenized pieces
//! would produce a different token boundary at the seam than the base was trained
//! on.
//!
//! [`ExpressiveTtsLayout::pack_record`] therefore REFUSES a
//! [`SpeechRecord`] whose `speaker` is `Some`, rather than
//! guessing where those ids belong. A rejected record is a caller who has not
//! pre-rendered yet; a silently accepted one would be a corpus that trains the
//! model on a prefix the base has never seen.
//!
//! # What this layout cannot express
//!
//! - **No `<|speech_end|>`.** The audio run is closed by `<|im_end|>`, which is
//!   also the checkpoint's `eos_token_id`. Do not invent a separate audio
//!   terminator.
//! - **One codebook only.** NeuCodec emits a single code per frame, and the base's
//!   audio ids are one contiguous run of 65_536. A frame carrying more than one
//!   code has nowhere to go and is rejected by index.
//! - **Control tokens sit AFTER the audio run** (`<|description|>` = 217_206
//!   follows `<|s_65535|>` = 217_205), which is exactly the fragmentation
//!   [`SpeechVocab`](crate::model::speech_lm::SpeechVocab) is documented as
//!   refusing to model. That is why this layout is a separate type and not a
//!   `SpeechVocab` built with unusual arguments.

use crate::error::{Error, Result};
use crate::model::speech_lm::pack::SpeechRecord;

/// `<|im_start|>`, opening every sequence.
///
/// This and every other id in this module were read from the
/// `added_tokens.json` and `config.json` of the
/// `Scicom-intl/Multilingual-Expressive-TTS-1.7B` checkpoint. They are FACTS
/// about that checkpoint's trained embedding rows, stated once here so nobody
/// re-derives them. Changing one silently retargets every token.
pub const IM_START: u32 = 151_644;

/// `<|im_end|>`, closing the audio run. Also the checkpoint's `eos_token_id`.
pub const IM_END: u32 = 151_645;

/// `<|endoftext|>`, the checkpoint's `pad_token_id`.
pub const ENDOFTEXT: u32 = 151_643;

/// `<|speech_start|>`, the switch from reading text to emitting audio codes.
pub const SPEECH_START: u32 = 151_669;

/// Id of `<|s_0|>`. Audio code `c` is `AUDIO_BASE + c`, contiguous to
/// `<|s_65535|>` = 217_205.
pub const AUDIO_BASE: u32 = 151_670;

/// `<|description|>`, opening the optional style description.
pub const DESCRIPTION: u32 = 217_206;

/// `<|description_category|>`, defined by the checkpoint and NOT used by this
/// layout: the model card documents no sequence containing it.
pub const DESCRIPTION_CATEGORY: u32 = 217_207;

/// Codes in the base's single audio codebook, `<|s_0|>` through `<|s_65535|>`.
pub const CODEBOOK_SIZE: usize = 65_536;

/// The checkpoint's `vocab_size`: the embedding and output-projection row count.
pub const VOCAB_SIZE: usize = 217_208;

/// The `Multilingual-Expressive-TTS-1.7B` token layout.
///
/// Every field is one of the module constants; the struct exists so a layout can
/// be passed around, compared, and matched against a checkpoint rather than
/// re-derived at each call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpressiveTtsLayout {
    seq_start: u32,
    speech_start: u32,
    seq_end: u32,
    description: u32,
    audio_base: u32,
    codebook_size: usize,
    vocab_size: usize,
}

impl Default for ExpressiveTtsLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpressiveTtsLayout {
    /// The layout of the published checkpoint.
    pub const fn new() -> Self {
        Self {
            seq_start: IM_START,
            speech_start: SPEECH_START,
            seq_end: IM_END,
            description: DESCRIPTION,
            audio_base: AUDIO_BASE,
            codebook_size: CODEBOOK_SIZE,
            vocab_size: VOCAB_SIZE,
        }
    }

    /// `<|im_start|>`.
    pub const fn seq_start(&self) -> u32 {
        self.seq_start
    }

    /// `<|speech_start|>`.
    pub const fn speech_start(&self) -> u32 {
        self.speech_start
    }

    /// `<|im_end|>`, which closes the audio run AND is the sampling stop
    /// condition. There is no separate speech-end token.
    pub const fn eos_id(&self) -> u32 {
        self.seq_end
    }

    /// `<|endoftext|>`, the checkpoint's pad id.
    pub const fn pad_id(&self) -> u32 {
        ENDOFTEXT
    }

    /// `<|description|>`.
    pub const fn description_id(&self) -> u32 {
        self.description
    }

    /// Id of `<|s_0|>`.
    pub const fn audio_base(&self) -> u32 {
        self.audio_base
    }

    /// Codes in the single codebook.
    pub const fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    /// The checkpoint's embedding row count.
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// True if a checkpoint with `rows` embedding rows matches this layout.
    pub const fn matches_embedding_rows(&self, rows: usize) -> bool {
        rows == self.vocab_size
    }

    /// Flat id for one audio code: `audio_base + code`.
    pub fn audio_token(&self, code: usize) -> Result<u32> {
        if code >= self.codebook_size {
            return Err(Error::InvalidArgument {
                arg: "code",
                reason: format!(
                    "audio code {code} is out of range for this layout's single codebook of \
                     {} codes; valid codes are 0..{}",
                    self.codebook_size, self.codebook_size
                ),
            });
        }
        let base = self.audio_base as usize;
        u32::try_from(base + code).map_err(|_| Error::ModelError {
            reason: format!("audio id {} exceeds the u32 id space", base + code),
        })
    }

    /// Inverse of [`audio_token`](Self::audio_token); `None` for a non-audio id.
    pub fn decode_audio_token(&self, id: u32) -> Option<usize> {
        let code = (id as usize).checked_sub(self.audio_base as usize)?;
        (code < self.codebook_size).then_some(code)
    }

    /// Flatten one record into the sequence documented at [module level](self).
    ///
    /// `record.speaker` MUST be `None`: see the module docs — the speaker is a
    /// `"{speaker}: "` prefix the caller renders and tokenizes together with the
    /// text, and a `Some` here is rejected rather than guessed at.
    ///
    /// `record.style`, when present, is emitted as the card's description
    /// variant: `<|description|>` followed by the style ids, between the text and
    /// `<|speech_start|>`. Those ids must come from the BASE tokenizer, like the
    /// text ids.
    pub fn pack_record(&self, record: &SpeechRecord<'_>) -> Result<Vec<u32>> {
        if record.speaker.is_some() {
            return Err(Error::InvalidArgument {
                arg: "speaker",
                reason: "this layout has no speaker delimiter: the base is trained on a literal \
                         \"{speaker}: \" prefix inside the tokenized text, so render \
                         \"{speaker}: {text}\" and tokenize that into `text`, leaving `speaker` \
                         as None"
                    .to_string(),
            });
        }

        let style_len = record.style.map_or(0, |s| s.len() + 1);
        let mut out = Vec::with_capacity(record.text.len() + style_len + record.frames.len() + 3);

        out.push(self.seq_start);
        out.extend_from_slice(record.text);
        if let Some(style) = record.style {
            out.push(self.description);
            out.extend_from_slice(style);
        }
        out.push(self.speech_start);

        for (i, frame) in record.frames.iter().enumerate() {
            if frame.len() != 1 {
                return Err(Error::InvalidArgument {
                    arg: "frames",
                    reason: format!(
                        "frame {i} holds {} codes; this layout has ONE codebook (NeuCodec emits \
                         one code per frame), so every frame must hold exactly 1 code",
                        frame.len()
                    ),
                });
            }
            let code = frame[0];
            let id = self.audio_token(code).map_err(|e| Error::InvalidArgument {
                arg: "frames",
                reason: format!("frame {i}: {e}"),
            })?;
            out.push(id);
        }

        out.push(self.seq_end);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    //! The sequence these tests pin IS the contract with
    //! `Scicom-intl/Multilingual-Expressive-TTS-1.7B`. Every expected vector below is
    //! hand-computed from the checkpoint's `added_tokens.json` ids, not from the code
    //! under test, so a change to the layout fails here rather than in a fine-tune
    //! that merely learns worse.

    use super::*;

    /// Two arbitrary text ids, well inside Qwen3's text region.
    const TEXT: [u32; 3] = [9707, 11, 1879];

    fn record<'a>(text: &'a [u32], frames: &'a [Vec<usize>]) -> SpeechRecord<'a> {
        SpeechRecord {
            speaker: None,
            style: None,
            text,
            frames,
        }
    }

    #[test]
    fn packs_the_exact_sequence_from_the_model_card() {
        let layout = ExpressiveTtsLayout::new();
        let frames = vec![vec![0usize], vec![1], vec![65_535]];
        let packed = layout
            .pack_record(&record(&TEXT, &frames))
            .expect("a one-codebook record packs");

        // <|im_start|> ++ text ++ <|speech_start|> ++ 151670 + c ++ <|im_end|>
        let expected: Vec<u32> = vec![
            151_644, 9707, 11, 1879, 151_669, 151_670, 151_671, 217_205, 151_645,
        ];
        assert_eq!(packed, expected);
    }

    #[test]
    fn code_zero_and_code_65535_pin_the_ends_of_the_audio_run() {
        let layout = ExpressiveTtsLayout::new();
        assert_eq!(layout.audio_token(0).expect("code 0 is in range"), 151_670);
        assert_eq!(
            layout.audio_token(65_535).expect("code 65535 is in range"),
            217_205
        );
        assert_eq!(layout.decode_audio_token(151_670), Some(0));
        assert_eq!(layout.decode_audio_token(217_205), Some(65_535));
        // <|description|> sits immediately after the audio run and is not audio.
        assert_eq!(layout.decode_audio_token(217_206), None);
        assert_eq!(layout.decode_audio_token(151_669), None);
    }

    #[test]
    fn code_65536_is_rejected_and_named_with_its_frame() {
        let layout = ExpressiveTtsLayout::new();
        let frames = vec![vec![7usize], vec![65_536]];
        let err = layout
            .pack_record(&record(&TEXT, &frames))
            .expect_err("code 65536 is past the codebook");
        let msg = err.to_string();
        assert!(msg.contains("65536"), "must name the code: {msg}");
        assert!(msg.contains("frame 1"), "must name the frame: {msg}");
    }

    #[test]
    fn a_frame_that_is_not_one_code_is_rejected_by_index_and_length() {
        let layout = ExpressiveTtsLayout::new();
        let frames = vec![vec![1usize], vec![2, 3, 4]];
        let err = layout
            .pack_record(&record(&TEXT, &frames))
            .expect_err("a 3-code frame has nowhere to go in a single-codebook layout");
        let msg = err.to_string();
        assert!(msg.contains("frame 1"), "must name the frame index: {msg}");
        assert!(msg.contains("3 codes"), "must name the length: {msg}");
    }

    #[test]
    fn a_some_speaker_is_refused_rather_than_guessed_at() {
        let layout = ExpressiveTtsLayout::new();
        let speaker = [1234u32];
        let frames = vec![vec![0usize]];
        let rec = SpeechRecord {
            speaker: Some(&speaker),
            style: None,
            text: &TEXT,
            frames: &frames,
        };
        let err = layout
            .pack_record(&rec)
            .expect_err("this layout has no speaker delimiter");
        let msg = err.to_string();
        assert!(
            msg.contains("{speaker}: {text}"),
            "the error must tell the caller to pre-render the prefix: {msg}"
        );
    }

    #[test]
    fn style_becomes_the_description_variant() {
        let layout = ExpressiveTtsLayout::new();
        let style = [100u32, 200];
        let frames = vec![vec![2usize]];
        let rec = SpeechRecord {
            speaker: None,
            style: Some(&style),
            text: &TEXT,
            frames: &frames,
        };
        let packed = layout
            .pack_record(&rec)
            .expect("the description variant packs");

        // <|im_start|> text <|description|> description <|speech_start|> audio <|im_end|>
        let expected: Vec<u32> = vec![
            151_644, 9707, 11, 1879, 217_206, 100, 200, 151_669, 151_672, 151_645,
        ];
        assert_eq!(packed, expected);
    }

    #[test]
    fn empty_frames_still_emit_both_delimiters() {
        let layout = ExpressiveTtsLayout::new();
        let frames: Vec<Vec<usize>> = Vec::new();
        let packed = layout
            .pack_record(&record(&TEXT, &frames))
            .expect("a record with no audio packs");
        assert_eq!(packed, vec![151_644, 9707, 11, 1879, 151_669, 151_645]);
    }

    #[test]
    fn empty_text_still_emits_both_delimiters() {
        let layout = ExpressiveTtsLayout::new();
        let frames = vec![vec![65_535usize]];
        let packed = layout
            .pack_record(&record(&[], &frames))
            .expect("a record with no text packs");
        assert_eq!(packed, vec![151_644, 151_669, 217_205, 151_645]);
    }

    #[test]
    fn the_checkpoint_ids_are_the_ones_this_layout_carries() {
        let layout = ExpressiveTtsLayout::new();
        assert_eq!(layout.seq_start(), 151_644);
        assert_eq!(layout.speech_start(), 151_669);
        assert_eq!(layout.eos_id(), 151_645);
        assert_eq!(layout.pad_id(), 151_643);
        assert_eq!(layout.description_id(), 217_206);
        assert_eq!(layout.audio_base(), 151_670);
        assert_eq!(layout.codebook_size(), 65_536);
        assert_eq!(layout.vocab_size(), 217_208);
        assert_eq!(DESCRIPTION_CATEGORY, 217_207);
        // The audio run ends exactly where <|description|> begins: no gap, no overlap.
        assert_eq!(AUDIO_BASE as usize + CODEBOOK_SIZE, DESCRIPTION as usize);
        assert!(layout.matches_embedding_rows(217_208));
        assert!(!layout.matches_embedding_rows(217_207));
    }
}
