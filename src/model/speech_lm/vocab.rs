//! Flat token layout shared by text and neural-audio-codec tokens.
//!
//! A text-to-speech LM is a causal decoder over ONE vocabulary: it reads text ids
//! and emits audio-codec ids. That only works if both live in a single flat id
//! space with a layout every downstream stage agrees on — embedding resize, loss
//! masking, sampling constraints, and audio decode all index into it.
//!
//! Nothing here is codec-specific. Every size is derived from [`CodecVocab`], so
//! a single-codebook codec (NeuCodec: 1 codebook x 65_536 entries, 50 frames/sec)
//! and a residual/interleaved codec (SNAC: 3-4 codebooks x 4096 entries, several
//! codes per frame) are both expressible without touching this file.
//!
//! Pure logic: no tensors, no `Runtime`, no device code. Testable without weights.

use crate::error::{Error, Result};

/// Description of a neural audio codec's token space.
///
/// A codec is fully described, for layout purposes, by three things:
/// - how many codebooks it has,
/// - how many entries each codebook holds,
/// - how many codes each codebook contributes to one frame.
///
/// The third field is what makes interleaved codecs work. NeuCodec emits one code
/// per frame from its one codebook. SNAC emits a hierarchy where deeper codebooks
/// fire more often per frame, so a frame is a flat run of codes whose codebook
/// index varies by position. `codes_per_codebook` records that shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecVocab {
    num_codebooks: usize,
    codebook_size: usize,
    codes_per_codebook: Vec<usize>,
}

impl CodecVocab {
    /// Codec where every codebook contributes exactly one code per frame.
    ///
    /// Covers NeuCodec (`new(1, 65_536)`) and flat residual codecs (`new(3, 4096)`).
    pub fn new(num_codebooks: usize, codebook_size: usize) -> Result<Self> {
        Self::with_frame_layout(codebook_size, vec![1; num_codebooks])
    }

    /// Codec with a per-codebook code count per frame, for interleaved layouts.
    ///
    /// `codes_per_codebook[c]` is how many codes codebook `c` emits in one frame.
    /// A SNAC-style hierarchy is `with_frame_layout(4096, vec![1, 2, 4])`.
    pub fn with_frame_layout(codebook_size: usize, codes_per_codebook: Vec<usize>) -> Result<Self> {
        let num_codebooks = codes_per_codebook.len();
        if num_codebooks == 0 {
            return Err(Error::InvalidArgument {
                arg: "num_codebooks",
                reason: "codec must have at least one codebook".to_string(),
            });
        }
        if codebook_size == 0 {
            return Err(Error::InvalidArgument {
                arg: "codebook_size",
                reason: "codebook must have at least one entry".to_string(),
            });
        }
        for (c, n) in codes_per_codebook.iter().enumerate() {
            if *n == 0 {
                return Err(Error::InvalidArgument {
                    arg: "codes_per_codebook",
                    reason: format!("codebook {c} contributes 0 codes per frame"),
                });
            }
        }
        // Ids are u32 on the wire (token ids into an embedding table), so the audio
        // span alone must fit even before text and control tokens are added.
        let span = num_codebooks
            .checked_mul(codebook_size)
            .filter(|s| *s <= u32::MAX as usize)
            .ok_or_else(|| Error::InvalidArgument {
                arg: "codebook_size",
                reason: format!(
                    "{num_codebooks} codebooks x {codebook_size} entries overflows the u32 id space"
                ),
            })?;
        debug_assert!(span > 0);
        Ok(Self {
            num_codebooks,
            codebook_size,
            codes_per_codebook,
        })
    }

    /// Number of codebooks.
    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    /// Entries in each codebook.
    pub fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    /// Total audio ids this codec occupies: `num_codebooks * codebook_size`.
    pub fn total_audio_tokens(&self) -> usize {
        self.num_codebooks * self.codebook_size
    }

    /// Codes in one frame, summed over codebooks.
    pub fn codes_per_frame(&self) -> usize {
        self.codes_per_codebook.iter().sum()
    }

    /// How many codes codebook `c` contributes per frame, or `None` if out of range.
    pub fn codes_of_codebook(&self, codebook: usize) -> Option<usize> {
        self.codes_per_codebook.get(codebook).copied()
    }

    /// Codebook index expected at each position within a frame.
    ///
    /// Length is [`codes_per_frame`](Self::codes_per_frame). This is the ground
    /// truth [`SpeechVocab::decode_frame`] validates a model's emissions against.
    pub fn frame_codebooks(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.codes_per_frame());
        for (c, n) in self.codes_per_codebook.iter().enumerate() {
            out.extend(std::iter::repeat_n(c, *n));
        }
        out
    }
}

/// Control tokens a TTS LM needs.
///
/// Deliberately minimal. A causal text-to-audio decoder needs exactly two things
/// from its control tokens: a delimiter around the text conditioning, and a
/// delimiter around the audio it generates. `EndOfAudio` is the stop condition
/// sampling checks; `StartOfAudio` is the prompt suffix that switches the model
/// from reading to speaking. No speaker, language, or style tokens are defined
/// here — those are task-specific and belong to whoever builds that task, added
/// through the same special-token list rather than baked into this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialToken {
    /// Marks the beginning of the text conditioning segment.
    StartOfText,
    /// Marks the end of the text conditioning segment.
    EndOfText,
    /// Switches the model from consuming text to emitting audio codes.
    StartOfAudio,
    /// Terminates generation; sampling stops here.
    EndOfAudio,
}

/// The flat id layout holding text, control, and audio tokens.
///
/// **The order of the three regions is load-bearing.** Downstream code depends on
/// it: embedding resize appends rows for control+audio to the pretrained matrix,
/// loss masking selects `>= audio_base` to score audio-only, and sampling
/// constrains logits to a contiguous audio slice. Reordering these regions
/// invalidates every checkpoint trained under the old layout.
///
/// ```text
/// [0, text_vocab)                                        text tokens
/// [text_vocab, text_vocab + num_specials)                control tokens
/// [audio_base, audio_base + num_codebooks*codebook_size) audio tokens
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechVocab {
    text_vocab_size: usize,
    specials: Vec<SpecialToken>,
    codec: CodecVocab,
}

impl SpeechVocab {
    /// Build the layout from the base model's text vocab, a codec, and controls.
    ///
    /// `specials` order fixes the control ids, so persist it alongside a
    /// checkpoint. Duplicates are rejected — a token with two ids would make
    /// [`special_id`](Self::special_id) ambiguous and split its gradient.
    pub fn new(
        text_vocab_size: usize,
        codec: CodecVocab,
        specials: Vec<SpecialToken>,
    ) -> Result<Self> {
        if text_vocab_size == 0 {
            return Err(Error::InvalidArgument {
                arg: "text_vocab_size",
                reason: "text vocabulary must be non-empty".to_string(),
            });
        }
        for (i, tok) in specials.iter().enumerate() {
            if specials.iter().take(i).any(|prev| prev == tok) {
                return Err(Error::InvalidArgument {
                    arg: "specials",
                    reason: format!("duplicate special token {tok:?} at index {i}"),
                });
            }
        }
        let total = text_vocab_size
            .checked_add(specials.len())
            .and_then(|n| n.checked_add(codec.total_audio_tokens()))
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "vocabulary of {text_vocab_size} text + {} control + {} audio tokens \
                     overflows the u32 id space",
                    specials.len(),
                    codec.total_audio_tokens()
                ),
            })?;
        debug_assert!(total > 0);
        Ok(Self {
            text_vocab_size,
            specials,
            codec,
        })
    }

    /// The codec this layout was built for.
    pub fn codec(&self) -> &CodecVocab {
        &self.codec
    }

    /// Size of the text region. Text ids are `[0, text_vocab_size)`.
    ///
    /// Text ids keep their ORIGINAL values from the base tokenizer. The pretrained
    /// embedding matrix is indexed by id, so shifting text ids would point every
    /// token at another token's learned row and destroy the pretrained model.
    /// New regions are therefore only ever appended after it.
    pub fn text_vocab_size(&self) -> usize {
        self.text_vocab_size
    }

    /// Number of control tokens.
    pub fn num_specials(&self) -> usize {
        self.specials.len()
    }

    /// First audio id. Audio occupies `[audio_base, total_size)`.
    pub fn audio_base(&self) -> usize {
        self.text_vocab_size + self.specials.len()
    }

    /// Total ids, i.e. the embedding/output-projection row count.
    pub fn total_size(&self) -> usize {
        self.audio_base() + self.codec.total_audio_tokens()
    }

    /// Id of a control token, or `None` if it was not included in the layout.
    pub fn special_id(&self, tok: SpecialToken) -> Option<u32> {
        let idx = self.specials.iter().position(|t| *t == tok)?;
        u32::try_from(self.text_vocab_size + idx).ok()
    }

    /// Flat id for `code` in `codebook`, codebook-major.
    ///
    /// `audio_base + codebook * codebook_size + code`. Codebook-major keeps each
    /// codebook's ids contiguous, so a residual codec can mask logits to the one
    /// codebook legal at the current frame position with a single range.
    pub fn audio_token(&self, codebook: usize, code: usize) -> Result<u32> {
        if codebook >= self.codec.num_codebooks {
            return Err(Error::InvalidArgument {
                arg: "codebook",
                reason: format!(
                    "codebook {codebook} out of range for {} codebooks",
                    self.codec.num_codebooks
                ),
            });
        }
        if code >= self.codec.codebook_size {
            return Err(Error::InvalidArgument {
                arg: "code",
                reason: format!(
                    "code {code} out of range for codebook size {}",
                    self.codec.codebook_size
                ),
            });
        }
        let id = self.audio_base() + codebook * self.codec.codebook_size + code;
        u32::try_from(id).map_err(|_| Error::ModelError {
            reason: format!("audio id {id} exceeds the u32 id space"),
        })
    }

    /// Inverse of [`audio_token`](Self::audio_token); `None` for non-audio ids.
    pub fn decode_audio_token(&self, id: u32) -> Option<(usize, usize)> {
        let id = id as usize;
        let offset = id.checked_sub(self.audio_base())?;
        if offset >= self.codec.total_audio_tokens() {
            return None;
        }
        Some((
            offset / self.codec.codebook_size,
            offset % self.codec.codebook_size,
        ))
    }

    /// True if `id` is a text token.
    pub fn is_text(&self, id: u32) -> bool {
        (id as usize) < self.text_vocab_size
    }

    /// True if `id` is a control token.
    pub fn is_special(&self, id: u32) -> bool {
        let id = id as usize;
        id >= self.text_vocab_size && id < self.audio_base()
    }

    /// True if `id` is an audio token.
    pub fn is_audio(&self, id: u32) -> bool {
        let id = id as usize;
        id >= self.audio_base() && id < self.total_size()
    }

    /// One frame's codes to ids, in frame position order.
    ///
    /// `codes.len()` must equal [`CodecVocab::codes_per_frame`]; position `i`
    /// belongs to the codebook named by [`CodecVocab::frame_codebooks`].
    pub fn encode_frame(&self, codes: &[usize]) -> Result<Vec<u32>> {
        let layout = self.codec.frame_codebooks();
        if codes.len() != layout.len() {
            return Err(Error::InvalidArgument {
                arg: "codes",
                reason: format!(
                    "expected {} codes per frame, got {}",
                    layout.len(),
                    codes.len()
                ),
            });
        }
        let mut ids = Vec::with_capacity(codes.len());
        for (codebook, code) in layout.iter().zip(codes.iter()) {
            ids.push(self.audio_token(*codebook, *code)?);
        }
        Ok(ids)
    }

    /// One frame's ids back to codes, validating codebook index against position.
    ///
    /// The positional check is the point of this method. A model that emits a
    /// codebook-2 token where codebook 0 belongs produces audio that decodes to
    /// noise with no other symptom, so the mismatch is raised as an error here
    /// rather than passed on to the codec.
    pub fn decode_frame(&self, ids: &[u32]) -> Result<Vec<usize>> {
        let layout = self.codec.frame_codebooks();
        if ids.len() != layout.len() {
            return Err(Error::InvalidArgument {
                arg: "ids",
                reason: format!("expected {} ids per frame, got {}", layout.len(), ids.len()),
            });
        }
        let mut codes = Vec::with_capacity(ids.len());
        for (pos, (expected, id)) in layout.iter().zip(ids.iter()).enumerate() {
            let (codebook, code) =
                self.decode_audio_token(*id)
                    .ok_or_else(|| Error::InvalidArgument {
                        arg: "ids",
                        reason: format!("id {id} at frame position {pos} is not an audio token"),
                    })?;
            if codebook != *expected {
                return Err(Error::InvalidArgument {
                    arg: "ids",
                    reason: format!(
                        "id {id} at frame position {pos} belongs to codebook {codebook}, \
                         expected codebook {expected}"
                    ),
                });
            }
            codes.push(code);
        }
        Ok(codes)
    }
}

#[cfg(test)]
#[path = "vocab_tests.rs"]
mod tests;
