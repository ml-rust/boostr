//! Host-side layout of the VoxCPM2 voice-clone prefill sequence: token ids
//! and the two position masks, built once and uploaded as tensors by
//! [`prefill`](crate::model::audio::voxcpm::model::VoxCpm2Model::prefill).
//!
//! # Reference ("voice clone") mode
//!
//! The reference TRANSCRIPT is NOT used — only the reference AUDIO. There is
//! no reference-text parameter anywhere on this path.
//!
//! With `z1 = zeros([1, patch_size, feat_dim])` the reference prefix is:
//!
//! ```text
//! tokens = [103] ++ [0; T_ref] ++ [104]
//! feats  = z1    ++ ref_feat   ++ z1
//! t_mask = [1]   ++ [0; T_ref] ++ [1]
//! a_mask = [0]   ++ [1; T_ref] ++ [0]
//! ```
//!
//! and the full sequence, of length `S = T_ref + 2 + text_length`, appends
//! the (already `101`-terminated) text prompt:
//!
//! ```text
//! text_token = ref_tokens ++ text_token
//! audio_feat = ref_feats  ++ zeros([text_length, patch_size, feat_dim])
//! text_mask  = ref_t_mask ++ [1; text_length]
//! audio_mask = ref_a_mask ++ [0; text_length]
//! ```
//!
//! # Why complementarity is checked, not assumed
//!
//! `combined_embed` is a SUM of `text_mask * text_embed` and `audio_mask *
//! feat_embed`. A position set in BOTH masks is double-counted; a position
//! set in NEITHER contributes exactly zero. Both stay shape-valid and both
//! run silently to a wrong model, so [`check_mask_complementarity`] rejects
//! them here, before anything is uploaded.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::model::config::{
    AUDIO_START_ID, REF_AUDIO_END_ID, REF_AUDIO_FILLER_ID, REF_AUDIO_START_ID,
};

/// Token ids and per-position masks for one voice-clone prefill sequence.
///
/// `token_ids` is `i64` because that is the index dtype
/// [`Embedding::forward`](crate::nn::Embedding::forward) normalizes to; the
/// masks are `f32` and are cast to the model's dtype at upload.
#[derive(Debug, Clone)]
pub struct SequenceLayout {
    /// `[S]` token ids. Reference-audio positions carry
    /// [`REF_AUDIO_FILLER_ID`], whose embedding is masked out.
    pub token_ids: Vec<i64>,
    /// `[S]`, 1.0 at every text position (the two delimiters plus the whole
    /// prompt), 0.0 elsewhere.
    pub text_mask: Vec<f32>,
    /// `[S]`, 1.0 at every reference-audio position, 0.0 elsewhere.
    pub audio_mask: Vec<f32>,
    /// Reference-audio patches, i.e. `ref_feat`'s leading axis.
    pub t_ref: usize,
    /// Prompt length, including its trailing [`AUDIO_START_ID`].
    pub text_length: usize,
}

impl SequenceLayout {
    /// Build the layout for `t_ref` reference-audio patches followed by
    /// `text_token_ids`.
    ///
    /// `text_token_ids` must be non-empty and must already end with
    /// [`AUDIO_START_ID`] — boostr does not tokenize on this path, so it
    /// cannot append the terminator itself, and a prompt missing it prefills
    /// a sequence the sampling loop was never primed for.
    ///
    /// Errors when `t_ref` is 0, when `text_token_ids` is empty, or when the
    /// prompt does not end with [`AUDIO_START_ID`].
    pub fn build(t_ref: usize, text_token_ids: &[u32]) -> Result<Self> {
        if t_ref == 0 {
            return Err(Error::InvalidArgument {
                arg: "t_ref",
                reason: "expected at least 1 reference-audio patch, got 0".to_string(),
            });
        }
        let text_length = text_token_ids.len();
        let last = *text_token_ids
            .last()
            .ok_or_else(|| Error::InvalidArgument {
                arg: "text_token_ids",
                reason: "expected a non-empty prompt, got 0 tokens".to_string(),
            })?;
        if last != AUDIO_START_ID {
            return Err(Error::InvalidArgument {
                arg: "text_token_ids",
                reason: format!(
                    "expected the prompt to end with AUDIO_START_ID \
                     ({AUDIO_START_ID}), got {last}"
                ),
            });
        }

        let seq_len = t_ref + 2 + text_length;
        let mut token_ids = Vec::with_capacity(seq_len);
        let mut text_mask = Vec::with_capacity(seq_len);
        let mut audio_mask = Vec::with_capacity(seq_len);

        // [103]: a TEXT position backed by a zero audio patch.
        token_ids.push(i64::from(REF_AUDIO_START_ID));
        text_mask.push(1.0);
        audio_mask.push(0.0);

        // [0; T_ref]: the reference-audio span. Filler ids, audio mask only.
        token_ids.extend(std::iter::repeat_n(i64::from(REF_AUDIO_FILLER_ID), t_ref));
        text_mask.extend(std::iter::repeat_n(0.0f32, t_ref));
        audio_mask.extend(std::iter::repeat_n(1.0f32, t_ref));

        // [104]: closes the span, again a TEXT position.
        token_ids.push(i64::from(REF_AUDIO_END_ID));
        text_mask.push(1.0);
        audio_mask.push(0.0);

        // The prompt: every position is text, none carries audio. The LAST
        // row of the whole sequence therefore lands here, which is why
        // `lm_hidden` comes back UN-fsq'd — see `prefill`.
        token_ids.extend(text_token_ids.iter().map(|&id| i64::from(id)));
        text_mask.extend(std::iter::repeat_n(1.0f32, text_length));
        audio_mask.extend(std::iter::repeat_n(0.0f32, text_length));

        check_mask_complementarity(&text_mask, &audio_mask)?;

        Ok(Self {
            token_ids,
            text_mask,
            audio_mask,
            t_ref,
            text_length,
        })
    }

    /// `S = T_ref + 2 + text_length`.
    pub fn seq_len(&self) -> usize {
        self.token_ids.len()
    }
}

/// Reject masks that are not complementary: every position must be in
/// EXACTLY one of them.
///
/// A position in both is summed twice into `combined_embed`; a position in
/// neither contributes zero. Neither is a shape error, so neither is caught
/// anywhere else.
pub fn check_mask_complementarity(text_mask: &[f32], audio_mask: &[f32]) -> Result<()> {
    if text_mask.len() != audio_mask.len() {
        return Err(Error::InvalidArgument {
            arg: "audio_mask",
            reason: format!(
                "expected the same length as text_mask ({}), got {}",
                text_mask.len(),
                audio_mask.len()
            ),
        });
    }
    for (position, (&text, &audio)) in text_mask.iter().zip(audio_mask).enumerate() {
        if (text + audio - 1.0).abs() > f32::EPSILON {
            return Err(Error::InvalidArgument {
                arg: "text_mask",
                reason: format!(
                    "expected text_mask + audio_mask == 1.0 at every position, got \
                     {text} + {audio} at position {position}: a position in both masks \
                     is double-counted in combined_embed, one in neither contributes zero"
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROMPT: [u32; 3] = [11, 22, AUDIO_START_ID];

    #[test]
    fn seq_len_is_t_ref_plus_two_plus_text_length() {
        for t_ref in [1usize, 5, 37] {
            let layout = SequenceLayout::build(t_ref, &PROMPT).expect("layout");
            assert_eq!(layout.seq_len(), t_ref + 2 + PROMPT.len());
            assert_eq!(layout.text_mask.len(), layout.seq_len());
            assert_eq!(layout.audio_mask.len(), layout.seq_len());
        }
    }

    #[test]
    fn masks_are_complementary_and_cover_every_position() {
        let layout = SequenceLayout::build(5, &PROMPT).expect("layout");
        check_mask_complementarity(&layout.text_mask, &layout.audio_mask).expect("complementary");
        assert_eq!(
            layout.audio_mask.iter().sum::<f32>(),
            5.0,
            "exactly T_ref audio positions"
        );
        assert_eq!(
            layout.text_mask.iter().sum::<f32>(),
            (2 + PROMPT.len()) as f32,
            "exactly the two delimiters plus the prompt are text positions"
        );
    }

    #[test]
    fn overlapping_masks_are_rejected() {
        // Position 1 set in BOTH: double-counted in combined_embed.
        let text = [1.0f32, 1.0, 0.0];
        let audio = [0.0f32, 1.0, 1.0];
        let err = check_mask_complementarity(&text, &audio).unwrap_err();
        assert!(err.to_string().contains("position 1"), "got {err}");
    }

    #[test]
    fn gap_in_masks_is_rejected() {
        // Position 2 in NEITHER: contributes exactly zero.
        let text = [1.0f32, 0.0, 0.0];
        let audio = [0.0f32, 1.0, 0.0];
        let err = check_mask_complementarity(&text, &audio).unwrap_err();
        assert!(err.to_string().contains("position 2"), "got {err}");
    }

    #[test]
    fn mismatched_mask_lengths_are_rejected() {
        assert!(check_mask_complementarity(&[1.0, 0.0], &[0.0]).is_err());
    }

    #[test]
    fn delimiters_and_prompt_land_at_the_right_positions() {
        let t_ref = 4usize;
        let layout = SequenceLayout::build(t_ref, &PROMPT).expect("layout");

        assert_eq!(layout.token_ids[0], i64::from(REF_AUDIO_START_ID));
        assert_eq!(layout.text_mask[0], 1.0);
        assert_eq!(layout.audio_mask[0], 0.0);

        for i in 1..=t_ref {
            assert_eq!(layout.token_ids[i], i64::from(REF_AUDIO_FILLER_ID));
            assert_eq!(layout.audio_mask[i], 1.0, "position {i} must be audio");
            assert_eq!(layout.text_mask[i], 0.0);
        }

        assert_eq!(layout.token_ids[t_ref + 1], i64::from(REF_AUDIO_END_ID));
        assert_eq!(layout.text_mask[t_ref + 1], 1.0);
        assert_eq!(layout.audio_mask[t_ref + 1], 0.0);

        let prompt: Vec<i64> = PROMPT.iter().map(|&id| i64::from(id)).collect();
        assert_eq!(&layout.token_ids[t_ref + 2..], prompt.as_slice());

        // The last row is a TEXT position — the premise behind `lm_hidden`
        // being un-fsq'd.
        let last = layout.seq_len() - 1;
        assert_eq!(layout.text_mask[last], 1.0);
        assert_eq!(layout.audio_mask[last], 0.0);
    }

    #[test]
    fn rejects_prompt_not_ending_in_audio_start() {
        let err = SequenceLayout::build(2, &[11, 22]).unwrap_err();
        assert!(err.to_string().contains("AUDIO_START_ID"), "got {err}");
    }

    #[test]
    fn rejects_empty_prompt_and_zero_reference() {
        assert!(SequenceLayout::build(2, &[]).is_err());
        assert!(SequenceLayout::build(0, &PROMPT).is_err());
    }
}
