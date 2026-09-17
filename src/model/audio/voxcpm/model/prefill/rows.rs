//! Host-side left-padded layout of a prefill batch: every row's
//! [`SequenceLayout`] shifted right to a common `S_max`, plus the per-row
//! start the attention kernels mask below.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::model::config::REF_AUDIO_FILLER_ID;
use crate::model::audio::voxcpm::model::sequence::SequenceLayout;

/// `B` rows laid out as `[B, S_max]`, each row's pad in front.
///
/// Pad positions carry [`REF_AUDIO_FILLER_ID`] (whose embedding the zero
/// `text_mask` removes anyway), `0.0` in BOTH masks (so they add nothing to
/// `combined_embed`) and, through [`PaddedBatch::kv_start`], are never
/// attended by a real position.
#[derive(Debug, Clone)]
pub struct PaddedBatch {
    /// Per-row layouts, unpadded, in input order.
    pub layouts: Vec<SequenceLayout>,
    /// Longest row, the shared sequence length.
    pub seq_len: usize,
    /// `[B * S_max]` token ids, row-major.
    pub token_ids: Vec<i64>,
    /// `[B * S_max]`, 1.0 at every real text position.
    pub text_mask: Vec<f32>,
    /// `[B * S_max]`, 1.0 at every real reference-audio position.
    pub audio_mask: Vec<f32>,
    /// `[B]`, `S_max - S_b`: the first real position of each row.
    pub kv_start: Vec<i32>,
}

impl PaddedBatch {
    /// Lay out `rows` of `(t_ref, text_token_ids)`; errors on an empty batch
    /// or on any row [`SequenceLayout::build`] rejects.
    pub fn build(rows: &[(usize, &[u32])]) -> Result<Self> {
        if rows.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "rows",
                reason: "expected at least 1 prefill row, got 0".to_string(),
            });
        }
        let layouts = rows
            .iter()
            .map(|&(t_ref, ids)| SequenceLayout::build(t_ref, ids))
            .collect::<Result<Vec<_>>>()?;
        let seq_len = layouts
            .iter()
            .map(SequenceLayout::seq_len)
            .max()
            .unwrap_or(0);
        let total = layouts.len() * seq_len;
        let mut token_ids = Vec::with_capacity(total);
        let mut text_mask = Vec::with_capacity(total);
        let mut audio_mask = Vec::with_capacity(total);
        let mut kv_start = Vec::with_capacity(layouts.len());
        for layout in &layouts {
            let pad = seq_len - layout.seq_len();
            token_ids.extend(std::iter::repeat_n(i64::from(REF_AUDIO_FILLER_ID), pad));
            token_ids.extend_from_slice(&layout.token_ids);
            text_mask.extend(std::iter::repeat_n(0.0f32, pad));
            text_mask.extend_from_slice(&layout.text_mask);
            audio_mask.extend(std::iter::repeat_n(0.0f32, pad));
            audio_mask.extend_from_slice(&layout.audio_mask);
            kv_start.push(i32::try_from(pad).map_err(|_| Error::InvalidArgument {
                arg: "rows",
                reason: format!("left pad {pad} does not fit the kernel's I32 kv_start"),
            })?);
        }
        Ok(Self {
            layouts,
            seq_len,
            token_ids,
            text_mask,
            audio_mask,
            kv_start,
        })
    }

    /// Rows in the batch.
    pub fn batch(&self) -> usize {
        self.layouts.len()
    }

    /// Whether any row is padded. `false` means every attention call can run
    /// unpadded, with no `kv_start` tensor at all.
    pub fn is_padded(&self) -> bool {
        self.kv_start.iter().any(|&start| start > 0)
    }

    /// Left pad of row `b`, in positions.
    pub fn pad(&self, row: usize) -> usize {
        self.seq_len - self.layouts[row].seq_len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::voxcpm::model::config::AUDIO_START_ID;

    const LONG: [u32; 3] = [11, 22, AUDIO_START_ID];
    const SHORT: [u32; 1] = [AUDIO_START_ID];

    /// One row is the plain layout: no pad, no start, ids and masks verbatim.
    #[test]
    fn single_row_is_the_unpadded_layout() {
        let batch = PaddedBatch::build(&[(2, &LONG)]).expect("batch");
        let layout = SequenceLayout::build(2, &LONG).expect("layout");
        assert_eq!(batch.batch(), 1);
        assert_eq!(batch.seq_len, layout.seq_len());
        assert_eq!(batch.token_ids, layout.token_ids);
        assert_eq!(batch.text_mask, layout.text_mask);
        assert_eq!(batch.audio_mask, layout.audio_mask);
        assert_eq!(batch.kv_start, [0]);
        assert!(!batch.is_padded());
    }

    /// Pads sit in FRONT: the short row's real content ends at `S_max - 1`,
    /// its pad positions carry the filler id and zero in both masks.
    #[test]
    fn shorter_rows_are_padded_in_front() {
        let batch = PaddedBatch::build(&[(2, &LONG), (0, &SHORT)]).expect("batch");
        let s = batch.seq_len;
        assert_eq!(s, 2 + 2 + LONG.len());
        assert_eq!(batch.kv_start, [0, (s - SHORT.len()) as i32]);
        assert!(batch.is_padded());
        assert_eq!(batch.pad(1), s - 1);

        let row1 = &batch.token_ids[s..2 * s];
        assert!(
            row1[..s - 1]
                .iter()
                .all(|&id| id == i64::from(REF_AUDIO_FILLER_ID))
        );
        assert_eq!(row1[s - 1], i64::from(AUDIO_START_ID));
        assert!(batch.text_mask[s..2 * s - 1].iter().all(|&m| m == 0.0));
        assert!(batch.audio_mask[s..2 * s - 1].iter().all(|&m| m == 0.0));
        assert_eq!(batch.text_mask[2 * s - 1], 1.0);

        // Row 0 is unpadded and verbatim.
        let layout = SequenceLayout::build(2, &LONG).expect("layout");
        assert_eq!(&batch.token_ids[..s], layout.token_ids.as_slice());
        assert_eq!(&batch.audio_mask[..s], layout.audio_mask.as_slice());
    }

    /// Equal-length rows need no pad, so the batch reports itself unpadded
    /// and the caller skips the `kv_start` tensor.
    #[test]
    fn equal_rows_are_not_padded() {
        let batch = PaddedBatch::build(&[(0, &LONG), (0, &LONG)]).expect("batch");
        assert!(!batch.is_padded());
        assert_eq!(batch.kv_start, [0, 0]);
    }

    #[test]
    fn empty_batch_and_bad_rows_are_rejected() {
        assert!(PaddedBatch::build(&[]).is_err());
        assert!(PaddedBatch::build(&[(0, &LONG), (0, &[11u32])]).is_err());
    }
}
