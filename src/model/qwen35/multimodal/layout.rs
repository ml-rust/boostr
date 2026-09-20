//! Row layout and IMROPE positions of a `qwen35` prompt holding images,
//! without tensors.
//!
//! An image arrives in the token ids as one `image_pad` placeholder between
//! `vision_start` and `vision_end`. The placeholder expands to the image's
//! `nx * ny` embedding rows. Positions follow the reference decoder:
//!
//! - a text token at position `p` has `t = h = w = p`, `e = 0`
//! - image row `i` of a run starting at `p0` has `t = p0`,
//!   `h = p0 + i / nx`, `w = p0 + i % nx`, `e = 0`
//! - the token after the run sits at `p0 + max(nx, ny)`
//!
//! The KV slot of every row is its index in the expanded sequence, so from
//! the first image on the rope position lags the slot.

use crate::error::{Error, Result};

/// The three vocabulary ids that bracket and stand in for an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionMarkers {
    /// `<|vision_start|>`, the token before the image rows.
    pub vision_start: u32,
    /// `<|vision_end|>`, the token after the image rows.
    pub vision_end: u32,
    /// `<|image_pad|>`, one per image in the ids; never reaches the model.
    pub image_pad: u32,
}

/// Token grid of one encoded image: `nx` columns by `ny` rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageGrid {
    pub nx: usize,
    pub ny: usize,
}

impl ImageGrid {
    /// Embedding rows the image occupies.
    pub fn n_tokens(&self) -> usize {
        self.nx * self.ny
    }

    /// Rope positions the image occupies: `max(nx, ny)`.
    pub fn n_pos(&self) -> usize {
        self.nx.max(self.ny)
    }
}

/// One run of rows in the expanded sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    /// Text ids `ids[start..end]` (placeholders excluded).
    Text { start: usize, end: usize },
    /// Every row of `images[index]`.
    Image { index: usize },
}

/// Expanded row layout plus the `[4, seq_len]` positions of every row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptLayout {
    /// Runs in sequence order.
    pub segments: Vec<Segment>,
    /// `[4, seq_len]` i32, row-major: all `t`, then `h`, then `w`, then `e`.
    pub positions: Vec<i32>,
    /// Rows in the expanded sequence.
    pub seq_len: usize,
    /// Rope position of the token that follows the prompt.
    pub next_rope_pos: usize,
}

impl PromptLayout {
    /// One stream of `positions`: `0 = t`, `1 = h`, `2 = w`, `3 = e`.
    pub fn stream(&self, index: usize) -> &[i32] {
        &self.positions[index * self.seq_len..(index + 1) * self.seq_len]
    }
}

/// Lay out `ids` with one placeholder per entry of `grids`, starting at
/// rope position `rope_pos`.
///
/// # Errors
///
/// [`Error::ModelError`] when the placeholder count differs from
/// `grids.len()`, a placeholder is not bracketed by `vision_start` and
/// `vision_end`, or a grid has a zero side.
pub fn plan_positions_only(
    ids: &[u32],
    grids: &[ImageGrid],
    markers: &VisionMarkers,
    rope_pos: usize,
) -> Result<PromptLayout> {
    let placeholders = ids.iter().filter(|&&id| id == markers.image_pad).count();
    if placeholders != grids.len() {
        return Err(Error::ModelError {
            reason: format!(
                "qwen35 prompt: {placeholders} image_pad placeholders in {} ids, {} images given",
                ids.len(),
                grids.len()
            ),
        });
    }
    for (index, grid) in grids.iter().enumerate() {
        if grid.nx == 0 || grid.ny == 0 {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35 prompt: image {index} has grid {}x{}, want both sides > 0",
                    grid.nx, grid.ny
                ),
            });
        }
    }

    let n_image_rows: usize = grids.iter().map(ImageGrid::n_tokens).sum();
    let seq_len = ids.len() - placeholders + n_image_rows;
    let mut t = Vec::with_capacity(seq_len);
    let mut h = Vec::with_capacity(seq_len);
    let mut w = Vec::with_capacity(seq_len);
    let mut segments = Vec::new();

    let mut pos = rope_pos;
    let mut image_index = 0usize;
    let mut text_start = 0usize;
    for (i, &id) in ids.iter().enumerate() {
        if id != markers.image_pad {
            t.push(pos as i32);
            h.push(pos as i32);
            w.push(pos as i32);
            pos += 1;
            continue;
        }
        let before = i.checked_sub(1).map(|j| ids[j]);
        let after = ids.get(i + 1).copied();
        if before != Some(markers.vision_start) || after != Some(markers.vision_end) {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35 prompt: image_pad at index {i} is bracketed by {before:?} and \
                     {after:?}, want vision_start {} and vision_end {}",
                    markers.vision_start, markers.vision_end
                ),
            });
        }
        if text_start < i {
            segments.push(Segment::Text {
                start: text_start,
                end: i,
            });
        }
        text_start = i + 1;
        let grid = grids[image_index];
        let p0 = pos;
        for r in 0..grid.n_tokens() {
            t.push(p0 as i32);
            h.push((p0 + r / grid.nx) as i32);
            w.push((p0 + r % grid.nx) as i32);
        }
        pos = p0 + grid.n_pos();
        segments.push(Segment::Image { index: image_index });
        image_index += 1;
    }
    if text_start < ids.len() {
        segments.push(Segment::Text {
            start: text_start,
            end: ids.len(),
        });
    }

    let mut positions = Vec::with_capacity(4 * seq_len);
    positions.extend_from_slice(&t);
    positions.extend_from_slice(&h);
    positions.extend_from_slice(&w);
    positions.resize(4 * seq_len, 0);
    Ok(PromptLayout {
        segments,
        positions,
        seq_len,
        next_rope_pos: pos,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: VisionMarkers = VisionMarkers {
        vision_start: 900,
        vision_end: 901,
        image_pad: 902,
    };

    #[test]
    fn text_only_is_identity() {
        let ids = [5, 6, 7];
        let layout = plan_positions_only(&ids, &[], &M, 0).unwrap();
        assert_eq!(layout.seq_len, 3);
        assert_eq!(layout.next_rope_pos, 3);
        assert_eq!(layout.segments, vec![Segment::Text { start: 0, end: 3 }]);
        assert_eq!(layout.positions, vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0]);
    }

    #[test]
    fn one_image_layout() {
        let ids = [1, 900, 902, 901, 2];
        let grid = ImageGrid { nx: 3, ny: 2 };
        let layout = plan_positions_only(&ids, &[grid], &M, 10).unwrap();
        assert_eq!(layout.seq_len, 4 + 6);
        assert_eq!(
            layout.segments,
            vec![
                Segment::Text { start: 0, end: 2 },
                Segment::Image { index: 0 },
                Segment::Text { start: 3, end: 5 },
            ]
        );
        // text 10, 11; image run at 12; vision_end at 12 + 3; text at 16.
        assert_eq!(layout.stream(0), &[10, 11, 12, 12, 12, 12, 12, 12, 15, 16]);
        assert_eq!(layout.stream(1), &[10, 11, 12, 12, 12, 13, 13, 13, 15, 16]);
        assert_eq!(layout.stream(2), &[10, 11, 12, 13, 14, 12, 13, 14, 15, 16]);
        assert_eq!(layout.stream(3), &[0; 10]);
        assert_eq!(layout.next_rope_pos, 17);
    }

    #[test]
    fn count_mismatch_is_an_error() {
        let ids = [1, 900, 902, 901, 2];
        let err = plan_positions_only(&ids, &[], &M, 0).unwrap_err();
        assert!(err.to_string().contains("1 image_pad"), "{err}");
        assert!(err.to_string().contains("0 images"), "{err}");
    }

    #[test]
    fn unbracketed_placeholder_is_an_error() {
        let ids = [1, 902, 2];
        let grid = ImageGrid { nx: 1, ny: 1 };
        assert!(plan_positions_only(&ids, &[grid], &M, 0).is_err());
    }

    #[test]
    fn zero_grid_is_an_error() {
        let ids = [900, 902, 901];
        let grid = ImageGrid { nx: 0, ny: 2 };
        assert!(plan_positions_only(&ids, &[grid], &M, 0).is_err());
    }
}
