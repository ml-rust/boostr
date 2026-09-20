//! Splice text embeddings and vision-tower rows into one
//! [`Qwen35Model::forward_qwen35_embeds`] input.

use super::layout::{ImageGrid, PromptLayout, Segment, VisionMarkers, plan_positions_only};
use crate::error::{Error, Result};
use crate::model::qwen35::Qwen35Model;
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One encoded image: the tower's `[nx * ny, hidden]` rows and its grid.
pub struct ImageEmbeds<R: Runtime> {
    /// `[nx * ny, hidden]`, raster order over the `(ny, nx)` grid.
    pub embeds: Tensor<R>,
    pub grid: ImageGrid,
}

/// A prompt ready for [`Qwen35Model::forward_qwen35_embeds`].
pub struct Qwen35PromptPlan<R: Runtime> {
    /// `[1, seq_len, hidden]`.
    pub embeds: Tensor<R>,
    /// `[4, seq_len]` i32 IMROPE streams.
    pub positions: Tensor<R>,
    /// The row layout the tensors were built from.
    pub layout: PromptLayout,
}

impl<R: Runtime<DType = DType>> Qwen35PromptPlan<R>
where
    R::Client: IndexingOps<R>,
{
    /// Expand `ids` (one `image_pad` per image, in order) into embeddings
    /// and positions starting at rope position `rope_pos`. Text rows come
    /// from [`Qwen35Model::embed_tokens`]; image rows are used as given.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when the placeholder and image counts differ,
    /// a placeholder is not bracketed by the markers, the positions exceed
    /// the rope table, an image's rows are not `[nx * ny, hidden]`, or its
    /// dtype differs from the text rows.
    pub fn build<C>(
        client: &C,
        model: &Qwen35Model<R>,
        ids: &[u32],
        images: &[ImageEmbeds<R>],
        markers: &VisionMarkers,
        rope_pos: usize,
    ) -> Result<Self>
    where
        C: ModelClient<R>,
    {
        let hidden = model.config().hidden_size;
        let grids: Vec<ImageGrid> = images.iter().map(|img| img.grid).collect();
        let layout = plan_positions_only(ids, &grids, markers, rope_pos)?;
        let max_pos = model.rope().cos_cache().shape()[0];
        if layout.next_rope_pos > max_pos {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35 prompt: rope positions {rope_pos}..{} exceed the rope table of {max_pos}",
                    layout.next_rope_pos
                ),
            });
        }
        for (index, img) in images.iter().enumerate() {
            let want = [img.grid.n_tokens(), hidden];
            if img.embeds.shape() != want.as_slice() {
                return Err(Error::ModelError {
                    reason: format!(
                        "qwen35 prompt: image {index} embeds are {:?}, want {want:?} for a {}x{} grid",
                        img.embeds.shape(),
                        img.grid.nx,
                        img.grid.ny
                    ),
                });
            }
        }

        let device = client.device();
        let text_ids: Vec<i64> = ids
            .iter()
            .filter(|&&id| id != markers.image_pad)
            .map(|&id| i64::from(id))
            .collect();
        let text_embeds = if text_ids.is_empty() {
            None
        } else {
            let id_tensor = Tensor::<R>::from_slice(&text_ids, &[1, text_ids.len()], device)?;
            Some(model.embed_tokens(client, &id_tensor)?)
        };

        let mut pieces = Vec::with_capacity(layout.segments.len());
        let mut text_off = 0usize;
        for segment in &layout.segments {
            match *segment {
                Segment::Text { start, end } => {
                    let len = end - start;
                    let all = text_embeds.as_ref().ok_or_else(|| Error::ModelError {
                        reason: "qwen35 prompt: text segment without text rows".into(),
                    })?;
                    pieces.push(all.narrow(1, text_off, len)?.contiguous()?);
                    text_off += len;
                }
                Segment::Image { index } => {
                    let img = &images[index];
                    if let Some(text) = &text_embeds
                        && img.embeds.dtype() != text.dtype()
                    {
                        return Err(Error::ModelError {
                            reason: format!(
                                "qwen35 prompt: image {index} embeds are {:?}, text rows are {:?}",
                                img.embeds.dtype(),
                                text.dtype()
                            ),
                        });
                    }
                    let n = img.grid.n_tokens();
                    pieces.push(img.embeds.contiguous()?.reshape(&[1, n, hidden])?);
                }
            }
        }
        let embeds = match pieces.as_slice() {
            [] => {
                return Err(Error::ModelError {
                    reason: "qwen35 prompt: no ids and no images".into(),
                });
            }
            [one] => one.clone(),
            many => {
                let refs: Vec<&Tensor<R>> = many.iter().collect();
                client.cat(&refs, 1)?
            }
        };
        let positions = Tensor::<R>::from_slice(&layout.positions, &[4, layout.seq_len], device)?;
        Ok(Self {
            embeds,
            positions,
            layout,
        })
    }

    /// Rope position of the token that follows the prompt.
    pub fn next_rope_pos(&self) -> usize {
        self.layout.next_rope_pos
    }

    /// Rows in the expanded sequence.
    pub fn seq_len(&self) -> usize {
        self.layout.seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen35::model::tiny_model;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    const M: VisionMarkers = VisionMarkers {
        vision_start: 3,
        vision_end: 4,
        image_pad: 5,
    };

    #[test]
    fn splices_text_and_image_rows() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device, 0x3535_0030);
        let hidden = model.config().hidden_size;
        let grid = ImageGrid { nx: 2, ny: 1 };
        let rows: Vec<f32> = (0..2 * hidden).map(|i| i as f32 + 100.0).collect();
        let image = ImageEmbeds {
            embeds: Tensor::<CpuRuntime>::from_slice(&rows, &[2, hidden], &device).unwrap(),
            grid,
        };
        let ids = [1u32, 3, 5, 4, 2];
        let plan = Qwen35PromptPlan::build(&client, &model, &ids, &[image], &M, 0).unwrap();
        assert_eq!(plan.seq_len(), 6);
        assert_eq!(plan.embeds.shape(), &[1, 6, hidden]);
        assert_eq!(plan.positions.shape(), &[4, 6]);
        assert_eq!(plan.next_rope_pos(), 6);

        let text_ids =
            Tensor::<CpuRuntime>::from_slice(&[1i64, 3, 4, 2], &[1, 4], &device).unwrap();
        let text = model
            .embed_tokens(&client, &text_ids)
            .unwrap()
            .to_vec::<f32>();
        let got = plan.embeds.to_vec::<f32>();
        assert_eq!(&got[..2 * hidden], &text[..2 * hidden]);
        assert_eq!(&got[2 * hidden..4 * hidden], &rows[..]);
        assert_eq!(&got[4 * hidden..], &text[2 * hidden..]);
        assert_eq!(
            plan.positions.to_vec::<i32>(),
            vec![
                0, 1, 2, 2, 4, 5, 0, 1, 2, 2, 4, 5, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0
            ]
        );
    }

    #[test]
    fn wrong_row_count_is_an_error() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device, 0x3535_0031);
        let hidden = model.config().hidden_size;
        let image = ImageEmbeds {
            embeds: Tensor::<CpuRuntime>::zeros(&[3, hidden], DType::F32, &device).unwrap(),
            grid: ImageGrid { nx: 2, ny: 1 },
        };
        let ids = [3u32, 5, 4];
        let err = match Qwen35PromptPlan::build(&client, &model, &ids, &[image], &M, 0) {
            Ok(_) => panic!("a 3-row image on a 2x1 grid must be rejected"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("[3, "), "{err}");
    }
}
