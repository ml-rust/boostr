//! Alignment matrix for word-timestamp DTW, matching HuggingFace
//! transformers' `WhisperGenerationMixin._extract_token_timestamps`.
//!
//! HF's reference (read from `transformers.models.whisper.generation_whisper`),
//! in the exact order HF runs it:
//!
//! ```text
//! weights = weights[..., : num_frames // 2]                     # 1. crop frames
//! weights = weights[:, :, num_input_ids:, :]                    # 2. drop prefix rows
//! if weights.shape[2] == 0: return timestamps                   #    (caller's concern)
//! std = torch.std(weights, dim=-2, keepdim=True, unbiased=False) # 3. normalize
//! mean = torch.mean(weights, dim=-2, keepdim=True)
//! weights = (weights - mean) / std
//! weights = _median_filter(weights, median_filter_width)         # 4. median filter
//! weights = weights.mean(dim=1)                                  # 5. mean over heads
//! # -> matrix, shape [T_total - num_input_ids, F]
//! ```
//!
//! Step 2 (dropping prefix rows) happens **before** step 3 (normalize): mean
//! and std are computed only over the rows DTW will actually see. Dropping
//! prefix rows from the *output* of this function instead would normalize
//! over the wrong row set and produce a different matrix — this function
//! therefore takes `skip_rows` (HF's `num_input_ids`) and drops those rows
//! itself, in the right place in the pipeline.
//!
//! `dim=-2` on a `[heads, tokens, frames]` tensor is the token axis, confirmed
//! from the source above. `_median_filter` pads `(pad_width, pad_width)` on
//! the last (frame) axis with `mode="reflect"`, then takes
//! `unfold(-1, filter_width, 1).sort()[..., pad_width]` — the true median for
//! an odd window, since `filter_width` is required odd and `pad_width` is
//! exactly the middle sorted index. When the frame axis is shorter than
//! `pad_width`, HF skips filtering entirely and returns the input unchanged;
//! this module does the same.
//!
//! ## What feeds DTW, and how timestamps map back
//!
//! This function's return value, `[T_total - skip_rows, F]`, is exactly HF's
//! `matrix` — one row per token DTW aligns (the forced prefix excluded, since
//! it always has timestamp 0.0 and skews the warp). DTW (outside this
//! function; HF's `_dynamic_time_warping`) turns it into `text_indices` and
//! `time_indices`, from which HF derives one `jump_time` per row of this
//! matrix. The full per-token timestamp array HF builds around that is
//! `[zeros(skip_rows), jump_times, jump_times[-1]]` — `skip_rows` leading
//! zeros (the prefix has no meaningful timestamp), one entry per row of this
//! matrix, and the LAST `jump_time` duplicated once more: the final generated
//! token is never fed back into the decoder (`generate_with_cross_attentions`
//! never captures a row for it — see [`CrossAttentionWeights`]'s doc comment
//! on `generate_with_cross_attentions`), so it has no row here and no
//! `jump_time` of its own, and HF reuses the previous one instead of leaving
//! it undefined. The resulting array therefore has `skip_rows + (T_total -
//! skip_rows) + 1 = T_total + 1` entries — one per captured row, plus that
//! one trailing duplicate.

use super::generate::CrossAttentionWeights;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{BinaryOps, PadMode, ReduceOps, ShapeOps, SortingOps, StatisticalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Builds the `[T_total - skip_rows, F]` DTW input matrix from captured
/// cross-attention.
///
/// * `weights` — cross-attention captured by
///   [`crate::model::audio::whisper_model::WhisperModel::generate_with_cross_attentions`].
/// * `alignment_heads` — `(decoder_layer, head)` pairs to average over.
/// * `num_frames` — the input feature (mel) frame count of the real audio.
///   Frames are cropped to `num_frames / 2`, matching HF's
///   `weights[..., : num_frames // 2]`.
/// * `median_filter_width` — odd window width for the median filter (HF's
///   `config.median_filter_width`, default 7).
/// * `skip_rows` — forced-prefix row count (HF's `num_input_ids`), dropped
///   before normalizing, matching HF's `weights[:, :, num_input_ids:, :]`.
///
/// # Errors
///
/// - `Error::InvalidArgument` if `alignment_heads` is empty, if
///   `median_filter_width` is zero or even, if `num_frames / 2` exceeds a
///   captured layer's encoder sequence length, or if `skip_rows` is `>=` a
///   captured layer's `T_total`.
/// - `Error::ModelError` if an alignment head names a decoder layer
///   `generate_with_cross_attentions` did not capture, if a head index is
///   `>=` that layer's head count, or if a captured layer's batch size is
///   not 1.
pub fn alignment_matrix<R, C>(
    client: &C,
    weights: &CrossAttentionWeights<R>,
    alignment_heads: &[(usize, usize)],
    num_frames: usize,
    median_filter_width: usize,
    skip_rows: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ShapeOps<R> + ReduceOps<R> + StatisticalOps<R> + SortingOps<R> + BinaryOps<R>,
{
    if median_filter_width == 0 || median_filter_width.is_multiple_of(2) {
        return Err(Error::InvalidArgument {
            arg: "median_filter_width",
            reason: format!("must be a positive odd number, got {median_filter_width}"),
        });
    }
    if alignment_heads.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "alignment_heads",
            reason: "must name at least one (layer, head) pair".to_string(),
        });
    }

    let frames = num_frames / 2;

    let mut planes: Vec<Tensor<R>> = Vec::with_capacity(alignment_heads.len());
    for &(layer, head) in alignment_heads {
        let layer_tensor = weights
            .layers
            .iter()
            .find(|(l, _)| *l == layer)
            .map(|(_, t)| t)
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "alignment head names decoder layer {layer}, which \
                     generate_with_cross_attentions did not capture: include \
                     {layer} in its `layers` argument"
                ),
            })?;

        let shape = layer_tensor.shape();
        let (batch, num_heads, t_total, src_len) = (shape[0], shape[1], shape[2], shape[3]);
        if batch != 1 {
            return Err(Error::ModelError {
                reason: format!(
                    "alignment_matrix requires batch=1 cross-attention weights, got batch={batch} for layer {layer}"
                ),
            });
        }
        if head >= num_heads {
            return Err(Error::ModelError {
                reason: format!(
                    "alignment head index {head} is out of range: decoder layer {layer} has {num_heads} heads"
                ),
            });
        }
        if frames > src_len {
            return Err(Error::InvalidArgument {
                arg: "num_frames",
                reason: format!(
                    "num_frames/2 ({frames}) exceeds the encoder sequence length ({src_len}) captured for layer {layer}"
                ),
            });
        }
        if skip_rows >= t_total {
            return Err(Error::InvalidArgument {
                arg: "skip_rows",
                reason: format!(
                    "skip_rows ({skip_rows}) must be less than the captured sequence length \
                     ({t_total}) for layer {layer}"
                ),
            });
        }

        // [B, H, T_total, S] -> [T_kept, F]: pick this head's plane, crop the
        // frame axis to the real-audio portion, then drop the forced-prefix
        // rows — BEFORE normalizing, matching HF's own order.
        let plane = layer_tensor
            .narrow(0, 0, 1)?
            .narrow(1, head, 1)?
            .squeeze(Some(1))
            .squeeze(Some(0))
            .narrow(1, 0, frames)?
            .narrow(0, skip_rows, t_total - skip_rows)?;
        planes.push(plane);
    }

    let refs: Vec<&Tensor<R>> = planes.iter().collect();
    let stacked = client.stack(&refs, 0)?; // [n_heads, T_kept, F]

    // Normalize over the token axis (HF's dim=-2), unbiased=False. Only the
    // kept (post-prefix-drop) rows enter this, matching HF exactly.
    let mean = client.mean(&stacked, &[1], true)?;
    let std = client.std(&stacked, &[1], true, 0)?;
    let centered = client.sub(&stacked, &mean)?;
    let normalized = client.div(&centered, &std)?;

    let filtered = median_filter(client, &normalized, median_filter_width)?;

    // Average the cross-attention heads.
    Ok(client.mean(&filtered, &[0], false)?)
}

/// Median filter along the last (frame) axis, matching HF's `_median_filter`:
/// reflect-pad `pad_width` on each side, then take the middle value of each
/// sorted sliding window. Returns the input unchanged when the frame axis is
/// not longer than `pad_width` — HF's own short-circuit.
fn median_filter<R, C>(client: &C, input: &Tensor<R>, filter_width: usize) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ShapeOps<R> + SortingOps<R>,
{
    let pad_width = filter_width / 2;
    let frame_axis_len = input.shape()[input.ndim() - 1];
    if frame_axis_len <= pad_width {
        return Ok(input.clone());
    }

    let padded = client.pad_mode(input, &[pad_width, pad_width], PadMode::Reflect)?;
    let windows = client.unfold(&padded, -1, filter_width, 1)?; // [..., F, filter_width]
    let sorted = client.sort(&windows, -1, false)?;
    let last = sorted.ndim() - 1;
    Ok(sorted
        .narrow(last as isize, pad_width, 1)?
        .squeeze(Some(last as isize)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::whisper_model::test_support::cpu_setup;

    /// Builds a `[1, H, T, S]` cross-attention tensor from per-head planes
    /// given as row-major `[T, S]` slices.
    fn layer_tensor(
        device: &numr::runtime::cpu::CpuDevice,
        heads: &[Vec<f32>],
        t: usize,
        s: usize,
    ) -> Tensor<numr::runtime::cpu::CpuRuntime> {
        let mut data = Vec::with_capacity(heads.len() * t * s);
        for head in heads {
            assert_eq!(head.len(), t * s);
            data.extend_from_slice(head);
        }
        Tensor::from_slice(&data, &[1, heads.len(), t, s], device).expect("build layer tensor")
    }

    /// The reference 4x4 matrix used by two tests below: two heads, 4 real
    /// tokens, `S = 6` cropped to `F = 4`. Computed independently from the
    /// reference formula (numpy, float64, rounded to f32 here) — see this
    /// file's header for the formula.
    #[rustfmt::skip]
    const EXPECTED_2HEAD_4X4: [f32; 16] = [
        -0.40436804, -0.40024937, -0.40436804, -0.40024937,
        -1.037_953_4, -1.010_099, -std::f32::consts::FRAC_1_SQRT_2,  0.11693604,
        -0.11028219, -0.08911538, -0.02354408,  0.6835627,
         1.5526036,  -0.31573394, -0.26726124, -0.40024937,
    ];

    #[test]
    fn matches_reference_formula_two_heads() {
        let (client, device) = cpu_setup();

        // T_total = 4, S = 6, num_frames = 8 -> crop to F = 4. No prefix.
        #[rustfmt::skip]
        let h0 = vec![
            0.1, 0.2, 0.3, 0.4, 0.9, 0.9,
            0.5, 0.1, 0.1, 0.3, 0.9, 0.9,
            0.2, 0.2, 0.5, 0.1, 0.9, 0.9,
            0.3, 0.3, 0.3, 0.1, 0.9, 0.9,
        ];
        #[rustfmt::skip]
        let h1 = vec![
            0.9, 0.05, 0.05, 0.0, 0.5, 0.5,
            0.1, 0.1, 0.7, 0.1, 0.5, 0.5,
            0.25, 0.25, 0.25, 0.25, 0.5, 0.5,
            0.0, 0.9, 0.05, 0.05, 0.5, 0.5,
        ];
        let tensor = layer_tensor(&device, &[h0, h1], 4, 6);
        let weights = CrossAttentionWeights {
            layers: vec![(0, tensor)],
        };

        let matrix = alignment_matrix(&client, &weights, &[(0, 0), (0, 1)], 8, 3, 0)
            .expect("alignment matrix");
        assert_eq!(matrix.shape(), &[4, 4]);

        let got: Vec<f32> = matrix.try_to_vec().expect("host copy for assertion");
        for (i, (g, e)) in got.iter().zip(EXPECTED_2HEAD_4X4.iter()).enumerate() {
            assert!((g - e).abs() < 1e-4, "index {i}: got {g}, expected {e}");
        }
    }

    #[test]
    fn drops_prefix_rows_before_normalizing() {
        let (client, device) = cpu_setup();

        // Same 4 "real" rows as `matches_reference_formula_two_heads`, with 2
        // prefix rows prepended whose values are wildly different from the
        // real rows. If prefix rows leaked into the mean/std computation
        // (the bug this test guards against), the result would differ
        // sharply from `EXPECTED_2HEAD_4X4`; dropping them before
        // normalizing (the fix) reproduces it exactly.
        #[rustfmt::skip]
        let h0 = vec![
            9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
           -9.0, -9.0, -9.0, -9.0, -9.0, -9.0,
            0.1, 0.2, 0.3, 0.4, 0.9, 0.9,
            0.5, 0.1, 0.1, 0.3, 0.9, 0.9,
            0.2, 0.2, 0.5, 0.1, 0.9, 0.9,
            0.3, 0.3, 0.3, 0.1, 0.9, 0.9,
        ];
        #[rustfmt::skip]
        let h1 = vec![
            9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
           -9.0, -9.0, -9.0, -9.0, -9.0, -9.0,
            0.9, 0.05, 0.05, 0.0, 0.5, 0.5,
            0.1, 0.1, 0.7, 0.1, 0.5, 0.5,
            0.25, 0.25, 0.25, 0.25, 0.5, 0.5,
            0.0, 0.9, 0.05, 0.05, 0.5, 0.5,
        ];
        let tensor = layer_tensor(&device, &[h0, h1], 6, 6);
        let weights = CrossAttentionWeights {
            layers: vec![(0, tensor)],
        };

        // T_total = 6 (2 prefix + 4 real), skip_rows = 2.
        let matrix = alignment_matrix(&client, &weights, &[(0, 0), (0, 1)], 8, 3, 2)
            .expect("alignment matrix");
        assert_eq!(matrix.shape(), &[4, 4]);

        let got: Vec<f32> = matrix.try_to_vec().expect("host copy for assertion");
        for (i, (g, e)) in got.iter().zip(EXPECTED_2HEAD_4X4.iter()).enumerate() {
            assert!((g - e).abs() < 1e-4, "index {i}: got {g}, expected {e}");
        }
    }

    #[test]
    fn skips_median_filter_when_frame_axis_not_longer_than_pad_width() {
        let (client, device) = cpu_setup();

        // T_total = 2, S = 1 (already <= pad_width = filter_width/2 = 1), one head.
        let h0 = vec![0.2, 0.8];
        let tensor = layer_tensor(&device, &[h0], 2, 1);
        let weights = CrossAttentionWeights {
            layers: vec![(0, tensor)],
        };

        // num_frames = 2 -> crop to F = 1.
        let matrix =
            alignment_matrix(&client, &weights, &[(0, 0)], 2, 3, 0).expect("alignment matrix");
        assert_eq!(matrix.shape(), &[2, 1]);

        let got: Vec<f32> = matrix.try_to_vec().expect("host copy for assertion");
        let expected = [-1.0f32, 1.0];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-4, "got {g}, expected {e}");
        }
    }

    #[test]
    fn rejects_even_median_filter_width() {
        let (client, device) = cpu_setup();
        let tensor = layer_tensor(&device, &[vec![0.1, 0.2, 0.3, 0.4]], 2, 2);
        let weights = CrossAttentionWeights {
            layers: vec![(0, tensor)],
        };

        let result = alignment_matrix(&client, &weights, &[(0, 0)], 4, 4, 0);
        assert!(result.is_err(), "an even filter width must be refused");
    }

    #[test]
    fn rejects_zero_median_filter_width() {
        let (client, device) = cpu_setup();
        let tensor = layer_tensor(&device, &[vec![0.1, 0.2, 0.3, 0.4]], 2, 2);
        let weights = CrossAttentionWeights {
            layers: vec![(0, tensor)],
        };

        let result = alignment_matrix(&client, &weights, &[(0, 0)], 4, 0, 0);
        assert!(result.is_err(), "a zero filter width must be refused");
    }

    #[test]
    fn rejects_uncaptured_layer() {
        let (client, device) = cpu_setup();
        let tensor = layer_tensor(&device, &[vec![0.1, 0.2, 0.3, 0.4]], 2, 2);
        let weights = CrossAttentionWeights {
            layers: vec![(0, tensor)],
        };

        // Layer 1 was never captured.
        let result = alignment_matrix(&client, &weights, &[(1, 0)], 4, 3, 0);
        assert!(result.is_err(), "an uncaptured layer must be refused");
    }

    #[test]
    fn rejects_out_of_range_head() {
        let (client, device) = cpu_setup();
        // Single head (H=1) layer tensor.
        let tensor = layer_tensor(&device, &[vec![0.1, 0.2, 0.3, 0.4]], 2, 2);
        let weights = CrossAttentionWeights {
            layers: vec![(0, tensor)],
        };

        let result = alignment_matrix(&client, &weights, &[(0, 1)], 4, 3, 0);
        assert!(result.is_err(), "an out-of-range head must be refused");
    }

    #[test]
    fn rejects_num_frames_exceeding_encoder_length() {
        let (client, device) = cpu_setup();
        // S = 2, so num_frames/2 must be <= 2.
        let tensor = layer_tensor(&device, &[vec![0.1, 0.2, 0.3, 0.4]], 2, 2);
        let weights = CrossAttentionWeights {
            layers: vec![(0, tensor)],
        };

        // num_frames = 10 -> crop target 5 > S=2.
        let result = alignment_matrix(&client, &weights, &[(0, 0)], 10, 3, 0);
        assert!(
            result.is_err(),
            "num_frames/2 exceeding the encoder length must be refused"
        );
    }

    #[test]
    fn rejects_skip_rows_at_or_past_t_total() {
        let (client, device) = cpu_setup();
        // T_total = 2.
        let tensor = layer_tensor(&device, &[vec![0.1, 0.2, 0.3, 0.4]], 2, 2);
        let weights = CrossAttentionWeights {
            layers: vec![(0, tensor)],
        };

        let result = alignment_matrix(&client, &weights, &[(0, 0)], 2, 3, 2);
        assert!(
            result.is_err(),
            "skip_rows == T_total must be refused, not silently empty the matrix"
        );
    }

    #[test]
    fn rejects_empty_alignment_heads() {
        let (client, device) = cpu_setup();
        let tensor = layer_tensor(&device, &[vec![0.1, 0.2, 0.3, 0.4]], 2, 2);
        let weights = CrossAttentionWeights {
            layers: vec![(0, tensor)],
        };

        let result = alignment_matrix(&client, &weights, &[], 4, 3, 0);
        assert!(result.is_err(), "empty alignment_heads must be refused");
    }
}
