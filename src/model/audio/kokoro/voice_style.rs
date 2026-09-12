//! Voice-pack tensor helpers: pick the style row for an utterance and split
//! it into the decoder and predictor halves the model consumes.
//!
//! Finding and reading the pack file is `boostr-audio`'s `VoiceResolver`;
//! these two functions only slice the loaded tensor.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Pick the style row matching a phoneme-count budget.
///
/// Kokoro voice packs are `[T, 1, 2*D]` or `[T, 2*D]`. Upstream indexes by
/// `len(phonemes) - 1`, clamped to `[0, T-1]`. Returns a `[1, 2*D]` row that
/// can then be split into decoder and predictor halves via
/// [`split_voice_style`].
pub fn select_voice_style<R: Runtime<DType = DType>>(
    voice_pack: &Tensor<R>,
    phoneme_count: usize,
) -> Result<Tensor<R>> {
    let shape = voice_pack.shape();
    let (rows, style_width) = match shape.len() {
        3 => {
            if shape[1] != 1 {
                return Err(Error::ModelError {
                    reason: format!("voice pack middle dim must be 1, got shape {shape:?}"),
                });
            }
            (shape[0], shape[2])
        }
        2 => (shape[0], shape[1]),
        _ => {
            return Err(Error::ModelError {
                reason: format!("voice pack rank must be 2 or 3, got shape {shape:?}"),
            });
        }
    };
    if rows == 0 {
        return Err(Error::ModelError {
            reason: "voice pack is empty".into(),
        });
    }
    let idx = phoneme_count.saturating_sub(1).min(rows - 1);
    let flat = match shape.len() {
        3 => voice_pack
            .reshape(&[rows, style_width])
            .map_err(|e| Error::ModelError {
                reason: format!("reshape voice pack: {e}"),
            })?,
        _ => voice_pack.clone(),
    };
    flat.narrow(0, idx, 1).map_err(|e| Error::ModelError {
        reason: format!("narrow voice pack: {e}"),
    })
}

/// Split a row-selected voice style `[1, 2*D]` into `(decoder_style [1, D],
/// predictor_style [1, D])`. Decoder half is the first `D` channels,
/// predictor half is the last `D` — matching `ref_s[:, :128]` and
/// `ref_s[:, 128:]` in the reference Kokoro Python source.
pub fn split_voice_style<R: Runtime<DType = DType>>(
    style_row: &Tensor<R>,
    style_dim: usize,
) -> Result<(Tensor<R>, Tensor<R>)> {
    let shape = style_row.shape();
    if shape.len() != 2 || shape[1] != 2 * style_dim {
        return Err(Error::ModelError {
            reason: format!(
                "style row shape must be [B, {}], got {shape:?}",
                2 * style_dim
            ),
        });
    }
    let decoder = style_row
        .narrow(1, 0, style_dim)
        .map_err(|e| Error::ModelError {
            reason: format!("narrow decoder style: {e}"),
        })?
        .contiguous()?;
    let predictor = style_row
        .narrow(1, style_dim, style_dim)
        .map_err(|e| Error::ModelError {
            reason: format!("narrow predictor style: {e}"),
        })?
        .contiguous()?;
    Ok((decoder, predictor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_voice_style_clamps_to_last_row() {
        use numr::runtime::cpu::{CpuDevice, CpuRuntime};
        let device = CpuDevice::new();
        // pack [3, 1, 4]: row 0 = 1s, row 1 = 2s, row 2 = 3s
        let data: Vec<f32> = (0..3).flat_map(|r| vec![(r + 1) as f32; 4]).collect();
        let pack = Tensor::<CpuRuntime>::from_slice(&data, &[3, 1, 4], &device).unwrap();
        let picked = select_voice_style(&pack, 2).unwrap();
        assert_eq!(picked.shape(), &[1, 4]);
        let v: Vec<f32> = picked.to_vec();
        assert_eq!(v, vec![2.0, 2.0, 2.0, 2.0]);
        // Out-of-range phoneme count clamps to the last row.
        let last = select_voice_style(&pack, 100).unwrap();
        let lv: Vec<f32> = last.to_vec();
        assert_eq!(lv, vec![3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn split_voice_style_halves_match() {
        use numr::runtime::cpu::{CpuDevice, CpuRuntime};
        let device = CpuDevice::new();
        let row = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1, 8],
            &device,
        )
        .unwrap();
        let (dec, pred) = split_voice_style(&row, 4).unwrap();
        assert_eq!(dec.shape(), &[1, 4]);
        assert_eq!(pred.shape(), &[1, 4]);
        let d: Vec<f32> = dec.to_vec();
        let p: Vec<f32> = pred.to_vec();
        assert_eq!(d, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(p, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn select_voice_style_rejects_bad_rank() {
        use numr::runtime::cpu::{CpuDevice, CpuRuntime};
        let device = CpuDevice::new();
        let bad = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device).unwrap();
        assert!(select_voice_style(&bad, 0).is_err());
    }

    #[test]
    fn split_voice_style_rejects_wrong_width() {
        use numr::runtime::cpu::{CpuDevice, CpuRuntime};
        let device = CpuDevice::new();
        let row = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[1, 6], &device).unwrap();
        // style_dim = 4 → expected [1, 8] but we have [1, 6]
        assert!(split_voice_style(&row, 4).is_err());
    }
}
