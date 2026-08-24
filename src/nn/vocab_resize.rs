//! Row-count growth for 2D weight matrices (embedding / lm_head tables).
//!
//! Architecture-agnostic math only. Policy — when growth is allowed, and
//! which weight it applies to — lives in `crate::model::vocab_growth`.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{ReduceOps, ShapeOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Grow `weight`'s row count to `target_rows`, mean-initializing new rows.
///
/// Mean-init, not random: a freshly initialized checkpoint row sits outside
/// the distribution the pretrained rows were trained under. At low precision
/// (BF16/FP16) that mismatch produces outsized gradients on the very first
/// step and can NaN the run before the new rows ever see real data. The
/// column-wise mean of the existing rows is already inside that
/// distribution, so training starts from a stable point.
///
/// `weight` must be rank 2 `[rows, dim]`. `target_rows < rows` is rejected —
/// shrinking silently drops information and is never the right call for a
/// checkpoint load path. `target_rows == rows` is a no-op clone.
pub fn resize_rows_mean_init<R, C>(
    client: &C,
    weight: &Tensor<R>,
    target_rows: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ReduceOps<R> + ShapeOps<R>,
{
    let shape = weight.shape();
    if shape.len() != 2 {
        return Err(Error::ModelError {
            reason: format!(
                "resize_rows_mean_init: weight must be rank 2 [rows, dim], got shape {shape:?}"
            ),
        });
    }
    let rows = shape[0];

    if target_rows < rows {
        return Err(Error::ModelError {
            reason: format!(
                "resize_rows_mean_init: cannot shrink rows from {rows} to {target_rows}"
            ),
        });
    }
    if target_rows == rows {
        return Ok(weight.clone());
    }

    let mean_row = client.mean(weight, &[0], true).map_err(Error::Numr)?;
    let new_rows = client
        .repeat(&mean_row, &[target_rows - rows, 1])
        .map_err(Error::Numr)?;
    client.cat(&[weight, &new_rows], 0).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn grown_rows_equal_column_mean_and_originals_are_untouched() {
        let (client, device) = cpu_setup();
        #[rustfmt::skip]
        let weight = Tensor::<CpuRuntime>::try_from_slice(
            &[
                1.0f32, 2.0, 3.0,
                3.0, 4.0, 5.0,
            ],
            &[2, 3],
            &device,
        ).unwrap();

        let grown = resize_rows_mean_init(&client, &weight, 4).unwrap();
        assert_eq!(grown.shape(), &[4, 3]);

        let data: Vec<f32> = grown.contiguous().unwrap().to_vec();
        // Original rows untouched.
        assert_eq!(&data[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&data[3..6], &[3.0, 4.0, 5.0]);
        // New rows equal the column-wise mean, exactly.
        assert_eq!(&data[6..9], &[2.0, 3.0, 4.0]);
        assert_eq!(&data[9..12], &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn rejects_non_rank_2_input() {
        let (client, device) = cpu_setup();
        let weight =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2, 1], &device)
                .unwrap();
        assert!(resize_rows_mean_init(&client, &weight, 4).is_err());
    }

    #[test]
    fn rejects_shrinking() {
        let (client, device) = cpu_setup();
        let weight =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device)
                .unwrap();
        assert!(resize_rows_mean_init(&client, &weight, 1).is_err());
    }

    #[test]
    fn equal_target_is_a_no_op() {
        let (client, device) = cpu_setup();
        let weight =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device)
                .unwrap();
        let out = resize_rows_mean_init(&client, &weight, 2).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        let data: Vec<f32> = out.contiguous().unwrap().to_vec();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
