//! Row regrouping of a 2-D [`QuantTensor`]: rows ordered `[groups, parts,
//! per]` become `[parts, groups, per]`, a whole-row byte permutation with no
//! dequantization.
//!
//! A block format packs along the column axis, so a row is one contiguous
//! byte run and reordering rows is a strided copy of the packed bytes. The
//! copy runs through `Runtime::copy_strided` and needs no client, so a
//! module constructor can call it on any runtime.

use crate::error::{Error, Result};
use crate::quant::tensor::QuantTensor;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = numr::dtype::DType>> QuantTensor<R> {
    /// Reorder the rows of a `[groups * parts * per, cols]` tensor from
    /// `[groups, parts, per]` to `[parts, groups, per]`.
    ///
    /// The activation contract carries over: it describes the input side,
    /// which a row permutation leaves as is.
    ///
    /// # Errors
    ///
    /// `QuantError` when the tensor is not 2-D or its row count is not a
    /// multiple of `groups * parts`, or when either is zero.
    pub fn regroup_rows(&self, groups: usize, parts: usize) -> Result<Self> {
        let shape = self.shape();
        let [rows, cols] = shape else {
            return Err(Error::QuantError {
                reason: format!("regroup_rows requires a 2-D QuantTensor, got shape {shape:?}"),
            });
        };
        let (rows, cols) = (*rows, *cols);
        let per = regroup_stride(rows, groups, parts)?;
        let row_bytes = self.format().storage_bytes(cols)?;

        let bytes = Tensor::<R>::from_storage_contiguous(
            self.storage().clone(),
            &[groups, parts, per * row_bytes],
        );
        let regrouped = bytes.permute(&[1, 0, 2])?.contiguous()?;
        let out = Self::from_storage(
            regrouped.storage().clone(),
            self.format(),
            &[rows, cols],
            self.device(),
        )?;
        Ok(match self.contract.clone() {
            Some(contract) => out.with_activation_contract(contract),
            None => out,
        })
    }
}

/// Rows per `(group, part)` cell of a `[groups, parts, per]` row order.
///
/// # Errors
///
/// `QuantError` when `groups` or `parts` is zero or `rows` is not a
/// multiple of `groups * parts`.
pub fn regroup_stride(rows: usize, groups: usize, parts: usize) -> Result<usize> {
    let cell = groups.checked_mul(parts).filter(|c| *c > 0);
    match cell {
        Some(cell) if rows.is_multiple_of(cell) => Ok(rows / cell),
        _ => Err(Error::QuantError {
            reason: format!(
                "regroup_rows: {rows} rows do not split into groups={groups} x parts={parts}"
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::QuantFormat;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    const COLS: usize = 32;

    /// Q8_0 rows whose bytes spell their row index, so a byte readback shows
    /// the row order directly.
    fn tagged(device: &CpuDevice, rows: usize) -> QuantTensor<CpuRuntime> {
        let row_bytes = QuantFormat::Q8_0.storage_bytes(COLS).unwrap();
        let data: Vec<u8> = (0..rows)
            .flat_map(|r| std::iter::repeat_n(r as u8, row_bytes))
            .collect();
        QuantTensor::<CpuRuntime>::from_bytes(&data, QuantFormat::Q8_0, &[rows, COLS], device)
            .unwrap()
    }

    #[test]
    fn interleaved_rows_become_blocked() {
        let device = CpuDevice::new();
        // 3 groups x 2 parts x 2 rows: [g0p0 g0p0 g0p1 g0p1 | g1.. | g2..]
        let qt = tagged(&device, 12);
        let out = qt.regroup_rows(3, 2).unwrap();
        assert_eq!(out.shape(), &[12, COLS]);
        let row_bytes = QuantFormat::Q8_0.storage_bytes(COLS).unwrap();
        let bytes = out.to_bytes().unwrap();
        let order: Vec<u8> = (0..12).map(|r| bytes[r * row_bytes]).collect();
        assert_eq!(order, [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]);
        for r in 0..12 {
            let row = &bytes[r * row_bytes..(r + 1) * row_bytes];
            assert!(row.iter().all(|b| *b == row[0]), "row {r} is torn");
        }
    }

    #[test]
    fn regroup_twice_with_swapped_arguments_is_identity() {
        let device = CpuDevice::new();
        let qt = tagged(&device, 24);
        let back = qt.regroup_rows(4, 3).unwrap().regroup_rows(3, 4).unwrap();
        assert_eq!(back.to_bytes().unwrap(), qt.to_bytes().unwrap());
    }

    #[test]
    fn rejects_bad_splits() {
        let device = CpuDevice::new();
        let qt = tagged(&device, 10);
        assert!(qt.regroup_rows(3, 2).is_err());
        assert!(qt.regroup_rows(0, 2).is_err());
        assert!(regroup_stride(12, 3, 2).is_ok_and(|per| per == 2));
    }
}
