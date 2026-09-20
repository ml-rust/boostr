//! Output-row regrouping of a linear layer: rows ordered `[groups, parts,
//! per]` become `[parts, groups, per]`, so a projection whose output
//! interleaves several parts per group (a per-head `[query | gate]`
//! layout) turns into one contiguous block per part, splittable by a
//! `narrow` at a block boundary instead of a strided copy.
//!
//! The permutation touches output rows only: an input-side Hadamard
//! rotation, a quantized weight's activation contract and the bias all
//! carry over. Dense and block-quantized weights regroup; an AWQ/GPTQ
//! packed weight interleaves output columns inside its packing and is
//! refused.

use super::dense::Linear;
use super::maybe_quant_linear::MaybeQuantLinear;
use super::maybe_rotated::MaybeRotatedLinear;
use super::quant_linear::QuantLinear;
use super::rotated_linear::RotatedLinear;
use crate::error::{Error, Result};
use crate::quant::regroup_rows::regroup_stride;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Reorder the leading axis of `t` from `[groups, parts, per]` to
/// `[parts, groups, per]`; trailing axes are untouched.
///
/// # Errors
///
/// `QuantError` when the leading axis is not a multiple of
/// `groups * parts`; numr errors from the copy propagate.
pub fn regroup_leading_axis<R: Runtime>(
    t: &Tensor<R>,
    groups: usize,
    parts: usize,
) -> Result<Tensor<R>> {
    let shape = t.shape().to_vec();
    let rows = shape.first().copied().ok_or_else(|| Error::QuantError {
        reason: "regroup_rows: a scalar has no rows".to_string(),
    })?;
    let per = regroup_stride(rows, groups, parts)?;
    let trailing: usize = shape[1..].iter().product();
    let dense = if t.is_contiguous() {
        t.clone()
    } else {
        t.contiguous()?
    };
    let cells = dense.reshape(&[groups, parts, per * trailing])?;
    let regrouped = cells.permute(&[1, 0, 2])?.contiguous()?;
    Ok(regrouped.reshape(&shape)?)
}

impl<R: Runtime> Linear<R> {
    /// Output rows regrouped from `[groups, parts, per]` to
    /// `[parts, groups, per]`, bias included. The result carries fresh
    /// parameter ids and keeps the trainable flag.
    ///
    /// # Errors
    ///
    /// As [`regroup_leading_axis`].
    pub fn regroup_rows(&self, groups: usize, parts: usize) -> Result<Self> {
        let weight = regroup_leading_axis(self.weight().tensor(), groups, parts)?;
        let bias = match self.bias() {
            Some(b) => Some(regroup_leading_axis(b.tensor(), groups, parts)?),
            None => None,
        };
        Ok(Self::new(weight, bias, self.weight().requires_grad()))
    }
}

impl<R: Runtime<DType = DType>> QuantLinear<R> {
    /// Output rows regrouped from `[groups, parts, per]` to
    /// `[parts, groups, per]`, bias included.
    ///
    /// # Errors
    ///
    /// As [`crate::quant::QuantTensor::regroup_rows`].
    pub fn regroup_rows(&self, groups: usize, parts: usize) -> Result<Self> {
        let weight = self.weight().regroup_rows(groups, parts)?;
        let bias = match self.bias() {
            Some(b) => Some(regroup_leading_axis(b, groups, parts)?),
            None => None,
        };
        Ok(Self::new(weight, bias))
    }
}

impl<R: Runtime<DType = DType>> MaybeQuantLinear<R> {
    /// Output rows regrouped from `[groups, parts, per]` to
    /// `[parts, groups, per]`.
    ///
    /// # Errors
    ///
    /// `ModelError` for a `DecomposedQuant` weight, whose packing
    /// interleaves output columns; otherwise as the variant's own
    /// `regroup_rows`.
    pub fn regroup_rows(&self, groups: usize, parts: usize) -> Result<Self> {
        match self {
            Self::Standard(linear) => Ok(Self::Standard(linear.regroup_rows(groups, parts)?)),
            Self::Quantized(qlinear) => Ok(Self::Quantized(qlinear.regroup_rows(groups, parts)?)),
            Self::DecomposedQuant(dq) => Err(Error::ModelError {
                reason: format!(
                    "regroup_rows: an AWQ/GPTQ packed weight of shape {:?} cannot have its \
                     output rows reordered",
                    dq.weight().shape()
                ),
            }),
        }
    }
}

impl<R: Runtime<DType = DType>> MaybeRotatedLinear<R> {
    /// Output rows regrouped from `[groups, parts, per]` to
    /// `[parts, groups, per]`; a Hadamard rotation stays attached, as it
    /// acts on the input side.
    ///
    /// # Errors
    ///
    /// As [`MaybeQuantLinear::regroup_rows`].
    pub fn regroup_rows(&self, groups: usize, parts: usize) -> Result<Self> {
        match self {
            Self::Plain(base) => Ok(Self::Plain(base.regroup_rows(groups, parts)?)),
            Self::Rotated(rotated) => {
                let base = rotated.base().regroup_rows(groups, parts)?;
                Ok(Self::Rotated(Box::new(RotatedLinear::new(
                    base,
                    rotated.rotation().clone(),
                )?)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::QuantFormat;
    use crate::quant::traits::{QuantMatmulOps, QuantizeOps};
    use crate::test_utils::cpu_setup;
    use numr::autograd::Var;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    const GROUPS: usize = 3;
    const PARTS: usize = 2;
    const PER: usize = 4;
    const ROWS: usize = GROUPS * PARTS * PER;
    const COLS: usize = 64;

    fn tensor(device: &CpuDevice, shape: &[usize], seed: u32) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let h = (i as u32).wrapping_mul(2_654_435_761u32).wrapping_add(seed);
                (h % 1000) as f32 / 500.0 - 1.0
            })
            .collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
    }

    /// Row `r` of the regrouped output equals row `source(r)` of the
    /// original.
    fn source(r: usize) -> usize {
        let (part, rest) = (r / (GROUPS * PER), r % (GROUPS * PER));
        let (group, i) = (rest / PER, rest % PER);
        (group * PARTS + part) * PER + i
    }

    #[test]
    fn dense_rows_move_as_whole_rows() {
        let (_, device) = cpu_setup();
        let w = tensor(&device, &[ROWS, COLS], 1);
        let out = regroup_leading_axis(&w, GROUPS, PARTS).unwrap();
        let (a, b): (Vec<f32>, Vec<f32>) = (w.to_vec(), out.to_vec());
        for r in 0..ROWS {
            assert_eq!(
                &b[r * COLS..(r + 1) * COLS],
                &a[source(r) * COLS..(source(r) + 1) * COLS]
            );
        }
    }

    #[test]
    fn quantized_forward_matches_original_with_rows_reordered() {
        let (client, device) = cpu_setup();
        let w = tensor(&device, &[ROWS, COLS], 7);
        let qt = client.quantize(&w, QuantFormat::Q8_0).unwrap();
        let original = QuantLinear::new(qt, None);
        let regrouped = original.regroup_rows(GROUPS, PARTS).unwrap();

        let x = tensor(&device, &[5, COLS], 11);
        let y0: Vec<f32> = client.quant_matmul(&x, original.weight()).unwrap().to_vec();
        let y1: Vec<f32> = client
            .quant_matmul(&x, regrouped.weight())
            .unwrap()
            .to_vec();
        for t in 0..5 {
            for r in 0..ROWS {
                assert_eq!(
                    y1[t * ROWS + r],
                    y0[t * ROWS + source(r)],
                    "token {t} row {r}"
                );
            }
        }
    }

    #[test]
    fn dense_layer_forward_matches_with_bias() {
        let (client, device) = cpu_setup();
        let w = tensor(&device, &[ROWS, COLS], 3);
        let b = tensor(&device, &[ROWS], 5);
        let linear = Linear::new(w, Some(b), false);
        let regrouped = linear.regroup_rows(GROUPS, PARTS).unwrap();
        let x = Var::new(tensor(&device, &[2, COLS], 9), false);
        let y0: Vec<f32> = linear.forward(&client, &x).unwrap().tensor().to_vec();
        let y1: Vec<f32> = regrouped.forward(&client, &x).unwrap().tensor().to_vec();
        for t in 0..2 {
            for r in 0..ROWS {
                let (got, want) = (y1[t * ROWS + r], y0[t * ROWS + source(r)]);
                assert!(
                    (got - want).abs() < 1e-4,
                    "token {t} row {r}: {got} vs {want}"
                );
            }
        }
    }
}
