//! Document-boundary masks for packed sequences.
//!
//! A packed row holds several documents back to back. A causal op that looks
//! back `d` positions must not read across a document start. These masks are
//! constants in the compute dtype, built once from segment ids and shared by
//! every layer.
//!
//! # Segment ids
//!
//! `ids` is an I32 `[batch, seq]` tensor. Equal ids in a row mean the same
//! document. Ids differ across neighbouring documents. Position 0 of a row
//! always starts a segment.
//!
//! # Construction
//!
//! numr `CompareOps::eq` supports I32 on CPU, CUDA and WGPU and returns a
//! 1/0 mask in the input dtype. The masks are built on the device with `eq`
//! and `cast`. The ids never leave the device.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{CompareOps, ShapeOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Boundary masks for one batch of packed rows. Cloning copies handles only.
pub struct SegmentMasks<R: Runtime> {
    ids: Tensor<R>,
    keep: Tensor<R>,
    taps: Vec<Tensor<R>>,
    max_kernel: usize,
    dtype: DType,
}

impl<R: Runtime> Clone for SegmentMasks<R> {
    fn clone(&self) -> Self {
        Self {
            ids: self.ids.clone(),
            keep: self.keep.clone(),
            taps: self.taps.clone(),
            max_kernel: self.max_kernel,
            dtype: self.dtype,
        }
    }
}

impl<R> SegmentMasks<R>
where
    R: Runtime<DType = DType>,
{
    /// Build the masks for conv kernels up to `max_kernel` taps.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `ids` is not an I32 `[batch, seq >= 1]`
    /// tensor, `max_kernel` is 0, or `dtype` is not a float dtype.
    pub fn from_ids<C>(client: &C, ids: &Tensor<R>, max_kernel: usize, dtype: DType) -> Result<Self>
    where
        C: RuntimeClient<R> + CompareOps<R> + TypeConversionOps<R> + ShapeOps<R>,
    {
        if ids.dtype() != DType::I32 || ids.shape().len() != 2 || ids.shape()[1] == 0 {
            return Err(Error::InvalidArgument {
                arg: "ids",
                reason: format!(
                    "expected I32 [batch, seq >= 1], got {:?} {:?}",
                    ids.dtype(),
                    ids.shape()
                ),
            });
        }
        if max_kernel == 0 {
            return Err(Error::InvalidArgument {
                arg: "max_kernel",
                reason: "kernel must be at least 1".into(),
            });
        }
        if !dtype.is_float() {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("expected a float compute dtype, got {dtype:?}"),
            });
        }
        let ids = ids.contiguous().map_err(Error::Numr)?;
        let (batch, seq) = (ids.shape()[0], ids.shape()[1]);

        let keep = same_segment(client, &ids, 1, dtype)?;
        let mut taps = Vec::with_capacity(max_kernel - 1);
        for d in 1..max_kernel {
            let mask = if d == 1 {
                keep.clone()
            } else {
                same_segment(client, &ids, d, dtype)?
            };
            taps.push(mask.reshape(&[batch, seq, 1]).map_err(Error::Numr)?);
        }
        Ok(Self {
            ids,
            keep,
            taps,
            max_kernel,
            dtype,
        })
    }

    /// The I32 `[batch, seq]` segment ids these masks came from. Cloning the
    /// result copies a handle.
    pub fn segment_ids(&self) -> &Tensor<R> {
        &self.ids
    }

    /// `[batch, seq]`: 1 where position `t` continues the segment of `t - 1`,
    /// 0 at a document start. Column 0 is always 0.
    pub fn keep(&self) -> &Tensor<R> {
        &self.keep
    }

    /// `[batch, seq, 1, 1]` view of [`Self::keep`], for state-space scans.
    ///
    /// # Errors
    ///
    /// [`Error::Numr`] when the reshape fails.
    pub fn keep_4d(&self) -> Result<Tensor<R>> {
        let s = self.keep.shape();
        self.keep.reshape(&[s[0], s[1], 1, 1]).map_err(Error::Numr)
    }

    /// `[batch, seq, 1]`: 1 where position `t - d` exists and shares the
    /// segment of `t`, else 0.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `d` is outside `1..max_kernel`.
    pub fn tap(&self, d: usize) -> Result<&Tensor<R>> {
        if d == 0 || d >= self.max_kernel {
            return Err(Error::InvalidArgument {
                arg: "d",
                reason: format!("expected 1..{}, got {d}", self.max_kernel),
            });
        }
        Ok(&self.taps[d - 1])
    }

    /// The largest conv kernel these masks cover.
    pub fn max_kernel(&self) -> usize {
        self.max_kernel
    }

    /// The compute dtype of the masks.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// `(batch, seq)` of the ids these masks came from.
    pub fn dims(&self) -> (usize, usize) {
        (self.keep.shape()[0], self.keep.shape()[1])
    }
}

/// `[batch, seq]` mask: 1 where `ids[t - d] == ids[t]` and `t >= d`, else 0.
fn same_segment<R, C>(client: &C, ids: &Tensor<R>, d: usize, dtype: DType) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + CompareOps<R> + TypeConversionOps<R> + ShapeOps<R>,
{
    let (batch, seq) = (ids.shape()[0], ids.shape()[1]);
    let lead = d.min(seq);
    let zeros = Tensor::<R>::zeros(&[batch, lead], dtype, ids.device()).map_err(Error::Numr)?;
    if lead == seq {
        return Ok(zeros);
    }
    let later = ids.narrow(1, d, seq - d).map_err(Error::Numr)?;
    let earlier = ids.narrow(1, 0, seq - d).map_err(Error::Numr)?;
    let eq = client.eq(&later, &earlier).map_err(Error::Numr)?;
    let eq = client.cast(&eq, dtype).map_err(Error::Numr)?;
    client.cat(&[&zeros, &eq], 1).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn ids(data: &[i32], shape: &[usize]) -> Tensor<CpuRuntime> {
        let (_, device) = cpu_setup();
        Tensor::<CpuRuntime>::from_slice(data, shape, &device).unwrap()
    }

    /// Row 0 holds documents of length 3, 1, 2. Row 1 is one document.
    #[test]
    fn masks_match_hand_values() {
        let (client, _) = cpu_setup();
        let t = ids(&[7, 7, 7, 2, 9, 9, 4, 4, 4, 4, 4, 4], &[2, 6]);
        for dtype in [DType::F32, DType::F64] {
            let m = SegmentMasks::from_ids(&client, &t, 4, dtype).unwrap();
            assert_eq!(m.dims(), (2, 6));
            assert_eq!(m.keep().dtype(), dtype);
            assert_eq!(m.segment_ids().dtype(), DType::I32);
            assert_eq!(m.segment_ids().shape(), &[2, 6]);
            assert_eq!(
                m.segment_ids().to_vec::<i32>(),
                vec![7, 7, 7, 2, 9, 9, 4, 4, 4, 4, 4, 4]
            );
            let host = |x: &Tensor<CpuRuntime>| crate::test_utils::to_f64(x);
            assert_eq!(
                host(m.keep()),
                vec![0., 1., 1., 0., 0., 1., 0., 1., 1., 1., 1., 1.]
            );
            assert_eq!(m.keep_4d().unwrap().shape(), &[2, 6, 1, 1]);
            assert_eq!(m.tap(1).unwrap().shape(), &[2, 6, 1]);
            assert_eq!(
                host(m.tap(1).unwrap()),
                vec![0., 1., 1., 0., 0., 1., 0., 1., 1., 1., 1., 1.]
            );
            assert_eq!(
                host(m.tap(2).unwrap()),
                vec![0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1.]
            );
            assert_eq!(
                host(m.tap(3).unwrap()),
                vec![0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]
            );
        }
    }

    /// A tap deeper than the row is all zeros. Kernel 1 has no taps.
    #[test]
    fn deep_tap_and_kernel_one() {
        let (client, _) = cpu_setup();
        let t = ids(&[1, 1], &[1, 2]);
        let m = SegmentMasks::from_ids(&client, &t, 4, DType::F32).unwrap();
        assert_eq!(crate::test_utils::to_f64(m.tap(3).unwrap()), vec![0., 0.]);
        let one = SegmentMasks::from_ids(&client, &t, 1, DType::F32).unwrap();
        assert!(one.tap(1).is_err());
        assert_eq!(crate::test_utils::to_f64(one.keep()), vec![0., 1.]);
    }

    #[test]
    fn rejects_bad_inputs() {
        let (client, device) = cpu_setup();
        let f = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0], &[1, 2], &device).unwrap();
        assert!(SegmentMasks::from_ids(&client, &f, 2, DType::F32).is_err());
        let t = ids(&[0, 0], &[1, 2]);
        assert!(SegmentMasks::from_ids(&client, &t, 0, DType::F32).is_err());
        assert!(SegmentMasks::from_ids(&client, &t, 2, DType::I32).is_err());
        let flat = ids(&[0, 0], &[2]);
        assert!(SegmentMasks::from_ids(&client, &flat, 2, DType::F32).is_err());
        let m = SegmentMasks::from_ids(&client, &t, 2, DType::F32).unwrap();
        assert!(m.tap(0).is_err());
        assert!(m.tap(2).is_err());
    }
}
