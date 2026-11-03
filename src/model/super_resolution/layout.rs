//! [`SrExecution`]: the dtype, memory layout and accumulator a loaded
//! super-resolution network runs with, and [`weight_for_layout`], the
//! OIHW → OHWI weight permute that runs once at load.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{Conv2dConfig, ConvAccumulation, ConvLayout, PaddingMode};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// How a loaded network runs. The load fixes every field.
///
/// # Layout
///
/// - [`ConvLayout::Nhwc`] is the CUDA F16 and BF16 path. The native CUDA
///   kernel reads NHWC input with OHWI weights. The load permutes each
///   weight once, so no forward call relayouts a weight.
/// - [`ConvLayout::Nchw`] is the CPU and F32 path. The weights stay in
///   PyTorch's OIHW order, and a frame in PyTorch's `[N, C, H, W]` order
///   needs no permute copy.
/// - Both layouts compute the same values within float tolerance. Only the
///   dimension order of input, output and weights differs.
///
/// Every forward input and output follows `layout`: `[N, C, H, W]` for
/// `Nchw`, `[N, H, W, C]` for `Nhwc`.
///
/// # Dtype
///
/// The load casts every weight to `dtype`. Every forward input must have
/// `dtype`. F16 and BF16 on CPU need the `f16` feature.
///
/// # Accumulation
///
/// `accumulation` passes through to every conv. See [`ConvAccumulation`]:
/// `AllowHalf` permits an F16 K sum on the native CUDA path only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SrExecution {
    /// Dtype of weights, input and output.
    pub dtype: DType,
    /// Dimension order of weights, input and output.
    pub layout: ConvLayout,
    /// Accumulator precision of every conv's K sum.
    pub accumulation: ConvAccumulation,
}

impl SrExecution {
    /// `dtype` in `layout`, with [`ConvAccumulation::F32`].
    pub fn new(dtype: DType, layout: ConvLayout) -> Self {
        Self {
            dtype,
            layout,
            accumulation: ConvAccumulation::F32,
        }
    }

    /// Set the accumulator precision.
    pub fn with_accumulation(mut self, accumulation: ConvAccumulation) -> Self {
        self.accumulation = accumulation;
        self
    }

    /// The fused-conv config for a stride-1 conv with `padding`.
    pub fn conv_config(&self, padding: PaddingMode) -> Conv2dConfig {
        Conv2dConfig::new(self.layout)
            .with_padding(padding)
            .with_accumulation(self.accumulation)
    }

    /// Index of the channel dimension in a 4-D image tensor.
    pub fn channel_dim(&self) -> usize {
        match self.layout {
            ConvLayout::Nchw => 1,
            ConvLayout::Nhwc => 3,
        }
    }

    /// `(n, c, h, w)` of a 4-D image tensor in this layout.
    pub fn image_dims(&self, shape: &[usize]) -> Result<(usize, usize, usize, usize)> {
        match (self.layout, shape) {
            (ConvLayout::Nchw, &[n, c, h, w]) | (ConvLayout::Nhwc, &[n, h, w, c]) => {
                Ok((n, c, h, w))
            }
            _ => Err(Error::InvalidArgument {
                arg: "input",
                reason: format!(
                    "expected a 4-D {} image tensor, got shape {shape:?}; pass {}",
                    layout_name(self.layout),
                    layout_order(self.layout)
                ),
            }),
        }
    }

    /// The shape of an `n × c × h × w` image in this layout.
    pub fn image_shape(&self, n: usize, c: usize, h: usize, w: usize) -> [usize; 4] {
        match self.layout {
            ConvLayout::Nchw => [n, c, h, w],
            ConvLayout::Nhwc => [n, h, w, c],
        }
    }
}

/// `weight`, an OIHW `[C_out, C_in, Kh, Kw]` tensor as PyTorch stores it,
/// in the order `layout` reads, as a contiguous tensor.
///
/// `Nchw` keeps OIHW. `Nhwc` permutes to OHWI `[C_out, Kh, Kw, C_in]` and
/// copies once.
pub fn weight_for_layout<R: Runtime>(weight: &Tensor<R>, layout: ConvLayout) -> Result<Tensor<R>> {
    if weight.shape().len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "weight",
            reason: format!(
                "expected a 4-D OIHW conv weight, got shape {:?}; pass the weight as PyTorch stores it",
                weight.shape()
            ),
        });
    }
    let ordered = match layout {
        ConvLayout::Nchw => weight.clone(),
        ConvLayout::Nhwc => weight.permute(&[0, 2, 3, 1])?,
    };
    Ok(ordered.contiguous()?)
}

fn layout_name(layout: ConvLayout) -> &'static str {
    match layout {
        ConvLayout::Nchw => "NCHW",
        ConvLayout::Nhwc => "NHWC",
    }
}

fn layout_order(layout: ConvLayout) -> &'static str {
    match layout {
        ConvLayout::Nchw => "[N, C, H, W]",
        ConvLayout::Nhwc => "[N, H, W, C]",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn nhwc_weight_is_ohwi() -> Result<()> {
        let device = CpuDevice::new();
        // [O=2, I=3, Kh=1, Kw=2], value = 100*o + 10*i + kw.
        let data: Vec<f32> = (0..2)
            .flat_map(|o| {
                (0..3).flat_map(move |i| (0..2).map(move |k| (100 * o + 10 * i + k) as f32))
            })
            .collect();
        let w = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3, 1, 2], &device)?;
        let ohwi = weight_for_layout(&w, ConvLayout::Nhwc)?;
        assert_eq!(ohwi.shape(), &[2, 1, 2, 3]);
        // out[o, 0, kw, i] = in[o, i, 0, kw].
        let got = ohwi.to_vec::<f32>();
        assert_eq!(&got[..6], &[0.0, 10.0, 20.0, 1.0, 11.0, 21.0]);
        Ok(())
    }

    #[test]
    fn nchw_weight_is_unchanged() -> Result<()> {
        let device = CpuDevice::new();
        let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let w = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3, 1, 2], &device)?;
        assert_eq!(
            weight_for_layout(&w, ConvLayout::Nchw)?.to_vec::<f32>(),
            data
        );
        Ok(())
    }

    #[test]
    fn rejects_non_4d_weight() -> Result<()> {
        let device = CpuDevice::new();
        let w = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device)?;
        let err = weight_for_layout(&w, ConvLayout::Nhwc)
            .err()
            .map(|e| e.to_string());
        assert!(err.as_deref().is_some_and(|e| e.contains("[2]")), "{err:?}");
        Ok(())
    }

    #[test]
    fn image_dims_follow_layout() {
        let nchw = SrExecution::new(DType::F32, ConvLayout::Nchw);
        let nhwc = SrExecution::new(DType::F32, ConvLayout::Nhwc);
        assert_eq!(nchw.image_dims(&[1, 3, 5, 7]).ok(), Some((1, 3, 5, 7)));
        assert_eq!(nhwc.image_dims(&[1, 5, 7, 3]).ok(), Some((1, 3, 5, 7)));
        assert_eq!(nhwc.image_shape(1, 3, 5, 7), [1, 5, 7, 3]);
        assert!(nchw.image_dims(&[3, 5, 7]).is_err());
    }
}
