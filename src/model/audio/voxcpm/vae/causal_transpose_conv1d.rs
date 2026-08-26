//! Causal (tail-trimmed) 1D transposed convolution: the upsampling stage of
//! each VoxCPM2 `DecoderBlock`.
//!
//! Upstream builds this as `ConvTranspose1d(kernel=2*stride,
//! padding=ceil(stride/2), output_padding=stride%2)`, then its `CausalConv1d`
//! wrapper runs the underlying conv with `padding=0, output_padding=0` and
//! trims `trim = padding*2 - output_padding` samples off the TAIL only
//! (transposed-conv padding shortens the output, so "removing padding" means
//! cropping the end, the mirror image of the encoder-side left-pad).
//!
//! `trim = 2*ceil(stride/2) - (stride % 2)` is computed once at construction
//! from `stride` alone (kernel size is always `2*stride` in this decoder) and
//! matches the spec table exactly: stride 8/6/5/2 -> trim 8/6/5/2.
//!
//! # TRAP
//!
//! `ConvTranspose1d.weight` layout is `[in_channels, out_channels/groups,
//! kernel]` — swapped from `Conv1d`'s `[out, in/groups, kernel]`. Getting
//! this backwards silently transposes channels at every upsample stage.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::client::VoxCpmClient;
use numr::dtype::DType;
use numr::ops::PaddingMode;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Causal ConvTranspose1d: `x [B, C_in, T] -> [B, C_out, T * stride]`.
pub struct CausalTransposeConv1d<R: Runtime> {
    weight: Tensor<R>,
    bias: Option<Tensor<R>>,
    stride: usize,
    trim: usize,
}

impl<R: Runtime<DType = DType>> CausalTransposeConv1d<R> {
    /// `weight`: `[in_channels, out_channels, kernel_size]` (`groups=1`
    /// throughout this decoder). `kernel_size` must equal `2 * stride`.
    pub fn new(weight: Tensor<R>, bias: Option<Tensor<R>>, stride: usize) -> Result<Self> {
        if stride == 0 {
            return Err(Error::InvalidArgument {
                arg: "stride",
                reason: "must be > 0".into(),
            });
        }
        let shape = weight.shape();
        if shape.len() != 3 || shape[2] != 2 * stride {
            return Err(Error::InvalidArgument {
                arg: "weight",
                reason: format!(
                    "expected [in, out, {}] (kernel = 2*stride), got {:?}",
                    2 * stride,
                    shape
                ),
            });
        }
        let padding = stride.div_ceil(2);
        let output_padding = stride % 2;
        let trim = padding * 2 - output_padding;
        Ok(Self {
            weight,
            bias,
            stride,
            trim,
        })
    }

    /// `x [B, C_in, T] -> [B, C_out, T * stride]`.
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
    {
        let raw = client
            .conv_transpose1d(
                x,
                &self.weight,
                self.bias.as_ref(),
                self.stride,
                PaddingMode::Valid,
                0,
                1,
                1,
            )
            .map_err(Error::Numr)?;

        let total = raw.shape()[2];
        if total <= self.trim {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "raw upsampled length {total} is too short to trim {} from the tail",
                    self.trim
                ),
            });
        }
        let keep = total - self.trim;
        raw.narrow(2, 0, keep)
            .map_err(Error::Numr)?
            .contiguous()
            .map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn output_length_is_stride_times_input() {
        let (client, device) = cpu_setup();
        let c_in = 2;
        let c_out = 3;
        let stride = 8;
        let k = 2 * stride;
        let weight = Tensor::<CpuRuntime>::from_slice(
            &vec![0.01f32; c_in * c_out * k],
            &[c_in, c_out, k],
            &device,
        )
        .unwrap();
        let conv = CausalTransposeConv1d::new(weight, None, stride).unwrap();

        let t = 5;
        let x = Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; c_in * t], &[1, c_in, t], &device)
            .unwrap();
        let out = conv.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, c_out, t * stride]);
        for v in out.contiguous().unwrap().to_vec::<f32>() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn trim_matches_spec_table() {
        let (_client, device) = cpu_setup();
        for (stride, expected_trim) in [(8usize, 8usize), (6, 6), (5, 5), (2, 2)] {
            let k = 2 * stride;
            let weight =
                Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; k], &[1, 1, k], &device).unwrap();
            let conv = CausalTransposeConv1d::new(weight, None, stride).unwrap();
            assert_eq!(conv.trim, expected_trim, "stride {stride}");
        }
    }

    #[test]
    fn rejects_wrong_kernel_size() {
        let (_client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 6], &[1, 1, 6], &device).unwrap();
        assert!(CausalTransposeConv1d::new(weight, None, 8).is_err());
    }
}
