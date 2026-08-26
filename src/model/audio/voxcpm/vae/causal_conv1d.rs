//! Causal (left-only-padded) 1D convolution used throughout the VoxCPM2
//! `AudioVAE` decoder.
//!
//! Upstream's `CausalConv1d` runs a plain `Conv1d` with `padding=0` and does
//! its own manual left-pad before calling it: `left = padding * 2 -
//! output_padding`, with `output_padding` always 0 for a non-transposed conv.
//! For a stride-1 conv, the nominal "same" padding is `dilation * (kernel -
//! 1)`, so `left = dilation * (kernel - 1)` and `right = 0` — output length
//! equals input length. This is verified against the spec table (k=7 d=1 ->
//! 6, k=1 -> 0, k=7 dilation d -> 6*d).
//!
//! Implemented as `numr::ops::PaddingMode::conv1d(left, 0)` on top of
//! [`Conv1d`], so the left-only pad is folded into the same convolution call
//! rather than a separate padding op.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::client::VoxCpmClient;
use crate::nn::Conv1d;
use numr::dtype::DType;
use numr::ops::PaddingMode;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Causal Conv1d: `x [B, C_in, T] -> [B, C_out, T]` (length preserved).
pub struct CausalConv1d<R: Runtime> {
    conv: Conv1d<R>,
}

impl<R: Runtime<DType = DType>> CausalConv1d<R> {
    /// `weight`: `[out_channels, in_channels/groups, kernel_size]`.
    pub fn new(
        weight: Tensor<R>,
        bias: Option<Tensor<R>>,
        kernel_size: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        if weight.shape().len() != 3 || weight.shape()[2] != kernel_size {
            return Err(Error::InvalidArgument {
                arg: "weight",
                reason: format!("expected [_, _, {kernel_size}], got {:?}", weight.shape()),
            });
        }
        let left_pad = dilation * (kernel_size - 1);
        Ok(Self {
            conv: Conv1d::new(
                weight,
                bias,
                1,
                PaddingMode::conv1d(left_pad, 0),
                dilation,
                groups,
                false,
            ),
        })
    }

    /// Strided causal Conv1d for the `AudioVAE` encoder's downsampling stage:
    /// `kernel_size` must equal `2 * stride`, `dilation = 1`. Left-pad follows
    /// the same `padding = ceil(stride/2)`, `output_padding = stride % 2`
    /// convention as [`super::causal_transpose_conv1d::CausalTransposeConv1d`],
    /// but applied as a LEFT-only pad on a plain (non-transposed) conv rather
    /// than a tail trim: `left_pad = 2*padding - output_padding`, right pad 0.
    /// With `T` a multiple of `stride`, output length is exactly `T / stride`.
    pub fn new_strided(
        weight: Tensor<R>,
        bias: Option<Tensor<R>>,
        stride: usize,
        groups: usize,
    ) -> Result<Self> {
        if stride == 0 {
            return Err(Error::InvalidArgument {
                arg: "stride",
                reason: "must be > 0".into(),
            });
        }
        let kernel_size = 2 * stride;
        if weight.shape().len() != 3 || weight.shape()[2] != kernel_size {
            return Err(Error::InvalidArgument {
                arg: "weight",
                reason: format!(
                    "expected [_, _, {kernel_size}] (kernel = 2*stride), got {:?}",
                    weight.shape()
                ),
            });
        }
        let padding = stride.div_ceil(2);
        let output_padding = stride % 2;
        let left_pad = 2 * padding - output_padding;
        Ok(Self {
            conv: Conv1d::new(
                weight,
                bias,
                stride,
                PaddingMode::conv1d(left_pad, 0),
                1,
                groups,
                false,
            ),
        })
    }

    /// `x [B, C_in, T] -> [B, C_out, T]`.
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
    {
        self.conv.forward_inference(client, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn preserves_length_and_is_causal() {
        let (client, device) = cpu_setup();
        let c = 2;
        let k = 7;
        let t = 10;
        // Depthwise identity-ish kernel: last tap = 1, rest = 0, so output[t]
        // == input[t] exactly (a pure delay-free copy) -- and, being causal,
        // output at any t must not depend on input at t+1..
        let mut w = vec![0.0f32; c * k];
        for ch in 0..c {
            w[ch * k + (k - 1)] = 1.0;
        }
        let weight = Tensor::<CpuRuntime>::from_slice(&w, &[c, 1, k], &device).unwrap();
        let conv = CausalConv1d::new(weight, None, k, 1, c).unwrap();

        let x_data: Vec<f32> = (0..(c * t)).map(|i| i as f32).collect();
        let x = Tensor::<CpuRuntime>::from_slice(&x_data, &[1, c, t], &device).unwrap();
        let out = conv.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, c, t]);
        let got: Vec<f32> = out.contiguous().unwrap().to_vec();
        assert_eq!(got, x_data, "last-tap kernel must reproduce input exactly");
    }

    #[test]
    fn pointwise_kernel_preserves_length() {
        let (client, device) = cpu_setup();
        let c_in = 3;
        let c_out = 5;
        let t = 4;
        let weight = Tensor::<CpuRuntime>::from_slice(
            &vec![0.1f32; c_out * c_in],
            &[c_out, c_in, 1],
            &device,
        )
        .unwrap();
        let bias =
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c_out], &[c_out], &device).unwrap();
        let conv = CausalConv1d::new(weight, Some(bias), 1, 1, 1).unwrap();
        let x = Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; c_in * t], &[1, c_in, t], &device)
            .unwrap();
        let out = conv.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, c_out, t]);
    }

    #[test]
    fn rejects_wrong_kernel_size() {
        let (_client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 6], &[2, 1, 3], &device).unwrap();
        assert!(CausalConv1d::new(weight, None, 7, 1, 2).is_err());
    }

    #[test]
    fn strided_output_length_matches_input_over_stride() {
        let (client, device) = cpu_setup();
        for stride in [2usize, 5, 8] {
            let c = 2;
            let k = 2 * stride;
            let t = stride * 4; // multiple of stride
            let weight =
                Tensor::<CpuRuntime>::from_slice(&vec![0.01f32; c * k], &[c, 1, k], &device)
                    .unwrap();
            let bias = Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], &device).unwrap();
            let conv = CausalConv1d::new_strided(weight, Some(bias), stride, c).unwrap();
            let x = Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; c * t], &[1, c, t], &device)
                .unwrap();
            let out = conv.forward(&client, &x).unwrap();
            assert_eq!(out.shape(), &[1, c, t / stride], "stride {stride}");
            for v in out.contiguous().unwrap().to_vec::<f32>() {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn new_strided_rejects_wrong_kernel_size() {
        let (_client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 6], &[2, 1, 3], &device).unwrap();
        assert!(CausalConv1d::new_strided(weight, None, 2, 2).is_err());
    }
}
