//! 2D convolution layer with autograd support

use crate::error::Result;
use numr::autograd::Var;
use numr::ops::{ConvOps, PaddingMode};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// 2D convolution layer: output = conv2d(input, weight) + bias
///
/// Weight: `[out_channels, in_channels/groups, kH, kW]`
/// Input:  `[batch, in_channels, height, width]`
/// Output: `[batch, out_channels, height_out, width_out]`
///
/// Supports autograd: when `trainable=true`, gradients flow through
/// to input, weight, and bias during backward pass.
pub struct Conv2d<R: Runtime> {
    weight: Var<R>,
    bias: Option<Var<R>>,
    stride: (usize, usize),
    padding: PaddingMode,
    dilation: (usize, usize),
    groups: usize,
}

impl<R: Runtime> Conv2d<R> {
    pub fn new(
        weight: Tensor<R>,
        bias: Option<Tensor<R>>,
        stride: (usize, usize),
        padding: PaddingMode,
        dilation: (usize, usize),
        groups: usize,
        trainable: bool,
    ) -> Self {
        Self {
            weight: Var::new(weight, trainable),
            bias: bias.map(|b| Var::new(b, trainable)),
            stride,
            padding,
            dilation,
            groups,
        }
    }

    /// Forward pass with autograd support.
    ///
    /// Input: `[batch, in_channels, height, width]`
    /// Output: `[batch, out_channels, height_out, width_out]`
    ///
    /// NOTE: var_conv2d does not exist in numr yet (being added concurrently).
    /// This currently falls back to forward_inference wrapped in a non-tracking Var.
    /// Once numr adds var_conv2d, this should be updated to call it directly
    /// (mirroring how Conv1d::forward calls var_conv1d).
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + ConvOps<R>,
    {
        let output = client
            .conv2d(
                input.tensor(),
                self.weight.tensor(),
                self.bias.as_ref().map(|b| b.tensor()),
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            .map_err(crate::error::Error::Numr)?;
        Ok(Var::new(output, false))
    }

    /// Forward pass without autograd (inference only, returns raw Tensor).
    pub fn forward_inference<C>(&self, client: &C, input: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + ConvOps<R>,
    {
        client
            .conv2d(
                input,
                self.weight.tensor(),
                self.bias.as_ref().map(|b| b.tensor()),
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            .map_err(crate::error::Error::Numr)
    }

    pub fn weight(&self) -> &Var<R> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Var<R>> {
        self.bias.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_conv2d_output_shape() {
        let (client, device) = cpu_setup();
        // weight: [out=4, in=3, kH=3, kW=3]
        let weight = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 108], &[4, 3, 3, 3], &device);
        let conv = Conv2d::new(weight, None, (1, 1), PaddingMode::Valid, (1, 1), 1, false);

        // input: [batch=2, channels=3, height=8, width=10]
        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 480], &[2, 3, 8, 10], &device),
            false,
        );
        let out = conv.forward(&client, &input).unwrap();
        // Valid padding: H_out = 8 - 3 + 1 = 6, W_out = 10 - 3 + 1 = 8
        assert_eq!(out.tensor().shape(), &[2, 4, 6, 8]);
    }

    #[test]
    fn test_conv2d_with_bias() {
        let (client, device) = cpu_setup();
        // Single in/out channel, kernel=1x1 -> effectively a multiply+bias
        let weight = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1, 1, 1, 1], &device);
        let bias = Tensor::<CpuRuntime>::from_slice(&[10.0f32], &[1], &device);
        let conv = Conv2d::new(
            weight,
            Some(bias),
            (1, 1),
            PaddingMode::Valid,
            (1, 1),
            1,
            false,
        );

        // input: [batch=1, channels=1, height=1, width=2]
        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 5.0], &[1, 1, 1, 2], &device),
            false,
        );
        let out = conv.forward(&client, &input).unwrap();
        let data: Vec<f32> = out.tensor().to_vec();
        // 3*2+10=16, 5*2+10=20
        assert_eq!(data, vec![16.0, 20.0]);
    }
}
