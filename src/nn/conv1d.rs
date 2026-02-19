//! 1D convolution layer

use crate::error::Result;
use numr::autograd::Var;
use numr::ops::{ConvOps, PaddingMode};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// 1D convolution layer: output = conv1d(input, weight) + bias
///
/// Weight: `[out_channels, in_channels/groups, kernel_size]`
/// Input:  `[batch, in_channels, length]`
/// Output: `[batch, out_channels, length_out]`
pub struct Conv1d<R: Runtime> {
    weight: Var<R>,
    bias: Option<Var<R>>,
    stride: usize,
    padding: PaddingMode,
    dilation: usize,
    groups: usize,
}

impl<R: Runtime> Conv1d<R> {
    pub fn new(
        weight: Tensor<R>,
        bias: Option<Tensor<R>>,
        stride: usize,
        padding: PaddingMode,
        dilation: usize,
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

    /// Forward pass.
    ///
    /// Input: `[batch, in_channels, length]`
    /// Output: `[batch, out_channels, length_out]`
    pub fn forward<C>(&self, client: &C, input: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + ConvOps<R>,
    {
        client
            .conv1d(
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
    fn test_conv1d_output_shape() {
        let (client, device) = cpu_setup();
        // weight: [out=4, in=3, kernel=3]
        let weight = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 36], &[4, 3, 3], &device);
        let conv = Conv1d::new(weight, None, 1, PaddingMode::Valid, 1, 1, false);

        // input: [batch=2, channels=3, length=10]
        let input = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 60], &[2, 3, 10], &device);
        let out = conv.forward(&client, &input).unwrap();
        // Valid padding: L_out = 10 - 3 + 1 = 8
        assert_eq!(out.shape(), &[2, 4, 8]);
    }

    #[test]
    fn test_conv1d_same_padding() {
        let (client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 12], &[4, 1, 3], &device);
        let conv = Conv1d::new(weight, None, 1, PaddingMode::Same, 1, 1, false);

        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 10], &[1, 1, 10], &device);
        let out = conv.forward(&client, &input).unwrap();
        assert_eq!(out.shape(), &[1, 4, 10]);
    }

    #[test]
    fn test_conv1d_with_bias() {
        let (client, device) = cpu_setup();
        // Single in/out channel, kernel=1 → effectively a multiply+bias
        let weight = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1, 1, 1], &device);
        let bias = Tensor::<CpuRuntime>::from_slice(&[10.0f32], &[1], &device);
        let conv = Conv1d::new(weight, Some(bias), 1, PaddingMode::Valid, 1, 1, false);

        let input = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 5.0], &[1, 1, 2], &device);
        let out = conv.forward(&client, &input).unwrap();
        let data: Vec<f32> = out.to_vec();
        // 3*2+10=16, 5*2+10=20
        assert_eq!(data, vec![16.0, 20.0]);
    }
}
