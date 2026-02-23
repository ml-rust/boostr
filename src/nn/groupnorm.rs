//! Group Normalization module
//!
//! GroupNorm: normalizes over groups of channels independently.
//! Delegates to numr's `var_group_norm` for forward + autograd.

use crate::error::{Error, Result};
use numr::autograd::{Var, var_group_norm};
use numr::ops::{NormalizationOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Group Normalization
///
/// Divides channels into `num_groups` groups and normalizes each group
/// independently. Input shape: `[batch, channels, *spatial]`.
///
/// weight (gamma): `[channels]`
/// bias (beta): `[channels]`
pub struct GroupNorm<R: Runtime> {
    weight: Var<R>,
    bias: Var<R>,
    num_groups: usize,
    eps: f32,
}

impl<R: Runtime> GroupNorm<R> {
    pub fn new(
        weight: Tensor<R>,
        bias: Tensor<R>,
        num_groups: usize,
        eps: f32,
        trainable: bool,
    ) -> Self {
        Self {
            weight: Var::new(weight, trainable),
            bias: Var::new(bias, trainable),
            num_groups,
            eps,
        }
    }

    /// Forward pass with autograd.
    ///
    /// Input: `[batch, channels, *spatial]`
    /// Output: same shape, normalized per group.
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        R: Runtime,
        C: RuntimeClient<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        var_group_norm(
            input,
            &self.weight,
            &self.bias,
            self.num_groups,
            self.eps,
            client,
        )
        .map_err(Error::Numr)
    }

    pub fn weight(&self) -> &Var<R> {
        &self.weight
    }

    pub fn bias(&self) -> &Var<R> {
        &self.bias
    }

    pub fn num_groups(&self) -> usize {
        self.num_groups
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::autograd::backward;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_groupnorm_output_shape() {
        let (client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device);
        let bias = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device);
        let norm = GroupNorm::new(weight, bias, 2, 1e-5, false);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 24], &[2, 4, 3], &device),
            false,
        );
        let out = norm.forward(&client, &input).unwrap();
        assert_eq!(out.tensor().shape(), &[2, 4, 3]);
    }

    #[test]
    fn test_groupnorm_zero_mean_per_group() {
        let (client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device);
        let bias = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device);
        let norm = GroupNorm::new(weight, bias, 2, 1e-5, false);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[
                    1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
                &[1, 4, 3],
                &device,
            ),
            false,
        );
        let out = norm.forward(&client, &input).unwrap();
        let data: Vec<f32> = out.tensor().to_vec();

        // Group 0 (ch 0,1) should have mean ~0
        let g0: f32 = data[0..6].iter().sum();
        assert!(g0.abs() < 1e-4, "group 0 mean should be ~0, sum={g0}");
    }

    #[test]
    fn test_groupnorm_backward() {
        let (client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device);
        let bias = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device);
        let norm = GroupNorm::new(weight, bias, 2, 1e-5, true);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                &[1, 4, 2],
                &device,
            ),
            true,
        );
        let out = norm.forward(&client, &input).unwrap();
        let loss = numr::autograd::var_sum(&out, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let d_input: Vec<f32> = grads.get(input.id()).unwrap().to_vec();
        assert_eq!(d_input.len(), 8);
        for v in &d_input {
            assert!(v.is_finite());
        }
    }
}
