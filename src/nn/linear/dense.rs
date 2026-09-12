use crate::error::Result;
use crate::nn::module::Module;
use numr::autograd::{Var, var_add, var_matmul, var_reshape, var_transpose};
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Dense linear layer: output = input @ weight^T + bias
///
/// Uses `Var<R>` throughout — autograd works during training,
/// near-zero overhead during inference.
pub struct Linear<R: Runtime> {
    weight: Var<R>,
    bias: Option<Var<R>>,
}

impl<R: Runtime> Linear<R> {
    /// Create from loaded tensors. `trainable` controls gradient tracking.
    pub fn new(weight: Tensor<R>, bias: Option<Tensor<R>>, trainable: bool) -> Self {
        Self {
            weight: Var::new(weight, trainable),
            bias: bias.map(|b| Var::new(b, trainable)),
        }
    }

    /// Create from tensors while preserving stable autograd IDs.
    ///
    /// Use this when rebuilding a layer from optimizer-updated tensors so the
    /// optimizer state keyed by `TensorId` remains attached to the same logical
    /// parameters across steps.
    pub fn with_ids(
        weight: Tensor<R>,
        weight_id: TensorId,
        bias: Option<(Tensor<R>, TensorId)>,
        trainable: bool,
    ) -> Self {
        Self {
            weight: Var::with_id(weight, weight_id, trainable),
            bias: bias.map(|(b, id)| Var::with_id(b, id, trainable)),
        }
    }

    /// Forward: input @ weight^T + bias
    ///
    /// input: `[..., in_features]`, output: `[..., out_features]`
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R>,
    {
        let w_t = var_transpose(&self.weight).map_err(crate::error::Error::Numr)?;
        let input_shape = input.shape().to_vec();

        if input_shape.len() <= 2 {
            let output = var_matmul(input, &w_t, client).map_err(crate::error::Error::Numr)?;
            return match &self.bias {
                Some(bias) => var_add(&output, bias, client).map_err(crate::error::Error::Numr),
                None => Ok(output),
            };
        }

        let last_axis = input_shape.len() - 1;
        let in_features = input_shape[last_axis];
        let leading: usize = input_shape[..last_axis].iter().product();
        let flat_input =
            var_reshape(input, &[leading, in_features]).map_err(crate::error::Error::Numr)?;
        let flat_output =
            var_matmul(&flat_input, &w_t, client).map_err(crate::error::Error::Numr)?;
        let flat_output = match &self.bias {
            Some(bias) => var_add(&flat_output, bias, client).map_err(crate::error::Error::Numr)?,
            None => flat_output,
        };

        let weight_shape = self.weight.tensor().shape();
        if weight_shape.is_empty() {
            return Err(crate::error::Error::ModelError {
                reason: "linear weight must have at least one dimension".into(),
            });
        }
        let mut output_shape = input_shape;
        output_shape[last_axis] = weight_shape[0];
        var_reshape(&flat_output, &output_shape).map_err(crate::error::Error::Numr)
    }

    pub fn weight(&self) -> &Var<R> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Var<R>> {
        self.bias.as_ref()
    }

    /// All parameters with their stable autograd IDs.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        let mut params = vec![(self.weight.id(), &self.weight)];
        if let Some(bias) = &self.bias {
            params.push((bias.id(), bias));
        }
        params
    }

    /// Trainable parameters with their stable autograd IDs.
    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters()
            .into_iter()
            .filter(|param| param.1.requires_grad())
            .collect()
    }

    /// Cheap duplicate that preserves every field's `TensorId`, for capturing
    /// this layer by owned value in a `'static` activation-checkpointing
    /// closure. Uses [`Var::alias`], not [`Clone`]: a `clone` would mint a
    /// fresh id for `weight`/`bias` and silently orphan their gradients.
    pub fn alias(&self) -> Self {
        Self {
            weight: self.weight.alias(),
            bias: self.bias.as_ref().map(Var::alias),
        }
    }
}

impl<R: Runtime> Module<R> for Linear<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = vec![self.weight()];
        if let Some(bias) = self.bias() {
            params.push(bias);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = vec![("weight".to_string(), self.weight())];
        if let Some(bias) = self.bias() {
            params.push(("bias".to_string(), bias));
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_linear_output_shape() {
        let (client, device) = cpu_setup();
        // weight: [out=4, in=3]
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device).unwrap();
        let linear = Linear::new(weight, None, false);

        // input: [2, 3]
        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device).unwrap(),
            false,
        );
        let out = linear.forward(&client, &input).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
    }

    #[test]
    fn test_linear_with_bias() {
        let (client, device) = cpu_setup();
        let weight =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap();
        let bias = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0], &[2], &device).unwrap();
        let linear = Linear::new(weight, Some(bias), false);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device).unwrap(),
            false,
        );
        let out = linear.forward(&client, &input).unwrap();
        let data: Vec<f32> = out.tensor().to_vec();
        // [1,2] @ [[1,0],[0,1]] + [10,20] = [1,2] + [10,20] = [11,22]
        assert_eq!(data, vec![11.0, 22.0]);
    }

    #[test]
    fn test_linear_batched() {
        let (client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device).unwrap();
        let linear = Linear::new(weight, None, false);

        // input: [4, 5, 3] — batched
        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 60], &[4, 5, 3], &device).unwrap(),
            false,
        );
        let out = linear.forward(&client, &input).unwrap();
        assert_eq!(out.shape(), &[4, 5, 2]);
    }
}
