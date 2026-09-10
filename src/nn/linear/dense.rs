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
