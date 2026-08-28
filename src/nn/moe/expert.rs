//! MoE Expert — individual SwiGLU MLP

use crate::error::{Error, Result};
use crate::nn::linear::Linear;
use crate::nn::maybe_lora::MaybeLoraLinear;
use crate::nn::module::Module;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_mul, var_silu};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Single expert MLP (SwiGLU architecture).
///
/// Architecture: `down_proj(silu(gate_proj(x)) * up_proj(x))`
///
/// Each projection is a [`MaybeLoraLinear`], so an expert can be LoRA-adapted
/// per projection without a separate expert type.
pub struct Expert<R: Runtime> {
    gate_proj: MaybeLoraLinear<R>,
    up_proj: MaybeLoraLinear<R>,
    down_proj: MaybeLoraLinear<R>,
}

impl<R: Runtime<DType = DType>> Expert<R> {
    /// Create from plain linear projections.
    pub fn new(gate_proj: Linear<R>, up_proj: Linear<R>, down_proj: Linear<R>) -> Self {
        Self {
            gate_proj: gate_proj.into(),
            up_proj: up_proj.into(),
            down_proj: down_proj.into(),
        }
    }

    /// Create from projections that may each carry a LoRA adapter.
    pub fn new_adapted(
        gate_proj: MaybeLoraLinear<R>,
        up_proj: MaybeLoraLinear<R>,
        down_proj: MaybeLoraLinear<R>,
    ) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Create from tensors. Expects:
    /// - gate_proj: `[intermediate, hidden]`
    /// - up_proj: `[intermediate, hidden]`
    /// - down_proj: `[hidden, intermediate]`
    pub fn from_tensors(
        gate_proj: Tensor<R>,
        up_proj: Tensor<R>,
        down_proj: Tensor<R>,
        trainable: bool,
    ) -> Self {
        Self::new(
            Linear::new(gate_proj, None, trainable),
            Linear::new(up_proj, None, trainable),
            Linear::new(down_proj, None, trainable),
        )
    }

    /// SwiGLU forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        R: Runtime<DType = numr::dtype::DType>,
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + QuantMatmulOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + ActivationOps<R> + ScalarOps<R> + BinaryOps<R> + DequantOps<R>,
    {
        let gate = self.gate_proj.forward(client, x)?;
        let up = self.up_proj.forward(client, x)?;

        let gate_silu = var_silu(&gate, client).map_err(Error::Numr)?;
        let hidden = var_mul(&gate_silu, &up, client).map_err(Error::Numr)?;
        self.down_proj.forward(client, &hidden)
    }

    /// Fold every adapter into its base weight, producing a plain expert.
    ///
    /// Mirrors [`LoraLinear::merge_into_base`](crate::nn::LoraLinear::merge_into_base)
    /// at expert granularity — for export and inference after training.
    pub fn merge_adapters<C>(&self, client: &C) -> Result<Self>
    where
        C: RuntimeClient<R> + TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
        R::Client: TensorOps<R> + BinaryOps<R> + ScalarOps<R> + DequantOps<R>,
    {
        Ok(Self::new(
            self.gate_proj.merge_into_base(client)?,
            self.up_proj.merge_into_base(client)?,
            self.down_proj.merge_into_base(client)?,
        ))
    }

    pub fn gate_proj(&self) -> &MaybeLoraLinear<R> {
        &self.gate_proj
    }

    pub fn up_proj(&self) -> &MaybeLoraLinear<R> {
        &self.up_proj
    }

    pub fn down_proj(&self) -> &MaybeLoraLinear<R> {
        &self.down_proj
    }

    /// All parameters with their stable autograd IDs, adapters included.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        let mut params = self.gate_proj.parameters();
        params.extend(self.up_proj.parameters());
        params.extend(self.down_proj.parameters());
        params
    }
}

impl<R: Runtime<DType = DType>> Module<R> for Expert<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        Expert::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        for (prefix, proj) in [
            ("gate_proj", &self.gate_proj),
            ("up_proj", &self.up_proj),
            ("down_proj", &self.down_proj),
        ] {
            params.extend(
                proj.named_parameters()
                    .into_iter()
                    .map(|(name, var)| (format!("{prefix}.{name}"), var)),
            );
        }
        params
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        Expert::parameters(self)
    }
}

#[cfg(test)]
mod tests;
