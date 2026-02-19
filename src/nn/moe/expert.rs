//! MoE Expert — individual SwiGLU MLP

use crate::error::{Error, Result};
use crate::nn::Linear;
use numr::autograd::{Var, var_mul, var_silu};
use numr::ops::{ActivationOps, ReduceOps, ScalarOps, ShapeOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Single expert MLP (SwiGLU architecture).
///
/// Architecture: `down_proj(silu(gate_proj(x)) * up_proj(x))`
pub struct Expert<R: Runtime> {
    gate_proj: Linear<R>,
    up_proj: Linear<R>,
    down_proj: Linear<R>,
}

impl<R: Runtime> Expert<R> {
    pub fn new(gate_proj: Linear<R>, up_proj: Linear<R>, down_proj: Linear<R>) -> Self {
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
        Self {
            gate_proj: Linear::new(gate_proj, None, trainable),
            up_proj: Linear::new(up_proj, None, trainable),
            down_proj: Linear::new(down_proj, None, trainable),
        }
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
            + ActivationOps<R>,
        R::Client: TensorOps<R> + ActivationOps<R> + ScalarOps<R>,
    {
        let gate = self.gate_proj.forward(client, x)?;
        let up = self.up_proj.forward(client, x)?;

        let gate_silu = var_silu(&gate, client).map_err(Error::Numr)?;
        let hidden = var_mul(&gate_silu, &up, client).map_err(Error::Numr)?;
        self.down_proj.forward(client, &hidden)
    }

    pub fn gate_proj(&self) -> &Linear<R> {
        &self.gate_proj
    }

    pub fn up_proj(&self) -> &Linear<R> {
        &self.up_proj
    }

    pub fn down_proj(&self) -> &Linear<R> {
        &self.down_proj
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_expert_forward_shape() {
        let (client, device) = cpu_setup();
        let hidden = 4;
        let inter = 8;

        let gate_w = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 32], &[inter, hidden], &device);
        let up_w = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 32], &[inter, hidden], &device);
        let down_w = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 32], &[hidden, inter], &device);

        let expert = Expert::from_tensors(gate_w, up_w, down_w, false);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 8], &[2, hidden], &device),
            false,
        );
        let out = expert.forward(&client, &input).unwrap();
        assert_eq!(out.shape(), &[2, hidden]);
    }
}
