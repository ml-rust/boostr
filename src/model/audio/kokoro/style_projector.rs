//! Kokoro / StyleTTS2 style projector.
//!
//! Each AdaIN call in the decoder needs a per-sample `(gamma, beta)` pair
//! derived from the voice's 256-d style vector. The StyleTTS2 reference keeps
//! one small projector per AdaIN site:
//!
//! ```text
//!     style [B, style_dim] → Linear(style_dim → 2*channels) → split → (gamma, beta)
//! ```
//!
//! `gamma` and `beta` are both `[B, channels]` and plug directly into
//! `nn::AdaIn1d::forward(...)`.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{MatmulOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Projector that produces AdaIN `(gamma, beta)` from a style vector.
pub struct StyleProjector<R: Runtime> {
    /// Shape `[2 * channels, style_dim]` — PyTorch Linear convention.
    weight: Tensor<R>,
    /// Shape `[2 * channels]`.
    bias: Tensor<R>,
    style_dim: usize,
    channels: usize,
}

impl<R: Runtime> StyleProjector<R> {
    pub fn new(weight: Tensor<R>, bias: Tensor<R>) -> Result<Self> {
        let w_shape = weight.shape();
        if w_shape.len() != 2 {
            return Err(Error::InvalidArgument {
                arg: "weight",
                reason: format!("expected rank-2 [2*C, style_dim], got {w_shape:?}"),
            });
        }
        let twoc = w_shape[0];
        let style_dim = w_shape[1];
        if !twoc.is_multiple_of(2) {
            return Err(Error::InvalidArgument {
                arg: "weight",
                reason: format!("first dim must be even (2 * channels), got {twoc}"),
            });
        }
        let channels = twoc / 2;
        if bias.shape() != [twoc] {
            return Err(Error::InvalidArgument {
                arg: "bias",
                reason: format!("expected shape [{twoc}], got {:?}", bias.shape()),
            });
        }
        Ok(Self {
            weight,
            bias,
            style_dim,
            channels,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn style_dim(&self) -> usize {
        self.style_dim
    }

    /// Forward: `style [B, style_dim]` → `(gamma [B, C], beta [B, C])`.
    #[allow(clippy::type_complexity)]
    pub fn forward<C>(&self, client: &C, style: &Tensor<R>) -> Result<(Tensor<R>, Tensor<R>)>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + MatmulOps<R> + TensorOps<R>,
    {
        let s_shape = style.shape();
        if s_shape.len() != 2 || s_shape[1] != self.style_dim {
            return Err(Error::InvalidArgument {
                arg: "style",
                reason: format!("expected [B, {}], got {s_shape:?}", self.style_dim),
            });
        }

        // style @ W^T + b → [B, 2C]
        let w_t = self.weight.transpose(0, 1).map_err(Error::Numr)?;
        let projected = client
            .matmul_bias(style, &w_t, &self.bias)
            .map_err(Error::Numr)?;

        let c = self.channels;
        let gamma = projected
            .narrow(1, 0, c)
            .map_err(Error::Numr)?
            .contiguous()?;
        let beta = projected
            .narrow(1, c, c)
            .map_err(Error::Numr)?
            .contiguous()?;
        Ok((gamma, beta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn zeros(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device)
    }

    #[test]
    fn split_yields_two_half_tensors_of_correct_shape() {
        let (client, device) = cpu_setup();
        // channels = 4, style_dim = 3 → weight [8, 3], bias [8].
        let weight = zeros(&[8, 3], &device);
        let bias = zeros(&[8], &device);
        let proj = StyleProjector::new(weight, bias).unwrap();

        let style =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let (gamma, beta) = proj.forward(&client, &style).unwrap();
        assert_eq!(gamma.shape(), &[2, 4]);
        assert_eq!(beta.shape(), &[2, 4]);
    }

    #[test]
    fn bias_controls_gamma_beta_when_weights_are_zero() {
        // With weights=0, output = bias; first C values land in gamma, next C in beta.
        let (client, device) = cpu_setup();
        let weight = zeros(&[4, 3], &device);
        let bias = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device);
        let proj = StyleProjector::new(weight, bias).unwrap();

        let style = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 3], &[1, 3], &device);
        let (gamma, beta) = proj.forward(&client, &style).unwrap();
        let g: Vec<f32> = gamma.to_vec();
        let b: Vec<f32> = beta.to_vec();
        assert_eq!(g, vec![10.0, 20.0]);
        assert_eq!(b, vec![30.0, 40.0]);
    }

    #[test]
    fn rejects_odd_output_dim() {
        let (_client, device) = cpu_setup();
        let weight = zeros(&[5, 3], &device);
        let bias = zeros(&[5], &device);
        assert!(StyleProjector::new(weight, bias).is_err());
    }

    #[test]
    fn rejects_mismatched_style_dim() {
        let (client, device) = cpu_setup();
        let weight = zeros(&[4, 3], &device);
        let bias = zeros(&[4], &device);
        let proj = StyleProjector::new(weight, bias).unwrap();

        let style = zeros(&[1, 5], &device);
        assert!(proj.forward(&client, &style).is_err());
    }
}
