//! `SnakeBeta`: the alias-free activation's inner nonlinearity.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use numr::autograd::{Var, var_snake_beta};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// `1e-9` guard the reference NeuCodec implementation adds to `beta` before dividing.
const SNAKE_EPS: f64 = 1e-9;

/// SnakeBeta: `x + (1 / (exp(beta) + 1e-9)) * sin(x * exp(alpha))^2`.
///
/// `alpha`/`beta` are per-channel `[C]` and stored in LOG scale in this
/// checkpoint (`alpha_logscale=True`). Both are exponentiated ONCE here, at
/// construction, and the LINEAR-scale tensors are what the module holds and
/// what the fused [`numr::autograd::var_snake_beta`] kernel consumes. Using
/// the log-scale values directly would be a silent, plausible-looking error.
///
/// With `trainable`, the gradient lands on the LINEAR-scale parameters.
pub struct SnakeBeta<R: Runtime> {
    /// `[C]`, LINEAR scale.
    alpha: Var<R>,
    /// `[C]`, LINEAR scale.
    beta: Var<R>,
}

impl<R: Runtime<DType = DType>> SnakeBeta<R> {
    /// `alpha`/`beta`: `[channels]`, log-scale as stored in the checkpoint.
    pub fn new(alpha: Tensor<R>, beta: Tensor<R>, trainable: bool) -> Result<Self>
    where
        R::Client: NeuCodecClient<R>,
    {
        if alpha.shape().len() != 1 || alpha.shape() != beta.shape() {
            return Err(Error::InvalidArgument {
                arg: "alpha/beta",
                reason: format!(
                    "both must be 1-D and equal length, got {:?} and {:?}",
                    alpha.shape(),
                    beta.shape()
                ),
            });
        }
        let alpha = alpha.exp().map_err(Error::Numr)?;
        let beta = beta.exp().map_err(Error::Numr)?;
        Ok(Self {
            alpha: Var::new(alpha, trainable),
            beta: Var::new(beta, trainable),
        })
    }

    pub fn channels(&self) -> usize {
        self.alpha.shape()[0]
    }

    /// `x [B, C, T] -> [B, C, T]`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let shape = x.shape().to_vec();
        if shape.len() != 3 || shape[1] != self.channels() {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("expected [B, {}, T], got {shape:?}", self.channels()),
            });
        }
        var_snake_beta(x, &self.alpha, &self.beta, 1, SNAKE_EPS, client).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn var(
        data: &[f32],
        shape: &[usize],
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Var<CpuRuntime> {
        Var::new(
            Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap(),
            false,
        )
    }

    /// alpha/beta are LOG-scale here. With both zero, `exp(0) = 1`, so
    /// SnakeBeta reduces to the textbook `x + sin^2(x)`.
    #[test]
    fn snake_beta_uses_log_scale_parameters() {
        let (client, device) = cpu_setup();
        let zeros = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device).unwrap();
        let snake = SnakeBeta::new(zeros.clone(), zeros, false).unwrap();
        let xs = [0.5f32, -1.25, 2.0, 0.0];
        let x = var(&xs, &[1, 2, 2], &device);
        let out = snake.forward(&client, &x).unwrap();
        let got: Vec<f32> = out.tensor().contiguous().unwrap().to_vec();
        for (g, &v) in got.iter().zip(xs.iter()) {
            let want = v + v.sin().powi(2);
            assert!(
                (g - want).abs() < 1e-5,
                "expected x + sin^2(x) = {want}, got {g}"
            );
        }
    }

    /// A non-zero log-alpha must actually change the frequency — guards against
    /// silently dropping the `exp`.
    #[test]
    fn snake_beta_alpha_scales_frequency() {
        let (client, device) = cpu_setup();
        let ln2 = std::f32::consts::LN_2; // exp(ln2) = 2
        let alpha = Tensor::<CpuRuntime>::from_slice(&[ln2], &[1], &device).unwrap();
        let beta = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device).unwrap();
        let snake = SnakeBeta::new(alpha, beta, false).unwrap();
        let x = var(&[0.7f32], &[1, 1, 1], &device);
        let got: Vec<f32> = snake
            .forward(&client, &x)
            .unwrap()
            .tensor()
            .contiguous()
            .unwrap()
            .to_vec();
        let want = 0.7f32 + (0.7f32 * 2.0).sin().powi(2);
        assert!(
            (got[0] - want).abs() < 1e-5,
            "expected {want}, got {}",
            got[0]
        );
    }
}
