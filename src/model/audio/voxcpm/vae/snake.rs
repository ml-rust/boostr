//! VoxCPM2's Snake activation.
//!
//! `x + (alpha + 1e-9).recip() * sin(alpha * x)^2`, with `alpha` shaped
//! `[1, C, 1]` in LINEAR scale.
//!
//! This is NOT the same activation as
//! [`crate::model::audio::neucodec::alias_free::SnakeBeta`]: that one has TWO
//! per-channel params (`alpha`, `beta`) stored in LOG scale
//! (`exp(alpha)`/`exp(beta)` at use). VoxCPM2's checkpoint has a single
//! `alpha` tensor per `Snake` module, used directly with no `exp`. Do not
//! unify the two — they are different activations with different checkpoint
//! layouts, only the name rhymes.
//!
//! Inference-only: operates on plain [`Tensor<R>`], no autograd tracking.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::client::VoxCpmClient;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// `1e-9` guard added to `alpha` before the reciprocal.
const SNAKE_EPS: f64 = 1e-9;

/// Snake activation with a single LINEAR-scale `alpha` per channel.
pub struct Snake<R: Runtime> {
    /// `[1, C, 1]`, LINEAR scale (checkpoint stores it pre-exponentiated,
    /// unlike NeuCodec's `SnakeBeta`).
    alpha: Tensor<R>,
    channels: usize,
}

impl<R: Runtime<DType = DType>> Snake<R> {
    /// `alpha`: `[1, channels, 1]`, LINEAR scale.
    pub fn new(alpha: Tensor<R>) -> Result<Self> {
        let shape = alpha.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[2] != 1 {
            return Err(Error::InvalidArgument {
                arg: "alpha",
                reason: format!("expected [1, C, 1], got {shape:?}"),
            });
        }
        let channels = shape[1];
        Ok(Self { alpha, channels })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// `x [B, C, T] -> [B, C, T]`.
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
    {
        let shape = x.shape().to_vec();
        if shape.len() != 3 || shape[1] != self.channels {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("expected [B, {}, T], got {shape:?}", self.channels),
            });
        }

        let scaled = client.mul(x, &self.alpha).map_err(Error::Numr)?;
        let s = client.sin(&scaled).map_err(Error::Numr)?;
        let s2 = client.mul(&s, &s).map_err(Error::Numr)?;

        let alpha_eps = client
            .add_scalar(&self.alpha, SNAKE_EPS)
            .map_err(Error::Numr)?;
        let recip = client.recip(&alpha_eps).map_err(Error::Numr)?;
        let term = client.mul(&s2, &recip).map_err(Error::Numr)?;
        client.add(x, &term).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn zero_alpha_reduces_to_x_plus_1e9_sin_sq() {
        // alpha = 0 => sin(0*x) = 0 => term = 0 / (0+eps) = 0, so out == x.
        let (client, device) = cpu_setup();
        let c = 2;
        let alpha =
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[1, c, 1], &device).unwrap();
        let snake = Snake::new(alpha).unwrap();
        let xs = [0.5f32, -1.25, 2.0, 0.0];
        let x = Tensor::<CpuRuntime>::from_slice(&xs, &[1, c, 2], &device).unwrap();
        let out = snake.forward(&client, &x).unwrap();
        let got: Vec<f32> = out.contiguous().unwrap().to_vec();
        for (g, v) in got.iter().zip(xs.iter()) {
            assert!((g - v).abs() < 1e-5, "expected {v}, got {g}");
        }
    }

    #[test]
    fn nonzero_alpha_matches_reference_formula() {
        let (client, device) = cpu_setup();
        let c = 1;
        let alpha_val = 1.3f32;
        let alpha = Tensor::<CpuRuntime>::from_slice(&[alpha_val], &[1, c, 1], &device).unwrap();
        let snake = Snake::new(alpha).unwrap();
        let x_val = 0.7f32;
        let x = Tensor::<CpuRuntime>::from_slice(&[x_val], &[1, c, 1], &device).unwrap();
        let got: Vec<f32> = snake
            .forward(&client, &x)
            .unwrap()
            .contiguous()
            .unwrap()
            .to_vec();
        let want = x_val + (1.0 / (alpha_val + 1e-9)) * (alpha_val * x_val).sin().powi(2);
        assert!(
            (got[0] - want).abs() < 1e-4,
            "expected {want}, got {}",
            got[0]
        );
    }

    #[test]
    fn rejects_wrong_alpha_shape() {
        let (_client, device) = cpu_setup();
        let bad = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device).unwrap();
        assert!(Snake::new(bad).is_err());
    }
}
