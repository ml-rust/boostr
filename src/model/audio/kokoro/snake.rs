//! Snake activation.
//!
//! Per Liu et al. 2020 ("Neural Networks Fail to Learn Periodic Functions and
//! How to Fix It"):
//!
//! ```text
//!     snake(x) = x + (1 / (α + ε)) * sin²(α · x)
//! ```
//!
//! Kokoro's ISTFTNet generator uses a learnable per-channel α shaped
//! `[1, C, 1]` in every `AdaINResBlock1` site (stored as `alpha1` / `alpha2`
//! `ParameterList`s in the state_dict). `ε` is a tiny positive constant that
//! keeps the reciprocal bounded when α is near zero — we follow the
//! `SnakeBeta` upstream in using `1e-9`.
//!
//! Composite op: no new kernel needed. Uses mul / sin / div / add from numr.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Apply snake activation with a per-channel α.
///
/// * `x` — activation of shape `[B, C, T]`.
/// * `alpha` — shape `[1, C, 1]` (matches the state_dict layout used by
///   `torch.nn.ParameterList([Parameter(torch.ones(1, ch, 1))])`).
/// * `eps` — denominator floor; pass `1e-9` to match upstream, or a larger
///   value if numerical stability is a concern.
pub fn snake<R, C>(client: &C, x: &Tensor<R>, alpha: &Tensor<R>, eps: f64) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + ScalarOps<R>,
{
    let x_shape = x.shape();
    if x_shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("snake expects rank-3 [B, C, T], got {x_shape:?}"),
        });
    }
    let channels = x_shape[1];
    let alpha_shape = alpha.shape();
    if alpha_shape != [1, channels, 1] {
        return Err(Error::InvalidArgument {
            arg: "alpha",
            reason: format!("snake alpha shape must be [1, {channels}, 1], got {alpha_shape:?}"),
        });
    }

    // α · x (broadcast across B and T).
    let ax = client.mul(alpha, x).map_err(Error::Numr)?;
    // sin²(α · x)
    let s = client.sin(&ax).map_err(Error::Numr)?;
    let s_sq = client.mul(&s, &s).map_err(Error::Numr)?;
    // 1 / (α + ε) — add scalar ε then reciprocate.
    let alpha_eps = client.add_scalar(alpha, eps).map_err(Error::Numr)?;
    let inv = client.recip(&alpha_eps).map_err(Error::Numr)?;
    // x + (1 / (α + ε)) · sin²(α · x)
    let scaled = client.mul(&inv, &s_sq).map_err(Error::Numr)?;
    client.add(x, &scaled).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn zero_input_stays_zero() {
        let (client, device) = cpu_setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 6], &[1, 2, 3], &device);
        let alpha = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[1, 2, 1], &device);
        let y = snake(&client, &x, &alpha, 1e-9).unwrap();
        for v in y.to_vec::<f32>() {
            assert!(v.abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn matches_formula_for_alpha_1() {
        // snake(x) = x + sin²(x). Check pointwise.
        let (client, device) = cpu_setup();
        let input: Vec<f32> = vec![-1.5, -0.5, 0.0, 0.5, 1.5];
        let x = Tensor::<CpuRuntime>::from_slice(&input, &[1, 1, 5], &device);
        let alpha = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1, 1, 1], &device);
        let y: Vec<f32> = snake(&client, &x, &alpha, 0.0).unwrap().to_vec();
        for (got, x_in) in y.iter().zip(&input) {
            let expected = x_in + x_in.sin().powi(2);
            assert!(
                (got - expected).abs() < 1e-5,
                "snake({x_in}) = {got} vs {expected}"
            );
        }
    }

    #[test]
    fn alpha_scales_frequency() {
        // For α=2, period halves: snake(x) = x + sin²(2x).
        let (client, device) = cpu_setup();
        let input: Vec<f32> = vec![0.5, 1.0];
        let x = Tensor::<CpuRuntime>::from_slice(&input, &[1, 1, 2], &device);
        let alpha = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1, 1, 1], &device);
        let y: Vec<f32> = snake(&client, &x, &alpha, 0.0).unwrap().to_vec();
        for (got, x_in) in y.iter().zip(&input) {
            let expected = x_in + 0.5 * (2.0 * x_in).sin().powi(2);
            assert!(
                (got - expected).abs() < 1e-5,
                "snake α=2({x_in}) = {got} vs {expected}"
            );
        }
    }

    #[test]
    fn rejects_bad_alpha_shape() {
        let (client, device) = cpu_setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 6], &[1, 2, 3], &device);
        // Should be [1, 2, 1]; give [2] instead.
        let alpha = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device);
        assert!(snake(&client, &x, &alpha, 1e-9).is_err());
    }

    #[test]
    fn rejects_bad_input_rank() {
        let (client, device) = cpu_setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[2, 2], &device);
        let alpha = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[1, 2, 1], &device);
        assert!(snake(&client, &x, &alpha, 1e-9).is_err());
    }
}
