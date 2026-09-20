//! The `ssm_alpha` and `ssm_beta` projections of a [`GdnBlock`](super::GdnBlock).
//!
//! Both read the same `x` and are narrow (`[value_heads, hidden_size]`),
//! so at decode each is one latency-bound launch. When both arrive dense
//! they are concatenated along the output axis into one
//! `[2 * value_heads, hidden_size]` linear at construction, and the
//! forward splits the result with two offset views. The split moves no
//! data: every consumer takes a strided view or copies it itself.
//!
//! Bit identity with the two separate projections: every f32 `x @ Wᵀ`
//! kernel on the CUDA path forms an output element as one `fma` chain
//! over `k`, independent of `N` and of the row index, and the CPU path
//! runs one dot product per output row with a fixed reduction order at
//! these shapes. The concatenated rows therefore hold the same bits the
//! separate projections would.

use crate::error::{Error, Result};
use crate::nn::{Linear, MaybeQuantLinear};
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{BinaryOps, MatmulOps, ShapeOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// The two raw gate projections.
pub(super) enum GateProjections<R: Runtime> {
    /// One dense linear `[2 * heads, hidden_size]`: rows `[0, heads)` are
    /// `ssm_alpha`, rows `[heads, 2 * heads)` are `ssm_beta`. The original
    /// weight ids stay so the importance-matrix tap still records each
    /// projection under its checkpoint name.
    Fused {
        linear: Box<Linear<R>>,
        heads: usize,
        alpha_id: TensorId,
        beta_id: TensorId,
    },
    /// Kept apart: at least one projection is block-quantized, so there is
    /// no dense weight to concatenate.
    Separate(Box<SeparateGates<R>>),
}

/// The two projections kept apart.
pub(super) struct SeparateGates<R: Runtime> {
    pub(super) alpha: MaybeQuantLinear<R>,
    pub(super) beta: MaybeQuantLinear<R>,
}

impl<R: Runtime<DType = DType>> GateProjections<R> {
    /// Fuse `alpha` and `beta` (`[heads, hidden_size]` each) when both are
    /// dense; keep them apart otherwise. A bias present on only one side is
    /// padded with zeros for the other.
    ///
    /// # Errors
    ///
    /// [`Error::Numr`] when the concatenation fails.
    pub(super) fn new(alpha: MaybeQuantLinear<R>, beta: MaybeQuantLinear<R>) -> Result<Self>
    where
        R::Client: ShapeOps<R>,
    {
        let (alpha_dense, beta_dense) = match (&alpha, &beta) {
            (MaybeQuantLinear::Standard(a), MaybeQuantLinear::Standard(b)) => (a, b),
            _ => return Ok(Self::Separate(Box::new(SeparateGates { alpha, beta }))),
        };
        let heads = alpha_dense.weight().tensor().shape()[0];
        let alpha_id = alpha_dense.weight().id();
        let beta_id = beta_dense.weight().id();
        let weight = Tensor::cat(
            &[alpha_dense.weight().tensor(), beta_dense.weight().tensor()],
            0,
        )
        .map_err(Error::Numr)?;
        let bias = fused_bias(alpha_dense, beta_dense, heads)?;
        Ok(Self::Fused {
            linear: Box::new(Linear::new(weight, bias, false)),
            heads,
            alpha_id,
            beta_id,
        })
    }

    /// `(alpha_raw, beta_raw)`, each `[batch, seq, heads]`, for `x`
    /// `[batch, seq, hidden_size]`. On the fused path both are views into
    /// one `[batch, seq, 2 * heads]` result; `beta_raw` starts at column
    /// `heads`.
    pub(super) fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<(Tensor<R>, Tensor<R>)>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + DequantOps<R> + MatmulOps<R>,
    {
        match self {
            Self::Fused {
                linear,
                heads,
                alpha_id,
                beta_id,
            } => {
                if crate::quant::imatrix::is_armed() {
                    crate::quant::imatrix::observe(*alpha_id, client, x.tensor())?;
                    crate::quant::imatrix::observe(*beta_id, client, x.tensor())?;
                }
                let both = linear.forward(client, x)?;
                let both = both.tensor();
                let alpha = both.narrow(-1, 0, *heads).map_err(Error::Numr)?;
                let beta = both.narrow(-1, *heads, *heads).map_err(Error::Numr)?;
                Ok((alpha, beta))
            }
            Self::Separate(gates) => {
                let SeparateGates { alpha, beta } = gates.as_ref();
                let alpha = alpha.forward(client, x)?;
                let beta = beta.forward(client, x)?;
                Ok((alpha.tensor().clone(), beta.tensor().clone()))
            }
        }
    }
}

/// The concatenated bias, `None` when neither projection has one.
fn fused_bias<R: Runtime<DType = DType>>(
    alpha: &Linear<R>,
    beta: &Linear<R>,
    heads: usize,
) -> Result<Option<Tensor<R>>>
where
    R::Client: ShapeOps<R>,
{
    let (a, b) = match (alpha.bias(), beta.bias()) {
        (None, None) => return Ok(None),
        (a, b) => (a, b),
    };
    let weight = alpha.weight().tensor();
    let zeros = || Tensor::<R>::zeros(&[heads], weight.dtype(), weight.device());
    let a = match a {
        Some(v) => v.tensor().clone(),
        None => zeros().map_err(Error::Numr)?,
    };
    let b = match b {
        Some(v) => v.tensor().clone(),
        None => zeros().map_err(Error::Numr)?,
    };
    Tensor::cat(&[&a, &b], 0).map(Some).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    const HEADS: usize = 4;
    const HIDDEN: usize = 8;

    fn tensor(seed: u64, shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let mut state = seed;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
            })
            .collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
    }

    fn dense(w: Tensor<CpuRuntime>, b: Option<Tensor<CpuRuntime>>) -> MaybeQuantLinear<CpuRuntime> {
        MaybeQuantLinear::Standard(Linear::new(w, b, false))
    }

    /// The fused projection's two halves equal the separate projections
    /// bit for bit, with and without a bias on one side.
    #[test]
    fn fused_halves_equal_separate_projections() {
        let (client, device) = cpu_setup();
        let wa = tensor(1, &[HEADS, HIDDEN], &device);
        let wb = tensor(2, &[HEADS, HIDDEN], &device);
        let bb = tensor(3, &[HEADS], &device);
        let x = Var::new(tensor(4, &[2, 3, HIDDEN], &device), false);

        let separate = GateProjections::Separate(Box::new(SeparateGates {
            alpha: dense(wa.clone(), None),
            beta: dense(wb.clone(), Some(bb.clone())),
        }));
        let fused = GateProjections::new(dense(wa, None), dense(wb, Some(bb))).unwrap();
        assert!(matches!(fused, GateProjections::Fused { heads: HEADS, .. }));

        let (alpha_sep, beta_sep) = separate.forward(&client, &x).unwrap();
        let (alpha_fused, beta_fused) = fused.forward(&client, &x).unwrap();
        assert_eq!(alpha_fused.shape(), &[2, 3, HEADS]);
        assert_eq!(beta_fused.shape(), &[2, 3, HEADS]);
        assert_eq!(
            alpha_fused.contiguous().unwrap().to_vec::<f32>(),
            alpha_sep.to_vec::<f32>()
        );
        assert_eq!(
            beta_fused.contiguous().unwrap().to_vec::<f32>(),
            beta_sep.to_vec::<f32>()
        );
    }

    #[test]
    fn fused_weight_is_the_two_stacked() {
        let (_client, device) = cpu_setup();
        let wa = tensor(5, &[HEADS, HIDDEN], &device);
        let wb = tensor(6, &[HEADS, HIDDEN], &device);
        let mut want = wa.to_vec::<f32>();
        want.extend(wb.to_vec::<f32>());
        let fused = GateProjections::new(dense(wa, None), dense(wb, None)).unwrap();
        let GateProjections::Fused { linear, .. } = fused else {
            panic!("dense pair must fuse");
        };
        assert_eq!(linear.weight().tensor().shape(), &[2 * HEADS, HIDDEN]);
        assert_eq!(linear.weight().tensor().to_vec::<f32>(), want);
        assert!(linear.bias().is_none());
    }
}
