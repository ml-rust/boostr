//! Encode (`bound` / `quantize_codes` / `quantize`) and decode
//! (`indices_to_codes`) paths of [`Fsq`].
//!
//! # Math (mirrors the reference exactly)
//!
//! For each of the `d = levels.len()` scalar dimensions, with `eps = 1e-3`:
//!
//! ```text
//! half_l     = (level - 1) * (1 + eps) / 2
//! offset     = 0.5 if level is even else 0.0
//! shift      = atanh(offset / half_l)          // 0 for odd levels
//! bound(z)   = tanh(z + shift) * half_l - offset          // FSQ.bound
//! half_width = level // 2
//! quantize(z) = round_ste(bound(z)) / half_width          // FSQ.quantize
//! ```
//!
//! Note the two are SEPARATE functions in the reference implementation (`Fsq::bound` and
//! `Fsq::quantize_codes`) and `bound` is NOT idempotent — its output range is
//! asymmetric, `(-half_l - offset, half_l - offset)`. `ResidualFsq` applies it
//! twice on purpose. Do not fuse them back together.
//!
//! `round_ste(x) = x + (round(x) - x).detach()`: forward value is `round(x)`,
//! backward gradient is the identity (straight-through estimator).
//!
//! Indices use a mixed-radix (cumulative-product) basis: `basis[0] = 1`,
//! `basis[i] = basis[i-1] * levels[i-1]`.
//!
//! ```text
//! codes_to_indices(code)   = round(sum((code * half_width + half_width) * basis))
//! indices_to_level_indices = (indices // basis) % levels
//! indices_to_codes(index)  = (level_indices - half_width) / half_width
//! ```

use crate::error::{Error, Result};
use crate::nn::fsq::codes::var_passthrough;
use numr::autograd::{Var, var_add, var_div, var_mul, var_sub, var_tanh};
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::Fsq;

impl<R: Runtime<DType = DType>> Fsq<R> {
    /// lucidrains/vector-quantize-pytorch's `FSQ.bound`: `tanh(z + shift) * half_l - offset`.
    ///
    /// Squashes `z` into the (asymmetric) interval
    /// `(-half_l - offset, half_l - offset)` per dimension. NO rounding, NO
    /// division by `half_width` — that is `quantize_codes`, a strictly
    /// different function.
    ///
    /// The asymmetry is why this is not idempotent: `bound(bound(z)) !=
    /// bound(z)`. [`ResidualFsq`](crate::nn::fsq::residual::ResidualFsq)
    /// relies on applying it twice (once to seed the residual, once inside
    /// `quantize_codes`) exactly as lucidrains/vector-quantize-pytorch's
    /// `ResidualFSQ.forward` does.
    ///
    /// Every step is a tracked `var_*` op, so gradients reach `z`.
    pub(crate) fn bound<C>(&self, z: &Var<R>, client: &C) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        // Genuine constants (precomputed from `levels`, independent of any
        // trainable parameter) — `Var::new(_, false)` is correct here.
        let shift = Var::new(self.shift.clone(), false);
        let half_l = Var::new(self.half_l.clone(), false);
        let offset = Var::new(self.offset.clone(), false);

        let shifted = var_add(z, &shift, client).map_err(Error::Numr)?;
        let tanh_val = var_tanh(&shifted, client).map_err(Error::Numr)?;
        let scaled = var_mul(&tanh_val, &half_l, client).map_err(Error::Numr)?;
        var_sub(&scaled, &offset, client).map_err(Error::Numr)
    }

    /// lucidrains/vector-quantize-pytorch's `FSQ.quantize`: `round_ste(bound(z)) / half_width`.
    ///
    /// Snaps `z` onto the FSQ grid, normalized to `[-1, 1]`-ish per dimension.
    /// Straight-through: forward value is the rounded grid point, backward
    /// gradient is `d(bound(z)) / dz / half_width` — every step here is a
    /// tracked `var_*` op, so gradients reach `z`.
    ///
    /// Named `quantize_codes` rather than `quantize` because
    /// [`Fsq::quantize`](Self::quantize) is the public encode entry point
    /// (projections + index packing) that wraps it.
    fn quantize_codes<C>(&self, z: &Var<R>, client: &C) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let half_width = Var::new(self.half_width.clone(), false);
        let bounded = self.bound(z, client)?;
        let rounded = self.round_ste(&bounded, client)?;
        var_div(&rounded, &half_width, client).map_err(Error::Numr)
    }

    /// Encode: quantize `z` and return `(codes, indices)`.
    ///
    /// `z`: `[..., input_dim]`. `codes`: `[..., input_dim]` (post `project_out`
    /// if projection is configured, else `[..., codebook_dim]`). `indices`:
    /// `[...]`, `DType::I32`.
    ///
    /// The straight-through estimator makes `codes` differentiable w.r.t. `z`
    /// (and w.r.t. `project_in`/`project_out` weights, if trainable); `indices`
    /// is a discrete byproduct and carries no gradient.
    pub fn quantize<C>(&self, client: &C, z: &Var<R>) -> Result<(Var<R>, Tensor<R>)>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        match z.shape().last().copied() {
            Some(last) if last == self.config.input_dim => {}
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "z",
                    reason: format!(
                        "expected last dimension {}, got shape {:?}",
                        self.config.input_dim,
                        z.shape()
                    ),
                });
            }
        }

        let projected = match &self.project_in {
            Some(linear) => linear.forward(client, z)?,
            // `Var::clone()` mints a fresh autograd id and would silently
            // disconnect this leaf from the caller's `z.id()` (the id the
            // caller looks gradients up by after `backward`). Use the
            // identity-preserving passthrough instead.
            None => var_passthrough(z),
        };

        let bounded = self.quantize_codes(&projected, client)?;
        let indices = self.codes_to_indices(client, bounded.tensor())?;

        let codes = match &self.project_out {
            Some(linear) => linear.forward(client, &bounded)?,
            None => bounded,
        };

        Ok((codes, indices))
    }

    /// Decode: `indices` (`[...]`, integer dtype) -> `codes` (`[...,
    /// input_dim]`).
    ///
    /// This is the decode path a decoder-only pipeline needs: reconstructs the
    /// normalized grid codes via mixed-radix unpack, then applies
    /// `project_out` if configured. `indices` carries no gradient (discrete),
    /// so the decoded codes are wrapped as a non-differentiable leaf before any
    /// (potentially trainable) `project_out` is applied.
    pub fn indices_to_codes<C>(&self, client: &C, indices: &Tensor<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R>,
    {
        let codes = self.decode_indices(client, indices)?;
        let codes = Var::new(codes, false);
        match &self.project_out {
            Some(linear) => linear.forward(client, &codes),
            None => Ok(codes),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::fsq::config::FsqConfig;
    use crate::nn::linear::Linear;
    use crate::test_utils::cpu_setup;
    use numr::autograd::backward;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn toy_fsq() -> (Fsq<CpuRuntime>, CpuClient, CpuDevice) {
        let (client, device) = cpu_setup();
        let config = FsqConfig::new(vec![4, 4], 2).unwrap();
        let fsq = Fsq::new(config, &device, None, None).unwrap();
        (fsq, client, device)
    }

    // --- grid values, hand-computed from the reference formula -----------

    /// For levels=[4,4]: half_width = 4 // 2 = 2, so
    /// `code = (level_index - 2) / 2` for level_index in {0,1,2,3} gives the
    /// grid {-1.0, -0.5, 0.0, 0.5} — every combination of the two dims should
    /// land exactly there. This is `indices_to_codes` alone (project_out is
    /// None for the toy config), so it's a pure check of the mixed-radix
    /// unpack + scale/shift math, independent of the tanh-based encode path.
    #[test]
    fn test_decode_grid_values_toy() {
        let (fsq, client, device) = toy_fsq();
        let expected_grid = [-1.0f32, -0.5, 0.0, 0.5];

        for index in 0..16i32 {
            let indices = Tensor::<CpuRuntime>::from_slice(&[index], &[1], &device).unwrap();
            let codes = fsq.indices_to_codes(&client, &indices).unwrap();
            let data: Vec<f32> = codes.tensor().contiguous().unwrap().to_vec();

            let dim0 = index % 4;
            let dim1 = index / 4;
            assert!(
                (data[0] - expected_grid[dim0 as usize]).abs() < 1e-5,
                "index {index}: dim0 = {}, expected {}",
                data[0],
                expected_grid[dim0 as usize]
            );
            assert!(
                (data[1] - expected_grid[dim1 as usize]).abs() < 1e-5,
                "index {index}: dim1 = {}, expected {}",
                data[1],
                expected_grid[dim1 as usize]
            );
        }
    }

    /// The quantized (encoded) codes must also land on the same discrete grid
    /// as the decode path, for saturating (large-magnitude) inputs where the
    /// tanh bound is unambiguous: driving z very negative saturates
    /// tanh(z+shift) -> -1, giving bounded_z -> -half_l - offset = -2.0015,
    /// which rounds to -2 -> code -1.0. Driving z very positive saturates
    /// tanh -> +1, giving bounded_z -> half_l - offset = 1.0015, which rounds
    /// to 1 -> code 0.5. z = 0 falls in between and must land on some point of
    /// the same 4-point grid.
    #[test]
    fn test_encode_grid_values_toy() {
        let (fsq, client, device) = toy_fsq();
        let allowed = [-1.0f32, -0.5, 0.0, 0.5];

        for &z_val in &[-5.0f32, 0.0, 5.0] {
            let z = Var::new(
                Tensor::<CpuRuntime>::from_slice(&[z_val, z_val], &[1, 2], &device).unwrap(),
                false,
            );
            let (codes, _) = fsq.quantize(&client, &z).unwrap();
            let data: Vec<f32> = codes.tensor().contiguous().unwrap().to_vec();
            for &v in &data {
                assert!(
                    allowed.iter().any(|&g| (g - v).abs() < 1e-4),
                    "z={z_val} produced off-grid code {v}"
                );
            }
        }

        // Saturating extremes hit the exact boundary grid points.
        let z_neg = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[-5.0f32, -5.0], &[1, 2], &device).unwrap(),
            false,
        );
        let (codes_neg, _) = fsq.quantize(&client, &z_neg).unwrap();
        let data_neg: Vec<f32> = codes_neg.tensor().contiguous().unwrap().to_vec();
        assert!((data_neg[0] - (-1.0)).abs() < 1e-4);
        assert!((data_neg[1] - (-1.0)).abs() < 1e-4);

        let z_pos = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[5.0f32, 5.0], &[1, 2], &device).unwrap(),
            false,
        );
        let (codes_pos, _) = fsq.quantize(&client, &z_pos).unwrap();
        let data_pos: Vec<f32> = codes_pos.tensor().contiguous().unwrap().to_vec();
        assert!((data_pos[0] - 0.5).abs() < 1e-4);
        assert!((data_pos[1] - 0.5).abs() < 1e-4);
    }

    // --- straight-through gradient -----------------------------------------

    /// Backward through `quantize` must reach `z` with a non-zero gradient.
    /// Uses small, asymmetric (non-zero, unequal-magnitude, mixed-sign) inputs
    /// so the tanh derivative is genuinely non-zero and a coincidental zero
    /// can't produce a false pass.
    #[test]
    fn test_straight_through_gradient_nonzero() {
        let (fsq, client, device) = toy_fsq();
        let z = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.31f32, -0.72], &[1, 2], &device).unwrap(),
            true,
        );

        let (codes, _) = fsq.quantize(&client, &z).unwrap();
        let loss = numr::autograd::var_sum(&codes, &[0, 1], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let grad = grads
            .get(z.id())
            .expect("straight-through estimator must propagate gradient to z");
        let grad_data: Vec<f32> = grad.contiguous().unwrap().to_vec();
        assert!(
            grad_data.iter().all(|&g| g != 0.0),
            "expected non-zero gradient on every element, got {grad_data:?}"
        );
    }

    // --- projection wiring ---------------------------------------------------

    #[test]
    fn test_projection_roundtrips_shape() {
        let (client, device) = cpu_setup();
        // input_dim=5, codebook_dim=2 (mirrors NeuCodec's dim != levels.len()).
        let config = FsqConfig::new(vec![4, 4], 5).unwrap();

        let w_in = Tensor::<CpuRuntime>::from_slice(
            &[0.1f32; 10], // [codebook_dim=2, input_dim=5]
            &[2, 5],
            &device,
        )
        .unwrap();
        let w_out = Tensor::<CpuRuntime>::from_slice(
            &[0.2f32; 10], // [input_dim=5, codebook_dim=2]
            &[5, 2],
            &device,
        )
        .unwrap();
        let project_in = Some(Linear::new(w_in, None, false));
        let project_out = Some(Linear::new(w_out, None, false));

        let fsq = Fsq::new(config, &device, project_in, project_out).unwrap();

        let z = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.5f32; 5], &[1, 5], &device).unwrap(),
            false,
        );
        let (codes, indices) = fsq.quantize(&client, &z).unwrap();
        assert_eq!(codes.shape(), &[1, 5]);
        assert_eq!(indices.shape(), &[1]);

        let decoded = fsq.indices_to_codes(&client, &indices).unwrap();
        assert_eq!(decoded.shape(), &[1, 5]);
    }
}
