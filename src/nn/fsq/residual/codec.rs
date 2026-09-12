//! `encode` / `decode` paths of [`ResidualFsq`].
//!
//! # The double-bound trap — do NOT "simplify" this away
//!
//! lucidrains/vector-quantize-pytorch's forward is:
//!
//! ```python
//! x = self.project_in(x)
//! residual = first(self.layers).bound(x)          # PURE bound: no round, no divide
//! for layer, scale in zip(self.layers, self.scales):
//!     quantized, indices = layer(residual / scale)  # FSQ.quantize -> round_ste(bound(z)) / half_width
//!     quantized = quantized * scale
//!     residual = residual - quantized.detach()
//!     quantized_out = quantized_out + quantized
//! quantized_out = self.project_out(quantized_out)
//! ```
//!
//! `bound` is therefore applied TWICE before the first rounding: once to seed
//! `residual`, and again inside `FSQ.quantize`. It looks redundant. It is not:
//! `bound(z) = tanh(z + shift) * half_l - offset` has the ASYMMETRIC output
//! range `(-half_l - offset, half_l - offset)`, so it is not idempotent and
//! `bound(bound(z)) != bound(z)` wherever `offset != 0` (i.e. every even level,
//! which includes NeuCodec's `levels = [4; 8]`).
//!
//! Measured against the real NeuCodec checkpoint, collapsing this to a single
//! `bound` changes **43.75% of the emitted indices**. Anyone tempted to delete
//! the pre-`bound` on line "seed the residual" below is introducing that bug.
//!
//! The decode path ([`ResidualFsq::decode`], lucidrains/vector-quantize-pytorch's `get_output_from_indices`)
//! has no such subtlety: per-quantizer codebook lookup, scale, sum, `project_out`.

use crate::error::{Error, Result};
use crate::nn::fsq::codes::var_passthrough;
use numr::autograd::{Var, var_add, var_div, var_mul, var_sub};
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::ResidualFsq;

impl<R: Runtime<DType = DType>> ResidualFsq<R> {
    /// Encode `x` (`[..., dim]`) into `(codes, indices)`.
    ///
    /// `codes`: `[..., dim]` (the summed, projected reconstruction).
    /// `indices`: `[..., num_quantizers]`, `DType::I32`.
    ///
    /// Follows lucidrains/vector-quantize-pytorch's `ResidualFSQ.forward` step for step, including the
    /// pre-`bound` that seeds `residual` — see this module's docs for why that
    /// second bound is load-bearing rather than redundant.
    pub fn encode<C>(&self, client: &C, x: &Var<R>) -> Result<(Var<R>, Tensor<R>)>
    where
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        match x.shape().last().copied() {
            Some(last) if last == self.config.dim => {}
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "x",
                    reason: format!(
                        "expected last dimension {}, got shape {:?}",
                        self.config.dim,
                        x.shape()
                    ),
                });
            }
        }

        let projected = match &self.project_in {
            Some(linear) => linear.forward(client, x)?,
            // Not `Var::clone()`: that mints a fresh TensorId and would orphan
            // the caller's `x.id()` from the gradient graph.
            None => var_passthrough(x),
        };

        let first = self.layers.first().ok_or_else(|| Error::ModelError {
            reason: "residual FSQ has no layers".to_string(),
        })?;
        // Seed the residual with the PURE bound. Deleting this line (or reusing
        // the already-bounded value from inside the loop) silently changes 43.75%
        // of NeuCodec's indices — `bound` is not idempotent.
        let mut residual = first.bound(&projected, client)?;

        let mut quantized_out: Option<Var<R>> = None;
        let mut all_indices: Vec<Tensor<R>> = Vec::with_capacity(self.layers.len());

        for (index, layer) in self.layers.iter().enumerate() {
            let scale_tensor = self.scales.get(index).ok_or_else(|| Error::ModelError {
                reason: format!("missing scale for quantizer {index}"),
            })?;
            // Constant derived from `levels` alone — a legitimate non-tracked leaf.
            let scale = Var::new(scale_tensor.clone(), false);

            let scaled = var_div(&residual, &scale, client).map_err(Error::Numr)?;
            let (quantized, indices) = layer.quantize(client, &scaled)?;
            let quantized = var_mul(&quantized, &scale, client).map_err(Error::Numr)?;

            // `quantized.detach()`: same pattern `round_ste` uses — wrap the raw
            // tensor as a fresh non-tracked leaf. `Var::clone()` would keep the
            // value but mint a new id while leaving the grad_fn semantics wrong.
            let detached = Var::new(quantized.tensor().clone(), false);
            residual = var_sub(&residual, &detached, client).map_err(Error::Numr)?;

            quantized_out = Some(match quantized_out {
                Some(acc) => var_add(&acc, &quantized, client).map_err(Error::Numr)?,
                None => quantized,
            });
            all_indices.push(indices);
        }

        let quantized_out = quantized_out.ok_or_else(|| Error::ModelError {
            reason: "residual FSQ produced no quantized output".to_string(),
        })?;

        let stack_dim = all_indices
            .first()
            .ok_or_else(|| Error::ModelError {
                reason: "residual FSQ produced no indices".to_string(),
            })?
            .shape()
            .len() as isize;
        let index_refs: Vec<&Tensor<R>> = all_indices.iter().collect();
        let indices = client.stack(&index_refs, stack_dim).map_err(Error::Numr)?;

        let codes = match &self.project_out {
            Some(linear) => linear.forward(client, &quantized_out)?,
            None => quantized_out,
        };
        Ok((codes, indices))
    }

    /// Decode `indices` (`[..., num_quantizers]`, integer dtype) into codes
    /// (`[..., dim]`) — lucidrains/vector-quantize-pytorch's `get_output_from_indices`.
    ///
    /// Per-quantizer codebook lookup, multiply by `scales[i]`, sum over
    /// quantizers, then `project_out`. Indices are discrete, so the summed codes
    /// are a non-differentiable leaf before the (possibly trainable) projection.
    pub fn decode<C>(&self, client: &C, indices: &Tensor<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let rank = indices.shape().len();
        let last_axis = rank.checked_sub(1).ok_or_else(|| Error::InvalidArgument {
            arg: "indices",
            reason: "expected at least one dimension (trailing quantizer axis)".to_string(),
        })?;
        if indices.shape().get(last_axis).copied() != Some(self.config.num_quantizers) {
            return Err(Error::InvalidArgument {
                arg: "indices",
                reason: format!(
                    "expected trailing dimension {} (num_quantizers), got shape {:?}",
                    self.config.num_quantizers,
                    indices.shape()
                ),
            });
        }

        let mut summed: Option<Var<R>> = None;
        for (index, layer) in self.layers.iter().enumerate() {
            let slice = indices
                .narrow(last_axis as isize, index, 1)
                .map_err(Error::Numr)?
                .squeeze(Some(last_axis as isize))
                .contiguous()
                .map_err(Error::Numr)?;
            let codes = layer.indices_to_codes(client, &slice)?;

            let scale_tensor = self.scales.get(index).ok_or_else(|| Error::ModelError {
                reason: format!("missing scale for quantizer {index}"),
            })?;
            let scale = Var::new(scale_tensor.clone(), false);
            let scaled = var_mul(&codes, &scale, client).map_err(Error::Numr)?;

            summed = Some(match summed {
                Some(acc) => var_add(&acc, &scaled, client).map_err(Error::Numr)?,
                None => scaled,
            });
        }

        let summed = summed.ok_or_else(|| Error::ModelError {
            reason: "residual FSQ has no layers".to_string(),
        })?;
        match &self.project_out {
            Some(linear) => linear.forward(client, &summed),
            None => Ok(summed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::ResidualFsqWeights;
    use super::*;
    use crate::nn::fsq::config::{FsqConfig, ResidualFsqConfig};
    use crate::nn::fsq::quantizer::Fsq;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    /// NeuCodec's grid: 8 dims, 4 levels each, no projections (dim == codebook_dim).
    const NEUCODEC_LEVELS: [u32; 8] = [4; 8];

    fn neucodec_residual() -> (ResidualFsq<CpuRuntime>, CpuClient, CpuDevice) {
        let (client, device) = cpu_setup();
        let config = ResidualFsqConfig::new(NEUCODEC_LEVELS.to_vec(), 8, 1).unwrap();
        let layer = Fsq::new(config.layer_config().unwrap(), &device, None, None).unwrap();
        let residual = ResidualFsq::new(
            config,
            ResidualFsqWeights {
                project_in: None,
                project_out: None,
                layers: vec![layer],
            },
            &device,
        )
        .unwrap();
        (residual, client, device)
    }

    /// `z` chosen so `bound(z) = 0.49` (just BELOW the rounding boundary) while
    /// `bound(bound(z)) = 0.527` (just ABOVE it), for `levels = 4`:
    ///
    /// ```text
    /// half_l = 3 * 1.001 / 2 = 1.5015, offset = 0.5, shift = atanh(0.5 / half_l)
    /// bound(0.44542) = 0.49    -> round -> 0 -> level index 2
    /// bound(0.49)    = 0.5267  -> round -> 1 -> level index 3
    /// ```
    ///
    /// So the single-bound and double-bound encodes MUST disagree here.
    const DOUBLE_BOUND_DISCRIMINATOR: f32 = 0.44542;

    // --- the double bound is real, and must never silently regress ------------

    /// `ResidualFsq::encode` applies `bound` twice (once to seed `residual`, once
    /// inside `Fsq::quantize`). A bare `Fsq::quantize` on the same input applies it
    /// once. Because `bound` is not idempotent, the two MUST produce different
    /// indices at [`DOUBLE_BOUND_DISCRIMINATOR`].
    ///
    /// If someone "simplifies" the pre-bound away, this test fails — which is the
    /// entire point: on the real NeuCodec checkpoint that change silently rewrites
    /// 43.75% of emitted indices.
    #[test]
    fn test_encode_applies_double_bound() {
        let (residual, client, device) = neucodec_residual();

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[DOUBLE_BOUND_DISCRIMINATOR; 8], &[1, 8], &device)
                .unwrap(),
            false,
        );

        // Double-bound (ResidualFSQ semantics): [1, num_quantizers = 1].
        let (_, double_bound_indices) = residual.encode(&client, &x).unwrap();
        let double_bound: Vec<i32> = double_bound_indices.contiguous().unwrap().to_vec();

        // Single-bound reference, constructed inline: a bare FSQ layer, which is
        // exactly `round_ste(bound(x)) / half_width` with NO pre-bound.
        let single_layer = Fsq::<CpuRuntime>::new(
            FsqConfig::new(NEUCODEC_LEVELS.to_vec(), 8).unwrap(),
            &device,
            None,
            None,
        )
        .unwrap();
        let (_, single_bound_indices) = single_layer.quantize(&client, &x).unwrap();
        let single_bound: Vec<i32> = single_bound_indices.contiguous().unwrap().to_vec();

        assert_ne!(
            double_bound, single_bound,
            "ResidualFsq::encode collapsed to a single bound — the pre-bound seeding \
             `residual` was removed or made idempotent"
        );
    }

    // --- round trip -----------------------------------------------------------

    /// `decode(indices)` must reproduce `encode`'s codes for a single quantizer
    /// with no projections, so the two paths are directly comparable.
    #[test]
    fn test_decode_round_trips_encode() {
        let (residual, client, device) = neucodec_residual();

        let values: Vec<f32> = (0..16).map(|i| (i as f32) * 0.37 - 3.0).collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&values, &[2, 8], &device).unwrap(),
            false,
        );

        let (codes, indices) = residual.encode(&client, &x).unwrap();
        assert_eq!(codes.shape(), &[2, 8]);
        assert_eq!(indices.shape(), &[2, 1]);

        let decoded = residual.decode(&client, &indices).unwrap();
        assert_eq!(decoded.shape(), &[2, 8]);

        let expected: Vec<f32> = codes.tensor().contiguous().unwrap().to_vec();
        let actual: Vec<f32> = decoded.tensor().contiguous().unwrap().to_vec();
        assert_eq!(expected.len(), actual.len());
        for (index, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (e - a).abs() < 1e-5,
                "element {index}: decode gave {a}, encode gave {e}"
            );
        }
    }
}
