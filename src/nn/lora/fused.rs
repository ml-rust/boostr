//! One [`LoraLinear`] over a fused base whose output rows concatenate
//! several projections that read the same input, each with its own
//! adapter or none.
//!
//! For parts `i = 1..n` with factors `A_i` `[r_i, in]`, `B_i`
//! `[out_i, r_i]` and scaling `s_i`, the fused adapter is
//!
//! ```text
//! A = [A_1; A_2; ...]                     [sum r_i, in]
//! B = blockdiag(s_1 B_1, s_2 B_2, ...)    [sum out_i, sum r_i]
//! scaling = 1
//! ```
//!
//! so output rows of part `i` read only the rank columns of `A_i`, and
//! `x A^T B^T` restricted to part `i` is `s_i (x A_i^T) B_i^T`. A part with
//! no adapter contributes zero rows to `B` and no rows to `A`.

use crate::error::{Error, Result};
use crate::nn::MaybeQuantLinear;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{ScalarOps, ShapeOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::LoraLinear;
use super::scaled::{base_dims, check_a};

/// One part's loaded adapter factors.
pub struct LoraFactors<R: Runtime> {
    /// `[rank, in_features]`.
    pub lora_a: Tensor<R>,
    /// `[out_features, rank]`, `out_features` being the part's own rows.
    pub lora_b: Tensor<R>,
    /// The part's scaling, folded into its block of the fused `lora_b`.
    pub scaling: f32,
}

/// One part of a fused base: its output row count, and its adapter if any.
pub struct FusedLoraPart<R: Runtime> {
    /// Rows of the fused base this part owns, in part order.
    pub out_features: usize,
    /// `None` for a part without an adapter.
    pub factors: Option<LoraFactors<R>>,
}

impl<R: Runtime<DType = DType>> LoraLinear<R> {
    /// One adapter over `base` `[sum out_i, in_features]` whose output rows
    /// are `parts` concatenated in order.
    ///
    /// Requires every part's `lora_a` to be `[r_i, in_features]` with
    /// `r_i > 0`, its `lora_b` to be `[out_i, r_i]`, and every factor to
    /// share one dtype (the base weight's dtype when the base is dense).
    /// The result has `lora_a` `[sum r_i, in_features]`, a block-diagonal
    /// `lora_b` `[sum out_i, sum r_i]` with each part's scaling folded in,
    /// and scaling `1`. It equals the parts' separate adapters with their
    /// outputs concatenated, up to float rounding from the folded scaling.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when the part rows do not sum to the base's
    /// `out_features`, no part has an adapter, a factor shape or dtype is
    /// wrong, or a scaling is not finite. numr errors from the device ops
    /// propagate.
    pub fn fused_block_diagonal<C>(
        client: &C,
        base: impl Into<MaybeQuantLinear<R>>,
        parts: Vec<FusedLoraPart<R>>,
        trainable: bool,
    ) -> Result<Self>
    where
        C: RuntimeClient<R> + ScalarOps<R> + ShapeOps<R>,
    {
        let base = base.into();
        let (out_features, in_features) = base_dims(&base)?;
        let rows: usize = parts.iter().map(|part| part.out_features).sum();
        if rows != out_features {
            return Err(Error::ModelError {
                reason: format!(
                    "fused LoRA parts own {rows} rows but the base has {out_features} output rows"
                ),
            });
        }
        check_parts(&base, &parts, in_features)?;
        let total_rank: usize = parts
            .iter()
            .filter_map(|part| part.factors.as_ref())
            .map(|factors| factors.lora_a.shape()[0])
            .sum();

        let mut a_parts: Vec<&Tensor<R>> = Vec::new();
        let mut b_blocks: Vec<Tensor<R>> = Vec::new();
        let mut rank_before = 0usize;
        let mut zero_rows_before = 0usize;
        for part in &parts {
            let Some(factors) = &part.factors else {
                zero_rows_before += part.out_features;
                continue;
            };
            let rank = factors.lora_a.shape()[0];
            let scaled = client.mul_scalar(&factors.lora_b, f64::from(factors.scaling))?;
            // `pad` takes `(before, after)` pairs from the last dim: rank
            // columns first, then output rows.
            let block = client.pad(
                &scaled,
                &[
                    rank_before,
                    total_rank - rank_before - rank,
                    zero_rows_before,
                    0,
                ],
                0.0,
            )?;
            a_parts.push(&factors.lora_a);
            b_blocks.push(block);
            rank_before += rank;
            zero_rows_before = 0;
        }
        if zero_rows_before > 0 {
            let last = b_blocks.pop().ok_or_else(|| Error::ModelError {
                reason: "fused LoRA: no part carries an adapter".to_string(),
            })?;
            b_blocks.push(client.pad(&last, &[0, 0, 0, zero_rows_before], 0.0)?);
        }

        let lora_a = client.cat(&a_parts, 0)?;
        let block_refs: Vec<&Tensor<R>> = b_blocks.iter().collect();
        let lora_b = client.cat(&block_refs, 0)?;
        Ok(Self {
            base,
            lora_a: Var::new(lora_a, trainable),
            lora_b: Var::new(lora_b, trainable),
            scaling: 1.0,
        })
    }
}

/// Check every adapted part's factors against `in_features` and one
/// shared dtype.
///
/// # Errors
///
/// [`Error::ModelError`] when no part has an adapter, or a shape, dtype or
/// scaling check fails.
fn check_parts<R: Runtime<DType = DType>>(
    base: &MaybeQuantLinear<R>,
    parts: &[FusedLoraPart<R>],
    in_features: usize,
) -> Result<()> {
    let base_dtype = base.weight().map(|weight| weight.tensor().dtype());
    let mut dtype = None;
    for (index, part) in parts.iter().enumerate() {
        let Some(factors) = &part.factors else {
            continue;
        };
        check_a(&factors.lora_a, in_features)?;
        let rank = factors.lora_a.shape()[0];
        if factors.lora_b.shape() != [part.out_features, rank] {
            return Err(Error::ModelError {
                reason: format!(
                    "fused LoRA part {index}: lora_b must be [{}, {rank}], got {:?}",
                    part.out_features,
                    factors.lora_b.shape()
                ),
            });
        }
        if !factors.scaling.is_finite() {
            return Err(Error::ModelError {
                reason: format!(
                    "fused LoRA part {index}: scaling {} is not finite",
                    factors.scaling
                ),
            });
        }
        let expected = *dtype.get_or_insert(base_dtype.unwrap_or(factors.lora_a.dtype()));
        for (name, got) in [
            ("lora_a", factors.lora_a.dtype()),
            ("lora_b", factors.lora_b.dtype()),
        ] {
            if got != expected {
                return Err(Error::ModelError {
                    reason: format!(
                        "fused LoRA part {index}: {name} is {got:?}, expected {expected:?} (the \
                         dense base or first adapted part's dtype)"
                    ),
                });
            }
        }
    }
    match dtype {
        Some(_) => Ok(()),
        None => Err(Error::ModelError {
            reason: "fused LoRA: no part carries an adapter".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    const IN: usize = 12;

    fn tensor(device: &CpuDevice, shape: &[usize], seed: f32) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + 1.0) * seed).sin() * 0.2)
            .collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
    }

    fn dense(w: &Tensor<CpuRuntime>) -> Linear<CpuRuntime> {
        Linear::new(w.clone(), None, false)
    }

    /// Two adapted parts and one bare part between them: the fused adapter
    /// equals the separate projections with their outputs concatenated.
    #[test]
    fn fused_equals_separate_projections_concatenated() {
        let (client, device) = cpu_setup();
        let (out1, out2, out3) = (3usize, 2usize, 4usize);
        let (r1, r3) = (2usize, 3usize);
        let w1 = tensor(&device, &[out1, IN], 0.11);
        let w2 = tensor(&device, &[out2, IN], 0.23);
        let w3 = tensor(&device, &[out3, IN], 0.37);
        let a1 = tensor(&device, &[r1, IN], 0.41);
        let b1 = tensor(&device, &[out1, r1], 0.53);
        let a3 = tensor(&device, &[r3, IN], 0.67);
        let b3 = tensor(&device, &[out3, r3], 0.79);
        let (s1, s3) = (2.0f32, 0.5f32);

        let fused_w = Tensor::cat(&[&w1, &w2, &w3], 0).unwrap();
        let fused = LoraLinear::fused_block_diagonal(
            &client,
            dense(&fused_w),
            vec![
                FusedLoraPart {
                    out_features: out1,
                    factors: Some(LoraFactors {
                        lora_a: a1.clone(),
                        lora_b: b1.clone(),
                        scaling: s1,
                    }),
                },
                FusedLoraPart {
                    out_features: out2,
                    factors: None,
                },
                FusedLoraPart {
                    out_features: out3,
                    factors: Some(LoraFactors {
                        lora_a: a3.clone(),
                        lora_b: b3.clone(),
                        scaling: s3,
                    }),
                },
            ],
            false,
        )
        .unwrap();
        assert_eq!(fused.lora_a().shape(), &[r1 + r3, IN]);
        assert_eq!(fused.lora_b().shape(), &[out1 + out2 + out3, r1 + r3]);

        let part1 = LoraLinear::from_scaled_weights(dense(&w1), a1, b1, s1, false).unwrap();
        let part3 = LoraLinear::from_scaled_weights(dense(&w3), a3, b3, s3, false).unwrap();
        let x = Var::new(tensor(&device, &[5, IN], 0.91), false);
        let y1 = part1.forward(&client, &x).unwrap();
        let y2 = dense(&w2).forward(&client, &x).unwrap();
        let y3 = part3.forward(&client, &x).unwrap();
        let want: Vec<f32> = Tensor::cat(&[y1.tensor(), y2.tensor(), y3.tensor()], 1)
            .unwrap()
            .to_vec();
        let got: Vec<f32> = fused.forward(&client, &x).unwrap().tensor().to_vec();

        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-5,
                "element {i}: fused {g} vs separate {w}"
            );
        }
    }

    #[test]
    fn rows_must_sum_to_base_out_features() {
        let (client, device) = cpu_setup();
        let w = tensor(&device, &[5, IN], 0.1);
        let result = LoraLinear::fused_block_diagonal(
            &client,
            dense(&w),
            vec![FusedLoraPart {
                out_features: 4,
                factors: Some(LoraFactors {
                    lora_a: tensor(&device, &[2, IN], 0.2),
                    lora_b: tensor(&device, &[4, 2], 0.3),
                    scaling: 1.0,
                }),
            }],
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn no_adapted_part_is_rejected() {
        let (client, device) = cpu_setup();
        let w = tensor(&device, &[5, IN], 0.1);
        let result = LoraLinear::fused_block_diagonal(
            &client,
            dense(&w),
            vec![
                FusedLoraPart {
                    out_features: 2,
                    factors: None,
                },
                FusedLoraPart {
                    out_features: 3,
                    factors: None,
                },
            ],
            false,
        );
        assert!(result.is_err());
    }
}
