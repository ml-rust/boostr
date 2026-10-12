//! [`MaybeRotatedLinear::forward_batch`]: several projections that share
//! one input, with any shared Hadamard rotation run once and every
//! unrotated base in one fused batched matmul.

use super::layer::MaybeRotatedLinear;
use crate::error::{Error, Result};
use crate::nn::linear::maybe_quant_linear::MaybeQuantLinear;
use crate::nn::linear::rotated_linear::RotatedLinear;
use crate::quant::QuantTensor;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{BinaryOps, FwhtOps, MatmulOps, ScalarOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime<DType = DType>> MaybeRotatedLinear<R> {
    /// Forward several projections that share one input, running any shared
    /// Hadamard rotation ONCE.
    ///
    /// `layers` is partitioned into unrotated members (`Plain` and `Lora`)
    /// and `Rotated` members, order preserved for the returned `Vec`. The
    /// unrotated bases run through one [`MaybeQuantLinear::forward_batch`]
    /// call over `input`. Each `Lora` member then adds its adapter to its
    /// base output ([`LoraLinear::add_adapter`](crate::nn::LoraLinear::add_adapter)),
    /// the same ops its own [`Self::forward`] runs after the base. Every
    /// `Rotated` member must carry the same rotation as the first one
    /// ([`HadamardRotation::same_rotation_as`](crate::nn::HadamardRotation::same_rotation_as))
    /// — the rotation then runs ONCE. When every `Rotated` base is
    /// block-quantized without a bias the rotation goes to
    /// `quant_matmul_batch_rotated`, which folds it into the activation
    /// quantization where the backend has that kernel; otherwise it runs as
    /// [`HadamardRotation::forward`](crate::nn::HadamardRotation::forward)
    /// and the bases run through a second [`MaybeQuantLinear::forward_batch`]
    /// call over the rotated activation. A single-member batch works on
    /// every path.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when two `Rotated` members disagree on
    /// rotation (different `block_size`, or different sign storage).
    pub fn forward_batch<C>(layers: &[&Self], client: &C, input: &Var<R>) -> Result<Vec<Var<R>>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + FwhtOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + DequantOps<R> + MatmulOps<R>,
    {
        let mut unrotated: Vec<&MaybeQuantLinear<R>> = Vec::new();
        // `(original index, the Rotated member)` — captured directly while
        // partitioning, so the rotation-agreement check below never needs
        // to re-match an unrotated layer against `Rotated`.
        let mut rotated: Vec<(usize, &RotatedLinear<R>)> = Vec::new();
        for (i, layer) in layers.iter().enumerate() {
            match layer {
                Self::Plain(base) => unrotated.push(base),
                Self::Lora(lora) => unrotated.push(lora.base()),
                Self::Rotated(rotated_layer) => rotated.push((i, rotated_layer.as_ref())),
            }
        }

        let mut unrotated_outs = if unrotated.is_empty() {
            Vec::new()
        } else {
            MaybeQuantLinear::forward_batch(&unrotated, client, input)?
        }
        .into_iter();
        let mut rotated_outs = forward_rotated(&rotated, client, input)?.into_iter();

        // Both groups' outputs keep the order of `layers`, so the n-th
        // unrotated/`Rotated` output lines up with the n-th unrotated/
        // `Rotated` member in one pass over `layers`.
        layers
            .iter()
            .map(|layer| match layer {
                Self::Plain(_) => next_output(&mut unrotated_outs, "unrotated"),
                Self::Lora(lora) => {
                    let base_out = next_output(&mut unrotated_outs, "unrotated")?;
                    lora.add_adapter(client, input, &base_out)
                }
                Self::Rotated(_) => next_output(&mut rotated_outs, "rotated"),
            })
            .collect()
    }
}

/// The next output of one group.
///
/// # Errors
///
/// [`Error::ModelError`] when the group returned fewer outputs than it had
/// members.
fn next_output<R: Runtime>(outs: &mut impl Iterator<Item = Var<R>>, group: &str) -> Result<Var<R>> {
    outs.next().ok_or_else(|| Error::ModelError {
        reason: format!("forward_batch: {group} output count did not match {group} input count"),
    })
}

/// The `Rotated` members of a batch over one shared rotation, in order.
///
/// # Errors
///
/// [`Error::ModelError`] when two members disagree on rotation.
fn forward_rotated<R, C>(
    rotated: &[(usize, &RotatedLinear<R>)],
    client: &C,
    input: &Var<R>,
) -> Result<Vec<Var<R>>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + QuantMatmulOps<R>
        + BinaryOps<R>
        + TypeConversionOps<R>
        + FwhtOps<R>,
    R::Client: TensorOps<R> + DequantOps<R> + MatmulOps<R>,
{
    let [(first_idx, first), rest @ ..] = rotated else {
        return Ok(Vec::new());
    };
    let first_rotation = first.rotation();
    for (i, member) in rest {
        let rotation = member.rotation();
        if !rotation.same_rotation_as(first_rotation) {
            return Err(Error::ModelError {
                reason: format!(
                    "forward_batch: Rotated layer at index {i} does not share layer \
                     {first_idx}'s rotation (block_size {} vs {}, signs storage {})",
                    rotation.block_size(),
                    first_rotation.block_size(),
                    if rotation.signs_ptr() == first_rotation.signs_ptr() {
                        "matched"
                    } else {
                        "did not match"
                    }
                ),
            });
        }
    }
    let quant_weights: Vec<&QuantTensor<R>> = rotated
        .iter()
        .filter_map(|(_, member)| member.quant_weight_without_bias())
        .collect();
    if quant_weights.len() == rotated.len() {
        // Every base is block-quantized without a bias: the rotation goes
        // with the batch, folded into the activation quantization where the
        // backend has that kernel.
        let rotation = first_rotation.rotation_for(input.tensor())?;
        Ok(client
            .quant_matmul_batch_rotated(input.tensor(), &rotation, &quant_weights)?
            .into_iter()
            .map(|t| Var::new(t, false))
            .collect())
    } else {
        let rotated_input = first_rotation.forward(client, input.tensor())?;
        let rotated_input = Var::new(rotated_input, false);
        let bases: Vec<&MaybeQuantLinear<R>> =
            rotated.iter().map(|(_, member)| member.base()).collect();
        MaybeQuantLinear::forward_batch(&bases, client, &rotated_input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::hadamard::HadamardRotation;
    use crate::nn::linear::dense::Linear;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    const IN: usize = 16;
    const BLOCK: usize = 8;

    fn plain_base(out: usize, seed: f32, device: &CpuDevice) -> MaybeQuantLinear<CpuRuntime> {
        let data: Vec<f32> = (0..out * IN).map(|i| ((i as f32) * seed).sin()).collect();
        let weight = Tensor::<CpuRuntime>::from_slice(&data, &[out, IN], device).unwrap();
        MaybeQuantLinear::Standard(Linear::new(weight, None, false))
    }

    fn rotation(signs: &[i8], device: &CpuDevice) -> HadamardRotation<CpuRuntime> {
        HadamardRotation::<CpuRuntime>::new(BLOCK, Some(signs), DType::F32, device).unwrap()
    }

    fn rotated(
        out: usize,
        seed: f32,
        rot: HadamardRotation<CpuRuntime>,
        device: &CpuDevice,
    ) -> MaybeRotatedLinear<CpuRuntime> {
        RotatedLinear::new(plain_base(out, seed, device), rot)
            .unwrap()
            .into()
    }

    fn plain(out: usize, seed: f32, device: &CpuDevice) -> MaybeRotatedLinear<CpuRuntime> {
        MaybeRotatedLinear::Plain(plain_base(out, seed, device))
    }

    fn adapted(out: usize, seed: f32, device: &CpuDevice) -> MaybeRotatedLinear<CpuRuntime> {
        let rank = 2;
        let a: Vec<f32> = (0..rank * IN)
            .map(|i| ((i as f32) * 0.7).cos() * 0.3)
            .collect();
        let b: Vec<f32> = (0..out * rank)
            .map(|i| ((i as f32) * 1.3).sin() * 0.3)
            .collect();
        let mut layer = plain(out, seed, device);
        layer
            .attach_adapter(
                Tensor::<CpuRuntime>::from_slice(&a, &[rank, IN], device).unwrap(),
                Tensor::<CpuRuntime>::from_slice(&b, &[out, rank], device).unwrap(),
                1.5,
            )
            .unwrap();
        layer
    }

    fn input(device: &CpuDevice) -> Var<CpuRuntime> {
        let x_data: Vec<f32> = (0..3 * IN).map(|i| (i as f32) * 0.05 - 0.3).collect();
        Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[3, IN], device).unwrap(),
            false,
        )
    }

    /// Batched outputs equal each layer's own forward, in order.
    fn assert_matches_per_layer(layers: &[&MaybeRotatedLinear<CpuRuntime>]) {
        let (client, device) = cpu_setup();
        let x = input(&device);
        let batched = MaybeRotatedLinear::forward_batch(layers, &client, &x).unwrap();
        assert_eq!(batched.len(), layers.len());
        for (i, (layer, out)) in layers.iter().zip(&batched).enumerate() {
            let solo = layer.forward(&client, &x).unwrap();
            assert_eq!(
                out.tensor().to_vec::<f32>(),
                solo.tensor().to_vec::<f32>(),
                "layer {i}"
            );
        }
    }

    #[test]
    fn two_rotated_layers_sharing_one_rotation_match_per_layer_forward() {
        let (_, device) = cpu_setup();
        let signs: Vec<i8> = (0..IN).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
        let rot = rotation(&signs, &device);
        let a = rotated(4, 0.01, rot.clone(), &device);
        let b = rotated(5, 0.02, rot, &device);
        assert_matches_per_layer(&[&a, &b]);
    }

    #[test]
    fn mixed_rotated_and_plain_match_per_layer_forward_order_preserved() {
        let (_, device) = cpu_setup();
        let signs: Vec<i8> = (0..IN).map(|i| if i % 2 == 0 { -1 } else { 1 }).collect();
        let rot = rotation(&signs, &device);
        let a = rotated(3, 0.01, rot.clone(), &device);
        let p = plain(4, 0.02, &device);
        let b = rotated(5, 0.03, rot, &device);
        assert_matches_per_layer(&[&a, &p, &b]);
    }

    /// An adapted member between a plain and a rotated one: every output
    /// equals that layer's own forward, in order.
    #[test]
    fn adapted_member_matches_per_layer_forward_order_preserved() {
        let (_, device) = cpu_setup();
        let signs: Vec<i8> = (0..IN).map(|i| if i % 2 == 0 { -1 } else { 1 }).collect();
        let p = plain(4, 0.02, &device);
        let l = adapted(3, 0.05, &device);
        let r = rotated(5, 0.03, rotation(&signs, &device), &device);
        let l2 = adapted(2, 0.07, &device);
        assert_matches_per_layer(&[&p, &l, &r, &l2]);
        assert_matches_per_layer(&[&l]);
    }

    #[test]
    fn rejects_rotated_layers_with_different_signs() {
        let (client, device) = cpu_setup();
        let signs_a: Vec<i8> = (0..IN).map(|i| if i % 2 == 0 { -1 } else { 1 }).collect();
        let signs_b: Vec<i8> = (0..IN).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let a = rotated(3, 0.01, rotation(&signs_a, &device), &device);
        let b = rotated(4, 0.02, rotation(&signs_b, &device), &device);
        let x = input(&device);
        assert!(MaybeRotatedLinear::forward_batch(&[&a, &b], &client, &x).is_err());
    }
}
