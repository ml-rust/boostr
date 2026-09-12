//! Folding a LoRA adapter into its dense base weight.

use crate::error::{Error, Result};
use crate::nn::{Linear, MaybeQuantLinear};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

use super::LoraLinear;

impl<R: Runtime<DType = DType>> LoraLinear<R> {
    /// Merge the adapter into the base weight, producing a plain `Linear`.
    ///
    /// Computes `W + scaling * (B @ A)`, matching the base weight layout
    /// `[out_features, in_features]` (`lora_b` is `[out, rank]`, `lora_a` is
    /// `[rank, in]`, so `B @ A` is `[out, in]`). The result carries no adapter
    /// and is not part of any gradient path — for export and inference after
    /// training. The base's bias, if any, is carried over unchanged.
    ///
    /// # Errors
    ///
    /// Only the dense (`Standard`) base can be merged. A quantized base
    /// (`Quantized` or `DecomposedQuant`) has no `Var<R>` weight to add the
    /// low-rank delta into — folding the adapter in would require
    /// requantizing the merged result, which this does not do. Keep the
    /// adapter separate (train and serve it alongside the quantized base)
    /// instead of merging.
    pub fn merge_into_base<C>(&self, client: &C) -> Result<Linear<R>>
    where
        C: RuntimeClient<R> + TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
        R::Client: TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
    {
        let base = match &self.base {
            MaybeQuantLinear::Standard(linear) => linear,
            MaybeQuantLinear::Quantized(_) | MaybeQuantLinear::DecomposedQuant(_) => {
                return Err(Error::ModelError {
                    reason: "cannot merge a LoRA adapter into a quantized base — merging would \
                             require requantizing the result; keep the adapter separate instead \
                             of merging"
                        .into(),
                });
            }
        };

        let ba = client
            .matmul(self.lora_b.tensor(), self.lora_a.tensor())
            .map_err(crate::error::Error::Numr)?;
        let scaled = client
            .mul_scalar(&ba, self.scaling as f64)
            .map_err(crate::error::Error::Numr)?;
        let merged_weight = client
            .add(base.weight().tensor(), &scaled)
            .map_err(crate::error::Error::Numr)?;

        let bias = base.bias().map(|b| b.tensor().clone());
        Ok(Linear::new(merged_weight, bias, false))
    }
}

#[cfg(test)]
mod tests {
    use super::super::layer::quantized_lora;
    use super::*;
    use numr::autograd::Var;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    /// Merging into a plain `Linear` must reproduce the adapted forward pass
    /// exactly, and must carry the base bias over unchanged. Uses a non-zero
    /// `lora_b` (via `from_weights`) — with the default zero-init the
    /// equivalence would hold trivially and prove nothing.
    #[test]
    fn test_merge_matches_forward_and_preserves_bias() {
        use crate::test_utils::cpu_setup;

        let (client, device) = cpu_setup();
        let (in_features, out_features, rank) = (3usize, 2usize, 2usize);

        let base_w: Vec<f32> = (0..out_features * in_features)
            .map(|i| (i as f32) * 0.1 - 0.3)
            .collect();
        let bias_v: Vec<f32> = vec![0.05, -0.05];
        let base = Linear::new(
            Tensor::<CpuRuntime>::from_slice(&base_w, &[out_features, in_features], &device)
                .unwrap(),
            Some(Tensor::<CpuRuntime>::from_slice(&bias_v, &[out_features], &device).unwrap()),
            false,
        );

        let a_vals: Vec<f32> = (0..rank * in_features)
            .map(|i| (i as f32) * 0.05 - 0.1)
            .collect();
        // Deliberately non-zero, unlike LoraLinear::new's zero-init.
        let b_vals: Vec<f32> = (0..out_features * rank)
            .map(|i| (i as f32) * 0.07 + 0.02)
            .collect();
        let lora_a =
            Tensor::<CpuRuntime>::from_slice(&a_vals, &[rank, in_features], &device).unwrap();
        let lora_b =
            Tensor::<CpuRuntime>::from_slice(&b_vals, &[out_features, rank], &device).unwrap();
        let lora = LoraLinear::from_weights(base, lora_a, lora_b, 8.0, false);

        let x_vals: Vec<f32> = (0..2 * in_features)
            .map(|i| (i as f32) * 0.2 - 0.4)
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_vals, &[2, in_features], &device).unwrap(),
            false,
        );

        let lora_out = lora.forward(&client, &x).expect("lora forward");
        let merged = lora.merge_into_base(&client).expect("merge");
        let merged_out = merged.forward(&client, &x).expect("merged forward");

        let lora_vals: Vec<f32> = lora_out.tensor().contiguous().expect("contig").to_vec();
        let merged_vals: Vec<f32> = merged_out.tensor().contiguous().expect("contig").to_vec();
        assert_eq!(lora_vals.len(), merged_vals.len());
        for (l, m) in lora_vals.iter().zip(merged_vals.iter()) {
            assert!((l - m).abs() < 1e-5, "lora={l} merged={m}");
        }

        let merged_bias: Vec<f32> = merged.bias().expect("bias preserved").tensor().to_vec();
        assert_eq!(merged_bias, bias_v);
    }

    /// Merging a LoRA adapter into a quantized base is not possible without
    /// requantizing the merged result — `merge_into_base` must error, not panic,
    /// and the error must say so explicitly.
    #[test]
    fn test_merge_into_base_errors_on_quantized_base() {
        use crate::test_utils::cpu_setup;

        let (client, device) = cpu_setup();
        let lora = quantized_lora(&client, &device, 4, 256, 2, 8.0);

        // `Linear<R>` is not `Debug`, so `expect_err` (which needs `T: Debug`)
        // cannot be used on this `Result<Linear<R>, _>`.
        let message = match lora.merge_into_base(&client) {
            Ok(_) => panic!("merging into a quantized base must fail"),
            Err(e) => e.to_string(),
        };
        assert!(
            message.contains("quantiz") && message.contains("requant"),
            "error must name the quantized base and the requantization requirement: {message}"
        );
    }
}
