//! The tracked LoRA forward pass.

use crate::error::Result;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_add, var_matmul, var_mul_scalar, var_transpose};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};

use super::LoraLinear;

impl<R: Runtime<DType = DType>> LoraLinear<R> {
    /// Forward: base(x) + (x @ A^T @ B^T) * scaling
    ///
    /// `base(x)` goes through
    /// [`MaybeQuantLinear::forward`](crate::nn::MaybeQuantLinear::forward), so this works
    /// identically whether the frozen base is dense, block-quantized, or
    /// decomposed — only the adapter path below ever needs a gradient.
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + QuantMatmulOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + BinaryOps<R> + ScalarOps<R> + DequantOps<R>,
    {
        let base_out = self.base.forward(client, input)?;

        // LoRA path: input @ A^T @ B^T * scaling
        let a_t = var_transpose(&self.lora_a).map_err(crate::error::Error::Numr)?;
        let lora_mid = var_matmul(input, &a_t, client).map_err(crate::error::Error::Numr)?;
        let b_t = var_transpose(&self.lora_b).map_err(crate::error::Error::Numr)?;
        let lora_out = var_matmul(&lora_mid, &b_t, client).map_err(crate::error::Error::Numr)?;

        // Scale and add — TRACKED.
        //
        // These must be `var_*` ops. Computing them on `.tensor()` and re-wrapping
        // with `Var::new` produces a LEAF with grad_fn = None, which severs the
        // graph: backward would reach neither `lora_a`/`lora_b` nor the base, so
        // every LoRA adapter would silently never train.
        let scaled = var_mul_scalar(&lora_out, self.scaling as f64, client)
            .map_err(crate::error::Error::Numr)?;
        let result = var_add(&base_out, &scaled, client).map_err(crate::error::Error::Numr)?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::super::layer::quantized_lora;
    use super::*;
    use crate::nn::Linear;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::{Tensor, TensorId};

    /// Gradients must reach the LoRA factors.
    ///
    /// Regression: the scale-and-add tail was built with `Var::new(...)` on raw
    /// tensors, producing a LEAF with no grad_fn. Backward then reached neither
    /// `lora_a`/`lora_b` nor the base, so EVERY LoRA adapter silently never
    /// trained — no error, no NaN, and the loss still falls because the rest of
    /// the network learns.
    #[test]
    fn test_lora_forward_propagates_gradient_to_factors() {
        use crate::test_utils::cpu_setup;
        use numr::autograd::{backward, var_sum};

        let (client, device) = cpu_setup();
        let (in_features, out_features, rank) = (4usize, 3usize, 2usize);

        // Asymmetric weights so a genuine zero gradient cannot pass by accident.
        let base_w: Vec<f32> = (0..out_features * in_features)
            .map(|i| (i as f32) * 0.1 - 0.5)
            .collect();
        let base = Linear::new(
            Tensor::<CpuRuntime>::from_slice(&base_w, &[out_features, in_features], &device)
                .unwrap(),
            None,
            false,
        );
        let lora =
            LoraLinear::new(base, rank, 16.0, &device).expect("lora new must succeed on CPU");

        let x_vals: Vec<f32> = (0..2 * in_features)
            .map(|i| (i as f32) * 0.25 - 0.75)
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_vals, &[2, in_features], &device).unwrap(),
            false,
        );

        let out = lora.forward(&client, &x).expect("lora forward");
        let loss = var_sum(&out, &[0, 1], false, &client).expect("reduce");
        let grads = backward(&loss, &client).expect("backward");

        // lora_b is zero-initialised, so d(loss)/d(lora_a) is zero at step 0 by
        // construction; lora_b is the factor that must receive signal immediately.
        let b_grad = grads
            .get(lora.lora_b.id())
            .expect("lora_b must receive a gradient");
        let b_vals: Vec<f32> = b_grad.contiguous().expect("contig").to_vec();
        let magnitude: f32 = b_vals.iter().map(|v| v.abs()).sum();
        assert!(
            magnitude > 1e-8,
            "lora_b gradient is all zeros ({magnitude}) — the LoRA graph is severed"
        );

        // And lora_a must at least be reachable in the graph.
        assert!(
            grads.get(lora.lora_a.id()).is_some(),
            "lora_a must be reachable from the loss"
        );
    }

    /// Ids must not affect numerics: a `with_ids`-built layer and an equivalent
    /// `from_weights`-built layer must produce identical forward output.
    #[test]
    fn test_with_ids_forward_matches_from_weights() {
        use crate::test_utils::cpu_setup;

        let (client, device) = cpu_setup();
        let (in_features, out_features, rank) = (3usize, 2usize, 2usize);

        let base_w: Vec<f32> = (0..out_features * in_features)
            .map(|i| (i as f32) * 0.1 - 0.3)
            .collect();
        let make_base = || {
            Linear::new(
                Tensor::<CpuRuntime>::from_slice(&base_w, &[out_features, in_features], &device)
                    .unwrap(),
                None,
                false,
            )
        };

        let a_vals: Vec<f32> = (0..rank * in_features)
            .map(|i| (i as f32) * 0.05 - 0.1)
            .collect();
        let b_vals: Vec<f32> = (0..out_features * rank)
            .map(|i| (i as f32) * 0.07 + 0.02)
            .collect();

        let with_ids_lora = LoraLinear::with_ids(
            make_base(),
            Tensor::<CpuRuntime>::from_slice(&a_vals, &[rank, in_features], &device).unwrap(),
            TensorId::new(),
            Tensor::<CpuRuntime>::from_slice(&b_vals, &[out_features, rank], &device).unwrap(),
            TensorId::new(),
            8.0,
            false,
        );
        let from_weights_lora = LoraLinear::from_weights(
            make_base(),
            Tensor::<CpuRuntime>::from_slice(&a_vals, &[rank, in_features], &device).unwrap(),
            Tensor::<CpuRuntime>::from_slice(&b_vals, &[out_features, rank], &device).unwrap(),
            8.0,
            false,
        );

        let x_vals: Vec<f32> = (0..2 * in_features)
            .map(|i| (i as f32) * 0.2 - 0.4)
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_vals, &[2, in_features], &device).unwrap(),
            false,
        );

        let with_ids_out = with_ids_lora
            .forward(&client, &x)
            .expect("with_ids forward");
        let from_weights_out = from_weights_lora
            .forward(&client, &x)
            .expect("from_weights forward");

        let with_ids_vals: Vec<f32> = with_ids_out.tensor().contiguous().expect("contig").to_vec();
        let from_weights_vals: Vec<f32> = from_weights_out
            .tensor()
            .contiguous()
            .expect("contig")
            .to_vec();
        assert_eq!(with_ids_vals.len(), from_weights_vals.len());
        for (w, f) in with_ids_vals.iter().zip(from_weights_vals.iter()) {
            assert!((w - f).abs() < 1e-5, "with_ids={w} from_weights={f}");
        }
    }

    /// `set_trainable(false)` must stop `forward` from recording an autograd
    /// graph — this is the actual fix: a file-loaded adapter used for inference
    /// must not build a graph nobody ever calls `backward` on, since dropping
    /// that graph is what overflowed a bounded worker stack in blazr.
    /// `set_trainable(true)` must restore tracking.
    #[test]
    fn test_set_trainable_toggles_graph_recording() {
        use crate::test_utils::cpu_setup;

        let (client, device) = cpu_setup();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device).unwrap();
        let base = Linear::new(weight, None, false);
        let mut lora =
            LoraLinear::new(base, 2, 4.0, &device).expect("lora new must succeed on CPU");

        // Input itself does not require grad — isolates the adapter's own flag.
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 6], &[2, 3], &device).unwrap(),
            false,
        );

        lora.set_trainable(false);
        assert!(!lora.is_trainable());
        let out_frozen = lora.forward(&client, &x).expect("forward while frozen");
        assert!(
            out_frozen.grad_fn().is_none(),
            "a frozen adapter forward must not record a graph"
        );
        assert!(!out_frozen.requires_grad());

        lora.set_trainable(true);
        assert!(lora.is_trainable());
        let out_tracked = lora.forward(&client, &x).expect("forward while trainable");
        assert!(
            out_tracked.grad_fn().is_some(),
            "a re-trainable adapter forward must record a graph"
        );
        assert!(out_tracked.requires_grad());
    }

    // --- QLoRA: adapter over a quantized base -----------------------------

    /// A LoRA adapter over a `MaybeQuantLinear::Quantized` base must still
    /// produce a finite, correctly-shaped forward output — this is the whole
    /// point of QLoRA: fine-tune directly on a quantized checkpoint.
    #[test]
    fn test_lora_forward_over_quantized_base_is_finite_and_correct_shape() {
        use crate::test_utils::cpu_setup;

        let (client, device) = cpu_setup();
        let (out_features, in_features, rank) = (4usize, 256usize, 2usize);
        let lora = quantized_lora(&client, &device, out_features, in_features, rank, 8.0);

        let x_vals: Vec<f32> = (0..2 * in_features)
            .map(|i| (i as f32 * 0.004) - 1.0)
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_vals, &[2, in_features], &device).unwrap(),
            false,
        );

        let out = lora
            .forward(&client, &x)
            .expect("lora forward over quantized base");
        assert_eq!(out.shape(), &[2, out_features]);

        let vals: Vec<f32> = out.tensor().contiguous().expect("contig").to_vec();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "quantized-base LoRA output must be finite: {vals:?}"
        );
    }

    /// The exact QLoRA arrangement the fix targets: a LoRA adapter over a
    /// quantized base feeds a SECOND, downstream quantized projection before the
    /// loss (`adapter -> quantized projection -> loss`). Before the fix, that
    /// downstream `MaybeQuantLinear::Quantized::forward` detached the graph, so
    /// `backward` reached neither this adapter nor anything upstream of it.
    #[test]
    fn test_lora_gradient_reaches_adapter_through_downstream_quantized_projection() {
        use crate::nn::linear::{MaybeQuantLinear, QuantLinear};
        use crate::quant::format::QuantFormat;
        use crate::quant::traits::QuantizeOps;
        use crate::test_utils::cpu_setup;
        use numr::autograd::{backward, var_sum};

        let (client, device) = cpu_setup();
        // First layer's out_features (32) doubles as the second layer's
        // in_features, so it must satisfy Q8_0's 32-element block size too.
        let (in_features, mid_features, out_features, rank) = (256usize, 32usize, 4usize, 2usize);
        let lora = quantized_lora(&client, &device, mid_features, in_features, rank, 8.0);

        let second_w: Vec<f32> = (0..out_features * mid_features)
            .map(|i| (i as f32 * 0.021).cos() * 0.4)
            .collect();
        let second_tensor =
            Tensor::<CpuRuntime>::from_slice(&second_w, &[out_features, mid_features], &device)
                .unwrap();
        let second_quant = client
            .quantize(&second_tensor, QuantFormat::Q8_0)
            .expect("Q8_0 quantize");
        let second_layer = MaybeQuantLinear::Quantized(QuantLinear::new(second_quant, None));

        let x_vals: Vec<f32> = (0..2 * in_features)
            .map(|i| (i as f32 * 0.005) - 0.5)
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_vals, &[2, in_features], &device).unwrap(),
            false,
        );

        let mid = lora.forward(&client, &x).expect("adapter forward");
        assert!(
            mid.requires_grad(),
            "adapter output must require grad (lora_a/lora_b are trainable)"
        );

        let out = second_layer
            .forward(&client, &mid)
            .expect("downstream quantized projection forward");
        assert!(
            out.requires_grad(),
            "downstream quantized projection must not detach the adapter's graph"
        );

        let loss = var_sum(&out, &[0, 1], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        assert!(
            grads.get(lora.lora_b().id()).is_some(),
            "lora_b must receive a gradient through the downstream quantized projection"
        );
        assert!(
            grads.get(lora.lora_a().id()).is_some(),
            "lora_a must receive a gradient through the downstream quantized projection"
        );
    }
}
