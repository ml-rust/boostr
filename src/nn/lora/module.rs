//! [`Module`] impl for [`LoraLinear`]: parameter enumeration.

use crate::nn::module::Module;
use numr::autograd::Var;
use numr::runtime::Runtime;

use super::LoraLinear;

impl<R: Runtime> Module<R> for LoraLinear<R> {
    // Enumerate the base as well as the adapters, and let the caller's
    // `requires_grad` filter decide what actually trains. Returning only the
    // adapters would be wrong in both directions: a QUANTIZED base already
    // contributes nothing here (`MaybeQuantLinear::parameters` is empty for
    // the quantized variants), so nothing needs suppressing for QLoRA; while
    // a DENSE base that a caller deliberately left trainable — oxidizr's
    // `lora.train_modules` opts a named projection's base back in even under
    // `freeze_base: true` — would silently vanish from the optimizer's
    // parameter set while still being checkpointed, so the weight would be
    // saved every step and never once updated.
    fn parameters(&self) -> Vec<&Var<R>> {
        // `Linear` also has an INHERENT `parameters()` returning `(TensorId, &Var)`
        // pairs, which shadows the trait method — disambiguate explicitly.
        let mut params = Module::parameters(&self.base);
        params.push(&self.lora_a);
        params.push(&self.lora_b);
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params: Vec<(String, &Var<R>)> = self
            .base
            .named_parameters()
            .into_iter()
            .map(|(name, var)| (format!("base.{name}"), var))
            .collect();
        params.push(("lora_a".to_string(), &self.lora_a));
        params.push(("lora_b".to_string(), &self.lora_b));
        params
    }
}

#[cfg(test)]
mod tests {
    use super::super::layer::quantized_lora;
    use super::*;
    use crate::nn::Linear;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    /// `Module::parameters` is UNFILTERED: it reports the dense base's
    /// weight/bias alongside the adapter factors, and the caller's
    /// `requires_grad` filter decides what actually trains.
    ///
    /// Reporting only the adapters would break a real caller. oxidizr's
    /// `lora.train_modules` deliberately opts a named projection's base back
    /// into training even under `freeze_base: true`; if `parameters()` dropped
    /// it, that weight would be checkpointed every step and never once updated
    /// by the optimizer — silently, with a healthy-looking run.
    ///
    /// A QUANTIZED base needs no special handling here: it has no `Var` weight,
    /// and `MaybeQuantLinear::parameters` is already empty for those variants,
    /// so this same code reports adapters alone. See
    /// `test_quantized_base_trainable_parameters_are_adapters_only`.
    #[test]
    fn test_module_parameters_reports_base_and_adapters() {
        let device = <CpuRuntime as Runtime>::default_device();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device).unwrap();
        let bias = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device).unwrap();
        let base = Linear::new(weight, Some(bias), true);
        let lora = LoraLinear::new(base, 2, 4.0, &device).expect("lora new must succeed on CPU");

        // base.weight + base.bias + lora_a + lora_b.
        assert_eq!(lora.parameters().len(), 4);

        let named = lora.named_parameters();
        assert_eq!(named.len(), 4);
        assert!(named.iter().any(|(n, _)| n == "lora_a"));
        assert!(named.iter().any(|(n, _)| n == "lora_b"));
        assert!(named.iter().any(|(n, _)| n == "base.weight"));
        assert!(named.iter().any(|(n, _)| n == "base.bias"));
    }

    /// The regression the doc comment above describes, pinned directly: a dense
    /// base left TRAINABLE must survive `trainable_parameters()`.
    #[test]
    fn test_trainable_base_is_not_dropped_from_trainable_parameters() {
        let device = <CpuRuntime as Runtime>::default_device();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device).unwrap();
        let base = Linear::new(weight, None, true); // deliberately trainable
        let lora = LoraLinear::new(base, 2, 4.0, &device).expect("lora new must succeed on CPU");

        // base.weight + lora_a + lora_b: the optimizer must be able to step all
        // three, or `lora.train_modules` silently stops working.
        assert_eq!(lora.trainable_parameters().len(), 3);
    }

    /// With a frozen base, only the adapter factors are trainable — this is
    /// what lets LoRA train an adapter alone.
    #[test]
    fn test_trainable_parameters_excludes_frozen_base() {
        let device = <CpuRuntime as Runtime>::default_device();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device).unwrap();
        let base = Linear::new(weight, None, false); // frozen
        let lora = LoraLinear::new(base, 2, 4.0, &device).expect("lora new must succeed on CPU");

        let trainable = lora.trainable_parameters();
        assert_eq!(trainable.len(), 2);
        assert_eq!(trainable[0].0, lora.lora_a.id());
        assert_eq!(trainable[1].0, lora.lora_b.id());
    }

    /// `trainable_parameters()` on a quantized-base adapter must report exactly
    /// the two adapter tensors and nothing from the base — the base has no
    /// `Var<R>` weight to report in the first place.
    #[test]
    fn test_quantized_base_trainable_parameters_are_adapters_only() {
        use crate::test_utils::cpu_setup;

        let (client, device) = cpu_setup();
        let lora = quantized_lora(&client, &device, 4, 256, 2, 8.0);

        let trainable = lora.trainable_parameters();
        assert_eq!(trainable.len(), 2);
        assert_eq!(trainable[0].0, lora.lora_a().id());
        assert_eq!(trainable[1].0, lora.lora_b().id());

        // The base itself has no `Var<R>` weight to have contributed one.
        assert!(lora.weight().is_none());

        let params = lora.parameters();
        assert_eq!(params.len(), 2);
    }
}
