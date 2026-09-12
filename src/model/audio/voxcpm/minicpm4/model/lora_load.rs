//! Optimizer write-back of adapter values across the layer stack.

use super::stack::MiniCpm4Model;
use crate::error::Result;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

impl<R: Runtime<DType = DType>> MiniCpm4Model<R> {
    /// Write back updated adapter values across every layer from an
    /// optimizer's `params` map, keeping every adapter's [`TensorId`]s. See
    /// [`crate::nn::MaybeLoraLinear::load_lora_parameters`] for the
    /// per-projection semantics. No prefix or target validation needed here
    /// — unlike [`Self::apply_lora`], lookup is by ID, not by dotted path,
    /// so there is no zero-match trap to guard against.
    pub fn load_lora_parameters(
        &mut self,
        params: &std::collections::HashMap<TensorId, Tensor<R>>,
    ) -> Result<usize> {
        let mut written = 0;
        for layer in self.layers.iter_mut() {
            written += layer.load_lora_parameters(params)?;
        }
        Ok(written)
    }
}

#[cfg(test)]
mod tests {
    use super::super::stack::tests::{filled, tiny_model};
    use super::*;
    use crate::nn::{LoraTargets, Module};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// Round-trip: adapt `["q_proj", "v_proj"]` across every layer, snapshot the
    /// fresh adapter values and ids, build a `params` map with DIFFERENT values
    /// for every adapter id, call [`MiniCpm4Model::load_lora_parameters`], and
    /// confirm both the returned count and that reading the adapters back
    /// through `trainable_parameters()` yields the NEW values under the SAME
    /// [`TensorId`]s — the optimizer keys its moment state on those ids, so an
    /// id change would silently reset Adam's state every step.
    #[test]
    fn load_lora_parameters_round_trips_new_values_and_preserves_ids() {
        let (_client, device) = cpu_setup();
        let mut model = tiny_model(&device);
        let targets = LoraTargets::new(["q_proj", "v_proj"]);
        let adapted = model
            .apply_lora(&targets, 2, 4.0, &device, "")
            .expect("apply_lora");

        let original: Vec<(TensorId, Vec<usize>)> = model
            .trainable_parameters()
            .into_iter()
            .map(|(id, var)| (id, var.tensor().shape().to_vec()))
            .collect();
        assert_eq!(original.len(), adapted * 2);

        let mut params: std::collections::HashMap<TensorId, Tensor<CpuRuntime>> =
            std::collections::HashMap::new();
        for (i, (id, shape)) in original.iter().enumerate() {
            params.insert(*id, filled(shape, 900 + i, &device));
        }

        let written = model
            .load_lora_parameters(&params)
            .expect("load_lora_parameters");
        assert_eq!(written, original.len());

        let updated = model.trainable_parameters();
        assert_eq!(updated.len(), original.len());
        for (id, _) in &original {
            let (_, var) = updated
                .iter()
                .find(|(uid, _)| uid == id)
                .unwrap_or_else(|| panic!("adapter {id} missing after load_lora_parameters"));
            let expected = params
                .get(id)
                .expect("params has this id")
                .contiguous()
                .expect("contiguous")
                .to_vec::<f32>();
            let actual = var
                .tensor()
                .contiguous()
                .expect("contiguous")
                .to_vec::<f32>();
            assert_eq!(actual, expected, "adapter {id} did not take the new value");
        }
    }

    /// A `params` map missing every adapter id writes nothing: `Ok(0)`, and
    /// every adapter value stays exactly what `apply_lora` set it to.
    #[test]
    fn load_lora_parameters_with_no_matching_ids_returns_zero_and_leaves_values() {
        let (_client, device) = cpu_setup();
        let mut model = tiny_model(&device);
        let targets = LoraTargets::new(["q_proj"]);
        model
            .apply_lora(&targets, 2, 4.0, &device, "")
            .expect("apply_lora");

        let before: Vec<(TensorId, Vec<f32>)> = model
            .trainable_parameters()
            .into_iter()
            .map(|(id, var)| {
                (
                    id,
                    var.tensor()
                        .contiguous()
                        .expect("contiguous")
                        .to_vec::<f32>(),
                )
            })
            .collect();

        let params: std::collections::HashMap<TensorId, Tensor<CpuRuntime>> =
            std::collections::HashMap::new();
        let written = model
            .load_lora_parameters(&params)
            .expect("load_lora_parameters");
        assert_eq!(written, 0);

        let after: Vec<(TensorId, Vec<f32>)> = model
            .trainable_parameters()
            .into_iter()
            .map(|(id, var)| {
                (
                    id,
                    var.tensor()
                        .contiguous()
                        .expect("contiguous")
                        .to_vec::<f32>(),
                )
            })
            .collect();
        assert_eq!(before, after);
    }

    /// A `params` map that carries `lora_a` for an adapted projection but not
    /// its paired `lora_b` is a torn update and errors, naming the projection.
    #[test]
    fn load_lora_parameters_errs_on_torn_update() {
        let (_client, device) = cpu_setup();
        let mut model = tiny_model(&device);
        let targets = LoraTargets::new(["q_proj"]);
        model
            .apply_lora(&targets, 2, 4.0, &device, "")
            .expect("apply_lora");

        let (lora_a, _) = model.layers[0]
            .self_attn
            .q_proj
            .adapters()
            .expect("q_proj is adapted");
        let lora_a_id = lora_a.id();
        let lora_a_shape = lora_a.tensor().shape().to_vec();

        let mut params: std::collections::HashMap<TensorId, Tensor<CpuRuntime>> =
            std::collections::HashMap::new();
        params.insert(lora_a_id, filled(&lora_a_shape, 42, &device));

        let err = model.load_lora_parameters(&params).unwrap_err();
        assert!(err.to_string().contains("q_proj"), "got {err}");
    }

    /// An unadapted (every projection `Plain`) model has no adapters to write
    /// back, so `load_lora_parameters` is `Ok(0)` regardless of `params`.
    #[test]
    fn load_lora_parameters_on_unadapted_model_returns_zero() {
        let (_client, device) = cpu_setup();
        let mut model = tiny_model(&device);
        let params: std::collections::HashMap<TensorId, Tensor<CpuRuntime>> =
            std::collections::HashMap::new();
        let written = model
            .load_lora_parameters(&params)
            .expect("load_lora_parameters");
        assert_eq!(written, 0);
    }
}
