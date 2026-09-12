//! [`VoxCpm2Model::load_lora_named`]: apply a NAME-keyed adapter map.

use super::VoxCpm2Model;
use crate::error::Result;
use crate::nn::{Module, named_tensors_to_id_map};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;

impl<R: Runtime<DType = DType>> VoxCpm2Model<R> {
    /// Load a LoRA adapter saved to safetensors — keyed by NAME, not by
    /// [`TensorId`](numr::tensor::TensorId) — and write it into this
    /// model's adapters.
    ///
    /// [`Self::load_lora_parameters`] exists for the in-PROCESS path:
    /// `SimpleTrainer::step`'s `TensorId`-keyed map, built from the SAME
    /// `Var`s this model still holds. A saved adapter file carries no such
    /// map — `TensorId`s are minted fresh by
    /// [`LoraLinear::new`](crate::nn::LoraLinear::new) every time
    /// [`Self::apply_lora`] runs, so they mean nothing across a save/load
    /// boundary or a second process. This function is the bridge: it reads
    /// each name off `self.named_parameters()`, resolves it to the `Var`'s
    /// current `TensorId`, and delegates to `load_lora_parameters`.
    ///
    /// # `apply_lora` MUST run first
    ///
    /// `named_parameters()` only lists `lora_a`/`lora_b` entries for
    /// projections that already carry an adapter — before `apply_lora`,
    /// every LoRA-suffixed name this function looks for is simply absent.
    /// Calling this on an unadapted model would then be a silent no-op:
    /// zero adapter names to resolve, so nothing to write, and `Ok(0)`
    /// looks identical to "the file matched everything". `tensors`
    /// non-empty against zero resolvable names is exactly the "stale/extra
    /// key" case below, so it errors instead — that is the failure this
    /// function exists to make impossible, not a case it degrades into.
    ///
    /// # Errors
    ///
    /// Hard-errors, never silently skips, on:
    /// - a `tensors` key matching no `lora_a`/`lora_b` `Var` in the model
    ///   (stale/extra key — the wrong `--targets` case)
    /// - an adapter `Var` with no matching key in `tensors` (missing key —
    ///   a partial adapter file)
    /// - a shape mismatch between a `Var` and its incoming tensor (the
    ///   wrong `--rank` case)
    ///
    /// Returns the number of adapter tensors written (2 per adapted
    /// projection present in `tensors`).
    pub fn load_lora_named(&mut self, tensors: &HashMap<String, Tensor<R>>) -> Result<usize> {
        let adapter_vars: Vec<(String, &Var<R>)> = self
            .named_parameters()
            .into_iter()
            .filter(|(name, _)| name.ends_with("lora_a") || name.ends_with("lora_b"))
            .collect();
        let by_id = named_tensors_to_id_map(&adapter_vars, tensors)?;
        self.load_lora_parameters(&by_id)
    }
}

#[cfg(test)]
mod tests {
    //! [`super::VoxCpm2Model::load_lora_named`] delegates entirely to
    //! [`crate::nn::named_tensors_to_id_map`] plus the pre-existing
    //! `load_lora_parameters` (see `super`'s doc comment). This module proves
    //! that exact mechanism end to end — apply_lora, build a save-shaped NAME
    //! map, resolve and write it back, read the values back off the live
    //! `Var`s — on [`AuxProjections`], a small REAL `Module<R>` in this tree
    //! that carries the identical `MaybeLoraLinear` fields `VoxCpm2Model` does.
    //!
    //! A full `VoxCpm2Model` needs a real `AudioVaeEncoder`/`AudioVaeDecoder`
    //! (four `EncoderBlock`/`DecoderBlock` stages each), which has no test
    //! fixture anywhere in this crate and is out of scope for this LoRA
    //! name-mapping fix — `vae_encoder`/`vae_decoder` are excluded from
    //! `named_parameters`/`apply_lora` entirely (see `loader.rs`'s doc
    //! comments), so they are irrelevant to what is under test here.

    use super::*;
    use crate::model::audio::voxcpm::fsq::AuxProjections;
    use crate::nn::named_tensors_to_id_map;
    use crate::nn::{LoraTargets, MaybeLoraLinear, MaybeQuantLinear, Weight};
    use numr::runtime::cpu::CpuRuntime;

    const DIM: usize = 2;

    fn square_linear(
        fill: f32,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> MaybeLoraLinear<CpuRuntime> {
        let weight = Tensor::<CpuRuntime>::from_slice(&[fill; DIM * DIM], &[DIM, DIM], device)
            .expect("weight");
        MaybeQuantLinear::from_weight(Weight::Standard(weight), None).into()
    }

    fn aux_projections(device: &<CpuRuntime as Runtime>::Device) -> AuxProjections<CpuRuntime> {
        AuxProjections {
            enc_to_lm_proj: square_linear(0.1, device),
            lm_to_dit_proj: square_linear(0.2, device),
            res_to_dit_proj: square_linear(0.3, device),
            fusion_concat_proj: square_linear(0.4, device),
            stop_proj: square_linear(0.5, device),
            stop_head: square_linear(0.6, device),
        }
    }

    /// Build a save-shaped `name -> tensor` map: every `lora_a`/`lora_b`
    /// currently in `aux`, filled with `fill` (distinguishable from the
    /// adapter's own zero/random init).
    fn adapter_tensor_map(
        aux: &AuxProjections<CpuRuntime>,
        fill: f32,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> HashMap<String, Tensor<CpuRuntime>> {
        let mut out = HashMap::new();
        for (name, var) in Module::named_parameters(aux) {
            if name.ends_with("lora_a") || name.ends_with("lora_b") {
                let numel: usize = var.shape().iter().product();
                let tensor =
                    Tensor::<CpuRuntime>::from_slice(&vec![fill; numel], var.shape(), device)
                        .expect("tensor");
                out.insert(name, tensor);
            }
        }
        out
    }

    #[test]
    fn happy_path_writes_named_values_into_the_right_vars() {
        let device = <CpuRuntime as Runtime>::default_device();
        let mut aux = aux_projections(&device);
        let targets = LoraTargets::new(["enc_to_lm_proj", "stop_head"]);
        let adapted = aux
            .apply_lora(&targets, 4, 8.0, &device, "")
            .expect("apply_lora");
        assert_eq!(adapted, 2);

        let tensors = adapter_tensor_map(&aux, 9.0, &device);
        assert_eq!(tensors.len(), 4); // 2 adapted projections * (lora_a, lora_b)

        let named: Vec<(String, &Var<CpuRuntime>)> = Module::named_parameters(&aux)
            .into_iter()
            .filter(|(name, _)| name.ends_with("lora_a") || name.ends_with("lora_b"))
            .collect();
        let by_id = named_tensors_to_id_map(&named, &tensors).expect("resolve");
        let written = aux.load_lora_parameters(&by_id).expect("load");
        assert_eq!(written, 4);

        for (name, var) in Module::named_parameters(&aux) {
            if name.ends_with("lora_a") || name.ends_with("lora_b") {
                assert_eq!(
                    var.tensor().to_vec::<f32>(),
                    vec![9.0f32; var.numel()],
                    "{name} did not land the saved value"
                );
            }
        }
    }

    #[test]
    fn rejects_extra_key_not_present_as_any_adapter_var() {
        let device = <CpuRuntime as Runtime>::default_device();
        let mut aux = aux_projections(&device);
        let targets = LoraTargets::new(["enc_to_lm_proj"]);
        aux.apply_lora(&targets, 4, 8.0, &device, "")
            .expect("apply_lora");

        let mut tensors = adapter_tensor_map(&aux, 9.0, &device);
        tensors.insert(
            "stop_head.lora_a".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; DIM * 4], &[DIM, 4], &device).expect("t"),
        );

        let named: Vec<(String, &Var<CpuRuntime>)> = Module::named_parameters(&aux)
            .into_iter()
            .filter(|(name, _)| name.ends_with("lora_a") || name.ends_with("lora_b"))
            .collect();
        let err = named_tensors_to_id_map(&named, &tensors).unwrap_err();
        assert!(err.to_string().contains("stop_head.lora_a"), "got {err}");
    }

    #[test]
    fn rejects_missing_key_for_an_adapted_projection() {
        let device = <CpuRuntime as Runtime>::default_device();
        let mut aux = aux_projections(&device);
        let targets = LoraTargets::new(["enc_to_lm_proj"]);
        aux.apply_lora(&targets, 4, 8.0, &device, "")
            .expect("apply_lora");

        let mut tensors = adapter_tensor_map(&aux, 9.0, &device);
        tensors.remove("enc_to_lm_proj.lora_b");

        let named: Vec<(String, &Var<CpuRuntime>)> = Module::named_parameters(&aux)
            .into_iter()
            .filter(|(name, _)| name.ends_with("lora_a") || name.ends_with("lora_b"))
            .collect();
        let err = named_tensors_to_id_map(&named, &tensors).unwrap_err();
        assert!(
            err.to_string().contains("enc_to_lm_proj.lora_b"),
            "got {err}"
        );
    }

    #[test]
    fn rejects_shape_mismatch_from_a_rank_change() {
        let device = <CpuRuntime as Runtime>::default_device();
        let mut aux = aux_projections(&device);
        let targets = LoraTargets::new(["enc_to_lm_proj"]);
        aux.apply_lora(&targets, 4, 8.0, &device, "")
            .expect("apply_lora");

        let mut tensors = adapter_tensor_map(&aux, 9.0, &device);
        // Rank 4 -> rank 8: doubles lora_a's row count.
        tensors.insert(
            "enc_to_lm_proj.lora_a".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[9.0f32; 8 * DIM], &[8, DIM], &device).expect("t"),
        );

        let named: Vec<(String, &Var<CpuRuntime>)> = Module::named_parameters(&aux)
            .into_iter()
            .filter(|(name, _)| name.ends_with("lora_a") || name.ends_with("lora_b"))
            .collect();
        let err = named_tensors_to_id_map(&named, &tensors).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("enc_to_lm_proj.lora_a"), "got {message}");
    }

    #[test]
    fn calling_before_apply_lora_errors_instead_of_silently_doing_nothing() {
        let device = <CpuRuntime as Runtime>::default_device();
        let aux = aux_projections(&device);

        // No apply_lora call: `named_parameters()` has zero lora_a/lora_b
        // entries, so ANY non-empty saved adapter map has nothing to match
        // against — this is the "silent no-op" failure `load_lora_named`'s doc
        // comment names, made impossible by `named_tensors_to_id_map`'s
        // extra-key check.
        let mut tensors = HashMap::new();
        tensors.insert(
            "enc_to_lm_proj.lora_a".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4 * DIM], &[4, DIM], &device).expect("t"),
        );

        let named: Vec<(String, &Var<CpuRuntime>)> = Module::named_parameters(&aux)
            .into_iter()
            .filter(|(name, _)| name.ends_with("lora_a") || name.ends_with("lora_b"))
            .collect();
        assert!(named.is_empty());
        let err = named_tensors_to_id_map(&named, &tensors).unwrap_err();
        assert!(
            err.to_string().contains("enc_to_lm_proj.lora_a"),
            "got {err}"
        );
    }
}
