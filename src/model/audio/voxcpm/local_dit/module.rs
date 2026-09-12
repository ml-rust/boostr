//! The [`Module`] impl for [`LocalDit`]: parameter enumeration under the
//! checkpoint's `estimator.*` naming.

use super::loader::LocalDit;
use crate::nn::{Module, child_params, extend_named};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Names mirror `feat_decoder.estimator.*` (checkpoint prefix `feat_decoder`
/// added by [`VoxCpm2Model`](crate::model::audio::voxcpm::model::VoxCpm2Model)).
/// `estimator.decoder.layers.{i}`/`estimator.decoder.norm` hardcode a
/// `decoder.` segment this struct's own field names (`layers`, `norm`) do
/// not carry — the checkpoint nests the transformer block stack under
/// `feat_decoder.estimator.decoder.*` (see the module doc's key layout).
/// `rope` and `time_embeddings` carry no `Var<R>` (`time_embeddings`'s
/// frequency table is a fixed, non-learned constant — see
/// [`SinusoidalPosEmb`](crate::nn::SinusoidalPosEmb)) and are correctly absent from every collection
/// below.
impl<R: Runtime<DType = DType>> Module<R> for LocalDit<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = child_params(&self.in_proj);
        params.extend(child_params(&self.cond_proj));
        params.extend(child_params(&self.out_proj));
        params.extend(child_params(&self.time_mlp));
        params.extend(child_params(&self.delta_time_mlp));
        for layer in &self.layers {
            params.extend(child_params(layer));
        }
        params.extend(child_params(&self.norm));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(
            &mut params,
            "estimator.in_proj",
            self.in_proj.named_parameters(),
        );
        extend_named(
            &mut params,
            "estimator.cond_proj",
            self.cond_proj.named_parameters(),
        );
        extend_named(
            &mut params,
            "estimator.out_proj",
            self.out_proj.named_parameters(),
        );
        extend_named(
            &mut params,
            "estimator.time_mlp",
            self.time_mlp.named_parameters(),
        );
        extend_named(
            &mut params,
            "estimator.delta_time_mlp",
            self.delta_time_mlp.named_parameters(),
        );
        for (i, layer) in self.layers.iter().enumerate() {
            extend_named(
                &mut params,
                &format!("estimator.decoder.layers.{i}"),
                layer.named_parameters(),
            );
        }
        extend_named(
            &mut params,
            "estimator.decoder.norm",
            self.norm.named_parameters(),
        );
        params
    }
}

#[cfg(test)]
mod tests {
    use super::super::loader::tests::model;
    use super::*;
    use crate::test_utils::cpu_setup;
    use std::collections::HashSet;

    /// [`LocalDit::parameters`]/[`LocalDit::named_parameters`] (via `Module`) on
    /// the 2-layer tiny model built by [`model`]: the largest sub-module this
    /// crate can build standalone in a unit test, reusing the same fixture the
    /// forward-pass tests build.
    #[test]
    fn module_enumeration_is_non_empty_with_unique_ids_and_names() {
        let (_client, device) = cpu_setup();
        let m = model(2, &device);

        let params = m.parameters();
        assert!(!params.is_empty(), "a 2-layer LocalDit must own parameters");

        let ids: HashSet<_> = params.iter().map(|var| var.id()).collect();
        assert_eq!(
            ids.len(),
            params.len(),
            "duplicate TensorId: two fields alias the same Var, so an optimizer \
             would double-step it"
        );

        let named = m.named_parameters();
        assert_eq!(
            named.len(),
            params.len(),
            "named_parameters() must enumerate exactly the same parameters as parameters()"
        );
        let names: HashSet<_> = named.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names.len(), named.len(), "duplicate parameter name");

        // Spot-check the dotted paths against the verified checkpoint key
        // layout in this module's loader doc (`estimator.*`, `estimator.decoder.*`).
        assert!(named.iter().any(|(n, _)| n == "estimator.in_proj.weight"));
        assert!(named.iter().any(|(n, _)| n == "estimator.in_proj.bias"));
        assert!(
            named
                .iter()
                .any(|(n, _)| n == "estimator.decoder.norm.weight")
        );
        assert!(
            named
                .iter()
                .any(|(n, _)| n == "estimator.decoder.layers.0.self_attn.q_proj.weight")
        );
        assert!(
            named
                .iter()
                .any(|(n, _)| n == "estimator.decoder.layers.1.mlp.down_proj.weight")
        );
        assert!(
            named
                .iter()
                .any(|(n, _)| n == "estimator.time_mlp.linear_1.weight")
        );
    }
}
