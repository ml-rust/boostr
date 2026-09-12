//! The [`Module`] impl for [`MiniCpm4Model`]: parameter enumeration.

use super::stack::MiniCpm4Model;
use crate::nn::{Module, child_params, extend_named};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Names ARE the field names (`embed_tokens`, `layers.{i}.*`, `norm`) —
/// this matches the `{prefix}.*` checkpoint layout
/// ([`crate::model::audio::voxcpm::minicpm4::loader`]) exactly, so
/// [`VoxCpm2Model`](crate::model::audio::voxcpm::model::VoxCpm2Model) need
/// only prefix by `base_lm`/`residual_lm` to reach the full checkpoint key.
/// `embed_tokens` is absent entirely on a `residual_lm` instantiation
/// (`vocab_size == 0`) — correctly contributing nothing rather than a
/// zero-filled placeholder. `rope` carries no `Var<R>` (a precomputed,
/// non-learned cos/sin cache, like every other `RoPE` table in this crate)
/// and is correctly absent below.
impl<R: Runtime<DType = DType>> Module<R> for MiniCpm4Model<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = self
            .embed_tokens
            .as_ref()
            .map(|embed_tokens| child_params(embed_tokens))
            .unwrap_or_default();
        for layer in &self.layers {
            params.extend(child_params(layer));
        }
        params.extend(child_params(&self.norm));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        if let Some(embed_tokens) = &self.embed_tokens {
            extend_named(&mut params, "embed_tokens", embed_tokens.named_parameters());
        }
        for (i, layer) in self.layers.iter().enumerate() {
            extend_named(
                &mut params,
                &format!("layers.{i}"),
                layer.named_parameters(),
            );
        }
        extend_named(&mut params, "norm", self.norm.named_parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::super::stack::tests::tiny_model;
    use super::*;
    use crate::test_utils::cpu_setup;

    /// [`MiniCpm4Model::parameters`]/[`named_parameters`] (via `Module`) on
    /// the same tiny fixture the forward-pass tests above build.
    /// `tiny_model` has NO `embed_tokens` (like the real `residual_lm` half
    /// of this model, though here for a different reason — the fixture
    /// simply never sets one), so this also pins that the absent table
    /// contributes nothing rather than a placeholder entry.
    #[test]
    fn module_enumeration_is_non_empty_with_unique_ids_and_names() {
        let (_client, device) = cpu_setup();
        let model = tiny_model(&device);

        let params = model.parameters();
        assert!(!params.is_empty());
        let ids: std::collections::HashSet<_> = params.iter().map(|v| v.id()).collect();
        assert_eq!(ids.len(), params.len(), "duplicate TensorId");

        let named = model.named_parameters();
        assert_eq!(named.len(), params.len());
        let names: std::collections::HashSet<_> = named.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), named.len(), "duplicate parameter name");

        assert!(!named.iter().any(|(n, _)| n.starts_with("embed_tokens")));
        assert!(
            named
                .iter()
                .any(|(n, _)| n == "layers.0.self_attn.q_proj.weight")
        );
        assert!(named.iter().any(|(n, _)| n == "norm.weight"));
    }
}
