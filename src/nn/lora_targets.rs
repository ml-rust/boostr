//! Target matching for `apply_lora` — which named projections in a model
//! tree get wrapped with a LoRA adapter.
//!
//! Matching is by DOT-SEGMENT, not substring: a target `q_proj` matches a
//! dotted checkpoint path `base_lm.layers.3.self_attn.q_proj` (and also
//! `self_attn.q_proj`) because `q_proj` is one of its `.`-separated
//! segments, but it does NOT match `xq_projy` — that string has no segment
//! equal to `q_proj`. This mirrors oxidizr's
//! `LoraConfig::train_module_matches` (`oxidizr/src/config.rs:378-385`),
//! which resolves the identical `module.split('.').any(|segment| segment ==
//! name)` rule for `train_modules`.

use crate::error::{Error, Result};
use crate::nn::maybe_lora::MaybeLoraLinear;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};
use std::collections::HashMap;

/// Named projections to wrap with a LoRA adapter, matched by dot-segment
/// against a full checkpoint-style path (e.g. `"q_proj"`, `"v_proj"`).
pub struct LoraTargets {
    names: Vec<String>,
}

impl LoraTargets {
    /// Build from any collection of target names, e.g. `["q_proj",
    /// "v_proj"]`.
    pub fn new(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            names: names.into_iter().map(Into::into).collect(),
        }
    }

    /// The configured target names, verbatim.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Whether `path`'s dot-separated segments include ANY of `self`'s
    /// names. See the module doc for why this is segment equality, not
    /// substring search.
    pub fn matches(&self, path: &str) -> bool {
        self.names
            .iter()
            .any(|name| path.split('.').any(|segment| segment == name.as_str()))
    }

    /// Join a parent path prefix and a local field/segment name the same
    /// way [`crate::nn::extend_named`] joins a container's prefix onto a
    /// child's `named_parameters()` names — `"{prefix}.{name}"`, or bare
    /// `name` when `prefix` is empty (the checkpoint root, e.g.
    /// `AuxProjections`'s six projections). Every `apply_lora` in the crate
    /// builds its per-projection paths through this one function so they are
    /// constructed identically to `named_parameters()`'s own paths — the
    /// two must never be built by separately hand-written logic, or a
    /// target can silently miss the projection it was meant to adapt.
    pub fn join(prefix: &str, name: &str) -> String {
        if prefix.is_empty() {
            name.to_string()
        } else {
            format!("{prefix}.{name}")
        }
    }

    /// Validate that every target name matches at least one of `candidates`
    /// (full dotted paths, e.g. from `Module::named_parameters()`).
    ///
    /// This is the zero-match trap guard: a target that matches NOTHING
    /// must error, naming the offending target(s), rather than silently
    /// wrapping zero projections while the caller's run looks healthy. Call
    /// this ONCE, against the full candidate set of the tree an `apply_lora`
    /// call is entered on — a container that calls this again for each
    /// child's OWN (necessarily narrower) candidate set would reject a
    /// perfectly valid multi-domain target list, since e.g. an `mlp` has no
    /// `q_proj` to match even when `q_proj` is valid elsewhere in the same
    /// tree. Do not call this from a delegated/composed `apply_lora` step —
    /// only from the entry point the caller invokes directly (see each
    /// `apply_lora`'s doc comment for whether it is such an entry point).
    pub fn ensure_all_match(&self, candidates: &[String]) -> Result<()> {
        let unmatched: Vec<&str> = self
            .names
            .iter()
            .filter(|name| {
                !candidates
                    .iter()
                    .any(|path| path.split('.').any(|segment| segment == name.as_str()))
            })
            .map(String::as_str)
            .collect();
        if unmatched.is_empty() {
            return Ok(());
        }
        let sample: Vec<&str> = candidates.iter().take(5).map(String::as_str).collect();
        Err(Error::InvalidArgument {
            arg: "targets",
            reason: format!(
                "LoRA target(s) {unmatched:?} matched no projection by dot-segment name out of \
                 {} candidate(s); available projections include {sample:?}",
                candidates.len()
            ),
        })
    }
}

/// Adapt `field` in place if its full dotted path — `{prefix}.{local_name}`,
/// via [`LoraTargets::join`] — matches `targets`. Returns `1` when adapted,
/// `0` when `field` was not targeted. Errors when targeted but `field`
/// already carries an adapter (see [`MaybeLoraLinear::apply_lora`]).
///
/// Shared by every leaf `apply_lora` in the crate so the match-then-wrap
/// step, and the path it matches against, are written exactly once.
pub fn adapt_if_targeted<R: Runtime<DType = DType>>(
    field: &mut MaybeLoraLinear<R>,
    targets: &LoraTargets,
    rank: usize,
    alpha: f32,
    device: &R::Device,
    prefix: &str,
    local_name: &str,
) -> Result<usize> {
    let path = LoraTargets::join(prefix, local_name);
    if !targets.matches(&path) {
        return Ok(0);
    }
    field.apply_lora(rank, alpha, device)?;
    Ok(1)
}

/// Write back `field`'s adapter values (if any) from `params`, tagging a
/// torn-update error with `local_name` so the caller learns WHICH
/// projection has a half-applied pair, not just which
/// [`TensorId`](numr::tensor::TensorId)s.
///
/// Shared by every leaf `load_lora_parameters` in the crate, mirroring how
/// [`adapt_if_targeted`] is shared by every leaf `apply_lora`. Unlike
/// `adapt_if_targeted`, `local_name` here does no path matching — see
/// [`MaybeLoraLinear::load_lora_parameters`], which looks adapters up by ID.
pub fn load_lora_child<R: Runtime<DType = DType>>(
    field: &mut MaybeLoraLinear<R>,
    params: &HashMap<TensorId, Tensor<R>>,
    local_name: &str,
) -> Result<usize> {
    field
        .load_lora_parameters(params)
        .map_err(|source| Error::ModelError {
            reason: format!("{local_name}: {source}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_dot_segment_not_substring() {
        let targets = LoraTargets::new(["q_proj"]);
        assert!(targets.matches("base_lm.layers.3.self_attn.q_proj"));
        assert!(targets.matches("self_attn.q_proj"));
        assert!(targets.matches("q_proj"));
        assert!(!targets.matches("xq_projy"));
        assert!(!targets.matches("q_proj_extra"));
    }

    #[test]
    fn join_handles_empty_prefix() {
        assert_eq!(LoraTargets::join("", "stop_proj"), "stop_proj");
        assert_eq!(
            LoraTargets::join("fsq_layer", "in_proj"),
            "fsq_layer.in_proj"
        );
    }

    #[test]
    fn ensure_all_match_reports_offending_targets() {
        let targets = LoraTargets::new(["q_proj", "roj"]);
        let candidates = vec!["layers.0.self_attn.q_proj.weight".to_string()];
        let err = targets.ensure_all_match(&candidates).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("roj"), "got {message}");
        assert!(!message.contains("\"q_proj\""), "got {message}");
    }

    #[test]
    fn ensure_all_match_passes_when_every_target_hits() {
        let targets = LoraTargets::new(["q_proj", "v_proj"]);
        let candidates = vec![
            "layers.0.self_attn.q_proj.weight".to_string(),
            "layers.0.self_attn.v_proj.weight".to_string(),
        ];
        assert!(targets.ensure_all_match(&candidates).is_ok());
    }
}
