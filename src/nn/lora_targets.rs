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
use numr::autograd::Var;
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

/// Push `local_name`'s full dotted path onto `names`, joined with `prefix`
/// via [`LoraTargets::join`] — the SAME construction [`adapt_if_targeted`]
/// uses to build the path it matches `targets` against.
///
/// Shared by every leaf `lora_projection_names` in the crate for the same
/// reason [`adapt_if_targeted`] is shared by every leaf `apply_lora`: a path
/// built by separately hand-written logic could drift from the one
/// `apply_lora` actually adapts, which would let a target validate and then
/// adapt nothing — the very failure [`LoraTargets::ensure_all_match`] exists
/// to catch.
pub fn push_projection_name(names: &mut Vec<String>, prefix: &str, local_name: &str) {
    names.push(LoraTargets::join(prefix, local_name));
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

/// Build the `TensorId`-keyed map a leaf `load_lora_parameters` needs, from
/// a NAME-keyed map — the only thing a saved adapter safetensors file
/// carries, since [`LoraLinear::new`](crate::nn::LoraLinear::new) mints a
/// fresh `TensorId` every process and it carries no meaning across a
/// save/load boundary.
///
/// `named` is (a filtered subset of) a tree's
/// [`crate::nn::Module::named_parameters()`] output — the caller decides
/// what counts as an adapter name (e.g. a `lora_a`/`lora_b` suffix filter).
/// Shared by every model's `load_lora_named` so the name-to-id resolution
/// and its three failure modes are written once, not re-implemented per
/// model tree.
///
/// Hard-errors, rather than silently skipping, on:
/// - a `tensors` key matching no name in `named` (stale/extra key — the
///   wrong `--targets` case)
/// - a `named` entry with no matching `tensors` key (missing key — a
///   partial adapter file)
/// - a shape mismatch between a `named` Var and its `tensors` entry (the
///   wrong `--rank` case)
pub fn named_tensors_to_id_map<R: Runtime>(
    named: &[(String, &Var<R>)],
    tensors: &HashMap<String, Tensor<R>>,
) -> Result<HashMap<TensorId, Tensor<R>>> {
    let known: std::collections::HashSet<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
    let mut extra: Vec<&str> = tensors
        .keys()
        .map(String::as_str)
        .filter(|k| !known.contains(k))
        .collect();
    if !extra.is_empty() {
        extra.sort_unstable();
        return Err(Error::InvalidArgument {
            arg: "tensors",
            reason: format!(
                "key(s) {extra:?} match no LoRA adapter Var in the model (stale/extra key, or \
                 apply_lora ran with different --targets than this file was saved with)"
            ),
        });
    }

    let mut by_id = HashMap::with_capacity(named.len());
    for (name, var) in named {
        let tensor = tensors.get(name).ok_or_else(|| Error::InvalidArgument {
            arg: "tensors",
            reason: format!(
                "missing key '{name}': the model has this LoRA adapter Var but no matching \
                 tensor was supplied (partial adapter file)"
            ),
        })?;
        if tensor.shape() != var.shape() {
            return Err(Error::InvalidArgument {
                arg: "tensors",
                reason: format!(
                    "shape mismatch for '{name}': model expects {:?}, adapter file has {:?} \
                     (likely a --rank mismatch between the saved adapter and this model)",
                    var.shape(),
                    tensor.shape()
                ),
            });
        }
        by_id.insert(var.id(), tensor.clone());
    }
    Ok(by_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    fn cpu_var(shape: &[usize], fill: f32) -> Var<CpuRuntime> {
        let device = <CpuRuntime as Runtime>::default_device();
        let numel: usize = shape.iter().product();
        let tensor =
            Tensor::<CpuRuntime>::from_slice(&vec![fill; numel], shape, &device).expect("tensor");
        Var::new(tensor, false)
    }

    #[test]
    fn named_tensors_to_id_map_writes_matching_names() {
        let device = <CpuRuntime as Runtime>::default_device();
        let a = cpu_var(&[2, 2], 1.0);
        let b = cpu_var(&[2, 2], 2.0);
        let named: Vec<(String, &Var<CpuRuntime>)> =
            vec![("x.lora_a".to_string(), &a), ("x.lora_b".to_string(), &b)];

        let mut tensors = HashMap::new();
        tensors.insert(
            "x.lora_a".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[9.0f32; 4], &[2, 2], &device).expect("t"),
        );
        tensors.insert(
            "x.lora_b".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[8.0f32; 4], &[2, 2], &device).expect("t"),
        );

        let by_id = named_tensors_to_id_map(&named, &tensors).expect("map");
        assert_eq!(by_id.len(), 2);
        assert_eq!(by_id[&a.id()].to_vec::<f32>(), vec![9.0; 4]);
        assert_eq!(by_id[&b.id()].to_vec::<f32>(), vec![8.0; 4]);
    }

    #[test]
    fn named_tensors_to_id_map_rejects_extra_key() {
        let a = cpu_var(&[2, 2], 1.0);
        let named: Vec<(String, &Var<CpuRuntime>)> = vec![("x.lora_a".to_string(), &a)];
        let device = <CpuRuntime as Runtime>::default_device();

        let mut tensors = HashMap::new();
        tensors.insert(
            "x.lora_a".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[9.0f32; 4], &[2, 2], &device).expect("t"),
        );
        tensors.insert(
            "y.lora_a".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[9.0f32; 4], &[2, 2], &device).expect("t"),
        );

        let err = named_tensors_to_id_map(&named, &tensors).unwrap_err();
        assert!(err.to_string().contains("y.lora_a"), "got {err}");
    }

    #[test]
    fn named_tensors_to_id_map_rejects_missing_key() {
        let a = cpu_var(&[2, 2], 1.0);
        let b = cpu_var(&[2, 2], 2.0);
        let named: Vec<(String, &Var<CpuRuntime>)> =
            vec![("x.lora_a".to_string(), &a), ("x.lora_b".to_string(), &b)];
        let device = <CpuRuntime as Runtime>::default_device();

        let mut tensors = HashMap::new();
        tensors.insert(
            "x.lora_a".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[9.0f32; 4], &[2, 2], &device).expect("t"),
        );

        let err = named_tensors_to_id_map(&named, &tensors).unwrap_err();
        assert!(err.to_string().contains("x.lora_b"), "got {err}");
    }

    #[test]
    fn named_tensors_to_id_map_rejects_shape_mismatch() {
        let a = cpu_var(&[2, 2], 1.0);
        let named: Vec<(String, &Var<CpuRuntime>)> = vec![("x.lora_a".to_string(), &a)];
        let device = <CpuRuntime as Runtime>::default_device();

        let mut tensors = HashMap::new();
        tensors.insert(
            "x.lora_a".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[9.0f32; 8], &[2, 4], &device).expect("t"),
        );

        let err = named_tensors_to_id_map(&named, &tensors).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("x.lora_a"), "got {message}");
        assert!(message.contains("[2, 2]"), "got {message}");
        assert!(message.contains("[2, 4]"), "got {message}");
    }

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
