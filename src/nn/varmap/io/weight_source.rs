//! Format-agnostic `VarMap` constructor over a
//! [`WeightSource`](crate::format::weight_source::WeightSource), and the
//! per-expert MoE stacking pass shared by every constructor in `io/`.

use super::super::core::VarMap;
use crate::error::Result;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;

impl<R: Runtime<DType = DType>> VarMap<R> {
    /// Load every name in `names` through a
    /// [`WeightSource`](crate::format::weight_source::WeightSource), keyed by
    /// `rename(name)`.
    ///
    /// This is the format-agnostic constructor: safetensors, GGUF and TCF all
    /// implement that trait, so one call site covers all three, and a
    /// caller that wraps the source in
    /// [`DenseWeightSource`](crate::format::weight_source::DenseWeightSource)
    /// gets every packed weight materialized to dense F32 without this
    /// function knowing that happened. `VarMap::from_gguf` stays the
    /// GGUF-specific convenience; it is the same shape with the source and
    /// the rename fixed.
    ///
    /// `rename` maps a stored tensor name to the name the model asks
    /// `VarBuilder` for: `gguf_to_hf_name` for a GGUF, identity for a
    /// safetensors checkpoint or a TCF written with HuggingFace names.
    ///
    /// Per-expert MoE tensors are stacked afterwards, exactly as `from_gguf`
    /// does — the pass matches on `.experts.{N}.` names and is a no-op for a
    /// source that has none.
    pub fn from_weight_source<S>(
        source: &mut S,
        names: &[String],
        rename: impl Fn(&str) -> String,
        device: &R::Device,
    ) -> Result<Self>
    where
        R::Client: numr::ops::ShapeOps<R>,
        S: crate::format::weight_source::WeightSource<R>,
    {
        let mut map = Self::new();
        for name in names {
            let weight = source.load_named_weight(name, device)?;
            map.insert_weight(rename(name), weight);
        }
        Self::stack_moe_experts(&mut map, device)?;
        Ok(map)
    }

    /// Stack per-expert MoE tensors into [num_experts, ...] tensors.
    ///
    /// Finds patterns like `*.experts.{N}.{proj}.weight` and stacks them into
    /// `*.experts.{proj}.weight`. Shared by `from_gguf` and
    /// `from_weight_source`.
    pub(super) fn stack_moe_experts(map: &mut Self, _device: &R::Device) -> Result<()>
    where
        R::Client: numr::ops::ShapeOps<R>,
    {
        use std::collections::BTreeMap;

        // Collect expert tensor groups: key = (prefix, proj_suffix), value = sorted (id, tensor)
        let mut groups: HashMap<String, BTreeMap<usize, String>> = HashMap::new();

        let all_names: Vec<String> = map.names().map(|s| s.to_string()).collect();
        for name in &all_names {
            // Match pattern: ...experts.{N}.{suffix}
            if let Some(experts_pos) = name.find(".experts.") {
                let after_experts = &name[experts_pos + ".experts.".len()..];
                if let Some(dot_pos) = after_experts.find('.') {
                    let id_str = &after_experts[..dot_pos];
                    if let Ok(expert_id) = id_str.parse::<usize>() {
                        let prefix = &name[..experts_pos];
                        let suffix = &after_experts[dot_pos + 1..];
                        let group_key = format!("{prefix}.experts.{suffix}");
                        groups
                            .entry(group_key)
                            .or_default()
                            .insert(expert_id, name.clone());
                    }
                }
            }
        }

        // Stack each group
        for (stacked_name, expert_entries) in &groups {
            if expert_entries.len() < 2 {
                continue;
            }

            // Only stack standard (non-quantized) tensors
            let mut tensors: Vec<Tensor<R>> = Vec::with_capacity(expert_entries.len());
            let mut all_standard = true;
            for name in expert_entries.values() {
                match map.get(name) {
                    Ok(w) if !w.is_quantized() => {
                        if let Ok(t) = w.as_tensor() {
                            tensors.push(t.clone());
                        } else {
                            all_standard = false;
                            break;
                        }
                    }
                    _ => {
                        all_standard = false;
                        break;
                    }
                }
            }

            if !all_standard || tensors.is_empty() {
                continue;
            }

            // Stack: each tensor is [dim_in, dim_out], result is [num_experts, dim_in, dim_out]
            let tensor_refs: Vec<&Tensor<R>> = tensors.iter().collect();
            let stacked = Tensor::<R>::stack(&tensor_refs, 0).map_err(|e| {
                crate::error::Error::ModelError {
                    reason: format!("Failed to stack expert tensors for {stacked_name}: {e}"),
                }
            })?;

            // Remove per-expert entries and insert stacked
            for name in expert_entries.values() {
                map.remove(name);
            }
            map.insert(stacked_name.clone(), stacked);
        }

        Ok(())
    }
}
