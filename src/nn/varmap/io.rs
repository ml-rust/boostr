//! Format-based constructors for VarMap (SafeTensors, GGUF).

use super::core::VarMap;
use crate::error::{Error, Result};
use crate::format::gguf::Gguf;
use crate::format::safetensors::SafeTensors;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;

impl<R: Runtime<DType = DType>> VarMap<R> {
    /// Load all tensors from a SafeTensors file.
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: &R::Device) -> Result<Self> {
        Self::from_safetensors_with_model_type(path, device, None)
    }

    /// Load all tensors from a SafeTensors file, normalizing tensor names for
    /// the given model type (e.g., "falcon", "gpt_neox", "dbrx").
    pub fn from_safetensors_with_model_type<P: AsRef<Path>>(
        path: P,
        device: &R::Device,
        model_type: Option<&str>,
    ) -> Result<Self> {
        use crate::format::safetensors_name_map::normalize_hf_name;
        let mut st = SafeTensors::open(path)?;
        let all = st.load_all::<R>(device)?;
        let mut map = Self::new();
        for (name, tensor) in all {
            let mapped = match model_type {
                Some(mt) => normalize_hf_name(mt, &name),
                None => name,
            };
            map.insert(mapped, tensor);
        }
        Ok(map)
    }

    /// Load tensors from sharded SafeTensors files.
    ///
    /// Reads `model.safetensors.index.json` from `dir`, which maps tensor names
    /// to shard filenames (e.g., `model-00001-of-00004.safetensors`).
    /// Loads each shard once and extracts all its tensors.
    pub fn from_safetensors_sharded<P: AsRef<Path>>(dir: P, device: &R::Device) -> Result<Self> {
        Self::from_safetensors_sharded_with_model_type(dir, device, None)
    }

    /// Load tensors from sharded SafeTensors files with name normalization.
    pub fn from_safetensors_sharded_with_model_type<P: AsRef<Path>>(
        dir: P,
        device: &R::Device,
        model_type: Option<&str>,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        let index_path = dir.join("model.safetensors.index.json");
        let index_str = std::fs::read_to_string(&index_path).map_err(|e| Error::ModelError {
            reason: format!("failed to read index file: {e}"),
        })?;

        let index: serde_json::Value =
            serde_json::from_str(&index_str).map_err(|e| Error::ModelError {
                reason: format!("failed to parse index JSON: {e}"),
            })?;

        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::ModelError {
                reason: "index.json missing 'weight_map' object".into(),
            })?;

        // Group tensor names by shard file to open each shard only once
        let mut shard_to_names: HashMap<String, Vec<String>> = HashMap::new();
        for (tensor_name, shard_val) in weight_map {
            let shard_file = shard_val.as_str().ok_or_else(|| Error::ModelError {
                reason: format!("weight_map value for '{tensor_name}' is not a string"),
            })?;
            shard_to_names
                .entry(shard_file.to_string())
                .or_default()
                .push(tensor_name.clone());
        }

        let mut map = Self::new();

        for (shard_file, names) in &shard_to_names {
            let shard_path = dir.join(shard_file);
            let mut st = SafeTensors::open(&shard_path)?;
            for name in names {
                let tensor = st.load_tensor::<R>(name, device)?;
                let mapped = match model_type {
                    Some(mt) => crate::format::safetensors_name_map::normalize_hf_name(mt, name),
                    None => name.clone(),
                };
                map.insert(mapped, tensor);
            }
        }

        Ok(map)
    }

    /// Load all tensors from a GGUF file.
    ///
    /// Unquantized tensors (F32, F16, BF16) are loaded as `Weight::Standard`.
    /// Quantized tensors (Q4_0, Q4K, etc.) are loaded as `Weight::Quantized`.
    pub fn from_gguf<P: AsRef<Path>>(path: P, device: &R::Device) -> Result<Self>
    where
        R::Client: numr::ops::ShapeOps<R>,
    {
        use crate::format::gguf::gguf_to_hf_name;

        let mut gguf = Gguf::open(path)?;
        let names: Vec<String> = gguf.tensor_names().map(|s| s.to_string()).collect();
        let mut map = Self::new();

        for name in &names {
            let hf_name = gguf_to_hf_name(name);
            let info = gguf.tensor_info(name)?.clone();
            if info.ggml_type.is_quantized() {
                let qt = gguf.load_tensor_quantized::<R>(name, device)?;
                map.insert_quant(hf_name, qt);
            } else {
                let t = gguf.load_tensor_f32::<R>(name, device)?;
                map.insert(hf_name, t);
            }
        }

        // Stack per-expert MoE tensors into single stacked tensors.
        // GGUF stores experts individually (experts.0.gate_proj.weight, experts.1.gate_proj.weight, ...)
        // but the model expects stacked tensors (experts.gate_proj.weight with shape [num_experts, ...]).
        Self::stack_moe_experts(&mut map, device)?;

        Ok(map)
    }

    /// Stack per-expert MoE tensors into [num_experts, ...] tensors.
    ///
    /// Finds patterns like `*.experts.{N}.{proj}.weight` and stacks them into
    /// `*.experts.{proj}.weight`.
    fn stack_moe_experts(map: &mut Self, _device: &R::Device) -> Result<()>
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::safetensors::save_safetensors;
    use crate::quant::QuantFormat;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn device() -> CpuDevice {
        CpuDevice::new()
    }

    #[test]
    fn test_varmap_from_safetensors_roundtrip() {
        let d = device();
        let tmp = tempfile::NamedTempFile::new().unwrap();

        let mut tensors = HashMap::new();
        tensors.insert(
            "a".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &d),
        );
        tensors.insert(
            "b".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0], &[2], &d),
        );
        save_safetensors(tmp.path(), &tensors, None).unwrap();

        let map = VarMap::<CpuRuntime>::from_safetensors(tmp.path(), &d).unwrap();
        assert_eq!(map.len(), 2);

        let a = map.get_tensor("a").unwrap();
        assert_eq!(a.shape(), &[3]);
        let data = a.to_vec::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);

        let b = map.get_tensor("b").unwrap();
        assert_eq!(b.shape(), &[2]);
    }

    #[test]
    fn test_varmap_from_gguf() {
        let d = device();
        let tmp = create_test_gguf_file();

        let map = VarMap::<CpuRuntime>::from_gguf(tmp.path(), &d).unwrap();
        assert_eq!(map.len(), 2);

        let f32_w = map.get("weight_f32").unwrap();
        assert!(!f32_w.is_quantized());
        let t = f32_w.as_tensor().unwrap();
        assert_eq!(t.shape(), &[4]);

        let q4_w = map.get("weight_q4").unwrap();
        assert!(q4_w.is_quantized());
        let qt = q4_w.as_quant_tensor().unwrap();
        assert_eq!(qt.shape(), &[32]);
        assert_eq!(qt.format(), QuantFormat::Q4_0);
    }

    #[test]
    fn test_varmap_from_safetensors_sharded() {
        let d = device();
        let dir = tempfile::TempDir::new().unwrap();

        // Create two shard files
        let mut shard1 = HashMap::new();
        shard1.insert(
            "layers.0.weight".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &d),
        );
        save_safetensors(
            dir.path().join("model-00001-of-00002.safetensors"),
            &shard1,
            None,
        )
        .unwrap();

        let mut shard2 = HashMap::new();
        shard2.insert(
            "layers.1.weight".to_string(),
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0, 5.0], &[3], &d),
        );
        save_safetensors(
            dir.path().join("model-00002-of-00002.safetensors"),
            &shard2,
            None,
        )
        .unwrap();

        // Create index.json
        let index = serde_json::json!({
            "metadata": {"total_size": 20},
            "weight_map": {
                "layers.0.weight": "model-00001-of-00002.safetensors",
                "layers.1.weight": "model-00002-of-00002.safetensors"
            }
        });
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&index).unwrap(),
        )
        .unwrap();

        // Load sharded
        let map = VarMap::<CpuRuntime>::from_safetensors_sharded(dir.path(), &d).unwrap();
        assert_eq!(map.len(), 2);

        let t0 = map.get_tensor("layers.0.weight").unwrap();
        assert_eq!(t0.shape(), &[2]);
        let data0: Vec<f32> = t0.to_vec();
        assert!((data0[0] - 1.0).abs() < 1e-6);

        let t1 = map.get_tensor("layers.1.weight").unwrap();
        assert_eq!(t1.shape(), &[3]);
        let data1: Vec<f32> = t1.to_vec();
        assert!((data1[2] - 5.0).abs() < 1e-6);
    }

    // ── GGUF test file helper ─────────────────────────────────────────

    fn create_test_gguf_file() -> tempfile::NamedTempFile {
        use crate::format::gguf::types::{GgmlType, GgufValueType};
        use std::io::Write;

        let mut buf = Vec::new();
        let gguf_magic: u32 = 0x46554747;

        buf.extend_from_slice(&gguf_magic.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());

        write_gguf_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_gguf_str(&mut buf, "test");

        write_gguf_str(&mut buf, "weight_f32");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        write_gguf_str(&mut buf, "weight_q4");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&32u64.to_le_bytes());
        buf.extend_from_slice(&(GgmlType::Q4_0 as u32).to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());

        let aligned = buf.len().div_ceil(32) * 32;
        buf.resize(aligned, 0);

        for f in [1.0f32, 2.0, 3.0, 4.0] {
            buf.extend_from_slice(&f.to_le_bytes());
        }

        let scale_bits = half::f16::from_f32(1.0).to_bits();
        buf.push((scale_bits & 0xFF) as u8);
        buf.push(((scale_bits >> 8) & 0xFF) as u8);
        buf.extend(std::iter::repeat_n(0x88u8, 16));

        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    fn write_gguf_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }
}
