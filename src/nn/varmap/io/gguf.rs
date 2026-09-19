//! `VarMap` constructor reading a GGUF file.

use super::super::core::VarMap;
use crate::error::Result;
use crate::format::gguf::Gguf;
use numr::dtype::DType;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> VarMap<R> {
    /// Load all tensors from a GGUF file.
    ///
    /// Unquantized tensors (F32, F16, BF16) are loaded as `Weight::Standard`.
    /// Quantized tensors (Q4_0, Q4K, etc.) are loaded as `Weight::Quantized`.
    ///
    /// Names are mapped with
    /// [`gguf_to_hf_name_for_arch`](crate::format::gguf::gguf_to_hf_name_for_arch)
    /// keyed on `general.architecture`, so an architecture with its own
    /// layer table (`qwen35`) gets it and every other file keeps the global
    /// mapping.
    pub fn from_gguf<P: AsRef<std::path::Path>>(path: P, device: &R::Device) -> Result<Self>
    where
        R::Client: numr::ops::ShapeOps<R>,
    {
        use crate::format::gguf::gguf_to_hf_name_for_arch;

        let mut gguf = Gguf::open(path)?;
        let arch = gguf.metadata().architecture().map(str::to_string);
        let names: Vec<String> = gguf.tensor_names().map(|s| s.to_string()).collect();
        let mut map = Self::new();
        let mut quant_formats = std::collections::HashSet::new();

        for name in &names {
            let hf_name = gguf_to_hf_name_for_arch(arch.as_deref(), name);
            let info = gguf.tensor_info(name)?.clone();
            if let Some(fmt) = info.ggml_type.to_quant_format() {
                quant_formats.insert(fmt);
            }
            if info.ggml_type.is_quantized() {
                let qt = gguf.load_tensor_quantized::<R>(name, device)?;
                map.insert_quant(hf_name, qt);
            } else {
                let t = gguf.load_tensor_f32::<R>(name, device)?;
                map.insert(hf_name, t);
            }
        }
        map.set_quant_formats(quant_formats.into_iter().collect());

        // Stack per-expert MoE tensors into single stacked tensors.
        // GGUF stores experts individually (experts.0.gate_proj.weight, experts.1.gate_proj.weight, ...)
        // but the model expects stacked tensors (experts.gate_proj.weight with shape [num_experts, ...]).
        Self::stack_moe_experts(&mut map, device)?;

        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::QuantFormat;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn device() -> CpuDevice {
        CpuDevice::new()
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

        assert_eq!(map.quant_formats(), &[QuantFormat::Q4_0]);
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
