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
        // A memory map lets each tensor's file pages leave resident memory
        // once copied, so a load never holds the file and the model at once.
        let mut gguf = Gguf::open_with_mmap(path, true)?;
        Self::from_gguf_reader(&mut gguf, device)
    }

    /// [`from_gguf`](Self::from_gguf) over a GGUF the caller already opened:
    /// a memory-mapped file, or bytes from [`Gguf::from_bytes`].
    pub fn from_gguf_reader(gguf: &mut Gguf, device: &R::Device) -> Result<Self>
    where
        R::Client: numr::ops::ShapeOps<R>,
    {
        use crate::format::gguf::gguf_to_hf_name_for_arch;

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
            // llama.cpp's `llama` files store query and key rows in pair
            // order. boostr rotates half-split, so the reader puts those rows
            // back on the host, before upload.
            if info.ggml_type.is_quantized() {
                let qt = gguf.load_tensor_quantized_hf_order::<R>(name, device)?;
                map.insert_quant(hf_name, qt);
            } else {
                let t = gguf.load_tensor_f32_hf_order::<R>(name, device)?;
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

    /// A `llama` file stores query/key rows in llama.cpp's pair order.
    /// `from_gguf` puts them back in HuggingFace order, for F32, F16, BF16
    /// and block-quantized rows alike, and leaves every other tensor as
    /// stored.
    #[test]
    fn llama_query_and_key_rows_return_to_hf_order() {
        use crate::format::gguf::rope_rows::{RopeRowOrder, rope_row_sources};
        use crate::format::gguf::types::{GgmlType, GgufValueType};
        use std::io::Write;

        // 2 query heads and 1 key head of size 4. Each stored row carries the
        // HF row index it came from, so the loaded order reads 0, 1, 2, ...
        let q_rows = rope_row_sources(8, 2, RopeRowOrder::ToGguf).unwrap();
        let k_rows = rope_row_sources(4, 1, RopeRowOrder::ToGguf).unwrap();
        let q8_row = |tag: usize| -> Vec<u8> {
            let mut row = half::f16::from_f32(tag as f32 + 1.0).to_le_bytes().to_vec();
            row.extend(std::iter::repeat_n(1u8, 32));
            row
        };

        let mut buf = Vec::new();
        buf.extend_from_slice(&0x46554747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&5u64.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());
        write_gguf_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_gguf_str(&mut buf, "llama");
        for (key, value) in [
            ("llama.attention.head_count", 2u32),
            ("llama.attention.head_count_kv", 1u32),
        ] {
            write_gguf_str(&mut buf, key);
            buf.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
            buf.extend_from_slice(&value.to_le_bytes());
        }
        let tensors: [(&str, &[u64], GgmlType, Vec<u8>); 5] = [
            (
                "blk.0.attn_q.bias",
                &[8],
                GgmlType::F32,
                q_rows
                    .iter()
                    .flat_map(|&r| (r as f32).to_le_bytes())
                    .collect(),
            ),
            (
                "blk.0.attn_k.bias",
                &[4],
                GgmlType::F16,
                k_rows
                    .iter()
                    .flat_map(|&r| half::f16::from_f32(r as f32).to_le_bytes())
                    .collect(),
            ),
            (
                "blk.0.attn_k.weight",
                &[2, 4],
                GgmlType::BF16,
                k_rows
                    .iter()
                    .flat_map(|&r| [r as f32, r as f32 + 0.5])
                    .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
                    .collect(),
            ),
            (
                "blk.0.attn_v.bias",
                &[4],
                GgmlType::F32,
                k_rows
                    .iter()
                    .flat_map(|&r| (r as f32).to_le_bytes())
                    .collect(),
            ),
            (
                "blk.0.attn_q.weight",
                &[32, 8],
                GgmlType::Q8_0,
                q_rows.iter().flat_map(|&r| q8_row(r)).collect(),
            ),
        ];
        let mut offset = 0u64;
        for (name, dims, ty, data) in &tensors {
            write_gguf_str(&mut buf, name);
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in *dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&(*ty as u32).to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
            offset += (data.len() as u64).div_ceil(32) * 32;
        }
        for (_, _, _, data) in &tensors {
            buf.resize(buf.len().div_ceil(32) * 32, 0);
            buf.extend_from_slice(data);
        }
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();

        let map = VarMap::<CpuRuntime>::from_gguf(file.path(), &device()).unwrap();
        let read =
            |name: &str| -> Vec<f32> { map.get(name).unwrap().as_tensor().unwrap().to_vec() };
        let iota = |n: usize| (0..n).map(|i| i as f32).collect::<Vec<_>>();
        assert_eq!(read("model.layers.0.self_attn.q_proj.bias"), iota(8));
        assert_eq!(read("model.layers.0.self_attn.k_proj.bias"), iota(4));
        let k_weight: Vec<f32> = (0..4).flat_map(|r| [r as f32, r as f32 + 0.5]).collect();
        assert_eq!(read("model.layers.0.self_attn.k_proj.weight"), k_weight);
        // The value projection keeps the stored order.
        let stored_v: Vec<f32> = k_rows.iter().map(|&r| r as f32).collect();
        assert_eq!(read("model.layers.0.self_attn.v_proj.bias"), stored_v);

        let qt = map
            .get("model.layers.0.self_attn.q_proj.weight")
            .unwrap()
            .as_quant_tensor()
            .unwrap();
        let bytes = qt.to_bytes().unwrap();
        let scales: Vec<f32> = bytes
            .chunks(34)
            .map(|row| half::f16::from_le_bytes([row[0], row[1]]).to_f32() - 1.0)
            .collect();
        assert_eq!(scales, iota(8));
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
