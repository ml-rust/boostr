//! VarMap: named collection of model weights (standard and quantized).

use crate::error::{Error, Result};
use crate::format::gguf::Gguf;
use crate::format::safetensors::SafeTensors;
use crate::nn::weight::Weight;
use crate::quant::tensor::QuantTensor;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;

/// Initialization strategy for new tensors.
#[derive(Debug, Clone, Copy)]
pub enum Init {
    /// All zeros
    Zeros,
    /// All ones
    Ones,
    /// Constant value
    Const(f32),
    /// Uniform random in `[-bound, bound]`
    Uniform(f32),
    /// Kaiming uniform (PyTorch Linear default): U(-1/sqrt(in), 1/sqrt(in))
    PyTorchLinear,
    /// PyTorch Embedding default: N(0, 1) approximated as U(-1, 1)
    PyTorchEmbedding,
}

/// Named collection of model weights (standard and quantized).
pub struct VarMap<R: Runtime> {
    data: HashMap<String, Weight<R>>,
}

impl<R: Runtime> VarMap<R> {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Insert a standard tensor.
    pub fn insert(&mut self, name: String, tensor: Tensor<R>) {
        self.data.insert(name, Weight::Standard(tensor));
    }

    /// Insert a quantized tensor.
    pub fn insert_quant(&mut self, name: String, tensor: QuantTensor<R>) {
        self.data.insert(name, Weight::Quantized(tensor));
    }

    /// Insert a weight directly.
    pub fn insert_weight(&mut self, name: String, weight: Weight<R>) {
        self.data.insert(name, weight);
    }

    /// Get a weight by name.
    pub fn get(&self, name: &str) -> Result<&Weight<R>> {
        self.data.get(name).ok_or_else(|| Error::ModelError {
            reason: format!("weight not found: {name}"),
        })
    }

    /// Get a standard tensor by name.
    pub fn get_tensor(&self, name: &str) -> Result<&Tensor<R>> {
        self.get(name)?.as_tensor()
    }

    /// Get a quantized tensor by name.
    pub fn get_quant_tensor(&self, name: &str) -> Result<&QuantTensor<R>> {
        self.get(name)?.as_quant_tensor()
    }

    /// Remove and return a weight by name (zero-copy extraction).
    pub fn take(&mut self, name: &str) -> Result<Weight<R>> {
        self.data.remove(name).ok_or_else(|| Error::ModelError {
            reason: format!("weight not found: {name}"),
        })
    }

    /// Remove and return a standard tensor by name (zero-copy extraction).
    pub fn take_tensor(&mut self, name: &str) -> Result<Tensor<R>> {
        self.take(name)?.into_tensor()
    }

    /// Remove and return a quantized tensor by name (zero-copy extraction).
    pub fn take_quant_tensor(&mut self, name: &str) -> Result<QuantTensor<R>> {
        self.take(name)?.into_quant_tensor()
    }

    /// Mutable access to a weight.
    pub fn get_mut(&mut self, name: &str) -> Result<&mut Weight<R>> {
        self.data.get_mut(name).ok_or_else(|| Error::ModelError {
            reason: format!("weight not found: {name}"),
        })
    }

    /// Set (overwrite) a weight.
    pub fn set(&mut self, name: String, weight: Weight<R>) {
        self.data.insert(name, weight);
    }

    /// All weight names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.data.keys().map(|s| s.as_str())
    }

    /// Number of weights.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over all weights.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Weight<R>)> {
        self.data.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// All standard tensors (for optimizers/serialization).
    pub fn all_tensors(&self) -> impl Iterator<Item = (&str, &Tensor<R>)> {
        self.data.iter().filter_map(|(k, v)| match v {
            Weight::Standard(t) => Some((k.as_str(), t)),
            Weight::Quantized(_) => None,
        })
    }

    /// Check if a name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.data.contains_key(name)
    }
}

impl<R: Runtime> Default for VarMap<R> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Format constructors ─────────────────────────────────────────────

impl<R: Runtime<DType = DType>> VarMap<R> {
    /// Load all tensors from a SafeTensors file.
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: &R::Device) -> Result<Self> {
        let mut st = SafeTensors::open(path)?;
        let all = st.load_all::<R>(device)?;
        let mut map = Self::new();
        for (name, tensor) in all {
            map.insert(name, tensor);
        }
        Ok(map)
    }

    /// Load all tensors from a GGUF file.
    ///
    /// Unquantized tensors (F32, F16, BF16) are loaded as `Weight::Standard`.
    /// Quantized tensors (Q4_0, Q4K, etc.) are loaded as `Weight::Quantized`.
    pub fn from_gguf<P: AsRef<Path>>(path: P, device: &R::Device) -> Result<Self> {
        let mut gguf = Gguf::open(path)?;
        let names: Vec<String> = gguf.tensor_names().map(|s| s.to_string()).collect();
        let mut map = Self::new();

        for name in &names {
            let info = gguf.tensor_info(name)?.clone();
            if info.ggml_type.is_quantized() {
                let qt = gguf.load_tensor_quantized::<R>(name, device)?;
                map.insert_quant(name.clone(), qt);
            } else {
                let t = gguf.load_tensor_f32::<R>(name, device)?;
                map.insert(name.clone(), t);
            }
        }

        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::safetensors::save_safetensors;
    use crate::quant::QuantFormat;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn device() -> CpuDevice {
        CpuDevice::new()
    }

    #[test]
    fn test_varmap_insert_and_get() {
        let d = device();
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert("w1".into(), Tensor::from_slice(&[1.0f32, 2.0], &[2], &d));
        assert_eq!(map.len(), 1);
        assert!(map.contains("w1"));
        assert!(!map.contains("w2"));

        let t = map.get_tensor("w1").unwrap();
        assert_eq!(t.shape(), &[2]);
    }

    #[test]
    fn test_varmap_mixed_weights() {
        let d = device();
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "norm.weight".into(),
            Tensor::from_slice(&[1.0f32], &[1], &d),
        );
        let data = vec![0u8; 18];
        let qt = QuantTensor::from_bytes(&data, QuantFormat::Q4_0, &[32], &d).unwrap();
        map.insert_quant("attn.weight".into(), qt);

        assert_eq!(map.len(), 2);
        assert!(!map.get("norm.weight").unwrap().is_quantized());
        assert!(map.get("attn.weight").unwrap().is_quantized());
    }

    #[test]
    fn test_varmap_take_tensor() {
        let d = device();
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert("w".into(), Tensor::from_slice(&[1.0f32, 2.0], &[2], &d));
        assert_eq!(map.len(), 1);

        let t = map.take_tensor("w").unwrap();
        assert_eq!(t.shape(), &[2]);
        assert_eq!(map.len(), 0);
        assert!(map.take_tensor("w").is_err());
    }

    #[test]
    fn test_varmap_names_and_iter() {
        let d = device();
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert("x".into(), Tensor::from_slice(&[1.0f32], &[1], &d));
        map.insert("y".into(), Tensor::from_slice(&[2.0f32], &[1], &d));

        let mut names: Vec<&str> = map.names().collect();
        names.sort();
        assert_eq!(names, vec!["x", "y"]);
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
