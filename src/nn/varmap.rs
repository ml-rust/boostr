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
    /// Kaiming (He) normal: N(0, sqrt(2 / fan_in))
    ///
    /// Standard initialization for ReLU networks. fan_in is the product of
    /// all dimensions except the last (output) dimension.
    Kaiming,
    /// Xavier (Glorot) normal: N(0, sqrt(2 / (fan_in + fan_out)))
    ///
    /// Standard initialization for Sigmoid/Tanh networks. Used in some
    /// attention weight initializations.
    Xavier,
    /// Normal distribution with given mean and standard deviation.
    Randn { mean: f64, stdev: f64 },
    /// Truncated normal: N(mean, stdev) clamped to [mean - 2*stdev, mean + 2*stdev]
    ///
    /// Used by GPT-2, BERT, and most modern LLMs for training stability.
    TruncatedNormal { mean: f64, stdev: f64 },
}

impl Init {
    /// Create a tensor initialized according to this strategy.
    ///
    /// # Arguments
    /// * `shape` - Shape of the tensor to create
    /// * `dtype` - Data type
    /// * `device` - Device to create on
    /// * `client` - Runtime client (needed for random ops)
    pub fn init_tensor<R, C>(
        &self,
        shape: &[usize],
        dtype: DType,
        device: &R::Device,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: numr::runtime::RuntimeClient<R>
            + numr::ops::RandomOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::TensorOps<R>,
    {
        // Trait bounds on the function provide the methods

        match *self {
            Init::Zeros => Ok(Tensor::zeros(shape, dtype, device)),
            Init::Ones => Ok(Tensor::ones(shape, dtype, device)),
            Init::Const(val) => {
                let t = Tensor::zeros(shape, dtype, device);
                client.add_scalar(&t, val as f64).map_err(Error::Numr)
            }
            Init::Uniform(bound) => {
                // U(-bound, bound) = rand() * 2*bound - bound
                let r = client.rand(shape, dtype).map_err(Error::Numr)?;
                let scaled = client
                    .mul_scalar(&r, 2.0 * bound as f64)
                    .map_err(Error::Numr)?;
                client
                    .add_scalar(&scaled, -(bound as f64))
                    .map_err(Error::Numr)
            }
            Init::PyTorchLinear => {
                // Kaiming uniform: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
                let fan_in = shape[0];
                let bound = 1.0 / (fan_in as f64).sqrt();
                let r = client.rand(shape, dtype).map_err(Error::Numr)?;
                let scaled = client.mul_scalar(&r, 2.0 * bound).map_err(Error::Numr)?;
                client.add_scalar(&scaled, -bound).map_err(Error::Numr)
            }
            Init::PyTorchEmbedding => {
                // N(0, 1) approximated as U(-1, 1)
                let r = client.rand(shape, dtype).map_err(Error::Numr)?;
                let scaled = client.mul_scalar(&r, 2.0).map_err(Error::Numr)?;
                client.add_scalar(&scaled, -1.0).map_err(Error::Numr)
            }
            Init::Kaiming => {
                // Kaiming/He normal: N(0, sqrt(2 / fan_in))
                let fan_in = if shape.len() >= 2 {
                    shape[..shape.len() - 1].iter().product::<usize>()
                } else {
                    shape[0]
                };
                let std = (2.0 / fan_in as f64).sqrt();
                let r = client.randn(shape, dtype).map_err(Error::Numr)?;
                client.mul_scalar(&r, std).map_err(Error::Numr)
            }
            Init::Xavier => {
                // Xavier/Glorot normal: N(0, sqrt(2 / (fan_in + fan_out)))
                let (fan_in, fan_out) = if shape.len() >= 2 {
                    let fi = shape[..shape.len() - 1].iter().product::<usize>();
                    let fo = shape[shape.len() - 1];
                    (fi, fo)
                } else {
                    (shape[0], shape[0])
                };
                let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
                let r = client.randn(shape, dtype).map_err(Error::Numr)?;
                client.mul_scalar(&r, std).map_err(Error::Numr)
            }
            Init::Randn { mean, stdev } => {
                let r = client.randn(shape, dtype).map_err(Error::Numr)?;
                let scaled = client.mul_scalar(&r, stdev).map_err(Error::Numr)?;
                if mean != 0.0 {
                    client.add_scalar(&scaled, mean).map_err(Error::Numr)
                } else {
                    Ok(scaled)
                }
            }
            Init::TruncatedNormal { mean, stdev } => {
                // Generate N(0, 1), clamp to [-2, 2], then scale by stdev and shift by mean
                let r = client.randn(shape, dtype).map_err(Error::Numr)?;
                let clamped = client.clamp(&r, -2.0, 2.0).map_err(Error::Numr)?;
                let scaled = client.mul_scalar(&clamped, stdev).map_err(Error::Numr)?;
                if mean != 0.0 {
                    client.add_scalar(&scaled, mean).map_err(Error::Numr)
                } else {
                    Ok(scaled)
                }
            }
        }
    }
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

    /// Load tensors from sharded SafeTensors files.
    ///
    /// Reads `model.safetensors.index.json` from `dir`, which maps tensor names
    /// to shard filenames (e.g., `model-00001-of-00004.safetensors`).
    /// Loads each shard once and extracts all its tensors.
    pub fn from_safetensors_sharded<P: AsRef<Path>>(dir: P, device: &R::Device) -> Result<Self> {
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
                map.insert(name.clone(), tensor);
            }
        }

        Ok(map)
    }

    /// Load all tensors from a GGUF file.
    ///
    /// Unquantized tensors (F32, F16, BF16) are loaded as `Weight::Standard`.
    /// Quantized tensors (Q4_0, Q4K, etc.) are loaded as `Weight::Quantized`.
    pub fn from_gguf<P: AsRef<Path>>(path: P, device: &R::Device) -> Result<Self> {
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

    fn client() -> numr::runtime::cpu::CpuClient {
        let d = device();
        CpuRuntime::default_client(&d)
    }

    #[test]
    fn test_init_zeros() {
        let d = device();
        let c = client();
        let t = Init::Zeros
            .init_tensor(&[2, 3], DType::F32, &d, &c)
            .unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        let data: Vec<f32> = t.to_vec();
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_init_kaiming() {
        let d = device();
        let c = client();
        // [out=64, in=128] → fan_in=128, std=sqrt(2/128)≈0.125
        let t = Init::Kaiming
            .init_tensor(&[64, 128], DType::F32, &d, &c)
            .unwrap();
        assert_eq!(t.shape(), &[64, 128]);
        let data: Vec<f32> = t.to_vec();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        // Mean should be close to 0
        assert!(mean.abs() < 0.1, "Kaiming mean too large: {mean}");
        // Std should be close to sqrt(2/128) ≈ 0.125
        let var: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std = var.sqrt();
        // fan_in = product of all dims except last = 64
        let expected_std = (2.0f32 / 64.0).sqrt();
        assert!(
            (std - expected_std).abs() < 0.05,
            "Kaiming std {std} vs expected {expected_std}"
        );
    }

    #[test]
    fn test_init_xavier() {
        let d = device();
        let c = client();
        // [256, 512] → fan_in=256, fan_out=512, std=sqrt(2/768)≈0.051
        let t = Init::Xavier
            .init_tensor(&[256, 512], DType::F32, &d, &c)
            .unwrap();
        assert_eq!(t.shape(), &[256, 512]);
        let data: Vec<f32> = t.to_vec();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 0.05, "Xavier mean too large: {mean}");
    }

    #[test]
    fn test_init_randn() {
        let d = device();
        let c = client();
        let t = Init::Randn {
            mean: 5.0,
            stdev: 0.1,
        }
        .init_tensor(&[1000], DType::F32, &d, &c)
        .unwrap();
        let data: Vec<f32> = t.to_vec();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!((mean - 5.0).abs() < 0.1, "Randn mean {mean} should be ~5.0");
    }

    #[test]
    fn test_init_truncated_normal() {
        let d = device();
        let c = client();
        let t = Init::TruncatedNormal {
            mean: 0.0,
            stdev: 0.02,
        }
        .init_tensor(&[10000], DType::F32, &d, &c)
        .unwrap();
        let data: Vec<f32> = t.to_vec();
        // All values should be within [-0.04, 0.04] (2*stdev)
        for &v in &data {
            assert!(
                (-0.04..=0.04).contains(&v),
                "Truncated normal value {v} out of range [-0.04, 0.04]"
            );
        }
    }
}
