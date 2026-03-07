//! VarMap: named collection of model weights (standard and quantized).

use crate::error::{Error, Result};
use crate::nn::weight::Weight;
use crate::quant::decomposed::DecomposedQuantTensor;
use crate::quant::tensor::QuantTensor;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;

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

    /// Insert a decomposed quantized tensor (AWQ/GPTQ).
    pub fn insert_decomposed_quant(&mut self, name: String, tensor: DecomposedQuantTensor<R>) {
        self.data
            .insert(name, Weight::DecomposedQuant(Box::new(tensor)));
    }

    /// Remove a weight by name.
    pub fn remove(&mut self, name: &str) -> Option<Weight<R>> {
        self.data.remove(name)
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

    /// Get a decomposed quantized tensor by name.
    pub fn get_decomposed_quant(&self, name: &str) -> Result<&DecomposedQuantTensor<R>> {
        self.get(name)?.as_decomposed_quant_tensor()
    }

    /// Remove and return a decomposed quantized tensor by name.
    pub fn take_decomposed_quant(&mut self, name: &str) -> Result<DecomposedQuantTensor<R>> {
        self.take(name)?.into_decomposed_quant_tensor()
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
            _ => None,
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

#[cfg(test)]
mod tests {
    use super::*;
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
}
