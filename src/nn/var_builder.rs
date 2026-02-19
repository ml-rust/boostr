//! VarBuilder: scoped access to weights in a VarMap.
//!
//! Provides prefix-based navigation for hierarchical weight names
//! (e.g., "model.layers.0.self_attn.q_proj.weight").

use crate::error::{Error, Result};
use crate::nn::varmap::VarMap;
use crate::nn::weight::Weight;
use crate::quant::tensor::QuantTensor;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Scoped access to weights in a `VarMap`.
///
/// VarBuilder holds a reference to a VarMap and a prefix string for
/// navigating hierarchical weight names (e.g., "model.layers.0.self_attn").
pub struct VarBuilder<'a, R: Runtime> {
    varmap: &'a mut VarMap<R>,
    prefix: String,
    device: &'a R::Device,
}

impl<'a, R: Runtime> VarBuilder<'a, R> {
    /// Create a root VarBuilder.
    pub fn new(varmap: &'a mut VarMap<R>, device: &'a R::Device) -> Self {
        Self {
            varmap,
            prefix: String::new(),
            device,
        }
    }

    /// Create a sub-builder with an additional prefix component.
    pub fn push_prefix(&mut self, segment: &str) -> VarBuilder<'_, R> {
        let prefix = if self.prefix.is_empty() {
            segment.to_string()
        } else {
            format!("{}.{}", self.prefix, segment)
        };
        VarBuilder {
            varmap: self.varmap,
            prefix,
            device: self.device,
        }
    }

    /// Alias for `push_prefix`.
    pub fn pp(&mut self, segment: &str) -> VarBuilder<'_, R> {
        self.push_prefix(segment)
    }

    /// Full name for a weight relative to this builder's prefix.
    fn full_name(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        }
    }

    /// Get a weight by name (relative to prefix).
    pub fn get(&self, name: &str) -> Result<&Weight<R>> {
        let full = self.full_name(name);
        self.varmap.get(&full)
    }

    /// Get a standard tensor by name (relative to prefix).
    pub fn get_tensor(&self, name: &str) -> Result<&Tensor<R>> {
        let full = self.full_name(name);
        self.varmap.get_tensor(&full)
    }

    /// Get a quantized tensor by name.
    pub fn get_quant_tensor(&self, name: &str) -> Result<&QuantTensor<R>> {
        let full = self.full_name(name);
        self.varmap.get_quant_tensor(&full)
    }

    /// Take a standard tensor by name, removing it from the map (zero-copy).
    pub fn take_tensor(&mut self, name: &str) -> Result<Tensor<R>> {
        let full = self.full_name(name);
        self.varmap.take_tensor(&full)
    }

    /// Take a quantized tensor by name, removing it from the map (zero-copy).
    pub fn take_quant_tensor(&mut self, name: &str) -> Result<QuantTensor<R>> {
        let full = self.full_name(name);
        self.varmap.take_quant_tensor(&full)
    }

    /// Get a standard tensor and validate its shape.
    pub fn get_with_shape(&self, name: &str, expected_shape: &[usize]) -> Result<&Tensor<R>> {
        let full = self.full_name(name);
        let t = self.varmap.get_tensor(&full)?;
        if t.shape() != expected_shape {
            return Err(Error::ModelError {
                reason: format!(
                    "shape mismatch for '{}': expected {:?}, got {:?}",
                    full,
                    expected_shape,
                    t.shape()
                ),
            });
        }
        Ok(t)
    }

    /// Device for this builder.
    pub fn device(&self) -> &R::Device {
        self.device
    }

    /// Current prefix.
    pub fn prefix(&self) -> &str {
        &self.prefix
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
    fn test_varbuilder_prefix() {
        let d = device();
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            Tensor::from_slice(&[1.0f32], &[1], &d),
        );

        let mut vb = VarBuilder::new(&mut map, &d);
        let mut vb = vb.pp("model");
        let mut vb = vb.pp("layers");
        let mut vb = vb.pp("0");
        let vb = vb.pp("self_attn");
        let t = vb.get_tensor("q_proj.weight").unwrap();
        assert_eq!(t.shape(), &[1]);
    }

    #[test]
    fn test_varbuilder_get_with_shape() {
        let d = device();
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "w".into(),
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &d),
        );

        let vb = VarBuilder::new(&mut map, &d);
        assert!(vb.get_with_shape("w", &[2, 2]).is_ok());
        assert!(vb.get_with_shape("w", &[4]).is_err());
    }

    #[test]
    fn test_varbuilder_take_tensor() {
        let d = device();
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "layer.weight".into(),
            Tensor::from_slice(&[1.0f32, 2.0], &[2], &d),
        );

        let mut vb = VarBuilder::new(&mut map, &d);
        let mut vb = vb.pp("layer");
        let t = vb.take_tensor("weight").unwrap();
        assert_eq!(t.shape(), &[2]);
        // Second take should fail — already removed
        assert!(vb.take_tensor("weight").is_err());
    }

    #[test]
    fn test_varbuilder_quant_prefix() {
        let d = device();
        let mut map = VarMap::<CpuRuntime>::new();
        let data = vec![0u8; 18];
        let qt = QuantTensor::from_bytes(&data, QuantFormat::Q4_0, &[32], &d).unwrap();
        map.insert_quant("layers.0.weight".into(), qt);

        let mut vb = VarBuilder::new(&mut map, &d);
        let mut vb = vb.pp("layers");
        let vb = vb.pp("0");
        let qt = vb.get_quant_tensor("weight").unwrap();
        assert_eq!(qt.shape(), &[32]);
    }
}
