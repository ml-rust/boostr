//! The [`VarBuilder`] type: construction, prefix navigation, and borrowing
//! getters.

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
    pub(super) varmap: &'a mut VarMap<R>,
    pub(super) prefix: String,
    pub(super) device: &'a R::Device,
    /// Base seed for reproducible initialization, if set via `with_seed`.
    ///
    /// `None` means "no seeding requested" — `take_or_init_tensor` falls back
    /// to the unseeded `Init::init_tensor` exactly as before `with_seed`
    /// existed.
    pub(super) seed: Option<u64>,
}

impl<'a, R: Runtime> VarBuilder<'a, R> {
    /// Create a root VarBuilder.
    pub fn new(varmap: &'a mut VarMap<R>, device: &'a R::Device) -> Self {
        Self {
            varmap,
            prefix: String::new(),
            device,
            seed: None,
        }
    }

    /// Request reproducible weight initialization from this builder onward.
    ///
    /// Every tensor `take_or_init_tensor` initializes (not loaded from a
    /// checkpoint) gets a per-tensor seed derived from `(seed, full_name)` —
    /// see `take_or_init_tensor` for why the derivation is name-based rather
    /// than a shared counter. The seed survives `pp()`/`push_prefix()` into
    /// every child builder, so calling `with_seed` once at the root seeds an
    /// entire model.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Create a sub-builder with an additional prefix component.
    ///
    /// The seed (if any) is carried into the child unchanged — losing it here
    /// would silently un-seed everything under this prefix.
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
            seed: self.seed,
        }
    }

    /// Alias for `push_prefix`.
    pub fn pp(&mut self, segment: &str) -> VarBuilder<'_, R> {
        self.push_prefix(segment)
    }

    /// Full name for a weight relative to this builder's prefix — the
    /// checkpoint key this builder reads `name` from.
    ///
    /// Public because a caller that REBUILDS a layer around a reshaped tensor
    /// (a grown or tied `lm_head`) mints a fresh autograd id and has to bind
    /// the same checkpoint key to it for an importance collection. Asking the
    /// builder is what keeps that key from being a second, drifting copy.
    pub fn full_name(&self, name: &str) -> String {
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

    /// Check if a name exists (relative to prefix).
    pub fn contains(&self, name: &str) -> bool {
        let full = self.full_name(name);
        self.varmap.contains(&full)
    }

    /// Current prefix.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

impl<R: Runtime> VarBuilder<'static, R> {
    /// Create a VarBuilder from a boxed VarMap.
    ///
    /// Takes ownership of the VarMap by boxing and leaking it to obtain a
    /// `'static` reference, which is required for `VarBuilder<'static, R>`.
    /// This is appropriate when the VarMap must outlive any particular scope.
    pub fn from_var_map(varmap: Box<VarMap<R>>, device: &'static R::Device) -> Self {
        let varmap_ref: &'static mut VarMap<R> = Box::leak(varmap);
        Self {
            varmap: varmap_ref,
            prefix: String::new(),
            device,
            seed: None,
        }
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
            Tensor::from_slice(&[1.0f32], &[1], &d).unwrap(),
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
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &d).unwrap(),
        );

        let vb = VarBuilder::new(&mut map, &d);
        assert!(vb.get_with_shape("w", &[2, 2]).is_ok());
        assert!(vb.get_with_shape("w", &[4]).is_err());
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
