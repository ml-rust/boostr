//! Owning `take_*` accessors: remove an entry from the map and hand it to the
//! caller, optionally as a layer, dequantized, or as a TP shard.

use crate::error::{Error, Result};
use crate::nn::linear::MaybeQuantLinear;
use crate::nn::weight::Weight;
use crate::quant::tensor::QuantTensor;
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::VarBuilder;

impl<R: Runtime> VarBuilder<'_, R> {
    /// Take a standard tensor by name, removing it from the map (zero-copy).
    pub fn take_tensor(&mut self, name: &str) -> Result<Tensor<R>> {
        let full = self.full_name(name);
        self.varmap.take_tensor(&full)
    }

    /// Take a standard tensor by name if it exists, returning `None` if absent.
    ///
    /// Useful for tensors that only exist in some architectures (e.g., attention
    /// biases, Q/K layer norms for Command-R).
    pub fn take_tensor_optional(&mut self, name: &str) -> Result<Option<Tensor<R>>> {
        if self.contains(name) {
            self.take_tensor(name).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Take a weight by name if it exists, returning `None` if absent.
    pub fn take_weight_optional(&mut self, name: &str) -> Result<Option<Weight<R>>> {
        if self.contains(name) {
            self.take_weight(name).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Take a `MaybeQuantLinear` if the weight exists, returning `None` if absent.
    pub fn take_maybe_quant_linear_optional(
        &mut self,
        name: &str,
        bias_name: Option<&str>,
    ) -> Result<Option<MaybeQuantLinear<R>>> {
        if self.contains(name) {
            self.take_maybe_quant_linear(name, bias_name).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Take a quantized tensor by name, removing it from the map (zero-copy).
    pub fn take_quant_tensor(&mut self, name: &str) -> Result<QuantTensor<R>> {
        let full = self.full_name(name);
        self.varmap.take_quant_tensor(&full)
    }

    /// Take a weight (standard or quantized) by name, removing it from the map.
    pub fn take_weight(&mut self, name: &str) -> Result<Weight<R>> {
        let full = self.full_name(name);
        self.varmap.take(&full)
    }

    /// Take a weight and construct a `MaybeQuantLinear` from it.
    ///
    /// If `bias_name` is provided, attempts to take a standard tensor for bias.
    ///
    /// The dense variant's weight is bound to the checkpoint key it was just
    /// read from, for an importance collection — see
    /// [`crate::quant::imatrix`]. This is the ONE place the binding can be
    /// made without guessing: the full dotted name is right here, and it is
    /// the same string the quantizer will see. The call is a no-op unless a
    /// collection is armed, and it never runs in a forward pass.
    pub fn take_maybe_quant_linear(
        &mut self,
        name: &str,
        bias_name: Option<&str>,
    ) -> Result<MaybeQuantLinear<R>> {
        let full = self.full_name(name);
        let weight = self.take_weight(name)?;
        let bias = match bias_name {
            Some(bn) => {
                if self.contains(bn) {
                    Some(self.take_tensor(bn)?)
                } else {
                    None
                }
            }
            None => None,
        };
        let layer = MaybeQuantLinear::from_weight(weight, bias);
        if let MaybeQuantLinear::Standard(linear) = &layer {
            crate::quant::imatrix::register_name(linear.weight().id(), &full);
        }
        Ok(layer)
    }

    /// Take a tensor by name, dequantizing if it's quantized.
    ///
    /// Useful for weights like embeddings that must be standard tensors
    /// but may be stored quantized in GGUF files.
    pub fn take_tensor_dequant(&mut self, name: &str, target_dtype: DType) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: DequantOps<R>,
    {
        match self.take_weight(name)? {
            Weight::Standard(t) => Ok(t),
            Weight::Quantized(qt) => {
                let client = R::default_client(self.device);
                client.dequantize(&qt, target_dtype)
            }
            Weight::DecomposedQuant(_) => Err(Error::ModelError {
                reason: "cannot dequantize decomposed quantized tensor to standard tensor".into(),
            }),
        }
    }

    /// Take a tensor and narrow it along `dim` for the given TP rank.
    ///
    /// Takes the full tensor from the VarMap, narrows to the rank's shard
    /// along `dim`, returns contiguous shard. The full tensor is removed
    /// from the VarMap (zero-copy take, then narrow).
    ///
    /// Column-parallel uses dim=0, row-parallel uses dim=1.
    pub fn take_tensor_shard(
        &mut self,
        name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<Tensor<R>> {
        let full = self.take_tensor(name)?;
        let shape = full.shape();

        if dim >= shape.len() {
            return Err(Error::ModelError {
                reason: format!(
                    "take_tensor_shard: dim {} out of range for {}D tensor '{}'",
                    dim,
                    shape.len(),
                    name
                ),
            });
        }

        let dim_size = shape[dim];
        if dim_size % world_size != 0 {
            return Err(Error::ModelError {
                reason: format!(
                    "take_tensor_shard: dim {} size ({}) not divisible by world_size ({}) for '{}'",
                    dim, dim_size, world_size, name
                ),
            });
        }

        let shard_size = dim_size / world_size;
        let start = rank * shard_size;

        full.narrow(dim as isize, start, shard_size)
            .and_then(|t| t.contiguous())
            .map_err(|e| Error::ModelError {
                reason: format!("take_tensor_shard narrow failed for '{}': {e}", name),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::varmap::VarMap;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn device() -> CpuDevice {
        CpuDevice::new()
    }

    #[test]
    fn test_varbuilder_take_tensor() {
        let d = device();
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "layer.weight".into(),
            Tensor::from_slice(&[1.0f32, 2.0], &[2], &d).unwrap(),
        );

        let mut vb = VarBuilder::new(&mut map, &d);
        let mut vb = vb.pp("layer");
        let t = vb.take_tensor("weight").unwrap();
        assert_eq!(t.shape(), &[2]);
        // Second take should fail — already removed
        assert!(vb.take_tensor("weight").is_err());
    }

    #[test]
    fn test_varbuilder_take_tensor_shard() {
        let d = device();
        let mut map = VarMap::<CpuRuntime>::new();
        // [4, 6] weight
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        map.insert(
            "weight".into(),
            Tensor::from_slice(&data, &[4, 6], &d).unwrap(),
        );

        let vb = VarBuilder::new(&mut map, &d);

        // Column-parallel shard (dim=0, rank=0, world_size=2) → [2, 6]
        // Re-insert since take removes it
        let data2: Vec<f32> = (0..24).map(|i| i as f32).collect();
        drop(vb);
        map.insert(
            "weight".into(),
            Tensor::from_slice(&data2, &[4, 6], &d).unwrap(),
        );
        let mut vb = VarBuilder::new(&mut map, &d);
        let shard = vb.take_tensor_shard("weight", 0, 0, 2).unwrap();
        assert_eq!(shard.shape(), &[2, 6]);

        // Row-parallel shard (dim=1, rank=1, world_size=2) → [4, 3]
        let data3: Vec<f32> = (0..24).map(|i| i as f32).collect();
        drop(vb);
        map.insert(
            "weight".into(),
            Tensor::from_slice(&data3, &[4, 6], &d).unwrap(),
        );
        let mut vb = VarBuilder::new(&mut map, &d);
        let shard = vb.take_tensor_shard("weight", 1, 1, 2).unwrap();
        assert_eq!(shard.shape(), &[4, 3]);
    }

    #[test]
    fn test_varbuilder_take_tensor_shard_not_divisible() {
        let d = device();
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "weight".into(),
            Tensor::from_slice(&[1.0f32; 15], &[3, 5], &d).unwrap(),
        );
        let mut vb = VarBuilder::new(&mut map, &d);
        // 3 not divisible by 2
        assert!(vb.take_tensor_shard("weight", 0, 0, 2).is_err());
    }
}
