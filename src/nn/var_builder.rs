//! VarBuilder: scoped access to weights in a VarMap.
//!
//! Provides prefix-based navigation for hierarchical weight names
//! (e.g., "model.layers.0.self_attn.q_proj.weight").

use crate::error::{Error, Result};
use crate::nn::linear::MaybeQuantLinear;
use crate::nn::varmap::VarMap;
use crate::nn::weight::Weight;
use crate::quant::tensor::QuantTensor;
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
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
    /// Base seed for reproducible initialization, if set via `with_seed`.
    ///
    /// `None` means "no seeding requested" — `take_or_init_tensor` falls back
    /// to the unseeded `Init::init_tensor` exactly as before `with_seed`
    /// existed.
    seed: Option<u64>,
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

    /// Take a tensor by name, initializing it if the VarMap has no entry.
    ///
    /// This is the constructor path a trainer needs: the same model code both
    /// builds a fresh model (empty VarMap → initialize) and restores one from a
    /// checkpoint (populated VarMap → take). Without it, a model written with
    /// `take_tensor` can only ever load, and training from scratch fails with
    /// "weight not found".
    ///
    /// A present tensor is validated against `shape`, so a checkpoint that
    /// disagrees with the config fails loudly instead of silently mis-shaping
    /// the model.
    pub fn take_or_init_tensor<C>(
        &mut self,
        name: &str,
        shape: &[usize],
        dtype: DType,
        init: crate::nn::Init,
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
        let full = self.full_name(name);
        if self.varmap.contains(&full) {
            let tensor = self.varmap.take_tensor(&full)?;
            if tensor.shape() != shape {
                return Err(Error::ModelError {
                    reason: format!(
                        "shape mismatch for '{full}': config expects {shape:?}, \
                         loaded weight is {:?}",
                        tensor.shape()
                    ),
                });
            }
            return Ok(tensor);
        }
        match self.seed {
            // Per-tensor seed derived from (base seed, full parameter name) —
            // deliberately NOT a shared counter incremented on every call.
            // A shared stream makes every parameter's values depend on
            // construction ORDER: adding one optional submodule would
            // silently reseed every parameter created after it, with no
            // compiler or test signal. Name-derived seeds are order-
            // independent — adding or removing a parameter perturbs only its
            // own seed.
            Some(base) => {
                let seed = derive_seed(base, &full);
                init.init_tensor_seeded::<R, C>(shape, dtype, self.device, client, seed)
            }
            None => init.init_tensor::<R, C>(shape, dtype, self.device, client),
        }
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
    pub fn take_maybe_quant_linear(
        &mut self,
        name: &str,
        bias_name: Option<&str>,
    ) -> Result<MaybeQuantLinear<R>> {
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
        Ok(MaybeQuantLinear::from_weight(weight, bias))
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

/// Derive a per-tensor seed from a base seed and a parameter's full dotted
/// name (e.g. `"model.layers.0.self_attn.q_proj.weight"`).
///
/// Deliberately NOT `std::collections::hash_map::DefaultHasher` (used
/// elsewhere in this workspace for `compute_model_config_hash` in oxidizr):
/// the standard library explicitly does NOT guarantee that hasher's
/// algorithm is stable across Rust releases. `compute_model_config_hash`
/// only compares hashes computed by the same running binary, so that's fine
/// for it — but a reproducibility seed is a claim that persists across
/// rebuilds and compiler upgrades. If `DefaultHasher`'s algorithm ever
/// changed, every seeded checkpoint's initial weights would silently change
/// meaning on the next compiler upgrade, with no error and no warning.
///
/// This is instead a fixed, explicit mix: FNV-1a over the seed's bytes
/// followed by the name's bytes, finalized with a SplitMix64 avalanche step.
/// Every step is plain wrapping arithmetic with no dependency on any
/// standard-library hasher, so its output is fixed forever by this source
/// code — the exact property a reproducibility guarantee needs.
fn derive_seed(base: u64, name: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for byte in base.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    for byte in name.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    // SplitMix64 finalizer: spreads the FNV hash across the full 64 bits so
    // seeds for similar names (e.g. differing by one trailing digit) don't
    // stay close together.
    let mut z = hash.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests;
