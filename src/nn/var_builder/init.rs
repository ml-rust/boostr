//! `take_or_init_tensor`: load-or-initialize, with name-derived per-tensor
//! seeds when the builder is seeded.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::VarBuilder;

impl<R: Runtime> VarBuilder<'_, R> {
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
mod tests {
    use super::*;
    use crate::nn::varmap::VarMap;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn device() -> CpuDevice {
        CpuDevice::new()
    }

    fn client() -> numr::runtime::cpu::CpuClient {
        let d = device();
        CpuRuntime::default_client(&d)
    }

    // ===== Seeded-path tests =====

    /// Two builders seeded identically must initialize the same (never-loaded)
    /// parameter to bit-identical values.
    #[test]
    fn test_varbuilder_with_seed_reproducible() {
        let d = device();
        let c = client();

        let mut map_a = VarMap::<CpuRuntime>::new();
        let mut vb_a = VarBuilder::new(&mut map_a, &d).with_seed(123);
        let t_a = vb_a
            .take_or_init_tensor(
                "weight",
                &[128, 64],
                DType::F32,
                crate::nn::Init::Kaiming,
                &c,
            )
            .unwrap();

        let mut map_b = VarMap::<CpuRuntime>::new();
        let mut vb_b = VarBuilder::new(&mut map_b, &d).with_seed(123);
        let t_b = vb_b
            .take_or_init_tensor(
                "weight",
                &[128, 64],
                DType::F32,
                crate::nn::Init::Kaiming,
                &c,
            )
            .unwrap();

        assert_eq!(t_a.to_vec::<f32>(), t_b.to_vec::<f32>());
    }

    /// The seed must survive `pp()` into a nested prefix: a child builder derived
    /// from a seeded parent must ALSO produce reproducible init, not silently
    /// fall back to unseeded randomness.
    #[test]
    fn test_varbuilder_seed_survives_push_prefix() {
        let d = device();
        let c = client();

        let mut map_a = VarMap::<CpuRuntime>::new();
        let mut root_a = VarBuilder::new(&mut map_a, &d).with_seed(456);
        let mut vb_a = root_a.pp("layers");
        let mut vb_a = vb_a.pp("0");
        let t_a = vb_a
            .take_or_init_tensor(
                "weight",
                &[32, 32],
                DType::F32,
                crate::nn::Init::PyTorchLinear,
                &c,
            )
            .unwrap();

        let mut map_b = VarMap::<CpuRuntime>::new();
        let mut root_b = VarBuilder::new(&mut map_b, &d).with_seed(456);
        let mut vb_b = root_b.pp("layers");
        let mut vb_b = vb_b.pp("0");
        let t_b = vb_b
            .take_or_init_tensor(
                "weight",
                &[32, 32],
                DType::F32,
                crate::nn::Init::PyTorchLinear,
                &c,
            )
            .unwrap();

        assert_eq!(t_a.to_vec::<f32>(), t_b.to_vec::<f32>());
    }

    /// Per-tensor seeds are derived from the NAME, not from call order: two
    /// differently-named tensors under the same base seed must get different
    /// values. If seeding instead used a shared incrementing counter, this
    /// would fail (both would land on the same counter value from a fresh
    /// builder), which is exactly the order-dependence bug name-derivation
    /// avoids.
    #[test]
    fn test_varbuilder_seed_is_name_derived_not_order_derived() {
        let d = device();
        let c = client();

        let mut map = VarMap::<CpuRuntime>::new();
        let mut vb = VarBuilder::new(&mut map, &d).with_seed(789);
        let t_a = vb
            .take_or_init_tensor(
                "weight_a",
                &[4096],
                DType::F32,
                crate::nn::Init::Randn {
                    mean: 0.0,
                    stdev: 1.0,
                },
                &c,
            )
            .unwrap();
        let t_b = vb
            .take_or_init_tensor(
                "weight_b",
                &[4096],
                DType::F32,
                crate::nn::Init::Randn {
                    mean: 0.0,
                    stdev: 1.0,
                },
                &c,
            )
            .unwrap();
        assert_ne!(t_a.to_vec::<f32>(), t_b.to_vec::<f32>());
    }

    /// A builder with no seed set behaves exactly like before `with_seed`
    /// existed: repeated init calls are NOT reproducible.
    #[test]
    fn test_varbuilder_without_seed_stays_unseeded() {
        let d = device();
        let c = client();

        let mut map_a = VarMap::<CpuRuntime>::new();
        let mut vb_a = VarBuilder::new(&mut map_a, &d);
        let t_a = vb_a
            .take_or_init_tensor(
                "weight",
                &[4096],
                DType::F32,
                crate::nn::Init::Randn {
                    mean: 0.0,
                    stdev: 1.0,
                },
                &c,
            )
            .unwrap();

        let mut map_b = VarMap::<CpuRuntime>::new();
        let mut vb_b = VarBuilder::new(&mut map_b, &d);
        let t_b = vb_b
            .take_or_init_tensor(
                "weight",
                &[4096],
                DType::F32,
                crate::nn::Init::Randn {
                    mean: 0.0,
                    stdev: 1.0,
                },
                &c,
            )
            .unwrap();

        assert_ne!(t_a.to_vec::<f32>(), t_b.to_vec::<f32>());
    }
}
