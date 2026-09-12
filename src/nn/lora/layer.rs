//! The [`LoraLinear`] type: construction, accessors, and in-place adapter
//! updates.

use crate::error::{Error, Result};
use crate::nn::MaybeQuantLinear;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

/// LoRA adapter wrapping a frozen base linear layer.
///
/// The base is [`MaybeQuantLinear`] rather than [`Linear`](crate::nn::Linear)
/// because LoRA's base is frozen by construction — only `lora_a`/`lora_b`
/// train. A frozen weight does not need to be dense: it can just as well be a
/// block-quantized GGUF weight (`MaybeQuantLinear::Quantized`) or a
/// decomposed AWQ/GPTQ one. A dense trainable adapter riding on a frozen
/// quantized base is exactly QLoRA, and it means fine-tuning can run directly
/// on a quantized checkpoint without ever dequantizing the base weights.
/// `MaybeQuantLinear::Standard` covers the plain dense case exactly, so this
/// is a strict generalization — no behavior change for existing dense-base
/// callers.
pub struct LoraLinear<R: Runtime> {
    /// Frozen base linear layer — dense, block-quantized, or decomposed.
    pub(super) base: MaybeQuantLinear<R>,
    /// Low-rank down-projection: [rank, in_features]
    pub(super) lora_a: Var<R>,
    /// Low-rank up-projection: [out_features, rank]
    pub(super) lora_b: Var<R>,
    /// Scaling factor: alpha / rank
    pub(super) scaling: f32,
}

impl<R: Runtime<DType = DType>> LoraLinear<R> {
    /// Create a LoRA adapter around an existing linear layer.
    ///
    /// - `base`: The frozen base linear layer — a plain `Linear` converts in
    ///   automatically (dense case), or pass a `MaybeQuantLinear` directly
    ///   for a quantized base (QLoRA)
    /// - `rank`: Low-rank dimension (typical: 4, 8, 16, 32)
    /// - `alpha`: Scaling factor (typical: rank or 2*rank)
    /// - `device`: Device to allocate LoRA weights on
    pub fn new(
        base: impl Into<MaybeQuantLinear<R>>,
        rank: usize,
        alpha: f32,
        device: &R::Device,
    ) -> Result<Self> {
        let base = base.into();
        let shape = base.shape();
        let out_features = *shape.first().ok_or_else(|| Error::ModelError {
            reason: "LoRA base weight has no dimensions to derive out_features from".into(),
        })?;
        let in_features = *shape.get(1).ok_or_else(|| Error::ModelError {
            reason: "LoRA base weight must have at least 2 dimensions [out_features, in_features]"
                .into(),
        })?;

        // Initialize A with Kaiming uniform (simple LCG PRNG), B with zeros (standard LoRA init)
        let a_data = {
            let bound = (1.0 / in_features as f64).sqrt() as f32;
            let mut state: u64 = 42;
            let data: Vec<f32> = (0..rank * in_features)
                .map(|_| {
                    // Simple LCG for deterministic init
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = (state >> 33) as f32 / (1u64 << 31) as f32; // [0, 1)
                    (u * 2.0 - 1.0) * bound
                })
                .collect();
            data
        };

        let lora_a = Var::new(
            Tensor::from_slice(&a_data, &[rank, in_features], device)?,
            true,
        );
        let lora_b = Var::new(
            Tensor::zeros(&[out_features, rank], DType::F32, device)?,
            true,
        );

        Ok(Self {
            base,
            lora_a,
            lora_b,
            scaling: alpha / rank as f32,
        })
    }

    /// Create from pre-loaded LoRA weights.
    ///
    /// `trainable` controls gradient tracking on `lora_a`/`lora_b`: `false` for
    /// inference or merge-only use (the adapter is fixed), `true` to resume
    /// training this adapter further (e.g. continued fine-tuning after a
    /// checkpoint load). The base layer's own trainability is unaffected —
    /// pass it separately when constructing `base`.
    pub fn from_weights(
        base: impl Into<MaybeQuantLinear<R>>,
        lora_a: Tensor<R>,
        lora_b: Tensor<R>,
        alpha: f32,
        trainable: bool,
    ) -> Self {
        let rank = lora_a.shape()[0];
        Self {
            base: base.into(),
            lora_a: Var::new(lora_a, trainable),
            lora_b: Var::new(lora_b, trainable),
            scaling: alpha / rank as f32,
        }
    }

    /// Create from adapter tensors while preserving stable autograd IDs.
    ///
    /// The stable-id counterpart of [`Self::from_weights`], mirroring
    /// [`Linear::with_ids`](crate::nn::Linear::with_ids). `Tensor::clone`
    /// mints a fresh tensor id, so a rebuild that routes optimizer-updated
    /// tensors through `from_weights` would hand every adapter a NEW
    /// `TensorId` each step — detaching the optimizer state keyed by that id
    /// and turning a resumed LoRA run into a fresh, never-converging one.
    /// Callers that rebuild a model from a `TensorId`-keyed parameter map
    /// must use this.
    ///
    /// `trainable` applies to `lora_a`/`lora_b` only; the base carries its own
    /// flag from however `base` was constructed.
    pub fn with_ids(
        base: impl Into<MaybeQuantLinear<R>>,
        lora_a: Tensor<R>,
        lora_a_id: TensorId,
        lora_b: Tensor<R>,
        lora_b_id: TensorId,
        alpha: f32,
        trainable: bool,
    ) -> Self {
        let rank = lora_a.shape()[0];
        Self {
            base: base.into(),
            lora_a: Var::with_id(lora_a, lora_a_id, trainable),
            lora_b: Var::with_id(lora_b, lora_b_id, trainable),
            scaling: alpha / rank as f32,
        }
    }

    /// The low-rank down-projection factor `[rank, in_features]`.
    pub fn lora_a(&self) -> &Var<R> {
        &self.lora_a
    }

    /// The low-rank up-projection factor `[out_features, rank]`.
    pub fn lora_b(&self) -> &Var<R> {
        &self.lora_b
    }

    /// Get reference to the base linear layer.
    pub fn base(&self) -> &MaybeQuantLinear<R> {
        &self.base
    }

    /// Overwrite `lora_a`/`lora_b` in place with new VALUES, keeping their
    /// stable [`TensorId`]s — the in-place counterpart of [`Self::with_ids`]
    /// for a training loop that keeps the same `LoraLinear` alive across
    /// steps instead of rebuilding one.
    ///
    /// An optimizer step (e.g. `AdamW::step`) is FUNCTIONAL: it writes a
    /// brand-new `Tensor<R>` (a fresh internal id) into the caller's
    /// `TensorId`-keyed param map, never mutating the old one in place. A
    /// naive `Var::new(new_tensor, true)` would inherit that fresh id and
    /// silently detach this adapter from the optimizer state and from the
    /// autograd graph's own id-keyed bookkeeping on the NEXT step. This pins
    /// the original `lora_a_id`/`lora_b_id` back on, exactly like
    /// `with_ids` does at construction.
    pub fn set_adapters_with_ids(
        &mut self,
        lora_a: Tensor<R>,
        lora_a_id: TensorId,
        lora_b: Tensor<R>,
        lora_b_id: TensorId,
    ) {
        self.lora_a = Var::with_id(lora_a, lora_a_id, true);
        self.lora_b = Var::with_id(lora_b, lora_b_id, true);
    }

    /// The base weight, if it is `Var`-wrapped — i.e. only when the base is
    /// dense (`MaybeQuantLinear::Standard`). A quantized base has no
    /// `Var<R>` weight: block-quantized storage carries nothing trainable,
    /// so `None` here signals "quantized base", not an error.
    pub fn weight(&self) -> Option<&Var<R>> {
        self.base.weight()
    }

    /// Get LoRA rank.
    pub fn rank(&self) -> usize {
        self.lora_a.tensor().shape()[0]
    }

    /// Get scaling factor.
    pub fn scaling(&self) -> f32 {
        self.scaling
    }

    /// Set `lora_a`/`lora_b`'s `requires_grad` to `trainable`, in place.
    ///
    /// A LoRA adapter LOADED FROM A FILE for inference must NOT record an
    /// autograd graph on every forward: `requires_grad = true` on either
    /// factor makes every op downstream of it tracked
    /// (`numr::autograd::var_ops`), and dropping that graph after inference
    /// (nothing ever calls `backward`) recurses over every recorded node —
    /// deep enough to overflow a bounded worker stack (2 MB tokio workers,
    /// observed in blazr). Call `set_trainable(false)` right after loading a
    /// frozen adapter; a caller that resumes fine-tuning it calls
    /// `set_trainable(true)` instead.
    pub fn set_trainable(&mut self, trainable: bool) {
        self.lora_a.set_requires_grad(trainable);
        self.lora_b.set_requires_grad(trainable);
    }

    /// `true` when `lora_a`/`lora_b` currently require grad. `LoraLinear::new`
    /// and `from_weights`/`with_ids` always set both factors to the SAME
    /// flag, so reading `lora_a` alone reflects both.
    pub fn is_trainable(&self) -> bool {
        self.lora_a.requires_grad()
    }

    /// Cheap duplicate that preserves `base`'s and `lora_a`/`lora_b`'s
    /// `TensorId`s, for capturing this adapter by owned value in a
    /// `'static` activation-checkpointing closure. `lora_a`/`lora_b` go
    /// through [`Var::alias`] — never [`Clone`] — so the optimizer, keyed by
    /// `TensorId`, still sees their gradients after the alias is dropped.
    pub fn alias(&self) -> Self {
        Self {
            base: self.base.alias(),
            lora_a: self.lora_a.alias(),
            lora_b: self.lora_b.alias(),
            scaling: self.scaling,
        }
    }
}

/// Build a `LoraLinear` whose frozen base is a Q6_K block-quantized weight
/// (`in_features` must be a multiple of Q6_K's 256-element block size).
///
/// Shared by the QLoRA tests in the sibling concern files.
#[cfg(test)]
pub(super) fn quantized_lora(
    client: &numr::runtime::cpu::CpuClient,
    device: &numr::runtime::cpu::CpuDevice,
    out_features: usize,
    in_features: usize,
    rank: usize,
    alpha: f32,
) -> LoraLinear<numr::runtime::cpu::CpuRuntime> {
    use crate::nn::linear::{MaybeQuantLinear, QuantLinear};
    use crate::quant::format::QuantFormat;
    use crate::quant::traits::QuantizeOps;
    use numr::runtime::cpu::CpuRuntime;

    let base_w: Vec<f32> = (0..out_features * in_features)
        .map(|i| (i as f32 * 0.013).sin() * 0.3)
        .collect();
    let base_tensor =
        Tensor::<CpuRuntime>::from_slice(&base_w, &[out_features, in_features], device).unwrap();
    let quant = client
        .quantize(&base_tensor, QuantFormat::Q6K)
        .expect("Q6_K quantize");
    let base = MaybeQuantLinear::Quantized(QuantLinear::new(quant, None));

    LoraLinear::new(base, rank, alpha, device).expect("lora new must succeed over a quantized base")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_lora_linear_creation() {
        let device = <CpuRuntime as Runtime>::default_device();
        let weight: Tensor<CpuRuntime> = Tensor::zeros(&[64, 32], DType::F32, &device).unwrap();
        let base = Linear::new(weight, None, false);
        let lora = LoraLinear::new(base, 8, 16.0, &device).expect("lora new must succeed on CPU");
        assert_eq!(lora.rank(), 8);
        assert!((lora.scaling() - 2.0).abs() < 1e-6); // alpha/rank = 16/8 = 2
    }

    /// `with_ids` must preserve BOTH supplied `TensorId`s exactly — this is its
    /// entire reason to exist over `from_weights`. A resumed run rebuilds
    /// `LoraLinear` from a `TensorId`-keyed optimizer-state map each step; if the
    /// ids drifted, the rebuilt adapter would detach from its optimizer state.
    #[test]
    fn test_with_ids_preserves_supplied_ids() {
        let device = <CpuRuntime as Runtime>::default_device();
        let make_base = || {
            let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device).unwrap();
            Linear::new(weight, None, false)
        };
        let a = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 6], &[2, 3], &device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&[0.2f32; 4], &[2, 2], &device).unwrap();
        let a_id = TensorId::new();
        let b_id = TensorId::new();

        let lora = LoraLinear::with_ids(make_base(), a, a_id, b, b_id, 4.0, true);

        assert_eq!(lora.lora_a().id(), a_id);
        assert_eq!(lora.lora_b().id(), b_id);
    }

    /// Contrast case: a resumed run reads adapter tensors OUT of a
    /// `TensorId`-keyed optimizer-state map by reference, so it must `.clone()`
    /// before handing them to `from_weights` — and `Tensor::clone` mints a FRESH
    /// `TensorId` (`numr::tensor::core::Tensor::clone`, confirmed by reading its
    /// impl: `Self { id: TensorId::new(), .. }`). This is precisely the
    /// detachment `with_ids` exists to avoid: rebuilding via `from_weights` from
    /// a stored map hands the adapter a new id every step. Proves the two
    /// constructors are NOT interchangeable for that caller.
    #[test]
    fn test_from_weights_mints_fresh_ids_unlike_with_ids() {
        let device = <CpuRuntime as Runtime>::default_device();
        let make_base = || {
            let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device).unwrap();
            Linear::new(weight, None, false)
        };
        // Simulates tensors held in a `TensorId`-keyed map, accessed by reference.
        let stored_a = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 6], &[2, 3], &device).unwrap();
        let stored_b = Tensor::<CpuRuntime>::from_slice(&[0.2f32; 4], &[2, 2], &device).unwrap();
        let stored_a_id = stored_a.id();
        let stored_b_id = stored_b.id();

        let lora =
            LoraLinear::from_weights(make_base(), stored_a.clone(), stored_b.clone(), 4.0, true);

        assert_ne!(
            lora.lora_a().id(),
            stored_a_id,
            "from_weights unexpectedly preserved the cloned lora_a id — the contrast \
             case no longer holds and with_ids may be redundant"
        );
        assert_ne!(
            lora.lora_b().id(),
            stored_b_id,
            "from_weights unexpectedly preserved the cloned lora_b id — the contrast \
             case no longer holds and with_ids may be redundant"
        );
    }

    /// `trainable` must apply to both factors, in both directions, via `with_ids`.
    #[test]
    fn test_with_ids_trainable_flag() {
        let device = <CpuRuntime as Runtime>::default_device();
        let make_base = || {
            let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device).unwrap();
            Linear::new(weight, None, false)
        };
        let a = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 6], &[2, 3], &device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&[0.2f32; 4], &[2, 2], &device).unwrap();

        let frozen = LoraLinear::with_ids(
            make_base(),
            a.clone(),
            TensorId::new(),
            b.clone(),
            TensorId::new(),
            4.0,
            false,
        );
        assert!(!frozen.lora_a().requires_grad());
        assert!(!frozen.lora_b().requires_grad());

        let trainable = LoraLinear::with_ids(
            make_base(),
            a,
            TensorId::new(),
            b,
            TensorId::new(),
            4.0,
            true,
        );
        assert!(trainable.lora_a().requires_grad());
        assert!(trainable.lora_b().requires_grad());
    }

    /// `rank()` and `scaling()` must derive from the supplied tensors/`alpha`,
    /// not from any assumption baked into `LoraLinear::new`'s init path.
    #[test]
    fn test_with_ids_derives_rank_and_scaling() {
        let device = <CpuRuntime as Runtime>::default_device();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 20], &[4, 5], &device).unwrap();
        let base = Linear::new(weight, None, false);
        let (rank, in_features, out_features) = (5usize, 5usize, 4usize);
        let a = Tensor::<CpuRuntime>::from_slice(
            &vec![0.1f32; rank * in_features],
            &[rank, in_features],
            &device,
        )
        .unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(
            &vec![0.2f32; out_features * rank],
            &[out_features, rank],
            &device,
        )
        .unwrap();

        let lora = LoraLinear::with_ids(base, a, TensorId::new(), b, TensorId::new(), 15.0, true);

        assert_eq!(lora.rank(), rank);
        assert!((lora.scaling() - 3.0).abs() < 1e-6); // alpha/rank = 15/5 = 3
    }

    #[test]
    fn test_from_weights_trainable_flag() {
        let device = <CpuRuntime as Runtime>::default_device();
        let make_base = || {
            let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device).unwrap();
            Linear::new(weight, None, false)
        };
        let a = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 6], &[2, 3], &device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&[0.2f32; 4], &[2, 2], &device).unwrap();

        let frozen = LoraLinear::from_weights(make_base(), a.clone(), b.clone(), 4.0, false);
        assert!(!frozen.lora_a.requires_grad());
        assert!(!frozen.lora_b.requires_grad());

        let trainable = LoraLinear::from_weights(make_base(), a, b, 4.0, true);
        assert!(trainable.lora_a.requires_grad());
        assert!(trainable.lora_b.requires_grad());
    }
}
