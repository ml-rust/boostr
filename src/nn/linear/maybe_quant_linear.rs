use super::dense::Linear;
use super::quant_linear::QuantLinear;
use crate::error::Result;
use crate::nn::module::Module;
use crate::nn::weight::Weight;
use crate::quant::decomposed::DecomposedQuantLinear;
use crate::quant::tensor::QuantTensor;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{BinaryOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// A linear layer that works with either standard or quantized weights.
///
/// During inference with GGUF models, some weights are quantized (Q4_K_M etc.)
/// while others (norms, embeddings) remain in full precision. This enum lets
/// model structs use a single field type for both cases.
pub enum MaybeQuantLinear<R: Runtime> {
    Standard(Linear<R>),
    Quantized(QuantLinear<R>),
    DecomposedQuant(Box<DecomposedQuantLinear<R>>),
}

impl<R: Runtime> MaybeQuantLinear<R> {
    /// Construct from a `Weight` (standard, quantized, or decomposed) plus optional bias tensor.
    pub fn from_weight(weight: Weight<R>, bias: Option<Tensor<R>>) -> Self {
        match weight {
            Weight::Standard(t) => Self::Standard(Linear::new(t, bias, false)),
            Weight::Quantized(qt) => Self::Quantized(QuantLinear::new(qt, bias)),
            Weight::DecomposedQuant(dq) => {
                Self::DecomposedQuant(Box::new(DecomposedQuantLinear::new(*dq, bias)))
            }
        }
    }

    /// The base weight, if it is `Var`-wrapped — i.e. only for the dense
    /// `Standard` variant. Block-quantized and decomposed storage carry no
    /// trainable `Var<R>`, so this is `None` for those, not a panic.
    pub fn weight(&self) -> Option<&Var<R>> {
        match self {
            Self::Standard(linear) => Some(linear.weight()),
            Self::Quantized(_) | Self::DecomposedQuant(_) => None,
        }
    }

    /// The base bias, if it is `Var`-wrapped. Mirrors [`Self::weight`]: only
    /// the dense `Standard` variant's bias is `Var`-wrapped; a quantized
    /// bias (when present) is a plain frozen `Tensor<R>`.
    pub fn bias(&self) -> Option<&Var<R>> {
        match self {
            Self::Standard(linear) => linear.bias(),
            Self::Quantized(_) | Self::DecomposedQuant(_) => None,
        }
    }

    /// Forward pass: works for standard, quantized, and decomposed quantized weights.
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>,
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + DequantOps<R> + numr::ops::MatmulOps<R>,
    {
        match self {
            // The importance-matrix tap — see `crate::quant::imatrix`. Off,
            // it costs one relaxed load of a process-wide `AtomicBool` and a
            // branch never taken: no field here, no constructor threading, no
            // allocation. The DENSE arm only, because an importance matrix
            // guides the quantization of a dense weight.
            Self::Standard(linear) => {
                if crate::quant::imatrix::is_armed() {
                    crate::quant::imatrix::observe(linear.weight().id(), client, input.tensor())?;
                }
                linear.forward(client, input)
            }
            // Forward always uses the fast quantized kernel. When `input`
            // needs no gradient (inference), the output stays a detached
            // leaf — zero extra allocation, unchanged from before. When it
            // does (QLoRA training), `attach_quant_linear_backward` wires up
            // a node whose backward dequantizes the FROZEN weight only then
            // — see `crate::quant::autograd` for why the base weight itself
            // never gets a gradient.
            Self::Quantized(qlinear) => {
                let out = qlinear.forward(client, input.tensor())?;
                if input.requires_grad() {
                    crate::quant::attach_quant_linear_backward(input, out, qlinear.weight())
                } else {
                    Ok(Var::new(out, false))
                }
            }
            // AWQ/GPTQ packed layouts have no elementwise dequant op in
            // `DequantOps` (only fused `int4_gemm`/`int4_gemm_gptq`/
            // `marlin_gemm`, which take an activation, not just the weight),
            // so there is no existing op to build a clean input-gradient
            // from without guessing at backward math. Left detached, same as
            // before, until such an op exists.
            Self::DecomposedQuant(dqlinear) => {
                let out = dqlinear.forward(client, input.tensor())?;
                Ok(Var::new(out, false))
            }
        }
    }

    /// Batched forward: compute multiple projections sharing the same input.
    ///
    /// When all layers are block-quantized, uses `quant_matmul_batch` to amortize
    /// activation preprocessing (e.g. Q8_1 quantization on CUDA).
    /// For decomposed quantized layers, falls back to individual forward passes.
    pub fn forward_batch<C>(
        layers: &[&MaybeQuantLinear<R>],
        client: &C,
        input: &Var<R>,
    ) -> Result<Vec<Var<R>>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>,
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + DequantOps<R> + numr::ops::MatmulOps<R>,
    {
        // Check if all are block-quantized (no bias) — enables batch path
        let all_quantized_no_bias = layers
            .iter()
            .all(|l| matches!(l, MaybeQuantLinear::Quantized(ql) if ql.bias().is_none()));

        if all_quantized_no_bias {
            let weights: Vec<&QuantTensor<R>> = layers
                .iter()
                .map(|l| match l {
                    MaybeQuantLinear::Quantized(ql) => ql.weight(),
                    _ => unreachable!(),
                })
                .collect();

            let outputs = client.quant_matmul_batch(input.tensor(), &weights)?;
            // Same detach-vs-attach split as the single-layer path: only
            // pay for the graph node when `input` actually needs a gradient.
            if input.requires_grad() {
                outputs
                    .into_iter()
                    .zip(weights)
                    .map(|(out, weight)| {
                        crate::quant::attach_quant_linear_backward(input, out, weight)
                    })
                    .collect()
            } else {
                Ok(outputs.into_iter().map(|t| Var::new(t, false)).collect())
            }
        } else {
            // Fallback: individual forward passes
            layers.iter().map(|l| l.forward(client, input)).collect()
        }
    }

    /// All trainable-capable parameters with their stable autograd IDs.
    ///
    /// Quantized variants are inference-only and therefore expose no `Var`
    /// parameters.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        match self {
            Self::Standard(linear) => linear.parameters(),
            Self::Quantized(_) | Self::DecomposedQuant(_) => Vec::new(),
        }
    }

    /// Trainable parameters with their stable autograd IDs.
    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters()
            .into_iter()
            .filter(|param| param.1.requires_grad())
            .collect()
    }

    /// Named standard parameters for checkpoint traversal.
    pub fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        match self {
            Self::Standard(linear) => linear.named_parameters(),
            Self::Quantized(_) | Self::DecomposedQuant(_) => Vec::new(),
        }
    }

    /// Cheap duplicate that preserves every `Var<R>`'s `TensorId`, for
    /// capturing this layer by owned value in a `'static`
    /// activation-checkpointing closure. Every variant delegates to that
    /// variant's own `alias()`, so a dense base still routes its weight
    /// through [`Var::alias`] — never [`Clone`] — and a quantized or
    /// decomposed base (no `Var<R>` to preserve an id for) shares its
    /// underlying storage cheaply instead.
    pub fn alias(&self) -> Self {
        match self {
            Self::Standard(linear) => Self::Standard(linear.alias()),
            Self::Quantized(qlinear) => Self::Quantized(qlinear.alias()),
            Self::DecomposedQuant(dqlinear) => Self::DecomposedQuant(Box::new(dqlinear.alias())),
        }
    }
}

// `QuantTensor::shape` is only available under `DType = DType`, so `shape`
// lives in its own block rather than constraining every other method on
// `MaybeQuantLinear` (`from_weight`, `forward`, `parameters`) that does not
// need it.
impl<R: Runtime<DType = numr::dtype::DType>> MaybeQuantLinear<R> {
    /// Logical weight shape `[out_features, in_features]`, for every variant.
    ///
    /// A block-quantized or decomposed base has no `Var<R>` weight to read
    /// `.shape()` off of, but its logical element shape is tracked
    /// regardless — this is what lets a LoRA adapter size its low-rank
    /// factors from a frozen QUANTIZED base, without caring whether that
    /// base is dense or quantized.
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Standard(linear) => linear.weight().tensor().shape(),
            Self::Quantized(qlinear) => qlinear.weight().shape(),
            Self::DecomposedQuant(dqlinear) => dqlinear.weight().shape(),
        }
    }
}

impl<R: Runtime> From<Linear<R>> for MaybeQuantLinear<R> {
    fn from(linear: Linear<R>) -> Self {
        Self::Standard(linear)
    }
}

impl<R: Runtime> Module<R> for MaybeQuantLinear<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        MaybeQuantLinear::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        MaybeQuantLinear::named_parameters(self)
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        MaybeQuantLinear::parameters(self)
    }

    fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        MaybeQuantLinear::trainable_parameters(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// `MaybeQuantLinear::shape` must report the logical `[out, in]` shape for
    /// every variant, including the quantized ones that have no `Var` weight.
    #[test]
    fn test_maybe_quant_linear_shape_quantized() {
        use crate::quant::format::QuantFormat;
        use crate::quant::traits::QuantizeOps;

        let (client, device) = cpu_setup();
        let (out_features, in_features) = (4usize, 256usize);
        let data: Vec<f32> = (0..out_features * in_features)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let weight =
            Tensor::<CpuRuntime>::from_slice(&data, &[out_features, in_features], &device).unwrap();
        let quant = client.quantize(&weight, QuantFormat::Q6K).unwrap();
        let maybe = MaybeQuantLinear::Quantized(QuantLinear::new(quant, None));

        assert_eq!(maybe.shape(), &[out_features, in_features]);
        assert!(maybe.weight().is_none());
        assert!(maybe.bias().is_none());
    }

    // --- QLoRA backward: quantized `MaybeQuantLinear::forward` must keep the
    // input on the autograd graph, not detach it. -----------------------------

    /// Backward through `MaybeQuantLinear::Quantized` must match a dense
    /// reference built from the SAME dequantized weight.
    ///
    /// Using the dequantized weight on both sides isolates the backward
    /// FORMULA from quantization noise — both paths compute the identical
    /// forward function, so any gradient mismatch can only come from a wrong
    /// adjoint, not from Q8_0 round-trip error. This is what proves
    /// `QuantLinearBackward`'s math is right, not merely present.
    ///
    /// Tolerance: `atol=1e-4, rtol=1e-3`. Both paths dequantize to bit-identical
    /// weight data, so the only source of difference is f32 summation-order
    /// noise between two different matmul call sites — numr's built-in
    /// `MatmulBackward` for `Standard` vs `QuantLinearBackward`'s own
    /// `client.matmul` — not quantization error.
    #[test]
    fn test_quantized_backward_matches_dense_reference() {
        use crate::quant::format::QuantFormat;
        use crate::quant::traits::{DequantOps, QuantizeOps};
        use numr::autograd::{backward, var_sum};

        let (client, device) = cpu_setup();
        // 64 is a multiple of Q8_0's 32-element block size.
        let (batch, out_features, in_features) = (4usize, 6usize, 64usize);

        let weight_data: Vec<f32> = (0..out_features * in_features)
            .map(|i| ((i as f32) * 0.017).sin() * 0.5)
            .collect();
        let weight =
            Tensor::<CpuRuntime>::from_slice(&weight_data, &[out_features, in_features], &device)
                .unwrap();
        let quant = client.quantize(&weight, QuantFormat::Q8_0).unwrap();
        let dequant_weight = client.dequantize(&quant, DType::F32).unwrap();

        let standard = MaybeQuantLinear::Standard(Linear::new(dequant_weight, None, false));
        let quantized = MaybeQuantLinear::Quantized(QuantLinear::new(quant, None));

        let input_data: Vec<f32> = (0..batch * in_features)
            .map(|i| ((i as f32) * 0.031).cos() * 0.2)
            .collect();
        let x_std = Var::new(
            Tensor::<CpuRuntime>::from_slice(&input_data, &[batch, in_features], &device).unwrap(),
            true,
        );
        let x_q = Var::new(
            Tensor::<CpuRuntime>::from_slice(&input_data, &[batch, in_features], &device).unwrap(),
            true,
        );

        let out_std = standard.forward(&client, &x_std).unwrap();
        let out_q = quantized.forward(&client, &x_q).unwrap();
        assert!(out_q.requires_grad());

        let loss_std = var_sum(&out_std, &[0, 1], false, &client).unwrap();
        let loss_q = var_sum(&out_q, &[0, 1], false, &client).unwrap();

        let grads_std = backward(&loss_std, &client).unwrap();
        let grads_q = backward(&loss_q, &client).unwrap();

        let grad_std: Vec<f32> = grads_std.get(x_std.id()).unwrap().to_vec();
        let grad_q: Vec<f32> = grads_q.get(x_q.id()).unwrap().to_vec();

        assert_eq!(grad_std.len(), grad_q.len());
        for (a, b) in grad_std.iter().zip(grad_q.iter()) {
            let diff = (a - b).abs();
            assert!(
                diff <= 1e-4 + 1e-3 * a.abs(),
                "grad mismatch: standard={a}, quantized={b}, diff={diff}"
            );
        }
    }

    /// A quantized projection's output must stay on the autograd graph when its
    /// input requires grad, and the gradient must actually reach that input —
    /// the defect this module fixes was a silent `Var::new(out, false)` detach
    /// that made every quantized projection a dead end for backprop.
    #[test]
    fn test_quantized_forward_requires_grad_reaches_input() {
        use crate::quant::format::QuantFormat;
        use crate::quant::traits::QuantizeOps;
        use numr::autograd::{backward, var_sum};

        let (client, device) = cpu_setup();
        let (out_features, in_features) = (4usize, 32usize);
        let weight_data: Vec<f32> = (0..out_features * in_features)
            .map(|i| i as f32 * 0.01)
            .collect();
        let weight =
            Tensor::<CpuRuntime>::from_slice(&weight_data, &[out_features, in_features], &device)
                .unwrap();
        let quant = client.quantize(&weight, QuantFormat::Q8_0).unwrap();
        let quantized = MaybeQuantLinear::Quantized(QuantLinear::new(quant, None));

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.1f32; in_features],
                &[1, in_features],
                &device,
            )
            .unwrap(),
            true,
        );

        let out = quantized.forward(&client, &x).unwrap();
        assert!(
            out.requires_grad(),
            "output must require grad when input does"
        );

        let loss = var_sum(&out, &[0, 1], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        assert!(
            grads.get(x.id()).is_some(),
            "gradient must reach the quantized projection's input"
        );
    }

    /// Inference must be unaffected by the QLoRA backward wiring: with
    /// `requires_grad == false`, `MaybeQuantLinear::forward` must take the same
    /// cheap detached path as before — same values, no `requires_grad`, and no
    /// extra dequantize.
    #[test]
    fn test_quantized_forward_no_grad_stays_detached_and_matches_direct_call() {
        use crate::quant::format::QuantFormat;
        use crate::quant::traits::QuantizeOps;

        let (client, device) = cpu_setup();
        let (out_features, in_features) = (4usize, 32usize);
        let weight_data: Vec<f32> = (0..out_features * in_features)
            .map(|i| i as f32 * 0.01)
            .collect();
        let weight =
            Tensor::<CpuRuntime>::from_slice(&weight_data, &[out_features, in_features], &device)
                .unwrap();
        // Quantizing the same float weight twice is a deterministic, pure
        // function — the two `QuantTensor`s are byte-for-byte identical, so
        // this gives two independent layers computing the same thing without
        // reaching into `QuantTensor`'s private fields to clone one.
        let quant_a = client.quantize(&weight, QuantFormat::Q8_0).unwrap();
        let quant_b = client.quantize(&weight, QuantFormat::Q8_0).unwrap();
        let maybe = MaybeQuantLinear::Quantized(QuantLinear::new(quant_a, None));
        let direct = QuantLinear::new(quant_b, None);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.1f32; in_features],
                &[1, in_features],
                &device,
            )
            .unwrap(),
            false,
        );

        let out = maybe.forward(&client, &x).unwrap();
        assert!(!out.requires_grad());

        let direct_out = direct.forward(&client, x.tensor()).unwrap();

        let via_maybe: Vec<f32> = out.tensor().to_vec();
        let via_direct: Vec<f32> = direct_out.to_vec();
        assert_eq!(
            via_maybe, via_direct,
            "requires_grad=false must take the same detached quant_matmul path as calling \
             QuantLinear::forward directly"
        );
    }
}
