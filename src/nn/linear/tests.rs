use super::*;
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::CpuRuntime;

#[test]
fn test_linear_output_shape() {
    let (client, device) = cpu_setup();
    // weight: [out=4, in=3]
    let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[4, 3], &device).unwrap();
    let linear = Linear::new(weight, None, false);

    // input: [2, 3]
    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device).unwrap(),
        false,
    );
    let out = linear.forward(&client, &input).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
}

#[test]
fn test_linear_with_bias() {
    let (client, device) = cpu_setup();
    let weight =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap();
    let bias = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0], &[2], &device).unwrap();
    let linear = Linear::new(weight, Some(bias), false);

    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device).unwrap(),
        false,
    );
    let out = linear.forward(&client, &input).unwrap();
    let data: Vec<f32> = out.tensor().to_vec();
    // [1,2] @ [[1,0],[0,1]] + [10,20] = [1,2] + [10,20] = [11,22]
    assert_eq!(data, vec![11.0, 22.0]);
}

#[test]
fn test_linear_batched() {
    let (client, device) = cpu_setup();
    let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device).unwrap();
    let linear = Linear::new(weight, None, false);

    // input: [4, 5, 3] — batched
    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 60], &[4, 5, 3], &device).unwrap(),
        false,
    );
    let out = linear.forward(&client, &input).unwrap();
    assert_eq!(out.shape(), &[4, 5, 2]);
}

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
        Tensor::<CpuRuntime>::from_slice(&vec![0.1f32; in_features], &[1, in_features], &device)
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
        Tensor::<CpuRuntime>::from_slice(&vec![0.1f32; in_features], &[1, in_features], &device)
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
