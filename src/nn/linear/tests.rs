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
