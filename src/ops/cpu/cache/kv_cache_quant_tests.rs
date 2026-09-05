use super::*;
use crate::test_utils::cpu_setup;

#[test]
fn test_fp8_roundtrip() {
    let (client, dev) = cpu_setup();
    let num_tokens = 4;
    let head_dim = 8;
    let data: Vec<f32> = (0..num_tokens * head_dim)
        .map(|i| (i as f32 * 0.3).sin())
        .collect();
    let input = Tensor::<CpuRuntime>::from_slice(&data, &[num_tokens, head_dim], &dev).unwrap();

    let (q, s) = client
        .quantize_kv_fp8_per_token(&input, num_tokens, head_dim)
        .unwrap();
    let output = client
        .dequantize_kv_fp8_per_token(&q, &s, num_tokens, head_dim, DType::F32)
        .unwrap();

    let out_data = output.to_vec::<f32>();
    let max_err: f32 = data
        .iter()
        .zip(out_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 0.1, "FP8 roundtrip error too high: {max_err}");
}

#[test]
fn test_fp8_per_tensor_roundtrip() {
    let (client, dev) = cpu_setup();
    let shape = [4, 8];
    let data: Vec<f32> = (0..32).map(|i| (i as f32 * 0.3).sin() * 3.0).collect();
    let input = Tensor::<CpuRuntime>::from_slice(&data, &shape, &dev).unwrap();

    let (q, s) = client.quantize_kv_fp8_per_tensor(&input).unwrap();
    assert_eq!(q.dtype(), DType::FP8E4M3);
    assert_eq!(s.shape(), &[1]);

    let output = client
        .dequantize_kv_fp8_per_tensor(&q, &s, DType::F32)
        .unwrap();
    let out_data = output.to_vec::<f32>();

    // FP8 e4m3 keeps 3 mantissa bits, i.e. a relative step of 1/8 near any
    // representable value; 15% covers that plus scale-rounding slop.
    let max_rel_err: f32 = data
        .iter()
        .zip(out_data.iter())
        .map(|(a, b)| (a - b).abs() / a.abs().max(1e-6))
        .fold(0.0f32, f32::max);
    assert!(
        max_rel_err < 0.15,
        "FP8 per-tensor roundtrip relative error too high: {max_rel_err}"
    );
}

#[test]
fn test_int4_roundtrip() {
    let (client, dev) = cpu_setup();
    let num_tokens = 2;
    let head_dim = 8;
    let data: Vec<f32> = (0..num_tokens * head_dim).map(|i| i as f32 * 0.1).collect();
    let input = Tensor::<CpuRuntime>::from_slice(&data, &[num_tokens, head_dim], &dev).unwrap();

    let (p, s, z) = client
        .quantize_kv_int4(&input, num_tokens, head_dim, Int4GroupSize::Group32)
        .unwrap();
    assert_eq!(p.shape(), &[num_tokens, head_dim / 2]);

    let output = client
        .dequantize_kv_int4(
            &p,
            &s,
            &z,
            num_tokens,
            head_dim,
            Int4GroupSize::Group32,
            DType::F32,
        )
        .unwrap();
    let out_data = output.to_vec::<f32>();
    let max_err: f32 = data
        .iter()
        .zip(out_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 0.2, "INT4 roundtrip error too high: {max_err}");
}

#[test]
fn test_int8_roundtrip() {
    let (client, dev) = cpu_setup();
    let num_tokens = 4;
    let head_dim = 8;
    let data: Vec<f32> = (0..num_tokens * head_dim)
        .map(|i| (i as f32 * 0.5).sin())
        .collect();
    let input = Tensor::<CpuRuntime>::from_slice(&data, &[num_tokens, head_dim], &dev).unwrap();

    let (q, s) = client
        .quantize_kv_int8(&input, num_tokens, head_dim)
        .unwrap();
    let output = client
        .dequantize_kv_int8(&q, &s, num_tokens, head_dim)
        .unwrap();

    let out_data = output.to_vec::<f32>();
    let max_err: f32 = data
        .iter()
        .zip(out_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 0.02, "INT8 roundtrip error too high: {max_err}");
}
