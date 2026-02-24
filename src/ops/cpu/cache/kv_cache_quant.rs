//! CPU fallback for KV cache quantization
//!
//! Provides correct (but not optimized) implementations for testing.

use crate::error::Result;
use crate::ops::traits::cache::kv_cache_quant::{Int4GroupSize, KvCacheQuantOps};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

#[allow(clippy::needless_range_loop)]
impl KvCacheQuantOps<CpuRuntime> for CpuClient {
    fn quantize_kv_fp8_per_token(
        &self,
        input: &Tensor<CpuRuntime>,
        num_tokens: usize,
        head_dim: usize,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let data = input.to_vec::<f32>();
        let device = input.device();
        let mut quantized = vec![0u8; num_tokens * head_dim];
        let mut scales = vec![0.0f32; num_tokens];

        for (t, scale_out) in scales.iter_mut().enumerate().take(num_tokens) {
            let offset = t * head_dim;
            let mut max_abs = 0.0f32;
            for d in 0..head_dim {
                max_abs = max_abs.max(data[offset + d].abs());
            }
            // Scale so max maps to 127 (symmetric INT8-like quant for CPU fallback)
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            *scale_out = scale;

            for d in 0..head_dim {
                let val = (data[offset + d] / scale).round().clamp(-127.0, 127.0) as i8;
                quantized[offset + d] = val as u8;
            }
        }

        let q_tensor =
            Tensor::<CpuRuntime>::from_slice(&quantized, &[num_tokens, head_dim], device);
        let s_tensor = Tensor::<CpuRuntime>::from_slice(&scales, &[num_tokens], device);
        Ok((q_tensor, s_tensor))
    }

    fn dequantize_kv_fp8_per_token(
        &self,
        quantized: &Tensor<CpuRuntime>,
        scales: &Tensor<CpuRuntime>,
        num_tokens: usize,
        head_dim: usize,
        _output_dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        let q_data = quantized.to_vec::<u8>();
        let s_data = scales.to_vec::<f32>();
        let device = quantized.device();

        let mut output = vec![0.0f32; num_tokens * head_dim];
        for t in 0..num_tokens {
            let scale = s_data[t];
            let offset = t * head_dim;
            for d in 0..head_dim {
                output[offset + d] = (q_data[offset + d] as i8 as f32) * scale;
            }
        }

        Ok(Tensor::<CpuRuntime>::from_slice(
            &output,
            &[num_tokens, head_dim],
            device,
        ))
    }

    fn quantize_kv_int4(
        &self,
        input: &Tensor<CpuRuntime>,
        num_tokens: usize,
        head_dim: usize,
        group_size: Int4GroupSize,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let data = input.to_vec::<f32>();
        let device = input.device();
        let gs = group_size as usize;
        let num_groups = (num_tokens * head_dim).div_ceil(gs);

        let mut packed = vec![0u8; num_tokens * head_dim / 2];
        let mut scales_vec = vec![0.0f32; num_groups];
        let mut zeros_vec = vec![0.0f32; num_groups];

        for g in 0..num_groups {
            let start = g * gs;
            let end = (start + gs).min(num_tokens * head_dim);

            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            for i in start..end {
                min_val = min_val.min(data[i]);
                max_val = max_val.max(data[i]);
            }

            let range = max_val - min_val;
            let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
            scales_vec[g] = scale;
            zeros_vec[g] = min_val;

            for i in start..end {
                let q = ((data[i] - min_val) / scale).round().clamp(0.0, 15.0) as u8;
                let byte_idx = i / 2;
                if i % 2 == 0 {
                    packed[byte_idx] |= q & 0xF;
                } else {
                    packed[byte_idx] |= (q & 0xF) << 4;
                }
            }
        }

        let p = Tensor::<CpuRuntime>::from_slice(&packed, &[num_tokens, head_dim / 2], device);
        let s = Tensor::<CpuRuntime>::from_slice(&scales_vec, &[num_groups], device);
        let z = Tensor::<CpuRuntime>::from_slice(&zeros_vec, &[num_groups], device);
        Ok((p, s, z))
    }

    fn dequantize_kv_int4(
        &self,
        packed: &Tensor<CpuRuntime>,
        scales: &Tensor<CpuRuntime>,
        zeros: &Tensor<CpuRuntime>,
        num_tokens: usize,
        head_dim: usize,
        group_size: Int4GroupSize,
    ) -> Result<Tensor<CpuRuntime>> {
        let p_data = packed.to_vec::<u8>();
        let s_data = scales.to_vec::<f32>();
        let z_data = zeros.to_vec::<f32>();
        let device = packed.device();
        let gs = group_size as usize;

        let total = num_tokens * head_dim;
        let mut output = vec![0.0f32; total];

        for i in 0..total {
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                p_data[byte_idx] & 0xF
            } else {
                (p_data[byte_idx] >> 4) & 0xF
            };
            let g = i / gs;
            output[i] = nibble as f32 * s_data[g] + z_data[g];
        }

        Ok(Tensor::<CpuRuntime>::from_slice(
            &output,
            &[num_tokens, head_dim],
            device,
        ))
    }

    fn quantize_kv_int8(
        &self,
        input: &Tensor<CpuRuntime>,
        num_tokens: usize,
        head_dim: usize,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let data = input.to_vec::<f32>();
        let device = input.device();
        let mut quantized = vec![0i8; num_tokens * head_dim];
        let mut scales = vec![0.0f32; num_tokens];

        for t in 0..num_tokens {
            let offset = t * head_dim;
            let mut max_abs = 0.0f32;
            for d in 0..head_dim {
                max_abs = max_abs.max(data[offset + d].abs());
            }
            let scale = max_abs / 127.0;
            scales[t] = scale;

            for d in 0..head_dim {
                quantized[offset + d] = if scale > 0.0 {
                    (data[offset + d] / scale).round().clamp(-127.0, 127.0) as i8
                } else {
                    0i8
                };
            }
        }

        let q = Tensor::<CpuRuntime>::from_slice(&quantized, &[num_tokens, head_dim], device);
        let s = Tensor::<CpuRuntime>::from_slice(&scales, &[num_tokens], device);
        Ok((q, s))
    }

    fn dequantize_kv_int8(
        &self,
        quantized: &Tensor<CpuRuntime>,
        scales: &Tensor<CpuRuntime>,
        num_tokens: usize,
        head_dim: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        let q_data = quantized.to_vec::<i8>();
        let s_data = scales.to_vec::<f32>();
        let device = quantized.device();

        let mut output = vec![0.0f32; num_tokens * head_dim];
        for t in 0..num_tokens {
            let offset = t * head_dim;
            for d in 0..head_dim {
                output[offset + d] = q_data[offset + d] as f32 * s_data[t];
            }
        }

        Ok(Tensor::<CpuRuntime>::from_slice(
            &output,
            &[num_tokens, head_dim],
            device,
        ))
    }
}

#[cfg(test)]
mod tests {
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
        let input = Tensor::<CpuRuntime>::from_slice(&data, &[num_tokens, head_dim], &dev);

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
    fn test_int4_roundtrip() {
        let (client, dev) = cpu_setup();
        let num_tokens = 2;
        let head_dim = 8;
        let data: Vec<f32> = (0..num_tokens * head_dim).map(|i| i as f32 * 0.1).collect();
        let input = Tensor::<CpuRuntime>::from_slice(&data, &[num_tokens, head_dim], &dev);

        let (p, s, z) = client
            .quantize_kv_int4(&input, num_tokens, head_dim, Int4GroupSize::Group32)
            .unwrap();
        assert_eq!(p.shape(), &[num_tokens, head_dim / 2]);

        let output = client
            .dequantize_kv_int4(&p, &s, &z, num_tokens, head_dim, Int4GroupSize::Group32)
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
        let input = Tensor::<CpuRuntime>::from_slice(&data, &[num_tokens, head_dim], &dev);

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
}
