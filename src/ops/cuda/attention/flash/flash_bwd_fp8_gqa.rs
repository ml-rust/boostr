//! FP8 gradient reduction for GQA groups, and the FP8 quantization-convention
//! tests that back it.
//!
//! Split out of `flash_bwd_fp8.rs`, which holds the kernel launcher. The tests
//! here isolate the `f32 <-> FP8` converters compiled into `flash_v2_bwd_fp8.cu`
//! from the backward kernel itself, and check that the group sum requantizes
//! exactly once.

use crate::error::Result;
use crate::ops::impl_generic::attention::sum_gqa_grads;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Reduce a GQA group of per-head FP8 gradients back to one KV head.
///
/// The kernel stores `raw = quantize(value * scale)`, so an element already
/// carries its `dk_scale`/`dv_scale` factor. Summing raw FP8 would round once per
/// group member and can leave E4M3's ~±448 range, so the group is dequantized to
/// F32, summed there, and requantized ONCE. The cast back to FP8 reapplies the
/// kernel's convention exactly: the F32 sum equals `scale * sum(real values)`,
/// which is the value the kernel would have written for the merged head.
pub(super) fn sum_gqa_grads_fp8(
    client: &CudaClient,
    grad: &Tensor<CudaRuntime>,
    num_kv_heads: usize,
    repeats: usize,
    dtype: DType,
) -> Result<Tensor<CudaRuntime>> {
    let grad_f32 = client.cast(grad, DType::F32)?;
    let summed = sum_gqa_grads(client, &grad_f32, num_kv_heads, repeats)?;
    Ok(client.cast(&summed, dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::cuda::kernels::{self, FLASH_V2_BWD_FP8_MODULE};
    use cudarc::driver::PushKernelArg;
    use cudarc::driver::safe::LaunchConfig;
    use numr::runtime::Device;
    use numr::runtime::Runtime;
    use numr::runtime::cuda::{CudaDevice, is_cuda_available};

    /// Same gate the CUDA integration tests use: the `cuda` feature can be on
    /// while no device is present, and the suite must skip, not fail.
    fn cuda_client() -> Option<(CudaClient, CudaDevice)> {
        if !is_cuda_available() {
            eprintln!("CUDA feature enabled but runtime unavailable, skipping");
            return None;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        Some((client, device))
    }

    /// Run `input` through the FP8 encoder compiled into `flash_v2_bwd_fp8.cu`.
    ///
    /// `probe` names the per-format probe kernel, `dtype` its storage type.
    /// Returns `(boostr_encode_numr_decode, boostr_encode_boostr_decode)`. The
    /// first column isolates the ENCODER (numr's decoder is the reference), the
    /// second adds boostr's decoder, so the two columns separate the converters.
    fn boostr_fp8_roundtrip(
        client: &CudaClient,
        device: &CudaDevice,
        input: &[f32],
        dtype: DType,
        probe: &str,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = input.len();
        let src = Tensor::<CudaRuntime>::from_slice(input, &[n], device).expect("input tensor");
        let raw = Tensor::<CudaRuntime>::empty(&[n], dtype, device).expect("raw tensor");
        let dec = Tensor::<CudaRuntime>::empty(&[n], DType::F32, device).expect("dec tensor");

        let module =
            kernels::get_or_load_module(client.context(), device.id(), FLASH_V2_BWD_FP8_MODULE)
                .expect("flash_v2_bwd_fp8 module");
        let func = kernels::get_kernel_function(&module, probe).expect("probe kernel");

        let block = 64u32;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let src_ptr = src.ptr();
        let raw_ptr = raw.ptr();
        let dec_ptr = dec.ptr();
        let n_i32 = n as i32;
        unsafe {
            let mut builder = client.stream().launch_builder(&func);
            builder.arg(&src_ptr);
            builder.arg(&raw_ptr);
            builder.arg(&dec_ptr);
            builder.arg(&n_i32);
            builder.launch(cfg).expect("probe launch");
        }
        client.stream().synchronize().expect("probe sync");

        let via_numr_decode = client
            .cast(&raw, DType::F32)
            .expect("raw cast")
            .to_vec::<f32>();
        (via_numr_decode, dec.to_vec::<f32>())
    }

    /// numr's own `f32 -> FP8 -> f32` round trip, the reference encoder.
    fn numr_fp8_roundtrip(
        client: &CudaClient,
        device: &CudaDevice,
        input: &[f32],
        dtype: DType,
    ) -> Vec<f32> {
        let n = input.len();
        let src = Tensor::<CudaRuntime>::from_slice(input, &[n], device).expect("input tensor");
        let q = client.cast(&src, dtype).expect("cast to fp8");
        client
            .cast(&q, DType::F32)
            .expect("cast to f32")
            .to_vec::<f32>()
    }

    /// The exact value the FP8 MQA backward parity test disagrees on.
    ///
    /// `0.59765625` sits in the `[0.5, 0.625)` binade, whose E4M3 ulp is
    /// `0.0625`; its neighbours are `0.5625` and `0.625`, and round-to-nearest
    /// gives `0.625`. If both encoders return `0.625` here, the cast is NOT the
    /// fault behind the parity failure and the wrong value is reaching it.
    #[test]
    fn fp8_e4m3_encodes_the_failing_parity_value_to_nearest() {
        let Some((client, device)) = cuda_client() else {
            return;
        };
        let input = [0.59765625f32, -0.59765625];
        let expected = [0.625f32, -0.625];

        let numr = numr_fp8_roundtrip(&client, &device, &input, DType::FP8E4M3);
        let (boostr_numr_decode, boostr) = boostr_fp8_roundtrip(
            &client,
            &device,
            &input,
            DType::FP8E4M3,
            "fp8_e4m3_roundtrip_probe",
        );

        assert_eq!(
            numr, expected,
            "numr cast f32->E4M3 must round 0.59765625 to 0.625"
        );
        assert_eq!(
            boostr_numr_decode, expected,
            "boostr f32_to_fp8_e4m3_raw must round 0.59765625 to 0.625"
        );
        assert_eq!(
            boostr, expected,
            "boostr fp8_e4m3_to_f32 must decode the encoded 0.59765625 as 0.625"
        );
    }

    /// Full E4M3 round-trip sweep: binade boundaries, mantissa carries, the
    /// smallest normal `2^-6`, subnormals down to `2^-9`, and the max finite
    /// `448`. Every mismatch is collected before failing, so one run reports the
    /// whole disagreement table instead of only its first row.
    #[test]
    fn fp8_e4m3_roundtrip_matches_round_to_nearest_even() {
        let Some((client, device)) = cuda_client() else {
            return;
        };
        // (input, IEEE round-to-nearest-even E4M3 value)
        let cases: [(f32, f32); 20] = [
            (0.0, 0.0),
            (0.5, 0.5),
            (0.5625, 0.5625),
            (0.59765625, 0.625),
            (0.625, 0.625),
            // Exact tie between 0.5 and 0.5625: round-half-to-even keeps 0.5.
            (0.53125, 0.5),
            // Mantissa rounds up out of the 3-bit field and must CARRY into the
            // exponent, not clamp to `0b111`.
            (0.9921875, 1.0),
            (1.9375, 2.0),
            (1.0, 1.0),
            (2.0, 2.0),
            // Smallest normal, and the binade just above it.
            (0.015625, 0.015625),
            (0.0234375, 0.0234375),
            (0.013671875, 0.013671875),
            // Subnormals: spacing is 2^-9 with no implicit leading 1.
            (0.0078125, 0.0078125),
            (0.00390625, 0.00390625),
            (0.001953125, 0.001953125),
            (0.001_464_843_8, 0.001_953_125),
            (0.01, 0.009765625),
            // Half the smallest subnormal ties to even, which is zero.
            (0.0009765625, 0.0),
            // Max finite E4M3.
            (448.0, 448.0),
        ];
        assert_fp8_roundtrip(
            &client,
            &device,
            &cases,
            DType::FP8E4M3,
            "fp8_e4m3_roundtrip_probe",
            "E4M3",
        );
    }

    /// Check every `(input, expected)` pair against numr's cast and both halves
    /// of boostr's converter pair, reporting the whole disagreement table.
    fn assert_fp8_roundtrip(
        client: &CudaClient,
        device: &CudaDevice,
        cases: &[(f32, f32)],
        dtype: DType,
        probe: &str,
        label: &str,
    ) {
        let input: Vec<f32> = cases.iter().map(|(x, _)| *x).collect();
        let expected: Vec<f32> = cases.iter().map(|(_, y)| *y).collect();

        let numr = numr_fp8_roundtrip(client, device, &input, dtype);
        let (boostr_numr_decode, boostr) =
            boostr_fp8_roundtrip(client, device, &input, dtype, probe);

        let mut mismatches = Vec::new();
        for ((((x, want), got_numr), got_enc), got_dec) in input
            .iter()
            .zip(&expected)
            .zip(&numr)
            .zip(&boostr_numr_decode)
            .zip(&boostr)
        {
            if got_numr != want || got_enc != want || got_dec != want {
                mismatches.push(format!(
                    "  in={x} expected={want} numr={got_numr} \
                     boostr_encode={got_enc} boostr_encode_decode={got_dec}"
                ));
            }
        }
        assert!(
            mismatches.is_empty(),
            "{label} round trip disagrees with round-to-nearest-even:\n{}",
            mismatches.join("\n")
        );
    }

    /// Full E5M2 round-trip sweep. E5M2 shares the converter family with E4M3,
    /// so it is checked against the same three defect classes: a half-to-even
    /// tie, a mantissa carry into the exponent, the subnormal range, and the max
    /// finite `57344`.
    #[test]
    fn fp8_e5m2_roundtrip_matches_round_to_nearest_even() {
        let Some((client, device)) = cuda_client() else {
            return;
        };
        // (input, IEEE round-to-nearest-even E5M2 value). E5M2 keeps 2 mantissa
        // bits, so the ulp in [0.5, 1.0) is 0.125.
        let cases: [(f32, f32); 18] = [
            (0.0, 0.0),
            (0.5, 0.5),
            (0.625, 0.625),
            (1.0, 1.0),
            (2.0, 2.0),
            // Exact tie between 0.5 and 0.625: round-half-to-even keeps 0.5.
            (0.5625, 0.5),
            // Exact tie between 0.625 and 0.75: round-half-to-even picks 0.75.
            (0.6875, 0.75),
            // Mantissa rounds up out of the 2-bit field and must CARRY into the
            // exponent, not clamp to `0b11`.
            (0.96875, 1.0),
            (1.9375, 2.0),
            // Smallest normal 2^-14, and the binade just above it.
            (6.1035156e-5, 6.1035156e-5),
            (7.6293945e-5, 7.6293945e-5),
            // Subnormals: spacing is 2^-16 with no implicit leading 1.
            (1.5258789e-5, 1.5258789e-5),
            (3.0517578e-5, 3.0517578e-5),
            (4.5776367e-5, 4.5776367e-5),
            // Between two subnormals, rounds up to 3 * 2^-16.
            (4.196167e-5, 4.5776367e-5),
            // Half the smallest subnormal ties to even, which is zero.
            (7.6293945e-6, 0.0),
            // Max finite E5M2, and a value just under it that must not overflow.
            (57344.0, 57344.0),
            (49152.0, 49152.0),
        ];

        assert_fp8_roundtrip(
            &client,
            &device,
            &cases,
            DType::FP8E5M2,
            "fp8_e5m2_roundtrip_probe",
            "E5M2",
        );
    }

    /// `sum_gqa_grads_fp8` must sum a GQA group in F32 and requantize ONCE.
    ///
    /// The four per-head values are each exact in E4M3 and sum to exactly
    /// `0.59765625`, the value the FP8 MQA parity test disagrees on. A correct
    /// single requantization returns `0.625`. This isolates the reduction and
    /// the final cast from the backward kernel that produces the gradients: a
    /// pass here puts the fault upstream, in the per-head dK the kernel writes.
    #[test]
    fn sum_gqa_grads_fp8_requantizes_the_group_sum_once() {
        let Some((client, device)) = cuda_client() else {
            return;
        };
        // Layout [b=1, heads=4, s=1, d=2]. Column 0 sums to 0.59765625;
        // column 1 sums to 1.0, which is exact in E4M3 and must survive.
        let per_head: [f32; 8] = [
            0.5, 0.25, // head 0
            0.0625, 0.25, // head 1
            0.03125, 0.25, // head 2
            0.00390625, 0.25, // head 3
        ];
        let src = Tensor::<CudaRuntime>::from_slice(&per_head, &[1, 4, 1, 2], &device)
            .expect("src tensor");
        let grad_fp8 = client.cast(&src, DType::FP8E4M3).expect("cast to fp8");

        // Every input is exact in E4M3, so the quantized inputs must round trip.
        let back = client
            .cast(&grad_fp8, DType::F32)
            .expect("cast back")
            .to_vec::<f32>();
        assert_eq!(
            back,
            per_head.to_vec(),
            "per-head inputs must be exact in E4M3 before the group sum is tested"
        );

        let summed =
            sum_gqa_grads_fp8(&client, &grad_fp8, 1, 4, DType::FP8E4M3).expect("group sum");
        assert_eq!(
            summed.shape(),
            [1, 1, 1, 2],
            "group sum must reduce to 1 KV head"
        );

        let got = client
            .cast(&summed, DType::F32)
            .expect("cast result")
            .to_vec::<f32>();
        assert_eq!(
            got,
            vec![0.625f32, 1.0],
            "F32 group sum 0.59765625 must requantize to 0.625, one E4M3 step; 1.0 must be exact"
        );
    }
}
