//! `quant_matmul_batch` must return what `quant_matmul` returns per weight.
//!
//! The batch quantizes the activation ONCE and reuses the record across every
//! weight; the single call quantizes per weight. Both records hold the same
//! per-token, per-32-value codes — the shared record only pads its token
//! stride to the widest tile — and each weight then runs the same kernel
//! variant, so the two answers are the same bytes, not merely close. A batch
//! that mixed formats (Q4_K beside Q6_K, as a K-quant mix produces) or sat in
//! a regime the batch routes differently is where this would break.

use super::helpers::*;
use boostr::QuantMatmulOps;
use boostr::quant::{QuantFormat, QuantTensor, QuantizeOps};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Quantize a deterministic `[n, k]` weight with the CPU quantizer. `salt`
/// separates the weights of one batch.
fn weight_bytes(format: QuantFormat, n: usize, k: usize, salt: usize) -> Vec<u8> {
    let device = CpuDevice::new();
    let (client, _) = setup_cpu();
    let values: Vec<f32> = (0..n * k)
        .map(|i| (((i + salt * 97) % 811) as f32 * 0.023).sin())
        .collect();
    let input = Tensor::<CpuRuntime>::from_slice(&values, &[n, k], &device).expect("weight tensor");
    client
        .quantize(&input, format)
        .expect("quantize")
        .to_bytes()
        .expect("to_bytes")
}

/// The token counts that reach each batch route: the dp4a GEMV batch, the
/// feature-major MMQ batch at a tile the DiT uses, and a token count past
/// the widest variant so the shared record's padding is exercised.
const BATCH_MS: [usize; 4] = [1, 24, 100, 300];

#[cfg(feature = "cuda")]
#[test]
fn cuda_quant_matmul_batch_matches_single_calls_bit_for_bit() {
    let k = 512usize;
    // Mixed formats and widths, as a q4_k_m attention block carries.
    let specs = [
        (QuantFormat::Q4K, 128usize),
        (QuantFormat::Q4K, 64usize),
        (QuantFormat::Q6K, 128usize),
    ];

    with_cuda_backend(|client, device| {
        let weights: Vec<QuantTensor<_>> = specs
            .iter()
            .enumerate()
            .map(|(i, (format, n))| {
                QuantTensor::from_bytes(
                    &weight_bytes(*format, *n, k, i),
                    *format,
                    &[*n, k],
                    &device,
                )
                .expect("CUDA QuantTensor")
            })
            .collect();
        let refs: Vec<&QuantTensor<_>> = weights.iter().collect();

        for m in BATCH_MS {
            let act_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect();
            let act = Tensor::from_slice(&act_data, &[m, k], &device).expect("activation");

            let batched = client
                .quant_matmul_batch(&act, &refs)
                .expect("quant_matmul_batch");
            assert_eq!(batched.len(), refs.len());
            for (w, out) in refs.iter().zip(&batched) {
                let single = client.quant_matmul(&act, w).expect("quant_matmul");
                assert_eq!(out.shape(), single.shape());
                let (a, b) = (out.to_vec::<f32>(), single.to_vec::<f32>());
                assert!(
                    a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "m={m} {:?} n={}: batched and single results differ",
                    w.format().expect("format"),
                    w.shape()[0]
                );
            }
        }
    });
}
