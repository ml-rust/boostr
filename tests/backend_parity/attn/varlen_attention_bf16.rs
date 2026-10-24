//! Backend parity for VarLenAttentionOps at BF16: CUDA vs CPU, forward and
//! backward.
//!
//! The CPU reference runs in F32 on the exact values the kernel reads: Q, K, V
//! and dO rounded to BF16 on the device, then widened back. The backward
//! reference also takes the CUDA forward O (BF16) and LSE (F32). The kernel
//! computes in F32 and rounds each stored tensor once: O and dQ at the store,
//! dK and dV when the launcher casts the F32 accumulator.
//!
//! Bound, with `u = 2^-8` (BF16 unit roundoff, 8 significant bits):
//! `|cuda - ref| <= 2u * |ref| + (u / 2) * ref_rms`. One round-to-nearest
//! store costs at most `u * |ref|`. The factor 2 and the `ref_rms` term cover
//! F32 summation order and `__expf`, which can push a value across one
//! rounding boundary. BF16 has F32's exponent range, so no subnormal term
//! applies. LSE stays F32 on both sides and takes the F32 relaxed bound.

use super::helpers::*;
use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::runtime::cuda::{CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

/// BF16 unit roundoff.
const U_BF16: f32 = 1.0 / 256.0;

/// Deterministic values, distinct per index and per seed.
fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.021 + seed;
            x.sin() * 0.8 + (x * 1.7).cos() * 0.3
        })
        .collect()
}

/// Uploads F32 data and rounds it to BF16 on the device.
fn bf16_tensor(data: &[f32], shape: &[usize], dev: &CudaDevice) -> Tensor<CudaRuntime> {
    Tensor::<CudaRuntime>::from_slice(data, shape, dev)
        .expect("upload fixture")
        .to_dtype(DType::BF16)
        .expect("cast fixture to BF16")
}

/// Reads a CUDA tensor back as F32 values.
fn read_f32(t: &Tensor<CudaRuntime>) -> Vec<f32> {
    if t.dtype() == DType::F32 {
        return t.to_vec::<f32>();
    }
    t.to_dtype(DType::F32)
        .expect("cast result to F32")
        .to_vec::<f32>()
}

/// Moves a CUDA tensor's F32 values into a CPU tensor of the same shape.
fn to_cpu(t: &Tensor<CudaRuntime>, dev: &CpuDevice) -> Tensor<CpuRuntime> {
    Tensor::<CpuRuntime>::from_slice(&read_f32(t), t.shape(), dev).expect("build CPU tensor")
}

fn assert_bf16_once(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    let n = expected.len() as f64;
    let ref_rms = (expected.iter().map(|&e| (e as f64).powi(2)).sum::<f64>() / n).sqrt() as f32;
    assert!(
        ref_rms > 1e-4,
        "{label}: reference RMS {ref_rms:.3e} is degenerate; fix the fixture"
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let bound = 2.0 * U_BF16 * e.abs() + 0.5 * U_BF16 * ref_rms;
        let diff = (a - e).abs();
        assert!(
            a.is_finite() && diff <= bound,
            "{label} at index {i}: cuda {a} vs reference {e} (diff {diff:.3e}, bound \
             {bound:.3e}); the result no longer reaches BF16 through one rounding"
        );
    }
}

/// One BF16 case: ragged batch, `seq_len_q != seq_len_k` in sequence 0, GQA.
fn run_case(head_dim: usize, causal: bool) {
    let (cpu, cpu_dev) = setup_cpu();
    let (nh, nkv) = (4usize, 2usize);
    let cu_q: Vec<i32> = vec![0, 5, 12, 15];
    let cu_k: Vec<i32> = vec![0, 9, 16, 19];
    let (batch, msq, msk) = (3usize, 7usize, 9usize);
    let (tq, tk) = (15usize, 19usize);
    let q_shape = [tq, nh, head_dim];
    let kv_shape = [tk, nkv, head_dim];
    let label = format!("varlen bf16 hd{head_dim} causal={causal}");

    with_cuda_backend(|cuda, dev| {
        let q = bf16_tensor(&values(tq * nh * head_dim, 0.2), &q_shape, &dev);
        let k = bf16_tensor(&values(tk * nkv * head_dim, 1.4), &kv_shape, &dev);
        let v = bf16_tensor(&values(tk * nkv * head_dim, 2.6), &kv_shape, &dev);
        let dout = bf16_tensor(&values(tq * nh * head_dim, 3.8), &q_shape, &dev);
        let csq = Tensor::<CudaRuntime>::from_slice(&cu_q, &[batch + 1], &dev).expect("cu_q");
        let csk = Tensor::<CudaRuntime>::from_slice(&cu_k, &[batch + 1], &dev).expect("cu_k");

        let (out, lse) = cuda
            .varlen_attention_fwd(
                &q, &k, &v, &csq, &csk, batch, nh, nkv, msq, msk, head_dim, causal, 0,
            )
            .unwrap_or_else(|e| panic!("{label}: CUDA forward: {e}"));
        let (dq, dk, dv) = cuda
            .varlen_attention_bwd(
                &dout, &q, &k, &v, &out, &lse, &csq, &csk, &cu_q, &cu_k, batch, nh, nkv, msq, msk,
                head_dim, causal, 0,
            )
            .unwrap_or_else(|e| panic!("{label}: CUDA backward: {e}"));
        for (name, t) in [("O", &out), ("dQ", &dq), ("dK", &dk), ("dV", &dv)] {
            assert_eq!(t.dtype(), DType::BF16, "{label}: {name} dtype");
        }

        let (q_c, k_c, v_c) = (
            to_cpu(&q, &cpu_dev),
            to_cpu(&k, &cpu_dev),
            to_cpu(&v, &cpu_dev),
        );
        let cu_q_c = det_i32_tensor(&cu_q, &[batch + 1], &cpu_dev);
        let cu_k_c = det_i32_tensor(&cu_k, &[batch + 1], &cpu_dev);
        let (out_r, lse_r) = cpu
            .varlen_attention_fwd(
                &q_c, &k_c, &v_c, &cu_q_c, &cu_k_c, batch, nh, nkv, msq, msk, head_dim, causal, 0,
            )
            .unwrap_or_else(|e| panic!("{label}: CPU forward: {e}"));
        assert_bf16_once(
            &read_f32(&out),
            &out_r.to_vec::<f32>(),
            &format!("{label} O"),
        );
        assert_parity_f32_relaxed(
            &read_f32(&lse),
            &lse_r.to_vec::<f32>(),
            &format!("{label} LSE"),
        );

        let (dq_r, dk_r, dv_r) = cpu
            .varlen_attention_bwd(
                &to_cpu(&dout, &cpu_dev),
                &q_c,
                &k_c,
                &v_c,
                &to_cpu(&out, &cpu_dev),
                &to_cpu(&lse, &cpu_dev),
                &cu_q_c,
                &cu_k_c,
                &cu_q,
                &cu_k,
                batch,
                nh,
                nkv,
                msq,
                msk,
                head_dim,
                causal,
                0,
            )
            .unwrap_or_else(|e| panic!("{label}: CPU backward: {e}"));
        let grads = [("dQ", &dq, dq_r), ("dK", &dk, dk_r), ("dV", &dv, dv_r)];
        for (name, actual, reference) in grads {
            let full = format!("{label} {name}");
            assert_bf16_once(&read_f32(actual), &reference.to_vec::<f32>(), &full);
        }
    });
}

#[test]
fn test_varlen_attention_bf16_parity_hd64() {
    run_case(64, true);
    run_case(64, false);
}

#[test]
fn test_varlen_attention_bf16_parity_hd128() {
    run_case(128, true);
    run_case(128, false);
}

#[test]
fn test_varlen_attention_bf16_parity_hd256() {
    run_case(256, true);
    run_case(256, false);
}
