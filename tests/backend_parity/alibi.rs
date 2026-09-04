//! Backend parity tests for AlibiOps.

use super::helpers::*;
use boostr::ops::traits::position::alibi::AlibiOps;

#[test]
fn test_alibi_add_bias_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, sq, sk) = (1, 4, 8, 8);

    // Initialize scores to zero so we can see the bias values
    let zeros = vec![0.0f32; b * h * sq * sk];
    let scores = numr::tensor::Tensor::from_slice(&zeros, &[b, h, sq, sk], &cpu_device).unwrap();

    cpu_client.alibi_add_bias(&scores, b, h, sq, sk).unwrap();
    let cpu_scores_vec = scores.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::position::alibi::AlibiOps as _;
        use numr::tensor::Tensor;
        let s = Tensor::from_slice(
            &vec![0.0f32; b * h * sq * sk],
            &[b, h, sq, sk],
            &cuda_device,
        )
        .unwrap();
        cuda_client.alibi_add_bias(&s, b, h, sq, sk).unwrap();
        assert_parity_f32(
            &s.to_vec::<f32>(),
            &cpu_scores_vec,
            "alibi_add_bias CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::position::alibi::AlibiOps as _;
        use numr::tensor::Tensor;
        let s = Tensor::from_slice(
            &vec![0.0f32; b * h * sq * sk],
            &[b, h, sq, sk],
            &wgpu_device,
        )
        .unwrap();
        wgpu_client.alibi_add_bias(&s, b, h, sq, sk).unwrap();
        assert_parity_f32(
            &s.to_vec::<f32>(),
            &cpu_scores_vec,
            "alibi_add_bias WGPU vs CPU",
        );
    });
}

// Covers the bf16 ALiBi kernels, which run on every supported device.
#[test]
fn test_alibi_add_bias_bf16_cuda() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, sq, sk) = (1, 4, 8, 8);

    // Reference computed on CPU in F32, per house rule for BF16 fixtures.
    let zeros = vec![0.0f32; b * h * sq * sk];
    let scores = numr::tensor::Tensor::from_slice(&zeros, &[b, h, sq, sk], &cpu_device).unwrap();
    cpu_client.alibi_add_bias(&scores, b, h, sq, sk).unwrap();
    let cpu_scores_vec = scores.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::position::alibi::AlibiOps as _;
        use numr::dtype::DType;
        use numr::tensor::Tensor;

        let s_f32 = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &vec![0.0f32; b * h * sq * sk],
            &[b, h, sq, sk],
            &cuda_device,
        )
        .unwrap();
        let s_bf16 = s_f32
            .to_dtype(DType::BF16)
            .expect("cast zeros fixture to BF16");

        cuda_client
            .alibi_add_bias(&s_bf16, b, h, sq, sk)
            .expect("BF16 alibi_add_bias must succeed");

        let result_f32 = s_bf16
            .to_dtype(DType::F32)
            .expect("cast BF16 result back to F32 for comparison")
            .to_vec::<f32>();
        // BF16 keeps ~8 mantissa bits, so tolerance is set by the dtype, not
        // by the op.
        assert_parity_f32_tol(
            &result_f32,
            &cpu_scores_vec,
            "alibi_add_bias BF16 CUDA vs CPU",
            4e-2,
            2e-2,
        );
    });
}

// Covers the two FP8 ALiBi kernels (`alibi_add_bias_fp8_e4m3`/`_e5m2`), which
// had no Rust dispatch site before this test and had never executed. There is
// no FP8 CPU reference (`alibi_add_bias` on CPU is F32 only), so the FP8 CUDA
// path is checked against the CUDA F32 path instead of against CPU.
#[cfg(feature = "cuda")]
fn run_alibi_add_bias_fp8_case(dtype: numr::dtype::DType, label: &str, rtol: f32, atol: f32) {
    use boostr::ops::traits::position::alibi::AlibiOps as _;
    use numr::tensor::Tensor;

    let (b, h, sq, sk) = (1, 4, 8, 8);

    with_cuda_backend(|cuda_client, cuda_device| {
        // F32 reference, computed on CUDA.
        let s_f32 = Tensor::from_slice(
            &vec![0.0f32; b * h * sq * sk],
            &[b, h, sq, sk],
            &cuda_device,
        )
        .unwrap();
        cuda_client
            .alibi_add_bias(&s_f32, b, h, sq, sk)
            .expect("F32 alibi_add_bias must succeed");
        let f32_result = s_f32.to_vec::<f32>();

        // Same fixture, quantized to FP8, biased in place, then dequantized
        // for comparison.
        let s_zero = Tensor::from_slice(
            &vec![0.0f32; b * h * sq * sk],
            &[b, h, sq, sk],
            &cuda_device,
        )
        .unwrap();
        let s_fp8 = s_zero
            .to_dtype(dtype)
            .unwrap_or_else(|e| panic!("cast zeros fixture to {dtype:?}: {e:?}"));
        cuda_client
            .alibi_add_bias(&s_fp8, b, h, sq, sk)
            .unwrap_or_else(|e| panic!("{label} alibi_add_bias must succeed: {e:?}"));
        let fp8_result = s_fp8
            .to_dtype(numr::dtype::DType::F32)
            .expect("cast FP8 result back to F32 for comparison")
            .to_vec::<f32>();

        // Gated on the CUDA runtime being present (with_cuda_backend), never
        // on matching an error string.
        assert_parity_f32_tol(
            &fp8_result,
            &f32_result,
            &format!("alibi_add_bias {label} CUDA vs F32 CUDA"),
            rtol,
            atol,
        );
    });
}

// The input fixture is zeros, so the kernel stores `quantize(bias)` and the
// only error is FP8's own rounding of that value — purely RELATIVE. So rtol
// carries the tolerance and atol stays near zero. An atol sized to the bias
// magnitudes (which reach ~1.75 here) would accept a wrong answer outright,
// since the comparator's tolerance is `atol + rtol * |reference|`.
//
// e4m3 keeps 3 explicit mantissa bits, so eps = 2^-4 = 0.0625.
#[test]
fn test_alibi_add_bias_fp8_e4m3_cuda() {
    #[cfg(feature = "cuda")]
    run_alibi_add_bias_fp8_case(numr::dtype::DType::FP8E4M3, "FP8E4M3", 7e-2, 1e-3);
}

// e5m2 keeps only 2 explicit mantissa bits, so eps = 2^-3 = 0.125.
#[test]
fn test_alibi_add_bias_fp8_e5m2_cuda() {
    #[cfg(feature = "cuda")]
    run_alibi_add_bias_fp8_case(numr::dtype::DType::FP8E5M2, "FP8E5M2", 1.4e-1, 1e-3);
}

#[test]
fn test_alibi_add_bias_causal_bf16_cuda() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, sq, sk, position) = (1, 4, 8, 16, 8);

    // Reference computed on CPU in F32, per house rule for BF16 fixtures.
    let zeros = vec![0.0f32; b * h * sq * sk];
    let scores = numr::tensor::Tensor::from_slice(&zeros, &[b, h, sq, sk], &cpu_device).unwrap();
    cpu_client
        .alibi_add_bias_causal(&scores, b, h, sq, sk, position)
        .unwrap();
    let cpu_scores_vec = scores.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::position::alibi::AlibiOps as _;
        use numr::dtype::DType;
        use numr::tensor::Tensor;

        let s_f32 = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &vec![0.0f32; b * h * sq * sk],
            &[b, h, sq, sk],
            &cuda_device,
        )
        .unwrap();
        let s_bf16 = s_f32
            .to_dtype(DType::BF16)
            .expect("cast zeros fixture to BF16");

        cuda_client
            .alibi_add_bias_causal(&s_bf16, b, h, sq, sk, position)
            .expect("BF16 alibi_add_bias_causal must succeed");

        let result_f32 = s_bf16
            .to_dtype(DType::F32)
            .expect("cast BF16 result back to F32 for comparison")
            .to_vec::<f32>();
        // BF16 keeps ~8 mantissa bits, so tolerance is set by the dtype, not
        // by the op.
        assert_parity_f32_tol(
            &result_f32,
            &cpu_scores_vec,
            "alibi_add_bias_causal BF16 CUDA vs CPU",
            4e-2,
            2e-2,
        );
    });
}

// ---------------------------------------------------------------------------
// alibi_attention_bwd: backward for the MATERIALIZED biased-attention path.
//
// `probs` must be a real softmax output, so the fixture below softmaxes raw
// scores on the host before feeding them in. Arbitrary values would exercise
// the kernels outside the contract they were written against.
//
// The BF16 case is the point of these tests: the BF16 entry points used to
// reinterpret their `__nv_bfloat16` buffers as `__half` and decode 1-8-7 bits
// with a 1-5-10 decoder.
// ---------------------------------------------------------------------------

const BWD_B: usize = 2;
const BWD_H: usize = 2;
const BWD_SQ: usize = 4;
const BWD_SK: usize = 6;
const BWD_HD: usize = 8;

/// Deterministic values in roughly [-0.5, 0.5].
fn bwd_vals(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + phase).sin() * 0.5)
        .collect()
}

/// Row-wise softmax over the last dimension, computed on the host in F32.
fn bwd_softmax_rows(scores: &[f32], cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; scores.len()];
    for (row_in, row_out) in scores.chunks(cols).zip(out.chunks_mut(cols)) {
        let max = row_in.iter().fold(f32::NEG_INFINITY, |m, x| m.max(*x));
        let mut sum = 0.0f32;
        for (o, x) in row_out.iter_mut().zip(row_in) {
            *o = (x - max).exp();
            sum += *o;
        }
        for o in row_out.iter_mut() {
            *o /= sum;
        }
    }
    out
}

/// (grad_output, probs, q, k, v) as flat F32 host data.
type BwdFixture = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// Deterministic backward inputs. `probs` is a real softmax output, so each
/// row over `BWD_SK` sums to 1 and the kernels run inside their contract.
fn bwd_fixture() -> BwdFixture {
    let qkv_q = BWD_B * BWD_H * BWD_SQ * BWD_HD;
    let qkv_k = BWD_B * BWD_H * BWD_SK * BWD_HD;
    let n_scores = BWD_B * BWD_H * BWD_SQ * BWD_SK;

    let grad_output = bwd_vals(qkv_q, 0.11);
    let q = bwd_vals(qkv_q, 0.73);
    let k = bwd_vals(qkv_k, 1.31);
    let v = bwd_vals(qkv_k, 2.17);

    // Raw scores -> host softmax, so every `probs` row sums to 1.
    let raw = bwd_vals(n_scores, 0.41);
    let probs = bwd_softmax_rows(&raw, BWD_SK);

    (grad_output, probs, q, k, v)
}

/// Tolerance scaled by the reference tensor's own magnitude, so a small-valued
/// gradient is not waved through by a fixed absolute tolerance.
#[cfg(feature = "cuda")]
fn assert_grad_close(actual: &[f32], expected: &[f32], op: &str, rtol: f32, atol_frac: f32) {
    let max_abs = expected.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(max_abs > 0.0, "{op}: reference gradient is all zeros");
    assert_parity_f32_tol(actual, expected, op, rtol, atol_frac * max_abs);
}

/// CPU reference gradients as flat F32 vectors: (dQ, dK, dV).
fn bwd_cpu_reference() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (cpu_client, cpu_device) = setup_cpu();
    let (go, probs, q, k, v) = bwd_fixture();
    let scale = (BWD_HD as f32).sqrt().recip();

    use numr::tensor::Tensor;
    let go_t = Tensor::from_slice(&go, &[BWD_B, BWD_H, BWD_SQ, BWD_HD], &cpu_device).unwrap();
    let p_t = Tensor::from_slice(&probs, &[BWD_B, BWD_H, BWD_SQ, BWD_SK], &cpu_device).unwrap();
    let q_t = Tensor::from_slice(&q, &[BWD_B, BWD_H, BWD_SQ, BWD_HD], &cpu_device).unwrap();
    let k_t = Tensor::from_slice(&k, &[BWD_B, BWD_H, BWD_SK, BWD_HD], &cpu_device).unwrap();
    let v_t = Tensor::from_slice(&v, &[BWD_B, BWD_H, BWD_SK, BWD_HD], &cpu_device).unwrap();

    let (dq, dk, dv) = cpu_client
        .alibi_attention_bwd(&go_t, &p_t, &q_t, &k_t, &v_t, BWD_B, BWD_H, BWD_HD, scale)
        .expect("CPU alibi_attention_bwd must succeed");

    (dq.to_vec::<f32>(), dk.to_vec::<f32>(), dv.to_vec::<f32>())
}

#[test]
fn test_alibi_attention_bwd_cpu_reference_is_sane() {
    let (dq, dk, dv) = bwd_cpu_reference();
    assert_eq!(dq.len(), BWD_B * BWD_H * BWD_SQ * BWD_HD);
    assert_eq!(dk.len(), BWD_B * BWD_H * BWD_SK * BWD_HD);
    assert_eq!(dv.len(), BWD_B * BWD_H * BWD_SK * BWD_HD);
    for (name, g) in [("dq", &dq), ("dk", &dk), ("dv", &dv)] {
        assert!(
            g.iter().all(|x| x.is_finite()),
            "{name}: CPU reference produced a non-finite gradient"
        );
        assert!(
            g.iter().any(|x| x.abs() > 1e-6),
            "{name}: CPU reference is entirely zero"
        );
    }
}

#[cfg(feature = "cuda")]
fn run_alibi_bwd_cuda_case(dtype: numr::dtype::DType, label: &str, rtol: f32, atol_frac: f32) {
    let (cpu_dq, cpu_dk, cpu_dv) = bwd_cpu_reference();

    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::position::alibi::AlibiOps as _;
        use numr::runtime::cuda::CudaRuntime;
        use numr::tensor::Tensor;

        let (go, probs, q, k, v) = bwd_fixture();
        let scale = (BWD_HD as f32).sqrt().recip();

        let mk = |data: &[f32], shape: [usize; 4]| {
            let t = Tensor::<CudaRuntime>::from_slice(data, &shape, &cuda_device).unwrap();
            if dtype == numr::dtype::DType::F32 {
                t
            } else {
                t.to_dtype(dtype)
                    .unwrap_or_else(|e| panic!("cast fixture to {dtype:?}: {e:?}"))
            }
        };

        let go_t = mk(&go, [BWD_B, BWD_H, BWD_SQ, BWD_HD]);
        let p_t = mk(&probs, [BWD_B, BWD_H, BWD_SQ, BWD_SK]);
        let q_t = mk(&q, [BWD_B, BWD_H, BWD_SQ, BWD_HD]);
        let k_t = mk(&k, [BWD_B, BWD_H, BWD_SK, BWD_HD]);
        let v_t = mk(&v, [BWD_B, BWD_H, BWD_SK, BWD_HD]);

        // Gated on the CUDA runtime being present (with_cuda_backend), never on
        // matching an error string: a kernel that fails to load MUST fail this
        // test, not silently skip it.
        let (dq, dk, dv) = cuda_client
            .alibi_attention_bwd(&go_t, &p_t, &q_t, &k_t, &v_t, BWD_B, BWD_H, BWD_HD, scale)
            .unwrap_or_else(|e| panic!("{label} alibi_attention_bwd must succeed: {e:?}"));

        let back = |t: Tensor<CudaRuntime>| -> Vec<f32> {
            if dtype == numr::dtype::DType::F32 {
                t.to_vec::<f32>()
            } else {
                t.to_dtype(numr::dtype::DType::F32)
                    .expect("cast result back to F32 for comparison")
                    .to_vec::<f32>()
            }
        };

        assert_grad_close(
            &back(dq),
            &cpu_dq,
            &format!("alibi_attention_bwd grad_q {label} CUDA vs CPU"),
            rtol,
            atol_frac,
        );
        assert_grad_close(
            &back(dk),
            &cpu_dk,
            &format!("alibi_attention_bwd grad_k {label} CUDA vs CPU"),
            rtol,
            atol_frac,
        );
        assert_grad_close(
            &back(dv),
            &cpu_dv,
            &format!("alibi_attention_bwd grad_v {label} CUDA vs CPU"),
            rtol,
            atol_frac,
        );
    });
}

#[test]
fn test_alibi_attention_bwd_f32_parity() {
    #[cfg(feature = "cuda")]
    run_alibi_bwd_cuda_case(numr::dtype::DType::F32, "F32", 1e-4, 1e-4);
}

// F16 keeps 10 explicit mantissa bits (eps 2^-11 ~ 4.9e-4); the reductions here
// run over at most SK=6 / SQ=4 terms, so a few multiples of eps is the budget.
#[test]
fn test_alibi_attention_bwd_f16_parity() {
    #[cfg(feature = "cuda")]
    run_alibi_bwd_cuda_case(numr::dtype::DType::F16, "F16", 8e-3, 3e-3);
}

// BF16 keeps only 7 explicit mantissa bits (eps 2^-8 ~ 3.9e-3), so its
// tolerance is set by the dtype, not by the op — roughly an order of magnitude
// looser than F16. Still far tighter than the garbage a __half decode of BF16
// bits produces, which is what this case exists to catch.
#[test]
fn test_alibi_attention_bwd_bf16_parity() {
    #[cfg(feature = "cuda")]
    run_alibi_bwd_cuda_case(numr::dtype::DType::BF16, "BF16", 6e-2, 2e-2);
}
