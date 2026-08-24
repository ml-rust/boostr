//! First numerical parity tests for the dedicated MQA/GQA CUDA kernels.
//!
//! `boostr::ops::cuda::attention::mqa_gqa` has no other caller in the tree, so
//! before this file nothing had ever executed `mqa_gqa.cu` / `mqa_gqa_bwd.cu`.
//! Every case here compares the dedicated kernel against an INDEPENDENT
//! implementation: the CPU `FlashAttentionOps` path (`impl_generic`
//! `standard_attention_fwd` / `standard_attention_bwd`), which is itself
//! parity-tested in `attention.rs`. The MQA kernel is never compared with
//! itself.
//!
//! Skip policy: an absent CUDA device (or an absent `cuda` / `f16` feature)
//! prints an unmistakable `SKIPPED:` line naming the case. When a device IS
//! present, the closure body is asserted to have run, so a silently skipped
//! backend fails the test instead of reporting green.

// The dtype tolerance table and the CUDA runners are unreachable in a build
// without the `cuda` feature; the `SKIPPED` stubs still name every case.
#![allow(dead_code)]

#[cfg(feature = "cuda")]
use super::helpers::{setup_cpu, with_cuda_backend};

/// One parity case. Plain data so the non-CUDA stubs can still name it.
#[derive(Clone, Copy)]
struct Case {
    label: &'static str,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    causal: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TestDType {
    F32,
    F16,
    BF16,
}

impl TestDType {
    fn name(self) -> &'static str {
        match self {
            TestDType::F32 => "f32",
            TestDType::F16 => "f16",
            TestDType::BF16 => "bf16",
        }
    }

    /// Tolerances are `(atol, rtol_of_reference_rms)`.
    ///
    /// `f32`: the kernel uses `__expf` (~2 ulp) and a different accumulation
    /// order than the reference matmul chain, so a few 1e-6 of drift per
    /// reduction step is expected; nothing larger is.
    /// `f16` / `bf16`: Q/K/V are ROUNDED TO THE STORAGE DTYPE before the kernel
    /// runs while the reference stays f32, so the floor is the input rounding
    /// itself — f16 eps is 9.8e-4, bf16 eps is 7.8e-3 — amplified by the score
    /// dot product and the softmax. The tolerance is set from that eps, not
    /// tuned to whatever the kernel happens to emit.
    fn fwd_tol(self) -> (f32, f32) {
        match self {
            TestDType::F32 => (1e-5, 1e-4),
            TestDType::F16 => (2e-3, 1e-2),
            TestDType::BF16 => (2e-2, 6e-2),
        }
    }

    /// Backward tolerances are looser than forward at every dtype: dK/dV are
    /// accumulated with `atomicAdd` across query-head blocks, so the summation
    /// order is non-deterministic run to run.
    fn bwd_tol(self) -> (f32, f32) {
        match self {
            TestDType::F32 => (1e-4, 1e-3),
            TestDType::F16 => (6e-3, 3e-2),
            TestDType::BF16 => (4e-2, 1e-1),
        }
    }

    /// F16/BF16 tensors need `numr`'s half support, pulled in by boostr's
    /// `f16` feature.
    fn enabled(self) -> bool {
        match self {
            TestDType::F32 => true,
            TestDType::F16 | TestDType::BF16 => cfg!(feature = "f16"),
        }
    }
}

// ============================================================================
// Forward parity
// ============================================================================

#[test]
fn mqa_gqa_fwd_mqa_hd32_noncausal_parity() {
    run_fwd_case(
        Case {
            label: "fwd mqa hd32 non-causal",
            batch: 2,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 16,
            seq_k: 16,
            head_dim: 32,
            causal: false,
        },
        TestDType::F32,
    );
}

/// Causal, square. A kernel that dropped the causal test would let query 0 mix
/// in keys 1..15, and every query row would pick up strictly future keys, so
/// the whole output diverges from the reference — not just the first row.
#[test]
fn mqa_gqa_fwd_mqa_hd32_causal_parity() {
    run_fwd_case(
        Case {
            label: "fwd mqa hd32 causal",
            batch: 2,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 16,
            seq_k: 16,
            head_dim: 32,
            causal: true,
        },
        TestDType::F32,
    );
}

#[test]
fn mqa_gqa_fwd_mqa_hd64_causal_parity() {
    run_fwd_case(
        Case {
            label: "fwd mqa hd64 causal",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 24,
            seq_k: 24,
            head_dim: 64,
            causal: true,
        },
        TestDType::F32,
    );
}

#[test]
fn mqa_gqa_fwd_mqa_hd128_causal_parity() {
    run_fwd_case(
        Case {
            label: "fwd mqa hd128 causal",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 24,
            seq_k: 24,
            head_dim: 128,
            causal: true,
        },
        TestDType::F32,
    );
}

#[test]
fn mqa_gqa_fwd_mqa_hd128_noncausal_parity() {
    run_fwd_case(
        Case {
            label: "fwd mqa hd128 non-causal",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 24,
            seq_k: 24,
            head_dim: 128,
            causal: false,
        },
        TestDType::F32,
    );
}

/// Intermediate GQA ratio (4 query heads per KV head). Discriminates the
/// `kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads)` mapping: a kernel
/// that used `q_head_idx % num_kv_heads` instead would pair head 1 with KV
/// head 1 rather than KV head 0, and every head but 0 would disagree.
#[test]
fn mqa_gqa_fwd_gqa_ratio4_hd64_noncausal_parity() {
    run_fwd_case(
        Case {
            label: "fwd gqa 8/2 hd64 non-causal",
            batch: 2,
            num_heads: 8,
            num_kv_heads: 2,
            seq_q: 16,
            seq_k: 16,
            head_dim: 64,
            causal: false,
        },
        TestDType::F32,
    );
}

#[test]
fn mqa_gqa_fwd_gqa_ratio4_hd64_causal_parity() {
    run_fwd_case(
        Case {
            label: "fwd gqa 8/2 hd64 causal",
            batch: 2,
            num_heads: 8,
            num_kv_heads: 2,
            seq_q: 16,
            seq_k: 16,
            head_dim: 64,
            causal: true,
        },
        TestDType::F32,
    );
}

/// `seq_len_q != seq_len_k`, non-causal. `seq_k = 37` is not a multiple of any
/// BLOCK_N the launcher can pick (128/64/32), so the last K tile is a partial
/// tail: a kernel that read a full BLOCK_N tile would pull garbage past
/// `seq_len_k` into the softmax.
#[test]
fn mqa_gqa_fwd_ragged_tail_hd32_noncausal_parity() {
    run_fwd_case(
        Case {
            label: "fwd mqa hd32 sq=5 sk=37 non-causal (tail block)",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 5,
            seq_k: 37,
            head_dim: 32,
            causal: false,
        },
        TestDType::F32,
    );
}

/// `seq_len_q != seq_len_k`, CAUSAL — the absolute-vs-relative query position
/// class. The repo convention (`flash_v2.cu` and
/// `impl_generic::attention::flash_standard::causal_window_mask`) is that query
/// row `i` sits at ABSOLUTE position `key_offset + i` with
/// `key_offset = seq_len_k - seq_len_q`, so with 5 queries against 37 keys the
/// queries are positions 32..36 and query 0 sees 33 keys. A kernel that masks
/// on the RELATIVE row index instead lets query 0 see only key 0, which is a
/// wildly different distribution — this assertion catches exactly that.
#[test]
fn mqa_gqa_fwd_ragged_tail_hd32_causal_parity() {
    run_fwd_case(
        Case {
            label: "fwd mqa hd32 sq=5 sk=37 causal (key_offset convention)",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 5,
            seq_k: 37,
            head_dim: 32,
            causal: true,
        },
        TestDType::F32,
    );
}

/// Multi-tile K with a partial tail (200 is not a multiple of 128/64/32) and a
/// multi-tile Q, so both loop nests run more than one iteration.
#[test]
fn mqa_gqa_fwd_multi_tile_hd64_noncausal_parity() {
    run_fwd_case(
        Case {
            label: "fwd mqa hd64 sq=48 sk=200 non-causal (multi tile + tail)",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 48,
            seq_k: 200,
            head_dim: 64,
            causal: false,
        },
        TestDType::F32,
    );
}

#[test]
fn mqa_gqa_fwd_multi_tile_hd64_causal_parity() {
    run_fwd_case(
        Case {
            label: "fwd mqa hd64 sq=48 sk=200 causal (multi tile + tail)",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 48,
            seq_k: 200,
            head_dim: 64,
            causal: true,
        },
        TestDType::F32,
    );
}

#[test]
fn mqa_gqa_fwd_mqa_hd64_causal_f16_parity() {
    run_fwd_case(
        Case {
            label: "fwd mqa hd64 causal",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 24,
            seq_k: 24,
            head_dim: 64,
            causal: true,
        },
        TestDType::F16,
    );
}

#[test]
fn mqa_gqa_fwd_mqa_hd64_causal_bf16_parity() {
    run_fwd_case(
        Case {
            label: "fwd mqa hd64 causal",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 24,
            seq_k: 24,
            head_dim: 64,
            causal: true,
        },
        TestDType::BF16,
    );
}

/// Closed-form causal discriminator, independent of the CPU path entirely.
///
/// With `causal = true` and `seq_len_q == seq_len_k`, query row 0 is at
/// absolute position 0 and may attend to key 0 only. Softmax over a single
/// unmasked key is exactly 1.0, so `out[b, h, 0, :]` must equal `v[b, kv, 0, :]`
/// bit-for-bit-ish, for EVERY query head. A kernel that ignored `causal`, or
/// that masked with `>` instead of `<`, or that let the tail of a K tile leak
/// in, produces a mixture of v[0..S] here and fails. The fixture makes v rows
/// distinct, so the mixture cannot coincidentally equal v[0].
#[test]
fn mqa_gqa_fwd_causal_first_query_equals_v0() {
    run_causal_first_row_case();
}

// ============================================================================
// Backward parity — dQ, dK and dV are each asserted separately
// ============================================================================

#[test]
fn mqa_gqa_bwd_mqa_hd32_noncausal_parity() {
    run_bwd_case(
        Case {
            label: "bwd mqa hd32 non-causal",
            batch: 2,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 16,
            seq_k: 16,
            head_dim: 32,
            causal: false,
        },
        TestDType::F32,
    );
}

#[test]
fn mqa_gqa_bwd_mqa_hd32_causal_parity() {
    run_bwd_case(
        Case {
            label: "bwd mqa hd32 causal",
            batch: 2,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 16,
            seq_k: 16,
            head_dim: 32,
            causal: true,
        },
        TestDType::F32,
    );
}

#[test]
fn mqa_gqa_bwd_mqa_hd64_noncausal_parity() {
    run_bwd_case(
        Case {
            label: "bwd mqa hd64 non-causal",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 24,
            seq_k: 24,
            head_dim: 64,
            causal: false,
        },
        TestDType::F32,
    );
}

/// head_dim 128 causal. The backward kernel skips query blocks with
/// `q_block_start = k_block`, which only lines up when BLOCK_M == BLOCK_N. At
/// head_dim 128 the launcher picks BLOCK_M=128, BLOCK_N=64 (or 64/32 on a
/// small-shared-memory GPU), so K block `n` starts at key `64n` while query
/// block `n` starts at query `128n`: the blocks in between are skipped and
/// their contribution to dK/dV (and dQ) is simply never accumulated. dK/dV
/// then come out too small in exactly the early rows.
#[test]
fn mqa_gqa_bwd_mqa_hd128_causal_parity() {
    run_bwd_case(
        Case {
            label: "bwd mqa hd128 causal (BLOCK_M != BLOCK_N q-block skip)",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 160,
            seq_k: 160,
            head_dim: 128,
            causal: true,
        },
        TestDType::F32,
    );
}

#[test]
fn mqa_gqa_bwd_mqa_hd128_noncausal_parity() {
    run_bwd_case(
        Case {
            label: "bwd mqa hd128 non-causal",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 24,
            seq_k: 24,
            head_dim: 128,
            causal: false,
        },
        TestDType::F32,
    );
}

/// GQA backward is where dK/dV accumulation is riskiest: four query heads
/// atomically add into ONE KV head. A kernel that wrote instead of accumulated
/// would leave dK/dV at a single head's contribution — a factor-of-4 error that
/// only a separate dK/dV assertion catches.
#[test]
fn mqa_gqa_bwd_gqa_ratio4_hd64_noncausal_parity() {
    run_bwd_case(
        Case {
            label: "bwd gqa 8/2 hd64 non-causal",
            batch: 2,
            num_heads: 8,
            num_kv_heads: 2,
            seq_q: 16,
            seq_k: 16,
            head_dim: 64,
            causal: false,
        },
        TestDType::F32,
    );
}

#[test]
fn mqa_gqa_bwd_gqa_ratio4_hd64_causal_parity() {
    run_bwd_case(
        Case {
            label: "bwd gqa 8/2 hd64 causal",
            batch: 2,
            num_heads: 8,
            num_kv_heads: 2,
            seq_q: 16,
            seq_k: 16,
            head_dim: 64,
            causal: true,
        },
        TestDType::F32,
    );
}

/// Ragged non-causal backward. dK/dV cover 37 key rows while dQ covers 5 —
/// the out-of-bounds dK/dV write class shows up here, because the tail K tile
/// is partial and the kernel indexes dK/dV by `k_start + k_row`.
#[test]
fn mqa_gqa_bwd_ragged_tail_hd32_noncausal_parity() {
    run_bwd_case(
        Case {
            label: "bwd mqa hd32 sq=5 sk=37 non-causal (tail block)",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 5,
            seq_k: 37,
            head_dim: 32,
            causal: false,
        },
        TestDType::F32,
    );
}

/// Ragged CAUSAL backward — same `key_offset` convention as the forward case.
/// With relative masking, keys 0..31 would receive no gradient at all while the
/// reference gives them the bulk of it, so dK/dV disagree by their own
/// magnitude.
#[test]
fn mqa_gqa_bwd_ragged_tail_hd32_causal_parity() {
    run_bwd_case(
        Case {
            label: "bwd mqa hd32 sq=5 sk=37 causal (key_offset convention)",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 5,
            seq_k: 37,
            head_dim: 32,
            causal: true,
        },
        TestDType::F32,
    );
}

#[test]
fn mqa_gqa_bwd_multi_tile_hd64_causal_parity() {
    run_bwd_case(
        Case {
            label: "bwd mqa hd64 sq=48 sk=200 causal (multi tile + tail)",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 48,
            seq_k: 200,
            head_dim: 64,
            causal: true,
        },
        TestDType::F32,
    );
}

#[test]
fn mqa_gqa_bwd_mqa_hd64_causal_f16_parity() {
    run_bwd_case(
        Case {
            label: "bwd mqa hd64 causal",
            batch: 1,
            num_heads: 8,
            num_kv_heads: 1,
            seq_q: 24,
            seq_k: 24,
            head_dim: 64,
            causal: true,
        },
        TestDType::F16,
    );
}

// ============================================================================
// Dispatch heuristic
// ============================================================================

#[test]
fn should_use_mqa_gqa_matches_documented_heuristic() {
    run_should_use_case();
}

// ============================================================================
// Fixtures and comparison
// ============================================================================

/// Deterministic, closed-form fixture. `phase` decorrelates Q, K, V and dO so a
/// transposed or swapped argument cannot cancel out.
#[cfg(feature = "cuda")]
fn det_data(shape: &[usize], phase: f32) -> Vec<f32> {
    let n: usize = shape.iter().product();
    (0..n)
        .map(|i| ((i as f32) * 0.1 + phase).sin() * 0.5)
        .collect()
}

/// Compare against the reference, printing max absolute difference AND the
/// reference RMS so a tiny-magnitude tensor cannot pass by being tiny.
#[cfg(feature = "cuda")]
fn report_and_assert(actual: &[f32], expected: &[f32], atol: f32, rtol_rms: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: element count mismatch: kernel {} vs reference {}",
        actual.len(),
        expected.len()
    );

    let mut max_abs = 0.0f32;
    let mut max_idx = 0usize;
    let mut sq_sum = 0.0f64;
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.is_finite(),
            "{label}: kernel produced non-finite value {a} at index {i} (reference {e})"
        );
        let diff = (a - e).abs();
        if diff > max_abs {
            max_abs = diff;
            max_idx = i;
        }
        sq_sum += (*e as f64) * (*e as f64);
    }
    let rms = (sq_sum / expected.len() as f64).sqrt() as f32;
    let tol = atol + rtol_rms * rms;

    eprintln!(
        "{label}: n={}, max_abs_diff={max_abs:.4e} (index {max_idx}), ref_rms={rms:.4e}, tol={tol:.4e}",
        expected.len()
    );

    assert!(
        rms > 1e-6,
        "{label}: reference RMS is {rms:.4e} — the fixture is degenerate, so agreement \
         would prove nothing. Fix the fixture, not the tolerance."
    );
    assert!(
        max_abs <= tol,
        "{label}: max_abs_diff {max_abs:.4e} at index {max_idx} exceeds tol {tol:.4e} \
         (ref_rms {rms:.4e}); kernel={} reference={}",
        actual[max_idx],
        expected[max_idx]
    );
}

// ============================================================================
// CUDA runners
// ============================================================================

#[cfg(feature = "cuda")]
fn skip_guard(case: &Case, dtype: TestDType, what: &str) -> bool {
    if !dtype.enabled() {
        eprintln!(
            "SKIPPED: mqa_gqa {what} [{}] '{}' — boostr built without the `f16` feature, \
             so {} tensors cannot be constructed",
            dtype.name(),
            case.label,
            dtype.name()
        );
        return false;
    }
    if !numr::runtime::cuda::is_cuda_available() {
        eprintln!(
            "SKIPPED: mqa_gqa {what} [{}] '{}' — the `cuda` feature is on but no CUDA \
             device is available at runtime",
            dtype.name(),
            case.label
        );
        return false;
    }
    true
}

#[cfg(feature = "cuda")]
fn to_cuda(
    data: &[f32],
    shape: &[usize],
    device: &numr::runtime::cuda::CudaDevice,
    dtype: TestDType,
) -> numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime> {
    use numr::dtype::DType;
    use numr::tensor::Tensor;
    let t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(data, shape, device).unwrap();
    match dtype {
        TestDType::F32 => t,
        TestDType::F16 => t.to_dtype(DType::F16).expect("cast fixture to F16"),
        TestDType::BF16 => t.to_dtype(DType::BF16).expect("cast fixture to BF16"),
    }
}

#[cfg(feature = "cuda")]
fn read_f32(t: &numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>) -> Vec<f32> {
    use numr::dtype::DType;
    if t.dtype() == DType::F32 {
        t.to_vec::<f32>()
    } else {
        t.to_dtype(DType::F32)
            .expect("cast kernel result back to F32 for comparison")
            .to_vec::<f32>()
    }
}

#[cfg(feature = "cuda")]
fn run_fwd_case(case: Case, dtype: TestDType) {
    use boostr::ops::cuda::attention::mqa_gqa::mqa_gqa_fwd;
    use boostr::ops::traits::attention::flash::FlashAttentionOps;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    if !skip_guard(&case, dtype, "fwd") {
        return;
    }

    let q_shape = [case.batch, case.num_heads, case.seq_q, case.head_dim];
    let kv_shape = [case.batch, case.num_kv_heads, case.seq_k, case.head_dim];
    let q_data = det_data(&q_shape, 0.0);
    let k_data = det_data(&kv_shape, 1.7);
    let v_data = det_data(&kv_shape, 3.1);

    // Independent reference: CPU flash path (impl_generic standard attention).
    let (cpu_client, cpu_device) = setup_cpu();
    let q = Tensor::<CpuRuntime>::from_slice(&q_data, &q_shape, &cpu_device).unwrap();
    let k = Tensor::<CpuRuntime>::from_slice(&k_data, &kv_shape, &cpu_device).unwrap();
    let v = Tensor::<CpuRuntime>::from_slice(&v_data, &kv_shape, &cpu_device).unwrap();
    let (cpu_out, _cpu_lse) = cpu_client
        .flash_attention_fwd(
            &q,
            &k,
            &v,
            case.num_heads,
            case.num_kv_heads,
            case.head_dim,
            case.causal,
            0,
            None,
        )
        .expect("CPU reference flash_attention_fwd failed");
    let expected = cpu_out.to_vec::<f32>();

    let (atol, rtol_rms) = dtype.fwd_tol();
    let label = format!("mqa_gqa {} [{}]", case.label, dtype.name());

    let mut ran = false;
    with_cuda_backend(|cuda_client, cuda_device| {
        ran = true;
        let q_c = to_cuda(&q_data, &q_shape, &cuda_device, dtype);
        let k_c = to_cuda(&k_data, &kv_shape, &cuda_device, dtype);
        let v_c = to_cuda(&v_data, &kv_shape, &cuda_device, dtype);

        let (out, lse) = mqa_gqa_fwd(
            &cuda_client,
            &q_c,
            &k_c,
            &v_c,
            case.num_heads,
            case.num_kv_heads,
            case.head_dim,
            case.causal,
        )
        .expect("mqa_gqa_fwd returned an error");

        assert_eq!(
            out.shape(),
            &q_shape,
            "{label}: forward output shape is wrong"
        );
        assert_eq!(
            lse.shape(),
            &[case.batch, case.num_heads, case.seq_q],
            "{label}: logsumexp shape is wrong"
        );

        let lse_host = lse.to_vec::<f32>();
        for (i, l) in lse_host.iter().enumerate() {
            assert!(
                l.is_finite(),
                "{label}: logsumexp[{i}] is {l}; the backward pass consumes this, \
                 so a non-finite entry poisons every gradient"
            );
        }

        report_and_assert(&read_f32(&out), &expected, atol, rtol_rms, &label);
    });
    assert!(
        ran,
        "{label}: the CUDA closure never executed, so nothing was verified — \
         refusing to report a pass"
    );
}

#[cfg(not(feature = "cuda"))]
fn run_fwd_case(case: Case, dtype: TestDType) {
    eprintln!(
        "SKIPPED: mqa_gqa fwd [{}] '{}' — boostr built without the `cuda` feature; \
         the MQA/GQA kernels are CUDA-only. Re-run with `--features cuda`.",
        dtype.name(),
        case.label
    );
}

#[cfg(feature = "cuda")]
fn run_bwd_case(case: Case, dtype: TestDType) {
    use boostr::ops::cuda::attention::mqa_gqa::{mqa_gqa_bwd, mqa_gqa_fwd};
    use boostr::ops::traits::attention::flash::FlashAttentionOps;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    if !skip_guard(&case, dtype, "bwd") {
        return;
    }

    let q_shape = [case.batch, case.num_heads, case.seq_q, case.head_dim];
    let kv_shape = [case.batch, case.num_kv_heads, case.seq_k, case.head_dim];
    let q_data = det_data(&q_shape, 0.0);
    let k_data = det_data(&kv_shape, 1.7);
    let v_data = det_data(&kv_shape, 3.1);
    let dout_data = det_data(&q_shape, 5.3);

    let (cpu_client, cpu_device) = setup_cpu();
    let q = Tensor::<CpuRuntime>::from_slice(&q_data, &q_shape, &cpu_device).unwrap();
    let k = Tensor::<CpuRuntime>::from_slice(&k_data, &kv_shape, &cpu_device).unwrap();
    let v = Tensor::<CpuRuntime>::from_slice(&v_data, &kv_shape, &cpu_device).unwrap();
    let dout = Tensor::<CpuRuntime>::from_slice(&dout_data, &q_shape, &cpu_device).unwrap();

    let (cpu_out, cpu_lse) = cpu_client
        .flash_attention_fwd(
            &q,
            &k,
            &v,
            case.num_heads,
            case.num_kv_heads,
            case.head_dim,
            case.causal,
            0,
            None,
        )
        .expect("CPU reference flash_attention_fwd failed");
    let (ref_dq, ref_dk, ref_dv) = cpu_client
        .flash_attention_bwd(
            &dout,
            &q,
            &k,
            &v,
            &cpu_out,
            &cpu_lse,
            case.num_heads,
            case.num_kv_heads,
            case.head_dim,
            case.causal,
            0,
        )
        .expect("CPU reference flash_attention_bwd failed");
    let expected_dq = ref_dq.to_vec::<f32>();
    let expected_dk = ref_dk.to_vec::<f32>();
    let expected_dv = ref_dv.to_vec::<f32>();

    let (atol, rtol_rms) = dtype.bwd_tol();
    let label = format!("mqa_gqa {} [{}]", case.label, dtype.name());

    let mut ran = false;
    with_cuda_backend(|cuda_client, cuda_device| {
        ran = true;
        let q_c = to_cuda(&q_data, &q_shape, &cuda_device, dtype);
        let k_c = to_cuda(&k_data, &kv_shape, &cuda_device, dtype);
        let v_c = to_cuda(&v_data, &kv_shape, &cuda_device, dtype);
        let dout_c = to_cuda(&dout_data, &q_shape, &cuda_device, dtype);

        // The backward kernel consumes the forward kernel's own O and LSE, so
        // the pair is exercised exactly as a caller would use it.
        let (out, lse) = mqa_gqa_fwd(
            &cuda_client,
            &q_c,
            &k_c,
            &v_c,
            case.num_heads,
            case.num_kv_heads,
            case.head_dim,
            case.causal,
        )
        .expect("mqa_gqa_fwd (feeding backward) returned an error");

        let (dq, dk, dv) = mqa_gqa_bwd(
            &cuda_client,
            &dout_c,
            &q_c,
            &k_c,
            &v_c,
            &out,
            &lse,
            case.num_heads,
            case.num_kv_heads,
            case.head_dim,
            case.causal,
        )
        .expect("mqa_gqa_bwd returned an error");

        assert_eq!(dq.shape(), &q_shape, "{label}: dQ shape is wrong");
        assert_eq!(dk.shape(), &kv_shape, "{label}: dK shape is wrong");
        assert_eq!(dv.shape(), &kv_shape, "{label}: dV shape is wrong");

        // Each gradient is asserted separately: a dQ-only check passes while
        // dK/dV are silently wrong.
        report_and_assert(
            &read_f32(&dq),
            &expected_dq,
            atol,
            rtol_rms,
            &format!("{label} dQ"),
        );
        report_and_assert(
            &read_f32(&dk),
            &expected_dk,
            atol,
            rtol_rms,
            &format!("{label} dK"),
        );
        report_and_assert(
            &read_f32(&dv),
            &expected_dv,
            atol,
            rtol_rms,
            &format!("{label} dV"),
        );
    });
    assert!(
        ran,
        "{label}: the CUDA closure never executed, so nothing was verified — \
         refusing to report a pass"
    );
}

#[cfg(not(feature = "cuda"))]
fn run_bwd_case(case: Case, dtype: TestDType) {
    eprintln!(
        "SKIPPED: mqa_gqa bwd [{}] '{}' — boostr built without the `cuda` feature; \
         the MQA/GQA kernels are CUDA-only. Re-run with `--features cuda`.",
        dtype.name(),
        case.label
    );
}

#[cfg(feature = "cuda")]
fn run_causal_first_row_case() {
    use boostr::ops::cuda::attention::mqa_gqa::mqa_gqa_fwd;

    let case = Case {
        label: "fwd mqa hd32 causal first-query closed form",
        batch: 1,
        num_heads: 8,
        num_kv_heads: 1,
        seq_q: 20,
        seq_k: 20,
        head_dim: 32,
        causal: true,
    };
    if !skip_guard(&case, TestDType::F32, "fwd") {
        return;
    }

    let q_shape = [case.batch, case.num_heads, case.seq_q, case.head_dim];
    let kv_shape = [case.batch, case.num_kv_heads, case.seq_k, case.head_dim];
    let q_data = det_data(&q_shape, 0.0);
    let k_data = det_data(&kv_shape, 1.7);
    let v_data = det_data(&kv_shape, 3.1);
    let v0: Vec<f32> = v_data[..case.head_dim].to_vec();

    let label = "mqa_gqa fwd causal first-query == v[0]";
    let mut ran = false;
    with_cuda_backend(|cuda_client, cuda_device| {
        ran = true;
        let q_c = to_cuda(&q_data, &q_shape, &cuda_device, TestDType::F32);
        let k_c = to_cuda(&k_data, &kv_shape, &cuda_device, TestDType::F32);
        let v_c = to_cuda(&v_data, &kv_shape, &cuda_device, TestDType::F32);

        let (out, _lse) = mqa_gqa_fwd(
            &cuda_client,
            &q_c,
            &k_c,
            &v_c,
            case.num_heads,
            case.num_kv_heads,
            case.head_dim,
            case.causal,
        )
        .expect("mqa_gqa_fwd returned an error");
        let host = read_f32(&out);

        for h in 0..case.num_heads {
            let start = h * case.seq_q * case.head_dim;
            let row = &host[start..start + case.head_dim];
            report_and_assert(row, &v0, 1e-5, 1e-4, &format!("{label} (query head {h})"));
        }
    });
    assert!(
        ran,
        "{label}: the CUDA closure never executed, so nothing was verified — \
         refusing to report a pass"
    );
}

#[cfg(not(feature = "cuda"))]
fn run_causal_first_row_case() {
    eprintln!(
        "SKIPPED: mqa_gqa fwd causal first-query closed form — boostr built without the \
         `cuda` feature; the MQA/GQA kernels are CUDA-only."
    );
}

#[cfg(feature = "cuda")]
fn run_should_use_case() {
    use boostr::ops::cuda::attention::mqa_gqa::should_use_mqa_gqa;

    // Documented heuristic: ratio >= 4 AND head_dim in {32, 64, 128}.
    assert!(should_use_mqa_gqa(8, 1, 32), "MQA at head_dim 32 qualifies");
    assert!(
        should_use_mqa_gqa(8, 2, 64),
        "ratio 4 at head_dim 64 qualifies"
    );
    assert!(
        should_use_mqa_gqa(32, 8, 128),
        "ratio 4 at head_dim 128 qualifies"
    );
    assert!(
        !should_use_mqa_gqa(8, 4, 64),
        "ratio 2 is below the threshold and must fall back to flash_v2"
    );
    assert!(
        !should_use_mqa_gqa(8, 1, 96),
        "head_dim 96 has no MQA/GQA kernel instantiation"
    );
    assert!(
        !should_use_mqa_gqa(8, 0, 64),
        "num_kv_heads = 0 must not divide by zero"
    );
}

#[cfg(not(feature = "cuda"))]
fn run_should_use_case() {
    eprintln!(
        "SKIPPED: should_use_mqa_gqa heuristic — boostr built without the `cuda` feature; \
         the function is gated behind it."
    );
}
