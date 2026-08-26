//! `attention_core_flash` backward on CUDA: the gradient that reaches
//! `FlashAttentionBackward` is a permuted VIEW, and the fused kernel indexes it
//! through a raw device pointer.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda,f16 --test attention_core_flash_bwd_cuda
//!
//! `attention_epilogue` ends the forward with
//! `permute([0,2,1,3]) -> contiguous -> reshape`. Its backward chain runs in
//! reverse: `ReshapeBackward` (which contiguizes), `ContiguousBackward`
//! (identity), then `PermuteBackward` — and `PermuteBackward` returns a
//! strided view. That view is `dout` for the flash backward node.
//!
//! Without the normalization in `FlashAttentionBackward::backward`, every case
//! here fails with
//! `flash_attention_bwd failed: invalid argument 'contiguity': backward
//! requires contiguous dout, output, lse`, raised by
//! `ops/cuda/attention/flash_bwd.rs`. `output` and `lse` are freshly allocated
//! by the forward and are always contiguous — `dout` is the one that is not.
//!
//! Neither GQA nor BF16 is required to trigger it: the MHA F32 case here fails
//! the same way, because the epilogue is shared. Both are covered anyway,
//! because the reported failure was a BF16 GQA model and GQA additionally
//! exercises the `repeat_interleave` / `sum_gqa_grads` path that reduces dK/dV
//! back to `num_kv_heads`.
//!
//! The CPU backend does NOT reproduce this: `standard_attention_bwd` consumes
//! `dout` through `matmul`, which accepts strided inputs.
//!
//! The F16/BF16 cases additionally cover a second, dtype-only defect in the
//! same backward: the kernel accumulated dQ with a 4-byte `atomicAdd` aimed at
//! a 2-byte element, which is a misaligned address at every odd element index.
//! See the per-test docs below.

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::model::{AttentionCoreSpec, AttentionKernel, attention_core_flash};
use boostr::nn::RoPE;
use numr::autograd::{Var, backward, var_cast, var_sum};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

// CUDA tests in this crate serialize on a process-wide lock.
static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
    CUDA_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

fn cuda_available() -> bool {
    numr::runtime::cuda::is_cuda_available()
}

const BATCH: usize = 2;
const SEQ: usize = 16;
/// The fused flash kernels are instantiated per head dim; 64 is the one the
/// reported Llama-3.2-1B config uses.
const HEAD_DIM: usize = 64;

/// Deterministic pseudo-random values, distinct per index.
fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.017 + seed;
            x.sin() * 0.9 + (x * 2.3).cos() * 0.4
        })
        .collect()
}

/// A leaf `Var` of the requested dtype, tracking gradients.
fn leaf(
    client: &CudaClient,
    shape: &[usize],
    seed: f32,
    dtype: DType,
    device: &CudaDevice,
) -> Var<CudaRuntime> {
    let n: usize = shape.iter().product();
    let t = Tensor::<CudaRuntime>::from_slice(&values(n, seed), shape, device).unwrap();
    let t = if dtype == DType::F32 {
        t
    } else {
        client.cast(&t, dtype).expect("cast leaf to test dtype")
    };
    Var::new(t, true)
}

/// Read a gradient back as f32 and assert it is finite and not all zero.
fn assert_finite_nonzero(client: &CudaClient, grad: &Tensor<CudaRuntime>, label: &str) {
    let grad = grad.contiguous().expect("gradient contiguous");
    let grad = if grad.dtype() == DType::F32 {
        grad
    } else {
        client
            .cast(&grad, DType::F32)
            .expect("cast gradient to f32")
    };
    let host = grad.to_vec::<f32>();
    assert!(!host.is_empty(), "{label}: empty gradient");
    assert!(
        host.iter().all(|x| x.is_finite()),
        "{label}: gradient contains a non-finite value"
    );
    assert!(
        host.iter().any(|x| x.abs() > 1e-6),
        "{label}: gradient is all zero"
    );
}

/// Forward + backward through `attention_core_flash` on CUDA, asserting that
/// backward succeeds and dQ/dK/dV are finite and non-zero.
fn run_case(num_heads: usize, num_kv_heads: usize, dtype: DType, label: &str) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);

    let q = leaf(
        &client,
        &[BATCH, SEQ, num_heads * HEAD_DIM],
        0.3,
        dtype,
        &device,
    );
    let kv_shape = [BATCH, SEQ, num_kv_heads * HEAD_DIM];
    let k = leaf(&client, &kv_shape, 1.1, dtype, &device);
    let v = leaf(&client, &kv_shape, 2.7, dtype, &device);

    let rope = RoPE::<CudaRuntime>::precompute_freqs(SEQ, HEAD_DIM, 10000.0, None, &device)
        .expect("rope cache builds");

    let spec = AttentionCoreSpec {
        num_heads,
        num_kv_heads,
        head_dim: HEAD_DIM,
        q_norm: None,
        k_norm: None,
        use_alibi: false,
        skip_rope: false,
        sliding_window: 0,
        // Ignored by `attention_core_flash`; stated for honesty.
        kernel: AttentionKernel::Flash,
    };

    let out = attention_core_flash(
        &client,
        &q,
        &k,
        &v,
        Some(rope.cos_cache()),
        Some(rope.sin_cache()),
        &spec,
    )
    .unwrap_or_else(|e| panic!("{label}: attention_core_flash forward failed: {e}"));

    let expected = [BATCH, SEQ, num_heads * HEAD_DIM];
    assert_eq!(out.shape(), &expected[..], "{label}: output shape");

    // Reduce in f32 so the loss reduction never becomes the thing under test.
    let out = var_cast(&out, DType::F32, &client).expect("cast attention output to f32");
    let loss = var_sum(&out, &[0, 1, 2], false, &client).expect("loss reduces");

    // THIS is the call that fails without the fix: the flash backward node
    // receives the permuted view produced by `PermuteBackward`.
    let grads = backward(&loss, &client)
        .unwrap_or_else(|e| panic!("{label}: backward through attention_core_flash failed: {e}"));

    for (name, leaf_var) in [("dQ", &q), ("dK", &k), ("dV", &v)] {
        let grad = grads
            .get(leaf_var.tensor().id())
            .unwrap_or_else(|| panic!("{label}: no {name}"));
        assert_finite_nonzero(&client, grad, &format!("{label}: {name}"));
    }
}

/// GQA at F32 — `num_kv_heads < num_heads`, so this also covers the
/// `repeat_interleave` / `sum_gqa_grads` reduction of dK/dV.
#[test]
fn flash_bwd_gqa_f32_survives_permuted_grad() {
    if !cuda_available() {
        eprintln!("flash_bwd_gqa_f32_survives_permuted_grad: CUDA not available, skipping");
        return;
    }
    let _lock = cuda_lock();
    run_case(8, 2, DType::F32, "GQA f32");
}

/// MHA at F32 — proves GQA is NOT required to trigger the failure. The
/// epilogue, and therefore the permuted gradient, is shared by every geometry.
#[test]
fn flash_bwd_mha_f32_survives_permuted_grad() {
    if !cuda_available() {
        eprintln!("flash_bwd_mha_f32_survives_permuted_grad: CUDA not available, skipping");
        return;
    }
    let _lock = cuda_lock();
    run_case(8, 8, DType::F32, "MHA f32");
}

/// GQA at BF16 — the exact dtype/geometry combination the Llama-3.2-1B LoRA run
/// reported. BF16 is not required to trigger the failure either, but this is
/// the configuration that surfaced it.
#[cfg(feature = "f16")]
#[test]
fn flash_bwd_gqa_bf16_survives_permuted_grad() {
    if !cuda_available() {
        eprintln!("flash_bwd_gqa_bf16_survives_permuted_grad: CUDA not available, skipping");
        return;
    }
    let _lock = cuda_lock();
    run_case(8, 2, DType::BF16, "GQA bf16");
}

/// MHA at BF16 — isolates the SECOND defect these cases caught, which is about
/// dtype alone and not about the permuted gradient: the F16/BF16 backward
/// kernels aimed a 4-byte `atomicAdd` at a 2-byte dQ element, so every odd
/// element index was a misaligned address and the launch died with
/// `CUDA_ERROR_MISALIGNED_ADDRESS`. GQA is NOT required — this MHA case fails
/// the same way, because dQ is indexed identically for every geometry. dQ is
/// now an F32 accumulator that the launcher casts down.
#[cfg(feature = "f16")]
#[test]
fn flash_bwd_mha_bf16_survives_permuted_grad() {
    if !cuda_available() {
        eprintln!("flash_bwd_mha_bf16_survives_permuted_grad: CUDA not available, skipping");
        return;
    }
    let _lock = cuda_lock();
    run_case(8, 8, DType::BF16, "MHA bf16");
}

/// GQA at F16 — the misaligned dQ atomic is identical in the F16 kernel
/// (`__half` is also 2 bytes), so F16 needs the same coverage as BF16.
#[cfg(feature = "f16")]
#[test]
fn flash_bwd_gqa_f16_survives_permuted_grad() {
    if !cuda_available() {
        eprintln!("flash_bwd_gqa_f16_survives_permuted_grad: CUDA not available, skipping");
        return;
    }
    let _lock = cuda_lock();
    run_case(8, 2, DType::F16, "GQA f16");
}

/// MHA at F16 — the F16 counterpart of the MHA BF16 case above.
#[cfg(feature = "f16")]
#[test]
fn flash_bwd_mha_f16_survives_permuted_grad() {
    if !cuda_available() {
        eprintln!("flash_bwd_mha_f16_survives_permuted_grad: CUDA not available, skipping");
        return;
    }
    let _lock = cuda_lock();
    run_case(8, 8, DType::F16, "MHA f16");
}
