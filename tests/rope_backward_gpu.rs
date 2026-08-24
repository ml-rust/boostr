//! Gradient flow through the FUSED GPU RoPE kernels (CUDA and WebGPU).
//!
//! Run with:
//!   cd boostr && cargo test --features cuda,f16 --test rope_backward_gpu
//!
//! The CPU backend composes RoPE out of `var_narrow`/`var_mul`/`var_sub`/
//! `var_add`/`var_cat`, so its graph is built for free — `tests/rope_backward.rs`
//! already covers it. The CUDA and WebGPU backends run a single fused kernel,
//! which used to return `Var::new(output, false)`: a DETACHED LEAF. Backward
//! stopped there, so `q_proj`, `k_proj` and everything upstream of them
//! (including the embedding table) silently received no gradient at all — no
//! error, no NaN.
//!
//! Every test here therefore checks two separate things:
//!
//! 1. **Existence** — `grads.get(x.id())` returns something finite and
//!    non-zero. Without the backward node this is `None` and the test fails.
//! 2. **Numerics** — the GPU gradient matches the CPU composed path's gradient
//!    on identical inputs. A backward node that runs but rotates the wrong way
//!    (un-negated sine) or pairs the wrong elements (split-half vs interleaved)
//!    passes the existence check and fails this one.
//!
//! The loss is `sum(rope(x) * w)` with a NON-UNIFORM `w`, not `sum(rope(x))`.
//! A uniform upstream gradient is a weak discriminator: it makes the standard
//! and interleaved adjoints agree on many entries and hides sign errors.

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use boostr::ops::RoPEOps;
use numr::autograd::{Var, backward, var_mul, var_sum};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

const BATCH: usize = 2;
const HEADS: usize = 3;
const SEQ: usize = 5;
const HEAD_DIM: usize = 8;
const HALF_DIM: usize = HEAD_DIM / 2;
const NUMEL: usize = BATCH * HEADS * SEQ * HEAD_DIM;
const SHAPE: [usize; 4] = [BATCH, HEADS, SEQ, HEAD_DIM];
const CACHE_SHAPE: [usize; 2] = [SEQ, HALF_DIM];

/// Which fused kernel to exercise.
#[derive(Clone, Copy)]
enum Variant {
    Standard,
    Interleaved,
    /// `attn_scale` deliberately != 1.0, so the scale is actually under test.
    Yarn(f32),
}

impl Variant {
    fn label(self) -> &'static str {
        match self {
            Variant::Standard => "standard",
            Variant::Interleaved => "interleaved",
            Variant::Yarn(_) => "yarn",
        }
    }
}

/// Deterministic, distinct-per-index values.
fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.017 + seed;
            x.sin() * 0.9 + (x * 2.3).cos() * 0.4
        })
        .collect()
}

/// Real RoPE caches, so `sin` is genuinely non-zero and the rotation is a
/// rotation. A cache of zeros would make a sign error in the adjoint invisible.
fn caches() -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0.0f32; SEQ * HALF_DIM];
    let mut sin = vec![0.0f32; SEQ * HALF_DIM];
    for pos in 0..SEQ {
        for i in 0..HALF_DIM {
            let freq = 1.0f32 / 10000f32.powf(2.0 * i as f32 / HEAD_DIM as f32);
            let angle = pos as f32 * freq;
            cos[pos * HALF_DIM + i] = angle.cos();
            sin[pos * HALF_DIM + i] = angle.sin();
        }
    }
    (cos, sin)
}

/// `dL/dx` for `L = sum(rope(x) * w)` on the CPU composed path — the reference.
fn cpu_reference_grad(variant: Variant) -> Vec<f32> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let (cos_data, sin_data) = caches();

    let x = Var::<CpuRuntime>::new(
        Tensor::from_slice(&values(NUMEL, 0.3), &SHAPE, &device).unwrap(),
        true,
    );
    let cos = Var::<CpuRuntime>::new(
        Tensor::from_slice(&cos_data, &CACHE_SHAPE, &device).unwrap(),
        false,
    );
    let sin = Var::<CpuRuntime>::new(
        Tensor::from_slice(&sin_data, &CACHE_SHAPE, &device).unwrap(),
        false,
    );
    let w = Var::<CpuRuntime>::new(
        Tensor::from_slice(&values(NUMEL, 1.9), &SHAPE, &device).unwrap(),
        false,
    );

    let out = match variant {
        Variant::Standard => client.apply_rope(&x, &cos, &sin),
        Variant::Interleaved => client.apply_rope_interleaved(&x, &cos, &sin),
        Variant::Yarn(scale) => client.apply_rope_yarn(&x, &cos, &sin, scale),
    }
    .expect("cpu rope forward");

    let weighted = var_mul(&out, &w, &client).expect("cpu weighting");
    let loss = var_sum(&weighted, &[0, 1, 2, 3], false, &client).expect("cpu loss reduces");
    let grads = backward(&loss, &client).expect("cpu backward");
    grads
        .get(x.tensor().id())
        .expect("cpu composed path produced no gradient — the reference itself is broken")
        .contiguous()
        .expect("cpu gradient contiguous")
        .to_vec::<f32>()
}

/// Compare a GPU gradient against the CPU reference, elementwise.
fn assert_matches_cpu(label: &str, got: &[f32], want: &[f32], rtol: f32, atol: f32) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: gradient length {} vs cpu {}",
        got.len(),
        want.len()
    );
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{label}: gradient contains a non-finite value"
    );
    assert!(
        got.iter().any(|v| v.abs() > 1e-6),
        "{label}: gradient is all zero"
    );
    for (i, (g, c)) in got.iter().zip(want.iter()).enumerate() {
        let tol = atol + rtol * c.abs();
        assert!(
            (g - c).abs() <= tol,
            "{label}: gradient mismatch at {i}: gpu={g}, cpu={c} (tol={tol})"
        );
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use numr::autograd::var_cast;
    use numr::dtype::DType;
    use numr::ops::TypeConversionOps;
    use numr::runtime::Runtime;
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    use std::sync::{Mutex, OnceLock};

    // CUDA tests in this crate serialize on a process-wide lock.
    static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
        CUDA_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    /// `dL/dx` for `L = sum(rope(x) * w)` through the fused CUDA kernel.
    ///
    /// The caches are handed over as F32 on purpose even in the BF16 case: that
    /// is what a real model does, and it exercises the cast inside
    /// `validate_rope_inputs` whose result the backward node has to reuse.
    fn cuda_grad(
        client: &CudaClient,
        device: &CudaDevice,
        variant: Variant,
        dtype: DType,
    ) -> Vec<f32> {
        let (cos_data, sin_data) = caches();
        let label = variant.label();

        let x_t = Tensor::<CudaRuntime>::from_slice(&values(NUMEL, 0.3), &SHAPE, device).unwrap();
        let x_t = if dtype == DType::F32 {
            x_t
        } else {
            client.cast(&x_t, dtype).expect("cast x to test dtype")
        };
        let x = Var::new(x_t, true);

        let cos = Var::new(
            Tensor::<CudaRuntime>::from_slice(&cos_data, &CACHE_SHAPE, device).unwrap(),
            false,
        );
        let sin = Var::new(
            Tensor::<CudaRuntime>::from_slice(&sin_data, &CACHE_SHAPE, device).unwrap(),
            false,
        );

        let out = match variant {
            Variant::Standard => client.apply_rope(&x, &cos, &sin),
            Variant::Interleaved => client.apply_rope_interleaved(&x, &cos, &sin),
            Variant::Yarn(scale) => client.apply_rope_yarn(&x, &cos, &sin, scale),
        }
        .unwrap_or_else(|e| panic!("cuda {label}: forward failed: {e}"));

        // Reduce in F32 so the loss reduction is never the thing under test.
        // `CastBackward` hands a BF16 gradient back down to the RoPE node, which
        // is exactly the dtype its saved caches carry.
        let out = var_cast(&out, DType::F32, client).expect("cast rope output to f32");
        let w = Var::new(
            Tensor::<CudaRuntime>::from_slice(&values(NUMEL, 1.9), &SHAPE, device).unwrap(),
            false,
        );
        let weighted = var_mul(&out, &w, client).expect("weighting");
        let loss = var_sum(&weighted, &[0, 1, 2, 3], false, client).expect("loss reduces");

        let grads = backward(&loss, client)
            .unwrap_or_else(|e| panic!("cuda {label}: backward failed: {e}"));

        // THIS is what the severed graph broke: no entry at all for the leaf.
        let grad = grads.get(x.tensor().id()).unwrap_or_else(|| {
            panic!(
                "cuda {label} ({dtype:?}): no gradient for the RoPE input — the fused \
                 kernel returned a detached leaf and backward stopped there"
            )
        });
        let grad = grad.contiguous().expect("gradient contiguous");
        let grad = if grad.dtype() == DType::F32 {
            grad
        } else {
            client
                .cast(&grad, DType::F32)
                .expect("cast gradient to f32")
        };
        grad.to_vec::<f32>()
    }

    fn run(variant: Variant, dtype: DType, rtol: f32, atol: f32) {
        if !numr::runtime::cuda::is_cuda_available() {
            eprintln!(
                "SKIPPED rope_backward_gpu::cuda [{} {dtype:?}]: CUDA not available",
                variant.label()
            );
            return;
        }
        let _lock = cuda_lock();
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let got = cuda_grad(&client, &device, variant, dtype);
        let want = cpu_reference_grad(variant);
        assert_matches_cpu(
            &format!("cuda {} {dtype:?}", variant.label()),
            &got,
            &want,
            rtol,
            atol,
        );
    }

    #[test]
    fn standard_f32() {
        run(Variant::Standard, DType::F32, 1e-4, 1e-5);
    }

    #[test]
    fn interleaved_f32() {
        run(Variant::Interleaved, DType::F32, 1e-4, 1e-5);
    }

    #[test]
    fn yarn_f32() {
        run(Variant::Yarn(1.7), DType::F32, 1e-4, 1e-5);
    }

    // BF16 keeps ~8 mantissa bits, so the tolerance is set by the dtype, not by
    // the kernel. The CPU reference stays in F32 — that asymmetry is the point.
    #[cfg(feature = "f16")]
    #[test]
    fn standard_bf16() {
        run(Variant::Standard, DType::BF16, 4e-2, 2e-2);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn interleaved_bf16() {
        run(Variant::Interleaved, DType::BF16, 4e-2, 2e-2);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn yarn_bf16() {
        run(Variant::Yarn(1.7), DType::BF16, 4e-2, 3e-2);
    }
}

// Named `webgpu`, not `wgpu`, so the module never shadows the `wgpu` crate.
#[cfg(feature = "wgpu")]
mod webgpu {
    use super::*;
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
    use std::sync::{Mutex, OnceLock};

    static WGPU_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn wgpu_lock() -> std::sync::MutexGuard<'static, ()> {
        WGPU_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    /// `dL/dx` for `L = sum(rope(x) * w)` through the fused WGSL shader.
    /// The WebGPU RoPE path is F32-only, so there is no BF16 counterpart.
    fn wgpu_grad(client: &WgpuClient, device: &WgpuDevice, variant: Variant) -> Vec<f32> {
        let (cos_data, sin_data) = caches();
        let label = variant.label();

        let x = Var::new(
            Tensor::<WgpuRuntime>::from_slice(&values(NUMEL, 0.3), &SHAPE, device).unwrap(),
            true,
        );
        let cos = Var::new(
            Tensor::<WgpuRuntime>::from_slice(&cos_data, &CACHE_SHAPE, device).unwrap(),
            false,
        );
        let sin = Var::new(
            Tensor::<WgpuRuntime>::from_slice(&sin_data, &CACHE_SHAPE, device).unwrap(),
            false,
        );

        let out = match variant {
            Variant::Standard => client.apply_rope(&x, &cos, &sin),
            Variant::Interleaved => client.apply_rope_interleaved(&x, &cos, &sin),
            Variant::Yarn(scale) => client.apply_rope_yarn(&x, &cos, &sin, scale),
        }
        .unwrap_or_else(|e| panic!("wgpu {label}: forward failed: {e}"));

        let w = Var::new(
            Tensor::<WgpuRuntime>::from_slice(&values(NUMEL, 1.9), &SHAPE, device).unwrap(),
            false,
        );
        let weighted = var_mul(&out, &w, client).expect("weighting");
        let loss = var_sum(&weighted, &[0, 1, 2, 3], false, client).expect("loss reduces");

        let grads = backward(&loss, client)
            .unwrap_or_else(|e| panic!("wgpu {label}: backward failed: {e}"));

        let grad = grads.get(x.tensor().id()).unwrap_or_else(|| {
            panic!(
                "wgpu {label}: no gradient for the RoPE input — the fused shader \
                 returned a detached leaf and backward stopped there"
            )
        });
        grad.contiguous()
            .expect("gradient contiguous")
            .to_vec::<f32>()
    }

    fn run(variant: Variant) {
        let _lock = wgpu_lock();
        let device = WgpuDevice::new(0);
        let client = match WgpuClient::new(device.clone()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "SKIPPED rope_backward_gpu::wgpu [{}]: no WebGPU adapter ({e:?})",
                    variant.label()
                );
                return;
            }
        };

        let got = wgpu_grad(&client, &device, variant);
        let want = cpu_reference_grad(variant);
        assert_matches_cpu(
            &format!("wgpu {}", variant.label()),
            &got,
            &want,
            1e-4,
            1e-5,
        );
    }

    #[test]
    fn standard_f32() {
        run(Variant::Standard);
    }

    #[test]
    fn interleaved_f32() {
        run(Variant::Interleaved);
    }

    #[test]
    fn yarn_f32() {
        run(Variant::Yarn(1.7));
    }
}
