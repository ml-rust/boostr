//! `RoPEOps` on a permuted view.
//!
//! Every attention block projects `[B, S, H*D]`, reshapes to `[B, S, H, D]`
//! and permutes to `[B, H, S, D]` before RoPE. That permute is a strided
//! view. The fused CUDA kernels read it through its strides and write a
//! dense `[B, H, S, D]`; the CPU path composes strided numr ops. Both must
//! return exactly what they return for a dense copy of the same view, and a
//! view the kernel cannot address must be refused, never rotated wrong.
//!
//! Run with:
//!   cd boostr && cargo test --test rope_strided_view
//!   cd boostr && cargo test --features cuda --test rope_strided_view

use boostr::ops::RoPEOps;
use numr::autograd::{Var, var_permute};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

const BATCH: usize = 2;
const HEADS: usize = 3;
const SEQ: usize = 5;
const HEAD_DIM: usize = 8;
const HALF_DIM: usize = HEAD_DIM / 2;
const NUMEL: usize = BATCH * HEADS * SEQ * HEAD_DIM;
/// The projection's own layout, before the permute.
const SEQ_MAJOR: [usize; 4] = [BATCH, SEQ, HEADS, HEAD_DIM];
const CACHE_SHAPE: [usize; 2] = [SEQ, HALF_DIM];

fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.017 + seed;
            x.sin() * 0.9 + (x * 2.3).cos() * 0.4
        })
        .collect()
}

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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// CPU: the permuted view and its dense copy rotate to the same bits.
#[test]
fn cpu_permuted_view_matches_dense_copy_bitwise() {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let (cos_data, sin_data) = caches();

    let seq_major =
        Tensor::<CpuRuntime>::from_slice(&values(NUMEL, 0.3), &SEQ_MAJOR, &device).unwrap();
    let view = var_permute(&Var::new(seq_major, false), &[0, 2, 1, 3]).unwrap();
    assert!(!view.tensor().is_contiguous(), "the permute must be a view");
    let dense = Var::new(view.tensor().contiguous().unwrap(), false);

    let cos = Var::new(
        Tensor::<CpuRuntime>::from_slice(&cos_data, &CACHE_SHAPE, &device).unwrap(),
        false,
    );
    let sin = Var::new(
        Tensor::<CpuRuntime>::from_slice(&sin_data, &CACHE_SHAPE, &device).unwrap(),
        false,
    );

    let from_view = client.apply_rope(&view, &cos, &sin).unwrap();
    let from_dense = client.apply_rope(&dense, &cos, &sin).unwrap();
    assert_eq!(from_view.shape(), &[BATCH, HEADS, SEQ, HEAD_DIM]);
    assert_eq!(
        bits(&from_view.tensor().contiguous().unwrap().to_vec::<f32>()),
        bits(&from_dense.tensor().contiguous().unwrap().to_vec::<f32>()),
    );
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use numr::autograd::{backward, var_mul, var_sum};
    use numr::dtype::DType;
    use numr::ops::TypeConversionOps;
    use numr::runtime::Runtime;
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    use std::sync::{Mutex, OnceLock};

    static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
        CUDA_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    fn setup(label: &str) -> Option<(CudaClient, CudaDevice)> {
        if !numr::runtime::cuda::is_cuda_available() {
            eprintln!("SKIPPED rope_strided_view::cuda [{label}]: CUDA not available");
            return None;
        }
        let device = CudaDevice::new(0);
        Some((CudaRuntime::default_client(&device), device))
    }

    #[derive(Clone, Copy)]
    enum Variant {
        Standard,
        Interleaved,
        Yarn(f32),
    }

    fn apply(
        client: &CudaClient,
        variant: Variant,
        x: &Var<CudaRuntime>,
        cos: &Var<CudaRuntime>,
        sin: &Var<CudaRuntime>,
    ) -> boostr::error::Result<Var<CudaRuntime>> {
        match variant {
            Variant::Standard => client.apply_rope(x, cos, sin),
            Variant::Interleaved => client.apply_rope_interleaved(x, cos, sin),
            Variant::Yarn(scale) => client.apply_rope_yarn(x, cos, sin, scale),
        }
    }

    fn cache_vars(device: &CudaDevice) -> (Var<CudaRuntime>, Var<CudaRuntime>) {
        let (cos_data, sin_data) = caches();
        (
            Var::new(
                Tensor::<CudaRuntime>::from_slice(&cos_data, &CACHE_SHAPE, device).unwrap(),
                false,
            ),
            Var::new(
                Tensor::<CudaRuntime>::from_slice(&sin_data, &CACHE_SHAPE, device).unwrap(),
                false,
            ),
        )
    }

    /// `[B, S, H, D]` on the device in `dtype`, permuted to `[B, H, S, D]`.
    fn permuted(client: &CudaClient, device: &CudaDevice, dtype: DType) -> Var<CudaRuntime> {
        let t = Tensor::<CudaRuntime>::from_slice(&values(NUMEL, 0.3), &SEQ_MAJOR, device).unwrap();
        let t = if dtype == DType::F32 {
            t
        } else {
            client.cast(&t, dtype).unwrap()
        };
        let view = var_permute(&Var::new(t, false), &[0, 2, 1, 3]).unwrap();
        assert!(!view.tensor().is_contiguous(), "the permute must be a view");
        view
    }

    fn host_f32(client: &CudaClient, t: &Tensor<CudaRuntime>) -> Vec<f32> {
        let t = t.contiguous().unwrap();
        let t = if t.dtype() == DType::F32 {
            t
        } else {
            client.cast(&t, DType::F32).unwrap()
        };
        t.to_vec::<f32>()
    }

    /// The kernel reads the view through its strides and the dense copy
    /// through row-major strides; the arithmetic per element is identical,
    /// so the outputs must agree bit for bit.
    fn view_matches_dense(variant: Variant, dtype: DType, label: &str) {
        let Some((client, device)) = setup(label) else {
            return;
        };
        let _lock = cuda_lock();
        let (cos, sin) = cache_vars(&device);
        let view = permuted(&client, &device, dtype);
        let dense = Var::new(view.tensor().contiguous().unwrap(), false);

        let from_view = apply(&client, variant, &view, &cos, &sin)
            .unwrap_or_else(|e| panic!("{label}: view forward failed: {e}"));
        let from_dense = apply(&client, variant, &dense, &cos, &sin).unwrap();
        assert_eq!(from_view.shape(), &[BATCH, HEADS, SEQ, HEAD_DIM]);
        assert!(
            from_view.tensor().is_contiguous(),
            "{label}: output is dense"
        );
        assert_eq!(
            bits(&host_f32(&client, from_view.tensor())),
            bits(&host_f32(&client, from_dense.tensor())),
            "{label}: permuted view and dense copy diverged"
        );
    }

    #[test]
    fn standard_f32_view_matches_dense() {
        view_matches_dense(Variant::Standard, DType::F32, "standard f32");
    }

    #[test]
    fn interleaved_f32_view_matches_dense() {
        view_matches_dense(Variant::Interleaved, DType::F32, "interleaved f32");
    }

    #[test]
    fn yarn_f32_view_matches_dense() {
        view_matches_dense(Variant::Yarn(1.7), DType::F32, "yarn f32");
    }

    #[cfg(feature = "f16")]
    #[test]
    fn standard_bf16_view_matches_dense() {
        view_matches_dense(Variant::Standard, DType::BF16, "standard bf16");
    }

    #[cfg(feature = "f16")]
    #[test]
    fn standard_f16_view_matches_dense() {
        view_matches_dense(Variant::Standard, DType::F16, "standard f16");
    }

    /// The CUDA result on the view agrees with the CPU composed path on the
    /// same view.
    #[test]
    fn standard_f32_view_matches_cpu() {
        let Some((client, device)) = setup("standard f32 vs cpu") else {
            return;
        };
        let _lock = cuda_lock();
        let (cos, sin) = cache_vars(&device);
        let view = permuted(&client, &device, DType::F32);
        let got = host_f32(
            &client,
            client.apply_rope(&view, &cos, &sin).unwrap().tensor(),
        );

        let cpu_device = CpuDevice::new();
        let cpu_client = CpuClient::new(cpu_device.clone());
        let (cos_data, sin_data) = caches();
        let seq_major =
            Tensor::<CpuRuntime>::from_slice(&values(NUMEL, 0.3), &SEQ_MAJOR, &cpu_device).unwrap();
        let cpu_view = var_permute(&Var::new(seq_major, false), &[0, 2, 1, 3]).unwrap();
        let cpu_cos = Var::new(
            Tensor::<CpuRuntime>::from_slice(&cos_data, &CACHE_SHAPE, &cpu_device).unwrap(),
            false,
        );
        let cpu_sin = Var::new(
            Tensor::<CpuRuntime>::from_slice(&sin_data, &CACHE_SHAPE, &cpu_device).unwrap(),
            false,
        );
        let want = cpu_client
            .apply_rope(&cpu_view, &cpu_cos, &cpu_sin)
            .unwrap()
            .tensor()
            .contiguous()
            .unwrap()
            .to_vec::<f32>();

        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let tol = 1e-7 + 1e-5 * w.abs();
            assert!(
                (g - w).abs() <= tol,
                "cuda vs cpu mismatch at {i}: {g} vs {w}"
            );
        }
    }

    /// Backward through the view: the gradient that reaches the `[B, S, H, D]`
    /// leaf must equal the one from the dense path.
    #[test]
    fn standard_f32_view_backward_matches_dense() {
        let Some((client, device)) = setup("standard f32 backward") else {
            return;
        };
        let _lock = cuda_lock();
        let (cos, sin) = cache_vars(&device);
        let w = Var::new(
            Tensor::<CudaRuntime>::from_slice(
                &values(NUMEL, 1.9),
                &[BATCH, HEADS, SEQ, HEAD_DIM],
                &device,
            )
            .unwrap(),
            false,
        );

        let grad_of = |through_view: bool| -> Vec<f32> {
            let leaf = Var::new(
                Tensor::<CudaRuntime>::from_slice(&values(NUMEL, 0.3), &SEQ_MAJOR, &device)
                    .unwrap(),
                true,
            );
            let view = var_permute(&leaf, &[0, 2, 1, 3]).unwrap();
            let x = if through_view {
                view
            } else {
                boostr::nn::var_contiguous(&view).unwrap()
            };
            let out = client.apply_rope(&x, &cos, &sin).unwrap();
            let loss = var_sum(
                &var_mul(&out, &w, &client).unwrap(),
                &[0, 1, 2, 3],
                false,
                &client,
            )
            .unwrap();
            let grads = backward(&loss, &client).unwrap();
            host_f32(
                &client,
                grads
                    .get(leaf.tensor().id())
                    .expect("gradient reaches the projection leaf"),
            )
        };

        let via_view = grad_of(true);
        let via_dense = grad_of(false);
        assert!(via_view.iter().any(|g| g.abs() > 1e-6), "gradient is zero");
        assert_eq!(bits(&via_view), bits(&via_dense));
    }

    /// A view whose head dimension is not unit-stride cannot be addressed by
    /// the kernel: it is refused with a named reason, not rotated wrong.
    #[test]
    fn non_unit_head_stride_is_rejected() {
        let Some((client, device)) = setup("rejection") else {
            return;
        };
        let _lock = cuda_lock();
        let (cos, sin) = cache_vars(&device);
        // Stored as [B, H, D, S]; swapping the last two axes yields a
        // [B, H, S, D] view whose D stride is S.
        let stored = Tensor::<CudaRuntime>::from_slice(
            &values(NUMEL, 0.3),
            &[BATCH, HEADS, HEAD_DIM, SEQ],
            &device,
        )
        .unwrap();
        let view = var_permute(&Var::new(stored, false), &[0, 1, 3, 2]).unwrap();
        assert_eq!(view.shape(), &[BATCH, HEADS, SEQ, HEAD_DIM]);

        let msg = match client.apply_rope(&view, &cos, &sin) {
            Ok(_) => panic!("a non-unit head stride must be refused"),
            Err(err) => err.to_string(),
        };
        assert!(
            msg.contains("unsupported layout") && msg.contains("unit stride"),
            "error must name the layout problem, got: {msg}"
        );

        // The same data made dense is accepted.
        let dense = Var::new(view.tensor().contiguous().unwrap(), false);
        client.apply_rope(&dense, &cos, &sin).unwrap();
    }
}
