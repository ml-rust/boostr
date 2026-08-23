//! `attention_core`: the `Masked` and `Flash` kernels must compute the SAME
//! function.
//!
//! Run with:
//!   cd boostr && cargo test --test attention_core_kernels
//!
//! `Masked` repeats the KV heads and materializes a `[1, 1, sq, sk]` additive
//! mask; `Flash` hands `num_kv_heads` and the window to the fused kernel and
//! materializes neither. They share the prologue (reshape, Q/K norm, RoPE), so
//! everything these tests compare is the attention step itself: GQA broadcast,
//! causality, and the sliding window.
//!
//! Two hazards these cases exist to catch, both of which keep every shape valid
//! and still emit fluent text:
//!
//! - The window sentinel and its inclusivity. `0` disables; otherwise row `i`
//!   keeps exactly `window` keys, the current token included. `Masked` gets
//!   this from `causal_window_mask` (triu/tril), `Flash` from the kernel's
//!   `window_size` loop bound — two independent implementations, so an
//!   off-by-one in either one shows up as a mismatch here.
//! - Absolute vs relative query position. Row `i` is at absolute position
//!   `sk - sq + i`. The `sq != sk` cases are the only ones where a relative
//!   reading differs from an absolute one.
//!
//! `window_actually_bites` guards the guard: if the chosen window were wide
//! enough to select every causally-legal key, every equivalence case would pass
//! with the window code deleted.
//!
//! Scope: F32 on the CPU backend. `f16`/`bf16` and the CUDA flash kernels take
//! different code paths and would need their own tolerances; nothing here says
//! anything about them.

use boostr::model::{AttentionCoreSpec, AttentionKernel, attention_core, attention_core_flash};
use boostr::nn::{RmsNorm, RoPE};
use numr::autograd::Var;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

const BATCH: usize = 2;
const HEAD_DIM: usize = 16;
const MAX_SEQ: usize = 32;
/// Both kernels run the same f32 math in a different order (`Masked` adds an
/// `f32::MIN` mask then softmaxes the whole row; `Flash` skips the masked keys
/// outright), so agreement is to float rounding, not bit-exact.
const TOL: f32 = 2e-5;

fn cpu_setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

/// Deterministic pseudo-random values, distinct per index.
fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.017 + seed;
            x.sin() * 0.9 + (x * 2.3).cos() * 0.4
        })
        .collect()
}

fn var(shape: &[usize], seed: f32, device: &CpuDevice) -> Var<CpuRuntime> {
    let n: usize = shape.iter().product();
    Var::new(
        Tensor::<CpuRuntime>::from_slice(&values(n, seed), shape, device),
        false,
    )
}

/// One attention geometry, run through both kernels.
struct Case {
    name: &'static str,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len_q: usize,
    seq_len_k: usize,
    /// `0` disables the window.
    sliding_window: usize,
    qk_norm: bool,
}

/// Run `attention_core` over `case` with the given kernel; returns the flat
/// `[B, S_q, H*D]` output.
fn run(case: &Case, kernel: AttentionKernel, sliding_window: usize) -> Vec<f32> {
    let (client, device) = cpu_setup();

    let q = var(
        &[BATCH, case.seq_len_q, case.num_heads * HEAD_DIM],
        0.3,
        &device,
    );
    let kv_shape = [BATCH, case.seq_len_k, case.num_kv_heads * HEAD_DIM];
    let k = var(&kv_shape, 1.1, &device);
    let v = var(&kv_shape, 2.7, &device);

    let rope = RoPE::<CpuRuntime>::precompute_freqs(MAX_SEQ, HEAD_DIM, 10000.0, None, &device)
        .expect("rope cache builds");

    // Non-unit norm weights: an ignored QK-norm would otherwise be invisible.
    let norm_weight = || {
        RmsNorm::<CpuRuntime>::new(
            Tensor::<CpuRuntime>::from_slice(&values(HEAD_DIM, 5.5), &[HEAD_DIM], &device),
            1e-6,
            false,
        )
    };
    let q_norm = case.qk_norm.then(norm_weight);
    let k_norm = case.qk_norm.then(norm_weight);

    let spec = AttentionCoreSpec {
        num_heads: case.num_heads,
        num_kv_heads: case.num_kv_heads,
        head_dim: HEAD_DIM,
        q_norm: q_norm.as_ref(),
        k_norm: k_norm.as_ref(),
        use_alibi: false,
        sliding_window,
        kernel,
    };

    let out = attention_core(
        &client,
        &q,
        &k,
        &v,
        rope.cos_cache(),
        rope.sin_cache(),
        &spec,
    )
    .unwrap_or_else(|e| panic!("{} / {:?}: attention_core failed: {e}", case.name, kernel));

    let expected = [BATCH, case.seq_len_q, case.num_heads * HEAD_DIM];
    assert_eq!(out.shape(), &expected[..], "{}: output shape", case.name);
    out.tensor()
        .contiguous()
        .expect("output contiguous")
        .to_vec::<f32>()
}

/// Same as [`run`] but calls [`attention_core_flash`] directly instead of
/// going through [`attention_core`]'s kernel selector — `spec.kernel` is set
/// to `Masked` deliberately, since `attention_core_flash` must ignore it.
fn run_flash_entry(case: &Case, sliding_window: usize) -> Vec<f32> {
    let (client, device) = cpu_setup();

    let q = var(
        &[BATCH, case.seq_len_q, case.num_heads * HEAD_DIM],
        0.3,
        &device,
    );
    let kv_shape = [BATCH, case.seq_len_k, case.num_kv_heads * HEAD_DIM];
    let k = var(&kv_shape, 1.1, &device);
    let v = var(&kv_shape, 2.7, &device);

    let rope = RoPE::<CpuRuntime>::precompute_freqs(MAX_SEQ, HEAD_DIM, 10000.0, None, &device)
        .expect("rope cache builds");

    let norm_weight = || {
        RmsNorm::<CpuRuntime>::new(
            Tensor::<CpuRuntime>::from_slice(&values(HEAD_DIM, 5.5), &[HEAD_DIM], &device),
            1e-6,
            false,
        )
    };
    let q_norm = case.qk_norm.then(norm_weight);
    let k_norm = case.qk_norm.then(norm_weight);

    let spec = AttentionCoreSpec {
        num_heads: case.num_heads,
        num_kv_heads: case.num_kv_heads,
        head_dim: HEAD_DIM,
        q_norm: q_norm.as_ref(),
        k_norm: k_norm.as_ref(),
        use_alibi: false,
        sliding_window,
        // Deliberately mismatched: `attention_core_flash` must ignore this
        // and always run the flash kernel, exactly like `attention_core_masked`
        // ignores a `Flash` request.
        kernel: AttentionKernel::Masked,
    };

    let out = attention_core_flash(
        &client,
        &q,
        &k,
        &v,
        rope.cos_cache(),
        rope.sin_cache(),
        &spec,
    )
    .unwrap_or_else(|e| panic!("{}: attention_core_flash failed: {e}", case.name));

    let expected = [BATCH, case.seq_len_q, case.num_heads * HEAD_DIM];
    assert_eq!(out.shape(), &expected[..], "{}: output shape", case.name);
    out.tensor()
        .contiguous()
        .expect("output contiguous")
        .to_vec::<f32>()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "mha, no window, no qk-norm",
            num_heads: 4,
            num_kv_heads: 4,
            seq_len_q: 8,
            seq_len_k: 8,
            sliding_window: 0,
            qk_norm: false,
        },
        Case {
            name: "gqa, no window, no qk-norm",
            num_heads: 4,
            num_kv_heads: 2,
            seq_len_q: 8,
            seq_len_k: 8,
            sliding_window: 0,
            qk_norm: false,
        },
        Case {
            name: "mha, window 5",
            num_heads: 4,
            num_kv_heads: 4,
            seq_len_q: 12,
            seq_len_k: 12,
            sliding_window: 5,
            qk_norm: false,
        },
        Case {
            name: "gqa, window 5",
            num_heads: 4,
            num_kv_heads: 2,
            seq_len_q: 12,
            seq_len_k: 12,
            sliding_window: 5,
            qk_norm: false,
        },
        Case {
            name: "gqa, no window, qk-norm",
            num_heads: 4,
            num_kv_heads: 2,
            seq_len_q: 8,
            seq_len_k: 8,
            sliding_window: 0,
            qk_norm: true,
        },
        Case {
            name: "gqa, window 5, qk-norm",
            num_heads: 4,
            num_kv_heads: 2,
            seq_len_q: 12,
            seq_len_k: 12,
            sliding_window: 5,
            qk_norm: true,
        },
        Case {
            name: "mha, window 4, qk-norm",
            num_heads: 4,
            num_kv_heads: 4,
            seq_len_q: 12,
            seq_len_k: 12,
            sliding_window: 4,
            qk_norm: true,
        },
        // sq != sk: query row `i` sits at absolute position `sk - sq + i`.
        Case {
            name: "chunked gqa (sq=3, sk=11), window 4",
            num_heads: 4,
            num_kv_heads: 2,
            seq_len_q: 3,
            seq_len_k: 11,
            sliding_window: 4,
            qk_norm: false,
        },
        Case {
            name: "chunked gqa (sq=3, sk=11), no window",
            num_heads: 4,
            num_kv_heads: 2,
            seq_len_q: 3,
            seq_len_k: 11,
            sliding_window: 0,
            qk_norm: false,
        },
        Case {
            name: "decode-shaped mha (sq=1, sk=9), window 3, qk-norm",
            num_heads: 4,
            num_kv_heads: 4,
            seq_len_q: 1,
            seq_len_k: 9,
            sliding_window: 3,
            qk_norm: true,
        },
    ]
}

/// The two kernels agree across GQA/MHA, windowed/unwindowed, QK-norm on/off,
/// and `sq != sk`.
#[test]
fn masked_and_flash_agree() {
    for case in cases() {
        let masked = run(&case, AttentionKernel::Masked, case.sliding_window);
        let flash = run(&case, AttentionKernel::Flash, case.sliding_window);
        let diff = max_abs_diff(&masked, &flash);
        assert!(
            diff <= TOL,
            "{}: Masked vs Flash max|d| = {diff:e} exceeds {TOL:e}",
            case.name
        );
        // A kernel that returned zeros (or ignored V) would agree perfectly.
        let magnitude = masked.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(
            magnitude > 1e-3,
            "{}: output is ~zero ({magnitude:e}); the comparison proves nothing",
            case.name
        );
    }
}

/// `attention_core_flash` must produce EXACTLY the same output as
/// `attention_core` with `kernel: Flash` — it is the same code, reached by a
/// narrower entry point, not a second implementation that could drift.
#[test]
fn flash_entry_point_matches_selector() {
    for case in cases() {
        let via_selector = run(&case, AttentionKernel::Flash, case.sliding_window);
        let via_entry = run_flash_entry(&case, case.sliding_window);
        assert_eq!(
            via_selector, via_entry,
            "{}: attention_core_flash diverged from attention_core(kernel: Flash)",
            case.name
        );
    }
}

/// Every windowed case must have a window narrow enough to mask keys that
/// causality alone would keep. Without this, `sliding_window` could be ignored
/// entirely — or its `0` sentinel inverted — and `masked_and_flash_agree` would
/// still pass on both kernels.
#[test]
fn window_actually_bites() {
    for case in cases().into_iter().filter(|c| c.sliding_window > 0) {
        for kernel in [AttentionKernel::Masked, AttentionKernel::Flash] {
            let windowed = run(&case, kernel, case.sliding_window);
            let unwindowed = run(&case, kernel, 0);
            let diff = max_abs_diff(&windowed, &unwindowed);
            assert!(
                diff > 1e-3,
                "{} / {kernel:?}: window {} changed nothing (max|d| = {diff:e}); \
                 the window is not being applied",
                case.name,
                case.sliding_window
            );
        }
    }
}

/// ALiBi has no home in the flash kernel: it takes no additive bias tensor and
/// writes causality itself. Falling back to `Masked`, or dropping the bias,
/// would produce a model that trains and emits fluent text while computing a
/// different function — so this combination must be rejected outright.
#[test]
fn alibi_with_flash_is_an_error() {
    let (client, device) = cpu_setup();
    let (num_heads, seq_len) = (4usize, 8usize);
    let q = var(&[BATCH, seq_len, num_heads * HEAD_DIM], 0.3, &device);
    let k = var(&[BATCH, seq_len, num_heads * HEAD_DIM], 1.1, &device);
    let v = var(&[BATCH, seq_len, num_heads * HEAD_DIM], 2.7, &device);
    let rope = RoPE::<CpuRuntime>::precompute_freqs(MAX_SEQ, HEAD_DIM, 10000.0, None, &device)
        .expect("rope cache builds");

    let spec = |kernel| AttentionCoreSpec::<CpuRuntime> {
        num_heads,
        num_kv_heads: num_heads,
        head_dim: HEAD_DIM,
        q_norm: None,
        k_norm: None,
        use_alibi: true,
        sliding_window: 0,
        kernel,
    };

    let result = attention_core(
        &client,
        &q,
        &k,
        &v,
        rope.cos_cache(),
        rope.sin_cache(),
        &spec(AttentionKernel::Flash),
    );
    let text = match result {
        Ok(_) => panic!("ALiBi + Flash returned a result; it must be rejected outright"),
        Err(e) => e.to_string(),
    };
    assert!(
        text.contains("alibi") || text.contains("ALiBi") || text.contains("use_alibi"),
        "error must name the ALiBi/Flash combination, got: {text}"
    );

    // The same spec on `Masked` is the supported combination and must work.
    attention_core(
        &client,
        &q,
        &k,
        &v,
        rope.cos_cache(),
        rope.sin_cache(),
        &spec(AttentionKernel::Masked),
    )
    .expect("ALiBi + Masked is the supported combination");
}
