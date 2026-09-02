//! Numerical parity between the paged attention PREFILL forward large tile
//! (`BLOCK_M=128, BLOCK_N=64`, the unsuffixed `paged_flash_attention_fwd_*`
//! kernels) and the small tile (`_small` kernels), forced explicitly through
//! `paged_attention_fwd_with_tile_for_test` in
//! `src/ops/cuda/attention/paged_attention_fwd.rs`.
//!
//! Before this file, the large tile had never been executed by this library
//! on any device: the kernel name was hardcoded to `_small`, so no test ever
//! covered a large-tile launch. The tile-selection policy in
//! `src/ops/cuda/attention/paged_attention_fwd_block_config.rs` (`fwd_prefer_large`) now
//! selects large only for F16/BF16, and only once the mechanism it is built
//! on — grid coverage at `head_dim=128`, free grid width at `head_dim=64` —
//! says the large tile's halved K-loop trip count repays its per-row cost.
//! This file does not exercise that selection logic (that is
//! `paged_attention.rs`'s own `#[cfg(test)]` unit tests, which need no
//! device); it proves the two kernels agree numerically, which the selection
//! logic depends on being true regardless of which side it picks.
//!
//! Tile forcing goes through `paged_attention_fwd_with_tile_for_test`, an
//! explicit-parameter entry point, NOT `BOOSTR_PAGED_PREFILL_TILE`: Rust test
//! binaries run every `#[test]` function on its own thread within one
//! process, so setting that process-wide env var from a test would race with
//! every other test reading or setting it. The explicit parameter has no
//! such race.
//!
//! `seq_len_q` values cover both sides of the tile-selection rule's flip
//! point without relying on it (this file forces each side directly): 32 and
//! 1024 span the head_dim=128 grid-coverage rule, 64 and 128 span the
//! head_dim=64 free-grid-width rule.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test paged_prefill_tile_parity

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::ops::cuda::attention::paged_attention_fwd::paged_attention_fwd_with_tile_for_test;
use boostr::ops::cuda::attention::paged_attention_fwd_block_config::fwd_prefill_tile_for_test;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::Runtime;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

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

fn cuda_setup() -> (CudaClient, CudaDevice) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    (client, device)
}

const BATCH: usize = 1;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 4;
const BLOCK_SIZE: usize = 16;

/// Deterministic pseudo-random values, distinct per index and per seed.
fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.017 + seed;
            x.sin() * 0.9 + (x * 2.3).cos() * 0.4
        })
        .collect()
}

fn assert_close(a: &[f32], b: &[f32], label: &str, tol: f32) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let threshold = tol + tol * y.abs();
        assert!(
            diff <= threshold,
            "{label} at index {i}: {x} vs {y} (diff={diff}, tol={threshold})"
        );
    }
}

#[test]
fn f16_prefill_small_and_large_tile_agree() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping f16_prefill_small_and_large_tile_agree");
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();

    for &head_dim in &[64usize, 128] {
        for &seq_len_q in &[32usize, 64, 128, 1024] {
            let seq_len_k = seq_len_q;
            let pages = seq_len_k.div_ceil(BLOCK_SIZE);

            let q_f32 = values(BATCH * NUM_HEADS * seq_len_q * head_dim, 0.3);
            let k_f32 = values(pages * BLOCK_SIZE * NUM_KV_HEADS * head_dim, 1.1);
            let v_f32 = values(pages * BLOCK_SIZE * NUM_KV_HEADS * head_dim, 2.7);

            let q = Tensor::<CudaRuntime>::from_slice(
                &q_f32,
                &[BATCH, NUM_HEADS, seq_len_q, head_dim],
                &device,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .expect("cast Q fixture to F16");
            let k = Tensor::<CudaRuntime>::from_slice(
                &k_f32,
                &[pages, BLOCK_SIZE, NUM_KV_HEADS, head_dim],
                &device,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .expect("cast K fixture to F16");
            let v = Tensor::<CudaRuntime>::from_slice(
                &v_f32,
                &[pages, BLOCK_SIZE, NUM_KV_HEADS, head_dim],
                &device,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .expect("cast V fixture to F16");
            // Reverse-order pages: the block-table indirection must be
            // exercised identically by both tiles, not just a sequential walk.
            let bt_data: Vec<i32> = (0..pages).map(|i| (pages - 1 - i) as i32).collect();
            let block_table =
                Tensor::<CudaRuntime>::from_slice(&bt_data, &[BATCH, pages], &device).unwrap();

            let (small_out, small_lse) = paged_attention_fwd_with_tile_for_test(
                &client,
                &q,
                &k,
                &v,
                &block_table,
                NUM_HEADS,
                NUM_KV_HEADS,
                seq_len_q,
                seq_len_k,
                head_dim,
                BLOCK_SIZE,
                true,
                false,
            )
            .expect("f16 small-tile prefill forward failed");
            let (large_out, large_lse) = paged_attention_fwd_with_tile_for_test(
                &client,
                &q,
                &k,
                &v,
                &block_table,
                NUM_HEADS,
                NUM_KV_HEADS,
                seq_len_q,
                seq_len_k,
                head_dim,
                BLOCK_SIZE,
                true,
                true,
            )
            .expect("f16 large-tile prefill forward failed");

            let small_out_f32 = small_out
                .to_dtype(DType::F32)
                .expect("cast small-tile F16 output back to F32 for comparison")
                .to_vec::<f32>();
            let large_out_f32 = large_out
                .to_dtype(DType::F32)
                .expect("cast large-tile F16 output back to F32 for comparison")
                .to_vec::<f32>();

            // Both tiles read the SAME already-F16-rounded Q/K/V and run the
            // SAME kernel formula on the SAME device — they differ only in
            // BLOCK_N (32 vs 64), i.e. only in the K-loop chunking of the
            // online-softmax reduction. That reduction runs in fp32
            // internally, so the only place F16 rounding can amplify a
            // difference is the final store of the output — one ULP of F16
            // (eps 9.8e-4) at most, not the input-quantization-dominated
            // error a cross-backend (CPU-vs-CUDA) comparison has to tolerate
            // (2e-3 atol / 1e-2 rtol in `tests/backend_parity/mqa_gqa_attention.rs`).
            // This tolerance is set tighter than that cross-backend bound,
            // not looser.
            assert_close(
                &large_out_f32,
                &small_out_f32,
                &format!("f16 prefill output head_dim={head_dim} seq_len_q={seq_len_q}"),
                2e-3,
            );
            // LSE is F32 end-to-end (never rounded to a storage dtype), so
            // its only source of divergence is fp32 reduction-order drift —
            // tighter still.
            assert_close(
                &large_lse.to_vec::<f32>(),
                &small_lse.to_vec::<f32>(),
                &format!("f16 prefill lse head_dim={head_dim} seq_len_q={seq_len_q}"),
                1e-4,
            );
        }
    }
}

#[test]
fn bf16_prefill_small_and_large_tile_agree() {
    if !cuda_available() {
        eprintln!("CUDA not available, skipping bf16_prefill_small_and_large_tile_agree");
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();

    for &head_dim in &[64usize, 128] {
        for &seq_len_q in &[32usize, 64, 128, 1024] {
            let seq_len_k = seq_len_q;
            let pages = seq_len_k.div_ceil(BLOCK_SIZE);

            let q_f32 = values(BATCH * NUM_HEADS * seq_len_q * head_dim, 0.4);
            let k_f32 = values(pages * BLOCK_SIZE * NUM_KV_HEADS * head_dim, 1.3);
            let v_f32 = values(pages * BLOCK_SIZE * NUM_KV_HEADS * head_dim, 2.9);

            let q = Tensor::<CudaRuntime>::from_slice(
                &q_f32,
                &[BATCH, NUM_HEADS, seq_len_q, head_dim],
                &device,
            )
            .unwrap()
            .to_dtype(DType::BF16)
            .expect("cast Q fixture to BF16");
            let k = Tensor::<CudaRuntime>::from_slice(
                &k_f32,
                &[pages, BLOCK_SIZE, NUM_KV_HEADS, head_dim],
                &device,
            )
            .unwrap()
            .to_dtype(DType::BF16)
            .expect("cast K fixture to BF16");
            let v = Tensor::<CudaRuntime>::from_slice(
                &v_f32,
                &[pages, BLOCK_SIZE, NUM_KV_HEADS, head_dim],
                &device,
            )
            .unwrap()
            .to_dtype(DType::BF16)
            .expect("cast V fixture to BF16");
            let bt_data: Vec<i32> = (0..pages).map(|i| (pages - 1 - i) as i32).collect();
            let block_table =
                Tensor::<CudaRuntime>::from_slice(&bt_data, &[BATCH, pages], &device).unwrap();

            let (small_out, small_lse) = paged_attention_fwd_with_tile_for_test(
                &client,
                &q,
                &k,
                &v,
                &block_table,
                NUM_HEADS,
                NUM_KV_HEADS,
                seq_len_q,
                seq_len_k,
                head_dim,
                BLOCK_SIZE,
                true,
                false,
            )
            .expect("bf16 small-tile prefill forward failed");
            let (large_out, large_lse) = paged_attention_fwd_with_tile_for_test(
                &client,
                &q,
                &k,
                &v,
                &block_table,
                NUM_HEADS,
                NUM_KV_HEADS,
                seq_len_q,
                seq_len_k,
                head_dim,
                BLOCK_SIZE,
                true,
                true,
            )
            .expect("bf16 large-tile prefill forward failed");

            let small_out_f32 = small_out
                .to_dtype(DType::F32)
                .expect("cast small-tile BF16 output back to F32 for comparison")
                .to_vec::<f32>();
            let large_out_f32 = large_out
                .to_dtype(DType::F32)
                .expect("cast large-tile BF16 output back to F32 for comparison")
                .to_vec::<f32>();

            // Same reasoning as the F16 case above, scaled to BF16's coarser
            // eps (7.8e-3 vs F16's 9.8e-4): both tiles read the same
            // already-BF16-rounded Q/K/V, run the same kernel formula on the
            // same device, and differ only in K-loop chunking of an
            // internally-fp32 reduction, so the only place rounding can
            // amplify a difference is the final BF16 store — at most a few
            // ULPs of BF16. Set tighter than the cross-backend (CPU-vs-CUDA)
            // BF16 bound of 2e-2 atol / 6e-2 rtol in
            // `tests/backend_parity/mqa_gqa_attention.rs`, which additionally
            // has to absorb input-quantization error this comparison does not.
            assert_close(
                &large_out_f32,
                &small_out_f32,
                &format!("bf16 prefill output head_dim={head_dim} seq_len_q={seq_len_q}"),
                1e-2,
            );
            // LSE is F32 end-to-end; only fp32 reduction-order drift applies.
            assert_close(
                &large_lse.to_vec::<f32>(),
                &small_lse.to_vec::<f32>(),
                &format!("bf16 prefill lse head_dim={head_dim} seq_len_q={seq_len_q}"),
                1e-4,
            );
        }
    }
}

/// The capability gate (`fwd_block_config_with_override`, exercised here
/// through the `fwd_prefill_tile_for_test` wrapper) must never return a tile
/// whose shared-memory need exceeds this device's opt-in limit, whether the
/// policy chose it or an override forced it — it must fall back instead of
/// attempting a launch that the device cannot fit. This covers exactly the
/// case a large tile that does not fit at all on the current device: at
/// head_dim=128, F32 needs a wide large-tile shared-memory footprint, and on
/// some devices that already exceeds the opt-in limit before the
/// performance policy (which never picks large for F32 anyway) is relevant.
#[test]
fn capability_gate_never_returns_a_tile_that_does_not_fit() {
    if !cuda_available() {
        eprintln!(
            "CUDA not available, skipping capability_gate_never_returns_a_tile_that_does_not_fit"
        );
        return;
    }
    let _lock = cuda_lock();
    let (_client, device) = cuda_setup();
    let device_index = device.id();

    for &head_dim in &[64usize, 128] {
        for &dtype in &[DType::F32, DType::F16, DType::BF16] {
            for &override_large in &[None, Some(true), Some(false)] {
                let (block_m, block_n, use_large, smem, max_smem) = fwd_prefill_tile_for_test(
                    head_dim,
                    dtype,
                    128,
                    NUM_HEADS,
                    BATCH,
                    device_index,
                    override_large,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "head_dim={head_dim} dtype={dtype:?} override={override_large:?}: \
                         capability-gated config must still resolve to a fitting tile, got \
                         error instead: {e}"
                    )
                });
                assert!(
                    smem <= max_smem,
                    "head_dim={head_dim} dtype={dtype:?} override={override_large:?}: \
                     selected block_m={block_m} block_n={block_n} use_large={use_large} needs \
                     {smem} bytes, exceeding this device's {max_smem} byte opt-in limit"
                );
            }
        }
    }
}
