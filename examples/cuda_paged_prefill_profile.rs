//! Profiling target for paged PREFILL attention forward (`seq_len_q > 1`).
//!
//! `seq_len_q == 1` routes `paged_attention_fwd` to the specialized decode
//! kernel (see `cuda_paged_decode_profile.rs`) — this binary always uses
//! `seq_len_q > 1` so every launch hits the tiled prefill kernel instead.
//!
//! The prefill kernel picks its tile in two stages: a capability gate (the
//! large `BLOCK_M=128, BLOCK_N=64` tile must fit this device's opt-in shared
//! memory at all), then a measured performance policy — large only for
//! F16/BF16, and only where grid coverage (`head_dim=128`) or free grid
//! width (`head_dim=64`) says its halved K-loop trip count is repaid; see
//! `fwd_prefer_large` in `src/ops/cuda/attention/paged_attention_fwd_block_config.rs`.
//! Otherwise it falls back to the smaller `_small` tile. Set
//! `BOOSTR_PAGED_PREFILL_TILE=large` or `=small` before running this binary
//! to force one side of that choice for A/B measurement on the same device —
//! `large` is still refused (falls back to `small`, with a stderr note) if
//! the device cannot fit it. Leave it unset (or `auto`) for the normal
//! policy-driven selection.
//!
//! ```text
//! cargo build --release --features cuda --example cuda_paged_prefill_profile
//! BOOSTR_PAGED_PREFILL_TILE=large \
//!   ncu --kernel-name regex:paged_flash_attention_fwd --launch-count 12 \
//!       --section SpeedOfLight --section Occupancy \
//!       ./target/release/examples/cuda_paged_prefill_profile
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example needs --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use boostr::ops::PagedAttentionOps;
    use boostr::{CudaDevice, CudaRuntime, DType, Runtime, RuntimeClient};
    use numr::ops::RandomOps;
    use numr::tensor::Tensor;

    /// Enough launches for a profiler to sample each shape.
    const ITERS: usize = 4;

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);

    let num_heads = 32usize;
    let num_kv_heads = 32usize;
    let block_size = 16usize;

    for &head_dim in &[64usize, 128] {
        for &dtype in &[DType::F32, DType::F16, DType::BF16] {
            for &seq_len_q in &[8usize, 32, 128, 512, 2048] {
                for &seq_len_k in &[512usize, 2048, 8192] {
                    let pages = seq_len_k.div_ceil(block_size);
                    let q = client
                        .rand(&[1, num_heads, seq_len_q, head_dim], dtype)
                        .unwrap();
                    let k = client
                        .rand(&[pages, block_size, num_kv_heads, head_dim], dtype)
                        .unwrap();
                    let v = client
                        .rand(&[pages, block_size, num_kv_heads, head_dim], dtype)
                        .unwrap();
                    // Reverse-order pages, so the walk is as scattered as a real
                    // allocator would leave it rather than sequential.
                    let bt_data: Vec<i32> = (0..pages).map(|i| (pages - 1 - i) as i32).collect();
                    let block_table = Tensor::from_slice(&bt_data, &[1, pages], &device).unwrap();

                    for _ in 0..ITERS {
                        let (out, lse) = client
                            .paged_attention_fwd(
                                &q,
                                &k,
                                &v,
                                &block_table,
                                num_heads,
                                num_kv_heads,
                                seq_len_q,
                                seq_len_k,
                                head_dim,
                                block_size,
                                true,
                            )
                            .unwrap();
                        std::hint::black_box(&out);
                        std::hint::black_box(&lse);
                    }
                    client.synchronize();
                }
            }
        }
    }
}
