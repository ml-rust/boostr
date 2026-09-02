//! Profiling target for paged PREFILL attention BACKWARD (`seq_len_q > 1`).
//!
//! `seq_len_q == 1` routes `paged_attention_fwd` to the specialized decode
//! kernel (see `cuda_paged_decode_profile.rs`) — this binary always uses
//! `seq_len_q > 1`, mirroring `cuda_paged_prefill_profile.rs`, so every
//! launch hits the tiled prefill kernel on the forward pass that feeds the
//! backward call, and the backward call itself stays in the prefill regime.
//!
//! Unlike the forward prefill tile, the backward large tile
//! (`BLOCK_M=128, BLOCK_N=64`) has never been measured on real hardware, so
//! `bwd_block_config` is SMALL-ONLY by default (see
//! `src/ops/cuda/attention/paged_attention_bwd_block_config.rs`). Set
//! `BOOSTR_PAGED_BWD_TILE=large` before running this binary to force the
//! large tile for A/B measurement on the same device — it is still refused
//! (falls back to `small`, with a stderr note) wherever the device cannot
//! fit it in opt-in shared memory; on the measurement device that means only
//! `head_dim=64` at F16/BF16 fits, every other combination falls back. Set
//! `BOOSTR_PAGED_BWD_TILE=small` (or leave it unset) for the small tile.
//!
//! ```text
//! cargo build --release --features cuda --example cuda_paged_bwd_profile
//! BOOSTR_PAGED_BWD_TILE=large \
//!   ncu --kernel-name regex:paged_flash_attention_bwd --launch-count 12 \
//!       --section SpeedOfLight --section Occupancy \
//!       ./target/release/examples/cuda_paged_bwd_profile
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
            // Backward recomputes the score tile and accumulates dK/dV, so it
            // costs several times the forward at the same shape. The sweep
            // stops below the forward example's top end to stay tractable
            // under a serializing profiler.
            for &seq_len_q in &[8usize, 32, 128, 512] {
                for &seq_len_k in &[512usize, 2048] {
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

                    // Real forward output/lse feed the backward call, so the
                    // measurement reflects a genuine fwd->bwd pair rather than
                    // random stand-ins.
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

                    // Incoming gradient, same shape as `out`/`q`: [B, H, S_q, D].
                    let dout = client
                        .rand(&[1, num_heads, seq_len_q, head_dim], dtype)
                        .unwrap();

                    for _ in 0..ITERS {
                        let (dq, dk_blocks, dv_blocks) = client
                            .paged_attention_bwd(
                                &dout,
                                &q,
                                &k,
                                &v,
                                &out,
                                &lse,
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
                        std::hint::black_box(&dq);
                        std::hint::black_box(&dk_blocks);
                        std::hint::black_box(&dv_blocks);
                    }
                    client.synchronize();
                }
            }
        }
    }
}
