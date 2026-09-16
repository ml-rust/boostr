//! Profiling target behind the short-query routing in
//! `FlashAttentionOps::flash_attention_fwd` for CUDA
//! (`src/ops/cuda/attention/flash/impl_ops.rs`, `short_query_fold`).
//!
//! Non-causal attention over a short query sequence (VoxCPM2's bidirectional
//! DiT: `[2, 16, 11, 128]` against 2 KV heads) can run two ways:
//! 1. `mqa_gqa_fwd`: one thread per query row, K/V staged through shared
//!    memory. A block holds `block_m` rows, so a short sequence leaves most of
//!    each block idle and every active thread walks its keys alone.
//! 2. The decode kernel with `S_q` folded into the head axis: `[B, H, S_q, D]`
//!    is the same bytes as `[B, H * S_q, 1, D]`, and because `H_kv` divides
//!    `H`, folded head `h * S_q + s` maps to KV head `h / (H / H_kv)` exactly
//!    as the unfolded head did. One block per query row, one warp per key,
//!    lanes split `D`.
//!
//! This sweeps `S_q = S_k` through both entries and prints host wall time
//! per call, bracketed by a device sync, so the fold's upper bound on `S_q`
//! (`SHORT_QUERY_FOLD_MAX`) is a measurement, not a guess. Run it under nsys
//! for per-kernel device time.
//!
//! ```text
//! cargo build --release --features cuda --example cuda_short_query_profile
//! nsys profile -t cuda -o short_query -f true \
//!     ./target/release/examples/cuda_short_query_profile
//! nsys stats -r cuda_gpu_kern_sum short_query.nsys-rep
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example needs --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use boostr::ops::FlashAttentionOps;
    use boostr::ops::cuda::attention::mqa_gqa::mqa_gqa_fwd;
    use boostr::{CudaDevice, CudaRuntime, DType, Runtime, RuntimeClient};
    use numr::ops::RandomOps;

    const ITERS: usize = 10;

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);

    let batch = 2usize;
    let num_heads = 16usize;
    let num_kv_heads = 2usize;
    let head_dim = 128usize;

    for &dtype in &[DType::F32, DType::BF16] {
        for &seq_len in &[5usize, 11, 16, 32, 64, 128, 256, 512, 1024, 2048] {
            let q = client
                .rand(&[batch, num_heads, seq_len, head_dim], dtype)
                .unwrap();
            let k = client
                .rand(&[batch, num_kv_heads, seq_len, head_dim], dtype)
                .unwrap();
            let v = client
                .rand(&[batch, num_kv_heads, seq_len, head_dim], dtype)
                .unwrap();

            // Side 1: dedicated MQA/GQA kernel, non-causal.
            let start = std::time::Instant::now();
            for _ in 0..ITERS {
                let out = mqa_gqa_fwd(
                    &client,
                    &q,
                    &k,
                    &v,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    false,
                )
                .unwrap();
                std::hint::black_box(&out);
            }
            client.synchronize();
            let dedicated = start.elapsed();

            // Side 2: decode kernel over the folded view. `S_q == 1` is what
            // routes `flash_attention_fwd` to the decode kernel.
            let q_folded = q
                .reshape(&[batch, num_heads * seq_len, 1, head_dim])
                .unwrap();
            let start = std::time::Instant::now();
            for _ in 0..ITERS {
                let out = client
                    .flash_attention_fwd(
                        &q_folded,
                        &k,
                        &v,
                        num_heads * seq_len,
                        num_kv_heads,
                        head_dim,
                        false,
                        0,
                        None,
                    )
                    .unwrap();
                std::hint::black_box(&out);
            }
            client.synchronize();
            let folded = start.elapsed();

            println!(
                "{dtype:?} S={seq_len:<4} mqa_gqa {:>8.1} us/iter   folded-decode {:>8.1} us/iter",
                dedicated.as_secs_f64() * 1e6 / ITERS as f64,
                folded.as_secs_f64() * 1e6 / ITERS as f64,
            );
        }
    }
}
