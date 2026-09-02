//! Profiling target for single-token decode attention.
//!
//! Decode (`seq_len_q == 1`) is the per-token hot path in serving. The decode
//! kernel launches `batch * num_heads` blocks and walks the whole KV sequence
//! serially inside each block, so the grid does not grow with `seq_len_k` — the
//! one axis that grows every decode step. This binary exists to measure that:
//! a profiler reports kernel time directly, where a host-side benchmark would
//! mix in allocation and launch overhead.
//!
//! Shapes are Llama-3-8B-class: 32 query heads, 8 KV heads (GQA ratio 4),
//! head_dim 128, swept over KV length.
//!
//! ```text
//! cargo build --release --features cuda --example cuda_decode_profile
//! ncu --kernel-name regex:decode --launch-count 6 \
//!     --section SpeedOfLight --section Occupancy \
//!     ./target/release/examples/cuda_decode_profile
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example needs --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use boostr::ops::FlashAttentionOps;
    use boostr::{CudaDevice, CudaRuntime, DType, Runtime, RuntimeClient};
    use numr::ops::RandomOps;

    /// Enough launches for a profiler to sample each shape.
    const ITERS: usize = 4;

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);

    let num_heads = 32usize;
    let head_dim = 128usize;

    // The decode fast path is gated to F32 and to num_kv_heads == num_heads,
    // so this configuration is what actually reaches it. The GQA case that real
    // serving uses takes a different kernel; measuring both is the point.
    for &num_kv_heads in &[32usize, 8] {
        for &seq_len_k in &[512usize, 4096, 16384] {
            let q = client
                .rand(&[1, num_heads, 1, head_dim], DType::F32)
                .unwrap();
            let k = client
                .rand(&[1, num_kv_heads, seq_len_k, head_dim], DType::F32)
                .unwrap();
            let v = client
                .rand(&[1, num_kv_heads, seq_len_k, head_dim], DType::F32)
                .unwrap();

            for _ in 0..ITERS {
                let out = client
                    .flash_attention_fwd(
                        &q,
                        &k,
                        &v,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        false,
                        0,
                        Some(seq_len_k),
                    )
                    .unwrap();
                std::hint::black_box(&out);
            }
            client.synchronize();
        }
    }
}
