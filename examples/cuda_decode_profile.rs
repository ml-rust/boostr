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

    // F16 is what real serving uses. Sweeping both dtypes, plus the window_size
    // toggle below, shows which kernel each configuration actually reaches.
    let dtypes: &[DType] = &[DType::F32, DType::F16];

    for &dtype in dtypes {
        for &num_kv_heads in &[32usize, 8] {
            for &seq_len_k in &[512usize, 4096, 16384] {
                let q = client.rand(&[1, num_heads, 1, head_dim], dtype).unwrap();
                let k = client
                    .rand(&[1, num_kv_heads, seq_len_k, head_dim], dtype)
                    .unwrap();
                let v = client
                    .rand(&[1, num_kv_heads, seq_len_k, head_dim], dtype)
                    .unwrap();

                // `window_size == 0` selects the decode kernel; a window
                // covering the whole cache is the same computation but routes
                // to the general tiled kernel — what a dtype without a decode
                // instantiation falls through to. Measuring both is the point.
                for &window_size in &[0usize, seq_len_k] {
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
                                window_size,
                                Some(seq_len_k),
                            )
                            .unwrap();
                        std::hint::black_box(&out);
                    }
                }
                client.synchronize();
            }
        }
    }
}
