//! Profiling target for paged single-token decode attention.
//!
//! The paged decode kernel walks the KV cache through a block table. Its grid
//! is one block per `(batch, Q head)` pair, which does not grow with
//! `seq_len_k` — the axis that grows every decode step. This binary measures
//! that: a profiler reports kernel time directly, where a host-side benchmark
//! would mix in allocation and launch overhead.
//!
//! Shapes are Llama-3-8B-class: 32 query heads, head_dim 128, page size 16,
//! swept over KV length.
//!
//! ```text
//! cargo build --release --features cuda --example cuda_paged_decode_profile
//! ncu --kernel-name regex:paged_decode --launch-count 12 \
//!     --section SpeedOfLight --section Occupancy \
//!     ./target/release/examples/cuda_paged_decode_profile
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
    let head_dim = 128usize;
    let block_size = 16usize;

    for &num_kv_heads in &[32usize, 8] {
        for &seq_len_k in &[512usize, 4096, 16384] {
            let pages = seq_len_k.div_ceil(block_size);
            let q = client
                .rand(&[1, num_heads, 1, head_dim], DType::F32)
                .unwrap();
            let k = client
                .rand(&[pages, block_size, num_kv_heads, head_dim], DType::F32)
                .unwrap();
            let v = client
                .rand(&[pages, block_size, num_kv_heads, head_dim], DType::F32)
                .unwrap();
            // Reverse-order pages, so the walk is as scattered as a real allocator
            // would leave it rather than sequential.
            let bt_data: Vec<i32> = (0..pages).map(|i| (pages - 1 - i) as i32).collect();
            let block_table = Tensor::from_slice(&bt_data, &[1, pages], &device).unwrap();

            for _ in 0..ITERS {
                let out = client
                    .paged_attention_fwd(
                        &q,
                        &k,
                        &v,
                        &block_table,
                        num_heads,
                        num_kv_heads,
                        1,
                        seq_len_k,
                        head_dim,
                        block_size,
                        false,
                    )
                    .unwrap();
                std::hint::black_box(&out);
            }
            client.synchronize();
        }
    }
}
