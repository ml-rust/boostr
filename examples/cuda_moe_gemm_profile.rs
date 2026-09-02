//! Profiling target for the MoE grouped GEMM.
//!
//! The launcher sizes `grid.y` from the TOTAL permuted token count and `grid.z`
//! from the expert count, because the per-expert counts live in device memory
//! and reading them back would stall. Every expert therefore launches enough
//! row tiles for every token, though it owns only its own slice.
//!
//! Shapes are Mixtral-class: an even token split across experts, so the useful
//! fraction of the grid is exactly `1 / num_experts`.
//!
//! ```text
//! cargo build --release --features cuda,f16 --example cuda_moe_gemm_profile
//! ncu --kernel-name regex:moe_grouped_gemm --launch-count 8 \
//!     --section SpeedOfLight ./target/release/examples/cuda_moe_gemm_profile
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example needs --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use boostr::ops::MoEOps;
    use boostr::{CudaDevice, CudaRuntime, DType, Runtime, RuntimeClient};
    use numr::ops::{MatmulOps, RandomOps};
    use numr::tensor::Tensor;

    /// Enough launches for a profiler to sample each shape.
    const ITERS: usize = 4;

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);

    // (num_experts, total permuted rows, in_dim, out_dim)
    let cases = [
        (8usize, 2048usize, 4096usize, 4096usize),
        (32usize, 2048usize, 1024usize, 1024usize),
    ];

    for &(num_experts, total_tokens, in_dim, out_dim) in &cases {
        let tokens = client.rand(&[total_tokens, in_dim], DType::F32).unwrap();
        let weights = client
            .rand(&[num_experts, in_dim, out_dim], DType::F32)
            .unwrap();

        // Even split, so every expert owns the same slice and the useful
        // fraction of the launched grid is exactly 1 / num_experts.
        let per_expert = total_tokens / num_experts;
        let offsets: Vec<i32> = (0..=num_experts).map(|e| (e * per_expert) as i32).collect();
        let expert_offsets = Tensor::from_slice(&offsets, &[num_experts + 1], &device).unwrap();

        for _ in 0..ITERS {
            let out = client
                .moe_grouped_gemm(&tokens, &weights, &expert_offsets)
                .unwrap();
            std::hint::black_box(&out);
        }
        client.synchronize();

        // Reference ceiling: one dense matmul of the same total FLOPs, through
        // numr's tuned kernel. An even split means the experts together do
        // exactly `total_tokens x in_dim x out_dim`, so the two are comparable
        // and the gap is the grouped kernel's own quality, not its shape.
        let dense_b = client.rand(&[in_dim, out_dim], DType::F32).unwrap();
        for _ in 0..ITERS {
            let out = client.matmul(&tokens, &dense_b).unwrap();
            std::hint::black_box(&out);
        }
        client.synchronize();
    }
}
