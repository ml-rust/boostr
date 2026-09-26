//! Profiling target for decode attention over an OVER-ALLOCATED KV cache.
//!
//! A served cache is preallocated to a budget and filled as the sequence
//! grows, so the keys the kernel reads are a prefix of a longer buffer. The
//! decode kernel takes that buffer's dim-2 extent as a stride and the key
//! count separately, so it reads the prefix alone — the same work whatever
//! the buffer's length.
//!
//! Same work need not be same time: the stride sets the distance between one
//! (batch, head) slice and the next, which is what decides how those reads
//! land in L2. This binary holds the key count fixed and sweeps the allocated
//! length around it, so a profiler reports the cost of the stride by itself
//! and a caller can tell whether over-allocating a cache costs it anything.
//!
//! ```text
//! cargo build --release --features cuda --example cuda_kv_stride_profile
//! ncu --kernel-name regex:decode --launch-count 64 \
//!     --metrics gpu__time_duration.sum \
//!     ./target/release/examples/cuda_kv_stride_profile
//! ```
//!
//! Read it by grouping the launches by shape: within one key count, any
//! spread across the strides is the stride's own cost.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example needs --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use boostr::ops::FlashAttentionOps;
    use boostr::ops::{AttnOutLayout, KvSeqLen};
    use boostr::{CudaDevice, CudaRuntime, DType, Runtime, RuntimeClient};
    use numr::ops::RandomOps;

    /// Enough launches for a profiler to sample each shape.
    const ITERS: usize = 4;

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);

    // A speech decoder's stack rather than a text LLM's: few heads, small
    // head_dim, and a sequence of a few hundred positions against a cache
    // budget sized for the longest request the slot may ever admit.
    let num_heads = 16usize;
    let num_kv_heads = 2usize;
    let head_dim = 64usize;
    let dtype = DType::F32;

    // Keys the kernel actually reads.
    let key_counts = [320usize, 640];
    // Positions the buffer is allocated for. The first entry of each pair is
    // the exact fit, which is the no-stride baseline.
    let capacities = [320usize, 512, 1024, 2048];

    for &keys in &key_counts {
        for &capacity in &capacities {
            if capacity < keys {
                continue;
            }
            let q = client.rand(&[1, num_heads, 1, head_dim], dtype).unwrap();
            let k = client
                .rand(&[1, num_kv_heads, capacity, head_dim], dtype)
                .unwrap();
            let v = client
                .rand(&[1, num_kv_heads, capacity, head_dim], dtype)
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
                        Some(KvSeqLen::Shared(keys)),
                        None,
                        AttnOutLayout::TokenMajor,
                    )
                    .unwrap();
                std::hint::black_box(&out);
            }
            client.synchronize();
        }
    }
}
