//! Profiling target that produced the measurement behind the MQA/GQA
//! dedicated-kernel routing gate.
//!
//! `should_use_mqa_gqa` (`src/ops/cuda/attention/mqa_gqa/block_config.rs`)
//! used to also gate on `num_heads / num_kv_heads >= 4`, a performance guess
//! with no measurement backing the `4`. This example runs the dedicated
//! kernel and the general tiled kernel on identical inputs across ratios both
//! above and below that old threshold, so the crossover point could be read
//! off a profile instead of assumed. It found none — the dedicated kernel won
//! a small, flat margin at every ratio — so the ratio condition was removed
//! and `should_use_mqa_gqa` is now a capability-only gate. This example is
//! kept as the reproduction for that measurement and for auditing any future
//! change to the gate.
//!
//! `tests/mqa_gqa_vs_flash_bench.rs` cannot answer this question: it calls
//! `should_use_mqa_gqa` and then ASSERTS the trait dispatch matches it on
//! every case, so it only ever exercises the dedicated kernel at shapes the
//! gate admits and the general kernel at shapes it doesn't — it can never
//! observe the general kernel's time on an admitted shape or the dedicated
//! kernel's time on a rejected one, which is exactly the comparison a
//! threshold audit needs.
//!
//! Two entry points, same shapes:
//! 1. `boostr::ops::cuda::attention::mqa_gqa::mqa_gqa_fwd` — the dedicated
//!    kernel, called directly. It does not consult the threshold, so it runs
//!    at every ratio in the sweep, not just ratio >= 4.
//! 2. `FlashAttentionOps::flash_attention_fwd` with `window_size = seq_len_k`
//!    (instead of `0`). `window_size == 0` is required both by the MQA/GQA
//!    gate and by the Hopper Flash-v3 path in `src/ops/cuda/attention/flash.rs`,
//!    so `window_size == 0` at ratio >= 4 would route back into the SAME
//!    dedicated kernel as side 1, making the comparison meaningless. A window
//!    covering the whole KV sequence is numerically identical to unrestricted
//!    causal attention (nothing gets masked that causal wasn't already
//!    masking) while forcing every ratio through the general tiled kernel in
//!    `flash_fwd::flash_attention_fwd_impl`. That function is `pub(super)` and
//!    not reachable from an example directly; going through the trait method
//!    with the window trick is the only public way to force it. This is the
//!    same technique `cuda_decode_profile.rs` uses to force its comparison off
//!    the decode kernel.
//!
//! ```text
//! cargo build --release --features cuda --example cuda_mqa_gqa_crossover_profile
//! ncu --kernel-name regex:'mqa_gqa_fwd|flash_attention_fwd' --launch-count 48 \
//!     --section SpeedOfLight --section Occupancy \
//!     ./target/release/examples/cuda_mqa_gqa_crossover_profile
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

    /// Enough launches per shape for a profiler to sample both kernels.
    const ITERS: usize = 4;

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);

    let num_heads = 32usize;
    let batch = 1usize;
    let dtype = DType::F32;

    // kv = 32, 16, 8, 4, 2, 1 -> ratio = 1, 2, 4, 8, 16, 32. The old ratio
    // threshold sat at 4; this sweep straddles it on both sides.
    let kv_heads_sweep = [32usize, 16, 8, 4, 2, 1];

    for &head_dim in &[64usize, 128] {
        for &seq_len in &[512usize, 4096] {
            for &num_kv_heads in &kv_heads_sweep {
                let q = client
                    .rand(&[batch, num_heads, seq_len, head_dim], dtype)
                    .unwrap();
                let k = client
                    .rand(&[batch, num_kv_heads, seq_len, head_dim], dtype)
                    .unwrap();
                let v = client
                    .rand(&[batch, num_kv_heads, seq_len, head_dim], dtype)
                    .unwrap();

                for _ in 0..ITERS {
                    // Side 1: dedicated MQA/GQA kernel, unconditionally — it
                    // does not check the ratio itself.
                    let out_dedicated =
                        mqa_gqa_fwd(&client, &q, &k, &v, num_heads, num_kv_heads, head_dim, true)
                            .unwrap();
                    std::hint::black_box(&out_dedicated);
                }

                for _ in 0..ITERS {
                    // Side 2: general tiled kernel, forced past both the
                    // MQA/GQA gate and the Flash-v3 gate by a full-width
                    // window (see module doc for why `window_size = 0`
                    // cannot be used here at ratio >= 4).
                    let out_general = client
                        .flash_attention_fwd(
                            &q,
                            &k,
                            &v,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                            true,
                            seq_len,
                            Some(seq_len),
                        )
                        .unwrap();
                    std::hint::black_box(&out_general);
                }

                client.synchronize();
            }
        }
    }
}
