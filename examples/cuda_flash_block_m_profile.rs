//! Profiling target for `seq_len_q`-aware `BLOCK_M` selection.
//!
//! Two sweeps, each isolating one side of the change:
//!
//! - **Control** (`head_dim = 96`): routes to the general tiled kernel
//!   (`flash_fwd::flash_attention_fwd_impl`, symbol `flash_attention_fwd_96`)
//!   via `flash_utils::block_config`, which still picks `BLOCK_M` from
//!   `(head_dim, device shared-memory limit)` only — no `seq_len_q` input.
//!   `head_dim = 96` is outside the MQA/GQA gate's `{32, 64, 128}` head_dim
//!   set (`mqa_gqa::should_use_mqa_gqa`), so it cannot reach the kernel the
//!   change touches; its numbers are expected to stay flat across the
//!   change and exist here as a negative control.
//! - **MQA/GQA** (`head_dim ∈ {64, 128}`): routes to the dedicated kernel
//!   (`mqa_gqa::mqa_gqa_fwd`, symbol `mqa_gqa_fwd_{head_dim}_{dtype}[_sm]`)
//!   via `mqa_gqa::block_config::mqa_fwd_block_config`, which the change
//!   made `seq_len_q`-aware: it now picks the `_sm` (small-tile) kernel
//!   variant when `seq_len_q` is small enough that the large tile would
//!   waste masked-off rows. `num_heads = 32`, `num_kv_heads = 8` is
//!   divisible, so `should_use_mqa_gqa` admits both head_dims regardless of
//!   ratio (see that function's doc — the ratio floor was removed after
//!   measurement found no crossover).
//!
//! Prediction for the MQA/GQA sweep: small `seq_len_q` shapes now launch the
//! `_sm` kernel symbol and cost less than they did under the old
//! always-large selection; large `seq_len_q` shapes still launch the
//! non-`_sm` symbol and are unchanged. The control sweep should show no
//! symbol change and no cost change at any `seq_len_q`.
//!
//! `seq_len_q == 1` is excluded from both sweeps: `flash.rs`'s
//! `flash_attention_fwd` routes `seq_len_q == 1` to the single-token decode
//! kernel (`flash_decode::decode_attention_fwd`) for `head_dim` 64/128,
//! which is a different kernel than either sweep measures. Both sweeps start
//! at 2 so every shape is unambiguously prefill.
//!
//! ## Why `head_dim = 96` reaches the general kernel, not a dedicated one
//!
//! `flash_attention_fwd` (`src/ops/cuda/attention/flash.rs`) has three
//! dispatch gates ahead of the general kernel:
//!
//! - Decode: only `head_dim` 64/128 and `seq_len_q == 1`. Avoided above.
//! - Flash-v3 (Hopper): only when `num_kv_heads == num_heads`.
//! - MQA/GQA dedicated kernels (`mqa_gqa::should_use_mqa_gqa`): admits ANY
//!   `head_dim ∈ {32, 64, 128}` with `num_heads` divisible by `num_kv_heads`
//!   — the ratio no longer matters, only capability
//!   (`src/ops/cuda/attention/mqa_gqa/block_config.rs`).
//!
//! `head_dim = 96` is outside `{32, 64, 128}` and so fails the MQA/GQA
//! `matches!` check unconditionally — no dependence on ratio, dtype, or GPU.
//! Pairing it with `num_kv_heads != num_heads` (32 query heads, 8 KV heads)
//! also statically fails the Flash-v3 gate's `num_kv_heads == num_heads`
//! check. Both exclusions hold from reading `flash.rs` alone, with no
//! dependence on which GPU runs this. `block_config_large`/
//! `block_config_small` both define a `head_dim = 96` block (`(64, 128)` or
//! `(32, 32)` — device shared-memory dependent), so `BLOCK_M` here is 64 or
//! 32, not the 128 that head_dim 64/128 would use; the sweep's {8, 16, 32,
//! 64} steps cross both candidate boundaries. Separately, on every device
//! available for this measurement, `head_dim = 96`'s large config needs more
//! shared memory than the device offers, so this path already runs the small
//! config unconditionally — another reason it cannot show the `seq_len_q`
//! rule's effect, on top of never reaching the kernel that rule changed.
//!
//! ```text
//! cargo build --release --features cuda --example cuda_flash_block_m_profile
//! ncu --kernel-name regex:'flash_attention_fwd_96|mqa_gqa_fwd_(64|128)_fp32' \
//!     --launch-count 144 \
//!     --section SpeedOfLight --section Occupancy \
//!     ./target/release/examples/cuda_flash_block_m_profile
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

    /// Enough launches per shape for a profiler to sample the kernel.
    const ITERS: usize = 4;

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);

    let num_heads = 32usize;
    let num_kv_heads = 8usize;
    let seq_len_k = 4096usize;
    let dtype = DType::F32;
    let causal = true;

    // Starts at 2 (not 1) to stay off the decode path; see module doc.
    let seq_len_q_sweep = [2usize, 4, 8, 16, 32, 64, 128, 256, 1024];

    // Control: head_dim = 96 never reaches the kernel the seq_len_q rule
    // changed (see module doc), so its numbers are expected to stay flat
    // across the change. Kept unchanged as the negative-control baseline.
    // Crosses both candidate BLOCK_M values (32 and 64) for head_dim=96 from
    // far under to well over.
    {
        let head_dim = 96usize;

        for &batch in &[1usize, 8] {
            for &seq_len_q in &seq_len_q_sweep {
                let q = client
                    .rand(&[batch, num_heads, seq_len_q, head_dim], dtype)
                    .unwrap();
                let k = client
                    .rand(&[batch, num_kv_heads, seq_len_k, head_dim], dtype)
                    .unwrap();
                let v = client
                    .rand(&[batch, num_kv_heads, seq_len_k, head_dim], dtype)
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
                            causal,
                            0,
                            None,
                        )
                        .unwrap();
                    std::hint::black_box(&out);
                }

                client.synchronize();
            }
        }
    }

    // MQA/GQA: head_dim 64 and 128 both pass should_use_mqa_gqa (divisible
    // head counts), so this sweep reaches mqa_gqa_fwd, where the new
    // seq_len_q-aware selection picks the _sm kernel variant for small
    // seq_len_q instead of always launching the large tile.
    for &head_dim in &[64usize, 128] {
        for &batch in &[1usize, 8] {
            for &seq_len_q in &seq_len_q_sweep {
                let q = client
                    .rand(&[batch, num_heads, seq_len_q, head_dim], dtype)
                    .unwrap();
                let k = client
                    .rand(&[batch, num_kv_heads, seq_len_k, head_dim], dtype)
                    .unwrap();
                let v = client
                    .rand(&[batch, num_kv_heads, seq_len_k, head_dim], dtype)
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
                            causal,
                            0,
                            None,
                        )
                        .unwrap();
                    std::hint::black_box(&out);
                }

                client.synchronize();
            }
        }
    }
}
