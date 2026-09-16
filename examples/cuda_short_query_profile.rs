//! Profiling target behind the short-query routing in
//! `FlashAttentionOps::flash_attention_fwd` for CUDA
//! (`src/ops/cuda/attention/flash/impl_ops.rs`, `short_query_fold`), and for
//! the F32 prefill path of `mqa_gqa_fwd`.
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
//! Default mode sweeps `S_q = S_k` through both entries and prints host wall
//! time per call, bracketed by a device sync, so the fold's upper bound on
//! `S_q` (`SHORT_QUERY_FOLD_MAX`) is a measurement, not a guess.
//!
//! `--prefill [--causal]` runs only `mqa_gqa_fwd` over the LM-prefill shapes
//! (H=16, H_kv=2, D=128, B in {1, 2}, S in {64, 256, 1024, 2048}, F32 and
//! BF16) so ncu can report per-kernel device time for the dedicated kernel
//! alone, in both causal settings:
//!
//! ```text
//! cargo build --release --features cuda --example cuda_short_query_profile
//! ncu --kernel-name regex:mqa_gqa_fwd --metrics gpu__time_duration.sum \
//!     --csv ./target/release/examples/cuda_short_query_profile --prefill --causal
//! ```
//!
//! `--flash [--causal] [--window N] [--sq N]` runs `flash_attention_fwd` over
//! the shapes the general `flash_v2.cu` kernel serves (head_dim 96 and 256,
//! which no dedicated kernel takes; H=8 against H_kv in {8, 2}; B=1; S in
//! {64, 256, 1024, 2048}; F32 and BF16). `--window N` adds one sliding-window
//! run at S=2048 after the sweep. `--sq N` caps the query length at N while S
//! stays the key length, for the decode and chunked-prefill shapes. Every
//! launch in this mode is `flash_attention_fwd_{head_dim}[_sm]_{dtype}`:
//!
//! ```text
//! ncu --kernel-name regex:flash_attention_fwd --metrics gpu__time_duration.sum \
//!     --csv ./target/release/examples/cuda_short_query_profile --flash --causal
//! ```
//!
//! Each shape's launches are preceded by a print of the shape, so the ncu
//! rows can be matched to shapes by order.

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

    let args: Vec<String> = std::env::args().skip(1).collect();
    let causal = args.iter().any(|a| a == "--causal");
    let prefill = args.iter().any(|a| a == "--prefill");
    let flash = args.iter().any(|a| a == "--flash");
    let seq_len_q_override = args
        .iter()
        .position(|a| a == "--sq")
        .and_then(|i| args.get(i + 1))
        .map(|w| w.parse::<usize>().expect("--sq takes a query length"));
    let window = args
        .iter()
        .position(|a| a == "--window")
        .and_then(|i| args.get(i + 1))
        .map(|w| w.parse::<usize>().expect("--window takes a key count"))
        .unwrap_or(0);

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);

    let num_heads = 16usize;
    let num_kv_heads = 2usize;
    let head_dim = 128usize;

    if flash {
        let run = |dtype: DType,
                   head_dim: usize,
                   kv_heads: usize,
                   seq_len: usize,
                   window: usize| {
            let heads = 8usize;
            let seq_len_q = seq_len_q_override.unwrap_or(seq_len).min(seq_len);
            let q = client
                .rand(&[1, heads, seq_len_q, head_dim], dtype)
                .unwrap();
            let k = client
                .rand(&[1, kv_heads, seq_len, head_dim], dtype)
                .unwrap();
            let v = client
                .rand(&[1, kv_heads, seq_len, head_dim], dtype)
                .unwrap();
            client.synchronize();
            println!(
                "shape {dtype:?} D={head_dim} Hkv={kv_heads} Sq={seq_len_q} S={seq_len} causal={causal} window={window}"
            );
            let start = std::time::Instant::now();
            for _ in 0..ITERS {
                let out = client
                    .flash_attention_fwd(
                        &q, &k, &v, heads, kv_heads, head_dim, causal, window, None,
                    )
                    .unwrap();
                std::hint::black_box(&out);
            }
            client.synchronize();
            println!(
                "  host {:>8.1} us/iter",
                start.elapsed().as_secs_f64() * 1e6 / ITERS as f64
            );
        };
        for &dtype in &[DType::F32, DType::BF16] {
            for &head_dim in &[96usize, 256] {
                for &kv_heads in &[8usize, 2] {
                    for &seq_len in &[64usize, 256, 1024, 2048] {
                        run(dtype, head_dim, kv_heads, seq_len, 0);
                    }
                }
            }
        }
        if window > 0 {
            for &dtype in &[DType::F32, DType::BF16] {
                for &head_dim in &[96usize, 256] {
                    run(dtype, head_dim, 8, 2048, window);
                }
            }
        }
        return;
    }

    if prefill {
        for &dtype in &[DType::F32, DType::BF16] {
            for &batch in &[1usize, 2] {
                for &seq_len in &[64usize, 256, 1024, 2048] {
                    let q = client
                        .rand(&[batch, num_heads, seq_len, head_dim], dtype)
                        .unwrap();
                    let k = client
                        .rand(&[batch, num_kv_heads, seq_len, head_dim], dtype)
                        .unwrap();
                    let v = client
                        .rand(&[batch, num_kv_heads, seq_len, head_dim], dtype)
                        .unwrap();
                    client.synchronize();
                    println!("shape {dtype:?} B={batch} S={seq_len} causal={causal}");
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
                            causal,
                        )
                        .unwrap();
                        std::hint::black_box(&out);
                    }
                    client.synchronize();
                    println!(
                        "  host {:>8.1} us/iter",
                        start.elapsed().as_secs_f64() * 1e6 / ITERS as f64
                    );
                }
            }
        }
        return;
    }

    let batch = 2usize;
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

            // Side 1: dedicated MQA/GQA kernel.
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
                    causal,
                )
                .unwrap();
                std::hint::black_box(&out);
            }
            client.synchronize();
            let dedicated = start.elapsed();

            // Side 2: decode kernel over the folded view. `S_q == 1` is what
            // routes `flash_attention_fwd` to the decode kernel. The fold has
            // no causal form, so this side always runs non-causal.
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
