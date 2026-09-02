//! Backend parity for the CUDA split-KV decode attention path.
//!
//! The decode kernel takes one of two grid shapes. Below the split threshold a
//! single block per `(batch, head)` walks the whole KV sequence; above it, the
//! sequence is cut into slices whose partial softmax statistics are merged by a
//! combine pass. `src/ops/cuda/attention/decode_split.rs` picks between them
//! from the device's compute-unit count and the KV length, so the shapes here
//! are chosen to land on each side of that choice and are stated per test.
//!
//! Both the output and the log-sum-exp are compared: the combine pass is the
//! only producer of a decode LSE, and a rescaling error there shows up in the
//! LSE before it shows up in the normalized output.

use super::helpers::*;
use boostr::ops::traits::attention::flash::FlashAttentionOps;

/// Runs one decode shape on CPU and CUDA and asserts output and LSE agree.
///
/// `kv_capacity` is the allocated KV extent; `seq_len_k` is how much of it the
/// kernel reads. Passing a larger capacity exercises the separate KV stride,
/// which is what a real paged-free cache uses.
#[allow(clippy::too_many_arguments)]
fn assert_decode_parity(
    label: &str,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len_k: usize,
    kv_capacity: usize,
    q_gain: f32,
) {
    let (cpu_client, cpu_device) = setup_cpu();

    let q_shape = [batch, num_heads, 1, head_dim];
    let kv_shape = [batch, num_kv_heads, seq_len_k, head_dim];

    let q_base = det_tensor(&q_shape, &cpu_device);
    let q_vec: Vec<f32> = q_base.to_vec::<f32>().iter().map(|x| x * q_gain).collect();
    let k_vec = det_tensor(&kv_shape, &cpu_device).to_vec::<f32>();
    let v_vec = det_tensor(&kv_shape, &cpu_device).to_vec::<f32>();

    let q = numr::tensor::Tensor::from_slice(&q_vec, &q_shape, &cpu_device).unwrap();
    let k = numr::tensor::Tensor::from_slice(&k_vec, &kv_shape, &cpu_device).unwrap();
    let v = numr::tensor::Tensor::from_slice(&v_vec, &kv_shape, &cpu_device).unwrap();

    let (cpu_out, cpu_lse) = cpu_client
        .flash_attention_fwd(
            &q,
            &k,
            &v,
            num_heads,
            num_kv_heads,
            head_dim,
            false,
            0,
            None,
        )
        .unwrap_or_else(|e| panic!("CPU decode failed for {label}: {e}"));
    let cpu_out_vec = cpu_out.to_vec::<f32>();
    let cpu_lse_vec = cpu_lse.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use numr::tensor::Tensor;

        // The CUDA tensors are allocated at capacity with the tail left as
        // whatever `zeros` wrote, so a kernel that reads past `seq_len_k`
        // diverges from CPU instead of silently agreeing.
        let cap_shape = [batch, num_kv_heads, kv_capacity, head_dim];
        let mut k_cap = vec![0.0f32; batch * num_kv_heads * kv_capacity * head_dim];
        let mut v_cap = vec![0.0f32; batch * num_kv_heads * kv_capacity * head_dim];
        for bh in 0..batch * num_kv_heads {
            let src = bh * seq_len_k * head_dim;
            let dst = bh * kv_capacity * head_dim;
            let n = seq_len_k * head_dim;
            k_cap[dst..dst + n].copy_from_slice(&k_vec[src..src + n]);
            v_cap[dst..dst + n].copy_from_slice(&v_vec[src..src + n]);
        }

        let q_c = Tensor::from_slice(&q_vec, &q_shape, &cuda_device).unwrap();
        let k_c = Tensor::from_slice(&k_cap, &cap_shape, &cuda_device).unwrap();
        let v_c = Tensor::from_slice(&v_cap, &cap_shape, &cuda_device).unwrap();

        let (cuda_out, cuda_lse) = cuda_client
            .flash_attention_fwd(
                &q_c,
                &k_c,
                &v_c,
                num_heads,
                num_kv_heads,
                head_dim,
                false,
                0,
                Some(seq_len_k),
            )
            .unwrap_or_else(|e| panic!("CUDA decode failed for {label}: {e}"));

        assert_parity_f32_tol(
            &cuda_out.to_vec::<f32>(),
            &cpu_out_vec,
            &format!("{label} output CUDA vs CPU"),
            1e-4,
            1e-6,
        );
        assert_parity_f32_tol(
            &cuda_lse.to_vec::<f32>(),
            &cpu_lse_vec,
            &format!("{label} lse CUDA vs CPU"),
            1e-4,
            1e-6,
        );
    });
}

/// Below the minimum chunk: the whole-sequence kernel runs, one block per
/// `(batch, head)`. This is the path that must not change.
#[test]
fn decode_short_sequence_takes_whole_sequence_path() {
    assert_decode_parity("decode_short", 1, 4, 4, 64, 64, 64, 1.0);
}

/// Long enough to split, with the KV length an exact multiple of the slice
/// count, so every slice is full.
#[test]
fn decode_split_with_even_slices_parity() {
    assert_decode_parity("decode_even_slices", 1, 2, 2, 128, 640, 640, 1.0);
}

/// KV length not divisible by the slice count, so the last slice is short. This
/// is the bound the split kernel computes rather than reads.
#[test]
fn decode_split_with_ragged_last_slice_parity() {
    assert_decode_parity("decode_ragged", 1, 3, 3, 64, 1000, 1000, 1.0);
}

/// Grouped-query decode: several query heads share one KV head, so the split
/// kernel's `kv_h` mapping is exercised alongside the slicing.
#[test]
fn decode_split_grouped_query_parity() {
    assert_decode_parity("decode_gqa", 1, 8, 2, 128, 768, 768, 1.0);
}

/// Reads a prefix of a larger allocation, so the KV stride differs from the KV
/// length. Every slice offset is computed against the stride, not the length.
#[test]
fn decode_split_with_capacity_beyond_length_parity() {
    assert_decode_parity("decode_capacity", 1, 4, 4, 64, 700, 1024, 1.0);
}

/// Amplified queries spread the scores across a wide range, so slices see very
/// different running maxima and the combine pass must rescale rather than sum.
#[test]
fn decode_split_wide_score_range_parity() {
    assert_decode_parity("decode_wide_scores", 1, 2, 2, 128, 896, 896, 40.0);
}

/// Batch above one widens the base grid, which lowers the slice count without
/// removing the split. Both the batch and the slice index address the partials.
#[test]
fn decode_split_batched_parity() {
    assert_decode_parity("decode_batched", 3, 4, 4, 64, 512, 512, 1.0);
}
