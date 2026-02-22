//! Integration tests for flash attention correctness.
//!
//! Tests verify that the CPU FlashAttentionOps (standard O(N²) fallback)
//! produces numerically correct results by comparing against manual
//! reference computations.

use boostr::Tensor;
use boostr::ops::traits::{FlashAttentionOps, PagedAttentionOps};
use numr::ops::{ActivationOps, BinaryOps, MatmulOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

fn setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

fn rand_tensor(shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
    Tensor::<CpuRuntime>::from_slice(&data, shape, device)
}

/// Reference standard attention: softmax(Q @ K^T / sqrt(d)) @ V
fn reference_attention(
    client: &CpuClient,
    q: &Tensor<CpuRuntime>,
    k: &Tensor<CpuRuntime>,
    v: &Tensor<CpuRuntime>,
    causal: bool,
) -> Tensor<CpuRuntime> {
    let head_dim = q.shape()[3];
    let seq_len_q = q.shape()[2];
    let seq_len_k = k.shape()[2];
    let scale = (head_dim as f64).sqrt().recip();

    let k_t = k.transpose(-2, -1).unwrap().contiguous();
    let scores = client.matmul(q, &k_t).unwrap();
    let scores = client.mul_scalar(&scores, scale).unwrap();

    let scores = if causal {
        let mask_data: Vec<f32> = (0..seq_len_q * seq_len_k)
            .map(|idx| {
                let i = idx / seq_len_k;
                let j = idx % seq_len_k;
                if j <= i { 0.0 } else { -1e9 }
            })
            .collect();
        let mask =
            Tensor::<CpuRuntime>::from_slice(&mask_data, &[1, 1, seq_len_q, seq_len_k], q.device());
        client.add(&scores, &mask).unwrap()
    } else {
        scores
    };

    let weights = client.softmax(&scores, -1).unwrap();
    client.matmul(&weights, v).unwrap()
}

fn max_abs_diff(client: &CpuClient, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> f32 {
    let diff = client.sub(a, b).unwrap();
    let abs_diff = client.abs(&diff).unwrap();
    let max = client.max(&abs_diff, &[], false).unwrap();
    max.to_vec::<f32>()[0]
}

#[test]
fn test_flash_v2_fwd_matches_reference() {
    let (client, device) = setup();
    let (b, h, s, d) = (2, 4, 16, 32);
    let q = rand_tensor(&[b, h, s, d], &device);
    let k = rand_tensor(&[b, h, s, d], &device);
    let v = rand_tensor(&[b, h, s, d], &device);

    let (flash_out, _lse) = client
        .flash_attention_fwd(&q, &k, &v, h, h, d, false, 0)
        .unwrap();
    let ref_out = reference_attention(&client, &q, &k, &v, false);

    let diff = max_abs_diff(&client, &flash_out, &ref_out);
    assert!(
        diff < 1e-5,
        "Flash v2 fwd output differs from reference by {diff}"
    );
}

#[test]
fn test_flash_v2_fwd_causal_matches_reference() {
    let (client, device) = setup();
    let (b, h, s, d) = (1, 2, 12, 16);
    let q = rand_tensor(&[b, h, s, d], &device);
    let k = rand_tensor(&[b, h, s, d], &device);
    let v = rand_tensor(&[b, h, s, d], &device);

    let (flash_out, _lse) = client
        .flash_attention_fwd(&q, &k, &v, h, h, d, true, 0)
        .unwrap();
    let ref_out = reference_attention(&client, &q, &k, &v, true);

    let diff = max_abs_diff(&client, &flash_out, &ref_out);
    assert!(
        diff < 1e-5,
        "Flash v2 causal fwd differs from reference by {diff}"
    );
}

#[test]
fn test_flash_v2_bwd_gradients_nonzero() {
    let (client, device) = setup();
    let (b, h, s, d) = (1, 2, 8, 16);
    let q = rand_tensor(&[b, h, s, d], &device);
    let k = rand_tensor(&[b, h, s, d], &device);
    let v = rand_tensor(&[b, h, s, d], &device);

    let (out, lse) = client
        .flash_attention_fwd(&q, &k, &v, h, h, d, false, 0)
        .unwrap();
    let dout = rand_tensor(&[b, h, s, d], &device);

    let (dq, dk, dv) = client
        .flash_attention_bwd(&dout, &q, &k, &v, &out, &lse, h, h, d, false, 0)
        .unwrap();

    for (name, grad) in [("dQ", &dq), ("dK", &dk), ("dV", &dv)] {
        let abs_sum = client.sum(&client.abs(grad).unwrap(), &[], false).unwrap();
        assert!(
            abs_sum.to_vec::<f32>()[0] > 1e-6,
            "{name} gradients are zero"
        );
    }
}

#[test]
fn test_gqa_correctness_various_ratios() {
    let (client, device) = setup();
    let (b, s, d) = (1, 8, 16);

    for (num_heads, num_kv_heads) in [(4, 4), (4, 2), (4, 1), (8, 2), (8, 1)] {
        let q = rand_tensor(&[b, num_heads, s, d], &device);
        let k = rand_tensor(&[b, num_kv_heads, s, d], &device);
        let v = rand_tensor(&[b, num_kv_heads, s, d], &device);

        let (out, lse) = client
            .flash_attention_fwd(&q, &k, &v, num_heads, num_kv_heads, d, false, 0)
            .unwrap();

        assert_eq!(out.shape(), &[b, num_heads, s, d]);
        assert_eq!(lse.shape(), &[b, num_heads, s]);

        let out_vec = out.to_vec::<f32>();
        assert!(
            out_vec.iter().all(|x| x.is_finite()),
            "GQA {num_heads}/{num_kv_heads} produced non-finite values"
        );

        let dout = rand_tensor(&[b, num_heads, s, d], &device);
        let (dq, dk, dv) = client
            .flash_attention_bwd(
                &dout,
                &q,
                &k,
                &v,
                &out,
                &lse,
                num_heads,
                num_kv_heads,
                d,
                false,
                0,
            )
            .unwrap();
        assert_eq!(dq.shape(), &[b, num_heads, s, d]);
        assert_eq!(dk.shape(), &[b, num_kv_heads, s, d]);
        assert_eq!(dv.shape(), &[b, num_kv_heads, s, d]);
    }
}

#[test]
fn test_sliding_window_correctness() {
    let (client, device) = setup();
    let (b, h, s, d) = (1, 2, 12, 16);
    let q = rand_tensor(&[b, h, s, d], &device);
    let k = rand_tensor(&[b, h, s, d], &device);
    let v = rand_tensor(&[b, h, s, d], &device);

    let window_size = 4;
    let (out_window, _) = client
        .flash_attention_fwd(&q, &k, &v, h, h, d, false, window_size)
        .unwrap();

    let (out_full, _) = client
        .flash_attention_fwd(&q, &k, &v, h, h, d, false, 0)
        .unwrap();

    // Sliding window restricts attention, outputs should differ
    let diff = max_abs_diff(&client, &out_window, &out_full);
    assert!(
        diff > 1e-6,
        "Sliding window output should differ from full attention"
    );

    let win_vec = out_window.to_vec::<f32>();
    assert!(
        win_vec.iter().all(|x| x.is_finite()),
        "Sliding window produced non-finite values"
    );
}

#[test]
fn test_paged_attention_output_correctness() {
    let (client, device) = setup();
    let (b, h, s, d) = (1, 2, 4, 16);
    let q = rand_tensor(&[b, h, s, d], &device);

    // KV in paged format: [num_blocks, block_size, num_kv_heads=1, head_dim]
    let block_size = 4;
    let num_blocks = 1;
    let k_blocks = rand_tensor(&[num_blocks, block_size, 1, d], &device);
    let v_blocks = rand_tensor(&[num_blocks, block_size, 1, d], &device);

    // Block table: sequence 0 uses block 0
    let bt_data: Vec<i32> = vec![0];
    let block_table = Tensor::<CpuRuntime>::from_slice(&bt_data, &[b, 1], &device);

    let (out, _lse) = client
        .paged_attention_fwd(
            &q,
            &k_blocks,
            &v_blocks,
            &block_table,
            h,
            s,
            s,
            d,
            block_size,
            false,
        )
        .unwrap();

    assert_eq!(out.shape(), &[b, h, s, d]);

    let out_vec = out.to_vec::<f32>();
    assert!(
        out_vec.iter().all(|x| x.is_finite()),
        "Paged attention produced non-finite values"
    );
}
