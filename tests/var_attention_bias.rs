//! Integration tests for `var_attention_with_bias`, the autograd attention
//! primitive that accepts an additive bias/mask (ALiBi and friends).

use boostr::ops::impl_generic::attention::multi_head_attention_impl;
use boostr::ops::{AlibiOps, AttentionCausality, var_attention_with_bias};
use numr::autograd::{GradStore, Var, backward, var_sum};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

fn det_data(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + phase).sin() * 0.5)
        .collect()
}

fn qkv(
    device: &CpuDevice,
    shape: &[usize],
    requires_grad: bool,
) -> (Var<CpuRuntime>, Var<CpuRuntime>, Var<CpuRuntime>) {
    let n: usize = shape.iter().product();
    let q = Var::new(
        Tensor::<CpuRuntime>::from_slice(&det_data(n, 0.0), shape, device).unwrap(),
        requires_grad,
    );
    let k = Var::new(
        Tensor::<CpuRuntime>::from_slice(&det_data(n, 1.1), shape, device).unwrap(),
        requires_grad,
    );
    let v = Var::new(
        Tensor::<CpuRuntime>::from_slice(&det_data(n, 2.3), shape, device).unwrap(),
        requires_grad,
    );
    (q, k, v)
}

fn grad_norm(grads: &GradStore<CpuRuntime>, var: &Var<CpuRuntime>, name: &str) -> TestResult<f32> {
    let g = grads
        .get(var.tensor().id())
        .ok_or_else(|| format!("no gradient for {name}"))?;
    // Backward produces strided gradients (transposes in the matmul rule), and
    // `to_vec` requires contiguous storage.
    let g = g.contiguous()?;
    Ok(g.to_vec::<f32>().iter().map(|x| x * x).sum::<f32>())
}

/// Gradients must reach Q, K and V — and be actually non-zero, not merely
/// present. A severed graph (re-wrapping a grad-path tensor in `Var::new`)
/// shows up here and nowhere else.
#[test]
fn gradients_flow_to_qkv_through_bias() -> TestResult {
    let (client, device) = setup();
    let (b, h, s, d) = (2, 2, 4, 8);
    let (q, k, v) = qkv(&device, &[b, h, s, d], true);

    let bias = Tensor::<CpuRuntime>::zeros(&[b, h, s, s], DType::F32, &device).unwrap();
    client.alibi_add_bias(&bias, b, h, s, s)?;

    let out = var_attention_with_bias(
        &client,
        &q,
        &k,
        &v,
        &bias,
        AttentionCausality::Mask { window_size: 0 },
        h,
    )?;
    let loss = var_sum(&out, &[0, 1, 2, 3], false, &client)?;
    let grads = backward(&loss, &client)?;

    for (var, name) in [(&q, "q"), (&k, "k"), (&v, "v")] {
        let norm = grad_norm(&grads, var, name)?;
        assert!(norm > 1e-8, "{name}: gradient norm is ~zero ({norm})");
    }
    Ok(())
}

/// A zero bias must reproduce the unbiased path exactly.
#[test]
fn zero_bias_matches_unbiased() -> TestResult {
    let (client, device) = setup();
    let (b, h, s, d) = (1, 2, 5, 4);
    let (q, k, v) = qkv(&device, &[b, h, s, d], false);

    let zero = Tensor::<CpuRuntime>::zeros(&[b, h, s, s], DType::F32, &device).unwrap();
    let biased = var_attention_with_bias(
        &client,
        &q,
        &k,
        &v,
        &zero,
        AttentionCausality::Bidirectional,
        h,
    )?;
    let plain = multi_head_attention_impl(&client, &q, &k, &v, None, h)?;

    let a = biased.tensor().to_vec::<f32>();
    let c = plain.tensor().to_vec::<f32>();
    assert_eq!(a.len(), c.len());
    for (i, (x, y)) in a.iter().zip(c.iter()).enumerate() {
        assert!((x - y).abs() < 1e-6, "idx {i}: {x} vs {y}");
    }
    Ok(())
}

/// A non-zero bias must change the result.
#[test]
fn nonzero_bias_changes_result() -> TestResult {
    let (client, device) = setup();
    let (b, h, s, d) = (1, 2, 5, 4);
    let (q, k, v) = qkv(&device, &[b, h, s, d], false);

    let zero = Tensor::<CpuRuntime>::zeros(&[b, h, s, s], DType::F32, &device).unwrap();
    let alibi = Tensor::<CpuRuntime>::zeros(&[b, h, s, s], DType::F32, &device).unwrap();
    client.alibi_add_bias(&alibi, b, h, s, s)?;

    let base = var_attention_with_bias(
        &client,
        &q,
        &k,
        &v,
        &zero,
        AttentionCausality::Bidirectional,
        h,
    )?;
    let shifted = var_attention_with_bias(
        &client,
        &q,
        &k,
        &v,
        &alibi,
        AttentionCausality::Bidirectional,
        h,
    )?;

    let a = base.tensor().to_vec::<f32>();
    let c = shifted.tensor().to_vec::<f32>();
    let max_diff = a
        .iter()
        .zip(c.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff > 1e-4, "bias had no effect (max diff {max_diff})");
    Ok(())
}

/// ALiBi bias from the existing kernel, against a hand-computed case.
///
/// H=1, S=2, D=1, q=k=1 so scores are all 1 and scale is 1. Slope for the only
/// head is 2^0 = 1, so bias[i][j] = -|i - j| and the biased scores are
/// [[1, 0], [0, 1]]. softmax gives [e/(e+1), 1/(e+1)] per row, and with
/// v = [3, 5] the outputs are 3*0.73106 + 5*0.26894 and its mirror.
#[test]
fn alibi_bias_matches_hand_computed() -> TestResult {
    let (client, device) = setup();
    let (b, h, s, d) = (1, 1, 2, 1);
    let ones = vec![1.0f32; b * h * s * d];
    let q = Var::new(
        Tensor::<CpuRuntime>::from_slice(&ones, &[b, h, s, d], &device).unwrap(),
        false,
    );
    let k = Var::new(
        Tensor::<CpuRuntime>::from_slice(&ones, &[b, h, s, d], &device).unwrap(),
        false,
    );
    let v = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[3.0f32, 5.0], &[b, h, s, d], &device).unwrap(),
        false,
    );

    let bias = Tensor::<CpuRuntime>::zeros(&[b, h, s, s], DType::F32, &device).unwrap();
    client.alibi_add_bias(&bias, b, h, s, s)?;

    let out = var_attention_with_bias(
        &client,
        &q,
        &k,
        &v,
        &bias,
        AttentionCausality::Bidirectional,
        h,
    )?;
    let got = out.tensor().to_vec::<f32>();

    let w_near = 1.0f32.exp() / (1.0f32.exp() + 1.0);
    let w_far = 1.0 / (1.0f32.exp() + 1.0);
    let expected = [3.0 * w_near + 5.0 * w_far, 3.0 * w_far + 5.0 * w_near];
    for (i, (x, y)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((x - y).abs() < 1e-5, "idx {i}: {x} vs {y}");
    }
    Ok(())
}

/// Same ALiBi bias, routed through `var_attention_with_bias` and through the
/// raw composition the LLaMA ALiBi block uses. Must agree bit-for-bit.
#[test]
fn alibi_matches_llama_composition() -> TestResult {
    let (client, device) = setup();
    let (b, h, s, d) = (2, 4, 6, 8);
    let (q, k, v) = qkv(&device, &[b, h, s, d], false);

    let bias = Tensor::<CpuRuntime>::zeros(&[b, h, s, s], DType::F32, &device).unwrap();
    client.alibi_add_bias(&bias, b, h, s, s)?;

    let got = var_attention_with_bias(
        &client,
        &q,
        &k,
        &v,
        &bias,
        AttentionCausality::Bidirectional,
        h,
    )?;

    // The LLaMA ALiBi path: zeros -> alibi_add_bias -> detached Var -> impl.
    let ref_bias = Tensor::<CpuRuntime>::zeros(&[b, h, s, s], DType::F32, &device).unwrap();
    client.alibi_add_bias(&ref_bias, b, h, s, s)?;
    let ref_var = Var::new(ref_bias, false);
    let want = multi_head_attention_impl(&client, &q, &k, &v, Some(&ref_var), h)?;

    let a = got.tensor().to_vec::<f32>();
    let c = want.tensor().to_vec::<f32>();
    for (i, (x, y)) in a.iter().zip(c.iter()).enumerate() {
        assert!((x - y).abs() < 1e-6, "idx {i}: {x} vs {y}");
    }
    Ok(())
}

/// `f32::MIN` entries must drive their attention weight to ~zero: with a full
/// causal mask, query 0 can only see key 0, so its output IS v[0].
#[test]
fn masked_positions_get_zero_weight() -> TestResult {
    let (client, device) = setup();
    let (b, h, s, d) = (1, 1, 4, 2);
    let (q, k, _) = qkv(&device, &[b, h, s, d], false);
    let v_data: Vec<f32> = (0..s * d).map(|i| (i as f32) + 1.0).collect();
    let v = Var::new(
        Tensor::<CpuRuntime>::from_slice(&v_data, &[b, h, s, d], &device).unwrap(),
        false,
    );

    let zero = Tensor::<CpuRuntime>::zeros(&[b, h, s, s], DType::F32, &device).unwrap();
    let out = var_attention_with_bias(
        &client,
        &q,
        &k,
        &v,
        &zero,
        AttentionCausality::Mask { window_size: 0 },
        h,
    )?;
    let got = out.tensor().to_vec::<f32>();

    for j in 0..d {
        assert!(
            (got[j] - v_data[j]).abs() < 1e-6,
            "query 0 leaked past the mask at dim {j}: {} vs {}",
            got[j],
            v_data[j]
        );
    }
    Ok(())
}
