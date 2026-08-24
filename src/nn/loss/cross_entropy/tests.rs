//! Tests for cross-entropy losses.

use super::*;
use crate::test_utils::cpu_setup;
use numr::autograd::backward;
use numr::runtime::cpu::CpuRuntime;

#[test]
fn test_cross_entropy_basic() {
    let (client, device) = cpu_setup();

    #[rustfmt::skip]
    let logits = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(
            &[2.0f32, 1.0, 0.1,   // sample 0: class 0 is highest
              0.1, 2.0, 1.0],     // sample 1: class 1 is highest
            &[2, 3],
            &device,
        ).unwrap(),
        true,
    );
    let targets = Tensor::<CpuRuntime>::try_from_slice(&[0i64, 1], &[2], &device).unwrap();

    let loss = cross_entropy_loss(&client, &logits, &targets).unwrap();
    assert_eq!(loss.shape(), &[] as &[usize]);
    let val: Vec<f32> = loss.tensor().to_vec();
    assert!(
        val[0] < 1.0,
        "loss={} should be < 1.0 for correct predictions",
        val[0]
    );
}

#[test]
fn test_cross_entropy_wrong_predictions() {
    let (client, device) = cpu_setup();

    let logits = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(
            &[
                0.1f32, 0.1, 2.0, // sample 0: class 2 is highest
                2.0, 0.1, 0.1, // sample 1: class 0 is highest
            ],
            &[2, 3],
            &device,
        )
        .unwrap(),
        false,
    );
    let targets = Tensor::<CpuRuntime>::try_from_slice(&[0i64, 1], &[2], &device).unwrap();

    let loss = cross_entropy_loss(&client, &logits, &targets).unwrap();
    let val: Vec<f32> = loss.tensor().to_vec();
    assert!(
        val[0] > 1.0,
        "loss={} should be > 1.0 for wrong predictions",
        val[0]
    );
}

#[test]
fn test_label_smoothing_reduces_confidence() {
    let (client, device) = cpu_setup();

    let logits = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(&[2.0f32, 1.0, 0.1, 0.1, 2.0, 1.0], &[2, 3], &device)
            .unwrap(),
        false,
    );
    let targets = Tensor::<CpuRuntime>::try_from_slice(&[0i64, 1], &[2], &device).unwrap();

    let loss_no_smooth = cross_entropy_loss(&client, &logits, &targets).unwrap();
    let loss_smooth = cross_entropy_loss_smooth(&client, &logits, &targets, 0.1).unwrap();

    let v0: Vec<f32> = loss_no_smooth.tensor().to_vec();
    let vs: Vec<f32> = loss_smooth.tensor().to_vec();

    assert!(
        vs[0] > v0[0],
        "smoothed loss {} should be > unsmoothed {}",
        vs[0],
        v0[0]
    );
}

#[test]
fn test_label_smoothing_zero_is_ce() {
    let (client, device) = cpu_setup();

    let logits = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(&[2.0f32, 1.0, 0.1, 0.1, 2.0, 1.0], &[2, 3], &device)
            .unwrap(),
        false,
    );
    let targets = Tensor::<CpuRuntime>::try_from_slice(&[0i64, 1], &[2], &device).unwrap();

    let loss_ce = cross_entropy_loss(&client, &logits, &targets).unwrap();
    let loss_smooth = cross_entropy_loss_smooth(&client, &logits, &targets, 0.0).unwrap();

    let v0: Vec<f32> = loss_ce.tensor().to_vec();
    let vs: Vec<f32> = loss_smooth.tensor().to_vec();
    assert!(
        (v0[0] - vs[0]).abs() < 1e-6,
        "smoothing=0 should match CE: {} vs {}",
        v0[0],
        vs[0]
    );
}

/// Negative log-likelihood of one row, straight from the definition.
fn nll_row(row: &[f32], target: usize) -> f64 {
    let max = row.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b)) as f64;
    let sum_exp: f64 = row.iter().map(|l| (*l as f64 - max).exp()).sum();
    max + sum_exp.ln() - row[target] as f64
}

#[test]
fn test_masked_all_ones_matches_unmasked() {
    let (client, device) = cpu_setup();

    let values = [
        2.0f32, 1.0, 0.1, // row 0
        0.1, 2.0, 1.0, // row 1
        -1.0, 0.5, 3.0, // row 2
        0.7, -0.2, 0.3, // row 3
    ];
    let logits = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(&values, &[4, 3], &device).unwrap(),
        false,
    );
    let targets = Tensor::<CpuRuntime>::try_from_slice(&[0i64, 1, 2, 1], &[4], &device).unwrap();
    let mask =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device).unwrap();

    let plain = cross_entropy_loss(&client, &logits, &targets).unwrap();
    let masked = cross_entropy_loss_masked(&client, &logits, &targets, &mask).unwrap();

    let p: Vec<f32> = plain.tensor().to_vec();
    let m: Vec<f32> = masked.tensor().to_vec();
    assert!(
        (p[0] - m[0]).abs() < 1e-6,
        "all-ones mask should match unmasked: {} vs {}",
        p[0],
        m[0]
    );
}

#[test]
fn test_masked_out_positions_are_excluded() {
    let (client, device) = cpu_setup();

    // Rows 1 and 3 are masked out and carry deliberately terrible logits.
    let values = [
        2.0f32, 1.0, 0.1, // row 0: kept
        -20.0, 5.0, 5.0, // row 1: masked out, target 0 is hopeless
        -1.0, 0.5, 3.0, // row 2: kept
        5.0, 5.0, -20.0, // row 3: masked out, target 2 is hopeless
    ];
    let logits = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(&values, &[4, 3], &device).unwrap(),
        false,
    );
    let targets = Tensor::<CpuRuntime>::try_from_slice(&[0i64, 0, 2, 2], &[4], &device).unwrap();
    let mask =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 1.0, 0.0], &[4], &device).unwrap();

    let masked = cross_entropy_loss_masked(&client, &logits, &targets, &mask).unwrap();
    let plain = cross_entropy_loss(&client, &logits, &targets).unwrap();

    let expected = (nll_row(&values[0..3], 0) + nll_row(&values[6..9], 2)) / 2.0;
    let m: Vec<f32> = masked.tensor().to_vec();
    let p: Vec<f32> = plain.tensor().to_vec();

    assert!(
        (m[0] as f64 - expected).abs() < 1e-5,
        "masked loss {} should equal kept-only loss {expected}",
        m[0]
    );
    assert!(
        (m[0] - p[0]).abs() > 1.0,
        "masked loss {} should differ from unmasked {}",
        m[0],
        p[0]
    );
}

#[test]
fn test_denominator_counts_only_kept_positions() {
    let (client, device) = cpu_setup();

    let kept = [
        2.0f32, 1.0, 0.1, // kept row 0
        -1.0, 0.5, 3.0, // kept row 1
    ];
    let kept_targets = [0i64, 2];
    let expected = (nll_row(&kept[0..3], 0) + nll_row(&kept[3..6], 2)) / 2.0;

    // Growing padding of masked-out rows must not change the loss.
    for pad in 0..5 {
        let mut values = kept.to_vec();
        let mut targets_data = kept_targets.to_vec();
        let mut mask_data = vec![1.0f32, 1.0];
        for _ in 0..pad {
            values.extend_from_slice(&[-30.0f32, 12.0, 7.5]);
            targets_data.push(0);
            mask_data.push(0.0);
        }
        let n = 2 + pad;

        let logits = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&values, &[n, 3], &device).unwrap(),
            false,
        );
        let targets = Tensor::<CpuRuntime>::try_from_slice(&targets_data, &[n], &device).unwrap();
        let mask = Tensor::<CpuRuntime>::try_from_slice(&mask_data, &[n], &device).unwrap();

        let loss = cross_entropy_loss_masked(&client, &logits, &targets, &mask).unwrap();
        let v: Vec<f32> = loss.tensor().to_vec();
        assert!(
            (v[0] as f64 - expected).abs() < 1e-5,
            "loss {} changed with {pad} masked-out rows, expected {expected}",
            v[0]
        );
    }
}

#[test]
fn test_all_zero_mask_errors() {
    let (client, device) = cpu_setup();

    let logits = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(&[2.0f32, 1.0, 0.1, 0.1, 2.0, 1.0], &[2, 3], &device)
            .unwrap(),
        false,
    );
    let targets = Tensor::<CpuRuntime>::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
    let mask = Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.0], &[2], &device).unwrap();

    let err = cross_entropy_loss_masked(&client, &logits, &targets, &mask).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("mask selected no positions"),
        "unexpected error: {msg}"
    );
}

#[test]
fn test_shape_mismatch_errors() {
    let (client, device) = cpu_setup();

    let values = [2.0f32, 1.0, 0.1, 0.1, 2.0, 1.0];
    let targets = Tensor::<CpuRuntime>::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
    let mask = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0], &[2], &device).unwrap();

    // logits must be rank 2
    let logits_3d = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(&values, &[1, 2, 3], &device).unwrap(),
        false,
    );
    let err = cross_entropy_loss_masked(&client, &logits_3d, &targets, &mask).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument { arg: "logits", .. }),
        "expected logits error, got {err}"
    );

    let logits = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(&values, &[2, 3], &device).unwrap(),
        false,
    );

    // targets length must be N
    let bad_targets = Tensor::<CpuRuntime>::try_from_slice(&[0i64, 1, 0], &[3], &device).unwrap();
    let err = cross_entropy_loss_masked(&client, &logits, &bad_targets, &mask).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument { arg: "targets", .. }),
        "expected targets error, got {err}"
    );

    // mask length must be N
    let bad_mask =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0, 1.0], &[3], &device).unwrap();
    let err = cross_entropy_loss_masked(&client, &logits, &targets, &bad_mask).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument { arg: "mask", .. }),
        "expected mask error, got {err}"
    );
}

#[test]
fn test_gradient_flows_only_to_kept_positions() {
    let (client, device) = cpu_setup();

    let values = [
        2.0f32, 1.0, 0.1, // row 0: kept
        0.1, 2.0, 1.0, // row 1: masked out
        -1.0, 0.5, 3.0, // row 2: kept
    ];
    let logits = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(&values, &[3, 3], &device).unwrap(),
        true,
    );
    let targets = Tensor::<CpuRuntime>::try_from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
    let mask = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 1.0], &[3], &device).unwrap();

    let loss = cross_entropy_loss_masked(&client, &logits, &targets, &mask).unwrap();
    let grads = backward(&loss, &client).unwrap();
    let g: Vec<f32> = grads.get(logits.id()).unwrap().to_vec();

    for (i, v) in g.iter().enumerate().take(6).skip(3) {
        assert_eq!(*v, 0.0, "masked-out grad[{i}] = {v} should be exactly zero");
    }
    assert!(
        g[0..3].iter().any(|v| v.abs() > 1e-6),
        "kept row 0 should have gradient, got {:?}",
        &g[0..3]
    );
    assert!(
        g[6..9].iter().any(|v| v.abs() > 1e-6),
        "kept row 2 should have gradient, got {:?}",
        &g[6..9]
    );
}

/// Cross-entropy at BF16 must match an F32 reference over the same logits.
///
/// 512 tokens over a 1024-way vocabulary with all-zero logits: every per-token
/// loss is exactly `ln(1024) = 6.9315`, so the loss is that number however the
/// mean is computed — unless the mean saturates.
///
/// It did. `cross_entropy_loss` ends in a mean over the flattened `[N, 1]`
/// per-token losses, and numr summed those 512 BF16 values in BF16. The
/// running sum stalled at `2048`, so the loss came back as exactly `4.0`
/// regardless of the logits. The same defect at a 512-token, 128256-way
/// vocabulary stalled at `4096` and reported `loss 8.0000` on every batch.
///
/// This test is dark under a plain `cargo test`: `f16` is not a default
/// boostr feature, and BF16 tensors need it.
#[cfg(feature = "f16")]
#[test]
fn test_cross_entropy_bf16_matches_f32_reference() {
    use half::bf16;

    let (client, device) = cpu_setup();

    const N: usize = 512;
    const V: usize = 1024;

    let target_data: Vec<i64> = (0..N as i64).map(|i| i % V as i64).collect();
    let targets = Tensor::<CpuRuntime>::try_from_slice(&target_data, &[N], &device).unwrap();

    let f32_logits = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(&vec![0.0f32; N * V], &[N, V], &device).unwrap(),
        false,
    );
    let f32_loss: f32 = cross_entropy_loss(&client, &f32_logits, &targets)
        .unwrap()
        .tensor()
        .item()
        .unwrap();
    assert!(
        (f32_loss - 6.931_472).abs() < 1e-4,
        "F32 reference moved: {f32_loss}"
    );

    let bf16_logits = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(&vec![bf16::from_f32(0.0); N * V], &[N, V], &device)
            .unwrap(),
        false,
    );
    let bf16_loss: bf16 = cross_entropy_loss(&client, &bf16_logits, &targets)
        .unwrap()
        .tensor()
        .item()
        .unwrap();
    let bf16_loss = bf16_loss.to_f32();

    assert!(
        (bf16_loss - f32_loss).abs() < 0.06,
        "BF16 loss {bf16_loss} does not match F32 reference {f32_loss}"
    );
}
