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
        Tensor::<CpuRuntime>::from_slice(
            &[2.0f32, 1.0, 0.1,   // sample 0: class 0 is highest
              0.1, 2.0, 1.0],     // sample 1: class 1 is highest
            &[2, 3],
            &device,
        ),
        true,
    );
    let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);

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
        Tensor::<CpuRuntime>::from_slice(
            &[
                0.1f32, 0.1, 2.0, // sample 0: class 2 is highest
                2.0, 0.1, 0.1, // sample 1: class 0 is highest
            ],
            &[2, 3],
            &device,
        ),
        false,
    );
    let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);

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
        Tensor::<CpuRuntime>::from_slice(&[2.0f32, 1.0, 0.1, 0.1, 2.0, 1.0], &[2, 3], &device),
        false,
    );
    let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);

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
        Tensor::<CpuRuntime>::from_slice(&[2.0f32, 1.0, 0.1, 0.1, 2.0, 1.0], &[2, 3], &device),
        false,
    );
    let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);

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
        Tensor::<CpuRuntime>::from_slice(&values, &[4, 3], &device),
        false,
    );
    let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 1], &[4], &device);
    let mask = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device);

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
        Tensor::<CpuRuntime>::from_slice(&values, &[4, 3], &device),
        false,
    );
    let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 2, 2], &[4], &device);
    let mask = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 1.0, 0.0], &[4], &device);

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
            Tensor::<CpuRuntime>::from_slice(&values, &[n, 3], &device),
            false,
        );
        let targets = Tensor::<CpuRuntime>::from_slice(&targets_data, &[n], &device);
        let mask = Tensor::<CpuRuntime>::from_slice(&mask_data, &[n], &device);

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
        Tensor::<CpuRuntime>::from_slice(&[2.0f32, 1.0, 0.1, 0.1, 2.0, 1.0], &[2, 3], &device),
        false,
    );
    let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);
    let mask = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device);

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
    let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device);
    let mask = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device);

    // logits must be rank 2
    let logits_3d = Var::new(
        Tensor::<CpuRuntime>::from_slice(&values, &[1, 2, 3], &device),
        false,
    );
    let err = cross_entropy_loss_masked(&client, &logits_3d, &targets, &mask).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument { arg: "logits", .. }),
        "expected logits error, got {err}"
    );

    let logits = Var::new(
        Tensor::<CpuRuntime>::from_slice(&values, &[2, 3], &device),
        false,
    );

    // targets length must be N
    let bad_targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 0], &[3], &device);
    let err = cross_entropy_loss_masked(&client, &logits, &bad_targets, &mask).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument { arg: "targets", .. }),
        "expected targets error, got {err}"
    );

    // mask length must be N
    let bad_mask = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[3], &device);
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
        Tensor::<CpuRuntime>::from_slice(&values, &[3, 3], &device),
        true,
    );
    let targets = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2], &[3], &device);
    let mask = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 1.0], &[3], &device);

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
