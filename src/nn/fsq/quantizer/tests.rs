//! Unit tests for [`super::Fsq`].
//!
//! Split into a sibling file to keep `quantizer.rs` readable; still
//! `#[cfg(test)]`-only and still the `tests` submodule of `quantizer`.

use super::*;
use crate::test_utils::cpu_setup;
use numr::autograd::backward;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

fn toy_fsq() -> (Fsq<CpuRuntime>, CpuClient, CpuDevice) {
    let (client, device) = cpu_setup();
    let config = FsqConfig::new(vec![4, 4], 2).unwrap();
    let fsq = Fsq::new(config, &device, None, None).unwrap();
    (fsq, client, device)
}

// --- grid values, hand-computed from the reference formula -----------

/// For levels=[4,4]: half_width = 4 // 2 = 2, so
/// `code = (level_index - 2) / 2` for level_index in {0,1,2,3} gives the
/// grid {-1.0, -0.5, 0.0, 0.5} — every combination of the two dims should
/// land exactly there. This is `indices_to_codes` alone (project_out is
/// None for the toy config), so it's a pure check of the mixed-radix
/// unpack + scale/shift math, independent of the tanh-based encode path.
#[test]
fn test_decode_grid_values_toy() {
    let (fsq, client, device) = toy_fsq();
    let expected_grid = [-1.0f32, -0.5, 0.0, 0.5];

    for index in 0..16i32 {
        let indices = Tensor::<CpuRuntime>::from_slice(&[index], &[1], &device);
        let codes = fsq.indices_to_codes(&client, &indices).unwrap();
        let data: Vec<f32> = codes.tensor().contiguous().unwrap().to_vec();

        let dim0 = index % 4;
        let dim1 = index / 4;
        assert!(
            (data[0] - expected_grid[dim0 as usize]).abs() < 1e-5,
            "index {index}: dim0 = {}, expected {}",
            data[0],
            expected_grid[dim0 as usize]
        );
        assert!(
            (data[1] - expected_grid[dim1 as usize]).abs() < 1e-5,
            "index {index}: dim1 = {}, expected {}",
            data[1],
            expected_grid[dim1 as usize]
        );
    }
}

/// The quantized (encoded) codes must also land on the same discrete grid
/// as the decode path, for saturating (large-magnitude) inputs where the
/// tanh bound is unambiguous: driving z very negative saturates
/// tanh(z+shift) -> -1, giving bounded_z -> -half_l - offset = -2.0015,
/// which rounds to -2 -> code -1.0. Driving z very positive saturates
/// tanh -> +1, giving bounded_z -> half_l - offset = 1.0015, which rounds
/// to 1 -> code 0.5. z = 0 falls in between and must land on some point of
/// the same 4-point grid.
#[test]
fn test_encode_grid_values_toy() {
    let (fsq, client, device) = toy_fsq();
    let allowed = [-1.0f32, -0.5, 0.0, 0.5];

    for &z_val in &[-5.0f32, 0.0, 5.0] {
        let z = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[z_val, z_val], &[1, 2], &device),
            false,
        );
        let (codes, _) = fsq.quantize(&client, &z).unwrap();
        let data: Vec<f32> = codes.tensor().contiguous().unwrap().to_vec();
        for &v in &data {
            assert!(
                allowed.iter().any(|&g| (g - v).abs() < 1e-4),
                "z={z_val} produced off-grid code {v}"
            );
        }
    }

    // Saturating extremes hit the exact boundary grid points.
    let z_neg = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[-5.0f32, -5.0], &[1, 2], &device),
        false,
    );
    let (codes_neg, _) = fsq.quantize(&client, &z_neg).unwrap();
    let data_neg: Vec<f32> = codes_neg.tensor().contiguous().unwrap().to_vec();
    assert!((data_neg[0] - (-1.0)).abs() < 1e-4);
    assert!((data_neg[1] - (-1.0)).abs() < 1e-4);

    let z_pos = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[5.0f32, 5.0], &[1, 2], &device),
        false,
    );
    let (codes_pos, _) = fsq.quantize(&client, &z_pos).unwrap();
    let data_pos: Vec<f32> = codes_pos.tensor().contiguous().unwrap().to_vec();
    assert!((data_pos[0] - 0.5).abs() < 1e-4);
    assert!((data_pos[1] - 0.5).abs() < 1e-4);
}

// --- straight-through gradient -----------------------------------------

/// Backward through `quantize` must reach `z` with a non-zero gradient.
/// Uses small, asymmetric (non-zero, unequal-magnitude, mixed-sign) inputs
/// so the tanh derivative is genuinely non-zero and a coincidental zero
/// can't produce a false pass.
#[test]
fn test_straight_through_gradient_nonzero() {
    let (fsq, client, device) = toy_fsq();
    let z = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.31f32, -0.72], &[1, 2], &device),
        true,
    );

    let (codes, _) = fsq.quantize(&client, &z).unwrap();
    let loss = numr::autograd::var_sum(&codes, &[0, 1], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let grad = grads
        .get(z.id())
        .expect("straight-through estimator must propagate gradient to z");
    let grad_data: Vec<f32> = grad.contiguous().unwrap().to_vec();
    assert!(
        grad_data.iter().all(|&g| g != 0.0),
        "expected non-zero gradient on every element, got {grad_data:?}"
    );
}

// --- projection wiring ---------------------------------------------------

#[test]
fn test_projection_required_when_dims_differ() {
    let (_, device) = cpu_setup();
    let config = FsqConfig::new(vec![4, 4], 5).unwrap();
    let err = Fsq::<CpuRuntime>::new(config, &device, None, None)
        .err()
        .unwrap();
    assert!(matches!(
        err,
        Error::InvalidArgument {
            arg: "project_in/project_out",
            ..
        }
    ));
}

#[test]
fn test_projection_rejected_when_dims_match() {
    let (_, device) = cpu_setup();
    let config = FsqConfig::new(vec![4, 4], 2).unwrap();
    let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[2, 2], &device);
    let project_in = Some(Linear::new(weight, None, false));
    let err = Fsq::<CpuRuntime>::new(config, &device, project_in, None)
        .err()
        .unwrap();
    assert!(matches!(
        err,
        Error::InvalidArgument {
            arg: "project_in/project_out",
            ..
        }
    ));
}

#[test]
fn test_projection_roundtrips_shape() {
    let (client, device) = cpu_setup();
    // input_dim=5, codebook_dim=2 (mirrors NeuCodec's dim != levels.len()).
    let config = FsqConfig::new(vec![4, 4], 5).unwrap();

    let w_in = Tensor::<CpuRuntime>::from_slice(
        &[0.1f32; 10], // [codebook_dim=2, input_dim=5]
        &[2, 5],
        &device,
    );
    let w_out = Tensor::<CpuRuntime>::from_slice(
        &[0.2f32; 10], // [input_dim=5, codebook_dim=2]
        &[5, 2],
        &device,
    );
    let project_in = Some(Linear::new(w_in, None, false));
    let project_out = Some(Linear::new(w_out, None, false));

    let fsq = Fsq::new(config, &device, project_in, project_out).unwrap();

    let z = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.5f32; 5], &[1, 5], &device),
        false,
    );
    let (codes, indices) = fsq.quantize(&client, &z).unwrap();
    assert_eq!(codes.shape(), &[1, 5]);
    assert_eq!(indices.shape(), &[1]);

    let decoded = fsq.indices_to_codes(&client, &indices).unwrap();
    assert_eq!(decoded.shape(), &[1, 5]);
}
