//! Unit tests for [`super::ResidualFsq`].
//!
//! Split into its own file (still `#[cfg(test)]`-only, included via `#[path]`
//! from `residual.rs`) to keep `residual.rs` under the nn/*.rs file-size limit,
//! exactly as `quantizer_tests.rs` does for `quantizer.rs`.

use super::*;
use crate::nn::fsq::config::FsqConfig;
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

/// NeuCodec's grid: 8 dims, 4 levels each, no projections (dim == codebook_dim).
const NEUCODEC_LEVELS: [u32; 8] = [4; 8];

fn neucodec_residual() -> (ResidualFsq<CpuRuntime>, CpuClient, CpuDevice) {
    let (client, device) = cpu_setup();
    let config = ResidualFsqConfig::new(NEUCODEC_LEVELS.to_vec(), 8, 1).unwrap();
    let layer = Fsq::new(config.layer_config().unwrap(), &device, None, None).unwrap();
    let residual = ResidualFsq::new(
        config,
        ResidualFsqWeights {
            project_in: None,
            project_out: None,
            layers: vec![layer],
        },
        &device,
    )
    .unwrap();
    (residual, client, device)
}

/// `z` chosen so `bound(z) = 0.49` (just BELOW the rounding boundary) while
/// `bound(bound(z)) = 0.527` (just ABOVE it), for `levels = 4`:
///
/// ```text
/// half_l = 3 * 1.001 / 2 = 1.5015, offset = 0.5, shift = atanh(0.5 / half_l)
/// bound(0.44542) = 0.49    -> round -> 0 -> level index 2
/// bound(0.49)    = 0.5267  -> round -> 1 -> level index 3
/// ```
///
/// So the single-bound and double-bound encodes MUST disagree here.
const DOUBLE_BOUND_DISCRIMINATOR: f32 = 0.44542;

// --- the double bound is real, and must never silently regress ------------

/// `ResidualFsq::encode` applies `bound` twice (once to seed `residual`, once
/// inside `Fsq::quantize`). A bare `Fsq::quantize` on the same input applies it
/// once. Because `bound` is not idempotent, the two MUST produce different
/// indices at [`DOUBLE_BOUND_DISCRIMINATOR`].
///
/// If someone "simplifies" the pre-bound away, this test fails — which is the
/// entire point: on the real NeuCodec checkpoint that change silently rewrites
/// 43.75% of emitted indices.
#[test]
fn test_encode_applies_double_bound() {
    let (residual, client, device) = neucodec_residual();

    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[DOUBLE_BOUND_DISCRIMINATOR; 8], &[1, 8], &device),
        false,
    );

    // Double-bound (ResidualFSQ semantics): [1, num_quantizers = 1].
    let (_, double_bound_indices) = residual.encode(&client, &x).unwrap();
    let double_bound: Vec<i32> = double_bound_indices.contiguous().unwrap().to_vec();

    // Single-bound reference, constructed inline: a bare FSQ layer, which is
    // exactly `round_ste(bound(x)) / half_width` with NO pre-bound.
    let single_layer = Fsq::<CpuRuntime>::new(
        FsqConfig::new(NEUCODEC_LEVELS.to_vec(), 8).unwrap(),
        &device,
        None,
        None,
    )
    .unwrap();
    let (_, single_bound_indices) = single_layer.quantize(&client, &x).unwrap();
    let single_bound: Vec<i32> = single_bound_indices.contiguous().unwrap().to_vec();

    assert_ne!(
        double_bound, single_bound,
        "ResidualFsq::encode collapsed to a single bound — the pre-bound seeding \
         `residual` was removed or made idempotent"
    );
}

// --- round trip -----------------------------------------------------------

/// `decode(indices)` must reproduce `encode`'s codes for a single quantizer
/// with no projections, so the two paths are directly comparable.
#[test]
fn test_decode_round_trips_encode() {
    let (residual, client, device) = neucodec_residual();

    let values: Vec<f32> = (0..16).map(|i| (i as f32) * 0.37 - 3.0).collect();
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&values, &[2, 8], &device),
        false,
    );

    let (codes, indices) = residual.encode(&client, &x).unwrap();
    assert_eq!(codes.shape(), &[2, 8]);
    assert_eq!(indices.shape(), &[2, 1]);

    let decoded = residual.decode(&client, &indices).unwrap();
    assert_eq!(decoded.shape(), &[2, 8]);

    let expected: Vec<f32> = codes.tensor().contiguous().unwrap().to_vec();
    let actual: Vec<f32> = decoded.tensor().contiguous().unwrap().to_vec();
    assert_eq!(expected.len(), actual.len());
    for (index, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (e - a).abs() < 1e-5,
            "element {index}: decode gave {a}, encode gave {e}"
        );
    }
}

// --- validation -----------------------------------------------------------

#[test]
fn test_wrong_layer_count_rejected() {
    let (_, device) = cpu_setup();
    let config = ResidualFsqConfig::new(vec![4, 4], 2, 3).unwrap();
    let layer_config = config.layer_config().unwrap();
    let layers = vec![
        Fsq::<CpuRuntime>::new(layer_config.clone(), &device, None, None).unwrap(),
        Fsq::<CpuRuntime>::new(layer_config, &device, None, None).unwrap(),
    ];

    let err = ResidualFsq::new(
        config,
        ResidualFsqWeights {
            project_in: None,
            project_out: None,
            layers,
        },
        &device,
    )
    .err()
    .unwrap();
    assert!(matches!(err, Error::ModelError { .. }), "got {err:?}");
}

#[test]
fn test_mismatched_projection_dims_rejected() {
    let (_, device) = cpu_setup();
    // dim = 5, codebook_dim = 2 -> project_in must be [2, 5], project_out [5, 2].
    let config = ResidualFsqConfig::new(vec![4, 4], 5, 1).unwrap();
    let layer =
        Fsq::<CpuRuntime>::new(config.layer_config().unwrap(), &device, None, None).unwrap();

    // Wrong out-features on project_in: [3, 5] instead of [2, 5].
    let bad_in = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 15], &[3, 5], &device);
    let good_out = Tensor::<CpuRuntime>::from_slice(&[0.2f32; 10], &[5, 2], &device);

    let err = ResidualFsq::new(
        config,
        ResidualFsqWeights {
            project_in: Some(Linear::new(bad_in, None, false)),
            project_out: Some(Linear::new(good_out, None, false)),
            layers: vec![layer],
        },
        &device,
    )
    .err()
    .unwrap();
    assert!(matches!(err, Error::ModelError { .. }), "got {err:?}");
}

#[test]
fn test_missing_projection_rejected_when_dims_differ() {
    let (_, device) = cpu_setup();
    let config = ResidualFsqConfig::new(vec![4, 4], 5, 1).unwrap();
    let layer =
        Fsq::<CpuRuntime>::new(config.layer_config().unwrap(), &device, None, None).unwrap();

    let err = ResidualFsq::new(
        config,
        ResidualFsqWeights {
            project_in: None,
            project_out: None,
            layers: vec![layer],
        },
        &device,
    )
    .err()
    .unwrap();
    assert!(
        matches!(
            err,
            Error::InvalidArgument {
                arg: "project_in/project_out",
                ..
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn test_projecting_inner_layer_rejected() {
    let (_, device) = cpu_setup();
    // Inner layers must be plain FSQ (upstream's are nn.Identity-projected).
    let config = ResidualFsqConfig::new(vec![4, 4], 2, 1).unwrap();
    let w_in = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 10], &[2, 5], &device);
    let w_out = Tensor::<CpuRuntime>::from_slice(&[0.2f32; 10], &[5, 2], &device);
    let projecting_layer = Fsq::<CpuRuntime>::new(
        FsqConfig::new(vec![4, 4], 5).unwrap(),
        &device,
        Some(Linear::new(w_in, None, false)),
        Some(Linear::new(w_out, None, false)),
    )
    .unwrap();

    let err = ResidualFsq::new(
        config,
        ResidualFsqWeights {
            project_in: None,
            project_out: None,
            layers: vec![projecting_layer],
        },
        &device,
    )
    .err()
    .unwrap();
    assert!(matches!(err, Error::ModelError { .. }), "got {err:?}");
}

#[test]
fn test_zero_num_quantizers_rejected() {
    let err = ResidualFsqConfig::new(vec![4, 4], 2, 0).err().unwrap();
    assert!(matches!(
        err,
        Error::InvalidArgument {
            arg: "num_quantizers",
            ..
        }
    ));
}
