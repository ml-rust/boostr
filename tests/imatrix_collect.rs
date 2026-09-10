//! End-to-end importance collection over `MaybeQuantLinear`, in a process of
//! its own.
//!
//! The collector is process-wide by design — one armed collection at a time,
//! owned by the thread that armed it — so this lives in its own integration
//! binary rather than in the unit suite, where another test's forward pass on
//! another thread would run while it was armed.
//!
//! Cargo runs the `#[test]` functions of ONE binary concurrently, so this file
//! holds exactly one test and drives every step of the flow in sequence.

use boostr::nn::{VarBuilder, VarMap};
use boostr::quant::imatrix;
use numr::autograd::Var;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

const WEIGHT_NAME: &str = "model.layers.0.mlp.down_proj.weight";

#[test]
fn collection_sums_squared_activations_per_column_under_the_checkpoint_name() {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    // Nothing is collected, and nothing is bound, while disarmed.
    assert!(!imatrix::is_armed());
    imatrix::register_name(numr::tensor::TensorId::new(), "ignored.weight");
    assert!(imatrix::registered_names().is_empty());

    // ARM BEFORE LOADING: the checkpoint name is bound by the `VarBuilder`
    // that reads the weight, which is the only place it exists without being
    // inferred from the model's structure.
    imatrix::arm().expect("collector arms");
    assert!(imatrix::is_armed());
    assert!(
        imatrix::arm().is_err(),
        "a second concurrent collection must be refused"
    );

    // A [2, 3] weight: 3 input columns, so the importance vector is 3 long.
    let mut var_map = VarMap::<CpuRuntime>::new();
    var_map.insert(
        WEIGHT_NAME.to_string(),
        Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 3], &device).expect("weight"),
    );
    // Each `pp` borrows its parent, so the prefix is built one binding at a
    // time; the full key the loader reads is
    // `model.layers.0.mlp.down_proj.weight`.
    let mut vb = VarBuilder::new(&mut var_map, &device);
    let mut model_vb = vb.pp("model");
    let mut layers_vb = model_vb.pp("layers");
    let mut layer0_vb = layers_vb.pp("0");
    let mut mlp_vb = layer0_vb.pp("mlp");
    let layer = mlp_vb
        .take_maybe_quant_linear("down_proj.weight", None)
        .expect("layer loads");
    assert_eq!(imatrix::registered_names(), vec![WEIGHT_NAME.to_string()]);

    // Two windows of two tokens each, so the sums accumulate across calls and
    // the row count adds up.
    let first =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], &device)
            .expect("first window");
    let second =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 2.0, 2.0, 2.0], &[1, 2, 3], &device)
            .expect("second window");
    for input in [first, second] {
        layer
            .forward(&client, &Var::new(input, false))
            .expect("forward");
    }

    let matrix = imatrix::finish::<CpuRuntime>(4).expect("collection finishes");
    assert!(!imatrix::is_armed(), "finish disarms the collector");
    assert_eq!(matrix.token_count(), 4);
    assert_eq!(matrix.len(), 1);

    let entry = matrix
        .get(WEIGHT_NAME)
        .expect("entry under the loader's key");
    assert_eq!(entry.in_features(), 3);
    assert_eq!(entry.rows, 4);
    // Column j: 1+16+1+4, 4+25+1+4, 9+36+1+4.
    assert_eq!(entry.sums, vec![22.0, 34.0, 50.0]);
    assert_eq!(entry.mean_square(), Some(vec![5.5, 8.5, 12.5]));

    // A weight this run never exercised has no entry at all — never a zero
    // vector, which would read as "measured, and unimportant".
    assert!(matrix.get("model.layers.0.mlp.up_proj.weight").is_none());

    // The file round-trips, and a width disagreement is refused rather than
    // read at the wrong shape.
    let parsed =
        boostr::quant::ImportanceMatrix::from_bytes(&matrix.to_bytes()).expect("file parses");
    assert_eq!(parsed, matrix);
    assert!(
        parsed
            .check_against(&[(WEIGHT_NAME.to_string(), 4)])
            .is_err()
    );
    let check = parsed
        .check_against(&[
            (WEIGHT_NAME.to_string(), 3),
            ("model.layers.0.mlp.up_proj.weight".to_string(), 3),
        ])
        .expect("matching widths");
    assert_eq!(check.present, vec![WEIGHT_NAME.to_string()]);
    assert_eq!(check.absent, vec!["model.layers.0.mlp.up_proj.weight"]);
    assert!(check.unknown.is_empty());

    // Disarmed again, an ordinary forward pass contributes nothing.
    let after = Tensor::<CpuRuntime>::from_slice(&[9.0f32; 3], &[1, 3], &device).expect("after");
    layer
        .forward(&client, &Var::new(after, false))
        .expect("forward while disarmed");
    assert!(imatrix::registered_names().is_empty());
}
