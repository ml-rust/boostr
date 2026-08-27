use super::*;
use crate::nn::moe::router::MoeRouterConfig;
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::CpuRuntime;

fn experts(
    num_experts: usize,
    hidden: usize,
    inter: usize,
    device: &<CpuRuntime as Runtime>::Device,
) -> Vec<Expert<CpuRuntime>> {
    (0..num_experts)
        .map(|expert_idx| {
            let scale = 0.05f32 + expert_idx as f32 * 0.02;
            let gw = Tensor::<CpuRuntime>::from_slice(
                &vec![scale; inter * hidden],
                &[inter, hidden],
                device,
            )
            .unwrap();
            let uw = Tensor::<CpuRuntime>::from_slice(
                &vec![scale + 0.01; inter * hidden],
                &[inter, hidden],
                device,
            )
            .unwrap();
            let dw = Tensor::<CpuRuntime>::from_slice(
                &vec![scale - 0.01; hidden * inter],
                &[hidden, inter],
                device,
            )
            .unwrap();
            Expert::from_tensors(gw, uw, dw, false)
        })
        .collect()
}

#[test]
fn test_moe_layer_forward_shape() {
    let (client, device) = cpu_setup();
    let hidden = 4;
    let inter = 8;
    let num_experts = 2;
    let top_k = 1;

    let gate_w =
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[num_experts, hidden], &device).unwrap();
    let config = MoeRouterConfig::new(num_experts, top_k);
    let router = MoeRouter::from_tensor(gate_w, config, false);

    let layer = MoeLayer::new(router, experts(num_experts, hidden, inter, &device), None);

    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[3, hidden], &device).unwrap(),
        false,
    );
    let result = layer.forward(&client, &input).unwrap();

    assert_eq!(result.output.shape(), &[3, hidden]);
    assert_eq!(result.z_loss.tensor().numel(), 1);
}

/// The shared expert is active for every token, so its contribution must be
/// weightable. Scaling by `s` must shift the output by exactly `(s - 1)`
/// times the shared expert's own output — not merely "change it".
#[test]
fn shared_expert_scale_weights_the_shared_contribution() {
    fn run(scale: Option<f32>) -> Vec<f32> {
        let (client, device) = cpu_setup();
        let (hidden, inter, num_experts, top_k) = (4, 8, 2, 1);

        let gate_w =
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[num_experts, hidden], &device)
                .unwrap();
        let router =
            MoeRouter::from_tensor(gate_w, MoeRouterConfig::new(num_experts, top_k), false);
        let shared = experts(1, hidden, inter, &device).pop();

        let mut layer = MoeLayer::new(router, experts(num_experts, hidden, inter, &device), shared);
        if let Some(scale) = scale {
            layer = layer.with_shared_expert_scale(scale).unwrap();
        }

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[3, hidden], &device).unwrap(),
            false,
        );
        let out = layer.forward(&client, &input).unwrap();
        out.output.tensor().contiguous().unwrap().to_vec()
    }

    // Baseline: no shared expert at all, so only routed experts contribute.
    let routed_only = {
        let (client, device) = cpu_setup();
        let (hidden, inter, num_experts, top_k) = (4, 8, 2, 1);
        let gate_w =
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[num_experts, hidden], &device)
                .unwrap();
        let router =
            MoeRouter::from_tensor(gate_w, MoeRouterConfig::new(num_experts, top_k), false);
        let layer = MoeLayer::new(router, experts(num_experts, hidden, inter, &device), None);
        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 12], &[3, hidden], &device).unwrap(),
            false,
        );
        let out = layer.forward(&client, &input).unwrap();
        out.output.tensor().contiguous().unwrap().to_vec::<f32>()
    };

    let unscaled = run(None);
    let half = run(Some(0.5));

    for i in 0..unscaled.len() {
        // Default stays exactly as before this option existed.
        let shared_contribution = unscaled[i] - routed_only[i];
        assert!(
            shared_contribution.abs() > 1e-6,
            "test setup is degenerate: shared expert contributes nothing"
        );
        // Scaling by 0.5 must halve precisely that contribution.
        let expected = routed_only[i] + 0.5 * shared_contribution;
        assert!(
            (half[i] - expected).abs() < 1e-5,
            "index {i}: expected {expected}, got {}",
            half[i]
        );
    }
}

#[test]
fn shared_expert_scale_rejects_non_finite() {
    let (_client, device) = cpu_setup();
    let gate_w = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[2, 4], &device).unwrap();
    let router = MoeRouter::from_tensor(gate_w, MoeRouterConfig::new(2, 1), false);
    let layer = MoeLayer::new(router, experts(2, 4, 8, &device), None);
    assert!(layer.with_shared_expert_scale(f32::NAN).is_err());
}

#[test]
fn z_loss_produces_gate_gradient() {
    let (client, device) = cpu_setup();
    let hidden = 4;
    let inter = 8;
    let num_experts = 3;
    let top_k = 2;

    // Asymmetric weights and inputs keep this from passing by accidental
    // symmetry if z_loss ever stops being connected to the gate.
    let gate_w = Tensor::<CpuRuntime>::from_slice(
        &[
            0.7f32, -0.2, 0.15, 0.4, -0.35, 0.6, 0.25, -0.1, 0.05, -0.45, 0.8, 0.3,
        ],
        &[num_experts, hidden],
        &device,
    )
    .unwrap();
    let router = MoeRouter::from_tensor(gate_w, MoeRouterConfig::new(num_experts, top_k), true);
    let layer = MoeLayer::new(router, experts(num_experts, hidden, inter, &device), None);
    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(
            &[
                0.3f32, -0.7, 1.1, 0.2, 0.8, 0.4, -0.3, 0.9, -0.6, 0.5, 0.7, -0.2,
            ],
            &[3, hidden],
            &device,
        )
        .unwrap(),
        false,
    );

    let result = layer.forward(&client, &input).unwrap();
    let grads = numr::autograd::backward(&result.z_loss, &client).unwrap();
    let gate_id = layer.router().gate().parameters()[0].0;
    let gate_grad = grads
        .get(gate_id)
        .expect("z_loss must produce a gradient for the gate weight")
        .contiguous()
        .unwrap();
    let magnitude: f32 = gate_grad.to_vec::<f32>().iter().map(|v| v.abs()).sum();
    assert!(
        magnitude > 1e-8,
        "gate gradient from z_loss is all zeros ({magnitude}) — graph is severed"
    );
}
