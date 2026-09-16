//! BF16/F16 dtype-preservation and f32-equivalence tests for the CPU
//! `FusedOptimizerOps` kernels. Requires the `f16` feature.
#![cfg(feature = "f16")]

mod common;

use boostr::FusedOptimizerOps;
use common::cpu_setup;
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Build a 1-D tensor of `dt` (BF16 or F16) from f32 values.
fn narrow_from_f32(
    vals: &[f32],
    dt: DType,
    device: &numr::runtime::cpu::CpuDevice,
) -> Tensor<CpuRuntime> {
    let shape = [vals.len()];
    match dt {
        DType::BF16 => {
            let d: Vec<half::bf16> = vals.iter().map(|&v| half::bf16::from_f32(v)).collect();
            Tensor::<CpuRuntime>::from_slice(&d, &shape, device).unwrap()
        }
        DType::F16 => {
            let d: Vec<half::f16> = vals.iter().map(|&v| half::f16::from_f32(v)).collect();
            Tensor::<CpuRuntime>::from_slice(&d, &shape, device).unwrap()
        }
        other => panic!("narrow_from_f32: {:?} is not a narrow float", other),
    }
}

/// Read a BF16 or F16 tensor back as f32.
fn narrow_to_f32(t: &Tensor<CpuRuntime>) -> Vec<f32> {
    match t.dtype() {
        DType::BF16 => t
            .to_vec::<half::bf16>()
            .into_iter()
            .map(|v| v.to_f32())
            .collect(),
        DType::F16 => t
            .to_vec::<half::f16>()
            .into_iter()
            .map(|v| v.to_f32())
            .collect(),
        other => panic!("narrow_to_f32: {:?} is not a narrow float", other),
    }
}

const NARROW_DTYPES: [DType; 2] = [DType::BF16, DType::F16];

#[test]
fn test_fused_adamw_accepts_narrow_dtypes() {
    let (client, device) = cpu_setup();

    for dt in NARROW_DTYPES {
        let param = narrow_from_f32(&[1.0, 2.0], dt, &device);
        let grad = narrow_from_f32(&[0.1, 0.2], dt, &device);
        let m = Tensor::<CpuRuntime>::zeros(&[2], dt, &device).unwrap();
        let v = Tensor::<CpuRuntime>::zeros(&[2], dt, &device).unwrap();

        // lr large enough that the step clears the narrow dtype's resolution.
        let lr = 0.1;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let wd = 0.0;
        let step_size = lr * (1.0_f64 - beta2).sqrt() / (1.0 - beta1);

        let (new_p, new_m, new_v) = client
            .fused_adamw_step(&param, &grad, &m, &v, lr, beta1, beta2, eps, wd, step_size)
            .unwrap();

        assert_eq!(new_p.dtype(), dt, "{dt:?}: param dtype must be preserved");
        assert_eq!(new_m.dtype(), dt, "{dt:?}: m dtype must be preserved");
        assert_eq!(new_v.dtype(), dt, "{dt:?}: v dtype must be preserved");

        let p = narrow_to_f32(&new_p);
        assert!(p[0] < 1.0, "{dt:?}: param should decrease, got {}", p[0]);
        assert!(
            narrow_to_f32(&new_m)[0] > 0.0,
            "{dt:?}: m should be positive"
        );
        assert!(
            narrow_to_f32(&new_v)[0] > 0.0,
            "{dt:?}: v should be positive"
        );
    }
}

#[test]
fn test_fused_adamw_narrow_matches_f32_kernel() {
    let (client, device) = cpu_setup();

    // Values exactly representable in BF16, so the only difference between
    // the two runs would be the arithmetic precision — which must be f32.
    let vals = [1.0f32, 2.0];
    let grads = [0.25f32, 0.5];

    let lr = 0.1;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    let wd = 0.0;
    let step_size = lr * (1.0_f64 - beta2).sqrt() / (1.0 - beta1);

    let p32 = Tensor::<CpuRuntime>::from_slice(&vals, &[2], &device).unwrap();
    let g32 = Tensor::<CpuRuntime>::from_slice(&grads, &[2], &device).unwrap();
    let m32 = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device).unwrap();
    let v32 = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device).unwrap();
    let (ref_p, _, _) = client
        .fused_adamw_step(&p32, &g32, &m32, &v32, lr, beta1, beta2, eps, wd, step_size)
        .unwrap();
    let ref_vals = ref_p.to_vec::<f32>();

    let p16 = narrow_from_f32(&vals, DType::BF16, &device);
    let g16 = narrow_from_f32(&grads, DType::BF16, &device);
    let m16 = Tensor::<CpuRuntime>::zeros(&[2], DType::BF16, &device).unwrap();
    let v16 = Tensor::<CpuRuntime>::zeros(&[2], DType::BF16, &device).unwrap();
    let (bf_p, _, _) = client
        .fused_adamw_step(&p16, &g16, &m16, &v16, lr, beta1, beta2, eps, wd, step_size)
        .unwrap();
    let bf_vals = narrow_to_f32(&bf_p);

    for (bf, r) in bf_vals.iter().zip(ref_vals.iter()) {
        assert!(
            (bf - r).abs() < 0.01,
            "BF16 kernel must widen to f32 internally: got {bf} vs f32 {r}"
        );
    }
}

#[test]
fn test_fused_sgd_accepts_narrow_dtypes() {
    let (client, device) = cpu_setup();

    for dt in NARROW_DTYPES {
        let param = narrow_from_f32(&[1.0, 2.0], dt, &device);
        let grad = narrow_from_f32(&[0.1, 0.2], dt, &device);

        let (new_p, new_buf) = client
            .fused_sgd_step(&param, &grad, None, 0.1, 0.0, 0.0, 0.0, false)
            .unwrap();

        assert_eq!(new_p.dtype(), dt, "{dt:?}: param dtype must be preserved");
        assert_eq!(
            new_buf.dtype(),
            dt,
            "{dt:?}: buffer dtype must be preserved"
        );

        let p = narrow_to_f32(&new_p);
        assert!((p[0] - 0.99).abs() < 0.01, "{dt:?}: got {}", p[0]);
        assert!((p[1] - 1.98).abs() < 0.02, "{dt:?}: got {}", p[1]);
    }
}

#[test]
fn test_fused_adagrad_accepts_narrow_dtypes() {
    let (client, device) = cpu_setup();

    for dt in NARROW_DTYPES {
        let param = narrow_from_f32(&[1.0, 2.0], dt, &device);
        let grad = narrow_from_f32(&[0.1, 0.2], dt, &device);
        let accum = Tensor::<CpuRuntime>::zeros(&[2], dt, &device).unwrap();

        let (new_p, new_acc) = client
            .fused_adagrad_step(&param, &grad, &accum, 0.1, 1e-10, 0.0)
            .unwrap();

        assert_eq!(new_p.dtype(), dt, "{dt:?}: param dtype must be preserved");
        assert_eq!(new_acc.dtype(), dt, "{dt:?}: accum dtype must be preserved");

        let p = narrow_to_f32(&new_p);
        assert!((p[0] - 0.9).abs() < 0.01, "{dt:?}: got {}", p[0]);
        assert!(narrow_to_f32(&new_acc)[0] > 0.0, "{dt:?}: accum must grow");
    }
}

#[test]
fn test_fused_lamb_accepts_narrow_dtypes() {
    let (client, device) = cpu_setup();

    for dt in NARROW_DTYPES {
        let param = narrow_from_f32(&[1.0, 2.0], dt, &device);
        let grad = narrow_from_f32(&[0.1, 0.2], dt, &device);
        let m = Tensor::<CpuRuntime>::zeros(&[2], dt, &device).unwrap();
        let v = Tensor::<CpuRuntime>::zeros(&[2], dt, &device).unwrap();

        let (update, new_m, new_v) = client
            .fused_lamb_step(&param, &grad, &m, &v, 0.9, 0.999, 1e-6, 0.0, 0.1, 0.001)
            .unwrap();

        assert_eq!(update.dtype(), dt, "{dt:?}: update dtype must match param");
        assert_eq!(new_m.dtype(), dt, "{dt:?}: m dtype must be preserved");
        assert_eq!(new_v.dtype(), dt, "{dt:?}: v dtype must be preserved");
        assert!(
            narrow_to_f32(&update)[0] > 0.0,
            "{dt:?}: update must be positive"
        );
    }
}

#[test]
fn test_fused_multi_tensor_adamw_accepts_bf16() {
    let (client, device) = cpu_setup();

    let p1 = narrow_from_f32(&[1.0, 2.0], DType::BF16, &device);
    let g1 = narrow_from_f32(&[0.1, 0.2], DType::BF16, &device);
    let m1 = Tensor::<CpuRuntime>::zeros(&[2], DType::BF16, &device).unwrap();
    let v1 = Tensor::<CpuRuntime>::zeros(&[2], DType::BF16, &device).unwrap();

    let lr = 0.1;
    let step_size = lr * (1.0_f64 - 0.999).sqrt() / (1.0 - 0.9);
    let groups = vec![(&p1, &g1, &m1, &v1)];

    let results = client
        .fused_multi_tensor_adamw(&groups, lr, 0.9, 0.999, 1e-8, 0.0, step_size)
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.dtype(), DType::BF16);
    assert!(narrow_to_f32(&results[0].0)[0] < 1.0);
}
