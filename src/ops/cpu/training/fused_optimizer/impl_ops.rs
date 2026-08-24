//! CPU implementation of FusedOptimizerOps
//!
//! Single-pass parameter updates using raw pointer arithmetic.
//! Each fused kernel reads all inputs and writes all outputs in one loop,
//! reducing memory traffic by 4-8x vs composing individual ops.

use crate::error::{Error, Result};
use crate::ops::traits::FusedOptimizerOps;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use super::{adagrad, adamw, lamb, sgd};

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
impl FusedOptimizerOps<CpuRuntime> for CpuClient {
    fn fused_adamw_step(
        &self,
        param: &Tensor<CpuRuntime>,
        grad: &Tensor<CpuRuntime>,
        m: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
        step_size: f64,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        validate_shapes(param, grad, "fused_adamw_step")?;
        validate_shapes(param, m, "fused_adamw_step")?;
        validate_shapes(param, v, "fused_adamw_step")?;

        match param.dtype() {
            DType::F32 => {
                adamw::fused_adamw_f32(param, grad, m, v, lr, beta1, beta2, eps, wd, step_size)
            }
            DType::F64 => {
                adamw::fused_adamw_f64(param, grad, m, v, lr, beta1, beta2, eps, wd, step_size)
            }
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => {
                adamw::fused_adamw_narrow(param, grad, m, v, lr, beta1, beta2, eps, wd, step_size)
            }
            dt => Err(Error::InvalidArgument {
                arg: "dtype",
                reason: unsupported_dtype_reason("fused_adamw_step", dt),
            }),
        }
    }

    fn fused_sgd_step(
        &self,
        param: &Tensor<CpuRuntime>,
        grad: &Tensor<CpuRuntime>,
        momentum_buf: Option<&Tensor<CpuRuntime>>,
        lr: f64,
        momentum: f64,
        dampening: f64,
        wd: f64,
        nesterov: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        validate_shapes(param, grad, "fused_sgd_step")?;
        if let Some(buf) = momentum_buf {
            validate_shapes(param, buf, "fused_sgd_step")?;
        }

        match param.dtype() {
            DType::F32 => sgd::fused_sgd_f32(
                param,
                grad,
                momentum_buf,
                lr,
                momentum,
                dampening,
                wd,
                nesterov,
            ),
            DType::F64 => sgd::fused_sgd_f64(
                param,
                grad,
                momentum_buf,
                lr,
                momentum,
                dampening,
                wd,
                nesterov,
            ),
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => sgd::fused_sgd_narrow(
                param,
                grad,
                momentum_buf,
                lr,
                momentum,
                dampening,
                wd,
                nesterov,
            ),
            dt => Err(Error::InvalidArgument {
                arg: "dtype",
                reason: unsupported_dtype_reason("fused_sgd_step", dt),
            }),
        }
    }

    fn fused_adagrad_step(
        &self,
        param: &Tensor<CpuRuntime>,
        grad: &Tensor<CpuRuntime>,
        accum: &Tensor<CpuRuntime>,
        lr: f64,
        eps: f64,
        wd: f64,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        validate_shapes(param, grad, "fused_adagrad_step")?;
        validate_shapes(param, accum, "fused_adagrad_step")?;

        match param.dtype() {
            DType::F32 => adagrad::fused_adagrad_f32(param, grad, accum, lr, eps, wd),
            DType::F64 => adagrad::fused_adagrad_f64(param, grad, accum, lr, eps, wd),
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => {
                adagrad::fused_adagrad_narrow(param, grad, accum, lr, eps, wd)
            }
            dt => Err(Error::InvalidArgument {
                arg: "dtype",
                reason: unsupported_dtype_reason("fused_adagrad_step", dt),
            }),
        }
    }

    fn fused_lamb_step(
        &self,
        param: &Tensor<CpuRuntime>,
        grad: &Tensor<CpuRuntime>,
        m: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
        bias_corr1: f64,
        bias_corr2: f64,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        validate_shapes(param, grad, "fused_lamb_step")?;
        validate_shapes(param, m, "fused_lamb_step")?;
        validate_shapes(param, v, "fused_lamb_step")?;

        match param.dtype() {
            DType::F32 => lamb::fused_lamb_f32(
                param, grad, m, v, beta1, beta2, eps, wd, bias_corr1, bias_corr2,
            ),
            DType::F64 => lamb::fused_lamb_f64(
                param, grad, m, v, beta1, beta2, eps, wd, bias_corr1, bias_corr2,
            ),
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => lamb::fused_lamb_narrow(
                param, grad, m, v, beta1, beta2, eps, wd, bias_corr1, bias_corr2,
            ),
            dt => Err(Error::InvalidArgument {
                arg: "dtype",
                reason: unsupported_dtype_reason("fused_lamb_step", dt),
            }),
        }
    }

    fn fused_multi_tensor_adamw(
        &self,
        groups: &[(
            &Tensor<CpuRuntime>,
            &Tensor<CpuRuntime>,
            &Tensor<CpuRuntime>,
            &Tensor<CpuRuntime>,
        )],
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
        step_size: f64,
    ) -> Result<Vec<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>> {
        groups
            .iter()
            .map(|(param, grad, m, v)| {
                self.fused_adamw_step(param, grad, m, v, lr, beta1, beta2, eps, wd, step_size)
            })
            .collect()
    }
}

/// Error text for a dtype no CPU fused-optimizer arm handles.
///
/// BF16 and F16 are handled, but only in a build with the `f16` feature: numr
/// cannot allocate or convert those dtypes without it. Say so, rather than
/// reporting them as plain "unsupported".
fn unsupported_dtype_reason(op: &str, dt: DType) -> String {
    match dt {
        DType::F16 | DType::BF16 => format!(
            "{}: dtype {:?} requires the `f16` feature (build boostr with --features f16)",
            op, dt
        ),
        _ => format!("{}: unsupported dtype {:?}", op, dt),
    }
}

fn validate_shapes(a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>, op: &str) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(Error::InvalidArgument {
            arg: "shape",
            reason: format!("{}: shape mismatch {:?} vs {:?}", op, a.shape(), b.shape()),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;

    #[test]
    fn test_fused_adamw_basic() {
        let (client, device) = cpu_setup();
        let param =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
        let grad =
            Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[4], &device).unwrap();
        let m = Tensor::<CpuRuntime>::zeros(&[4], DType::F32, &device).unwrap();
        let v = Tensor::<CpuRuntime>::zeros(&[4], DType::F32, &device).unwrap();

        let lr = 1e-3;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let wd = 0.01;
        let bc1 = 1.0 - beta1;
        let bc2 = (1.0_f64 - beta2).sqrt();
        let step_size = lr * bc2 / bc1;

        let (new_p, new_m, new_v) = client
            .fused_adamw_step(&param, &grad, &m, &v, lr, beta1, beta2, eps, wd, step_size)
            .unwrap();

        let p_data = new_p.to_vec::<f32>();
        assert!(p_data[0] < 1.0, "param should decrease: {}", p_data[0]);
        assert!(new_m.to_vec::<f32>()[0] > 0.0, "m should be positive");
        assert!(new_v.to_vec::<f32>()[0] > 0.0, "v should be positive");
    }

    #[test]
    fn test_fused_sgd_basic() {
        let (client, device) = cpu_setup();
        let param = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let grad = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device).unwrap();

        let (new_p, _buf) = client
            .fused_sgd_step(&param, &grad, None, 0.1, 0.0, 0.0, 0.0, false)
            .unwrap();

        let p = new_p.to_vec::<f32>();
        assert!((p[0] - 0.99).abs() < 1e-6);
        assert!((p[1] - 1.98).abs() < 1e-6);
    }

    #[test]
    fn test_fused_multi_tensor_adamw() {
        let (client, device) = cpu_setup();

        let p1 = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let g1 = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device).unwrap();
        let m1 = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device).unwrap();
        let v1 = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device).unwrap();

        let p2 = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0, 5.0], &[3], &device).unwrap();
        let g2 = Tensor::<CpuRuntime>::from_slice(&[0.3f32, 0.4, 0.5], &[3], &device).unwrap();
        let m2 = Tensor::<CpuRuntime>::zeros(&[3], DType::F32, &device).unwrap();
        let v2 = Tensor::<CpuRuntime>::zeros(&[3], DType::F32, &device).unwrap();

        let lr = 1e-3;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let wd = 0.01;
        let bc1 = 1.0 - beta1;
        let bc2 = (1.0_f64 - beta2).sqrt();
        let step_size = lr * bc2 / bc1;

        let groups = vec![(&p1, &g1, &m1, &v1), (&p2, &g2, &m2, &v2)];

        let results = client
            .fused_multi_tensor_adamw(&groups, lr, beta1, beta2, eps, wd, step_size)
            .unwrap();

        assert_eq!(results.len(), 2);

        // Verify results match individual calls
        let (ref_p1, ref_m1, ref_v1) = client
            .fused_adamw_step(&p1, &g1, &m1, &v1, lr, beta1, beta2, eps, wd, step_size)
            .unwrap();
        let (ref_p2, ref_m2, ref_v2) = client
            .fused_adamw_step(&p2, &g2, &m2, &v2, lr, beta1, beta2, eps, wd, step_size)
            .unwrap();

        assert_eq!(results[0].0.to_vec::<f32>(), ref_p1.to_vec::<f32>());
        assert_eq!(results[0].1.to_vec::<f32>(), ref_m1.to_vec::<f32>());
        assert_eq!(results[0].2.to_vec::<f32>(), ref_v1.to_vec::<f32>());
        assert_eq!(results[1].0.to_vec::<f32>(), ref_p2.to_vec::<f32>());
        assert_eq!(results[1].1.to_vec::<f32>(), ref_m2.to_vec::<f32>());
        assert_eq!(results[1].2.to_vec::<f32>(), ref_v2.to_vec::<f32>());
    }

    /// Build a 1-D tensor of `dt` (BF16 or F16) from f32 values.
    #[cfg(feature = "f16")]
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
    #[cfg(feature = "f16")]
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

    #[cfg(feature = "f16")]
    const NARROW_DTYPES: [DType; 2] = [DType::BF16, DType::F16];

    #[cfg(feature = "f16")]
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

    #[cfg(feature = "f16")]
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

    #[cfg(feature = "f16")]
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

    #[cfg(feature = "f16")]
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

    #[cfg(feature = "f16")]
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

    #[cfg(feature = "f16")]
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

    #[test]
    fn test_fused_adagrad_basic() {
        let (client, device) = cpu_setup();
        let param = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let grad = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device).unwrap();
        let accum = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device).unwrap();

        let (new_p, new_acc) = client
            .fused_adagrad_step(&param, &grad, &accum, 0.1, 1e-10, 0.0)
            .unwrap();

        let p = new_p.to_vec::<f32>();
        assert!((p[0] - 0.9).abs() < 1e-4);
        assert!((p[1] - 1.9).abs() < 1e-4);
        assert!(new_acc.to_vec::<f32>()[0] > 0.0);
    }
}
