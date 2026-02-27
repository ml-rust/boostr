//! SSM scan algorithms for inference (raw Tensor, no autograd).

use super::types::SsmInferenceInput;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{ActivationOps, ScalarOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Single SSM recurrence step on raw tensors (no autograd overhead).
///
/// Takes current hidden state `h: [B, nheads, headdim, d_state]` and single-timestep
/// inputs, returns `(y_t: [B, 1, nheads, headdim], h_new)`.
pub fn ssm_step_inference<R, C>(
    _client: &C,
    input: &SsmInferenceInput<'_, R>,
    h: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + UnaryOps<R> + ActivationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + numr::ops::BinaryOps<R>,
{
    let shape = input.x.shape();
    // x: [B, 1, nheads, headdim], dt: [B, 1, nheads], b: [B, 1, ngroups, d_state], c: same
    let batch = shape[0];
    let nheads = input.config.nheads;
    let headdim = input.config.headdim;
    let d_state = input.config.d_state;
    let ngroups = input.config.ngroups;

    // Squeeze time dim: x_t [B, nheads, headdim]
    let x_t = input
        .x
        .reshape(&[batch, nheads, headdim])
        .map_err(Error::Numr)?;
    let x_t = x_t.contiguous();

    // dt_t: [B, nheads, 1, 1]
    let dt_t = input
        .dt
        .reshape(&[batch, nheads, 1, 1])
        .map_err(Error::Numr)?;
    let dt_t = dt_t.contiguous();

    // B_t: [B, ngroups, 1, d_state]
    let b_t = input
        .b
        .reshape(&[batch, ngroups, 1, d_state])
        .map_err(Error::Numr)?;
    let b_t = b_t.contiguous();

    // C_t: [B, ngroups, d_state]
    let c_t = input
        .c
        .reshape(&[batch, ngroups, d_state])
        .map_err(Error::Numr)?;
    let c_t = c_t.contiguous();

    // A: [nheads] -> [1, nheads, 1, 1]
    let a_broad = input.a.reshape(&[1, nheads, 1, 1]).map_err(Error::Numr)?;

    // decay = exp(dt_t * A): [B, nheads, 1, 1]
    let dt_a = dt_t.mul(&a_broad).map_err(Error::Numr)?;
    let decay = dt_a.exp().map_err(Error::Numr)?;

    // h = decay * h + dt_t * B_t * x_t
    let mut h_new = decay.mul(h).map_err(Error::Numr)?;

    let b_t_expanded = if ngroups == nheads {
        b_t.reshape(&[batch, nheads, 1, d_state])
            .map_err(Error::Numr)?
    } else {
        b_t // [B, ngroups(=1), 1, d_state] broadcasts
    };

    let x_t_col = x_t
        .reshape(&[batch, nheads, headdim, 1])
        .map_err(Error::Numr)?;
    let dt_x = dt_t.mul(&x_t_col).map_err(Error::Numr)?;
    let input_term = dt_x.mul(&b_t_expanded).map_err(Error::Numr)?;
    h_new = h_new.add(&input_term).map_err(Error::Numr)?;

    // y_t = h @ C_t: [B, nheads, headdim, d_state] @ [B, ngroups, d_state, 1]
    let c_t_col = c_t
        .reshape(&[batch, ngroups, d_state, 1])
        .map_err(Error::Numr)?;
    let y_t = h_new.matmul(&c_t_col).map_err(Error::Numr)?;
    let mut y_t = y_t
        .reshape(&[batch, nheads, headdim])
        .map_err(Error::Numr)?;

    // D skip connection
    if let Some(d_param) = input.d_param {
        let d_broad = d_param.reshape(&[1, nheads, 1]).map_err(Error::Numr)?;
        let d_x = d_broad.mul(&x_t).map_err(Error::Numr)?;
        y_t = y_t.add(&d_x).map_err(Error::Numr)?;
    }

    // y_t: [B, 1, nheads, headdim]
    let y_t = y_t
        .reshape(&[batch, 1, nheads, headdim])
        .map_err(Error::Numr)?;
    Ok((y_t, h_new))
}

/// Sequential SSM scan on raw tensors for inference prefill.
///
/// Processes a full sequence and returns `(output: [B, S, nheads, headdim], final_h)`.
pub fn ssm_forward_sequential_inference<R, C>(
    client: &C,
    input: &SsmInferenceInput<'_, R>,
    h_init: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + UnaryOps<R> + ActivationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + numr::ops::BinaryOps<R> + numr::ops::ShapeOps<R>,
{
    let shape = input.x.shape();
    let seq_len = shape[1];

    let mut h = h_init.clone();
    let mut outputs: Vec<Tensor<R>> = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        // Slice single timestep
        let x_t = input.x.narrow(1, t, 1).map_err(Error::Numr)?.contiguous();
        let b_t = input.b.narrow(1, t, 1).map_err(Error::Numr)?.contiguous();
        let c_t = input.c.narrow(1, t, 1).map_err(Error::Numr)?.contiguous();
        let dt_t = input.dt.narrow(1, t, 1).map_err(Error::Numr)?.contiguous();

        let step_input = SsmInferenceInput {
            x: &x_t,
            a: input.a,
            b: &b_t,
            c: &c_t,
            d_param: input.d_param,
            dt: &dt_t,
            config: input.config,
        };

        let (y_t, h_new) = ssm_step_inference(client, &step_input, &h)?;
        h = h_new;
        outputs.push(y_t);
    }

    // Concatenate along dim 1: [B, S, nheads, headdim]
    let output_refs: Vec<&Tensor<R>> = outputs.iter().collect();
    let output = Tensor::cat(&output_refs, 1).map_err(Error::Numr)?;
    Ok((output, h))
}

#[cfg(test)]
mod tests {
    use super::super::scan::ssm_forward_sequential;
    use super::super::types::SsmInput;
    use super::*;
    use crate::model::mamba::mamba2::Mamba2Config;
    use crate::test_utils::cpu_setup;
    use numr::autograd::Var;
    use numr::dtype::DType;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_ssm_step_inference_matches_var() {
        let (client, device) = cpu_setup();
        let config = Mamba2Config::new(4)
            .with_nheads(1)
            .with_d_state(2)
            .with_expand(1)
            .with_use_d(true)
            .with_dt_softplus(false)
            .with_use_dt_bias(false);

        let x_data = [1.0f32, 0.5, 0.0, -0.5];
        let x_t = Tensor::<CpuRuntime>::from_slice(&x_data, &[1, 1, 1, 4], &device);
        let a_t = Tensor::<CpuRuntime>::from_slice(&[-1.0f32], &[1], &device);
        let b_t = Tensor::<CpuRuntime>::from_slice(&[0.5f32, 0.3], &[1, 1, 1, 2], &device);
        let c_t = Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.8], &[1, 1, 1, 2], &device);
        let d_p = Tensor::<CpuRuntime>::from_slice(&[0.1f32], &[1], &device);
        let dt_t = Tensor::<CpuRuntime>::from_slice(&[0.5f32], &[1, 1, 1], &device);
        let h = Tensor::<CpuRuntime>::zeros(&[1, 1, 4, 2], DType::F32, &device);

        let input = SsmInferenceInput {
            x: &x_t,
            a: &a_t,
            b: &b_t,
            c: &c_t,
            d_param: Some(&d_p),
            dt: &dt_t,
            config: &config,
        };

        let (y, h_new) = ssm_step_inference(&client, &input, &h).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1, 4]);
        assert_eq!(h_new.shape(), &[1, 1, 4, 2]);

        let y_data: Vec<f32> = y.to_vec();
        assert!(y_data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_sequential_inference_matches_var() {
        let (client, device) = cpu_setup();
        let config = Mamba2Config::new(4)
            .with_nheads(1)
            .with_d_state(2)
            .with_expand(1)
            .with_use_d(false)
            .with_dt_softplus(false)
            .with_use_dt_bias(false);

        let x_data: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let a_data = [-1.0f32];
        let b_data = [1.0f32, 0.0, 0.0, 1.0];
        let c_data = [1.0f32, 1.0, 1.0, 1.0];
        let dt_data = [0.5f32, 0.5];

        // Var-based (existing)
        let x_var = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[1, 2, 1, 4], &device),
            false,
        );
        let a_var = Var::new(
            Tensor::<CpuRuntime>::from_slice(&a_data, &[1], &device),
            false,
        );
        let b_var = Var::new(
            Tensor::<CpuRuntime>::from_slice(&b_data, &[1, 2, 1, 2], &device),
            false,
        );
        let c_var = Var::new(
            Tensor::<CpuRuntime>::from_slice(&c_data, &[1, 2, 1, 2], &device),
            false,
        );
        let dt_var = Var::new(
            Tensor::<CpuRuntime>::from_slice(&dt_data, &[1, 2, 1], &device),
            false,
        );

        let var_input = SsmInput {
            x: &x_var,
            a: &a_var,
            b: &b_var,
            c: &c_var,
            d_param: None,
            dt: &dt_var,
            config: &config,
        };
        let out_var = ssm_forward_sequential(&client, &var_input).unwrap();

        // Tensor-based (inference)
        let x_t = Tensor::<CpuRuntime>::from_slice(&x_data, &[1, 2, 1, 4], &device);
        let a_t = Tensor::<CpuRuntime>::from_slice(&a_data, &[1], &device);
        let b_t = Tensor::<CpuRuntime>::from_slice(&b_data, &[1, 2, 1, 2], &device);
        let c_t = Tensor::<CpuRuntime>::from_slice(&c_data, &[1, 2, 1, 2], &device);
        let dt_t = Tensor::<CpuRuntime>::from_slice(&dt_data, &[1, 2, 1], &device);
        let h_init = Tensor::<CpuRuntime>::zeros(&[1, 1, 4, 2], DType::F32, &device);

        let inf_input = SsmInferenceInput {
            x: &x_t,
            a: &a_t,
            b: &b_t,
            c: &c_t,
            d_param: None,
            dt: &dt_t,
            config: &config,
        };
        let (out_inf, _) = ssm_forward_sequential_inference(&client, &inf_input, &h_init).unwrap();

        let var_data: Vec<f32> = out_var.tensor().to_vec();
        let inf_data: Vec<f32> = out_inf.to_vec();
        assert_eq!(var_data.len(), inf_data.len());
        for (i, (v, t)) in var_data.iter().zip(inf_data.iter()).enumerate() {
            assert!((v - t).abs() < 1e-5, "mismatch at {i}: var={v}, inf={t}");
        }
    }
}
