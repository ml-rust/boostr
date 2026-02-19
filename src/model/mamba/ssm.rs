//! SSM (Structured State Space) scan algorithms for Mamba2.
//!
//! Provides sequential scan and chunked SSD forward passes,
//! composed entirely from numr autograd primitives.

use crate::error::{Error, Result};
use crate::model::mamba::mamba2::Mamba2Config;
use numr::autograd::{
    Var, var_add, var_cat, var_exp, var_matmul, var_mul, var_narrow, var_reshape,
};
use numr::dtype::DType;
use numr::ops::{ActivationOps, ReduceOps, ScalarOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Bundled SSM inputs to reduce parameter count.
pub struct SsmInput<'a, R: Runtime> {
    pub x: &'a Var<R>,
    pub a: &'a Var<R>,
    pub b: &'a Var<R>,
    pub c: &'a Var<R>,
    pub d_param: Option<&'a Var<R>>,
    pub dt: &'a Var<R>,
    pub config: &'a Mamba2Config,
}

/// Make a Var contiguous (copies data if non-contiguous).
/// Preserves the requires_grad flag but not the grad_fn (treated as a leaf).
pub fn var_contiguous<R: Runtime>(v: &Var<R>) -> Var<R> {
    if v.tensor().is_contiguous() {
        v.clone()
    } else {
        Var::new(v.tensor().contiguous(), v.requires_grad())
    }
}

/// Sequential SSM scan: for each position t:
///   h[t] = exp(dt[t] * A) * h[t-1] + dt[t] * B[t] * x[t]
///   y[t] = (C[t] @ h[t]) + D * x[t]
///
/// x: [B, S, nheads, headdim], A: [nheads], B: [B, S, ngroups, d_state],
/// C: [B, S, ngroups, d_state], dt: [B, S, nheads], D: [nheads] (optional)
pub fn ssm_forward_sequential<R, C>(client: &C, input: &SsmInput<'_, R>) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + UnaryOps<R> + ActivationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let shape = input.x.shape();
    let batch = shape[0];
    let seq_len = shape[1];
    let nheads = input.config.nheads;
    let headdim = input.config.headdim;
    let d_state = input.config.d_state;
    let ngroups = input.config.ngroups;

    let device = input.x.tensor().device();

    // Initialize h to zeros: [B, nheads, headdim, d_state]
    let h_tensor = Tensor::<R>::zeros(
        &[batch, nheads, headdim, d_state],
        input.x.tensor().dtype(),
        device,
    );
    let mut h = Var::new(h_tensor, false);

    let mut outputs: Vec<Var<R>> = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let (y_t, h_new) = ssm_step(
            client,
            input.x,
            input.a,
            input.b,
            input.c,
            input.d_param,
            input.dt,
            &h,
            t,
            batch,
            nheads,
            headdim,
            d_state,
            ngroups,
        )?;
        h = h_new;
        outputs.push(y_t);
    }

    // Concatenate along dim 1: [B, S, nheads, headdim]
    let output_refs: Vec<&Var<R>> = outputs.iter().collect();
    var_cat(&output_refs, 1, client).map_err(Error::Numr)
}

/// Chunked SSD forward: splits sequence into chunks for O(S*chunk_size) instead of O(S²).
///
/// Falls back to sequential scan for sequences shorter than chunk_size.
pub fn ssm_forward_chunked<R, C>(client: &C, input: &SsmInput<'_, R>) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + ReduceOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let shape = input.x.shape();
    let batch = shape[0];
    let seq_len = shape[1];
    let chunk_size = input.config.chunk_size;

    // If sequence fits in one chunk, use sequential
    if seq_len <= chunk_size {
        return ssm_forward_sequential(client, input);
    }

    let nchunks = seq_len.div_ceil(chunk_size);
    let nheads = input.config.nheads;
    let headdim = input.config.headdim;
    let d_state = input.config.d_state;

    let device = input.x.tensor().device();
    let dtype = input.x.tensor().dtype();

    // Initialize hidden state
    let h_tensor = Tensor::<R>::zeros(&[batch, nheads, headdim, d_state], dtype, device);
    let mut h = Var::new(h_tensor, false);

    let mut chunk_outputs: Vec<Var<R>> = Vec::with_capacity(nchunks);

    for chunk_idx in 0..nchunks {
        let start = chunk_idx * chunk_size;
        let len = (seq_len - start).min(chunk_size);

        // Extract chunk slices
        let x_chunk = var_narrow(input.x, 1, start, len).map_err(Error::Numr)?;
        let b_chunk = var_narrow(input.b, 1, start, len).map_err(Error::Numr)?;
        let c_chunk = var_narrow(input.c, 1, start, len).map_err(Error::Numr)?;
        let dt_chunk = var_narrow(input.dt, 1, start, len).map_err(Error::Numr)?;

        let chunk_input = SsmInput {
            x: &x_chunk,
            a: input.a,
            b: &b_chunk,
            c: &c_chunk,
            d_param: input.d_param,
            dt: &dt_chunk,
            config: input.config,
        };

        let (chunk_out, h_new) = ssm_chunk_with_state(client, &chunk_input, &h)?;

        chunk_outputs.push(chunk_out);
        h = h_new;
    }

    // Concatenate chunk outputs: [B, S, nheads, headdim]
    let chunk_refs: Vec<&Var<R>> = chunk_outputs.iter().collect();
    var_cat(&chunk_refs, 1, client).map_err(Error::Numr)
}

/// SSM scan over a chunk, returning both output and final hidden state.
fn ssm_chunk_with_state<R, C>(
    client: &C,
    input: &SsmInput<'_, R>,
    h_init: &Var<R>,
) -> Result<(Var<R>, Var<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + UnaryOps<R> + ActivationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let shape = input.x.shape();
    let batch = shape[0];
    let seq_len = shape[1];
    let nheads = input.config.nheads;
    let headdim = input.config.headdim;
    let d_state = input.config.d_state;
    let ngroups = input.config.ngroups;

    let mut h = h_init.clone();
    let mut outputs: Vec<Var<R>> = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let (y_t, h_new) = ssm_step(
            client,
            input.x,
            input.a,
            input.b,
            input.c,
            input.d_param,
            input.dt,
            &h,
            t,
            batch,
            nheads,
            headdim,
            d_state,
            ngroups,
        )?;
        h = h_new;
        outputs.push(y_t);
    }

    let output_refs: Vec<&Var<R>> = outputs.iter().collect();
    let out = var_cat(&output_refs, 1, client).map_err(Error::Numr)?;
    Ok((out, h))
}

/// Single SSM step: computes one timestep of the recurrence.
///
/// Returns (y_t shaped [B, 1, nheads, headdim], updated h).
#[allow(clippy::too_many_arguments)]
fn ssm_step<R, C>(
    client: &C,
    x: &Var<R>,
    a: &Var<R>,
    b: &Var<R>,
    c: &Var<R>,
    d_param: Option<&Var<R>>,
    dt: &Var<R>,
    h: &Var<R>,
    t: usize,
    batch: usize,
    nheads: usize,
    headdim: usize,
    d_state: usize,
    ngroups: usize,
) -> Result<(Var<R>, Var<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + UnaryOps<R> + ActivationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    // Extract x_t: [B, 1, nheads, headdim] -> [B, nheads, headdim]
    let x_t = var_contiguous(&var_narrow(x, 1, t, 1).map_err(Error::Numr)?);
    let x_t = var_reshape(&x_t, &[batch, nheads, headdim]).map_err(Error::Numr)?;

    // Extract dt_t: [B, 1, nheads] -> [B, nheads, 1, 1] for broadcasting
    let dt_t = var_contiguous(&var_narrow(dt, 1, t, 1).map_err(Error::Numr)?);
    let dt_t = var_reshape(&dt_t, &[batch, nheads, 1, 1]).map_err(Error::Numr)?;

    // Extract B_t: [B, 1, ngroups, d_state] -> [B, ngroups, 1, d_state]
    let b_t = var_contiguous(&var_narrow(b, 1, t, 1).map_err(Error::Numr)?);
    let b_t = var_reshape(&b_t, &[batch, ngroups, 1, d_state]).map_err(Error::Numr)?;

    // Extract C_t: [B, 1, ngroups, d_state] -> [B, ngroups, d_state]
    let c_t = var_contiguous(&var_narrow(c, 1, t, 1).map_err(Error::Numr)?);
    let c_t = var_reshape(&c_t, &[batch, ngroups, d_state]).map_err(Error::Numr)?;

    // A: [nheads] -> [1, nheads, 1, 1] for broadcasting with h
    let a_broad = var_reshape(a, &[1, nheads, 1, 1]).map_err(Error::Numr)?;

    // decay = exp(dt_t * A): [B, nheads, 1, 1]
    let dt_a = var_mul(&dt_t, &a_broad, client).map_err(Error::Numr)?;
    let decay = var_exp(&dt_a, client).map_err(Error::Numr)?;

    // h = decay * h
    let mut h = var_mul(&decay, h, client).map_err(Error::Numr)?;

    // Expand B_t for broadcasting with h [B, nheads, headdim, d_state]
    // ngroups=1: [B, 1, 1, d_state] broadcasts naturally
    // ngroups=nheads: reshape to [B, nheads, 1, d_state]
    let b_t_expanded = if ngroups == nheads {
        var_reshape(&b_t, &[batch, nheads, 1, d_state]).map_err(Error::Numr)?
    } else {
        b_t // [B, ngroups, 1, d_state] broadcasts when ngroups=1
    };

    // x_t: [B, nheads, headdim] -> [B, nheads, headdim, 1]
    let x_t_col = var_reshape(&x_t, &[batch, nheads, headdim, 1]).map_err(Error::Numr)?;

    // input_term = dt_t * B_t * x_t: [B, nheads, headdim, d_state]
    let dt_x = var_mul(&dt_t, &x_t_col, client).map_err(Error::Numr)?;
    let input_term = var_mul(&dt_x, &b_t_expanded, client).map_err(Error::Numr)?;

    // h = h + input_term
    h = var_add(&h, &input_term, client).map_err(Error::Numr)?;

    // y_t = h @ C_t: [B, nheads, headdim, d_state] @ [B, ngroups, d_state, 1]
    let c_t_col = var_reshape(&c_t, &[batch, ngroups, d_state, 1]).map_err(Error::Numr)?;
    let y_t = var_matmul(&h, &c_t_col, client).map_err(Error::Numr)?;
    let mut y_t = var_reshape(&y_t, &[batch, nheads, headdim]).map_err(Error::Numr)?;

    // D skip connection
    if let Some(d_var) = d_param {
        let d_broad = var_reshape(d_var, &[1, nheads, 1]).map_err(Error::Numr)?;
        let d_x = var_mul(&d_broad, &x_t, client).map_err(Error::Numr)?;
        y_t = var_add(&y_t, &d_x, client).map_err(Error::Numr)?;
    }

    // y_t: [B, nheads, headdim] -> [B, 1, nheads, headdim]
    let y_t = var_reshape(&y_t, &[batch, 1, nheads, headdim]).map_err(Error::Numr)?;
    Ok((y_t, h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_sequential_ssm_known_values() {
        let (client, device) = cpu_setup();
        let config = Mamba2Config::new(4)
            .with_nheads(1)
            .with_d_state(2)
            .with_expand(1)
            .with_use_d(false)
            .with_dt_softplus(false)
            .with_use_dt_bias(false);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                &[1, 2, 1, 4],
                &device,
            ),
            false,
        );
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[-1.0f32], &[1], &device),
            false,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[1, 2, 1, 2], &device),
            false,
        );
        let c = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 2, 1, 2], &device),
            false,
        );
        let dt = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.5f32, 0.5], &[1, 2, 1], &device),
            false,
        );

        let input = SsmInput {
            x: &x,
            a: &a,
            b: &b,
            c: &c,
            d_param: None,
            dt: &dt,
            config: &config,
        };
        let out = ssm_forward_sequential(&client, &input).unwrap();
        assert_eq!(out.shape(), &[1, 2, 1, 4]);

        let data: Vec<f32> = out.tensor().to_vec();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_chunked_matches_sequential() {
        let (client, device) = cpu_setup();
        let config = Mamba2Config::new(4)
            .with_nheads(1)
            .with_d_state(2)
            .with_expand(1)
            .with_use_d(true)
            .with_dt_softplus(false)
            .with_use_dt_bias(false)
            .with_chunk_size(3);

        let seq_len = 6;
        let x_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[1, seq_len, 1, 4], &device),
            false,
        );
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[-0.5f32], &[1], &device),
            false,
        );
        let b_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.05 + 0.1).collect();
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&b_data, &[1, seq_len, 1, 2], &device),
            false,
        );
        let c_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.03 + 0.2).collect();
        let c = Var::new(
            Tensor::<CpuRuntime>::from_slice(&c_data, &[1, seq_len, 1, 2], &device),
            false,
        );
        let d_param = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.5f32], &[1], &device),
            false,
        );
        let dt_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.1, 0.2, 0.3];
        let dt = Var::new(
            Tensor::<CpuRuntime>::from_slice(&dt_data, &[1, seq_len, 1], &device),
            false,
        );

        let input = SsmInput {
            x: &x,
            a: &a,
            b: &b,
            c: &c,
            d_param: Some(&d_param),
            dt: &dt,
            config: &config,
        };

        let out_seq = ssm_forward_sequential(&client, &input).unwrap();
        let out_chunk = ssm_forward_chunked(&client, &input).unwrap();

        let seq_data: Vec<f32> = out_seq.tensor().to_vec();
        let chunk_data: Vec<f32> = out_chunk.tensor().to_vec();

        assert_eq!(seq_data.len(), chunk_data.len());
        for (i, (s, c)) in seq_data.iter().zip(chunk_data.iter()).enumerate() {
            assert!((s - c).abs() < 1e-4, "mismatch at {i}: seq={s}, chunk={c}");
        }
    }
}
