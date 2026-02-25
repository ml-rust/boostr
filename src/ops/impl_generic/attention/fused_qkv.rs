//! Generic fused QKV projection implementation
//!
//! Composes numr primitives: matmul, split, reshape, transpose, add.
//! All backends can delegate to these functions.

use crate::error::{Error, Result};
use numr::ops::{BinaryOps, MatmulOps, ReduceOps, ShapeOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Fused QKV projection: input @ weight.T [+ bias] → split → reshape → (Q, K, V)
///
/// - `input`: `[B, S, H]`
/// - `weight`: `[(Hq + 2*Hkv), H]` where Hq = num_heads * head_dim, Hkv = num_kv_heads * head_dim
/// - `bias`: optional `[(Hq + 2*Hkv)]`
/// - Returns (Q, K, V) each in `[B, heads, S, D]` layout
#[allow(clippy::too_many_arguments)]
pub fn fused_qkv_projection_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    weight: &Tensor<R>,
    bias: Option<&Tensor<R>>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>
where
    R: Runtime,
    C: RuntimeClient<R> + MatmulOps<R> + BinaryOps<R> + ShapeOps<R>,
{
    let input_shape = input.shape();
    if input_shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: format!("expected 3D [B, S, H], got {}D", input_shape.len()),
        });
    }
    let batch_size = input_shape[0];
    let seq_len = input_shape[1];
    let hidden_dim = input_shape[2];

    let hq = num_heads * head_dim;
    let hkv = num_kv_heads * head_dim;
    let total_proj = hq + 2 * hkv;

    let weight_shape = weight.shape();
    if weight_shape != [total_proj, hidden_dim] {
        return Err(Error::InvalidArgument {
            arg: "weight",
            reason: format!(
                "expected [{}, {}], got {:?}",
                total_proj, hidden_dim, weight_shape
            ),
        });
    }

    // input_2d: [B*S, H]
    let input_2d = input.reshape(&[batch_size * seq_len, hidden_dim])?;

    // weight_t: [H, total_proj] (zero-copy view)
    let weight_t = weight.transpose(-2, -1)?;

    // qkv: [B*S, total_proj]
    let mut qkv = client.matmul(&input_2d, &weight_t).map_err(Error::Numr)?;

    // Add bias if present
    if let Some(b) = bias {
        qkv = client.add(&qkv, b).map_err(Error::Numr)?;
    }

    // qkv_3d: [B, S, total_proj]
    let qkv_3d = qkv.reshape(&[batch_size, seq_len, total_proj])?;

    // Split into Q, K, V along last dimension using narrow
    let q_flat = qkv_3d.narrow(-1, 0, hq)?;
    let k_flat = qkv_3d.narrow(-1, hq, hkv)?;
    let v_flat = qkv_3d.narrow(-1, hq + hkv, hkv)?;

    // Reshape and transpose to [B, heads, S, D]
    let q = q_flat
        .contiguous()
        .reshape(&[batch_size, seq_len, num_heads, head_dim])?
        .transpose(1, 2)?
        .contiguous();

    let k = k_flat
        .contiguous()
        .reshape(&[batch_size, seq_len, num_kv_heads, head_dim])?
        .transpose(1, 2)?
        .contiguous();

    let v = v_flat
        .contiguous()
        .reshape(&[batch_size, seq_len, num_kv_heads, head_dim])?
        .transpose(1, 2)?
        .contiguous();

    Ok((q, k, v))
}

/// Fused output projection with residual: attn_out @ weight.T [+ bias] + residual
///
/// - `attn_out`: `[B, S, Hq*D]`
/// - `weight`: `[H, Hq*D]`
/// - `bias`: optional `[H]`
/// - `residual`: `[B, S, H]`
pub fn fused_output_projection_residual_impl<R, C>(
    client: &C,
    attn_out: &Tensor<R>,
    weight: &Tensor<R>,
    bias: Option<&Tensor<R>>,
    residual: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + MatmulOps<R> + BinaryOps<R>,
{
    let attn_shape = attn_out.shape();
    if attn_shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "attn_out",
            reason: format!("expected 3D [B, S, Hq*D], got {}D", attn_shape.len()),
        });
    }
    let batch_size = attn_shape[0];
    let seq_len = attn_shape[1];
    let proj_dim = attn_shape[2];

    let weight_shape = weight.shape();
    let hidden_dim = weight_shape[0];

    // attn_2d: [B*S, Hq*D]
    let attn_2d = attn_out.reshape(&[batch_size * seq_len, proj_dim])?;

    // weight_t: [Hq*D, H]
    let weight_t = weight.transpose(-2, -1)?;

    // proj: [B*S, H]
    let mut proj = client.matmul(&attn_2d, &weight_t).map_err(Error::Numr)?;

    // Add bias if present
    if let Some(b) = bias {
        proj = client.add(&proj, b).map_err(Error::Numr)?;
    }

    // proj_3d: [B, S, H]
    let proj_3d = proj.reshape(&[batch_size, seq_len, hidden_dim])?;

    // Add residual
    let output = client.add(&proj_3d, residual).map_err(Error::Numr)?;

    Ok(output)
}

/// Backward pass for fused QKV projection
///
/// Given dQ [B, num_heads, S, D], dK [B, num_kv_heads, S, D], dV [B, num_kv_heads, S, D],
/// computes gradients w.r.t. input, weight, and bias.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn fused_qkv_projection_bwd_impl<R, C>(
    client: &C,
    dq: &Tensor<R>,
    dk: &Tensor<R>,
    dv: &Tensor<R>,
    input: &Tensor<R>,
    weight: &Tensor<R>,
    has_bias: bool,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(Tensor<R>, Tensor<R>, Option<Tensor<R>>)>
where
    R: Runtime,
    C: RuntimeClient<R> + MatmulOps<R> + BinaryOps<R> + ShapeOps<R> + ReduceOps<R>,
{
    let input_shape = input.shape();
    let batch_size = input_shape[0];
    let seq_len = input_shape[1];
    let hidden_dim = input_shape[2];

    let hq = num_heads * head_dim;
    let hkv = num_kv_heads * head_dim;

    // Transpose dQ/dK/dV from [B, heads, S, D] to [B, S, heads, D] then flatten
    let dq_flat = dq
        .transpose(1, 2)?
        .contiguous()
        .reshape(&[batch_size * seq_len, hq])?;

    let dk_flat = dk
        .transpose(1, 2)?
        .contiguous()
        .reshape(&[batch_size * seq_len, hkv])?;

    let dv_flat = dv
        .transpose(1, 2)?
        .contiguous()
        .reshape(&[batch_size * seq_len, hkv])?;

    // Concatenate dQ, dK, dV → d_qkv: [B*S, total_proj]
    let d_qkv = client
        .cat(&[&dq_flat, &dk_flat, &dv_flat], -1)
        .map_err(Error::Numr)?;

    // d_input = d_qkv @ weight → [B*S, H]
    let d_input_2d = client.matmul(&d_qkv, weight).map_err(Error::Numr)?;
    let d_input = d_input_2d.reshape(&[batch_size, seq_len, hidden_dim])?;

    // d_weight = d_qkv.T @ input_2d → [total_proj, H]
    let input_2d = input.reshape(&[batch_size * seq_len, hidden_dim])?;
    let d_qkv_t = d_qkv.transpose(-2, -1)?;
    let d_weight = client
        .matmul(&d_qkv_t.contiguous(), &input_2d)
        .map_err(Error::Numr)?;

    // d_bias = sum(d_qkv, dim=0) → [total_proj]
    let d_bias = if has_bias {
        let db = client.sum(&d_qkv, &[0], false).map_err(Error::Numr)?;
        Some(db)
    } else {
        None
    };

    Ok((d_input, d_weight, d_bias))
}

/// Backward pass for fused output projection with residual
///
/// Given d_output [B, S, H], computes gradients w.r.t. attn_out, weight, bias, residual.
#[allow(clippy::type_complexity)]
pub fn fused_output_projection_residual_bwd_impl<R, C>(
    client: &C,
    d_output: &Tensor<R>,
    attn_out: &Tensor<R>,
    weight: &Tensor<R>,
    has_bias: bool,
) -> Result<(Tensor<R>, Tensor<R>, Option<Tensor<R>>, Tensor<R>)>
where
    R: Runtime,
    C: RuntimeClient<R> + MatmulOps<R> + BinaryOps<R> + ReduceOps<R>,
{
    let d_shape = d_output.shape();
    let batch_size = d_shape[0];
    let seq_len = d_shape[1];
    let hidden_dim = d_shape[2];

    let attn_shape = attn_out.shape();
    let proj_dim = attn_shape[2];

    // d_residual = d_output (identity branch of residual addition)
    let d_residual = d_output.clone();

    // d_output_2d: [B*S, H]
    let d_output_2d = d_output.reshape(&[batch_size * seq_len, hidden_dim])?;

    // d_attn_out = d_output_2d @ weight → [B*S, Hq*D]
    let d_attn_2d = client.matmul(&d_output_2d, weight).map_err(Error::Numr)?;
    let d_attn_out = d_attn_2d.reshape(&[batch_size, seq_len, proj_dim])?;

    // d_weight = d_output_2d.T @ attn_out_2d → [H, Hq*D]
    let attn_2d = attn_out.reshape(&[batch_size * seq_len, proj_dim])?;
    let d_output_2d_t = d_output_2d.transpose(-2, -1)?;
    let d_weight = client
        .matmul(&d_output_2d_t.contiguous(), &attn_2d)
        .map_err(Error::Numr)?;

    // d_bias = sum(d_output_2d, dim=0) → [H]
    let d_bias = if has_bias {
        let db = client.sum(&d_output_2d, &[0], false).map_err(Error::Numr)?;
        Some(db)
    } else {
        None
    };

    Ok((d_attn_out, d_weight, d_bias, d_residual))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::dtype::DType;
    use numr::runtime::cpu::CpuRuntime;

    fn assert_close(actual: &Tensor<CpuRuntime>, expected: &Tensor<CpuRuntime>, label: &str) {
        let a = actual.contiguous().to_vec::<f32>();
        let b = expected.contiguous().to_vec::<f32>();
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-5,
                "{label} mismatch at {i}: got {x}, expected {y}"
            );
        }
    }

    /// Verify QKV backward numerically against a manual reference.
    /// Forward: qkv = input @ W^T + bias, then split+reshape.
    /// Backward: d_input = d_qkv @ W, d_weight = d_qkv^T @ input, d_bias = sum(d_qkv, dim=0).
    #[test]
    fn test_fused_qkv_bwd_numerical() {
        use numr::ops::{MatmulOps, ReduceOps, ShapeOps};

        let (client, device) = cpu_setup();
        let (batch, seq, hidden) = (1, 2, 8);
        let (num_heads, num_kv_heads, head_dim) = (2, 1, 4);
        let hq = num_heads * head_dim;
        let hkv = num_kv_heads * head_dim;
        let total_proj = hq + 2 * hkv;

        // Use deterministic values (sequential) for reproducibility
        let n_input = batch * seq * hidden;
        let input_data: Vec<f32> = (0..n_input).map(|i| (i as f32) * 0.01).collect();
        let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[batch, seq, hidden], &device);

        let n_weight = total_proj * hidden;
        let weight_data: Vec<f32> = (0..n_weight).map(|i| (i as f32) * 0.005).collect();
        let weight = Tensor::<CpuRuntime>::from_slice(&weight_data, &[total_proj, hidden], &device);

        // Upstream gradients
        let dq_data: Vec<f32> = (0..batch * num_heads * seq * head_dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
            .collect();
        let dq =
            Tensor::<CpuRuntime>::from_slice(&dq_data, &[batch, num_heads, seq, head_dim], &device);

        let dk_data: Vec<f32> = (0..batch * num_kv_heads * seq * head_dim)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.01)
            .collect();
        let dk = Tensor::<CpuRuntime>::from_slice(
            &dk_data,
            &[batch, num_kv_heads, seq, head_dim],
            &device,
        );

        let dv_data: Vec<f32> = (0..batch * num_kv_heads * seq * head_dim)
            .map(|i| ((i % 3) as f32 - 1.0) * 0.01)
            .collect();
        let dv = Tensor::<CpuRuntime>::from_slice(
            &dv_data,
            &[batch, num_kv_heads, seq, head_dim],
            &device,
        );

        // Compute backward via our implementation
        let (d_input, d_weight, d_bias) = fused_qkv_projection_bwd_impl(
            &client,
            &dq,
            &dk,
            &dv,
            &input,
            &weight,
            true,
            num_heads,
            num_kv_heads,
            head_dim,
        )
        .unwrap();

        // Compute reference manually:
        // 1. Reconstruct d_qkv from dq, dk, dv
        let bs = batch * seq;
        let dq_flat = dq
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .reshape(&[bs, hq])
            .unwrap();
        let dk_flat = dk
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .reshape(&[bs, hkv])
            .unwrap();
        let dv_flat = dv
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .reshape(&[bs, hkv])
            .unwrap();
        let d_qkv_ref = client.cat(&[&dq_flat, &dk_flat, &dv_flat], -1).unwrap();

        // 2. ref_d_input = d_qkv @ weight
        let ref_d_input_2d = client.matmul(&d_qkv_ref, &weight).unwrap();
        let ref_d_input = ref_d_input_2d.reshape(&[batch, seq, hidden]).unwrap();

        // 3. ref_d_weight = d_qkv^T @ input_2d
        let input_2d = input.reshape(&[bs, hidden]).unwrap();
        let d_qkv_t = d_qkv_ref.transpose(-2, -1).unwrap().contiguous();
        let ref_d_weight = client.matmul(&d_qkv_t, &input_2d).unwrap();

        // 4. ref_d_bias = sum(d_qkv, dim=0)
        let ref_d_bias = client.sum(&d_qkv_ref, &[0], false).unwrap();

        assert_close(&d_input, &ref_d_input, "d_input");
        assert_close(&d_weight, &ref_d_weight, "d_weight");
        assert_close(&d_bias.unwrap(), &ref_d_bias, "d_bias");
    }

    /// Verify output projection backward numerically.
    ///
    /// Forward: output = attn_out @ W^T + bias + residual
    /// So: d_residual = d_output, d_attn_out = d_output @ W,
    ///     d_weight = d_output^T @ attn_out, d_bias = sum(d_output, dim=0).
    #[test]
    fn test_fused_output_projection_residual_bwd_numerical() {
        use numr::ops::{MatmulOps, ReduceOps};

        let (client, device) = cpu_setup();
        let (batch, seq, hidden, proj_dim) = (1, 3, 8, 8);

        let n_attn = batch * seq * proj_dim;
        let attn_data: Vec<f32> = (0..n_attn).map(|i| (i as f32) * 0.02).collect();
        let attn_out =
            Tensor::<CpuRuntime>::from_slice(&attn_data, &[batch, seq, proj_dim], &device);

        let n_weight = hidden * proj_dim;
        let weight_data: Vec<f32> = (0..n_weight).map(|i| (i as f32) * 0.01).collect();
        let weight = Tensor::<CpuRuntime>::from_slice(&weight_data, &[hidden, proj_dim], &device);

        let n_dout = batch * seq * hidden;
        let d_output_data: Vec<f32> = (0..n_dout).map(|i| ((i % 5) as f32 - 2.0) * 0.01).collect();
        let d_output =
            Tensor::<CpuRuntime>::from_slice(&d_output_data, &[batch, seq, hidden], &device);

        // Compute backward via our implementation
        let (d_attn_out, d_weight, d_bias, d_residual) =
            fused_output_projection_residual_bwd_impl(&client, &d_output, &attn_out, &weight, true)
                .unwrap();

        // Reference implementation
        let bs = batch * seq;
        let d_output_2d = d_output.reshape(&[bs, hidden]).unwrap();

        // ref_d_attn = d_output_2d @ weight
        let ref_d_attn_2d = client.matmul(&d_output_2d, &weight).unwrap();
        let ref_d_attn = ref_d_attn_2d.reshape(&[batch, seq, proj_dim]).unwrap();

        // ref_d_weight = d_output_2d^T @ attn_2d
        let attn_2d = attn_out.reshape(&[bs, proj_dim]).unwrap();
        let d_output_2d_t = d_output_2d.transpose(-2, -1).unwrap().contiguous();
        let ref_d_weight = client.matmul(&d_output_2d_t, &attn_2d).unwrap();

        // ref_d_bias = sum(d_output_2d, dim=0)
        let ref_d_bias = client.sum(&d_output_2d, &[0], false).unwrap();

        assert_close(&d_residual, &d_output, "d_residual");
        assert_close(&d_attn_out, &ref_d_attn, "d_attn_out");
        assert_close(&d_weight, &ref_d_weight, "d_weight");
        assert_close(&d_bias.unwrap(), &ref_d_bias, "d_bias");
    }

    /// Verify QKV backward with no bias returns None for d_bias.
    #[test]
    fn test_fused_qkv_bwd_no_bias() {
        let (client, device) = cpu_setup();
        let (batch, seq, hidden) = (1, 2, 8);
        let (num_heads, num_kv_heads, head_dim) = (2, 1, 4);
        let total_proj = num_heads * head_dim + 2 * num_kv_heads * head_dim;

        let dq =
            Tensor::<CpuRuntime>::ones(&[batch, num_heads, seq, head_dim], DType::F32, &device);
        let dk =
            Tensor::<CpuRuntime>::ones(&[batch, num_kv_heads, seq, head_dim], DType::F32, &device);
        let dv =
            Tensor::<CpuRuntime>::ones(&[batch, num_kv_heads, seq, head_dim], DType::F32, &device);
        let input = Tensor::<CpuRuntime>::ones(&[batch, seq, hidden], DType::F32, &device);
        let weight = Tensor::<CpuRuntime>::ones(&[total_proj, hidden], DType::F32, &device);

        let (_, _, d_bias) = fused_qkv_projection_bwd_impl(
            &client,
            &dq,
            &dk,
            &dv,
            &input,
            &weight,
            false,
            num_heads,
            num_kv_heads,
            head_dim,
        )
        .unwrap();

        assert!(d_bias.is_none());
    }
}
