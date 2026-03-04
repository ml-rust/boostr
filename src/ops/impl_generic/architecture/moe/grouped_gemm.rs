//! MoE grouped GEMM: per-expert matmul on contiguous token groups.
//!
//! THE algorithm — same for all backends.
//! Composes numr primitives: narrow, matmul, cat, silu, gelu.

use crate::error::{Error, Result};
use crate::ops::traits::architecture::moe::MoEActivation;
use numr::dtype::DType;
use numr::ops::{ActivationOps, MatmulOps, ShapeOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Grouped GEMM: per-expert matmul on contiguous token groups.
///
/// For each expert e, computes:
///   output[offsets[e]..offsets[e+1]] = tokens[offsets[e]..offsets[e+1]] @ weights[e]
pub fn moe_grouped_gemm_impl<R, C>(
    client: &C,
    permuted_tokens: &Tensor<R>,
    expert_weights: &Tensor<R>,
    expert_offsets: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + MatmulOps<R> + ShapeOps<R> + TensorOps<R>,
{
    validate_grouped_gemm_args(permuted_tokens, expert_weights, expert_offsets)?;

    let ew_shape = expert_weights.shape();
    let num_experts = ew_shape[0];
    let out_dim = ew_shape[2];
    let total_tokens = permuted_tokens.shape()[0];

    // `to_vec()` here is called only by the CPU backend — the CUDA and WebGPU backends
    // use fused kernels and never reach this function for grouped GEMM. On CPU there is
    // no device memory, so this is not a GPU↔CPU transfer.
    let offsets = expert_offsets.to_vec::<i32>();

    let mut expert_outputs: Vec<Tensor<R>> = Vec::with_capacity(num_experts);

    for e in 0..num_experts {
        let start = offsets[e] as usize;
        let end = offsets[e + 1] as usize;
        let count = end - start;

        if count == 0 {
            continue;
        }

        let expert_tokens = permuted_tokens
            .narrow(0, start, count)
            .map_err(Error::Numr)?
            .contiguous();

        let weight = expert_weights
            .narrow(0, e, 1)
            .map_err(Error::Numr)?
            .contiguous()
            .reshape(&[ew_shape[1], out_dim])
            .map_err(Error::Numr)?;

        // matmul: [count, in_dim] @ [in_dim, out_dim] → [count, out_dim]
        let result = client
            .matmul(&expert_tokens, &weight)
            .map_err(Error::Numr)?;
        expert_outputs.push(result);
    }

    if expert_outputs.is_empty() {
        // All experts have zero tokens — return empty
        let device = permuted_tokens.device();
        return Ok(Tensor::<R>::empty(
            &[total_tokens, out_dim],
            permuted_tokens.dtype(),
            device,
        ));
    }

    // Concatenate all expert outputs along dim 0
    let refs: Vec<&Tensor<R>> = expert_outputs.iter().collect();
    let output = client.cat(&refs, 0).map_err(Error::Numr)?;

    Ok(output)
}

/// Fused grouped GEMM with activation.
///
/// Same as `moe_grouped_gemm_impl` but applies activation after each expert's matmul.
pub fn moe_grouped_gemm_fused_impl<R, C>(
    client: &C,
    permuted_tokens: &Tensor<R>,
    expert_weights: &Tensor<R>,
    expert_offsets: &Tensor<R>,
    activation: MoEActivation,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + MatmulOps<R> + ShapeOps<R> + TensorOps<R> + ActivationOps<R>,
{
    validate_grouped_gemm_args(permuted_tokens, expert_weights, expert_offsets)?;

    let ew_shape = expert_weights.shape();
    let num_experts = ew_shape[0];
    let out_dim = ew_shape[2];
    let total_tokens = permuted_tokens.shape()[0];

    // `to_vec()` here is called only by the CPU backend — see comment in
    // `moe_grouped_gemm_impl`. Not a GPU↔CPU transfer.
    let offsets = expert_offsets.to_vec::<i32>();

    let mut expert_outputs: Vec<Tensor<R>> = Vec::with_capacity(num_experts);

    for e in 0..num_experts {
        let start = offsets[e] as usize;
        let end = offsets[e + 1] as usize;
        let count = end - start;

        if count == 0 {
            continue;
        }

        let expert_tokens = permuted_tokens
            .narrow(0, start, count)
            .map_err(Error::Numr)?
            .contiguous();

        let weight = expert_weights
            .narrow(0, e, 1)
            .map_err(Error::Numr)?
            .contiguous()
            .reshape(&[ew_shape[1], out_dim])
            .map_err(Error::Numr)?;

        let result = client
            .matmul(&expert_tokens, &weight)
            .map_err(Error::Numr)?;

        // Apply activation
        let activated = match activation {
            MoEActivation::SiLU => client.silu(&result).map_err(Error::Numr)?,
            MoEActivation::GeLU => client.gelu(&result).map_err(Error::Numr)?,
            MoEActivation::None => result,
        };

        expert_outputs.push(activated);
    }

    if expert_outputs.is_empty() {
        let device = permuted_tokens.device();
        return Ok(Tensor::<R>::empty(
            &[total_tokens, out_dim],
            permuted_tokens.dtype(),
            device,
        ));
    }

    let refs: Vec<&Tensor<R>> = expert_outputs.iter().collect();
    let output = client.cat(&refs, 0).map_err(Error::Numr)?;

    Ok(output)
}

fn validate_grouped_gemm_args<R: Runtime>(
    permuted_tokens: &Tensor<R>,
    expert_weights: &Tensor<R>,
    expert_offsets: &Tensor<R>,
) -> Result<()> {
    let pt_shape = permuted_tokens.shape();
    let ew_shape = expert_weights.shape();
    let eo_shape = expert_offsets.shape();

    if pt_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "permuted_tokens",
            reason: format!("expected 2D [total, in_dim], got {}D", pt_shape.len()),
        });
    }
    if ew_shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "expert_weights",
            reason: format!(
                "expected 3D [num_experts, in_dim, out_dim], got {}D",
                ew_shape.len()
            ),
        });
    }
    if eo_shape.len() != 1 {
        return Err(Error::InvalidArgument {
            arg: "expert_offsets",
            reason: format!("expected 1D [num_experts+1], got {}D", eo_shape.len()),
        });
    }
    if pt_shape[1] != ew_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "expert_weights",
            reason: format!(
                "in_dim mismatch: tokens has {}, weights has {}",
                pt_shape[1], ew_shape[1]
            ),
        });
    }
    if eo_shape[0] != ew_shape[0] + 1 {
        return Err(Error::InvalidArgument {
            arg: "expert_offsets",
            reason: format!(
                "expected {} entries (num_experts+1), got {}",
                ew_shape[0] + 1,
                eo_shape[0]
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_grouped_gemm_shapes() {
        let (client, device) = cpu_setup();
        let num_experts = 3;
        let in_dim = 4;
        let out_dim = 6;
        let total_tokens = 8;

        let tokens_data: Vec<f32> = (0..total_tokens * in_dim)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let tokens =
            Tensor::<CpuRuntime>::from_slice(&tokens_data, &[total_tokens, in_dim], &device);

        let weights_data: Vec<f32> = (0..num_experts * in_dim * out_dim)
            .map(|i| (i as f32 * 0.05).cos() * 0.1)
            .collect();
        let weights = Tensor::<CpuRuntime>::from_slice(
            &weights_data,
            &[num_experts, in_dim, out_dim],
            &device,
        );

        let offsets = Tensor::<CpuRuntime>::from_slice(&[0i32, 3, 5, 8], &[4], &device);

        let output = moe_grouped_gemm_impl(&client, &tokens, &weights, &offsets).unwrap();
        assert_eq!(output.shape(), &[total_tokens, out_dim]);
    }

    #[test]
    fn test_grouped_gemm_fused_silu() {
        let (client, device) = cpu_setup();
        const NUM_EXPERTS: usize = 2;
        const IN_DIM: usize = 4;
        const OUT_DIM: usize = 4;
        const TOTAL: usize = 4;

        let tokens =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; TOTAL * IN_DIM], &[TOTAL, IN_DIM], &device);
        let weights = Tensor::<CpuRuntime>::from_slice(
            &[0.1f32; NUM_EXPERTS * IN_DIM * OUT_DIM],
            &[NUM_EXPERTS, IN_DIM, OUT_DIM],
            &device,
        );
        let offsets = Tensor::<CpuRuntime>::from_slice(&[0i32, 2, 4], &[3], &device);

        let plain = moe_grouped_gemm_impl(&client, &tokens, &weights, &offsets).unwrap();
        let fused =
            moe_grouped_gemm_fused_impl(&client, &tokens, &weights, &offsets, MoEActivation::SiLU)
                .unwrap();

        // Fused should equal silu(plain) element-wise
        let plain_vec = plain.to_vec::<f32>();
        let fused_vec = fused.to_vec::<f32>();
        for (i, (&p, &f)) in plain_vec.iter().zip(fused_vec.iter()).enumerate() {
            let expected = p * (1.0 / (1.0 + (-p).exp())); // silu
            assert!(
                (f - expected).abs() < 1e-5,
                "fused silu mismatch at {}: got {}, expected {}",
                i,
                f,
                expected
            );
        }
    }
}
