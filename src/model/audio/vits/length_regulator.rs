//! Length regulation: exact port of the alignment-matrix arithmetic inside
//! HF `VitsModel.forward` (`transformers.models.vits.modeling_vits`), the
//! part between the duration predictor and the flow / decoder.
//!
//! Converts a per-input-token `log_duration` into: integer per-token
//! durations, the resulting output length per batch element, a
//! `_generate_path`-style monotonic alignment matrix, and the prior
//! distribution (means/log-variances) expanded from input length to output
//! length via that alignment matrix.
//!
//! The one place this reads a tensor value back to the host is
//! `output_length` (the batch's longest predicted length): every downstream
//! tensor's shape depends on it, and no on-device op can size a tensor from
//! data it has not yet read. This is a single small `[batch]`-sized
//! reduction result, read once for shape, not for data processing — the same
//! category as reading `.shape()`, not a compute-on-CPU shortcut.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, CumulativeOps, MatmulOps, ReduceOps, ScalarOps, ShapeOps,
    TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Client ops [`regulate_length`] composes.
pub trait LengthRegulatorClient<R: Runtime>:
    RuntimeClient<R>
    + BinaryOps<R>
    + CompareOps<R>
    + CumulativeOps<R>
    + MatmulOps<R>
    + ReduceOps<R>
    + ScalarOps<R>
    + ShapeOps<R>
    + TypeConversionOps<R>
    + UnaryOps<R>
    + UtilityOps<R>
{
}

impl<R, C> LengthRegulatorClient<R> for C
where
    R: Runtime,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + CompareOps<R>
        + CumulativeOps<R>
        + MatmulOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + UtilityOps<R>,
{
}

/// Every tensor [`regulate_length`] produces from a `log_duration`.
pub struct LengthRegulated<R: Runtime> {
    /// `ceil(exp(log_duration) * input_padding_mask * length_scale)`.
    /// `[batch, 1, input_length]`.
    pub duration: Tensor<R>,
    /// `clamp_min(sum(duration, [1, 2]), 1)`, as `I64`. `[batch]`.
    pub predicted_lengths: Tensor<R>,
    /// `1.0` for `t < predicted_lengths[b]`, else `0.0`. `[batch, 1,
    /// output_length]`.
    pub output_padding_mask: Tensor<R>,
    /// Monotonic alignment matrix: `1.0` where output step `t` is assigned
    /// to input token `i`. `[batch, 1, output_length, input_length]`.
    pub attn: Tensor<R>,
    /// `attn.squeeze(1) @ prior_means`, transposed. `[batch, flow_size,
    /// output_length]`.
    pub expanded_prior_means: Tensor<R>,
    /// Same expansion, over `prior_log_variances`. `[batch, flow_size,
    /// output_length]`.
    pub expanded_prior_log_variances: Tensor<R>,
}

/// `log_duration`: `[batch, 1, input_length]` (the stochastic duration
/// predictor's reverse-path output). `input_padding_mask`: `[batch, 1,
/// input_length]`. `prior_means` / `prior_log_variances`: `[batch,
/// input_length, flow_size]` (the text encoder's outputs, unchanged — HF
/// passes these into `torch.matmul` with no transpose). `speaking_rate`:
/// HF's `model.speaking_rate`; `length_scale = 1.0 / speaking_rate`.
pub fn regulate_length<R, C>(
    client: &C,
    log_duration: &Tensor<R>,
    input_padding_mask: &Tensor<R>,
    prior_means: &Tensor<R>,
    prior_log_variances: &Tensor<R>,
    speaking_rate: f64,
) -> Result<LengthRegulated<R>>
where
    R: Runtime<DType = DType>,
    C: LengthRegulatorClient<R>,
{
    if !(speaking_rate.is_finite() && speaking_rate > 0.0) {
        return Err(Error::InvalidArgument {
            arg: "speaking_rate",
            reason: format!("expected a finite positive rate, got {speaking_rate}"),
        });
    }
    let batch = log_duration.shape()[0];
    let input_length = log_duration.shape()[2];
    let length_scale = 1.0 / speaking_rate;

    let exp_log_duration = client.exp(log_duration).map_err(Error::Numr)?;
    let masked = client
        .mul(&exp_log_duration, input_padding_mask)
        .map_err(Error::Numr)?;
    let scaled = client
        .mul_scalar(&masked, length_scale)
        .map_err(Error::Numr)?;
    let duration = client.ceil(&scaled).map_err(Error::Numr)?; // [batch, 1, input_length]

    let duration_sum = client.sum(&duration, &[1, 2], false).map_err(Error::Numr)?; // [batch]
    let predicted_lengths_f = client
        .clamp(&duration_sum, 1.0, f64::INFINITY)
        .map_err(Error::Numr)?;
    let predicted_lengths = client
        .cast(&predicted_lengths_f, DType::I64)
        .map_err(Error::Numr)?;

    // The sole host read: sizes the output-length axis. See the module doc.
    let max_len = client
        .max(&predicted_lengths_f, &[0], false)
        .and_then(|t| t.contiguous())
        .map_err(Error::Numr)?;
    let max_len_f32 = client
        .cast(&max_len, DType::F32)
        .map_err(Error::Numr)?
        .contiguous()
        .map_err(Error::Numr)?;
    let max_len_vec: Vec<f32> = max_len_f32.to_vec();
    let output_length = max_len_vec.first().copied().unwrap_or(0.0).round().max(0.0) as usize;

    let indices = client
        .arange(0.0, output_length as f64, 1.0, DType::F32)
        .map_err(Error::Numr)?; // [output_length]
    let indices_row = indices.unsqueeze(0).map_err(Error::Numr)?; // [1, output_length]
    let lengths_col = predicted_lengths_f
        .unsqueeze(1)
        .and_then(|t| t.contiguous())
        .map_err(Error::Numr)?; // [batch, 1]
    let output_mask_2d = client.lt(&indices_row, &lengths_col).map_err(Error::Numr)?; // [batch, output_length]
    let output_padding_mask = output_mask_2d
        .unsqueeze(1)
        .and_then(|t| t.contiguous())
        .map_err(Error::Numr)?; // [batch, 1, output_length]

    // cum_duration[b, i] = sum(duration[b, 0, 0..=i]).
    let cum_duration = client.cumsum(&duration, -1).map_err(Error::Numr)?; // [batch, 1, input_length]
    let cum_duration_col = cum_duration
        .reshape(&[batch * input_length, 1])
        .map_err(Error::Numr)?;
    let out_indices = client
        .arange(0.0, output_length as f64, 1.0, DType::F32)
        .map_err(Error::Numr)?
        .unsqueeze(0)
        .map_err(Error::Numr)?; // [1, output_length]
    let valid = client
        .lt(&out_indices, &cum_duration_col)
        .map_err(Error::Numr)?; // [batch*input_length, output_length]
    let valid = valid
        .reshape(&[batch, input_length, output_length])
        .map_err(Error::Numr)?;

    // padded[b, i, t] = valid[b, i, t] - valid[b, i-1, t] (0 for i == 0):
    // isolates the output steps newly covered at input position i.
    let shifted = client
        .pad(&valid, &[0, 0, 1, 0], 0.0)
        .map_err(Error::Numr)?; // [batch, input_length + 1, output_length]
    let shifted = shifted
        .narrow(1, 0, input_length)
        .and_then(|t| t.contiguous())
        .map_err(Error::Numr)?;
    let path = client.sub(&valid, &shifted).map_err(Error::Numr)?; // [batch, input_length, output_length]

    let path_t = path
        .unsqueeze(1)
        .and_then(|t| t.transpose(2, 3))
        .and_then(|t| t.contiguous())
        .map_err(Error::Numr)?; // [batch, 1, output_length, input_length]
    let attn_mask = {
        let a = input_padding_mask
            .unsqueeze(2)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?; // [batch, 1, 1, input_length]
        let b = output_padding_mask
            .unsqueeze(3)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?; // [batch, 1, output_length, 1]
        client.mul(&a, &b).map_err(Error::Numr)? // [batch, 1, output_length, input_length]
    };
    let attn = client.mul(&path_t, &attn_mask).map_err(Error::Numr)?;

    let attn_matmul = attn.squeeze(Some(1)).contiguous().map_err(Error::Numr)?; // [batch, output_length, input_length]
    let expanded_prior_means = client
        .matmul(&attn_matmul, prior_means)
        .and_then(|t| t.transpose(1, 2))
        .and_then(|t| t.contiguous())
        .map_err(Error::Numr)?;
    let expanded_prior_log_variances = client
        .matmul(&attn_matmul, prior_log_variances)
        .and_then(|t| t.transpose(1, 2))
        .and_then(|t| t.contiguous())
        .map_err(Error::Numr)?;

    Ok(LengthRegulated {
        duration,
        predicted_lengths,
        output_padding_mask,
        attn,
        expanded_prior_means,
        expanded_prior_log_variances,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// Hand-worked example: 1 batch, 2 input tokens with pre-`ceil` durations
    /// 1.5 and 0.7 (`length_scale = 1`) — safely mid-bin, not near an
    /// integer, so an `exp(ln(x))` round-trip rounding either way still
    /// `ceil`s to the intended 2 and 1. `output_length = 3`; the path
    /// assigns output steps `{0,1}` to input 0 and `{2}` to input 1.
    #[test]
    fn path_construction_matches_hand_example() {
        let (client, device) = cpu_setup();
        let log_duration =
            Tensor::<CpuRuntime>::from_slice(&[1.5f32.ln(), 0.7f32.ln()], &[1, 1, 2], &device)
                .unwrap();
        let input_padding_mask =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[1, 1, 2], &device).unwrap();
        // flow_size = 1, distinct means per input token so the expansion is checkable.
        let prior_means =
            Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0], &[1, 2, 1], &device).unwrap();
        let prior_log_variances =
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[1, 2, 1], &device).unwrap();

        let out = regulate_length(
            &client,
            &log_duration,
            &input_padding_mask,
            &prior_means,
            &prior_log_variances,
            1.0,
        )
        .unwrap();

        assert_eq!(out.duration.to_vec::<f32>(), vec![2.0, 1.0]);
        assert_eq!(out.predicted_lengths.to_vec::<i64>(), vec![3]);
        assert_eq!(out.output_padding_mask.shape(), &[1, 1, 3]);
        assert_eq!(out.output_padding_mask.to_vec::<f32>(), vec![1.0, 1.0, 1.0]);

        assert_eq!(out.attn.shape(), &[1, 1, 3, 2]);
        // attn[b,0,t,i]: row-major over (t=3, i=2).
        assert_eq!(
            out.attn.to_vec::<f32>(),
            vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "output steps 0,1 assign to input 0; step 2 assigns to input 1"
        );

        assert_eq!(out.expanded_prior_means.shape(), &[1, 1, 3]);
        assert_eq!(
            out.expanded_prior_means.to_vec::<f32>(),
            vec![10.0, 10.0, 20.0]
        );
        assert_eq!(
            out.expanded_prior_log_variances.to_vec::<f32>(),
            vec![0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn rejects_non_positive_speaking_rate() {
        let (client, device) = cpu_setup();
        let log_duration =
            Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1, 1, 1], &device).unwrap();
        let mask = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1, 1, 1], &device).unwrap();
        let means = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1, 1, 1], &device).unwrap();
        let log_vars = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1, 1, 1], &device).unwrap();
        let err = regulate_length(&client, &log_duration, &mask, &means, &log_vars, 0.0);
        assert!(err.is_err());
    }

    #[test]
    fn batched_lengths_use_the_longest_for_output_length() {
        let (client, device) = cpu_setup();
        // batch 0: one token, pre-ceil duration 1.0 (ln(1.0) == 0.0 exactly,
        // no round-trip rounding risk) -> ceils to 1. batch 1: one token,
        // pre-ceil duration 2.4 (safely mid-bin) -> ceils to 3.
        let log_duration =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32.ln(), 2.4f32.ln()], &[2, 1, 1], &device)
                .unwrap();
        let mask = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2, 1, 1], &device).unwrap();
        let means = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2, 1, 1], &device).unwrap();
        let log_vars =
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2, 1, 1], &device).unwrap();

        let out = regulate_length(&client, &log_duration, &mask, &means, &log_vars, 1.0).unwrap();
        assert_eq!(out.predicted_lengths.to_vec::<i64>(), vec![1, 3]);
        assert_eq!(out.output_padding_mask.shape(), &[2, 1, 3]);
        assert_eq!(
            out.output_padding_mask.to_vec::<f32>(),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "batch 0 only has 1 valid output step, batch 1 has all 3"
        );
    }
}
