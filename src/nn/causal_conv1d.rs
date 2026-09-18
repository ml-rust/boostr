//! Depthwise causal conv1d with a carried sliding window.
//!
//! One function serves prefill and decode. The window holds the last
//! `kernel - 1` inputs the layer saw, so the conv runs over
//! `[window | x]` with `PaddingMode::Valid` and returns exactly `seq`
//! outputs. A zero window is a fresh sequence: the result equals a conv
//! with `kernel - 1` zeros of left padding. A one-token `x` is a decode
//! step.
//!
//! Callers: Mamba2 (`model/mamba/mamba2/conv.rs`, `cached_conv`) and Gated
//! DeltaNet (`model/hybrid/blocks/gdn`).
//!
//! # Layout
//!
//! - `x`: `[batch, channels, seq]`
//! - `weight`: `[channels, 1, kernel]`, numr's depthwise layout with
//!   `groups = channels`
//! - `bias`: `[channels]` or `None`
//! - `window`: `[batch, channels, kernel - 1]`

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{ConvOps, PaddingMode, ShapeOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Run the causal conv over `x` with `window` as left context.
///
/// Returns `(out, window)`:
///
/// - `out`: `[batch, channels, seq]`
/// - `window`: `[batch, channels, kernel - 1]`, the last `kernel - 1`
///   columns of `[window | x]`. When `seq < kernel - 1` the old window's
///   tail survives in front of `x`.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a shape disagrees with the layout above.
pub fn causal_conv1d<R, C>(
    client: &C,
    x: &Tensor<R>,
    weight: &Tensor<R>,
    bias: Option<&Tensor<R>>,
    window: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ConvOps<R> + ShapeOps<R>,
{
    let (batch, channels, kernel) = check_shapes(x, weight, window)?;
    let seq = x.shape()[2];
    let keep = kernel - 1;

    let input = client.cat(&[window, x], 2).map_err(Error::Numr)?;
    let out = client
        .conv1d(&input, weight, bias, 1, PaddingMode::Valid, 1, channels)
        .map_err(Error::Numr)?;

    let window = if keep == 0 {
        Tensor::<R>::zeros(&[batch, channels, 0], x.dtype(), x.device())?
    } else {
        input
            .narrow(2, seq, keep)
            .map_err(Error::Numr)?
            .contiguous()?
    };
    Ok((out, window))
}

/// Return `(batch, channels, kernel)` after checking every operand.
fn check_shapes<R: Runtime>(
    x: &Tensor<R>,
    weight: &Tensor<R>,
    window: &Tensor<R>,
) -> Result<(usize, usize, usize)> {
    let xs = x.shape();
    if xs.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("expected [batch, channels, seq], got {xs:?}"),
        });
    }
    let (batch, channels) = (xs[0], xs[1]);
    let ws = weight.shape();
    if ws.len() != 3 || ws[0] != channels || ws[1] != 1 || ws[2] == 0 {
        return Err(Error::InvalidArgument {
            arg: "weight",
            reason: format!("expected [{channels}, 1, kernel >= 1], got {ws:?}"),
        });
    }
    let kernel = ws[2];
    let want = [batch, channels, kernel - 1];
    if window.shape() != want {
        return Err(Error::InvalidArgument {
            arg: "window",
            reason: format!("expected {want:?}, got {:?}", window.shape()),
        });
    }
    Ok((batch, channels, kernel))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// Prefill of 6 tokens equals 6 carried single steps, 3 channels, kernel 4.
    #[test]
    fn prefill_matches_carried_steps() {
        let (client, device) = cpu_setup();
        let (batch, channels, kernel, seq) = (1, 3, 4, 6);

        let x: Vec<f32> = (0..batch * channels * seq)
            .map(|i| ((i * 7) % 11) as f32 * 0.25 - 1.0)
            .collect();
        let w: Vec<f32> = (0..channels * kernel)
            .map(|i| ((i * 5) % 9) as f32 * 0.125 - 0.5)
            .collect();
        let b: Vec<f32> = (0..channels).map(|c| c as f32 * 0.1).collect();
        let x = Tensor::<CpuRuntime>::from_slice(&x, &[batch, channels, seq], &device).unwrap();
        let w = Tensor::<CpuRuntime>::from_slice(&w, &[channels, 1, kernel], &device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&b, &[channels], &device).unwrap();
        let zero = Tensor::<CpuRuntime>::zeros(&[batch, channels, kernel - 1], DType::F32, &device)
            .unwrap();

        let (full, full_window) = causal_conv1d(&client, &x, &w, Some(&b), &zero).unwrap();
        assert_eq!(full.shape(), &[batch, channels, seq]);
        assert_eq!(full_window.shape(), &[batch, channels, kernel - 1]);

        let mut window = zero;
        let mut steps = Vec::with_capacity(seq);
        for t in 0..seq {
            let xt = x.narrow(2, t, 1).unwrap().contiguous().unwrap();
            let (o, next) = causal_conv1d(&client, &xt, &w, Some(&b), &window).unwrap();
            assert_eq!(o.shape(), &[batch, channels, 1]);
            steps.push(o);
            window = next;
        }
        let refs: Vec<&Tensor<CpuRuntime>> = steps.iter().collect();
        let stepped = client.cat(&refs, 2).unwrap();

        let a = full.to_vec::<f32>();
        let s = stepped.to_vec::<f32>();
        for (i, (p, q)) in a.iter().zip(&s).enumerate() {
            assert!((p - q).abs() < 1e-5, "idx={i}: prefill {p} step {q}");
        }
        let fw = full_window.to_vec::<f32>();
        let sw = window.to_vec::<f32>();
        assert_eq!(fw, sw);
        // The window is the last kernel-1 inputs.
        let tail = x.narrow(2, seq - (kernel - 1), kernel - 1).unwrap();
        assert_eq!(fw, tail.contiguous().unwrap().to_vec::<f32>());
    }

    /// A short prefill keeps the old window's tail in front of the new tokens.
    #[test]
    fn short_prefill_keeps_old_window_tail() {
        let (client, device) = cpu_setup();
        let w = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 1, 4], &device)
            .unwrap();
        let window =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 1, 3], &device).unwrap();
        let x = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0], &[1, 1, 2], &device).unwrap();
        let (out, next) = causal_conv1d(&client, &x, &w, None, &window).unwrap();
        assert_eq!(out.to_vec::<f32>(), vec![16.0, 35.0]);
        assert_eq!(next.to_vec::<f32>(), vec![3.0, 10.0, 20.0]);
    }

    #[test]
    fn rejects_window_shape_mismatch() {
        let (client, device) = cpu_setup();
        let w = Tensor::<CpuRuntime>::zeros(&[2, 1, 4], DType::F32, &device).unwrap();
        let x = Tensor::<CpuRuntime>::zeros(&[1, 2, 5], DType::F32, &device).unwrap();
        let window = Tensor::<CpuRuntime>::zeros(&[1, 2, 2], DType::F32, &device).unwrap();
        assert!(causal_conv1d(&client, &x, &w, None, &window).is_err());
    }
}
