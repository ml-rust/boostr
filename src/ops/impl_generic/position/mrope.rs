//! Interleaved multi-section RoPE (IMROPE) — impl_generic.
//!
//! The rule is documented on [`MRopeOps`](crate::ops::traits::position::MRopeOps).
//! This file composes it from numr ops: one `embedding_lookup` per table
//! gathers all four streams at once, a one-hot stream selector picks the
//! stream each pair reads, and the rotation is the split-half formula on the
//! first `n_rot` dims.

use crate::error::{Error, Result};
use numr::autograd::{Var, var_add, var_cat, var_mul, var_narrow, var_sub};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Stream index (`0 = t, 1 = h, 2 = w, 3 = e`) each rotated pair reads.
///
/// Port of the `is_imrope` branch of `ggml_mrope_cache_init`:
/// `sector = pair % sum(sections)`, then `sector % 3` picks `t`/`h`/`w`
/// when `sector < 3 * sections[stream]`, else `e`.
///
/// # Errors
///
/// `InvalidArgument` when `sections` sum to zero (ggml asserts
/// `sections[0] > 0 || sections[1] > 0 || sections[2] > 0`).
pub fn mrope_pair_streams(sections: [usize; 4], half_rot: usize) -> Result<Vec<usize>> {
    let sect_dims: usize = sections.iter().sum();
    if sect_dims == 0 {
        return Err(Error::InvalidArgument {
            arg: "sections",
            reason: "sections must not all be zero".into(),
        });
    }
    Ok((0..half_rot)
        .map(|pair| {
            let sector = pair % sect_dims;
            match sector % 3 {
                1 if sector < 3 * sections[1] => 1,
                2 if sector < 3 * sections[2] => 2,
                0 if sector < 3 * sections[0] => 0,
                _ => 3,
            }
        })
        .collect())
}

/// Build the one-hot stream selector `[4, 1, half_rot]`: `sel[s, 0, i] = 1`
/// when pair `i` reads stream `s`. Pure function of `sections` and
/// `half_rot` — build it once per layer and reuse it across calls to
/// [`apply_mrope_interleaved_impl`].
///
/// # Errors
///
/// `InvalidArgument` when `sections` sum to zero or sum above `half_rot`.
pub fn mrope_stream_selector<R: Runtime<DType = DType>>(
    sections: [usize; 4],
    half_rot: usize,
    device: &R::Device,
) -> Result<Tensor<R>> {
    let sect_dims: usize = sections.iter().sum();
    if sect_dims > half_rot {
        return Err(Error::InvalidArgument {
            arg: "sections",
            reason: format!("sections sum to {sect_dims}, must be at most half_rot={half_rot}"),
        });
    }
    let streams = mrope_pair_streams(sections, half_rot)?;
    let mut sel = vec![0.0f32; 4 * half_rot];
    for (pair, &stream) in streams.iter().enumerate() {
        sel[stream * half_rot + pair] = 1.0;
    }
    Tensor::<R>::from_slice(&sel, &[4, 1, half_rot], device).map_err(Error::Numr)
}

/// Apply interleaved multi-section RoPE.
///
/// - `x`: `[batch, seq, heads, head_dim]`
/// - `cos_cache`, `sin_cache`: `[max_pos, n_rot / 2]`
/// - `positions`: `[4, seq]`, I32 or I64
/// - `selector`: `[4, 1, n_rot / 2]` one-hot stream selector from
///   [`mrope_stream_selector`], built once per layer
///
/// See [`MRopeOps`](crate::ops::traits::position::MRopeOps) for the rule.
pub fn apply_mrope_interleaved_impl<R, C>(
    client: &C,
    x: &Var<R>,
    cos_cache: &Var<R>,
    sin_cache: &Var<R>,
    positions: &Tensor<R>,
    selector: &Tensor<R>,
    n_rot: usize,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TypeConversionOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + BinaryOps<R>,
    R::Client: TensorOps<R> + ShapeOps<R> + TypeConversionOps<R>,
{
    let x_shape = x.tensor().shape().to_vec();
    if x_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!(
                "expected 4D [batch, seq, heads, head_dim], got {}D",
                x_shape.len()
            ),
        });
    }
    let seq = x_shape[1];
    let head_dim = x_shape[3];
    if n_rot == 0 || !n_rot.is_multiple_of(2) || n_rot > head_dim {
        return Err(Error::InvalidArgument {
            arg: "n_rot",
            reason: format!("n_rot={n_rot} must be even, nonzero and at most head_dim={head_dim}"),
        });
    }
    let half_rot = n_rot / 2;
    for (arg, cache) in [("cos_cache", cos_cache), ("sin_cache", sin_cache)] {
        let shape = cache.tensor().shape();
        if shape.len() != 2 || shape[1] != half_rot {
            return Err(Error::InvalidArgument {
                arg,
                reason: format!("expected [max_pos, {half_rot}], got {shape:?}"),
            });
        }
    }
    let pos_shape = positions.shape();
    if pos_shape != [4, seq] {
        return Err(Error::InvalidArgument {
            arg: "positions",
            reason: format!("expected [4, seq={seq}], got {pos_shape:?}"),
        });
    }
    let sel_shape = selector.shape();
    if sel_shape != [4, 1, half_rot] {
        return Err(Error::InvalidArgument {
            arg: "selector",
            reason: format!("expected [4, 1, half_rot={half_rot}], got {sel_shape:?}"),
        });
    }
    let x_dtype = x.tensor().dtype();

    // Gather every stream's row at once: `[4, seq, half_rot]`, then keep the
    // selected stream per pair: `[seq, half_rot]`.
    let select = |cache: &Var<R>| -> Result<Var<R>> {
        let rows = client.embedding_lookup(cache.tensor(), positions)?;
        let sel = if selector.dtype() == rows.dtype() {
            selector.clone()
        } else {
            client.cast(selector, rows.dtype())?
        };
        let picked = client.mul(&rows, &sel)?;
        let picked = client.sum(&picked, &[0], false)?;
        let picked = if picked.dtype() == x_dtype {
            picked
        } else {
            client.cast(&picked, x_dtype)?
        };
        let picked = picked.reshape(&[1, seq, 1, half_rot])?;
        Ok(Var::new(picked, false))
    };
    let cos = select(cos_cache)?;
    let sin = select(sin_cache)?;

    let x1 = var_narrow(x, -1, 0, half_rot).map_err(Error::Numr)?;
    let x2 = var_narrow(x, -1, half_rot, half_rot).map_err(Error::Numr)?;

    let x1_cos = var_mul(&x1, &cos, client).map_err(Error::Numr)?;
    let x2_sin = var_mul(&x2, &sin, client).map_err(Error::Numr)?;
    let out1 = var_sub(&x1_cos, &x2_sin, client).map_err(Error::Numr)?;

    let x1_sin = var_mul(&x1, &sin, client).map_err(Error::Numr)?;
    let x2_cos = var_mul(&x2, &cos, client).map_err(Error::Numr)?;
    let out2 = var_add(&x1_sin, &x2_cos, client).map_err(Error::Numr)?;

    if n_rot == head_dim {
        return var_cat(&[&out1, &out2], -1, client).map_err(Error::Numr);
    }
    let pass = var_narrow(x, -1, n_rot, head_dim - n_rot).map_err(Error::Numr)?;
    var_cat(&[&out1, &out2, &pass], -1, client).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::impl_generic::attention::apply_rope_impl;
    use crate::test_utils::cpu_setup;
    use numr::autograd::var_permute;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    const SEQ: usize = 5;
    const HEADS: usize = 2;
    const HD: usize = 8;
    const N_ROT: usize = 4;
    const BASE: f32 = 10_000.0;
    const MAX_POS: usize = 16;

    fn freq(pair: usize) -> f32 {
        1.0 / BASE.powf(2.0 * pair as f32 / N_ROT as f32)
    }

    fn tables(device: &CpuDevice) -> (Var<CpuRuntime>, Var<CpuRuntime>) {
        let half = N_ROT / 2;
        let mut cos = vec![0.0f32; MAX_POS * half];
        let mut sin = vec![0.0f32; MAX_POS * half];
        for p in 0..MAX_POS {
            for i in 0..half {
                let a = p as f32 * freq(i);
                cos[p * half + i] = a.cos();
                sin[p * half + i] = a.sin();
            }
        }
        let t = |d: &[f32]| {
            Var::new(
                Tensor::<CpuRuntime>::from_slice(d, &[MAX_POS, half], device).unwrap(),
                false,
            )
        };
        (t(&cos), t(&sin))
    }

    fn input(device: &CpuDevice) -> (Vec<f32>, Var<CpuRuntime>) {
        let data: Vec<f32> = (0..SEQ * HEADS * HD)
            .map(|i| ((i as f32) * 0.37).sin())
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[1, SEQ, HEADS, HD], device).unwrap(),
            false,
        );
        (data, x)
    }

    fn positions(device: &CpuDevice, streams: [&[i32]; 4]) -> Tensor<CpuRuntime> {
        let data: Vec<i32> = streams.iter().flat_map(|s| s.iter().copied()).collect();
        Tensor::<CpuRuntime>::from_slice(&data, &[4, SEQ], device).unwrap()
    }

    #[test]
    fn pair_streams_follow_ggml_interleave() {
        // qwen35: [11, 11, 10, 0], 32 pairs -> t,h,w,t,h,w,... no `e`.
        let s = mrope_pair_streams([11, 11, 10, 0], 32).unwrap();
        for (pair, &stream) in s.iter().enumerate() {
            assert_eq!(stream, pair % 3, "pair {pair}");
        }
        assert_eq!(mrope_pair_streams([1, 1, 0, 0], 2).unwrap(), vec![0, 1]);
        // sector 1: 1 % 3 == 1 but 1 < 3 * sections[1] = 0 fails -> `e`.
        assert_eq!(mrope_pair_streams([1, 0, 1, 0], 2).unwrap(), vec![0, 3]);
        assert!(mrope_pair_streams([0, 0, 0, 0], 2).is_err());
    }

    #[test]
    fn equal_streams_match_partial_neox_rope() {
        let (client, device) = cpu_setup();
        let (cos, sin) = tables(&device);
        let (_, x) = input(&device);
        let t: Vec<i32> = (0..SEQ as i32).collect();
        let pos = positions(&device, [&t, &t, &t, &t]);

        for sections in [[1usize, 1, 0, 0], [1, 0, 1, 0]] {
            let sel = mrope_stream_selector::<CpuRuntime>(sections, N_ROT / 2, &device).unwrap();
            let out =
                apply_mrope_interleaved_impl(&client, &x, &cos, &sin, &pos, &sel, N_ROT).unwrap();
            let got: Vec<f32> = out.tensor().contiguous().unwrap().to_vec();

            // Reference: plain split-half RoPE on x[.., ..N_ROT] in [B, H, S, D].
            let xr = var_narrow(&x, -1, 0, N_ROT).unwrap();
            let xr = var_permute(&xr, &[0, 2, 1, 3]).unwrap();
            let xr = Var::new(xr.tensor().contiguous().unwrap(), false);
            // `apply_rope_impl` narrows the table to the sequence itself.
            let rot = apply_rope_impl(&client, &xr, &cos, &sin).unwrap();
            let rot = var_permute(&rot, &[0, 2, 1, 3]).unwrap();
            let rot: Vec<f32> = rot.tensor().contiguous().unwrap().to_vec();
            let x_data: Vec<f32> = x.tensor().to_vec();

            for s in 0..SEQ {
                for h in 0..HEADS {
                    for d in 0..HD {
                        let i = (s * HEADS + h) * HD + d;
                        let want = if d < N_ROT {
                            rot[(s * HEADS + h) * N_ROT + d]
                        } else {
                            x_data[i]
                        };
                        assert!(
                            (got[i] - want).abs() < 1e-5,
                            "sections {sections:?} s={s} h={h} d={d}: got {} want {want}",
                            got[i]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn distinct_streams_pick_t_then_h() {
        let (client, device) = cpu_setup();
        let (cos, sin) = tables(&device);
        let (data, x) = input(&device);
        let t: Vec<i32> = (0..SEQ as i32).collect();
        let h: Vec<i32> = t.iter().map(|p| p + 3).collect();
        let w: Vec<i32> = t.iter().map(|p| p + 7).collect();
        let e = vec![0i32; SEQ];
        let pos = positions(&device, [&t, &h, &w, &e]);

        let sel = mrope_stream_selector::<CpuRuntime>([1, 1, 0, 0], N_ROT / 2, &device).unwrap();
        let out = apply_mrope_interleaved_impl(&client, &x, &cos, &sin, &pos, &sel, N_ROT).unwrap();
        let got: Vec<f32> = out.tensor().contiguous().unwrap().to_vec();

        // Token 2, head 1: pair 0 at position t=2, pair 1 at position h=5.
        let base = (2 * HEADS + 1) * HD;
        let (x0, x1, x2, x3) = (data[base], data[base + 1], data[base + 2], data[base + 3]);
        let a0 = 2.0 * freq(0);
        let a1 = 5.0 * freq(1);
        let want = [
            x0 * a0.cos() - x2 * a0.sin(),
            x1 * a1.cos() - x3 * a1.sin(),
            x0 * a0.sin() + x2 * a0.cos(),
            x1 * a1.sin() + x3 * a1.cos(),
        ];
        for (d, want) in want.iter().enumerate() {
            assert!(
                (got[base + d] - want).abs() < 1e-5,
                "d={d}: got {} want {want}",
                got[base + d]
            );
        }
        for d in N_ROT..HD {
            assert_eq!(got[base + d], data[base + d], "pass-through d={d}");
        }
    }

    #[test]
    fn fall_through_sector_reads_e_stream() {
        let (client, device) = cpu_setup();
        let (cos, sin) = tables(&device);
        let (data, x) = input(&device);
        let t: Vec<i32> = (0..SEQ as i32).collect();
        let w: Vec<i32> = t.iter().map(|p| p + 7).collect();
        let e: Vec<i32> = t.iter().map(|p| p + 9).collect();
        let pos = positions(&device, [&t, &t, &w, &e]);

        let sel = mrope_stream_selector::<CpuRuntime>([1, 0, 1, 0], N_ROT / 2, &device).unwrap();
        let out = apply_mrope_interleaved_impl(&client, &x, &cos, &sin, &pos, &sel, N_ROT).unwrap();
        let got: Vec<f32> = out.tensor().contiguous().unwrap().to_vec();

        // Token 1, head 0: pair 1 is sector 1, which ggml sends to `e` = 10.
        let base = HEADS * HD;
        let (x1, x3) = (data[base + 1], data[base + 3]);
        let a1 = 10.0 * freq(1);
        assert!((got[base + 1] - (x1 * a1.cos() - x3 * a1.sin())).abs() < 1e-5);
        assert!((got[base + 3] - (x1 * a1.sin() + x3 * a1.cos())).abs() < 1e-5);
    }

    #[test]
    fn rejects_bad_shapes() {
        let (client, device) = cpu_setup();
        let (cos, sin) = tables(&device);
        let (_, x) = input(&device);
        let t: Vec<i32> = (0..SEQ as i32).collect();
        let pos = positions(&device, [&t, &t, &t, &t]);
        let sel = mrope_stream_selector::<CpuRuntime>([1, 1, 0, 0], N_ROT / 2, &device).unwrap();
        assert!(apply_mrope_interleaved_impl(&client, &x, &cos, &sin, &pos, &sel, 3).is_err());
        assert!(apply_mrope_interleaved_impl(&client, &x, &cos, &sin, &pos, &sel, 6).is_err());
        let short = Tensor::<CpuRuntime>::from_slice(&t, &[1, SEQ], &device).unwrap();
        assert!(
            apply_mrope_interleaved_impl(&client, &x, &cos, &sin, &short, &sel, N_ROT).is_err()
        );
        let bad_sel = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4, 1, 1], &device).unwrap();
        assert!(
            apply_mrope_interleaved_impl(&client, &x, &cos, &sin, &pos, &bad_sel, N_ROT).is_err()
        );
    }
}
