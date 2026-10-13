//! The per-tensor edits of [`restructure`](super::restructure): the
//! value-head reorder of one Gated DeltaNet layer, `-exp(A_log)`, and the
//! `1 + w` norm offset.
//!
//! Every edit stays on the tensor's device and returns the tensor's dtype.
//! The reorder is a permutation in that dtype. `1 + w` and `-exp(A_log)`
//! compute in F32 through [`in_f32`], as the llama.cpp converter does, so
//! an F16 or BF16 result rounds once.

use super::reorder::{reorder_qkv_v_rows, reorder_v_heads};
use crate::error::{Error, Result};
use crate::model::config::GdnConfig;
use crate::nn::{VarMap, Weight};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ShapeOps, TypeConversionOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) fn model_error(reason: String) -> Error {
    Error::ModelError { reason }
}

/// Replace the dense tensor `name` with `f` of it.
pub(super) fn edit<R: Runtime<DType = DType>>(
    map: &mut VarMap<R>,
    name: &str,
    f: impl FnOnce(&Tensor<R>) -> Result<Tensor<R>>,
) -> Result<()> {
    match map.remove(name) {
        Some(Weight::Standard(t)) => {
            let edited =
                f(&t).map_err(|e| model_error(format!("qwen35: editing {name} failed: {e}")))?;
            map.insert(name.to_string(), edited);
            Ok(())
        }
        Some(_) => Err(model_error(format!(
            "qwen35: {name} is quantized; the Hugging Face layout edits need F32, F16 or BF16"
        ))),
        None => Err(model_error(format!(
            "qwen35: the Hugging Face checkpoint has no {name}"
        ))),
    }
}

/// `f(t)` computed in F32, returned in `t`'s dtype.
///
/// A float tensor narrower than F32 (F16, BF16, FP8) is cast to F32 on
/// its device, edited, and cast back once. Any other tensor, F32 and F64
/// among them, is edited in its own dtype.
pub(super) fn in_f32<R: Runtime<DType = DType>>(
    t: &Tensor<R>,
    f: impl FnOnce(&Tensor<R>) -> Result<Tensor<R>>,
) -> Result<Tensor<R>>
where
    R::Client: TypeConversionOps<R>,
{
    let dtype = t.dtype();
    if !dtype.is_float() || dtype.size_in_bytes() >= DType::F32.size_in_bytes() {
        return f(t);
    }
    let client = R::default_client(t.device());
    let wide = client.cast(t, DType::F32)?;
    Ok(client.cast(&f(&wide)?, dtype)?)
}

/// `1 + w`, computed in F32 by [`in_f32`], in `w`'s dtype.
pub(super) fn plus_one<R: Runtime<DType = DType>>(w: &Tensor<R>) -> Result<Tensor<R>>
where
    R::Client: BinaryOps<R> + TypeConversionOps<R>,
{
    in_f32(w, |w| {
        let one = Tensor::<R>::full_scalar(&[1], w.dtype(), 1.0, w.device())?;
        Ok(R::default_client(w.device()).add(w, &one)?)
    })
}

/// `-exp(a_log)`, computed in F32 by [`in_f32`], in `a_log`'s dtype.
fn neg_exp<R: Runtime<DType = DType>>(a_log: &Tensor<R>) -> Result<Tensor<R>>
where
    R::Client: UnaryOps<R> + TypeConversionOps<R>,
{
    in_f32(a_log, |a_log| {
        let client = R::default_client(a_log.device());
        Ok(client.neg(&client.exp(a_log)?)?)
    })
}

/// The value-head reorder and the `A_log` edit of the GDN layer whose
/// tensors sit under `prefix` (`model.layers.{i}.linear_attn`).
pub(super) fn edit_gdn_layer<R: Runtime<DType = DType>>(
    map: &mut VarMap<R>,
    prefix: &str,
    gdn: &GdnConfig,
) -> Result<()>
where
    R::Client: ShapeOps<R> + UnaryOps<R> + TypeConversionOps<R>,
{
    let (h_k, rep, s) = (gdn.key_heads, gdn.head_repeat(), gdn.state_size);
    for name in ["in_proj_qkv.weight", "conv1d.weight"] {
        edit(map, &format!("{prefix}.{name}"), |t| {
            reorder_qkv_v_rows(t, gdn)
        })?;
    }
    edit(map, &format!("{prefix}.in_proj_z.weight"), |t| {
        reorder_v_heads(t, 0, h_k, rep, s)
    })?;
    for name in ["alpha_proj.weight", "beta_proj.weight", "dt_bias"] {
        edit(map, &format!("{prefix}.{name}"), |t| {
            reorder_v_heads(t, 0, h_k, rep, 1)
        })?;
    }
    edit(map, &format!("{prefix}.a_neg_exp"), |t| {
        neg_exp(&reorder_v_heads(t, 0, h_k, rep, 1)?)
    })?;
    edit(map, &format!("{prefix}.out_proj.weight"), |t| {
        reorder_v_heads(t, 1, h_k, rep, s)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{QuantFormat, QuantTensor};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn t(v: &[f32], shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
        Tensor::from_slice(v, shape, device).unwrap()
    }

    #[test]
    fn plus_one_keeps_the_dtype() {
        let device = CpuDevice::new();
        let w = t(&[0.5, -1.0, 0.0], &[3], &device);
        let out = plus_one(&w).unwrap();
        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(out.to_vec::<f32>(), vec![1.5, 0.0, 1.0]);
    }

    #[test]
    fn neg_exp_of_zero_is_minus_one() {
        let device = CpuDevice::new();
        let out = neg_exp(&t(&[0.0, 0.0], &[2], &device)).unwrap();
        assert_eq!(out.to_vec::<f32>(), vec![-1.0, -1.0]);
    }

    /// Weight-like values: norm weights near zero and `A_log` in `(-2, 3)`.
    const VALUES: [f32; 8] = [0.3, -1.7, 2.5e-3, 0.0, -0.61, 1.33, 2.71, -0.047];

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// An F32 tensor takes the ops in its own dtype, with no cast.
    #[test]
    fn f32_edits_match_the_ops_in_place() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let w = t(&VALUES, &[VALUES.len()], &device);
        let one = Tensor::<CpuRuntime>::full_scalar(&[1], DType::F32, 1.0, &device).unwrap();
        let want = client.add(&w, &one).unwrap();
        assert_eq!(
            bits(&plus_one(&w).unwrap().to_vec::<f32>()),
            bits(&want.to_vec::<f32>())
        );
        let want = client.neg(&client.exp(&w).unwrap()).unwrap();
        let got = neg_exp(&w).unwrap();
        assert_eq!(got.dtype(), DType::F32);
        assert_eq!(bits(&got.to_vec::<f32>()), bits(&want.to_vec::<f32>()));
    }

    /// A half tensor's edit equals the F32 edit of its values, rounded
    /// once to its dtype on the host: `1 + w` in host F32 arithmetic,
    /// `-exp` through the F32 path above.
    #[cfg(feature = "f16")]
    fn check_half<H: numr::dtype::Element + bytemuck::Pod>(
        dtype: DType,
        from: fn(f32) -> H,
        to: fn(H) -> f32,
    ) {
        let device = CpuDevice::new();
        let n = VALUES.len();
        let half: Vec<H> = VALUES.iter().map(|&v| from(v)).collect();
        let exact: Vec<f32> = half.iter().map(|&h| to(h)).collect();
        let w = Tensor::<CpuRuntime>::from_slice(&half, &[n], &device).unwrap();
        let round = |v: &[f32]| -> Vec<u32> { v.iter().map(|&x| to(from(x)).to_bits()).collect() };
        let host = |out: Tensor<CpuRuntime>| -> Vec<u32> {
            assert_eq!(out.dtype(), dtype);
            bits(&out.to_vec::<H>().into_iter().map(to).collect::<Vec<_>>())
        };

        let sums: Vec<f32> = exact.iter().map(|v| 1.0 + v).collect();
        assert_eq!(host(plus_one(&w).unwrap()), round(&sums), "{dtype:?} 1 + w");

        let wide = neg_exp(&t(&exact, &[n], &device)).unwrap().to_vec::<f32>();
        assert_eq!(host(neg_exp(&w).unwrap()), round(&wide), "{dtype:?} -exp");
    }

    #[cfg(feature = "f16")]
    #[test]
    fn bf16_edits_round_the_f32_result_once() {
        check_half(DType::BF16, half::bf16::from_f32, half::bf16::to_f32);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn f16_edits_round_the_f32_result_once() {
        check_half(DType::F16, half::f16::from_f32, half::f16::to_f32);
    }

    #[test]
    fn missing_and_quantized_tensors_are_errors() {
        let device = CpuDevice::new();
        let mut map = VarMap::<CpuRuntime>::new();
        let err = edit(&mut map, "a", plus_one).unwrap_err().to_string();
        assert!(err.contains("has no a"), "{err}");

        let q =
            QuantTensor::<CpuRuntime>::from_bytes(&[0u8; 34], QuantFormat::Q8_0, &[32], &device)
                .unwrap();
        map.insert_quant("q".into(), q);
        let err = edit(&mut map, "q", plus_one).unwrap_err().to_string();
        assert!(err.contains("quantized"), "{err}");
    }
}
