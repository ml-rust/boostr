//! The per-tensor edits of [`restructure`](super::restructure): the
//! value-head reorder of one Gated DeltaNet layer, `-exp(A_log)`, and the
//! `1 + w` norm offset. Every edit stays in the tensor's dtype and on its
//! device.

use super::reorder::reorder_v_heads;
use crate::error::{Error, Result};
use crate::model::config::GdnConfig;
use crate::nn::{VarMap, Weight};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ShapeOps, UnaryOps};
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

/// `1 + w`, with the one in `w`'s dtype.
pub(super) fn plus_one<R: Runtime<DType = DType>>(w: &Tensor<R>) -> Result<Tensor<R>>
where
    R::Client: BinaryOps<R>,
{
    let one = Tensor::<R>::full_scalar(&[1], w.dtype(), 1.0, w.device())?;
    Ok(R::default_client(w.device()).add(w, &one)?)
}

/// `-exp(a_log)`.
fn neg_exp<R: Runtime<DType = DType>>(a_log: &Tensor<R>) -> Result<Tensor<R>>
where
    R::Client: UnaryOps<R>,
{
    let client = R::default_client(a_log.device());
    Ok(client.neg(&client.exp(a_log)?)?)
}

/// `t` (`[q | k | v, ..]` rows) with only the V rows reordered.
fn reorder_v_tail<R: Runtime<DType = DType>>(t: &Tensor<R>, gdn: &GdnConfig) -> Result<Tensor<R>>
where
    R::Client: ShapeOps<R>,
{
    let qk = 2 * gdn.key_dim();
    let rows = t.shape().first().copied().unwrap_or(0);
    if rows != gdn.qkv_dim() {
        return Err(model_error(format!(
            "{rows} rows; q, k and v need {}",
            gdn.qkv_dim()
        )));
    }
    let head = t.narrow(0, 0, qk)?.contiguous()?;
    let tail = reorder_v_heads(
        &t.narrow(0, qk, rows - qk)?,
        0,
        gdn.key_heads,
        gdn.head_repeat(),
        gdn.state_size,
    )?;
    Ok(Tensor::cat(&[&head, &tail], 0)?)
}

/// The value-head reorder and the `A_log` edit of the GDN layer whose
/// tensors sit under `prefix` (`model.layers.{i}.linear_attn`).
pub(super) fn edit_gdn_layer<R: Runtime<DType = DType>>(
    map: &mut VarMap<R>,
    prefix: &str,
    gdn: &GdnConfig,
) -> Result<()>
where
    R::Client: ShapeOps<R> + UnaryOps<R>,
{
    let (h_k, rep, s) = (gdn.key_heads, gdn.head_repeat(), gdn.state_size);
    for name in ["in_proj_qkv.weight", "conv1d.weight"] {
        edit(map, &format!("{prefix}.{name}"), |t| reorder_v_tail(t, gdn))?;
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

    /// Two key heads, two value heads each, head width 1: key dim 2,
    /// q/k rows `0..4`, v rows `4..8`.
    fn gdn() -> GdnConfig {
        GdnConfig {
            hidden_size: 4,
            conv_kernel: 2,
            state_size: 1,
            key_heads: 2,
            value_heads: 4,
            inner_size: 4,
            rms_eps: 1e-6,
            chunk_size: 64,
            v_grouped: false,
        }
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

    #[test]
    fn only_the_v_rows_move() {
        let device = CpuDevice::new();
        let rows: Vec<f32> = (0..8).map(|r| r as f32).collect();
        let out = reorder_v_tail(&t(&rows, &[8, 1], &device), &gdn()).unwrap();
        assert_eq!(
            out.to_vec::<f32>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 7.0]
        );
        assert!(reorder_v_tail(&t(&rows[..6], &[6, 1], &device), &gdn()).is_err());
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
