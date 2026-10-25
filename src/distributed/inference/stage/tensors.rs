//! Conversion between a stage's wire payload and its device tensors.

use anyhow::{Result, bail};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::super::transport::{LayerAssignment, StagePayload};
use crate::model::LoadedModel;
use crate::model::traits::ModelClient;

/// The hidden state and residual `payload` holds, as `[1, seq_len, hidden]`
/// tensors in the model's dtype. Token ids go through the embedding.
///
/// # Errors
///
/// Token ids on a stage without the embedding, an activation on the stage
/// with it, a size that does not match `seq_len`, or a failed upload.
pub(super) fn input_vars<R>(
    model: &LoadedModel<R>,
    device: &R::Device,
    assignment: &LayerAssignment,
    seq_len: usize,
    payload: StagePayload,
) -> Result<(Var<R>, Option<Var<R>>)>
where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R> + TypeConversionOps<R>,
{
    match (payload, assignment.has_embedding) {
        (StagePayload::Tokens(ids), true) => {
            if ids.len() != seq_len {
                bail!(
                    "the frame covers {seq_len} positions and holds {} token ids",
                    ids.len()
                );
            }
            let ids = Tensor::<R>::from_slice(&ids, &[1, seq_len], device)?;
            Ok((model.forward_embed(&ids)?, None))
        }
        (StagePayload::Activation(a), false) => {
            let hidden = model.hidden_size();
            let want = seq_len * hidden;
            let mlp_len = a.mlp.as_ref().map_or(want, Vec::len);
            if a.hidden.len() != want || mlp_len != want {
                bail!(
                    "{seq_len} positions of hidden size {hidden} need {want} values, and the \
                     activation holds {} hidden and {} residual values",
                    a.hidden.len(),
                    a.mlp.as_ref().map_or(0, Vec::len)
                );
            }
            let mlp = a
                .mlp
                .map(|m| device_var(model, device, &m, seq_len))
                .transpose()?;
            Ok((device_var(model, device, &a.hidden, seq_len)?, mlp))
        }
        (StagePayload::Tokens(_), false) => bail!(
            "stage layers {}..{} has no embedding and takes an activation, not token ids",
            assignment.start_layer,
            assignment.end_layer
        ),
        (StagePayload::Activation(_), true) => {
            bail!("the stage with the embedding takes token ids, not an activation")
        }
    }
}

/// `values` as a `[1, seq_len, hidden]` tensor in the model's dtype.
fn device_var<R>(
    model: &LoadedModel<R>,
    device: &R::Device,
    values: &[f32],
    seq_len: usize,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R> + TypeConversionOps<R>,
{
    let t = Tensor::<R>::from_slice(values, &[1, seq_len, model.hidden_size()], device)?;
    let dtype = model.dtype();
    let t = if dtype == DType::F32 {
        t
    } else {
        R::default_client(device).cast(&t, dtype)?
    };
    Ok(Var::new(t, false))
}

/// `tensor` on the host as `f32` values in row-major order. Another dtype
/// casts on its device first.
pub(super) fn host_f32<R>(tensor: &Tensor<R>) -> Result<Vec<f32>>
where
    R: Runtime<DType = DType>,
    R::Client: TypeConversionOps<R>,
{
    let wide = if tensor.dtype() == DType::F32 {
        tensor.clone()
    } else {
        R::default_client(tensor.device()).cast(tensor, DType::F32)?
    };
    let wide = if wide.is_contiguous() {
        wide
    } else {
        wide.contiguous()?
    };
    Ok(wide.try_to_vec::<f32>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_host_f32_reads_a_narrowed_view_and_a_half_tensor() {
        let (client, device) = cpu_setup();
        let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2], &device)
            .unwrap();
        assert_eq!(host_f32(&t.narrow(1, 1, 1).unwrap()).unwrap(), [3.0, 4.0]);
        let half = client.cast(&t, DType::F16).unwrap();
        assert_eq!(host_f32(&half).unwrap(), [1.0, 2.0, 3.0, 4.0]);
    }
}
