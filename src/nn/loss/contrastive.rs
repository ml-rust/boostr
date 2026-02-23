//! InfoNCE / contrastive loss (CLIP, SimCLR, retrieval models).

use super::all_dims;
use crate::error::{Error, Result};
use numr::autograd::{
    Var, var_gather, var_log_softmax, var_matmul, var_mean, var_mul_scalar, var_neg, var_transpose,
};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, IndexingOps, MatmulOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// InfoNCE / contrastive loss
///
/// `loss = -mean(diag(log_softmax(similarity / temperature)))`
///
/// where `similarity[i,j] = dot(query[i], key[j])`.
///
/// - `queries`: `[N, D]` query embeddings (L2-normalized recommended)
/// - `keys`: `[N, D]` key embeddings (positive pairs aligned by index)
/// - `temperature`: scaling factor (typically 0.07 for CLIP)
pub fn contrastive_loss<R, C>(
    client: &C,
    queries: &Var<R>,
    keys: &Var<R>,
    temperature: f64,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + IndexingOps<R>
        + MatmulOps<R>,
    R::Client: ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + IndexingOps<R>
        + MatmulOps<R>,
{
    if queries.shape().len() != 2 || keys.shape().len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "queries/keys",
            reason: "expected 2D tensors [N, D]".to_string(),
        });
    }
    if queries.shape() != keys.shape() {
        return Err(Error::InvalidArgument {
            arg: "keys",
            reason: format!(
                "shape mismatch: queries {:?} vs keys {:?}",
                queries.shape(),
                keys.shape()
            ),
        });
    }

    let n = queries.shape()[0];

    // similarity = queries @ keys^T / temperature -> [N, N]
    let keys_t = var_transpose(keys).map_err(Error::Numr)?;
    let sim = var_matmul(queries, &keys_t, client).map_err(Error::Numr)?;
    let sim_scaled = var_mul_scalar(&sim, 1.0 / temperature, client).map_err(Error::Numr)?;

    // log_softmax along rows (dim=1): each query's distribution over keys
    let log_probs = var_log_softmax(&sim_scaled, -1, client).map_err(Error::Numr)?;

    // Targets: diagonal (positive pair for query i is key i)
    let targets = Tensor::<R>::from_slice(
        &(0..n as i64).collect::<Vec<_>>(),
        &[n],
        queries.tensor().device(),
    );
    let targets_expanded = targets
        .unsqueeze(1)
        .map_err(Error::Numr)?
        .broadcast_to(&[n, 1])
        .map_err(Error::Numr)?;

    // Gather diagonal: log_probs[i, i]
    let selected = var_gather(&log_probs, 1, &targets_expanded, client).map_err(Error::Numr)?;

    // loss = -mean(selected)
    let neg = var_neg(&selected, client).map_err(Error::Numr)?;
    let loss = var_mean(&neg, &all_dims(neg.shape().len()), false, client).map_err(Error::Numr)?;

    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_contrastive_loss_basic() {
        let (client, device) = cpu_setup();

        let embeddings = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[
                    1.0f32, 0.0, 0.0, 0.0, // embed 0
                    0.0, 1.0, 0.0, 0.0, // embed 1
                    0.0, 0.0, 1.0, 0.0, // embed 2
                ],
                &[3, 4],
                &device,
            ),
            false,
        );

        let loss = contrastive_loss(&client, &embeddings, &embeddings, 0.07).unwrap();
        let val: Vec<f32> = loss.tensor().to_vec();
        assert!(val[0].is_finite(), "loss should be finite, got {}", val[0]);
        assert!(val[0] >= 0.0, "contrastive loss should be >= 0");
    }

    #[test]
    fn test_contrastive_loss_shape_validation() {
        let (client, device) = cpu_setup();

        let q = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device),
            false,
        );
        let k = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device),
            false,
        );
        assert!(contrastive_loss(&client, &q, &k, 0.07).is_err());
    }
}
