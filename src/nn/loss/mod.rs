//! Loss functions for neural network training.
//!
//! | Function | Use case |
//! |---|---|
//! | [`cross_entropy_loss`] | Multi-class classification |
//! | [`cross_entropy_loss_smooth`] | Classification with label smoothing (regularization) |
//! | [`focal_loss`] | Class-imbalanced classification (object detection) |
//! | [`mse_loss`] | Regression |
//! | [`kl_div_loss`] | Knowledge distillation, distribution matching |
//! | [`contrastive_loss`] | Contrastive / self-supervised representation learning |

pub mod contrastive;
pub mod cross_entropy;
pub mod focal;
pub mod kl_div;
pub mod mse;

pub use contrastive::contrastive_loss;
pub use cross_entropy::{cross_entropy_loss, cross_entropy_loss_smooth};
pub use focal::focal_loss;
pub use kl_div::kl_div_loss;
pub use mse::mse_loss;

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Flatten targets to [N, 1] for gather operations.
fn prepare_targets<R: Runtime<DType = DType>>(
    targets: &Tensor<R>,
    batch_size: usize,
) -> Result<Tensor<R>> {
    targets
        .reshape(&[batch_size])
        .and_then(|t| t.unsqueeze(1))
        .and_then(|t| t.broadcast_to(&[batch_size, 1]))
        .map_err(Error::Numr)
}

/// All dimension indices for a tensor (for full reductions).
fn all_dims(ndim: usize) -> Vec<usize> {
    (0..ndim).collect()
}

/// Compute batch size from shape (product of all dims except last).
fn batch_size(shape: &[usize]) -> usize {
    if shape.len() <= 1 {
        1
    } else {
        shape[..shape.len() - 1].iter().product()
    }
}
