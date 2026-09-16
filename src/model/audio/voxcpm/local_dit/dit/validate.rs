use crate::error::{Error, Result};
use crate::model::audio::voxcpm::local_dit::loader::LocalDit;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> LocalDit<R> {
    /// Validate a `[batch, patch_size, feat_dim]` input, returning its batch.
    /// `expected_batch` pins the batch against an earlier input.
    ///
    /// `pub(super)` so the sibling CFM sampler validates `z`/`cond` up front
    /// with the same rules, instead of waiting for the first estimator call.
    pub(in crate::model::audio::voxcpm::local_dit) fn check_patch_input(
        &self,
        arg: &'static str,
        v: &Var<R>,
        expected_batch: Option<usize>,
    ) -> Result<usize> {
        let shape = v.shape();
        if shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg,
                reason: format!(
                    "expected 3D [batch, patch_size, feat_dim], got {}D {shape:?}",
                    shape.len()
                ),
            });
        }
        if shape[1] != self.patch_size || shape[2] != self.feat_dim {
            return Err(Error::InvalidArgument {
                arg,
                reason: format!(
                    "expected [batch, {}, {}], got {shape:?}",
                    self.patch_size, self.feat_dim
                ),
            });
        }
        if let Some(batch) = expected_batch
            && shape[0] != batch
        {
            return Err(Error::InvalidArgument {
                arg,
                reason: format!("batch {} does not match x's batch {batch}", shape[0]),
            });
        }
        Ok(shape[0])
    }

    /// Validate `mu: [batch, mu_tokens * hidden_dim]` and return `mu_tokens`
    /// — derived from the width, never hardcoded to 2. `pub(super)` for the
    /// same reason as [`Self::check_patch_input`].
    pub(in crate::model::audio::voxcpm::local_dit) fn check_mu(
        &self,
        mu: &Var<R>,
        batch: usize,
    ) -> Result<usize> {
        let shape = mu.shape();
        if shape.len() != 2 || shape[0] != batch {
            return Err(Error::InvalidArgument {
                arg: "mu",
                reason: format!(
                    "expected 2D [{batch}, k * {}], got {shape:?}",
                    self.hidden_dim
                ),
            });
        }
        if shape[1] == 0 || !shape[1].is_multiple_of(self.hidden_dim) {
            return Err(Error::InvalidArgument {
                arg: "mu",
                reason: format!(
                    "width {} is not a nonzero multiple of hidden_dim {}",
                    shape[1], self.hidden_dim
                ),
            });
        }
        Ok(shape[1] / self.hidden_dim)
    }

    /// Validate a pre-projected `cond_h: [batch, patch_size, hidden_dim]`
    /// (the output of `LocalDit::project_cond`) against `x`'s batch.
    pub(super) fn check_cond_hidden(&self, cond_h: &Var<R>, batch: usize) -> Result<()> {
        let shape = cond_h.shape();
        if shape.len() != 3
            || shape[0] != batch
            || shape[1] != self.patch_size
            || shape[2] != self.hidden_dim
        {
            return Err(Error::InvalidArgument {
                arg: "cond_h",
                reason: format!(
                    "expected 3D [{batch}, {}, {}], got {shape:?}",
                    self.patch_size, self.hidden_dim
                ),
            });
        }
        Ok(())
    }

    /// Validate a pre-tokenized `mu_tok: [batch, mu_tokens, hidden_dim]` and
    /// return `mu_tokens` — read from the shape, never hardcoded. Counterpart
    /// to [`Self::check_mu`] for `forward_with_mu_tokens`, whose `mu`
    /// argument already carries the sequence-position axis [`Self::check_mu`]
    /// derives from a flat width.
    pub(super) fn check_mu_tokens(&self, mu_tok: &Var<R>, batch: usize) -> Result<usize> {
        let shape = mu_tok.shape();
        if shape.len() != 3 || shape[0] != batch || shape[2] != self.hidden_dim {
            return Err(Error::InvalidArgument {
                arg: "mu_tok",
                reason: format!(
                    "expected 3D [{batch}, mu_tokens, {}], got {shape:?}",
                    self.hidden_dim
                ),
            });
        }
        if shape[1] == 0 {
            return Err(Error::InvalidArgument {
                arg: "mu_tok",
                reason: format!("expected at least one mu token, got {shape:?}"),
            });
        }
        Ok(shape[1])
    }
}

/// Validate a `[batch]` scalar-per-sample timestep (`t` or `dt`).
pub(super) fn check_timestep<R: Runtime<DType = DType>>(
    arg: &'static str,
    v: &Var<R>,
    batch: usize,
) -> Result<()> {
    let shape = v.shape();
    if shape.len() != 1 || shape[0] != batch {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!("expected 1D [{batch}], got {shape:?}"),
        });
    }
    Ok(())
}
