//! [`ClapProjection`]: a tower's two-layer projection into the shared
//! embedding space, and the L2 normalization after it. Port of
//! transformers' `ClapProjectionLayer` and `F.normalize(x, dim=-1)`.

use super::linear::{dense, linear};
use crate::error::Result;
use crate::nn::{Activation, Linear, VarBuilder};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, ScalarOps, TensorOps, UnaryOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// `F.normalize`'s floor on the norm.
const NORM_EPS: f64 = 1e-12;

/// `linear2(act(linear1(x)))`.
pub struct ClapProjection<R: Runtime> {
    linear1: Linear<R>,
    linear2: Linear<R>,
    act: Activation,
}

impl<R: Runtime<DType = DType>> ClapProjection<R> {
    /// Reads `linear1.*` and `linear2.*` under `vb`.
    pub fn from_varbuilder(vb: &mut VarBuilder<'_, R>, act: Activation) -> Result<Self> {
        Ok(Self {
            linear1: linear(vb, "linear1", true)?,
            linear2: linear(vb, "linear2", true)?,
            act,
        })
    }

    /// The unit-length embeddings of `pooled` (`[batch, hidden]`):
    /// `[batch, projection_dim]`.
    pub fn embed<C>(&self, client: &C, pooled: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ActivationOps<R>
            + UnaryOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + UtilityOps<R>,
        R::Client: TensorOps<R>,
    {
        let h = dense(client, &self.linear1, pooled)?;
        let h = self.act.forward(client, &h)?;
        let h = dense(client, &self.linear2, &h)?;
        let norm = client.sqrt(&client.sum(&client.mul(&h, &h)?, &[1], true)?)?;
        let norm = client.clamp(&norm, NORM_EPS, f64::INFINITY)?;
        Ok(client.div(&h, &norm)?)
    }
}
