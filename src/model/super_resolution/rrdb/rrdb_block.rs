//! [`RrdbBlock`]: three residual dense blocks under one scaled residual.

use crate::error::{Error, Result};
use crate::model::super_resolution::client::SrClient;
use crate::model::super_resolution::layout::SrExecution;
use crate::model::super_resolution::rrdb::config::RrdbConfig;
use crate::model::super_resolution::rrdb::dense_block::{RESIDUAL_SCALE, ResidualDenseBlock};
use crate::nn::VarBuilder;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One RRDB: `x + 0.2 · rdb3(rdb2(rdb1(x)))`.
///
/// The outer residual runs as `mul_scalar` then `add`. It cannot fuse into
/// `rdb3.conv5`: that conv already adds the residual of `rdb3`, and
/// `x + 0.2 · (y + 0.2 · z)` rounds differently from any single fused
/// epilogue.
pub struct RrdbBlock<R: Runtime> {
    name: String,
    rdb1: ResidualDenseBlock<R>,
    rdb2: ResidualDenseBlock<R>,
    rdb3: ResidualDenseBlock<R>,
    exec: SrExecution,
}

impl<R: Runtime<DType = DType>> RrdbBlock<R> {
    /// Take `rdb1`, `rdb2` and `rdb3` under `vb`'s prefix, such as
    /// `body.0`.
    ///
    /// # Errors
    ///
    /// A missing key or a shape mismatch. The error names the key.
    pub fn from_varbuilder<C: SrClient<R>>(
        client: &C,
        vb: &mut VarBuilder<'_, R>,
        config: &RrdbConfig,
        exec: &SrExecution,
    ) -> Result<Self> {
        let name = vb.prefix().to_string();
        let rdb1 = ResidualDenseBlock::from_varbuilder(client, &mut vb.pp("rdb1"), config, exec)?;
        let rdb2 = ResidualDenseBlock::from_varbuilder(client, &mut vb.pp("rdb2"), config, exec)?;
        let rdb3 = ResidualDenseBlock::from_varbuilder(client, &mut vb.pp("rdb3"), config, exec)?;
        Ok(Self {
            name,
            rdb1,
            rdb2,
            rdb3,
            exec: *exec,
        })
    }

    /// Run the block on `x` and return a new contiguous tensor.
    ///
    /// `x` has `num_feat` channels in the execution layout. `buffers` are
    /// two feature buffers of `num_feat + 4 · num_grow_ch` channels at the
    /// resolution of `x`, in separate storage. Their contents on entry do
    /// not matter. The call overwrites both. `x` must not share storage with
    /// either buffer.
    ///
    /// # Errors
    ///
    /// A buffer of the wrong shape, or a conv or elementwise error. Each
    /// error names the conv or the op and the shapes.
    pub fn forward<C: SrClient<R>>(
        &self,
        client: &C,
        x: &Tensor<R>,
        buffers: &[Tensor<R>; 2],
    ) -> Result<Tensor<R>> {
        let [a, b] = buffers;
        let (_, feat, _, _) = self.exec.image_dims(x.shape())?;
        let head = a.narrow(self.exec.channel_dim() as isize, 0, feat)?;
        client
            .copy_into(&head, x)
            .map_err(|e| self.op_error("copy of the block input into the feature buffer", x, e))?;
        self.rdb1.forward_into(client, a, b)?;
        self.rdb2.forward_into(client, b, a)?;
        let inner = self.rdb3.forward(client, a)?;
        let scaled = client
            .mul_scalar(&inner, RESIDUAL_SCALE)
            .map_err(|e| self.op_error("residual scale", x, e))?;
        client
            .add(x, &scaled)
            .map_err(|e| self.op_error("residual add", x, e))
    }

    /// Full checkpoint key prefix of the block, such as `body.4`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `rdb1`, `rdb2` and `rdb3` in order.
    pub fn dense_blocks(&self) -> [&ResidualDenseBlock<R>; 3] {
        [&self.rdb1, &self.rdb2, &self.rdb3]
    }

    fn op_error(&self, op: &str, x: &Tensor<R>, e: numr::error::Error) -> Error {
        Error::model_op(
            format!(
                "RRDB '{}' {op} on input {:?} {:?}: {e}",
                self.name,
                x.shape(),
                x.dtype()
            ),
            e,
        )
    }
}
