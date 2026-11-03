//! [`ResidualDenseBlock`]: five convs over a growing channel concat, run
//! without a concat.
//!
//! The reference block computes `conv_k(cat(x, f_1, .., f_{k-1}))`. Here
//! `x` and every `f_k` sit side by side in one feature buffer, so the concat
//! of the first `m` channel groups is the view `buf.narrow(C, 0, m)`.
//! `conv_k` reads that view and writes `f_k` into the next channel slice of
//! the same buffer. The two views are disjoint channel ranges of one
//! storage, which numr's exact disjointness rule accepts.

use crate::error::{Error, Result};
use crate::model::super_resolution::client::SrClient;
use crate::model::super_resolution::conv::SrConv;
use crate::model::super_resolution::layout::SrExecution;
use crate::model::super_resolution::rrdb::config::{RDB_CONVS, RrdbConfig};
use crate::nn::VarBuilder;
use numr::dtype::DType;
use numr::ops::{ConvActivation, ConvResidual};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Kernel size of every RRDBNet conv.
pub(super) const KERNEL: usize = 3;

/// Negative slope of every LeakyReLU in RRDBNet.
pub(super) const LEAKY_SLOPE: f64 = 0.2;

/// Factor on every residual branch of a dense block and of an RRDB.
pub(super) const RESIDUAL_SCALE: f64 = 0.2;

/// One residual dense block: `x + 0.2 · conv5(cat(x, f_1, .., f_4))`, with
/// `f_k = leaky_relu(conv_k(cat(x, f_1, .., f_{k-1})), 0.2)`.
///
/// `conv_k` for `k` in `1..=4` maps `num_feat + (k - 1) · num_grow_ch`
/// channels to `num_grow_ch`. `conv5` maps
/// [`RrdbConfig::dense_channels`] to `num_feat`.
pub struct ResidualDenseBlock<R: Runtime> {
    /// `conv1..conv4`, each followed by LeakyReLU.
    growing: Vec<SrConv<R>>,
    conv5: SrConv<R>,
    num_feat: usize,
    num_grow_ch: usize,
    exec: SrExecution,
}

impl<R: Runtime<DType = DType>> ResidualDenseBlock<R> {
    /// Take `conv1..conv5`, each `.weight` and `.bias`, under `vb`'s
    /// prefix, such as `body.0.rdb1`.
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
        let mut load = |k: usize, out: usize| {
            SrConv::from_varbuilder(
                client,
                &mut *vb,
                &format!("conv{k}"),
                None,
                config.dense_conv_in_channels(k),
                out,
                KERNEL,
                exec,
            )
        };
        let growing = (1..RDB_CONVS)
            .map(|k| load(k, config.num_grow_ch))
            .collect::<Result<Vec<_>>>()?;
        let conv5 = load(RDB_CONVS, config.num_feat)?;
        Ok(Self {
            growing,
            conv5,
            num_feat: config.num_feat,
            num_grow_ch: config.num_grow_ch,
            exec: *exec,
        })
    }

    /// Run the block from `src` and write its output into channels
    /// `[0, num_feat)` of `dst`.
    ///
    /// `src` and `dst` are two feature buffers of
    /// `num_feat + 4 · num_grow_ch` channels in the execution layout. The
    /// block input sits in channels `[0, num_feat)` of `src`. The call
    /// overwrites channels `[num_feat, ..)` of `src` with `f_1..f_4`. `src`
    /// and `dst` must not share storage.
    ///
    /// # Errors
    ///
    /// A buffer with the wrong channel count, or a conv error. Each error
    /// names the conv and the shapes.
    pub fn forward_into<C: SrClient<R>>(
        &self,
        client: &C,
        src: &Tensor<R>,
        dst: &Tensor<R>,
    ) -> Result<()> {
        self.check_buffer("dst", dst)?;
        let out = dst.narrow(self.channel_dim(), 0, self.num_feat)?;
        let residual = self.grow(client, src)?;
        self.conv5.forward_into(
            client,
            &out,
            src,
            ConvActivation::None,
            Some(ConvResidual {
                tensor: &residual,
                scale: RESIDUAL_SCALE,
            }),
        )
    }

    /// Run the block from `src` and return its output as a new contiguous
    /// `num_feat`-channel tensor.
    ///
    /// `src` follows the rules of [`Self::forward_into`].
    pub fn forward<C: SrClient<R>>(&self, client: &C, src: &Tensor<R>) -> Result<Tensor<R>> {
        let residual = self.grow(client, src)?;
        self.conv5.forward(
            client,
            src,
            ConvActivation::None,
            Some(ConvResidual {
                tensor: &residual,
                scale: RESIDUAL_SCALE,
            }),
        )
    }

    /// `conv1..conv5` in order.
    pub fn convs(&self) -> impl Iterator<Item = &SrConv<R>> {
        self.growing.iter().chain(std::iter::once(&self.conv5))
    }

    /// Write `f_1..f_4` into `src`, and return the block input view
    /// `src[0, num_feat)` that the residual of `conv5` adds.
    fn grow<C: SrClient<R>>(&self, client: &C, src: &Tensor<R>) -> Result<Tensor<R>> {
        self.check_buffer("src", src)?;
        let dim = self.channel_dim();
        for (k, conv) in self.growing.iter().enumerate() {
            let filled = self.num_feat + k * self.num_grow_ch;
            let input = src.narrow(dim, 0, filled)?;
            let out = src.narrow(dim, filled, self.num_grow_ch)?;
            conv.forward_into(
                client,
                &out,
                &input,
                ConvActivation::LeakyRelu(LEAKY_SLOPE),
                None,
            )?;
        }
        Ok(src.narrow(dim, 0, self.num_feat)?)
    }

    fn channel_dim(&self) -> isize {
        self.exec.channel_dim() as isize
    }

    fn check_buffer(&self, arg: &'static str, buf: &Tensor<R>) -> Result<()> {
        let want = self.num_feat + (RDB_CONVS - 1) * self.num_grow_ch;
        let (_, c, _, _) = self.exec.image_dims(buf.shape())?;
        if c == want {
            return Ok(());
        }
        Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "dense block with conv5 '{}' needs a {want}-channel feature buffer, got shape {:?} in {:?}; allocate num_feat + 4 · num_grow_ch channels",
                self.conv5.name(),
                buf.shape(),
                self.exec.layout
            ),
        })
    }
}
