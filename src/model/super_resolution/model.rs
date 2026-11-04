//! [`SrModel`]: one type over both super-resolution networks.

use crate::error::Result;
use crate::model::super_resolution::client::SrClient;
use crate::model::super_resolution::compact::SrVggCompact;
use crate::model::super_resolution::rrdb::RrdbNet;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A loaded super-resolution network.
///
/// The variant owns the network and its weights. `SrModel` holds no other
/// copy.
pub enum SrModel<R: Runtime> {
    /// SRVGGNetCompact, such as `realesr-general-x4v3`.
    Compact(Box<SrVggCompact<R>>),
    /// RRDBNet at x2, such as `RealESRGAN_x2plus` or `2x_PureVision`.
    Rrdb(Box<RrdbNet<R>>),
}

impl<R: Runtime<DType = DType>> SrModel<R> {
    /// The integer upscale factor of [`SrModel::forward_native`].
    pub fn native_scale(&self) -> usize {
        match self {
            Self::Compact(net) => net.config().upscale,
            Self::Rrdb(net) => net.config().scale,
        }
    }

    /// Run the network's native forward.
    ///
    /// Compact runs [`SrVggCompact::forward_raw`] at its upscale. RRDB runs
    /// [`RrdbNet::forward`] at x2. Input and output follow the layout and
    /// dtype of the [`super::SrExecution`] the model loaded with.
    ///
    /// # Errors
    ///
    /// The errors of the network's forward. Each names the input shape and
    /// dtype.
    pub fn forward_native<C: SrClient<R>>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>> {
        match self {
            Self::Compact(net) => net.forward_raw(client, x),
            Self::Rrdb(net) => net.forward(client, x),
        }
    }
}
