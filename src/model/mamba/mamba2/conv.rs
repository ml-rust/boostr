//! Mamba2 conv helpers: the causal prefill conv and the cached decode step.

use super::layer::Mamba2;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{ConvOps, PaddingMode, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime> Mamba2<R> {
    /// Prefill: full conv1d with causal padding, saves conv state.
    pub(in crate::model::mamba::mamba2) fn prefill_conv<C>(
        &self,
        client: &C,
        xbc_ncl: &Tensor<R>,
        seq_len: usize,
        batch: usize,
        x: &Tensor<R>,
        state: &mut crate::inference::SsmState<R>,
    ) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + ConvOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ConvOps<R>,
    {
        let conv_out = self.conv1d.forward_inference(client, xbc_ncl)?;
        let conv_out = conv_out
            .narrow(2, 0, seq_len)
            .map_err(Error::Numr)?
            .contiguous()?;

        let conv_window = self.config.d_conv - 1;
        if seq_len >= conv_window {
            let tail = xbc_ncl
                .narrow(2, seq_len - conv_window, conv_window)
                .map_err(Error::Numr)?
                .contiguous()?;
            state.update_conv_state(tail);
        } else {
            let conv_channels = self.config.conv_channels();
            let mut new_conv =
                Tensor::<R>::zeros(&[batch, conv_channels, conv_window], x.dtype(), x.device())?;
            let offset = conv_window - seq_len;
            if state.is_initialized() && offset > 0 {
                let old_tail = state
                    .conv_state()
                    .narrow(2, conv_window - offset, offset)
                    .map_err(Error::Numr)?
                    .contiguous()?;
                new_conv = new_conv
                    .slice_assign(&old_tail, 2, 0)
                    .map_err(Error::Numr)?;
            }
            new_conv = new_conv
                .slice_assign(xbc_ncl, 2, offset)
                .map_err(Error::Numr)?;
            state.update_conv_state(new_conv);
        }

        Ok(conv_out)
    }

    /// Decode (seq_len=1): manual conv step using cached state.
    pub(in crate::model::mamba::mamba2) fn decode_conv(
        &self,
        xbc_ncl: &Tensor<R>,
        batch: usize,
        x: &Tensor<R>,
        state: &mut crate::inference::SsmState<R>,
    ) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + ScalarOps<R> + ConvOps<R>,
    {
        let conv_window = self.config.d_conv - 1;
        let conv_channels = self.config.conv_channels();

        let old_state = if conv_window > 1 {
            state
                .conv_state()
                .narrow(2, 1, conv_window - 1)
                .map_err(Error::Numr)?
                .contiguous()?
        } else {
            Tensor::<R>::zeros(&[batch, conv_channels, 0], x.dtype(), x.device())?
        };

        let mut new_state =
            Tensor::<R>::zeros(&[batch, conv_channels, conv_window], x.dtype(), x.device())?;
        if conv_window > 1 {
            new_state = new_state
                .slice_assign(&old_state, 2, 0)
                .map_err(Error::Numr)?;
        }
        new_state = new_state
            .slice_assign(xbc_ncl, 2, conv_window - 1)
            .map_err(Error::Numr)?;
        state.update_conv_state(new_state.clone());

        let conv_input_refs = [&new_state, xbc_ncl];
        let conv_input = Tensor::cat(&conv_input_refs, 2).map_err(Error::Numr)?;

        let conv_weight = self.conv1d.weight().tensor();
        let conv_bias = self.conv1d.bias().map(|b| b.tensor());
        // The full conv input is [B, C, d_conv], which with valid padding gives [B, C, 1]
        conv_input
            .conv1d(
                conv_weight,
                conv_bias,
                1,
                PaddingMode::Valid,
                1,
                self.config.conv_channels(),
            )
            .map_err(Error::Numr)
    }
}
