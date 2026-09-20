//! Prefill half of the GDN forward: the per-token chain from the post-SiLU
//! conv output to the chunked recurrence, for `seq > 1`. The `seq == 1`
//! twin is [`GatedDeltaNetOps::gdn_step_from_conv`], which a backend may
//! fuse into one kernel.

use super::layer::{GdnBlock, slice_heads};
use crate::error::{Error, Result};
use crate::ops::traits::architecture::gated_delta_net::GatedDeltaNetOps;
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, NormalizationOps, ShapeOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> GdnBlock<R> {
    /// Steps 4 to 8 of the block forward over `seq > 1` tokens: split
    /// `qkv` `[batch, seq, qkv_dim]`, L2-normalize q and k, tile the key
    /// heads, build the gates from the raw `alpha` and `beta_raw`
    /// projections `[batch, seq, H_v]`, then `gdn_chunk_prefill`.
    ///
    /// Returns `(o: [batch, seq, H_v, S], state: [batch, H_v, S, S])`.
    pub(super) fn prefill_recurrence<C>(
        &self,
        client: &C,
        qkv: &Tensor<R>,
        alpha: &Tensor<R>,
        beta_raw: &Tensor<R>,
        ssm_state: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>)>
    where
        C: GatedDeltaNetOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + NormalizationOps<R>
            + ShapeOps<R>,
    {
        let cfg = &self.cfg;
        let shape = qkv.shape();
        let (batch, seq) = (shape[0], shape[1]);
        let (h_k, h_v, s) = (cfg.key_heads, cfg.value_heads, cfg.state_size);
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();

        // 4. Split.
        let q = slice_heads(qkv, 0, key_dim, &[batch, seq, h_k, s])?;
        let k = slice_heads(qkv, key_dim, key_dim, &[batch, seq, h_k, s])?;
        let v = slice_heads(qkv, 2 * key_dim, value_dim, &[batch, seq, h_v, s])?;

        // 5. L2 norm over the head dim.
        let q = client
            .l2_normalize(&q, -1, cfg.rms_eps)
            .map_err(Error::Numr)?;
        let k = client
            .l2_normalize(&k, -1, cfg.rms_eps)
            .map_err(Error::Numr)?;

        // 6. Tiled repeat: value head h_v reads key head h_v % H_k.
        let rep = cfg.head_repeat();
        let (q, k) = if rep > 1 {
            let tile = [1, 1, rep, 1];
            (
                client.repeat(&q, &tile).map_err(Error::Numr)?,
                client.repeat(&k, &tile).map_err(Error::Numr)?,
            )
        } else {
            (q, k)
        };

        // 7. Gates. `ssm_a` already holds `-exp(A_log)`.
        let beta = client.sigmoid(beta_raw).map_err(Error::Numr)?;
        let g = client.add(alpha, &self.ssm_dt_bias).map_err(Error::Numr)?;
        let g = client.softplus(&g).map_err(Error::Numr)?;
        let g = client.mul(&g, &self.ssm_a).map_err(Error::Numr)?;

        // 8. Chunked recurrence.
        client.gdn_chunk_prefill(&q, &k, &v, &g, &beta, ssm_state, cfg.chunk_size)
    }
}
