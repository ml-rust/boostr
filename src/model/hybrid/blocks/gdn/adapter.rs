//! Unmerged LoRA adapters on a [`GdnBlock`]'s projections:
//! [`GdnBlock::attach_adapter`] for `attn_qkv`, `attn_gate` and `ssm_out`,
//! [`GdnBlock::attach_gate_adapters`] for the `ssm_alpha`/`ssm_beta` pair.
//!
//! Factors arrive in the block's layout: value heads in tiled order, and
//! `ssm_out` columns grouped when `v_grouped` is set. A caller holding
//! factors in another layout reorders them first.

use super::layer::GdnBlock;
use crate::error::Result;
use crate::nn::LoraFactors;
use numr::dtype::DType;
use numr::ops::{ScalarOps, ShapeOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};

/// One [`GdnBlock`] projection that takes an adapter on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnProjection {
    /// `attn_qkv`, `[qkv_dim, hidden_size]`.
    AttnQkv,
    /// `attn_gate` (`z`), `[value_dim, hidden_size]`.
    AttnGate,
    /// `ssm_out`, `[hidden_size, value_dim]`.
    SsmOut,
}

impl<R: Runtime<DType = DType>> GdnBlock<R> {
    /// Attach an unmerged adapter to `projection` through
    /// [`MaybeRotatedLinear::attach_factors`](crate::nn::MaybeRotatedLinear::attach_factors).
    /// `activation_dtype` is the dtype the block's input arrives in.
    ///
    /// # Errors
    ///
    /// As `attach_factors`: a Hadamard-rotated or already adapted
    /// projection, or a factor shape that does not fit it.
    pub fn attach_adapter<C: TypeConversionOps<R>>(
        &mut self,
        client: &C,
        projection: GdnProjection,
        factors: LoraFactors<R>,
        activation_dtype: DType,
    ) -> Result<()> {
        let linear = match projection {
            GdnProjection::AttnQkv => &mut self.attn_qkv,
            GdnProjection::AttnGate => &mut self.attn_gate,
            GdnProjection::SsmOut => &mut self.ssm_out,
        };
        linear.attach_factors(client, factors, activation_dtype)
    }

    /// Attach unmerged adapters to `ssm_alpha` (`alpha`) and `ssm_beta`
    /// (`beta`), each `lora_a` `[rank, hidden_size]` and `lora_b`
    /// `[value_heads, rank]`, either `None`. A dense pair, fused at
    /// construction, takes one block-diagonal adapter whose rows for a
    /// projection without an adapter are zero.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`](crate::error::Error::ModelError) when both are
    /// `None`, a targeted projection already carries an adapter, or a
    /// factor shape is wrong. The block is unchanged on error.
    pub fn attach_gate_adapters<C>(
        &mut self,
        client: &C,
        alpha: Option<LoraFactors<R>>,
        beta: Option<LoraFactors<R>>,
        activation_dtype: DType,
    ) -> Result<()>
    where
        C: RuntimeClient<R> + ScalarOps<R> + ShapeOps<R> + TypeConversionOps<R>,
    {
        self.gates = self
            .gates
            .with_adapters(client, alpha, beta, activation_dtype)?;
        Ok(())
    }
}
