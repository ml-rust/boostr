//! `LocalEncoder` — VoxCPM2's `feat_encoder` (local encoder / "locenc"),
//! inference only.
//!
//! ```text
//! x [B, T, 4, 64]
//!   -> in_proj: Linear(64 -> 1024, WITH bias)          [B, T, 4, 1024]
//!   -> prepend special_token, broadcast [1,1,1,1024]   [B, T, 5, 1024]
//!      -> [B, T, 5, 1024] = [CLS, p0, p1, p2, p3] per (batch, frame)
//!   -> reshape [(B*T), 5, 1024]      each (batch, frame) is independent
//!   -> 12x pre-norm transformer layer (bidirectional GQA 16/2, head_dim 128)
//!   -> final RmsNorm over all 5 positions
//!   -> CLS-pool: take position 0, reshape            [B, T, 1024]
//! ```
//!
//! Patch folding (`[T, P, D]` -> `[T, 4, 64]`) happens upstream in the
//! audio-feature stage; this module takes the input already folded and does
//! NOT implement that fold.
//!
//! Built from plain [`Var<R>`]-wrapped weights (`requires_grad = false`)
//! rather than autograd-tracked training params — same inference-only
//! posture as `AudioVaeEncoder`/`AudioVaeDecoder` in this module.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::bidirectional::layer::BidirectionalLayer;
use crate::model::traits::ModelClient;
use crate::nn::{MaybeQuantLinear, RmsNorm, RoPE, var_contiguous};
use numr::autograd::{Var, var_broadcast_to, var_cat, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;

pub struct LocalEncoder<R: Runtime> {
    /// [`MaybeQuantLinear`], not plain `Linear`: a GGUF stores this
    /// projection block-quantized, and the quantized variant multiplies it
    /// PACKED through `quant_matmul` rather than expanding it to dense F32 at
    /// load. Its 4-D `[B, T, num_patches, patch_dim]` input is fine —
    /// `quant_matmul`'s contract is `[..., M, K]`, the same leading-dims rule
    /// dense `matmul` follows, so nothing reshapes here.
    pub(crate) in_proj: MaybeQuantLinear<R>,
    /// `[1, 1, 1, hidden_dim]`, broadcast to `[B, T, 1, hidden_dim]` and
    /// prepended along the patch axis.
    ///
    /// DENSE, deliberately: it is a learned constant that is concatenated,
    /// never multiplied, so there is no packed kernel it could feed.
    pub(crate) special_token: Var<R>,
    pub(crate) layers: Vec<BidirectionalLayer<R>>,
    pub(crate) norm: RmsNorm<R>,
    pub(crate) rope: RoPE<R>,
    pub(crate) hidden_dim: usize,
}

impl<R: Runtime<DType = DType>> LocalEncoder<R> {
    /// `x: [B, T, num_patches, patch_dim]` -> `[B, T, hidden_dim]`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        // `TypeConversionOps` is what `MaybeQuantLinear::forward` adds over a
        // dense `Linear::forward`, here for `in_proj` and for every
        // projection inside the layer stack.
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>,
    {
        let shape = x.shape().to_vec();
        if shape.len() != 4 {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "expected 4D [B, T, num_patches, patch_dim], got {}D",
                    shape.len()
                ),
            });
        }
        let (batch, seq_t, num_patches, _patch_dim) = (shape[0], shape[1], shape[2], shape[3]);

        // in_proj: patch_dim -> hidden_dim, WITH bias — the only biased
        // Linear in this module.
        let projected = self.in_proj.forward(client, x)?;

        // Prepend the learned CLS/special token, broadcast [1,1,1,H] ->
        // [B, T, 1, H], then concat along the patch axis (dim 2).
        // `broadcast_to` yields a strided view; `cat` copies from its inputs
        // and refuses one, so materialize it here.
        let special = var_contiguous(
            &var_broadcast_to(&self.special_token, &[batch, seq_t, 1, self.hidden_dim])
                .map_err(Error::Numr)?,
        )?;
        let with_cls = var_cat(&[&special, &projected], 2, client).map_err(Error::Numr)?;

        // [B, T, num_patches+1, H] -> [(B*T), num_patches+1, H]: each
        // (batch, frame) pair is an independent length-5 sequence.
        let seq_len = num_patches + 1;
        let flat = var_reshape(&with_cls, &[batch * seq_t, seq_len, self.hidden_dim])
            .map_err(Error::Numr)?;

        let mut h = flat;
        for layer in &self.layers {
            h = layer.forward(client, &h, &self.rope)?;
        }
        let h = self.norm.forward(client, &h)?;

        // CLS-pool: position 0 only, reshape [(B*T), 1, H] -> [B, T, H].
        // `narrow` yields a view over a strided slice; `reshape` needs it
        // materialized before it can reinterpret the layout.
        let cls = var_contiguous(&var_narrow(&h, 1, 0, 1).map_err(Error::Numr)?)?;
        var_reshape(&cls, &[batch, seq_t, self.hidden_dim]).map_err(Error::Numr)
    }
}
