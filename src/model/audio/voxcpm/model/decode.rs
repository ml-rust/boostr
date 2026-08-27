//! Unit C1 of the VoxCPM2 end-to-end orchestrator: turn the patches
//! [`super::generate::GenerateState`] collected into a waveform.
//!
//! ```text
//! patches: Vec<Var<R>>, each [1, patch_size, feat_dim]
//!   -> unfold_patches -> latent [1, feat_dim, n_patches * patch_size]
//!   -> chunked_decode::decode_latent_windowed (sr bucket 3, 48 kHz) -> waveform [1, 1, samples]
//! ```
//!
//! [`unfold_patches`] is the exact inverse of
//! [`super::patches::fold_patches`]: patch `t`'s slot `p` lands at time index
//! `t * patch_size + p`. Every other axis order is shape-valid and scrambles
//! time, so this file mirrors `fold_patches`'s reshape/permute rather than
//! reinventing the unfold.
//!
//! # Reference mode only
//!
//! In REFERENCE (voice-clone) mode `context_len == 0`, so the reference does
//! NOT trim the decoded waveform. A trim applies only in continuation mode,
//! which this orchestrator does not implement, so [`VoxCpm2Model::decode_patches`]
//! never trims for that reason. (`chunked_decode` does trim internally, per
//! window, to remove the overlap context added for chunking — an unrelated,
//! purely-internal trim that leaves the total sample count unchanged.)
//!
//! # The one device read
//!
//! There is none in this file. [`VoxCpm2Model::decode_patches`] returns the
//! decoded `Tensor<R>` as-is; converting it to `Vec<f32>` for a WAV file is a
//! later unit's job, at the boundary right before the write.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::client::VoxCpmClient;
use crate::model::audio::voxcpm::model::chunked_decode::decode_latent_windowed;
use crate::model::audio::voxcpm::model::loader::VoxCpm2Model;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{ShapeOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Unfold generated patches back into an `AudioVAE` latent.
///
/// `patches[t]` must be `[1, patch_size, feat_dim]`, in generation order.
/// Produces `[1, feat_dim, patches.len() * patch_size]` such that
/// `folded[t * patch_size + p][c] == unfolded[c][t * patch_size + p]` — the
/// exact inverse of [`super::patches::fold_patches`]'s
/// `view(64, -1, 4).permute(1, 2, 0)`.
///
/// Errors on an empty slice, or when any patch is not `[1, patch_size,
/// feat_dim]`.
pub fn unfold_patches<R: Runtime<DType = DType>>(
    patches: &[Var<R>],
    patch_size: usize,
    feat_dim: usize,
) -> Result<Tensor<R>>
where
    R::Client: ShapeOps<R>,
{
    if patches.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "patches",
            reason: "expected at least 1 patch, got 0".to_string(),
        });
    }

    let expected = [1, patch_size, feat_dim];
    let mut refs = Vec::with_capacity(patches.len());
    for (i, patch) in patches.iter().enumerate() {
        let shape = patch.shape();
        if shape != expected.as_slice() {
            return Err(Error::InvalidArgument {
                arg: "patches",
                reason: format!("patch {i}: expected {expected:?}, got {shape:?}"),
            });
        }
        refs.push(patch.tensor());
    }

    // [n_patches, patch_size, feat_dim], patches[t][p][c] preserved exactly
    // as fold_patches's own output would be.
    let stacked = Tensor::cat(&refs, 0)?;
    let frames = patches.len() * patch_size;

    // Inverse of fold_patches's `reshape([feat_dim, t_ref, patch_size])
    // .permute([1, 2, 0])`: permute back with the inverse permutation
    // ([1,2,0])^-1 = [2,0,1], then merge (t_ref, patch_size) back into one
    // frame axis exactly as the original reshape split it.
    Ok(stacked
        .permute(&[2, 0, 1])?
        .contiguous()?
        .reshape(&[feat_dim, frames])?
        .unsqueeze(0)?)
}

impl<R: Runtime<DType = DType>> VoxCpm2Model<R> {
    /// Decode generated patches to a waveform.
    ///
    /// Unfolds `patches` with [`unfold_patches`] using `self.config`'s patch
    /// geometry, then runs [`super::loader::VoxCpm2Model::vae_decoder`] at
    /// the fixed 48 kHz sample-rate bucket
    /// ([`crate::model::audio::voxcpm::vae::decoder::DEFAULT_SR_BUCKET`]),
    /// windowed via [`super::chunked_decode::decode_latent_windowed`] so peak
    /// decoder activation memory is bounded by a fixed number of latent
    /// frames rather than growing linearly with the utterance length.
    /// Returns `[1, 1, samples]` at [`crate::model::audio::voxcpm::vae::decoder::SAMPLE_RATE`]
    /// Hz, un-trimmed (reference/voice-clone mode has no context to trim).
    pub fn decode_patches<C>(&self, client: &C, patches: &[Var<R>]) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
        R::Client: ShapeOps<R> + TypeConversionOps<R>,
    {
        let latent = unfold_patches(patches, self.config.patch_size, self.config.feat_dim)?;
        // The transformer stack runs at whatever dtype it was loaded at; the
        // AudioVAE is always left at its own (F32). This is the boundary
        // between them, mirroring `prefill`'s cast of the encoder's F32
        // reference features up into the stack's dtype.
        let latent = latent.to_dtype(self.vae_decoder.dtype())?;
        decode_latent_windowed(client, &self.vae_decoder, &latent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::voxcpm::model::patches::fold_patches;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// Split a `fold_patches`-shaped tensor `[t_ref, patch_size, feat_dim]`
    /// into `t_ref` per-patch `Var`s of `[1, patch_size, feat_dim]`, mimicking
    /// how [`super::super::generate::GenerateState::patches`] is populated.
    fn split_into_patches(folded: &Tensor<CpuRuntime>, t_ref: usize) -> Vec<Var<CpuRuntime>> {
        (0..t_ref)
            .map(|t| {
                let row = folded.narrow(0, t, 1).expect("narrow");
                Var::new(row, false)
            })
            .collect()
    }

    /// `unfold_patches` inverts `fold_patches` exactly: every value encodes
    /// its own `(channel, frame)` index, so a transposed unfold (any axis
    /// order other than the one derived here) cannot pass.
    #[test]
    fn unfold_inverts_fold_for_self_indexing_values() {
        let (_client, device) = cpu_setup();
        let (feat_dim, patch_size, t_ref) = (3usize, 2usize, 4usize);
        let frames = patch_size * t_ref;

        let data: Vec<f32> = (0..feat_dim * frames)
            .map(|i| {
                let (c, t) = (i / frames, i % frames);
                (c * 100 + t) as f32
            })
            .collect();
        let latent =
            Tensor::<CpuRuntime>::from_slice(&data, &[1, feat_dim, frames], &device).expect("in");

        let folded = fold_patches(&latent, patch_size, feat_dim).expect("fold");
        let patches = split_into_patches(&folded, t_ref);

        let unfolded = unfold_patches(&patches, patch_size, feat_dim).expect("unfold");
        assert_eq!(unfolded.shape(), &[1, feat_dim, frames]);

        let want: Vec<f32> = latent.contiguous().expect("contig").to_vec();
        let got: Vec<f32> = unfolded.contiguous().expect("contig").to_vec();
        assert_eq!(got, want);
    }

    #[test]
    fn unfold_rejects_an_empty_patch_list() {
        let patches: Vec<Var<CpuRuntime>> = Vec::new();
        let err = unfold_patches(&patches, 4, 64).expect_err("empty slice must error");
        let msg = err.to_string();
        assert!(msg.contains('0'), "{msg}");
    }

    #[test]
    fn unfold_rejects_a_patch_with_the_wrong_shape() {
        let (_client, device) = cpu_setup();
        let bad =
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4 * 3], &[1, 4, 3], &device).expect("in");
        let patches = vec![Var::new(bad, false)];
        let err = unfold_patches(&patches, 4, 64).expect_err("feat_dim 3 != 64 must error");
        let msg = err.to_string();
        assert!(msg.contains("patch 0"), "{msg}");
        assert!(msg.contains("[1, 4, 64]"), "{msg}");
        assert!(msg.contains("[1, 4, 3]"), "{msg}");
    }
}
